import os
import sys
import yaml
import torch
import random
import argparse
import numpy as np
import optuna
import pandas as pd
import json
import logging
import warnings
from typing import Union, List, Dict, Any
from scipy.optimize import curve_fit

# --- 1. STRICT Determinism Setup ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Local imports
from MatImba.utils.losses import *
from MatImba.models.megnet import MEGNet
from MatImba.utils.evaluate import get_obj
from MatImba.trainer import CgcnnTrainer, LossExplosionError
from MatImba.dataset.crystalgraph import CgcnnDataset
from MatImba.utils.struct2graph import (
    SimpleCrystalConverter,
    FlattenGaussianDistanceConverter,
    GaussianDistanceConverter,
    AtomFeaturesExtractor,
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console Handler (always active)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

def setup_file_logging(log_file_path):
    """Dynamically attaches a file handler to the root logger."""
    # Remove existing file handlers to prevent writing to old logs
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    # Add new file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
def seed_everything(seed: int):
    """
    Sets seeds for all relevant libraries to ensure reproducibility.
    Uses warn_only=True to prevent crashes on non-deterministic GNN ops.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def narrow_range(low, high, value, factor=2.0):
    new_low = max(low, value / factor)
    new_high = min(high, value * factor)
    return new_low, new_high

class RobustObjective:
    """
    Stage 1: Screening.
    Runs 3 seeds per trial. Implements 'Fail Fast' to discard obviously unstable params.
    """
    def __init__(self, expt_config, fold, model_name, datafiles, 
                 bond_converter, atom_converter, target_name, first_best_params=None):
        self.expt_config = expt_config
        self.fold = fold
        self.model_name = model_name
        self.datafiles = datafiles
        self.bond_converter = bond_converter
        self.atom_converter = atom_converter
        self.target_name = target_name
        self.first_best_params = first_best_params
        
        self.seeds_per_trial = 3     
        self.stability_penalty = 2.0 
        self.best_observed_mae = float('inf')

    def __call__(self, trial):
        # 1. Suggest Parameters
        config_scheduler = self.expt_config['scheduler'].get('name')
        
        if self.first_best_params:
            scheduler_type = trial.suggest_categorical('scheduler_type', [self.first_best_params['scheduler_type']])
        elif config_scheduler is not None:
            scheduler_type = trial.suggest_categorical('scheduler_type', [config_scheduler])
        else:
            scheduler_type = trial.suggest_categorical('scheduler_type', ['ReduceLROnPlateau', 'CosineAnnealingLR', 'OneCycleLR'])

        if self.first_best_params:
            p = self.first_best_params
            lr = trial.suggest_float('lr', *narrow_range(1e-4, 1e-2, p['lr']), log=True)
            weight_decay = trial.suggest_float('weight_decay', *narrow_range(1e-7, 1e-4, p['weight_decay']), log=True)
            
            if scheduler_type == 'ReduceLROnPlateau':
                patience = trial.suggest_int('patience', max(5, p['patience']-10), min(50, p['patience']+10))
                factor = trial.suggest_float('factor', max(0.05, p['factor']-0.2), min(0.9, p['factor']+0.2))
                min_lr = trial.suggest_float('min_lr', *narrow_range(1e-6, 1e-4, p['min_lr']), log=True)
                sched_params = {'patience': patience, 'factor': factor, 'min_lr': min_lr}
            elif scheduler_type == 'CosineAnnealingLR':
                eta_min = trial.suggest_float('eta_min', *narrow_range(1e-7, 1e-5, p['eta_min']), log=True)
                sched_params = {'T_max': self.expt_config['train']['epoch_range'], 'eta_min': eta_min}
            elif scheduler_type == 'OneCycleLR':
                max_lr = trial.suggest_float('max_lr', *narrow_range(1e-3, 0.1, p['max_lr']), log=True)
                pct_start = trial.suggest_float('pct_start', 0.1, 0.5)
                div_factor = trial.suggest_float('div_factor', 10, 50)
                dummy_loader = self._get_loader(seed=0)[0]
                sched_params = {'max_lr': max_lr, 'epochs': self.expt_config['train']['epoch_range'], 
                                'steps_per_epoch': len(dummy_loader), 'pct_start': pct_start, 'div_factor': div_factor}
        else:
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
            if scheduler_type == 'ReduceLROnPlateau':
                 sched_params = {'patience': trial.suggest_int('patience', 10, 30), 
                                 'factor': trial.suggest_float('factor', 0.1, 0.5), 
                                 'min_lr': trial.suggest_float('min_lr', 1e-6, 1e-4, log=True)}
            elif scheduler_type == 'CosineAnnealingLR':
                 sched_params = {'T_max': self.expt_config['train']['epoch_range'], 
                                 'eta_min': trial.suggest_float('eta_min', 1e-7, 1e-5, log=True)}
            elif scheduler_type == 'OneCycleLR':
                max_lr = trial.suggest_float('max_lr', 1e-3, 0.1, log=True)
                pct_start = trial.suggest_float('pct_start', 0.1, 0.5)
                div_factor = trial.suggest_float('div_factor', 10, 25)
                dummy_loader = self._get_loader(seed=0)[0]
                sched_params = {'max_lr': max_lr, 'epochs': self.expt_config['train']['epoch_range'], 
                                'steps_per_epoch': len(dummy_loader), 'pct_start': pct_start, 'div_factor': div_factor}

        optim_params = {
            'lr': lr if scheduler_type != 'OneCycleLR' else max_lr / div_factor,
            'weight_decay': weight_decay,
            'betas': self.expt_config['optimiser'].get('parameters', {}).get('betas', (0.85, 0.99)),
            'eps': self.expt_config['optimiser'].get('parameters', {}).get('eps', 1e-08)
        }
        
        # Save params to trial for retrieval later
        trial.set_user_attr("optim_params", optim_params)
        trial.set_user_attr("sched_params", sched_params)
        trial.set_user_attr("scheduler_type", scheduler_type)

        results = []
        base_seed = 2000 + trial.number * 100 
        
        for i in range(self.seeds_per_trial):
            seed = base_seed + i
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val_mae = self._run_single_seed(trial, seed, optim_params, sched_params, scheduler_type)
                results.append(val_mae)
                
                # Fail Fast: If first seed is > 1.3x worse than best observed, kill it.
                if i == 0 and self.best_observed_mae != float('inf'):
                    if val_mae > self.best_observed_mae * 1.3:
                        logger.info(f"Trial {trial.number} pruned early (Seed 0 MAE {val_mae:.1f} > 1.3x Best {self.best_observed_mae:.1f})")
                        raise optuna.TrialPruned()

            except LossExplosionError:
                logger.warning(f"Trial {trial.number} exploded on seed {seed}. Pruning.")
                raise optuna.TrialPruned()

        current_mean = np.mean(results)
        if current_mean < self.best_observed_mae:
            self.best_observed_mae = current_mean

        mean_mae = np.mean(results)
        std_mae = np.std(results)
        final_score = mean_mae + self.stability_penalty * std_mae
        
        trial.set_user_attr("mean_mae", mean_mae)
        trial.set_user_attr("std_mae", std_mae)
        
        return final_score

    def _get_loader(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        data_set_creator = CgcnnDataset(
            datafile=self.datafiles, target_name=self.target_name, 
            bond_converter=self.bond_converter, atom_converter=self.atom_converter, 
            random_seed=seed
        )
        return data_set_creator.prepare_data(
            reweight=self.expt_config['data'].get('reweight', 'log_inv'), 
            generator=g, worker_init_fn=seed_worker
        )

    def _run_single_seed(self, trial, seed, optim_params, sched_params, scheduler_type):
        seed_everything(seed)
        train_loader, val_loader, test_loader = self._get_loader(seed)
        
        model_kwargs = {
            "edge_input_shape": self.bond_converter.get_shape(),
            "node_input_shape": self.atom_converter.get_shape(),
            "state_input_shape": self.expt_config["model"]["state_input_shape"],
            "device": self.expt_config["model"].get('device', None)
        }
        if 'fds' in self.expt_config:
            model_kwargs.update({'fds': True, **self.expt_config['fds']})
        
        model = MEGNet(**model_kwargs)
        
        loss_func = get_obj(self.expt_config['loss']['loss'])()
        optimiser = get_obj(self.expt_config['optimiser']['name'])(model.parameters(), **optim_params)
        scheduler = get_obj(scheduler_type)(optimiser, **sched_params)
        
        hpo_dir = os.path.join(self.expt_config['save']['basedir'], self.expt_config['save']['outdir'], 'hpo_trials')
        os.makedirs(hpo_dir, exist_ok=True)

        # LOGGING: Setup specific log file for this trial
        trial_name = f"{self.model_name}_t{trial.number}_s{seed}"
        setup_file_logging(os.path.join(hpo_dir, f"{trial_name}.log"))
        
        trainer = CgcnnTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            loss_func=loss_func, optimiser=optimiser, scheduler=scheduler, scheduler_type=scheduler_type,
            name=trial_name, **self.expt_config['train'], outdir=hpo_dir
        )
        if hasattr(model, 'FDS'): model.FDS.device = trainer.device
        
        metrics = trainer.fit()
        if 'error' in metrics:
            raise LossExplosionError(metrics['error'])
            
        final_mae = metrics.get('test_mae', trainer.best_l1_loss)
        if final_mae is None or np.isinf(final_mae):
            raise LossExplosionError("Invalid MAE returned")
            
        return final_mae

# --- Helper Functions ---
def get_opt_params(study):
    try:
        best_trial = study.best_trial
        return best_trial.params, best_trial.value
    except:
        return None, float('inf')

def _load_fold0_params(outdir):
    try:
        # We try to load the JSON first as it's the verified robust one
        json_path = os.path.join(outdir, 'fold_0_best_params.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        # Fallback to DB if JSON missing (legacy)
        return get_opt_params(optuna.load_study(study_name='fold_0', storage=f"sqlite:///{outdir}/optuna_study_fold_0.db"))[0]
    except:
        return None

# --- NEW: Candidate Verification Logic ---
def verify_top_candidates(study, objective_fn, top_k=3, n_seeds=5):
    """
    Takes the top K candidates from the study and runs them on N *new* seeds.
    Returns the params of the most robust candidate.
    """
    logger.info(f"--- Stage 2: Verifying Top {top_k} Candidates on {n_seeds} New Seeds ---")
    
    # 1. Get Top Candidates
    df = study.trials_dataframe()
    complete_df = df[df.state == 'COMPLETE'].sort_values('value')
    
    if complete_df.empty:
        logger.warning("No complete trials to verify.")
        return None
        
    top_trials = complete_df.head(top_k)
    candidates = []
    
    # 2. Extract Params for Candidates
    for _, row in top_trials.iterrows():
        trial_number = row['number']
        trial = study.trials[trial_number]
        
        candidates.append({
            'id': trial_number,
            'hpo_params': trial.params,  # <--- Ensured Key
            'optim_params': trial.user_attrs['optim_params'], 
            'sched_params': trial.user_attrs['sched_params'], 
            'scheduler_type': trial.user_attrs['scheduler_type'],
            'original_score': row['value']
        })

    best_candidate = None
    best_robust_score = float('inf')

    # 3. Verification Loop
    # New base seed for verification (distinct from HPO seeds)
    base_seed = 9000 
    
    for cand in candidates:
        logger.info(f"Verifying Candidate {cand['id']} (Orig Score: {cand['original_score']:.2f})...")
        
        results = []
        try:
            for i in range(n_seeds):
                seed = base_seed + i
                # Reuse the single_seed runner from objective
                # We need a dummy trial object just to pass the ID/logging if needed, 
                # but we can reuse the internal logic if we refactor or just call manual trainer logic.
                # Actually, easier to call objective._run_single_seed with a dummy trial object
                class DummyTrial:
                    number = f"verify_{cand['id']}"
                
                mae = objective_fn._run_single_seed(
                    DummyTrial(), seed, cand['optim_params'], cand['sched_params'], cand['scheduler_type']
                )
                results.append(mae)
                
        except LossExplosionError:
            logger.warning(f"Candidate {cand['id']} exploded during verification. Discarding.")
            continue
            
        mean_mae = np.mean(results)
        std_mae = np.std(results)
        # Using a slightly higher penalty for final selection to be ultra-safe
        robust_score = mean_mae + 2.0 * std_mae
        
        logger.info(f"  -> Mean: {mean_mae:.2f} | Std: {std_mae:.2f} | Score: {robust_score:.2f}")
        
        if robust_score < best_robust_score:
            best_robust_score = robust_score
            # We return the FULL structured config to avoid reconstruction errors
            best_candidate = {
                'scheduler_type': cand['scheduler_type'],
                'optim_params': cand['optim_params'],
                'sched_params': cand['sched_params'],
                'hpo_params': cand['hpo_params']
            }
            
    if best_candidate is None:
        logger.warning("All candidates failed verification! Falling back to best study trial (reconstructing).")
        # Reconstruct fallback
        trial = study.best_trial
        return {
            'scheduler_type': trial.user_attrs['scheduler_type'],
            'optim_params': trial.user_attrs['optim_params'],
            'sched_params': trial.user_attrs['sched_params'],
            'hpo_params': trial.params
        }
        
    logger.info(f"Winner: Candidate with score {best_robust_score:.2f}")
    return best_candidate

def run(config_path, config_file):
    with open(os.path.join(config_path, config_file)) as config:
        expt_config = yaml.full_load(config)
    
    # Consistency Checks
    if expt_config['train'].get('weighted_loss', False) and not expt_config['data'].get('reweight', False):
        print("Warning: 'weighted_loss' is True but 'reweight' is missing. Defaulting to 'log_inv'.")
        expt_config['data']['reweight'] = 'log_inv'

    basedir = expt_config['save']['basedir']
    outdir = os.path.join(basedir, expt_config['save']['outdir'])
    os.makedirs(outdir, exist_ok=True)
    base_seed = expt_config['data']['seed']
    
    # Setup Converters
    bond_centers = np.linspace(0, expt_config['data']['cutoff'], expt_config['data']['edge_embed_size'])
    if expt_config["data"]["add_z_bond_coord"]:
        bond_converter = FlattenGaussianDistanceConverter(centers=bond_centers)
    else:
        bond_converter = GaussianDistanceConverter(centers=bond_centers)
    atom_converter = AtomFeaturesExtractor(expt_config["data"]["atom_features"])
    target_name = expt_config['data']['target_name']
    
    hpo_trials = expt_config['hpo']['trials']
    multi_runs = expt_config.get('multi_runs', 1)

    for fold in expt_config["data"]["folds"]:
        model_name = f'fold_{fold}'
        datafiles = {
            'train': os.path.join(expt_config['data']['data_loc'], model_name, 'train.pickle.gz'),
            'test': os.path.join(expt_config['data']['data_loc'], model_name, 'test.pickle.gz')
        }
        
        db_file = f"{outdir}/optuna_study_fold_{fold}.db"
        storage_name = "sqlite:///" + db_file
        best_params_for_runs = None
        best_params_file = os.path.join(outdir, f'fold_{fold}_best_params.json')
        
        # --- HPO Phase ---
        if hpo_trials > 0:
            study = optuna.create_study(direction='minimize', storage=storage_name, study_name=f'fold_{fold}', load_if_exists=True)
            first_best_params, first_best_value = get_opt_params(study)
            
            if fold != 0 and not first_best_params:
                fold_0_params = _load_fold0_params(outdir)
                if fold_0_params: study.enqueue_trial(fold_0_params)
            
            # Count only COMPLETE trials to ensure sufficient sampling
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            target_trials = (hpo_trials if fold == 0 else (1 + 2 * hpo_trials // 3))
            remaining = target_trials - len(completed_trials)
            
            # Helper to create objective (needed for both optimization and verification)
            objective_fn = RobustObjective(
                expt_config, fold, model_name, datafiles, 
                bond_converter, atom_converter, target_name, first_best_params
            )

            if remaining > 0:
                logger.info(f"Running {remaining} Robust HPO trials for fold {fold} (3 seeds/trial)...")
                study.optimize(objective_fn, n_trials=remaining)
            
            # --- STAGE 2: VERIFICATION ---
            # Instead of just taking the best result from the limited 3-seed runs,
            # we verify the top 5 candidates on 5 NEW seeds.
            # SKIP if we already have the verified params file
            if os.path.exists(best_params_file):
                logger.info(f"Verified params found in {best_params_file}. Skipping Stage 2 Verification.")
                with open(best_params_file, 'r') as f:
                    best_params_for_runs = json.load(f)
            else:
                best_params_for_runs = verify_top_candidates(study, objective_fn, top_k=5, n_seeds=5)
                # Save the params even if HPO was skipped but verification wasn't
                study.trials_dataframe().to_csv(os.path.join(outdir, f'fold_{fold}_all_trials.csv'), index=False)
                if best_params_for_runs:
                    with open(best_params_file, 'w') as f: json.dump(best_params_for_runs, f, indent=4)
        else:
             if os.path.exists(db_file):
                study = optuna.load_study(study_name=f'fold_{fold}', storage=storage_name)
                best_params_for_runs, _ = get_opt_params(study)

        # --- Multi-Run Phase ---
        results_file = os.path.join(outdir, f'fold_{fold}_results.csv')
        finished_ids = []
        if os.path.exists(results_file):
            try: finished_ids = pd.read_csv(results_file)['run_id'].unique().tolist()
            except: pass
            
        for run_id in range(multi_runs):
            if run_id in finished_ids: continue
                
            run_name = f'{model_name}_run{run_id}'
            # LOGGING: Setup log file for this specific run
            setup_file_logging(os.path.join(outdir, f'{run_name}.log'))
            logger.info(f"Starting run {run_id} for fold {fold}...")
            
            seed = base_seed + run_id
            seed_everything(seed)
            g = torch.Generator()
            g.manual_seed(seed)
            
            data_set_creator = CgcnnDataset(
                datafile=datafiles, target_name=target_name,
                bond_converter=bond_converter,
                atom_converter=atom_converter,
                random_seed=seed
            )
            train_loader, val_loader, test_loader = data_set_creator.prepare_data(
                reweight=expt_config['data'].get('reweight', 'log_inv'),
                generator=g, worker_init_fn=seed_worker
            )
            
            # Setup Params
            base_optim = expt_config['optimiser'].get('parameters', {})
            base_sched = expt_config['scheduler'].get('parameters', {})
            sched_type = expt_config['scheduler'].get('name', 'CosineAnnealingLR')
            optim_params = base_optim.copy()
            sched_params = base_sched.copy()
            
            if best_params_for_runs:
                # Use the Structured dictionary directly if available
                if 'optim_params' in best_params_for_runs:
                    sched_type = best_params_for_runs['scheduler_type']
                    optim_params = best_params_for_runs['optim_params']
                    sched_params = best_params_for_runs['sched_params']
                else:
                    # Legacy Fallback (Reconstruction logic)
                    sched_type = best_params_for_runs['scheduler_type']
                    optim_params.update({
                        'lr': best_params_for_runs['lr'],
                        'weight_decay': best_params_for_runs['weight_decay']
                    })
                    if sched_type == 'ReduceLROnPlateau': 
                        sched_params.update({
                            'patience': best_params_for_runs['patience'],
                            'factor': best_params_for_runs['factor'],
                            'min_lr': best_params_for_runs['min_lr']
                        })
                    elif sched_type == 'CosineAnnealingLR': 
                        sched_params.update({
                            'T_max': expt_config['train']['epoch_range'],
                            'eta_min': best_params_for_runs['eta_min']
                        })
                    elif sched_type == 'OneCycleLR':
                        optim_params['lr'] = best_params_for_runs['max_lr'] / best_params_for_runs['div_factor']
                        sched_params.update({
                            'max_lr': best_params_for_runs['max_lr'],
                            'epochs': expt_config['train']['epoch_range'],
                            'steps_per_epoch': len(train_loader),
                            'pct_start': best_params_for_runs['pct_start'],
                            'div_factor': best_params_for_runs['div_factor']
                        })
            else:
                # Defaults
                optim_params.setdefault('lr', 0.01); optim_params.setdefault('weight_decay', 1e-6)
                if sched_type == 'ReduceLROnPlateau': sched_params.update({'factor': 0.2, 'patience': 20, 'min_lr': 1e-5})
                elif sched_type == 'CosineAnnealingLR': sched_params.update({
                    'T_max': expt_config['train']['epoch_range'], 'eta_min': 1e-6
                })
                elif sched_type == 'OneCycleLR':
                     sched_params.update({
                         'max_lr': 0.01, 'epochs': expt_config['train']['epoch_range'],
                         'steps_per_epoch': len(train_loader), 'pct_start': 0.3,
                         'div_factor': 25
                     })
                     optim_params['lr'] = sched_params['max_lr'] / sched_params['div_factor']

            model = MEGNet(
                edge_input_shape=bond_converter.get_shape(),
                node_input_shape=atom_converter.get_shape(),
                state_input_shape=expt_config["model"]["state_input_shape"],
                device=expt_config["model"].get('device', None),
                fds=('fds' in expt_config), **expt_config.get('fds', {})
            )
            
            loss_func = get_obj(expt_config['loss']['loss'])()
            optimiser = get_obj(expt_config['optimiser']['name'])(model.parameters(), **optim_params)
            scheduler = get_obj(sched_type)(optimiser, **sched_params)
            
            run_name = f'{model_name}_run{run_id}'
            trainer = CgcnnTrainer(
                model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                loss_func=loss_func, optimiser=optimiser, scheduler=scheduler, scheduler_type=sched_type,
                name=run_name, **expt_config['train'], outdir=outdir
            )
            if hasattr(model, 'FDS'): model.FDS.device = trainer.device
            
            metrics_result = trainer.fit()
            if 'error' in metrics_result: continue
            
            final_metrics = metrics_result
            final_metrics.update({'seed': seed, 'run_id': run_id, 'fold': fold})
            try:
                trainer.plot_dynamics()
                trainer.plot_awareness_space()
            except: pass
            
            pd.DataFrame([final_metrics]).to_csv(
                results_file, mode='a',
                header=not os.path.exists(results_file),
                index=False
            )
            logger.info(f"Saved results for run {run_id} to {results_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cd', type=str, default='expt_configs', help='Experiment configuration directory')
    parser.add_argument('--cf', type=str, default='log_kvrh.yaml', help='Experiment configuration file')
    args = parser.parse_args()
    run(args.cd, args.cf)

if __name__ == '__main__':
    main()