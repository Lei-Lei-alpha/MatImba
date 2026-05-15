import os
import sys
import torch
import logging
import argparse
import yaml
import numpy as np
import pandas as pd

import warnings
from typing import Optional, Dict

# Reuse your existing imports
from MatImba.utils.losses import *
from MatImba.models.megnet import MEGNet
from MatImba.trainer import CgcnnTrainer, LossExplosionError
from MatImba.dataset.crystalgraph import CgcnnDataset
from MatImba.utils.struct2graph import (
    GaussianDistanceConverter, 
    FlattenGaussianDistanceConverter, 
    AtomFeaturesExtractor
)
from MatImba.utils.evaluate import get_obj
from run_trainer import seed_everything, seed_worker, setup_file_logging

# Initialize Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

# --- 1. BSAM Optimizer Implementation ---
class BSAM(torch.optim.Optimizer):
    """
    Balanced Sharpness-Aware Minimization (BSAM).
    
    Wraps a base optimizer (e.g., AdamW) to perform the SAM update:
    1. Compute gradients of the loss.
    2. Ascend in the direction of the gradient (perturb weights).
    3. Compute gradients at the perturbed state.
    4. Descend (update weights) using the perturbed gradients.
    
    Args:
        params: Model parameters.
        base_optimizer: The underlying optimizer class (e.g. torch.optim.AdamW).
        rho (float): The neighborhood size for perturbation.
        adaptive (bool): If True, uses element-wise adaptive perturbation (ASAM).
        **kwargs: Arguments for the base optimizer.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(BSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Ascent Step: Perturb weights based on current gradients.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                
                # Perturbation e_w
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                
                # Apply perturbation: w = w + e_w
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Descent Step: Revert perturbation and update weights using gradients 
        calculated at the perturbed state.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # Revert: w = w - e_w (back to original state)
                p.data = self.state[p]["old_p"]

        # Base optimizer update
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        # Calculate norm of gradients for scaling
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("BSAM requires `first_step` and `second_step` calls manually.")


# --- 2. BSAM Trainer ---
class BSAMTrainer(CgcnnTrainer):
    """
    Extension of CgcnnTrainer to support the two-step BSAM optimization loop.
    """
    def train(self, epoch: int, dataloader: Optional[object] = None):
        """
        Overrides the standard training loop to implement the SAM/BSAM logic:
        1. Forward -> Loss -> Backward (Get Gradient)
        2. Optimizer.first_step() (Perturb Weights)
        3. Forward -> Loss -> Backward (Get Gradient at Perturbed State)
        4. Optimizer.second_step() (Update Weights)
        """
        if dataloader is None:
            dataloader = self.train_loader

        # Reset Meters
        self.meter_train_time.reset()
        self.meter_train_loss.reset()
        if self.meter_train_penalty: self.meter_train_penalty.reset()

        self.model.train()
        end = torch.cuda.Event(enable_timing=True)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        
        # DIL Warmup (if applicable)
        if self.dil_inform and hasattr(self.loss_func, "lambda_dcor"):
            warmup_factor = float(epoch) / float(self.warmup_epochs) if epoch < self.warmup_epochs else 1.0
            self.loss_func.lambda_dcor = self.dil_config.get("lambda", 1.0) * warmup_factor

        for idx, batch in enumerate(dataloader):
            batch = batch.to(self.device, non_blocking=True)
            
            # Helper to calculate loss
            def compute_loss():
                outputs = self.model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.state, batch.batch, batch.bond_batch
                )
                if outputs.ndim == 1: outputs = outputs.unsqueeze(1)
                if batch.y.ndim == 1: batch.y = batch.y.unsqueeze(1)

                current_weights = None
                if self.weighted_loss:
                    current_weights = batch.omega
                    if current_weights.ndim == 1: current_weights = current_weights.unsqueeze(1)
                    if current_weights.shape != outputs.shape: current_weights = current_weights.expand_as(outputs)

                current_rou = None
                if self.dil_inform:
                    current_rou = batch.rou
                    if current_rou.ndim == 1: current_rou = current_rou.unsqueeze(1)
                    if current_rou.shape[1] != outputs.shape[1]: current_rou = current_rou.expand_as(outputs)
                    
                    loss, penalty = self.loss_func(outputs, batch.y, current_rou, weights=current_weights)
                    return loss, penalty
                elif self.weighted_loss:
                    return self.loss_func(outputs, batch.y, current_weights), 0.0
                else:
                    return self.loss_func(outputs, batch.y), 0.0

            # --- STEP 1: Compute Gradients at w ---
            loss, penalty_val = compute_loss()
            
            if torch.isnan(loss) or loss.item() > 1e10:
                raise LossExplosionError(f"Loss explosion at step 1: {loss.item()}")
                
            loss.backward()
            
            # Optional: Clip gradients of the first step to stabilize the perturbation direction
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # --- STEP 2: Ascent (Perturb w -> w + e) ---
            self.optimiser.first_step(zero_grad=True)

            # --- STEP 3: Compute Gradients at w + e ---
            loss_perturbed, _ = compute_loss()
            loss_perturbed.backward()

            # --- STEP 4: Descent (Update w) ---
            self.optimiser.second_step(zero_grad=True)

            # Update Meters (logging the original loss, not perturbed)
            self.meter_train_loss.update(loss.item(), batch.y.size(0))
            if self.meter_train_penalty:
                self.meter_train_penalty.update(penalty_val.item(), batch.y.size(0))

        end.record()
        torch.cuda.synchronize()
        self.meter_train_time.update(start.elapsed_time(end) / 1000.0) # approx time
        
        if hasattr(self.model, "update_fds"):
            self.model.update_fds(dataloader, epoch)

        return self.meter_train_loss.avg, (self.meter_train_penalty.avg if self.meter_train_penalty else 0.0)


# --- 3. Main Execution Logic (Mirrors run_trainer.py) ---
def run_bsam(config_path, config_file):
    # Load Config
    with open(os.path.join(config_path, config_file)) as config:
        expt_config = yaml.full_load(config)

    # Basic Setup
    basedir = expt_config['save']['basedir']
    outdir = os.path.join(basedir, expt_config['save']['outdir'], 'BSAM')
    os.makedirs(outdir, exist_ok=True)
    
    # Force Weighted Loss for BSAM (Balanced SAM requires importance weights)
    if not expt_config['data'].get('reweight', False):
        logger.info("BSAM requires reweighting. Forcing 'log_inv' weights.")
        expt_config['data']['reweight'] = 'log_inv'
    expt_config['train']['weighted_loss'] = True

    # Setup Data Converters
    bond_centers = np.linspace(0, expt_config['data']['cutoff'], expt_config['data']['edge_embed_size'])
    if expt_config["data"]["add_z_bond_coord"]:
        bond_converter = FlattenGaussianDistanceConverter(centers=bond_centers)
    else:
        bond_converter = GaussianDistanceConverter(centers=bond_centers)
    atom_converter = AtomFeaturesExtractor(expt_config["data"]["atom_features"])
    target_name = expt_config['data']['target_name']

    # Default BSAM Parameters
    rho = expt_config.get('bsam', {}).get('rho', 0.05)
    adaptive = expt_config.get('bsam', {}).get('adaptive', False)
    
    logger.info(f"--- Starting BSAM Run (rho={rho}, adaptive={adaptive}) ---")

    for fold in expt_config["data"]["folds"]:
        model_name = f'fold_{fold}_bsam'
        seed_everything(expt_config['data']['seed'])
        
        # Load Data
        datafiles = {
            'train': os.path.join(expt_config['data']['data_loc'], f'fold_{fold}', 'train.pickle.gz'),
            'test': os.path.join(expt_config['data']['data_loc'], f'fold_{fold}', 'test.pickle.gz')
        }
        
        data_set_creator = CgcnnDataset(
            datafile=datafiles, target_name=target_name,
            bond_converter=bond_converter, atom_converter=atom_converter,
            random_seed=expt_config['data']['seed']
        )
        
        g = torch.Generator()
        g.manual_seed(expt_config['data']['seed'])
        train_loader, val_loader, test_loader = data_set_creator.prepare_data(
            reweight=expt_config['data']['reweight'], generator=g, worker_init_fn=seed_worker
        )

        # Initialize Model
        model = MEGNet(
            edge_input_shape=bond_converter.get_shape(),
            node_input_shape=atom_converter.get_shape(),
            state_input_shape=expt_config["model"]["state_input_shape"],
            device=expt_config["model"].get('device', None),
            fds=('fds' in expt_config), **expt_config.get('fds', {})
        )

        # Setup Optimizer: Wrap AdamW with BSAM
        base_optim_cls = torch.optim.AdamW
        base_optim_params = expt_config['optimiser'].get('parameters', {})
        
        # BSAM Optimizer
        optimiser = BSAM(
            model.parameters(), 
            base_optimizer=base_optim_cls,
            rho=rho,
            adaptive=adaptive,
            **base_optim_params
        )

        # Scheduler (Standard)
        sched_type = expt_config['scheduler'].get('name', 'CosineAnnealingLR')
        sched_params = expt_config['scheduler'].get('parameters', {})
        
        # Adjust scheduler defaults if needed
        if sched_type == 'CosineAnnealingLR' and 'T_max' not in sched_params:
            sched_params['T_max'] = expt_config['train']['epoch_range']

        scheduler = get_obj(sched_type)(optimiser.base_optimizer, **sched_params)

        # Loss Function (Must support weights for BSAM)
        loss_func = get_obj(expt_config['loss']['loss'])()
        
        # Initialize BSAM Trainer
        setup_file_logging(os.path.join(outdir, f'{model_name}.log'))
        
        trainer = BSAMTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            loss_func=loss_func, optimiser=optimiser, scheduler=scheduler, scheduler_type=sched_type,
            name=model_name, **expt_config['train'], outdir=outdir
        )
        if hasattr(model, 'FDS'): model.FDS.device = trainer.device

        logger.info(f"Running BSAM on Fold {fold}...")
        metrics = trainer.fit()
        
        # Save Results
        results_file = os.path.join(outdir, f'bsam_results.csv')
        final_metrics = metrics
        final_metrics.update({'fold': fold, 'rho': rho})
        pd.DataFrame([final_metrics]).to_csv(
            results_file, mode='a', header=not os.path.exists(results_file), index=False
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cd', type=str, default='expt_configs', help='Experiment configuration directory')
    parser.add_argument('--cf', type=str, required=True, help='Experiment configuration file')
    args = parser.parse_args()
    
    run_bsam(args.cd, args.cf)
