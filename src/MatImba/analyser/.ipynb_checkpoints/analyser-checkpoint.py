import os
import sys
import logging
import yaml
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

# Local imports 
from MatImba.dataset import CgcnnDataset
from MatImba.models import MEGNet
from MatImba.trainer import CgcnnTrainer
from MatImba.utils import (
    AtomFeaturesExtractor,
    FlattenGaussianDistanceConverter,
    GaussianDistanceConverter,
    SimpleCrystalConverter,
)
from MatImba.utils.evaluate import get_obj
from MatImba.utils.losses import calc_alpha, calc_ser_nd, calc_sera
from MatImba.vis import draw_y_equals_x

# Initialize logger
logger = logging.getLogger(__name__)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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


def evaluate_ckpt(
    ckpt_path: str, 
    config_file: str, 
    fold: int = 0,
    run_id: int = 0,
    data_loc_override: Optional[str] = None
) -> CgcnnTrainer:
    """
    Loads a model from a checkpoint and recreates the Trainer environment for evaluation.

    Args:
        ckpt_path (str): Path to the .pth.tar checkpoint file.
        config_file (str): Path to the experiment YAML config.
        fold (int): Fold index for data loading.
        data_loc_override (str, optional): Override data location from config.

    Returns:
        CgcnnTrainer: A trainer instance with the loaded model, ready for .predict().
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    logger.info(f"Loading configuration from {config_file}...")
    with open(config_file) as config:
        expt_config = yaml.full_load(config)

    # --- 1. Reconstruct Converters ---
    cutoff = expt_config['data']['cutoff']
    edge_embed_size = expt_config['data']['edge_embed_size']
    
    if expt_config["data"]["add_z_bond_coord"]:
        bond_converter = FlattenGaussianDistanceConverter(
            centers=np.linspace(0, cutoff, edge_embed_size)
        )
    else:
        bond_converter = GaussianDistanceConverter(
            centers=np.linspace(0, cutoff, edge_embed_size)
        )
        
    atom_converter = AtomFeaturesExtractor(expt_config["data"]["atom_features"])
    target_name = expt_config['data']['target_name']

    # --- 2. Prepare Data ---
    model_name_dir = f'fold_{fold}'
    base_data_loc = data_loc_override if data_loc_override else expt_config['data']['data_loc']

    datafiles = {
        'train': os.path.join(base_data_loc, model_name_dir, 'train.pickle.gz'),
        'test': os.path.join(base_data_loc, model_name_dir, 'test.pickle.gz')
    }

    # Validation check for data files
    if not os.path.exists(datafiles['test']):
        logger.warning(f"Test data not found at {datafiles['test']}. Dataloaders might be empty.")

    seed = expt_config["data"]["seed"] + run_id
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3. Initialize Model ---
    model = MEGNet(
        edge_input_shape=bond_converter.get_shape(),
        node_input_shape=atom_converter.get_shape(),
        state_input_shape=expt_config["model"]["state_input_shape"],
        device=device
    )

    # --- 4. Load Weights ---
    logger.info(f"Loading checkpoint weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Handle different checkpoint saving styles
    if "model" in checkpoint and "states" in checkpoint["model"]:
        # Format used in your trainer.py
        model.load_state_dict(checkpoint["model"]["states"], strict = False)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model"]["states"], strict = False)
    else:
        # Fallback: assume checkpoint IS the state dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    
    # --- 5. Initialize Trainer ---
    loss_func = get_obj(expt_config['loss']['loss'])()
    
    trainer = CgcnnTrainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        loss_func=loss_func, 
        name=f'eval_fold_{fold}',
        epoch_range=0, # No training intended
        weighted_loss=expt_config['train']['weighted_loss'], 
        dil_inform=expt_config['train']['dil_inform'],
        outdir=os.path.join(
            expt_config['save']['basedir'],
            expt_config['save']['outdir']
        )
    )
    
    return trainer

class ml_pred():
    def __init__(self, targets, preds, relevances, densities, train_log = None):
        self.targets = targets
        self.preds = preds
        self.maes = np.abs(targets - preds)
        self.relevances = relevances
        self.densities = densities
        self.train_log = train_log
        self.r2_score = r2_score(self.targets, self.preds)
        self.alpha = 1 - np.abs(np.corrcoef(1 / densities, self.maes)[0][-1])
        self.mae_wb()
        self.get_sers()
        self.sera = calc_sera(targets, preds, relevances)

    def mae_wb(self, bins = "fd"):
        self.hist, self.bin_edges = np.histogram(self.targets, bins = bins)
        self.x = (self.bin_edges[:-1] + self.bin_edges[1:])/2
        self.nbins = len(self.x)
        self.bin_width = (self.bin_edges[-1] - self.bin_edges[0]) / self.nbins
        label_locs = np.fmin(np.digitize(self.targets, self.bin_edges), self.nbins)
        self.binned_AEs = np.zeros(len(self.x))
        
        for j in range(len(self.x)):
            locs = np.where(label_locs == j + 1)
            if np.any(locs):
                self.binned_AEs[j] = self.maes[locs].mean()
            else:
                self.binned_AEs[j] = np.nan
            
    def get_sers(self, sampling = 50):
        self.t_s = np.linspace(0, 1, sampling)
        self.sers = np.zeros(sampling)
        
        for j, t in enumerate(self.t_s):
            self.sers[j] = calc_ser_nd(self.targets, self.preds, self.relevances, t)
    
    def save(self, filepath):
        """
        Saves the prediction results and all calculated metrics to a compressed 
        NumPy archive (.npz) for fast retrieval.
        """
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # We pack all scalar and string values into 0-dimensional arrays so npz accepts them
        data_dict = {
            'targets': self.targets,
            'preds': self.preds,
            'maes': self.maes,
            'relevances': self.relevances if self.relevances is not None else np.array([None]),
            'densities': self.densities if self.densities is not None else np.array([None]),
            'train_log': self.train_log if self.train_log is not None else "NONE",
            'r2_score': np.array(self.r2_score),
            'alpha': np.array(self.alpha),
            'sera': np.array(self.sera),
            'hist': self.hist,
            'bin_edges': self.bin_edges,
            'x': self.x,
            'nbins': np.array(self.nbins),
            'bin_width': np.array(self.bin_width),
            'binned_AEs': self.binned_AEs,
            't_s': self.t_s,
            'sers': self.sers
        }
        np.savez_compressed(filepath, **data_dict)
        print(f"Saved ml_pred data to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Instantiates an ml_pred object directly from a saved .npz file, bypassing 
        __init__ to avoid redundant and expensive recalculations.
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Create an empty instance without triggering __init__
        instance = cls.__new__(cls)
        
        # Restore Core Arrays
        instance.targets = data['targets']
        instance.preds = data['preds']
        instance.maes = data['maes']
        
        # Safely handle 'None' values saved as string/object fallbacks
        instance.relevances = None if (data['relevances'].size == 1 and data['relevances'][0] is None) else data['relevances']
        instance.densities = None if (data['densities'].size == 1 and data['densities'][0] is None) else data['densities']
        
        log_val = str(data['train_log'])
        instance.train_log = None if log_val == "NONE" else log_val
        
        # Restore pre-calculated Metrics and Bins
        instance.r2_score = float(data['r2_score'])
        instance.alpha = float(data['alpha'])
        instance.sera = float(data['sera'])
        
        instance.hist = data['hist']
        instance.bin_edges = data['bin_edges']
        instance.x = data['x']
        instance.nbins = int(data['nbins'])
        instance.bin_width = float(data['bin_width'])
        instance.binned_AEs = data['binned_AEs']
        
        instance.t_s = data['t_s']
        instance.sers = data['sers']
        
        return instance

    @classmethod
    def from_csv(cls, filepath, train_log=None):
        """
        Instantiates an ml_pred object dynamically from a standard test predictions .csv file.
        Expects columns: 'labels', 'predictions', 'relevance', 'density'.
        """
        df = pd.read_csv(filepath)
        
        targets = df['labels'].values
        preds = df['predictions'].values
        
        relevances = df['relevance'].values if 'relevance' in df.columns else None
        densities = df['density'].values if 'density' in df.columns else None
        
        return cls(targets, preds, relevances, densities, train_log=train_log)


class CombinedPred:
    """Helper class to cleanly concatenate multiple runs sharing the same test set."""
    def __init__(self, preds_list):
        self.targets = np.concatenate([p.targets for p in preds_list])
        self.preds = np.concatenate([p.preds for p in preds_list])
        if hasattr(preds_list[0], 'relevances') and preds_list[0].relevances is not None:
            self.relevances = np.concatenate([p.relevances for p in preds_list])
        else:
            self.relevances = None
            
        # We assume the binning structure (x, hist, bin_width) is identical across 
        # runs because they share the exact same test set.
        if hasattr(preds_list[0], 'x'):
            self.x = preds_list[0].x
            self.hist = preds_list[0].hist
            self.bin_width = preds_list[0].bin_width

class imba_analyser():
    def __init__(self, *ml_preds, labels=None, outdir=None):
        """
        Accepts individual prediction objects OR lists of prediction objects.
        e.g., imba_analyser([run1, run2, run3], [ctrl1, ctrl2, ctrl3])
        """
        self.num_preds = len(ml_preds)
        if self.num_preds == 0:
            print("No model prediction specified!")
        else:
            self.all_preds = ml_preds
        
        self.labels = labels if labels is not None else [f"Model {i}" for i in range(self.num_preds)]
        self.outdir = os.getcwd() if outdir is None else outdir
        self.results = {}

    # =========================================================
    # NEW: Robust filter to drop exploded runs automatically
    # =========================================================
    def _filter_bad_runs(self, runs, r2_drop_threshold=0.05):
        """Filters out runs that have an R2 score significantly worse than the ensemble best."""
        if not isinstance(runs, (list, tuple, np.ndarray)) or len(runs) < 2:
            return runs
            
        r2_scores = []
        for p in runs:
            if hasattr(p, 'r2_score') and p.r2_score is not None:
                r2_scores.append(float(p.r2_score))
            else:
                # Compute R2 on the fly if not explicitly saved
                r2_scores.append(r2_score(np.ravel(p.targets), np.ravel(p.preds)))
                
        best_r2 = np.max(r2_scores)
        valid_runs = []
        
        for i, p in enumerate(runs):
            if r2_scores[i] >= (best_r2 - r2_drop_threshold):
                valid_runs.append(p)
            else:
                print(f"--> Dropped exploded run from ensemble: R2={r2_scores[i]:.3f} (Ensemble Best={best_r2:.3f})")
                
        # Fallback to all runs if everything somehow failed
        final_runs = valid_runs if valid_runs else runs
        return valid_runs if valid_runs else runs

    def _ensure_combined(self, pred_input):
        """Checks if input is a list of runs. If so, filters bad ones and concatenates."""
        if isinstance(pred_input, (list, tuple, np.ndarray)):
            filtered_input = self._filter_bad_runs(pred_input)
            return CombinedPred(filtered_input)
        return pred_input

    def plot_logs(self, *ml_preds, skip=25, model_names=None, file_names=None):
        if len(ml_preds) == 0:
            ml_preds = self.all_preds
        
        if model_names is None:
            model_names = self.labels
            
        # Extract the first VALID run using the filter
        first_runs = [self._filter_bad_runs(p)[0] if isinstance(p, (list, tuple, np.ndarray)) else p for p in ml_preds]
        train_logs = [pred.train_log for pred in first_runs if hasattr(pred, 'train_log') and pred.train_log is not None]
        
        for i, train_log in enumerate(train_logs):
            log_df = pd.read_csv(train_log)
            fig, ax = plt.subplots(figsize=(3.8, 2.8))
            sc = ax.scatter(log_df["awareness"][skip:], log_df["r2_score"][skip:], c=log_df["epoch"][skip:], cmap="RdYlBu")
            cbar = plt.colorbar(sc, label="Epoch")
            ax.set_xlabel("DIL awareness", fontsize=10)
            ax.set_ylabel("R2 Score", fontsize=10)
            if model_names:
                ax.text(0.25, 0.9, model_names[i], va="center", ha="center", transform=ax.transAxes)
                        
            plt.tight_layout(pad=0.5)
            
            if file_names is not None:
                plt.savefig(os.path.join(self.outdir, file_names[i]), dpi=600)
            plt.show()

    def calculate_adaptive_discovery_metrics(self, ml_pred, rel_threshold=0.5, 
                                             budget_ratio=1.0, discover_mode='auto'):
        ml_pred = self._ensure_combined(ml_pred) # Filter is applied here automatically
        
        targets = np.ravel(ml_pred.targets)
        preds = np.ravel(ml_pred.preds)
        relevances = np.ravel(ml_pred.relevances)

        if relevances is None:
             return {'Error': 'No relevance scores found in ml_pred.'}
        
        is_relevant = relevances > rel_threshold
        if not np.any(is_relevant):
            return {'Error': f'No samples found with relevance > {rel_threshold}'}

        head_mask = ~is_relevant
        head_median = np.median(targets[head_mask]) if np.any(head_mask) else np.median(targets)

        gt_high_mask = is_relevant & (targets >= head_median)
        gt_low_mask  = is_relevant & (targets < head_median)
        
        n_high = np.sum(gt_high_mask)
        n_low  = np.sum(gt_low_mask)
        
        mode = discover_mode.lower()
        total_relevant = np.sum(is_relevant)

        if mode == 'auto':
            has_high = n_high > (0.05 * total_relevant)
            has_low  = n_low  > (0.05 * total_relevant)
            
            if has_high and has_low: mode = 'both'
            elif has_low:            mode = 'low'
            else:                    mode = 'high'

        all_indices = np.arange(len(targets))
        candidate_indices = set()
        target_gt_indices = set()
        
        def select_candidates(gt_mask, ascending):
            n_target = np.sum(gt_mask)
            if n_target == 0: return 0
            
            k = int(np.ceil(n_target * budget_ratio))
            k = min(k, len(targets))
            
            selection = np.argsort(preds)[:k] if ascending else np.argsort(preds)[-k:]
            candidate_indices.update(selection)
            target_gt_indices.update(all_indices[gt_mask]) 
            return k

        k_used_total = 0
        if mode == 'both':
            k_used_total += select_candidates(gt_high_mask, ascending=False)
            k_used_total += select_candidates(gt_low_mask, ascending=True)
        elif mode == 'high':
            k_used_total += select_candidates(gt_high_mask, ascending=False)
        elif mode == 'low':
            k_used_total += select_candidates(gt_low_mask, ascending=True)
            
        if len(target_gt_indices) == 0:
            return {'Error': f'No relevant samples found in mode "{mode}"'}

        hits = len(candidate_indices.intersection(target_gt_indices))
        precision = hits / k_used_total if k_used_total > 0 else 0.0
        recall = hits / len(target_gt_indices)
        
        tail_mask = (gt_high_mask | gt_low_mask) if mode == 'both' else (gt_high_mask if mode == 'high' else gt_low_mask)
        tail_mae = np.mean(np.abs(targets[tail_mask] - preds[tail_mask]))

        return {
            'Discovery Mode': mode,
            'Tail Size (GT)': len(target_gt_indices),
            'Budget Ratio': f"{budget_ratio}x ({k_used_total} tests)",
            'Extrap. Precision': round(precision, 4),
            'Tail Recall': round(recall, 4),
            'Tail MAE': round(tail_mae, 4),
            'Breakdown': f'High={n_high}, Low={n_low}'
        }

    def parity_plot(
        self, *ml_preds, target_name="", model_names=None, combined=True,
        colors='#4575b4', xeqy_cs='#cc7c71', axes=None, linewidths=0
    ):
        if len(ml_preds) == 0: ml_preds = self.all_preds
        if model_names is None: model_names = self.labels

        if combined and len(ml_preds) > 1:
            if axes is None:
                n_cols = 2
                n_rows = int(np.ceil(len(ml_preds) / n_cols))
                fig, axes = plt.subplots(n_cols, n_rows, layout='compressed', figsize=(6, 6))
                axes = axes.flatten()
                
            colors = [colors] * len(ml_preds) if isinstance(colors, str) else colors
            xeqy_cs = [xeqy_cs] * len(ml_preds) if isinstance(xeqy_cs, str) else xeqy_cs
                
            for i, pred_group in enumerate(ml_preds):
                ml_pred = self._ensure_combined(pred_group) # Filter applied automatically
                
                axes[i].scatter(ml_pred.targets, ml_pred.preds, color=colors[i], linewidths=linewidths, alpha=0.5, s=15)
                draw_y_equals_x(axes[i], colour=xeqy_cs[i])
                
                axes[i].set_xlabel(f"True {target_name}")
                axes[i].set_ylabel(f"Predicted {target_name}")
                mae = np.mean(np.abs(ml_pred.targets - ml_pred.preds))
                axes[i].text(0.3, 0.90, "MAE = %.2f" % mae, va="center", ha="center", fontsize=10, transform=axes[i].transAxes)
                axes[i].text(0.8, 0.15, model_names[i], va="center", ha="center", fontsize=10, transform=axes[i].transAxes)

        else:
            if axes is None:
                fig, ax = plt.subplots(figsize=(3.0, 2.8), layout="compressed")
            else:
                ax = axes[0] if hasattr(axes, '__len__') else axes
                
            for i, pred_group in enumerate(ml_preds):
                ml_pred = self._ensure_combined(pred_group)
                ax.scatter(ml_pred.targets, ml_pred.preds, color=colors, linewidths=linewidths, alpha=0.5, s=15)
                draw_y_equals_x(ax, colour=xeqy_cs)
                ax.set_xlabel(f"True {target_name}", fontsize=10)
                ax.set_ylabel(f"Predicted {target_name}", fontsize=10)
                mae = np.mean(np.abs(ml_pred.targets - ml_pred.preds))
                ax.text(0.3, 0.90, "MAE = %.2f" % mae, va="center", ha="center", transform=ax.transAxes)
                if model_names:
                    ax.text(0.8, 0.15, model_names[i], va="center", ha="center", fontsize=10, transform=ax.transAxes)

    def compare_mae_wb(self, *ml_preds_groups, model_names=None, bins="fd", target_name="", ax=None):
        if len(ml_preds_groups) == 0: ml_preds_groups = self.all_preds
        if model_names is None: model_names = self.labels
        if ax is None: fig, ax = plt.subplots(figsize=(3.2, 2.8), layout="compressed")
        
        axtwin = ax.twinx()
        markers = ['s', 'o', '^', 'v', '<', '>', 'd', 'p', '*', 'h']
        e_colors = ['#4d4d4d', '#2166ac', '#b2182b', '#35978f', '#ff7f0e'] 
        f_colors = ['#e0e0e0', '#d1e5f0', '#fff5eb', '#c7eae5', '#ffbb78']
        
        hist_plotted = False
        global_max_mae = 0 

        for i, pred_group in enumerate(ml_preds_groups):
            model_name = model_names[i] if i < len(model_names) else f"model_{i}"
            is_ensemble = isinstance(pred_group, (list, tuple, np.ndarray))
            runs = list(pred_group) if is_ensemble else [pred_group]
            
            # Filter Bad Runs explicitly for iterating over single runs
            if is_ensemble:
                runs = self._filter_bad_runs(runs)
            
            base = runs[0]
            valid_mask = ~np.isnan(base.binned_AEs)
            x_vals = base.x[valid_mask]
            
            if not hist_plotted:
                axtwin.bar(x_vals, base.hist[valid_mask], color='#4d4d4d', alpha=0.15, 
                           width=base.bin_width, linewidth=0)
                hist_plotted = True

            all_binned = np.vstack([p.binned_AEs[valid_mask] for p in runs])
            
            median_binned = np.nanmedian(all_binned, axis=0)
            
            safe_max = np.nanpercentile(median_binned, 95) * 1.5 
            global_max_mae = max(global_max_mae, safe_max)
            
            c_idx = i % len(e_colors)
            
            if len(runs) > 1:
                lower_bound = np.nanpercentile(all_binned, 25, axis=0) 
                upper_bound = np.nanpercentile(all_binned, 75, axis=0) 
                
                upper_bound = np.clip(upper_bound, a_min=None, a_max=safe_max * 2)
                
                yerr_lower = np.maximum(median_binned - lower_bound, 0)
                yerr_upper = np.maximum(upper_bound - median_binned, 0)
                
                ax.errorbar(x_vals, median_binned, yerr=[yerr_lower, yerr_upper], 
                            c=e_colors[c_idx], marker=markers[c_idx], ms=6,
                            markerfacecolor=f_colors[c_idx], alpha=0.8, label=model_name, 
                            linestyle='none', capsize=0, elinewidth=1.5)
            else:
                ax.plot(x_vals, median_binned, c=e_colors[c_idx], marker=markers[c_idx], ms=6,
                        markerfacecolor=f_colors[c_idx], alpha=0.8, label=model_name, linestyle='none')

        axtwin.set_ylabel(r"Testset Counts", fontsize=10)
        ax.set_xlabel(target_name if target_name else r"$y$", fontsize=10)
        ax.set_ylabel(r"Median MAE (Test)", fontsize=10)
        ax.set_zorder(axtwin.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.grid(False)
        axtwin.grid(False)
        
        if global_max_mae > 0:
            ax.set_ylim(-0.05 * global_max_mae, global_max_mae)
            
        ax.legend(fontsize=8, loc='upper center', handletextpad=0.5, borderpad=0.2, borderaxespad=0.3, labelspacing=0.6)

    def plot_sers(self, *ml_preds_groups, model_names=None, target_name="", ax=None):
        if len(ml_preds_groups) == 0: ml_preds_groups = self.all_preds
        if model_names is None: model_names = self.labels
            
        colors = ['#4d4d4d', '#2166ac', '#b2182b', '#35978f', '#ff7f0e']
        linestyles = ['-', (0, (5,5)), (0, (3,5,1,5)), (0, (5,1)), (0, (1,1))]
        if ax is None: fig, ax = plt.subplots(figsize=(3.0, 2.8), layout="compressed")
        
        global_max_ser = 0
        
        for i, pred_group in enumerate(ml_preds_groups):
            model_name = model_names[i] if i < len(model_names) else f"Model {i}"
            is_ensemble = isinstance(pred_group, (list, tuple, np.ndarray))
            runs = list(pred_group) if is_ensemble else [pred_group]
            
            # Filter Bad Runs explicitly for iterating over single runs
            if is_ensemble:
                runs = self._filter_bad_runs(runs)
            
            t_s = runs[0].t_s
            all_sers = np.vstack([p.sers for p in runs])
            
            median_sers = np.nanmedian(all_sers, axis=0)
            
            safe_max = np.nanmax(median_sers[:-int(len(median_sers)*0.05)]) * 1.5
            if not np.isnan(safe_max):
                global_max_ser = max(global_max_ser, safe_max)
            
            c_idx = i % len(colors)
            ax.plot(t_s, median_sers, ls=linestyles[c_idx], color=colors[c_idx], 
                    lw=2, alpha=0.8, label=model_name)
            
            if len(runs) > 1:
                lower_bound = np.nanpercentile(all_sers, 25, axis=0)
                upper_bound = np.nanpercentile(all_sers, 75, axis=0)
                ax.fill_between(t_s, lower_bound, upper_bound, 
                                color=colors[c_idx], alpha=0.15, linewidth=0)
                
        if target_name:
            clean_target = target_name.replace('$', '')
            clean_target = clean_target.replace('{{', '{').replace('}}', '}')
            ax.set_xlabel(rf"Relevance $\phi({clean_target})$")
        else:
            ax.set_xlabel(r"Relevance $\phi(y)$", fontsize=10)
            
        ax.set_ylabel("Median SER", fontsize=10)
        
        if global_max_ser > 0:
            ax.set_ylim(-0.05 * global_max_ser, global_max_ser)
            
        ax.legend(fontsize=8, loc='upper right', handletextpad=0.5, borderpad=0.2, borderaxespad=0.3, labelspacing=0.6)

    def _get_tail_mask(self, ml_pred, config):
        key, val = list(config.items())[0]
        
        if key == 'quantile':
            threshold = np.quantile(ml_pred.targets, val)
            return ml_pred.targets > threshold, threshold, 'targets'
        elif key == 'relevance':
            if ml_pred.relevances is None: raise ValueError("Missing 'relevance' in ml_pred object.")
            return ml_pred.relevances > val, val, 'relevance'
        elif key in ['labels', 'absolute', 'targets']:
            return ml_pred.targets > val, val, 'targets'
        else:
            raise ValueError(f"Unknown threshold type: {key}")

    def _split_data(self, ml_pred, threshold_config):
        if threshold_config is None: threshold_config = {'quantile': 0.90}
        tail_mask, threshold_val, threshold_col = self._get_tail_mask(ml_pred, threshold_config)
        
        def get_slice(arr, mask):
            return arr[mask] if arr is not None else np.array([])

        return {
            'tail': {
                'preds': get_slice(ml_pred.preds, tail_mask), 
                'gt': get_slice(ml_pred.targets, tail_mask),
                'relevance': get_slice(ml_pred.relevances, tail_mask)
            },
            'head': {
                'preds': get_slice(ml_pred.preds, ~tail_mask), 
                'gt': get_slice(ml_pred.targets, ~tail_mask),
                'relevance': get_slice(ml_pred.relevances, ~tail_mask)
            },
            'metadata': {
                'threshold_val': threshold_val,
                'threshold_col': threshold_col,
                'config': threshold_config,
                'total_count': len(ml_pred.targets)
            }
        }

    def plot_marginal_kde(self, ax, gt, preds, n_total, subset_label, color, 
                          orientation='horizontal', head_bounds=None):
        if len(gt) < 5: 
            print(f"Not enough {subset_label} samples for KDE.")
            return

        def plot_single_kde_curve(data, weight, label, c, linestyle='-', alpha=1.0, is_filled=False):
            try:
                kde = gaussian_kde(data)
                buffer = (data.max() - data.min()) * 0.1
                if buffer == 0: buffer = 1e-6
                grid = np.linspace(data.min() - buffer, data.max() + buffer, 500)
                density = kde(grid) * weight 

                if orientation == 'horizontal':
                    if is_filled:
                        ax.fill_between(grid, density, color=c, alpha=alpha, label=label)
                    else:
                        ax.plot(grid, density, color=c, linestyle=linestyle, lw=1.5, label=label)
                else: 
                    if is_filled:
                        ax.fill_betweenx(grid, density, color=c, alpha=alpha, label=label)
                    else:
                        ax.plot(density, grid, color=c, linestyle=linestyle, lw=1.5, label=label)
            except np.linalg.LinAlgError:
                print(f"Singular matrix in KDE for {label}")

        is_two_sided = False
        if head_bounds is not None and subset_label == 'Tail':
            head_min, head_max = head_bounds
            low_mask = gt < head_min
            high_mask = gt > head_max
            
            if np.sum(low_mask) > 2 and np.sum(high_mask) > 2:
                is_two_sided = True
                w_low = np.sum(low_mask) / n_total
                plot_single_kde_curve(gt[low_mask], w_low, 'Low Tail GT', '#1f77b4', linestyle='--')
                plot_single_kde_curve(preds[low_mask], w_low, 'Low Tail Pred', '#1f77b4', alpha=0.3, is_filled=True)

                w_high = np.sum(high_mask) / n_total
                plot_single_kde_curve(gt[high_mask], w_high, 'High Tail GT', '#d62728', linestyle='--')
                plot_single_kde_curve(preds[high_mask], w_high, 'High Tail Pred', '#d62728', alpha=0.3, is_filled=True)

        if not is_two_sided:
            weight = len(gt) / n_total
            plot_single_kde_curve(gt, weight, f'{subset_label} Truth', color, linestyle='--')
            plot_single_kde_curve(preds, weight, f'{subset_label} Pred', color, alpha=0.3, is_filled=True)

        ax.grid(True, alpha=0.2)
        if orientation == 'horizontal':
            ax.set_yticks([])
            ax.text(0.125, 0.5, 'Tail', va='center', ha='center', transform=ax.transAxes)
        else:
            ax.set_xticks([])
            ax.set_xlabel(r'$\rho$', labelpad=9, fontsize=10)
            ax.xaxis.set_label_position("top")
            ax.text(0.5, 0.825, 'Head', va='center', ha='center', transform=ax.transAxes, rotation = 90)

    def plot_split_parity(self, ax, split_data, target_name = "", model_name = None):
        head_gt = split_data['head']['gt']
        head_preds = split_data['head']['preds']
        tail_gt = split_data['tail']['gt']
        tail_preds = split_data['tail']['preds']
        
        head_min, head_max = np.min(head_gt), np.max(head_gt)
        
        mae_head = np.mean(np.abs(head_gt - head_preds))
        ax.scatter(head_gt, head_preds, c='gray', alpha=0.3, s=20, label=f'Head: {mae_head:.2f}')

        low_mask = tail_gt < head_min
        high_mask = tail_gt > head_max
        
        if np.any(low_mask):
            mae_low = np.mean(np.abs(tail_gt[low_mask] - tail_preds[low_mask]))
            ax.scatter(tail_gt[low_mask], tail_preds[low_mask], 
                       c='#1f77b4', alpha=0.6, s=20, label=f'Low Tail: {mae_low:.2f}')
            ax.axvline(head_min, color='k', ls=':', lw=1.5)
            ax.axhline(head_min, color='k', ls=':', lw=1.5)

        if np.any(high_mask):
            mae_high = np.mean(np.abs(tail_gt[high_mask] - tail_preds[high_mask]))
            ax.scatter(tail_gt[high_mask], tail_preds[high_mask], 
                       c='#d62728', alpha=0.6, s=20, label=f'High Tail: {mae_high:.2f}')
            ax.axvline(head_max, color='k', ls=':', lw=1.5)
            ax.axhline(head_max, color='k', ls=':', lw=1.5)
        
        other_mask = ~(low_mask | high_mask)
        if np.any(other_mask):
             mae_other = np.mean(np.abs(tail_gt[other_mask] - tail_preds[other_mask]))
             ax.scatter(tail_gt[other_mask], tail_preds[other_mask], 
                        c='orange', alpha=0.6, s=20, label=f'Other Tail: {mae_other:.2f}')

        all_vals = np.concatenate([head_gt, tail_gt, head_preds, tail_preds])
        min_v, max_v = np.min(all_vals), np.max(all_vals)
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)

        ax.legend(fontsize=8, loc='upper left', handletextpad=0.5, borderpad=0.2, borderaxespad=0.3, labelspacing=0.6)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(f'True {target_name}', fontsize=10)
        ax.set_ylabel(f'Pred {target_name}', fontsize=10)

    def create_composite_plot(self, ml_pred=None, axes=None, target_name="", threshold_config={'relevance': 0.80}, title=None, file_name=None):
        if ml_pred is None: return
        
        ml_pred_combined = self._ensure_combined(ml_pred) # Filter is applied here automatically
        split_data = self._split_data(ml_pred_combined, threshold_config)
        
        n_total = split_data['metadata']['total_count']
        head_gt = split_data['head']['gt']
        
        if len(head_gt) > 0:
            head_bounds = (head_gt.min(), head_gt.max())
        else:
            head_bounds = (split_data['tail']['gt'].min(), split_data['tail']['gt'].max())

        is_standalone = axes is None
        if is_standalone:
            fig = plt.figure(figsize=(3.5, 3))
            gs = gridspec.GridSpec(2, 2, width_ratios=[7, 1], height_ratios=[1, 7], wspace=0.05, hspace=0.05)
            ax_main = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        else:
            try: ax_main, ax_top, ax_right = axes
            except: ax_top, _, ax_main, ax_right = axes

        self.plot_split_parity(ax_main, split_data, target_name)
        
        self.plot_marginal_kde(ax_top, split_data['tail']['gt'], split_data['tail']['preds'], 
                               n_total, 'Tail', '#d62728', orientation='horizontal', head_bounds=head_bounds)
        
        self.plot_marginal_kde(ax_right, split_data['head']['gt'], split_data['head']['preds'], 
                               n_total, 'Head', 'gray', orientation='vertical', head_bounds=None)

        if title: ax_top.set_title(title, loc='center', fontsize=10)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        if file_name: plt.savefig(os.path.join(self.outdir, file_name), dpi=300, bbox_inches='tight')
        if is_standalone: plt.show()

    def sumarry_plot(self, *ml_preds_groups, model_names=None, target_name="", parity_titles=None, file_name=None, fig_labels=None):
        if len(ml_preds_groups) == 0: ml_preds_groups = self.all_preds
        num_plots = len(ml_preds_groups) + 2
        ncols = 3 if num_plots != 4 else 2
        nrows = int(np.ceil(num_plots / ncols))
        
        fig = plt.figure(figsize=((2.8 + 0.15) * ncols + 0.15, (2.75 + 0.1) * nrows + 0.1))
        gs_main = gridspec.GridSpec(nrows, ncols, figure=fig)
        label_axes = []
        
        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            if i < len(ml_preds_groups):
                inner_gs = gridspec.GridSpecFromSubplotSpec(8, 8, subplot_spec=gs_main[i])
                ax_top = fig.add_subplot(inner_gs[0, 0:7])
                ax_main = fig.add_subplot(inner_gs[1:8, 0:7])
                ax_right = fig.add_subplot(inner_gs[1:8, 7])
                
                ax_top.sharex(ax_main)
                ax_right.sharey(ax_main)
                title = parity_titles[i] if parity_titles else None
                
                self.create_composite_plot(ml_preds_groups[i], axes=[ax_main, ax_top, ax_right], target_name=target_name, title=title)
                label_axes.append(ax_top)

            elif i == len(ml_preds_groups):
                ax_ser = fig.add_subplot(gs_main[r, c])
                self.plot_sers(*ml_preds_groups, model_names=model_names, target_name=target_name, ax=ax_ser)
                label_axes.append(ax_ser)

            elif i == len(ml_preds_groups) + 1:
                ax_mae = fig.add_subplot(gs_main[r, c])
                self.compare_mae_wb(*ml_preds_groups, target_name=target_name, model_names=model_names, ax=ax_mae)
                label_axes.append(ax_mae)

        if fig_labels is None:
            start = ord('a')
            for i, ax in enumerate(label_axes):
                ax.set_title(chr(start + i), loc='left', fontsize=10, y=1.05)
        plt.tight_layout()
        if file_name: plt.savefig(file_name, dpi=600, bbox_inches='tight')
        plt.show()