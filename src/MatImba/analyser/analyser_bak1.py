import os
import sys
import logging
import yaml
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import r2_score

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

class imba_analyser():
    def __init__(self, *ml_preds, labels = None, outdir = None):
        self.num_preds = len(ml_preds)
        if self.num_preds == 0:
            print("No model prediction specified!")
        else:
            self.all_preds = ml_preds
        
        self.labels = labels if labels is not None else [f"Model {i}" for i in range(self.num_preds)]
        self.outdir = os.getcwd() if outdir is None else outdir
        self.results = {}

    def plot_logs(self, *ml_preds, skip = 25, model_names = None, file_names = None):
        if len(ml_preds) == 0:
            ml_preds = self.all_preds
        
        if model_names is None:
            model_names = self.labels
        train_logs = [pd.read_csv(ml_pred.train_log) for ml_pred in ml_preds if ml_pred.train_log is not None]
        for i, train_log in enumerate(train_logs):
            log_df = pd.read_csv(train_log)
            fig, ax = plt.subplots(figsize = (3.8, 2.8))
            sc = plt.scatter(log_df["awareness"][skip:], log_df["r2_score"][skip:], c = log_df["epoch"][skip:], cmap = "RdYlBu")
            cbar = plt.colorbar(sc, label = "Epoch")
            ax.set_xlabel("DIL awareness")
            ax.set_ylabel("R2 Score")
            if model_names:
                ax.text(0.25, 0.9, model_names[i], va = "center", ha = "center", transform = ax.transAxes)
                        
            plt.tight_layout(pad = 0.5)
            
            if file_names is not None:
                plt.savefig(os.path.join(self.outdir, file_names[i]), dpi = 600)
            plt.show()
        

    def parity_plot(
        self, *ml_preds, target_name = "", model_names = None, combined = True,
        colors = '#4575b4', xeqy_cs = '#cc7c71', axes = None, linewidths = 0
    ):
        if len(ml_preds) == 0:
            ml_preds = self.all_preds
        
        if model_names is None:
            model_names = self.labels

        if combined and len(ml_preds) > 1:
            if axes is None:
                n_cols = 2
                n_rows = int(np.ceil(len(ml_preds) / n_cols))
                width = 0.85*3
                height = 0.9*2.8
                w_space = 0.05*3
                h_space = 0.035*2.8
                fig, axes = plt.subplots(
                    n_cols, n_rows, layout = 'compressed',
                    figsize = ((width + w_space) * n_cols + w_space, (height + h_space) * n_cols + h_space)
                )
                axes = axes.flatten()
            if isinstance(colors, str):
                colors = [colors] * len(ml_preds)
            if isinstance(xeqy_cs, str):
                xeqy_cs = [xeqy_cs] * len(ml_preds)
                
            for i, ml_pred in enumerate(ml_preds):
                
                axes[i].scatter(ml_pred.targets, ml_pred.preds, color = colors[i], linewidths = linewidths, alpha = 0.75)
                draw_y_equals_x(axes[i], colour = xeqy_cs[i])
                xaxis_label = target_name
                axes[i].set_xlabel(f"True {xaxis_label}")
                axes[i].set_ylabel(f"Predicted {xaxis_label}")
                mae = np.abs(ml_pred.targets - ml_pred.preds).mean()
                axes[i].text(0.3, 0.90, "MAE = %.2f" % mae, va = "center", ha = "center", fontsize = 10, transform = axes[i].transAxes)
                axes[i].text(0.8, 0.15, model_names[i], va = "center", ha = "center", fontsize = 10, transform = axes[i].transAxes)

        else:
            if axes is None:
                fig, axes = plt.subplots(figsize = (3.0, 2.8), layout = "compressed")
            elif hasattr(axes, '__len__'):
                axes = axes[0]
                
            for i, ml_pred in enumerate(ml_preds):
                axes.scatter(ml_pred.targets, ml_pred.preds, color = color, linewidths = linewidths, alpha = 0.65)
                draw_y_equals_x(axes, colour = xeqy_c)
                xaxis_label = target_name
                axes.set_xlabel(f"True {xaxis_label}")
                axes.set_ylabel(f"Predicted {xaxis_label}")
                mae = np.abs(ml_pred.targets - ml_pred.preds).mean()
                axes.text(0.3, 0.90, "MAE = %.2f" % mae, va = "center", ha = "center", transform = axes.transAxes)
                if model_names:
                    axes.text(0.8, 0.15, model_names[i], va = "center", ha = "center", fontsize = 10, transform = axes[i].transAxes)
            

    def mae_wb(self, *ml_preds, target_name = "", model_names = None,
               bins = "fd", ax = None, file_names = None):
        
        if len(ml_preds) == 0:
            ml_preds = self.all_preds
        
        if model_names is None:
            model_names = self.labels

        for i, ml_pred in enumerate(ml_preds):
            model_name = model_names[i] if model_names is not None else f"model_{i}"
            if ax is None:
                fig, ax = plt.subplots(figsize = (3.2, 2.8))
            ax.bar(ml_pred.x, ml_pred.hist, color = '#92c5de', width = 0.85 * ml_pred.bin_width, linewidth = 0, alpha = 0.85)
        
            xaxis_label = "Target" if target_name == "" else target_name
                
            ax.set_xlabel(xaxis_label)
            ax.set_ylabel(r"Testset Counts")
            ax.tick_params(axis='y', colors='#2166ac')
            ax.yaxis.label.set_color('#2166ac')
            
            x = ml_pred.x[~np.isnan(ml_pred.binned_AEs)]
            hist = ml_pred.hist[~np.isnan(ml_pred.binned_AEs)]
            binned_AEs = ml_pred.binned_AEs[~np.isnan(ml_pred.binned_AEs)]

            axtwin = ax.twinx()
            axtwin.plot(x, binned_AEs, c = "#b2182b", marker="s", markerfacecolor="#fddbc7")
        
            axtwin.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Test}}$ ")
            axtwin.tick_params(axis='y', colors='#b2182b')
            axtwin.yaxis.label.set_color('#b2182b')
            l_preds[i]
            axtwin.set_position([0.18, 0.18, 0.65, 0.75])
            plt.tight_layout(pad = 0.5)
               
            if file_names is not None:
                plt.savefig(os.path.join(self.outdir, file_names[i]), dpi=600)
            plt.show()
    
    def compare_mae_wb(self, *ml_preds, model_names = None, bins = "fd",
               target_name = "", ax = None):
        
        if len(ml_preds) == 0:
            ml_preds = self.all_preds

        if model_names is None:
            model_names = self.labels
        if ax is None:
            fig, ax = plt.subplots(figsize = (3.2, 2.8), layout = "compressed")
        
        axtwin = ax.twinx()
        markers = ['s', 'o', '^', 'v', '<', '>', 'd', 'p', '*', 'h', 'H', '8', 'P', 'X']
        e_colors = ['#4d4d4d', '#2166ac', '#b2182b', '#35978f'] 
        f_colors = ['#e0e0e0', '#d1e5f0', '#fff5eb', '#c7eae5']
        
        for i, ml_pred in enumerate(ml_preds):
            model_name = model_names[i] if model_names is not None else f"model_{i}"
        
            x = ml_pred.x[~np.isnan(ml_pred.binned_AEs)]
            hist = ml_pred.hist[~np.isnan(ml_pred.binned_AEs)]
            binned_AEs = ml_pred.binned_AEs[~np.isnan(ml_pred.binned_AEs)]

            ax.plot(
                x, binned_AEs, c = e_colors[i], marker = markers[i], ms = 7,
                markerfacecolor = f_colors[i], alpha = 0.6, label = model_name
            )
        axtwin.bar(x, hist, color = '#4d4d4d', alpha = 0.4, width = ml_pred.bin_width, linewidth = 0)

        axtwin.set_ylabel(r"Testset Counts")
        # axtwin.tick_params(axis = 'y', colors='#2166ac')
        # axtwin.yaxis.label.set_color('#2166ac')
        # axtwin.grid(color='#d1e5f0', alpha = 0.65)
        
        xaxis_label = "y" if target_name == "" else target_name
        ax.set_xlabel(xaxis_label)
        ax.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Test}}$ ")
        ax.set_zorder(axtwin.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.grid(False)
        axtwin.grid(False)
        ax.legend()

    
    def plot_sers(self, *ml_preds, model_names = None, ax = None):
        if len(ml_preds) == 0:
            ml_preds = self.all_preds
        
        if model_names is None:
            model_names = self.labels
            
        colors = ['#4d4d4d', '#2166ac', '#b2182b', '#35978f']
        linestyles = [
            '-', (0, (5,5)), (0, (3,5,1,5)), (0, (5,1)), (0, (1,1)),
            (0, (3,1,1,1,1,1)),  (0,(5,1)), (0, (3,1,1,1)), (0, (3,5,1,5,1,5))
        ]
        if ax is None:
            fig, ax = plt.subplots(figsize = (3.0, 2.8), layout = "compressed")
        
        for i, model_name in enumerate(model_names):
            ax.plot(
                ml_preds[i].t_s, ml_preds[i].sers,
                ls = linestyles[i], color = colors[i],
                lw = 2, alpha = 0.6, label = model_name
            )
            
        ax.set_xlabel(r"Relevance $\phi(y)$")
        ax.set_ylabel("SER")
        ax.legend()


    def sumarry_plot(self, *ml_preds, model_names = None, target_name = "", file_name = None, fig_labels = None):
        if len(ml_preds) == 0:
            ml_preds = self.all_preds
        num_axes = len(ml_preds) + 2
        ncols = 3 if num_axes !=4 else 2
        nrows = int(np.ceil(len(ml_preds) / ncols))
        width = 2.65
        height = 2.35
        w_space = 0.15
        h_space = 0.1
        
        fig, axes = plt.subplots(
                nrows, ncols, layout = 'compressed',
                figsize = ((width + w_space) * ncols + w_space, (height + h_space) * nrows + h_space)
            )
        axes = axes.flatten()
        
        self.parity_plot(*ml_preds, target_name = target_name, colors = ['#bababa', '#abd9e9', '#fddbc7', '#c7eae5'],
                    xeqy_cs = ["#4d4d4d", "#2166ac", "#b2182b", "#01665e"], axes = axes[:len(ml_preds)])
        
        self.plot_sers(*ml_preds, ax = axes[len(ml_preds)])
        self.compare_mae_wb(
            *ml_preds, target_name = target_name,
            model_names = model_names, ax = axes[len(ml_preds) + 1]
        )
        if fig_labels is None:
            start = ord('a')
            for i in range(len(axes)):
                axes[i].set_title(chr(start + i), loc = 'left', fontsize = 10)
        
        if file_name:
            plt.savefig(file_name, dpi = 600)