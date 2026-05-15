import os
import sys
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torch_geometric.loader import DataLoader
from torchmetrics.regression import R2Score

# Local imports
from .utils.losses import (
    WeightedL1Loss, WeightedMSELoss,
    WeightedHuberLoss, WeightedFocalMSELoss,
    WeightedFocalL1Loss, ESRLoss,
    NaiiveDILALoss, SmoothDILALoss,
    StableDILALoss, calc_alpha, calc_sera,
    naiive_calc_alpha
)
from .utils.utils import AverageMeter, ProgressMeter, save_checkpoint

# Environment-aware tqdm import
python_env = os.environ.get("_", "").split("/")[-1]
if python_env == "jupyter":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Initialize logger for this module
logger = logging.getLogger(__name__)

class LossExplosionError(Exception):
    """
    Custom exception raised when the loss value becomes NaN or infinitely large.
    This usually indicates exploding gradients or numerical instability.
    """
    pass

class CgcnnTrainer:
    """
    A robust trainer for Graph Neural Networks (CGCNN/MEGNet) with support for 
    Domain Invariant Learning (DIL) and Feature Distribution Smoothing (FDS).

    This trainer handles the full lifecycle: training loops, validation, 
    metric tracking (MAE, MSE, SERA, R2), checkpointing, and visualization.

    Args:
        model (nn.Module): The PyTorch Geometric model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader, optional): DataLoader for the test set.
        optimiser (Optimizer, optional): PyTorch optimizer. Defaults to AdamW.
        scheduler (_LRScheduler, optional): Learning rate scheduler.
        scheduler_type (str, optional): Type of default scheduler if none provided 
            ("ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"). Defaults to "ReduceLROnPlateau".
        loss_func (nn.Module, optional): Custom loss module. If None, defaults to HuberLoss or DIL-aware loss.
        epoch_range (Union[int, Tuple[int, int], range], optional): Number of epochs or range. Defaults to 200.
        weighted_loss (bool, optional): Whether to apply sample weights (omega) during loss calculation.
        alpha_metric (str, optional): Metric for DIL awareness ("dcor" or "pcc"). Defaults to "dcor".
        dil_inform (bool, optional): Whether to use DIL-aware loss functions.
        dil_config (Dict[str, Any], optional): Configuration for DIL (lambda, method, warmup).
        outdir (str, optional): Directory to save logs and checkpoints. Defaults to current working directory.
        name (str, optional): Name of the experiment/model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimiser: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_type: str = "ReduceLROnPlateau",
        loss_func: Optional[nn.Module] = None,
        epoch_range: Union[int, Tuple[int, int], range, None] = None,
        weighted_loss: bool = False,
        alpha_metric: str = "dcor",
        dil_inform: bool = False,
        dil_config: Optional[Dict[str, Any]] = None,
        outdir: Optional[str] = None,
        name: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        self.model = model
        # Robust device detection: check model first, fallback to available hardware
        self.device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # --- Optimizer & Scheduler Setup ---
        self.optimiser = optimiser if optimiser is not None else AdamW(
            self.model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-08, weight_decay=1e-6
        )
        self.scheduler_type = scheduler_type

        # Parse epoch range
        if isinstance(epoch_range, int):
            self.epoch_range = range(epoch_range)
        elif isinstance(epoch_range, (tuple, list)) and len(epoch_range) == 2:
            self.epoch_range = range(epoch_range[0], epoch_range[1] + 1)
        else:
            self.epoch_range = range(200)
        
        # Initialize default scheduler if not provided
        max_epochs = self.epoch_range.stop
        if scheduler is None:
            if self.scheduler_type == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimiser, T_max=max_epochs, eta_min=1e-6
                )
            elif self.scheduler_type == "OneCycleLR":
                steps_per_epoch = len(train_loader) if train_loader else 1
                self.scheduler = OneCycleLR(
                    self.optimiser,
                    max_lr=0.01,
                    epochs=max_epochs,
                    steps_per_epoch=steps_per_epoch,
                )
            else:  # Default ReduceLROnPlateau
                self.scheduler = ReduceLROnPlateau(
                    self.optimiser, factor=0.2, patience=20, min_lr=1e-5
                )
        else:
            self.scheduler = scheduler

        self.last_lr = self.scheduler.get_last_lr() if hasattr(self.scheduler, 'get_last_lr') else [0.01]
        
        # --- Data Imbalance Level aware configuration ---
        self.weighted_loss = weighted_loss
        self.alpha_metric = alpha_metric
        self.dil_inform = dil_inform
        self.dil_config = dil_config or {}
        self.warmup_epochs = self.dil_config.get("warmup_epochs", 20)
        
        # --- Output Configuration ---
        self.outdir = outdir if outdir is not None else os.getcwd()
        os.makedirs(self.outdir, exist_ok=True)
        self.name = type(self.model).__name__ if name is None else name

        # --- Initialization Steps ---
        self._setup_loss_function(loss_func)
        self._init_validation_metrics() # Create criteria objects once
        self._init_meters()             # Create AverageMeters once
        self._log_init_info()

    def _setup_loss_function(self, loss_func: Optional[nn.Module]):
        """Configures the loss function based on DIL settings or user input."""
        if self.dil_inform:
            method = self.dil_config.get("method", "stable")
            lam = self.dil_config.get("lambda", 1.0)
            
            if method == "naive":
                logger.info("Initializing NaiveDILALoss (Batch PCC)...")
                self.loss_func = NaiveDILALoss(base_metric="huber", lambda_pcc=lam)
            elif method == "smooth":
                logger.info("Initializing SmoothDILALoss (Momentum Stats)...")
                self.loss_func = SmoothDILALoss(base_metric="huber", lambda_dcor=lam, momentum=0.1)
            else:
                logger.info("Initializing StableDILALoss (dCor + SMAPE)...")
                self.loss_func = StableDILALoss(base_metric="huber", lambda_dcor=lam)
        elif loss_func is None:
            self.loss_func = HuberLoss()
        else:
            self.loss_func = loss_func

    def _init_validation_metrics(self):
        """Initialize validation/test criteria once to avoid overhead in loops."""
        self.criterion_mse = nn.MSELoss().to(self.device)
        self.criterion_l1 = nn.L1Loss().to(self.device)
        self.criterion_esr = ESRLoss().to(self.device)
        self.criterion_r2 = R2Score(multioutput="uniform_average").to(self.device)
    
    def _init_meters(self):
        """Initialize all AverageMeters once."""
        # Train Meters
        self.meter_train_time = AverageMeter("Time", ":6.2f")
        loss_name = type(self.loss_func).__name__
        self.meter_train_loss = AverageMeter(f"Loss ({loss_name})", ":.3f")
        self.meter_train_penalty = AverageMeter("Penalty", ":.4f") if self.dil_inform else None

        # Val Meters
        self.meter_val_time = AverageMeter("Time", ":6.3f")
        self.meter_val_mse = AverageMeter("Loss (MSE)", ":.3f")
        self.meter_val_l1 = AverageMeter("Loss (L1)", ":.3f")
        self.meter_val_esr = AverageMeter("Loss (ESR)", ":.3f")
    

    def _log_init_info(self):
        """Logs the initialization summary to console/file."""
        logger.info(f"--- Trainer Initialized ---")
        logger.info(f"Python: {sys.version.split()[0]}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.name}")
        logger.info(f"DIL Informed: {self.dil_inform}")
        logger.info(f"Loss Function: {type(self.loss_func).__name__}")
        logger.info(f"Output Directory: {self.outdir}")
        logger.info(f"Optimizer: {type(self.optimiser).__name__}")
        logger.info(f"Scheduler: {self.scheduler_type}")
        logger.info(f"---------------------------------")

    def train(self, epoch: int, dataloader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        Executes one training epoch.

        Args:
            epoch (int): Current epoch number.
            dataloader (DataLoader, optional): Training dataloader. Defaults to self.train_loader.

        Returns:
            Tuple[float, float]: Average loss and Average DIL penalty for the epoch.
        
        Raises:
            LossExplosionError: If loss becomes NaN or excessively high.
        """
        if dataloader is None:
            dataloader = self.train_loader

        # 1. Reset Meters
        self.meter_train_time.reset()
        self.meter_train_loss.reset()
        if self.meter_train_penalty:
            self.meter_train_penalty.reset()

        # 2. Setup ProgressMeter (Lightweight wrapper, okay to re-init for correct batch_fmtstr)
        progress_items = [self.meter_train_time, self.meter_train_loss]
        if self.meter_train_penalty:
            progress_items.append(self.meter_train_penalty)
        
        progress = ProgressMeter(
            len(dataloader), progress_items, prefix=f"Epoch: [{epoch}]"
        )

        self.model.train()
        end = time.time()

        # DIL Warmup Scheduling: Gradually increase the regularization lambda
        if self.dil_inform and hasattr(self.loss_func, "lambda_dcor"):
            warmup_factor = float(epoch) / float(self.warmup_epochs) if epoch < self.warmup_epochs else 1.0
            self.loss_func.lambda_dcor = self.dil_config.get("lambda", 1.0) * warmup_factor

        for idx, batch in enumerate(dataloader):
            batch = batch.to(self.device, non_blocking=True)
            self.optimiser.zero_grad()

            # Forward pass: Supports standard PyG Graph signatures
            outputs = self.model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.state, batch.batch, batch.bond_batch
            )

            # Robust Dimension Handling (N,) -> (N, 1)
            if outputs.ndim == 1: outputs = outputs.unsqueeze(1)
            if batch.y.ndim == 1: batch.y = batch.y.unsqueeze(1)

            # --- Pre-process Auxiliary Data (Weights & Density) ---
            current_weights = None
            if self.weighted_loss:
                if not hasattr(batch, 'omega'):
                     raise ValueError("weighted_loss is True but batch has no 'omega' attribute.")
                current_weights = batch.omega
                if current_weights.ndim == 1:
                    current_weights = current_weights.unsqueeze(1)
                if current_weights.shape != outputs.shape:
                    current_weights = current_weights.expand_as(outputs)

            current_rou = None
            if self.dil_inform:
                if not hasattr(batch, 'rou'):
                    raise ValueError("dil_inform is True but batch has no 'rou' (density) attribute.")
                current_rou = batch.rou
                if current_rou.ndim == 1:
                    current_rou = current_rou.unsqueeze(1)
                if current_rou.shape[1] != outputs.shape[1]:
                    current_rou = current_rou.expand_as(outputs)

            # --- Loss Calculation ---
            if self.dil_inform:
                # DIL loss requires predictions, targets, and density/invariant factor (rou), Optional Weights
                loss, penalty_val = self.loss_func(outputs, batch.y, current_rou, weights=current_weights)
                
                if self.meter_train_penalty:
                    self.meter_train_penalty.update(penalty_val.item(), batch.y.size(0))
                    
            elif self.weighted_loss:
                # Weighted loss requires weights (omega)
                loss = self.loss_func(outputs, batch.y, current_weights)
            else:
                loss = self.loss_func(outputs, batch.y)

            # Safety check
            if torch.isnan(loss) or loss.item() > 1e10:
                raise LossExplosionError(f"Loss explosion: {loss.item()}")

            self.meter_train_loss.update(loss.item(), batch.y.size(0))
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()

            self.meter_train_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                progress.display(idx)

        # Hook for Feature Distribution Smoothing (FDS) updates at end of epoch
        if hasattr(self.model, "update_fds"):
            self.model.update_fds(dataloader, epoch)
            logger.info(f"Epoch {epoch}: FDS updated!")

        return self.meter_train_loss.avg, (self.meter_train_penalty.avg if self.meter_train_penalty else 0.0)

    def validate(self, prefix: str = "Val", dataloader: Any = None) -> Tuple[float, ...]:
        """
        Evaluates the model on the validation or test set.

        Args:
            prefix (str): Label for logging (e.g., "Val", "Test").
            dataloader (DataLoader, optional): Loader to evaluate. Defaults to self.val_loader.

        Returns:
            Tuple[float, ...]: Tuple containing (MSE, L1, ESR, SERA, Scaled_Error, R2, Awareness).
        """
        if dataloader is None:
            dataloader = self.val_loader
            if dataloader is None:
                # Return dummy values if no loader exists
                return (float("inf"),) * 5 + (float("-inf"), float("-inf"))

        # 1. Reset Meters
        self.meter_val_time.reset()
        self.meter_val_mse.reset()
        self.meter_val_l1.reset()
        self.meter_val_esr.reset()

        self.model.eval()
        
        # Accumulators for global metrics (R2, SERA, Awareness)
        all_labels, all_preds, all_relevances, all_densities, all_losses_raw = [], [], [], [], []

        with torch.no_grad():
            end = time.time()
            for idx, batch in enumerate(dataloader):
                # non_blocking=True allows async data transfer if pinned_memory is used
                batch = batch.to(self.device, non_blocking=True)
                
                outputs = self.model(
                    batch.x, batch.edge_index,
                    batch.edge_attr, batch.state,
                    batch.batch, batch.bond_batch
                )

                # Dimension check
                if outputs.ndim == 1: outputs = outputs.unsqueeze(1)
                if batch.y.ndim == 1: batch.y = batch.y.unsqueeze(1)

                # Batch-level metrics
                loss_l1 = self.criterion_l1(outputs, batch.y)
                self.meter_val_l1.update(loss_l1.item(), batch.y.size(0))
                loss_mse = self.criterion_mse(outputs, batch.y)                
                self.meter_val_mse.update(loss_mse.item(), batch.y.size(0))

                # --- Accumulate raw tensors on GPU ---
                all_labels.append(batch.y)
                all_preds.append(outputs)
                # Calculate abs error per batch on GPU to save memory later? 
                # No, we store it to match previous logic, but keep on GPU.
                all_losses_raw.append((outputs - batch.y).abs())

                # Metadata for DIL/Imbalanced regression
                # Note: 'rou' is usually stored as inverse density or density depending on implementation.
                # Here we calculate reciprocal, implying 'rou' is treated as a scaling factor.
                dens = batch.rou
                phi = batch.phi
                
                if dens.ndim == 1: dens = dens.unsqueeze(1)
                if phi.ndim == 1: phi = phi.unsqueeze(1)
                
                # Expand to match targets
                if dens.shape[1] != outputs.shape[1]: dens = dens.expand_as(outputs)
                if phi.shape[1] != outputs.shape[1]: phi = phi.expand_as(outputs)

                all_densities.append(dens)
                all_relevances.append(phi)

                self.meter_val_time.update(time.time() - end)
                end = time.time()

        # Concatenate full dataset
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_relevances = torch.cat(all_relevances, dim=0)
        all_densities = torch.cat(all_densities, dim=0)
        all_losses_raw = torch.cat(all_losses_raw, dim=0)

        # --- Calculate Global Metrics ---
        mse = self.meter_val_mse.avg
        l1 = self.meter_val_l1.avg
        esr = self.criterion_esr(all_preds, all_labels).item()
        sera_scalar = calc_sera(all_labels, all_preds, all_relevances, t=0.5).mean().item()
        # Calculate Awareness (Dependence between Error and Density)
        # alpha_metric decides if we use Pearson Correlation (pcc) or Distance Correlation (dcor)
        alpha_func = naive_calc_alpha if self.alpha_metric == "pcc" else calc_alpha
        alpha_tensor = alpha_func(all_labels, all_preds, all_densities)
        awareness_scalar = alpha_tensor.mean().item()

        # 3. Scaled Error (MAD normalization)
        # Calculate Mean Absolute Deviation of labels on GPU
        mad_labels = (all_labels - all_labels.mean(dim=0)).abs().sum()
        # Avoid division by zero if mad is 0 (unlikely in regression)
        scaled_error = (all_losses_raw.sum() / (mad_labels + 1e-8)).item()
        
        r2_acc = self.criterion_r2(all_preds, all_labels).item()

        logger.info(
            f" * {prefix}: MSE {mse:.3f}\tL1 {l1:.3f}\t"
            f"SERA {sera_scalar:.3f}\tAWARENESS {awareness_scalar:.3f}"
        )

        return (
            mse, l1, esr,
            sera_scalar,
            scaled_error,
            r2_acc,
            awareness_scalar,
        )

    def fit(self):
        """
        Runs the full training loop over the specified epoch range.
        
        Handles:
            - Epoch iteration
            - Validation calls
            - Checkpoint saving (tracking Best Loss, SERA, R2, and Awareness)
            - Logging to CSV
            - Error handling
        
        Returns:
            Dict[str, Any]: Test results (if test_loader exists) or empty dict.
        """
        torch.cuda.empty_cache()
        self.best_l1_loss = float("inf")
        self.best_sera = float("inf")
        self.best_r2 = float("-inf")
        self.best_robust_score = float("inf") # We want to MINIMIZE this distance
        self.best_robust_state = None
        self.min_log_sera = float("inf")

        # Track the maximum SERA seen (usually epoch 0) to normalize
        self.ema_r2 = None
        self.ema_aware = None
        self.ema_log_sera = None

        # --- Initialize Smoothing Variables for robust score ---
        # Beta controls the smoothing: 0.7 means "retain 70% history, add 30% new"
        # This makes the metric resistant to single-epoch spikes.
        self.betas = {
            'r2': 0.7,    # Low beta: Keep R2 responsive (it's naturally stable)
            'aware': 0.90, # Med-High beta: Filter noise in awareness
            'sera': 0.85   # High beta: Aggressively smooth volatile SERA
        }
        self.ema_r2 = None
        self.ema_aware = None
        self.ema_sera = None

        log_path = os.path.join(self.outdir, f"{self.name}_val_log.csv")
        # Initialize log file
        with open(log_path, "w") as f:
            f.write("epoch,mae,sera,scaled_error,r2_score,awareness,robust_score,penalty\n")

        try:
            for epoch in self.epoch_range:
                # Train
                train_loss, train_penalty = self.train(epoch)
                # Validate
                val_metrics = self.validate()
                val_mse, val_l1, val_esr, sera, scaled_error, r2_acc, awareness = val_metrics

                if self.scheduler_type == "ReduceLROnPlateau":
                    self.scheduler.step(val_l1)
                else:
                    self.scheduler.step()

                # Log Learning Rate changes
                current_lr = self.scheduler.get_last_lr()
                if getattr(self, "last_lr", []) != current_lr:
                    self.last_lr = current_lr
                    logger.info(f"=> Learning rate changed to: {self.last_lr}")

                # --- Update Smoothed Metrics (EMA) ---
                # We use Log10 for SERA to handle magnitude differences robustly
                log_sera = np.log10(sera + 1e-8)
                
                if self.ema_r2 is None:
                    # First epoch initialization
                    self.ema_r2, self.ema_aware, self.ema_log_sera = r2_acc, awareness, log_sera
                else:
                    # Update EMA
                    self.ema_r2 = self.betas['r2'] * self.ema_r2 + (1 - self.betas['r2']) * r2_acc
                    self.ema_aware = self.betas['aware'] * self.ema_aware + (1 - self.betas['aware']) * awareness
                    self.ema_log_sera = self.betas['sera'] * self.ema_log_sera + (1 - self.betas['sera']) * log_sera
                    
                # Smart Update: Only update best SERA if the model is decent (MAE < 1.1 x Best)
                if val_l1 < (self.best_l1_loss * 1.1 if self.best_l1_loss != float('inf') else float('inf')):
                    if self.ema_log_sera < self.min_log_sera:
                        self.min_log_sera = self.ema_log_sera

                # --- ROBUST SCORE CALCULATION ---
                # 1. Base Metric: MAE
                base_error = val_l1

                # 2. Penalties (Scale-Invariant)
                # Smooth Hinge Awareness Penalty
                # Target = 0.8. Quadratic decay.
                # Values < 0.8 get penalized non-linearly (stronger penalty for lower values).
                # Values >= 0.8 get 0 penalty.
                AWARE_THRESHOLD = 0.8
                penalty_aware = max(0.0, AWARE_THRESHOLD - self.ema_aware) ** 2
                
                # Log Difference = Ratio penalty. 
                # difference of 1.0 means 1 order of magnitude degradation (10x worse SERA)
                # We treat that as a heavy penalty.
                penalty_sera = max(0.0, self.ema_log_sera - self.min_log_sera)

                # Weights: # w_aware=0.5 to balance the squaring effect (0.2^2 = 0.04 * 0.5 = 0.02 = 2% penalty)
                w_aware = 0.5
                w_sera = 0.10
                
                # 3. Aggregation (Multiplicative)
                #    Robust Defect = MAE * (1 + Penalties)
                #    This prevents a model with high Error (bad MAE) from winning just via penalties.
                robust_score = base_error * (1.0 + w_aware * penalty_aware + w_sera * penalty_sera)

                # --- CHECKPOINTING ---
                # Best Overall (Standard)
                is_best = val_l1 < self.best_l1_loss
                if is_best: 
                    self.best_l1_loss = val_l1
                    logger.info(f"New Best MAE: {self.best_l1_loss:.4f}")
                    
                is_r2_best = r2_acc > self.best_r2
                if is_r2_best: self.best_r2 = r2_acc
                # Best Tail Performance
                is_sera_best = sera < self.best_sera
                if is_sera_best: self.best_sera = sera
                
                # 2% Safety Gate:
                # We strictly REJECT any "Robust" checkpoint if its MAE is > 5% worse than best seen.
                # This forces the trainer to find a model that is BOTH robust AND accurate.

                is_dil_aware_best = (
                    robust_score < self.best_robust_score and 
                    val_l1 < (self.best_l1_loss * 1.02) and 
                    r2_acc > 0
                )
                
                if is_dil_aware_best:
                    self.best_robust_score = robust_score
                    self.best_robust_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    logger.info(f"New Best Robust Model! (Score: {robust_score:.4f} | MAE: {val_l1:.3f} | Aware: {awareness:.3f})")

                # Save Checkpoint
                save_checkpoint(
                    state={
                        "epoch": epoch,
                        "best_loss": self.best_l1_loss,
                        "model": {
                            "name": type(self.model).__name__,
                            "states": self.model.state_dict(),
                            # Add generic param saving if model supports it
                            "init_params": getattr(self.model, "init_params", {}), 
                            "fds_params": getattr(self.model, "fds_params", {}),
                        },
                        "optimiser": self.optimiser.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    is_best=is_best,
                    is_dil_aware_best=is_dil_aware_best,
                    is_sera_best=is_sera_best,
                    is_r2_best=is_r2_best,
                    outdir=self.outdir,
                    prefix=self.name,
                )

                # Append metrics to CSV
                with open(log_path, "a") as f:
                    f.write(f"{epoch},{val_l1},{sera},{scaled_error},{r2_acc},{awareness},{robust_score},{train_penalty}\n")

        except LossExplosionError as e:
            logger.info(f"Training stopped: {e}")
            return {"error": str(e)}
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")

        logger.info("--- Training Finished ---")

        if self.test_loader:
            # Reload the best robust model from memory if it exists
            if self.best_robust_state is not None:
                logger.info(f"Restoring best robust model (Score: {self.best_robust_score:.4f}) for testing...")
                self.model.load_state_dict(self.best_robust_state)
                self.model.to(self.device)
            else:
                logger.info("Warning: No robust model recorded. Using last epoch model.")
            return self._run_test_evaluation()
        else:
            return {}

    def _run_test_evaluation(self) -> Dict[str, float]:
        """Runs evaluation on the held-out test set and saves predictions."""
        logger.info("--- Final Test Set Evaluation ---")
        metrics = self.validate(prefix="Test", dataloader=self.test_loader)
        test_mse, test_l1, test_esr, sera, test_scaled, test_r2, awareness = metrics

        logger.info("--- Generating Test Set Predictions ---")
        # Run prediction
        test_labels, test_preds, test_relevances, test_densities = self.predict(
            dataloader=self.test_loader
        )

        # Save to CSV
        data_dict = {}
        
        def add_cols(name, data):
            if data.ndim == 1:
                data_dict[name] = data
            else:
                for i in range(data.shape[1]):
                    data_dict[f"{name}_{i}"] = data[:, i]

        add_cols("labels", test_labels)
        add_cols("predictions", test_preds)
        add_cols("relevance", test_relevances)
        add_cols("density", test_densities)

        pred_path = os.path.join(self.outdir, f"{self.name}_test_predictions.csv")
        pd.DataFrame(data_dict).to_csv(pred_path, index=False)
        logger.info(f"Test predictions saved to {pred_path}")

        return {
            "test_mse": test_mse,
            "test_mae": test_l1,
            "test_esr": test_esr,
            "test_sera": sera,
            "test_scaled_error": test_scaled,
            "test_r2": test_r2,
            "test_awareness": awareness,
        }

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, ...]:
        """
        Generates predictions for a given dataloader.

        Args:
            dataloader (DataLoader): Input data.

        Returns:
            Tuple[np.ndarray, ...]: (Labels, Predictions, Relevances, Densities)
        """
        self.model.eval()
        results = {
            "labels": [],
            "preds": [],
            "relevances": [],
            "densities": []
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", leave=False):
                batch = batch.to(self.device, non_blocking=True)
                outputs = self.model(
                    batch.x, batch.edge_index,
                    batch.edge_attr, batch.state,
                    batch.batch, batch.bond_batch
                )

                if outputs.ndim == 1: outputs = outputs.unsqueeze(1)
                if batch.y.ndim == 1: batch.y = batch.y.unsqueeze(1)

                # Handle metadata
                dens = batch.rou if hasattr(batch, 'rou') else torch.ones_like(batch.y)
                phi = batch.phi if hasattr(batch, 'phi') else torch.ones_like(batch.y)
                if dens.ndim == 1: dens = dens.unsqueeze(1)
                if phi.ndim == 1: phi = phi.unsqueeze(1)
                
                if dens.shape[1] != outputs.shape[1]: dens = dens.expand_as(outputs)
                if phi.shape[1] != outputs.shape[1]: phi = phi.expand_as(outputs)

                results["labels"].append(batch.y.cpu())
                results["preds"].append(outputs.cpu())
                results["relevances"].append(phi.cpu())
                results["densities"].append(dens.cpu())

        # Convert to Numpy
        def concat_numpy(key):
            return torch.cat(results[key], dim=0).numpy()

        final_labels = concat_numpy("labels")
        final_preds = concat_numpy("preds")
        final_rels = concat_numpy("relevances")
        final_dens = concat_numpy("densities")

        # Optional Squeeze for backward compatibility
        if final_labels.shape[1] == 1:
            return (
                final_labels.squeeze(1),
                final_preds.squeeze(1),
                final_rels.squeeze(1),
                final_dens.squeeze(1),
            )
            
        return final_labels, final_preds, final_rels, final_dens

    def plot_dynamics(self, compare_configs: Optional[List[Any]] = None) -> None:
        """
        Plots training dynamics (Loss and Awareness) comparing this model 
        to other configurations if provided.

        Args:
            compare_configs (List[Any], optional): List of other Trainer instances to compare.
        """
        log_path = os.path.join(self.outdir, f"{self.name}_val_log.csv")
        if not os.path.exists(log_path):
            logger.info("No log file found for plotting dynamics.")
            return

        df = pd.read_csv(log_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.8), constrained_layout=True)
        
        # Plot Loss
        ax1.plot(df["epoch"], df["mae"], label=f"{self.name} Val MAE")
        if compare_configs:
            for other in compare_configs:
                other_path = os.path.join(other.outdir, f"{other.name}_val_log.csv")
                if os.path.exists(other_path):
                    other_df = pd.read_csv(other_path)
                    ax1.plot(other_df["epoch"], other_df["mae"], label=other.name)
        
        ax1.set_title("Loss Dynamics")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Val MAE")
        ax1.legend()

        # Plot Awareness
        ax2.plot(df["epoch"], df["awareness"], label=f"{self.name} Awareness")
        if compare_configs:
            for other in compare_configs:
                other_path = os.path.join(other.outdir, f"{other.name}_val_log.csv")
                if os.path.exists(other_path):
                    other_df = pd.read_csv(other_path)
                    ax2.plot(other_df["epoch"], other_df["awareness"], label=other.name)
                    
        ax2.set_title("Awareness Dynamics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("DIL Awareness")
        ax2.legend()

        plot_path = os.path.join(self.outdir, f"{self.name}_dynamics_plot.jpg")
        plt.savefig(plot_path, dpi=600)
        plt.close()

    def plot_awareness_space(self, skip: int = 25) -> None:
        """
        Plots the Accuracy vs Awareness trade-off space.
        
        Args:
            skip (int): Number of initial epochs to skip (burn-in period).
        """
        log_path = os.path.join(self.outdir, f"{self.name}_val_log.csv")
        if not os.path.exists(log_path):
            return

        df = pd.read_csv(log_path)
        if len(df) <= skip:
            return

        fig, ax = plt.subplots(figsize=(3.8, 2.8), constrained_layout=True)
        
        sc = ax.scatter(
            df["awareness"][skip:], 
            df["r2_score"][skip:], 
            c=df["epoch"][skip:], 
            cmap="RdYlBu"
        )
        cbar = plt.colorbar(sc, label="Epoch")
        
        ax.set_xlabel("DIL Awareness")
        ax.set_ylabel("R2 Score")
        ax.set_title("Awareness Space")
        
        plot_path = os.path.join(self.outdir, f"{self.name}_awareness_space.jpg")
        plt.savefig(plot_path, dpi=600)
        plt.close()