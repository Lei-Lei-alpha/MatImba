import os
import time
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import *
from .utils.losses import *
from torchmetrics.regression import R2Score
from torch.optim.lr_scheduler import *
from .utils.utils import save_checkpoint, AverageMeter, ProgressMeter
import sys


class LossExplosionError(Exception):
    pass

class CgcnnTrainer:
    def __init__(self, model, train_loader = None, val_loader = None, test_loader = None,
                 optimiser = None, scheduler = None, scheduler_type='ReduceLROnPlateau',
                 loss_func = None, epoch_range = None, weighted_loss = False,
                 dil_inform = False, outdir = None, name = None,
                 log_file = None, device = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimiser = optimiser if optimiser is not None else \
        AdamW(self.model.parameters(), lr = 0.01, betas = (0.85, 0.99), eps = 1e-08, weight_decay = 1e-6)
        self.scheduler_type = scheduler_type
        if scheduler is None:
            if self.scheduler_type == 'CosineAnnealingLR':
                self.scheduler = CosineAnnealingLR(self.optimiser, T_max=epoch_range[-1] + 1 if epoch_range else 200, eta_min=1e-6)
            elif self.scheduler_type == 'OneCycleLR':
                steps_per_epoch = len(train_loader) if train_loader else 1
                self.scheduler = OneCycleLR(self.optimiser, max_lr=0.01, epochs=epoch_range[-1] + 1 if epoch_range else 200, steps_per_epoch=steps_per_epoch)
            else:  # Default ReduceLROnPlateau
                self.scheduler = ReduceLROnPlateau(self.optimiser, factor = 0.2, patience = 20, min_lr = 1e-5)
        else:
            self.scheduler = scheduler
        self.last_lr = self.scheduler.get_last_lr()
        if loss_func is None:
            self.loss_func = HuberLoss()
        else:
            self.loss_func = loss_func
        self.epoch_range = range(epoch_range) if isinstance(epoch_range, int) else epoch_range
        self.outdir = os.getcwd() if outdir is None else outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.name = type(self.model).__name__ if name is None else name
        self.weighted_loss = weighted_loss
        self.dil_inform = dil_inform
        self.log_file = os.path.join(self.outdir, log_file) if log_file else os.path.join(self.outdir, f'{self.name}.log')
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

        # Log basic information at the beginning
        self.log(f"Python version: {sys.version}")
        self.log(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'No CUDA'}")
        self.log(f"Optimizer parameters: {str(self.optimiser)}")
        self.log(f"Scheduler parameters: {str(self.scheduler)}")
        train_size = len(self.train_loader.dataset) if self.train_loader else 'N/A'
        val_size = len(self.val_loader.dataset) if self.val_loader else 'N/A'
        self.log(f"Train set size: {train_size}")
        self.log(f"Validation set size: {val_size}")

    def log(self, text: str) -> None:
        with open(self.log_file, 'a') as file:
            file.write(text + '\n')
        file.close()
    
    def train(self, epoch, dataloader = None):
        if dataloader is None:
            dataloader = self.train_loader
        batch_time = AverageMeter('Time', ':6.2f')
        loss_func_name = type(self.loss_func).__name__
        losses = AverageMeter(f'Loss ({loss_func_name})', ':.3f')
        
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, losses],
            prefix = f'Epoch: [{epoch}]',
            log_file = self.log_file
        )

        criterion_maes = nn.L1Loss(reduction = "none")
    
        self.model.train()
        end = time.time()
        features, labels = [], []
        
        # EMA for loss to reduce per-batch noise (decay=0.99 is a good starting point; adjust as needed)
        ema_decay = 0.995
        avg_loss = None
        
        for idx, batch in enumerate(dataloader):
            batch.to(self.device)
            self.optimiser.zero_grad()

            if hasattr(self.model, 'FDS'):
                outputs, encodings = self.model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.state, batch.batch, batch.bond_batch,
                    batch.y, epoch
                    )
                
                features.extend(encodings.detach())
                labels.append(batch.y)

            else:
                outputs = self.model(
                            batch.x, batch.edge_index, batch.edge_attr,
                            batch.state, batch.batch, batch.bond_batch
                        )

            if self.weighted_loss:
                loss = self.loss_func(outputs, batch.y, batch.omega)
            else:
                loss = self.loss_func(outputs, batch.y)

            if self.dil_inform:
                maes = criterion_maes(outputs, batch.y)
                pcc = torch.corrcoef(torch.stack(((1/batch.rou).squeeze(), maes.squeeze())))[0][-1]
                awareness_contribution = torch.exp(2.5 * (torch.abs(pcc)))
                scale = 20
                loss = (1 + awareness_contribution / scale) * loss
                           
            if torch.isnan(loss) or loss.item() > 1e10:
                raise LossExplosionError(f"Loss explosion: {loss.item()}")
    
            losses.update(loss.item(), batch.y.size(0))

            # Update EMA-averaged loss
            if avg_loss is None:
                avg_loss = loss.item()
            else:
                avg_loss = ema_decay * avg_loss + (1 - ema_decay) * loss.item()
    
            loss.backward()
            self.optimiser.step()

            current_lr = self.scheduler.get_last_lr()
            if self.last_lr != current_lr:
                self.last_lr = current_lr
                self.log(f"=> Learning rate changed to: {self.last_lr}")

            batch_time.update(time.time() - end)
            end = time.time()
            
            if idx % 10 == 0:
                progress.display(idx)
                
        if hasattr(self.model, 'FDS'):
            features = torch.stack(features)
            labels = torch.cat(labels)
            self.model.FDS.update_last_epoch_stats(epoch)
            self.model.FDS.update_running_stats(features, labels, epoch)
        
        return losses.avg
    
    def validate(self, prefix = 'Val', dataloader = None):
        if dataloader is None:
            dataloader = self.val_loader
        
        batch_time = AverageMeter('Time', ':6.3f')
        losses_mse = AverageMeter('Loss (MSE)', ':.3f')
        losses_l1 = AverageMeter('Loss (L1)', ':.3f')
        losses_esr = AverageMeter('Loss (ESR)', ':.3f')
        
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, losses_mse, losses_l1, losses_esr],
            prefix = f'{prefix}: ',
            log_file = self.log_file
        )
    
        criterion_mse = nn.MSELoss().to(self.device)
        criterion_l1 = nn.L1Loss().to(self.device)
        criterion_esr = ESRLoss().to(self.device)
        criterion_maes = nn.L1Loss(reduction = "none").to(self.device)
        criterion_r2 = R2Score().to(self.device)
    
        self.model.eval()
        all_labels = []
        all_preds = []
        all_losses = []
        all_relevances = []
        inv_densities = []
        
        with torch.no_grad():
            end = time.time()
            for idx, batch in enumerate(dataloader):
                end = time.time()
                batch.to(self.device)
                
                outputs = self.model(
                    batch.x, batch.edge_index, 
                    batch.edge_attr, batch.state, 
                    batch.batch, batch.bond_batch
                ).squeeze()
                
                loss_mse = criterion_mse(outputs, batch.y)
                loss_l1 = criterion_l1(outputs, batch.y)
                loss_esr = criterion_esr(outputs, batch.y)
                loss_all = criterion_maes(outputs, batch.y)
    
                losses_mse.update(loss_mse.item(), batch.y.size(0))
                losses_l1.update(loss_l1.item(), batch.y.size(0))
                losses_esr.update(loss_esr.item(), batch.y.size(0))

                if outputs.dim() == 0:
                    all_labels.append(batch.y.unsqueeze(0).detach())
                    all_preds.append(outputs.unsqueeze(0).detach())
                    all_losses.append(loss_all.unsqueeze(0).detach())
                    inv_densities.append((1 / batch.rou).unsqueeze(0).detach())
                    all_relevances.append(batch.phi.unsqueeze(0).detach())
                else:
                    all_labels.append(batch.y.detach())
                    all_preds.append(outputs.detach())
                    all_losses.append(loss_all.detach())
                    inv_densities.append((1 / batch.rou).detach())
                    all_relevances.append(batch.phi.detach())
                    
                batch_time.update(time.time() - end)
                end = time.time()
                if idx % 10 == 0:
                    progress.display(idx)

        all_labels = torch.cat(all_labels, dim=0).reshape(-1)
        all_preds = torch.cat(all_preds, dim=0).reshape(-1)
        all_relevances = torch.cat(all_relevances, dim=0).reshape(-1)
        inv_densities = torch.cat(inv_densities, dim=0).reshape(-1)
        all_losses = torch.cat(all_losses, dim=0).reshape(-1)

        sera = calc_sera(all_labels, all_preds, all_relevances, t = 0.4)
        scaled_error = all_losses.sum() / torch.abs(all_labels - all_labels.mean()).sum()
        
        r2_acc = criterion_r2(all_labels, all_preds)
        awareness = 1 - torch.abs(torch.corrcoef(torch.stack((inv_densities, all_losses), dim=0)))[0, 1]
            
        self.log(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tESR {losses_esr.avg:.3f}\tAWARENESS {awareness.item():.3f} * ")
    
        return losses_mse.avg, losses_l1.avg, losses_esr.avg, sera, scaled_error.item(), r2_acc.item(), awareness.item()
                
        
    def fit(self, weighted_loss = False, dil_inform = False):
        torch.cuda.empty_cache()
        self.best_l1_loss = torch.inf
        self.best_sera = torch.inf
        self.best_r2 = - torch.inf
        self.best_eta_alph_area = - torch.inf

        cmd = f"echo 'epoch,mae,sera,scaled_error,r2_score,awareness,etaxalpha' > {self.outdir}/{self.name}_val_log.csv"
        os.system(cmd)
        
        for epoch in self.epoch_range:
            train_loss = self.train(epoch)
            val_mse, val_l1, val_esr, sera, scaled_error, r2_acc, awareness = self.validate()
            
            if self.scheduler_type == 'ReduceLROnPlateau':
                self.scheduler.step(val_l1)
            else:
                self.scheduler.step()
            
            current_lr = self.scheduler.get_last_lr()[0]
            if current_lr < self.last_lr[0]:  # For ReduceLROnPlateau, log reductions
                self.last_lr = current_lr
                self.log(f"LR reduced to {current_lr}")

            eta_alpha_area = r2_acc * awareness
            
            is_best = val_l1 < self.best_l1_loss
            self.best_l1_loss = min(val_l1, self.best_l1_loss)
            is_sera_best = sera < self.best_sera
            self.best_sera = min(sera, self.best_sera)
            is_r2_best = r2_acc > self.best_r2
            
            is_dil_aware_best = eta_alpha_area > self.best_eta_alph_area
            self.best_eta_alph_area = max(eta_alpha_area, self.best_eta_alph_area)
            self.log(f"Best 'MAE' Loss: {self.best_l1_loss:.3f}")
            
            save_checkpoint(state = 
                            {
                                'epoch': epoch,
                                'best_loss': self.best_l1_loss,
                                "model": {"name": type(self.model).__name__,
                                          'init_params': {
                                              'edge_input_shape': self.model.edge_input_shape,
                                              'node_input_shape': self.model.node_input_shape,
                                              'state_input_shape': self.model.state_input_shape,
                                              'node_embedding_size': self.model.node_embedding_size,
                                              'embedding_size': self.model.embedding_size,
                                              'n_blocks': self.model.n_blocks,
                                              'fds': self.model.fds
                                          },
                                          "states": self.model.state_dict(keep_vars = True),
                                          "fds_params": self.model.fds_params
                                         },
                                "optimiser": {"name": type(self.optimiser).__name__,
                                              "states": self.optimiser.state_dict()},
                                "scheduler": {"name": type(self.scheduler).__name__, 
                                              "states": self.scheduler.state_dict()},
                            },
                            is_best = is_best,  is_dil_aware_best = is_dil_aware_best,
                            is_sera_best = is_sera_best, is_r2_best = is_r2_best,
                            outdir = self.outdir, prefix = self.name
                           )
            
            cmd = f"echo '{epoch},{val_l1},{sera},{scaled_error},{r2_acc},{awareness},{eta_alpha_area}' >> {self.outdir}/{self.name}_val_log.csv"
            os.system(cmd)
            
            self.log(f"Epoch #{epoch}: Train loss [{train_loss:.4f}];\n"
                     f"Val loss: MSE [{val_mse:.4f}], L1 [{val_l1:.4f}], ESR [{val_esr:.4f}];\n"
                     f"Val metrics: SERA [{sera:.4f}] SCALED_ERROR [{scaled_error:.4f}] R2 Score [{r2_acc:.4f}]")
        
        if self.test_loader:
            test_mse, test_l1, test_esr, sera, test_scaled_error, test_r2_acc, awareness = self.validate(prefix = "Test", dataloader = self.test_loader)
            return {
                'test_mse': test_mse,
                'test_mae': test_l1,
                'test_esr': test_esr,
                'test_sera': sera,
                'test_scaled_error': test_scaled_error,
                'test_r2': test_r2_acc,
                'test_awareness': awareness
            }
        else:
            return {}
        
    def predict(self, dataloader):
        all_labels = []
        all_preds = []
        all_densities = []
        all_relevances = []
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                all_labels.extend(batch.y.cpu().numpy())
    
        all_labels = np.array(all_labels).reshape(-1)
        all_preds = np.array(all_preds).reshape(-1)
        all_relevances = np.array(all_relevances).reshape(-1)
        all_densities = np.array(all_densities).reshape(-1)
        return all_labels, all_preds, all_relevances, all_densities

    def plot_dynamics(self, compare_configs=None):
        df = pd.read_csv(f'{self.outdir}/{self.name}_val_log.csv')
        plt.figure(figsize=(6,2.8), layout = 'compressed')
        plt.subplot(1,2,1)
        plt.plot(df['epoch'], df['mae'], label=f'{self.name} Val MAE')
        if compare_configs:
            for other in compare_configs:
                other_df = pd.DataFrame(other.dynamics_log)
                plt.plot(other_df['epoch'], other_df['val_mae'], label=other.name)
        plt.title('Loss Dynamics')
        plt.xlabel('Epoch')
        plt.ylabel('Val MAE')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(df['epoch'], df['awareness'], label=f'{self.name} Awareness')
        if compare_configs:
            for other in compare_configs:
                other_df = pd.read_csv(f'{other.outdir}/{other.name}_val_log.csv')
                plt.plot(other_df['epoch'], other_df['awareness'], label=other.name)
        plt.title('Awareness Dynamics')
        plt.xlabel('Epoch')
        plt.ylabel('DIL Awareness')
        plt.legend()
        plt.savefig(os.path.join(self.outdir, f'{self.name}_dynamics_plot.jpg'), dpi = 600)
        plt.close()

    def plot_awareness_space(self, skip = 25, model_name = None):
        df = pd.read_csv(f'{self.outdir}/{self.name}_val_log.csv')
        fig, ax = plt.subplots(figsize = (3.8, 2.8), layout = 'compressed')
        sc = plt.scatter(df["awareness"][skip:], df["r2_score"][skip:], c = df["epoch"][skip:], cmap = "RdYlBu")
        cbar = plt.colorbar(sc, label = "Epoch")
        ax.set_xlabel("DIL Awareness")
        ax.set_ylabel("R2 Score")
        plt.title(f'Awareness Space')
        plt.savefig(os.path.join(self.outdir, f'{self.name}_awareness_space.jpg'), dpi = 600)
        plt.close()