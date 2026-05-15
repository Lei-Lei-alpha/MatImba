import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def calibrate_features(features: torch.Tensor, m1: torch.Tensor, v1: torch.Tensor, 
                       m2: torch.Tensor, v2: torch.Tensor, clip_min: float = 0.1, 
                       clip_max: float = 5.0) -> torch.Tensor:
    """
    Calibrates features using first and second-order statistics.
    
    This function aligns the distribution of the input features (characterized by m1, v1)
    to a target distribution (characterized by m2, v2) via affine transformation.

    Args:
        features (torch.Tensor): Input feature batch of shape (N, feature_dim).
        m1 (torch.Tensor): Running mean of the current bin.
        v1 (torch.Tensor): Running variance of the current bin.
        m2 (torch.Tensor): Smoothed mean (target) of the current bin.
        v2 (torch.Tensor): Smoothed variance (target) of the current bin.
        clip_min (float, optional): Minimum scaling factor clamp. Defaults to 0.1.
        clip_max (float, optional): Maximum scaling factor clamp. Defaults to 5.0.

    Returns:
        torch.Tensor: Calibrated features with smoothed statistics.
    """
    eps = 1e-8
    factor = torch.clamp(v2 / (v1 + eps), clip_min, clip_max)
    return (features - m1) * torch.sqrt(factor) + m2

class FDS(nn.Module):
    """
    Feature Distribution Smoothing (FDS) module for coping with continuous imbalance.
    
    FDS estimates the statistics (mean and variance) of features for different 
    target bins and smooths them using a symmetric kernel (Gaussian, Laplace, etc.).
    It then calibrates features during training to mitigate bias toward head regions.

    Args:
        feature_dim (int): Dimensionality of the input features.
        bucket_num (int, optional): Number of bins for continuous target discretization. Defaults to 30.
        start_update (int, optional): Epoch to start updating statistics. Defaults to 0.
        start_smooth (int, optional): Epoch to start applying feature calibration. Defaults to 1.
        kernel (str, optional): Smoothing kernel type ('gaussian', 'laplace', 'triang'). Defaults to 'gaussian'.
        kernel_size (int, optional): Size of the smoothing kernel window. Defaults to 3.
        sigma (float, optional): Standard deviation for Gaussian/Laplace kernels. Defaults to 2.0.
        momentum (float, optional): Momentum factor for running statistics updates. Defaults to 0.85.
    """
    def __init__(self, feature_dim: int, bucket_num: int = 30, start_update: int = 0,
                 start_smooth: int = 1, kernel: str = 'gaussian', kernel_size: int = 3,
                 sigma: float = 2.0, momentum: float = 0.85):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.momentum = momentum
        self.half_ks = (kernel_size - 1) // 2

        self.register_buffer('kernel_window', self._build_kernel())
        self.register_buffer('epoch', torch.tensor(start_update, dtype=torch.long))
        
        # Initialize statistics buffers
        for name in ['running_mean', 'running_mean_last', 'smoothed_mean_last']:
            self.register_buffer(name, torch.zeros(bucket_num, feature_dim))
        for name in ['running_var', 'running_var_last', 'smoothed_var_last']:
            self.register_buffer(name, torch.ones(bucket_num, feature_dim))

        self._low_lim: Optional[float] = None
        self._up_lim: Optional[float] = None

    def _build_kernel(self) -> torch.Tensor:
        """Constructs the smoothing kernel window."""
        x = torch.arange(-self.half_ks, self.half_ks + 1, dtype=torch.float32)
        if self.kernel == 'gaussian':
            k = torch.exp(-0.5 * (x / self.sigma) ** 2)
        elif self.kernel == 'laplace':
            k = torch.exp(-torch.abs(x) / self.sigma)
        elif self.kernel == 'triang':
            k = torch.relu(1.0 - torch.abs(x) / (self.half_ks + 1.0))
        else:
            raise ValueError("Kernel must be 'gaussian', 'laplace', or 'triang'")
        return k / k.sum()

    def _smooth_stats(self, stat: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D convolution to smooth statistics across adjacent buckets.
        
        Args:
            stat (torch.Tensor): Input statistics tensor of shape (bucket_num, feature_dim).
            
        Returns:
            torch.Tensor: Smoothed statistics of the same shape.
        """
        # Input: [bucket, feat] -> Transpose/Unsqueeze -> [1, feat, bucket]
        input_stat = stat.transpose(0, 1).unsqueeze(0)
        kernel_weight = self.kernel_window.view(1, 1, -1).repeat(self.feature_dim, 1, 1)
        
        conv = F.conv1d(
            input_stat, kernel_weight, padding=int(self.half_ks), groups=self.feature_dim
        )
        return conv.squeeze(0).transpose(0, 1)

    def label_bucketize(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Discretizes continuous labels into bucket indices.
        
        Args:
            labels (torch.Tensor): Continuous target values.
            
        Returns:
            torch.Tensor: Long tensor of bucket indices.
        """
        device = labels.device
        if self._low_lim is None:
            value_range = labels.max() - labels.min()
            delta = value_range.clamp_min(1e-6)
            self._low_lim = labels.min() - 0.5 * delta / self.bucket_num
            self._up_lim = labels.max() + 0.5 * delta / self.bucket_num

        edges = torch.linspace(self._low_lim, self._up_lim, self.bucket_num + 1, device=device)
        bins = torch.bucketize(labels, edges, right=True)
        return torch.clamp(bins, 0, self.bucket_num - 1)

    def update(self, features, labels, epoch: int):
        """
        Updates the running mean and variance statistics based on the current batch.
        
        This should be called once per epoch with the full training set or 
        aggregated batch statistics.

        Args:
            features (torch.Tensor): Feature representations.
            labels (torch.Tensor): Corresponding target labels.
            epoch (int): Current training epoch.
        """
        if epoch < self.start_update or epoch <= self.epoch.item():
            return

        self.epoch.fill_(epoch)
        bins = self.label_bucketize(labels).long()

        # Calculate current statistics
        counts = torch.bincount(bins, minlength=self.bucket_num).to(features.dtype)
        sum_f = torch.zeros(self.bucket_num, self.feature_dim, device=features.device, dtype=features.dtype)
        sum_f.scatter_add_(0, bins.unsqueeze(1).expand(-1, self.feature_dim), features)
        curr_mean = sum_f / counts.clamp_min(1e-8).unsqueeze(1)

        centered = features - curr_mean[bins]
        sum_sq = torch.zeros_like(sum_f)
        sum_sq.scatter_add_(0, bins.unsqueeze(1).expand(-1, self.feature_dim), centered ** 2)
        curr_var = sum_sq / counts.clamp_min(1e-8).unsqueeze(1)

        # Momentum update
        alpha = self.momentum
        self.running_mean.mul_(alpha).add_(curr_mean, alpha=1 - alpha)
        self.running_var.mul_(alpha).add_(curr_var, alpha=1 - alpha)

        # Smoothing
        self.running_mean_last.copy_(self.running_mean)
        self.running_var_last.copy_(self.running_var)
        self.smoothed_mean_last.copy_(self._smooth_stats(self.running_mean_last))
        self.smoothed_var_last.copy_(self._smooth_stats(self.running_var_last))

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                epoch: Optional[int] = None) -> torch.Tensor:
        """
        Calibrates the features during the forward pass if FDS is active.

        Args:
            features (torch.Tensor): Input latent features.
            labels (torch.Tensor, optional): Target values (required for bucket lookup during training).
            epoch (int, optional): Current training epoch.

        Returns:
            torch.Tensor: Calibrated features (if active) or original features.
        """
        if epoch is None or epoch < self.start_smooth or labels is None:
            return features

        bins = self.label_bucketize(labels)
        m1, v1 = self.running_mean_last[bins], self.running_var_last[bins]
        m2, v2 = self.smoothed_mean_last[bins], self.smoothed_var_last[bins]
        return calibrate_features(features, m1, v1, m2, v2)

    def reset(self):
        """Reset all stats"""
        self.epoch.fill_(self.start_update)
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.running_mean_last.zero_()
        self.running_var_last.fill_(1.0)
        self.smoothed_mean_last.zero_()
        self.smoothed_var_last.fill_(1.0)
        self._low_lim = None
        self._up_lim = None
