# loss.py: Define the loss functions
import torch
import numpy as np
from typing import Union, List, Tuple, Optional
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import simpson
from torch.nn.modules.loss import _Loss

TensorLike = Union[torch.Tensor, np.ndarray, List[float], Tuple[float, ...]]

def _sanitize_inputs(input: TensorLike, target: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to convert inputs to float32 tensors and ensure shapes match."""
    input_t = torch.as_tensor(input, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    
    if input_t.ndim == 1:
        input_t = input_t.view(-1, 1)
    if target_t.ndim == 1:
        target_t = target_t.view(-1, 1)
        
    if input_t.shape != target_t.shape:
        raise ValueError(f"Shape mismatch: input {input_t.shape} vs target {target_t.shape}")
    
    return input_t, target_t

def _apply_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Helper to apply reduction to a loss tensor."""
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def _get_base_loss(input: torch.Tensor, target: torch.Tensor, metric: str, weights: Optional[TensorLike] = None) -> torch.Tensor:
    """
    Global helper to calculate element-wise base loss and apply weights.
    Handles weight conversion (sanitization) and broadcasting.
    """
    # 1. Calculate raw element-wise loss
    metric_lower = metric.lower()
    if metric_lower in ['l1', 'mae']:
        loss = F.l1_loss(input, target, reduction='none')
    elif metric_lower == 'mse':
        loss = F.mse_loss(input, target, reduction='none')
    elif metric_lower == 'huber':
        loss = F.huber_loss(input, target, reduction='none')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # 2. Apply weights if provided
    if weights is not None:
        # Convert to tensor, ensure float, move to correct device
        w_t = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
        
        # Handle Dimension Mismatch for Broadcasting
        # Case: Loss is (N, 1) but Weights are (N,) -> Unsqueeze Weights to (N, 1)
        if w_t.ndim == 1 and loss.ndim > 1:
            w_t = w_t.view(-1, 1)
            
        # Case: Loss is (N, C) and Weights are (N, 1) -> Broadcast automatically works
        # Case: Loss is (N, C) and Weights are (N, C) -> Element-wise works
        
        loss = loss * w_t
        
    return loss


# ==============================================================================
# 1. Standard Weighted Losses
# ==============================================================================

class WeightedL1Loss(_Loss):
    """Mean Absolute Error (MAE) with optional element-wise weighting."""
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)

    def forward(self, input: TensorLike, target: TensorLike, weights: Optional[TensorLike] = None) -> torch.Tensor:
        input_t, target_t = _sanitize_inputs(input, target)
        loss = _get_base_loss(input_t, target_t, 'l1', weights)
        return _apply_reduction(loss, self.reduction)


class WeightedMSELoss(_Loss):
    """Mean Squared Error (MSE) with optional element-wise weighting."""
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)

    def forward(self, input: TensorLike, target: TensorLike, weights: Optional[TensorLike] = None) -> torch.Tensor:
        input_t, target_t = _sanitize_inputs(input, target)
        loss = _get_base_loss(input_t, target_t, 'mse', weights)
        return _apply_reduction(loss, self.reduction)

class WeightedHuberLoss(_Loss):
    """Huber Loss (Robust Regression) with optional element-wise weighting."""
    def __init__(self, reduction: str = 'mean', delta: float = 1.0) -> None:
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input: TensorLike, target: TensorLike, weights: Optional[TensorLike] = None) -> torch.Tensor:
        input_t, target_t = _sanitize_inputs(input, target)
        
        # F.huber_loss does not accept weights directly in all versions, 
        # so we compute with reduction='none' first.
        loss = F.huber_loss(input_t, target_t, reduction='none', delta=self.delta)
        
        if weights is not None:
            w_t = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
            loss = loss * w_t
            
        return _apply_reduction(loss, self.reduction)

# ==============================================================================
# 2. Focal Losses (Outlier/Hard Sample Mining)
# ==============================================================================

class WeightedFocalL1Loss(_Loss):
    """
    L1 Loss weighted by a function of the error magnitude.
    """
    def __init__(self, reduction: str = 'mean', activate: str = 'sigmoid',
                 beta: float = 0.2, gamma: float = 1.0) -> None:
        super().__init__(reduction=reduction)
        if activate not in {"sigmoid", "tanh"}:
            raise ValueError("activate must be 'sigmoid' or 'tanh'")
        self.activate = activate
        self.beta = beta
        self.gamma = gamma

    def forward(self, input: TensorLike, target: TensorLike, weights: Optional[TensorLike] = None) -> torch.Tensor:
        input_t, target_t = _sanitize_inputs(input, target)
        
        # Base L1
        l1_loss = torch.abs(input_t - target_t)
        
        if self.activate == 'tanh':
            modulation = (torch.tanh(self.beta * l1_loss)) ** self.gamma
        else:
            modulation = (2 * torch.sigmoid(self.beta * l1_loss) - 1) ** self.gamma
            
        loss = l1_loss * modulation
        
        if weights is not None:
            w_t = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
            loss = loss * w_t
            
        return _apply_reduction(loss, self.reduction)

class WeightedFocalMSELoss(_Loss):
    """
    MSE Loss weighted by a function of the error magnitude to focus on hard examples.
    """
    def __init__(self, reduction: str = 'mean', activate: str = 'sigmoid',
                 beta: float = 0.2, gamma: float = 1.0) -> None:
        super().__init__(reduction=reduction)
        if activate not in {"sigmoid", "tanh"}:
            raise ValueError("activate must be 'sigmoid' or 'tanh'")
        self.activate = activate
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, input: TensorLike, target: TensorLike, weights: Optional[TensorLike] = None) -> torch.Tensor:
        input_t, target_t = _sanitize_inputs(input, target)
        
        # Base MSE
        mse_loss = (input_t - target_t) ** 2
        
        # Modulation factor based on absolute error
        abs_error = torch.abs(input_t - target_t)
        
        if self.activate == 'tanh':
            modulation = (torch.tanh(self.beta * abs_error)) ** self.gamma
        else:
            modulation = (2 * torch.sigmoid(self.beta * abs_error) - 1) ** self.gamma
            
        loss = mse_loss * modulation
        
        if weights is not None:
            w_t = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
            loss = loss * w_t
            
        return _apply_reduction(loss, self.reduction)

class WeightedFocalL1Loss(_Loss):
    def __init__(self, reduction: str = 'mean', activate: str = 'sigmoid',
                 beta: float = 0.2, gamma: float = 1) -> None:
        super().__init__(reduction = reduction)
        assert activate in {"sigmoid", "tanh"}
        self.activate = activate
        self.beta = beta
        self.gamma = gamma

    def forward(self, input: tensor, target: tensor, weights: tensor = None) -> tensor:
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= (torch.tanh(self.beta * torch.abs(inputs - targets))) ** self.gamma if self.activate == 'tanh' else \
        (2 * torch.sigmoid(self.beta * torch.abs(inputs - targets)) - 1) ** self.gamma
        if weights is not None:
            loss *= weights.expand_as(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss

# ==============================================================================
# 3. Statistical Losses (ISR / ESR)
# ==============================================================================
            
def ISR(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Calculates Internally Studentized Residuals (ISR) for multi-channel data.
    Performs C independent simple linear regressions in parallel.

    Args:
        x: Predictor with shape (N, C) or (N,).
        y: Target with shape (N, C) or (N,).

    Returns:
        torch.Tensor: Studentized residuals with shape (N, C).
    """
    # 1. Input Sanitization & Shape Standardization
    # Convert to tensor and force float32
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    # Ensure inputs are at least 2D: (N, 1) if they were (N,)
    if x_t.ndim == 1:
        x_t = x_t.view(-1, 1)
    if y_t.ndim == 1:
        y_t = y_t.view(-1, 1)

    # 2. Validation
    N, C_x = x_t.shape
    _, C_y = y_t.shape

    if x_t.shape != y_t.shape:
        raise ValueError(f"Shape mismatch: input {x_t.shape} vs target {y_t.shape}. "
                         "Channels and observations must align.")
    if N <= 2:
        raise ValueError(f"Insufficient observations (N={N}). Need N > 2.")

    # Constants for numerical stability
    EPS = 1e-8

    # 3. Regression Statistics (Computed column-wise along dim=0)
    # Means -> Shape: (1, C) for correct broadcasting
    mean_x = torch.mean(x_t, dim=0, keepdim=True)
    mean_y = torch.mean(y_t, dim=0, keepdim=True)

    # Centered variables -> Shape: (N, C)
    diff_x = x_t - mean_x
    diff_y = y_t - mean_y

    # Sum of Squares -> Shape: (1, C)
    # We keepdim=True to ensure easy broadcasting later
    S_xx = torch.sum(diff_x**2, dim=0, keepdim=True)
    S_xy = torch.sum(diff_x * diff_y, dim=0, keepdim=True)
    
    # Coefficients -> Shape: (1, C)
    beta1 = S_xy / (S_xx + EPS)
    beta0 = mean_y - (beta1 * mean_x)

    # 4. Predictions and Residuals
    # Broadcasting: (1, C) + (1, C) * (N, C) -> (N, C)
    y_hat = beta0 + (beta1 * x_t)
    residuals = y_t - y_hat

    # 5. Leverage (Hat Matrix Diagonal) -> Shape: (N, C)
    # Formula: h_ii = 1/n + (x_i - mean_x)^2 / S_xx
    # Note: This computes the leverage of the i-th point within the c-th channel regression.
    h_ii = (1.0 / N) + (diff_x**2 / (S_xx + EPS))

    # 6. Standard Error of Regression (Sigma Hat) -> Shape: (1, C)
    sse = torch.sum(residuals**2, dim=0, keepdim=True)
    std_error = torch.sqrt(sse / (N - 2))

    # 7. Studentization -> Shape: (N, C)
    # SE_regression varies by observation i due to leverage h_ii
    se_regression_i = std_error * torch.sqrt(1 - h_ii)
    
    # Avoid division by zero
    studentized_residuals = residuals / (se_regression_i + EPS)

    return studentized_residuals


class ESRLoss(nn.Module):
    """
    Calculates the Externally Studentized Residual (ESR) Loss, also known as 
    Studentized Deleted Residuals. This loss is more sensitive to outliers 
    than standard MSE or ISR.
    
    Formula: 
        t_i = r_i * sqrt( (n - p - 1) / (n - p - r_i^2) )
        where r_i is the ISR, n is sample size, p is number of parameters (2).
    """
    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-6) -> None:
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'. Default: 'mean'.
            epsilon (float): Small value to prevent division by zero or sqrt of negative numbers.
        """
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(
        self, 
        input: TensorLike, 
        target: TensorLike, 
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input: Predictor variables (N, C) or (N,).
            target: Target variables (N, C) or (N,).
            weights: Optional rescaling weight for each element. Shape must broadcast.

        Returns:
            torch.Tensor: The computed loss.
        """
        # 1. Compute Internally Studentized Residuals (ISR)
        # We assume compute_isr_multichannel is defined as in the previous step
        isr = ISR(input, target)
        
        # 2. Get dimensions
        # n = number of observations (rows)
        n = isr.size(0)

        # We need n > 3 because degrees of freedom = n - p - 1.
        # With p=2 (slope+intercept), we need n - 3 > 0.
        if n <= 3:
            print(n)
            raise ValueError(f"ESRLoss requires at least 4 samples (n={n}) to be numerically stable.")

        # 3. Convert ISR to ESR (Studentized Deleted Residuals)
        # Formula: t = isr * sqrt((n - 3) / (n - 2 - isr^2))
        numerator = n - 3

        # Stability Check:
        # If isr^2 is very close to (n-2), the denominator -> 0 (Exploding gradients).
        # We clamp the denominator to ensure it stays positive.
        denominator = (n - 2) - torch.square(isr)
        denominator = torch.clamp(denominator, min=self.epsilon)

        scaling_factor = torch.sqrt(numerator / denominator)
        studentized_residuals = isr * scaling_factor

        # 4. Calculate Absolute Loss
        loss = torch.abs(studentized_residuals)

        # 5. Apply Weights (if provided)
        if weights is not None:
            # Ensure weights are a tensor and on the correct device
            w_t = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
            loss = loss * w_t

        # 6. Reduction
        return _apply_reduction(loss, self.reduction)


# ==============================================================================
# 4. Correlation / Density Aware Losses (DILA)
# ==============================================================================

class NaiiveDILALoss(nn.Module):
    def __init__(self, base_metric='huber', lambda_pcc=0.25, beta=0.99, epsilon=1e-8):
        """
        Args:
            base_metric: 'l1', 'mse', or 'huber'
            lambda_pcc: Weight of the correlation penalty. 
                        Higher = stricter independence between error and density.
            beta: Momentum for the running average of statistics (0.9 to 0.99 recommended).
                  Helps smooth the noisy trajectory.
            epsilon: Stability term for division.
        """
        super().__init__()
        self.base_metric = base_metric
        self.lambda_pcc = lambda_pcc
        self.beta = beta
        self.epsilon = epsilon
        # Register buffers to store running statistics (these are not trainable parameters)
        # We track Covariance and Variances to approximate global PCC
        self.register_buffer('running_cov', torch.tensor(0.0))
        self.register_buffer('running_var_inv_dens', torch.tensor(1.0))
        self.register_buffer('running_var_mae', torch.tensor(1.0))

    def forward(
        self, input: torch.Tensor, target: torch.Tensor,
        density: torch.Tensor, weights: Optional[TensorLike] = None
    ) -> tuple:
        """
        input: (N, C) or (N,)
        target: (N, C) or (N,)
        density: (N, C) or (N,) - Local label density
        """
        # 1. Calculate Base Task Loss (Per sample)
        # We keep reduction='none' to allow weighting if needed later, 
        # but for PCC we need the raw values.
        input_t, target_t = _sanitize_inputs(input, target)
        density_t = torch.as_tensor(density, dtype=torch.float32, device=input_t.device)
        
        # 1. Base Task Loss (uses shared helper)
        task_losses = _get_base_loss(input_t, target_t, self.base_metric, weights)
        # Ensure shapes match
        if input_t.ndim == 1:
            input_t = input_t.unsqueeze(1)
            target_t = target_t.unsqueeze(1)
            density_t = density_t.unsqueeze(1)

        if density_t.shape != input_t.shape:
            if density_t.ndim == 1: density_t = density_t.view(-1, 1)
            if density_t.shape[0] == input_t.shape[0]:
                 density_t = density_t.expand_as(input_t)

        # 2. 2. Prepare Variables for Correlation
        # We want to minimize correlation between MAE and (1/Density)
        # i.e., We don't want Rare items (High 1/Density) to have High MAE.
        current_maes = torch.abs(input_t - target_t) 
        inv_densities = 1.0 / torch.clamp(density, min=self.epsilon)

        # 3. Differentiable PCC Calculation (Batch Level)
        # Center variables
        mae_centered = current_maes - current_maes.mean(dim=0, keepdim=True)
        dens_centered = inv_densities - inv_densities.mean(dim=0, keepdim=True)
        # Calculate batch statistics
        batch_cov = (mae_centered * dens_centered).mean(dim=0)
        batch_var_mae = (mae_centered ** 2).mean(dim=0)
        batch_var_dens = (dens_centered ** 2).mean(dim=0)
        
        # 4. Update Running Statistics (EMA)
        # This stabilizes the gradient. We detach because we don't want to 
        # backpropagate through the history of the training.
        if self.training:
            with torch.no_grad():
                self.running_cov = self.beta * self.running_cov + (1 - self.beta) * batch_cov.mean()

        # 5. Compute Correlation Penalty
        # We use the current batch's centered data but normalize using the 
        # smoothed variances to prevent division by zero or massive jumps 
        # in gradients when a batch is homogeneous.
        
        # Note: We optimize the *batch* correlation directly here to get 
        # immediate gradients, but we could use the smoothed stats for the denominator.
        # Here, we stick to batch statistics for the gradient, but you can mix in running stats.
        
        denom = torch.sqrt(batch_var_mae * batch_var_dens) + self.epsilon
        pcc = batch_cov / denom
        
        # We want PCC to be 0. So we minimize |PCC| or PCC^2.
        # PCC^2 is often smoother for optimization (convex parabola).
        pcc_loss_val = torch.mean(pcc ** 2)

        # 6. Combine Losses
        # Total = Mean(TaskLoss) + lambda * PCC_Penalty
        final_loss = task_losses.mean() * (1 + self.lambda_pcc * pcc_loss_val)
        
        # --- CRITICAL: RETURN TUPLE ---
        return final_loss, pcc_loss_val

class StableNaiiveDILALoss(nn.Module):
    def __init__(self, base_metric='huber', lambda_pcc=0.25, beta=0.95, epsilon=1e-6):
        """
        Improvements:
        - Reduced beta to 0.95 (was 0.99) for faster adaptation of stats.
        - Uses Global EMA for normalization to prevent "Batch-Overfitting".
        """
        super().__init__()
        self.base_metric = base_metric
        self.lambda_pcc = lambda_pcc
        self.beta = beta
        self.epsilon = epsilon
        
        # Buffers for EMA (Global Statistics)
        self.register_buffer('running_var_mae', torch.tensor(1.0))
        self.register_buffer('running_var_dens', torch.tensor(1.0))

    def forward(self, input: torch.Tensor, target: torch.Tensor, density: torch.Tensor, weights: Optional[TensorLike] = None) -> tuple:
        # 1. Base Task Loss
        input_t, target_t = _sanitize_inputs(input, target)
        density_t = torch.as_tensor(density, dtype=torch.float32, device=input_t.device)
        task_losses = _get_base_loss(input_t, target_t, self.base_metric, weights)
        
        if input_t.ndim == 1:
            input_t = input_t.unsqueeze(1)
            target_t = target_t.unsqueeze(1)
            density_t = density_t.unsqueeze(1)
        if density_t.ndim == 1: density_t = density_t.view(-1, 1)
        if density_t.shape != input_t.shape: density_t = density_t.expand_as(input_t)

        # 2. Prepare Variables
        current_maes = torch.abs(input_t - target_t) 
        inv_densities = 1.0 / torch.clamp(density_t, min=self.epsilon)
        
        # Center variables (Batch Mean is fine for centering)
        mae_centered = current_maes - current_maes.mean(dim=0, keepdim=True)
        dens_centered = inv_densities - inv_densities.mean(dim=0, keepdim=True)
        
        # 3. Calculate Batch Variances
        batch_var_mae = (mae_centered ** 2).mean(dim=0)
        batch_var_dens = (dens_centered ** 2).mean(dim=0)
        
        # 4. Update Global Variances (EMA)
        # We track the "Global Scale" of the problem here
        if self.training:
            with torch.no_grad():
                self.running_var_mae = self.beta * self.running_var_mae + (1 - self.beta) * batch_var_mae.mean()
                self.running_var_dens = self.beta * self.running_var_dens + (1 - self.beta) * batch_var_dens.mean()

        # 5. Calculate Robust Correlation Penalty
        # Numerator: Current Batch Covariance (Gradient flows here)
        batch_cov = (mae_centered * dens_centered).mean(dim=0)
        
        # Denominator: HYBRID Normalization
        # We mix Batch Variance with Global Variance to stabilize the gradient.
        # If Batch is too small/noisy, Global takes over.
        global_std_mae = torch.sqrt(self.running_var_mae)
        global_std_dens = torch.sqrt(self.running_var_dens)
        
        # Use global stats for normalization to align Batch PCC with Global PCC
        denom = (global_std_mae * global_std_dens) + self.epsilon
        
        # We are effectively minimizing Covariance scaled by Global Variance
        pcc_proxy = batch_cov / denom
        
        # Metric: Minimize PCC^2
        pcc_loss_val = torch.mean(pcc_proxy ** 2) 

        # 6. Combine
        final_loss = task_losses.mean() * (1 + self.lambda_pcc * pcc_loss_val)
        
        return final_loss, pcc_loss_val

class StableDILALoss(nn.Module):
    """
    [ROBUST SCALED LOSS] 
    Optimizes dCov on Z-Scored inputs. 
    This ensures the penalty magnitude is consistent regardless of density units.
    """
    def __init__(self, base_metric='huber', lambda_dcor=1.0, epsilon=1e-8):
        super().__init__()
        self.base_metric = base_metric
        self.lambda_dcor = lambda_dcor
        self.epsilon = epsilon
        self.warmup_lambda = None # Placeholder for dynamic lambda update

    def forward(self, input: torch.Tensor, target: torch.Tensor, density: torch.Tensor, weights: Optional[torch.Tensor] = None) -> tuple:
        input_t, target_t = _sanitize_inputs(input, target)
        density_t = torch.as_tensor(density, dtype=torch.float32, device=input_t.device)
        task_losses = _get_base_loss(input_t, target_t, self.base_metric, weights)
        
        if input_t.ndim == 1:
            input_t = input_t.unsqueeze(1)
            target_t = target_t.unsqueeze(1)
            density_t = density_t.unsqueeze(1)
            task_losses = task_losses.unsqueeze(1)
        if density_t.ndim == 1: density_t = density_t.view(-1, 1)
        if density_t.shape != input_t.shape: density_t = density_t.expand_as(input_t)
            
        # 1. Metric: Log-L1 Error (Monotonic, safe for 0)
        error_metric = torch.log1p(torch.abs(input_t - target_t))
        inv_densities = 1.0 / torch.clamp(density_t, min=self.epsilon)

        # 2. Z-SCORE NORMALIZATION (The Fix)
        # We normalize the batch to Zero Mean, Unit Variance.
        # This prevents '1/density' scale from blowing up the distance matrix.
        # We add 1e-5 to std to allow gradients to flow even if std is small.
        error_norm = (error_metric - error_metric.mean(dim=0, keepdim=True)) / (error_metric.std(dim=0, keepdim=True) + 1e-5)
        dens_norm = (inv_densities - inv_densities.mean(dim=0, keepdim=True)) / (inv_densities.std(dim=0, keepdim=True) + 1e-5)

        # 3. Distance Matrices on NORMALIZED Data
        X = dens_norm.T.unsqueeze(-1)
        Y = error_norm.T.unsqueeze(-1)
        
        a = torch.cdist(X, X, p=2)
        b = torch.cdist(Y, Y, p=2)
        
        # 4. Double Centering
        a_mean_row = a.mean(dim=2, keepdim=True)
        a_mean_col = a.mean(dim=1, keepdim=True)
        a_mean_grand = a.mean(dim=(1, 2), keepdim=True)
        A_centered = a - a_mean_row - a_mean_col + a_mean_grand
        
        b_mean_row = b.mean(dim=2, keepdim=True)
        b_mean_col = b.mean(dim=1, keepdim=True)
        b_mean_grand = b.mean(dim=(1, 2), keepdim=True)
        B_centered = b - b_mean_row - b_mean_col + b_mean_grand
        
        # 5. Compute dCov (on normalized data, this is roughly dCor)
        dcov2 = (A_centered * B_centered).mean(dim=(1, 2))
        dcov2 = torch.clamp(dcov2, min=1e-12)
        
        # We optimize the Distance Covariance directly.
        # Since inputs are normalized, this value is bounded and stable.
        dcov_loss = torch.sqrt(dcov2).mean()
        
        # 6. Final Loss
        mean_task_loss = task_losses.mean()
        
        # Dynamic lambda (handled by trainer, or default self.lambda_dcor)
        current_lambda = self.warmup_lambda if self.warmup_lambda is not None else self.lambda_dcor
        
        # Multiplicative Penalty
        # Since dcov_loss is now scale-invariant (~0 to 1), lambda=0.25 means 25% penalty.
        # We detach mean_task_loss to decouple magnitude gradients.
        penalty_term = mean_task_loss.detach() * current_lambda * dcov_loss
        
        final_loss = mean_task_loss + penalty_term
        
        return final_loss, dcov_loss
                

class SmoothDILALoss(nn.Module):
    """
    [SMOOTH STABLE LOSS] 
    Uses Momentum (EMA) for normalization statistics to fix oscillation.
    
    Why this helps:
    - Standard StableDILALoss normalizes using *Batch Stats*, which jitter.
    - SmoothDILALoss normalizes using *Running Stats*, which are stable.
    """
    def __init__(self, base_metric='huber', lambda_dcor=1.0, momentum=0.1, epsilon=1e-8):
        super().__init__()
        self.base_metric = base_metric
        self.lambda_dcor = lambda_dcor
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Buffers to track global statistics (Initialized lazily on first forward)
        self.register_buffer('running_mean_e', torch.tensor(0.0))
        self.register_buffer('running_var_e', torch.tensor(1.0))
        self.register_buffer('running_mean_d', torch.tensor(0.0))
        self.register_buffer('running_var_d', torch.tensor(1.0))
        self.is_initialized = False

    def _get_base_loss(self, input: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.base_metric.lower() in ['l1', 'mae']:
            loss = F.l1_loss(input, target, reduction='none')
        elif self.base_metric.lower() == 'mse':
            loss = F.mse_loss(input, target, reduction='none')
        elif self.base_metric.lower() == 'huber':
            loss = F.huber_loss(input, target, reduction='none')
        else:
            raise ValueError(f"Unknown metric: {self.base_metric}")

        if weights is not None:
            w_t = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
            if w_t.ndim == 1 and loss.ndim > 1:
                w_t = w_t.view(-1, 1)
            loss = loss * w_t
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, density: torch.Tensor, weights: Optional[TensorLike] = None) -> tuple:
        input_t, target_t = _sanitize_inputs(input, target)
        task_losses = self._get_base_loss(input_t, target_t, weights)
        
        if input.ndim == 1:
            input = input.unsqueeze(1)
            target = target.unsqueeze(1)
            density = density.unsqueeze(1)
            task_losses = task_losses.unsqueeze(1)
            
        # 1. Metric: Log-L1 Error
        error_metric = torch.log1p(torch.abs(input - target))
        inv_densities = 1.0 / torch.clamp(density, min=self.epsilon)

        # 2. Update Running Statistics (Momentum)
        if self.training:
            with torch.no_grad():
                # Calculate current batch stats (averaged across channels for scalar stability)
                batch_mean_e = error_metric.mean()
                batch_var_e = error_metric.var(unbiased=False)
                batch_mean_d = inv_densities.mean()
                batch_var_d = inv_densities.var(unbiased=False)
                
                if not self.is_initialized:
                    self.running_mean_e = batch_mean_e
                    self.running_var_e = batch_var_e
                    self.running_mean_d = batch_mean_d
                    self.running_var_d = batch_var_d
                    self.is_initialized = True
                else:
                    m = self.momentum
                    self.running_mean_e = (1 - m) * self.running_mean_e + m * batch_mean_e
                    self.running_var_e = (1 - m) * self.running_var_e + m * batch_var_e
                    self.running_mean_d = (1 - m) * self.running_mean_d + m * batch_mean_d
                    self.running_var_d = (1 - m) * self.running_var_d + m * batch_var_d

        # 3. Normalize using RUNNING stats (Stable Reference)
        # Treated as constants (detach) so we don't backprop through the history
        std_e = torch.sqrt(self.running_var_e + 1e-5)
        std_d = torch.sqrt(self.running_var_d + 1e-5)
        
        error_norm = (error_metric - self.running_mean_e) / std_e
        dens_norm = (inv_densities - self.running_mean_d) / std_d

        # 4. Distance Matrices on STABLE Data
        X = dens_norm.T.unsqueeze(-1)
        Y = error_norm.T.unsqueeze(-1)
        
        a = torch.cdist(X, X, p=2)
        b = torch.cdist(Y, Y, p=2)
        
        # 5. Double Centering
        a_mean_row = a.mean(dim=2, keepdim=True)
        a_mean_col = a.mean(dim=1, keepdim=True)
        a_mean_grand = a.mean(dim=(1, 2), keepdim=True)
        A_centered = a - a_mean_row - a_mean_col + a_mean_grand
        
        b_mean_row = b.mean(dim=2, keepdim=True)
        b_mean_col = b.mean(dim=1, keepdim=True)
        b_mean_grand = b.mean(dim=(1, 2), keepdim=True)
        B_centered = b - b_mean_row - b_mean_col + b_mean_grand
        
        # 6. Compute dCov
        dcov2 = (A_centered * B_centered).mean(dim=(1, 2))
        dcov2 = torch.clamp(dcov2, min=1e-12)
        
        dcov_loss = torch.sqrt(dcov2).mean()
        
        # 7. Final Loss
        mean_task_loss = task_losses.mean()
        
        # Check for warmup lambda (set by trainer)
        current_lambda = getattr(self, 'warmup_lambda', self.lambda_dcor)
        
        # Apply Penalty
        penalty_term = mean_task_loss.detach() * current_lambda * dcov_loss
        final_loss = mean_task_loss + penalty_term
        
        return final_loss, dcov_loss
        

def calc_ser_nd(labels: torch.Tensor, preds: torch.Tensor, relevances: torch.Tensor, t: float) -> torch.Tensor:
    """
    Calculates Sum of Squared Errors (SER) for relevant samples, per channel.

    Args:
        labels (Tensor): Shape (N,) or (N, C)
        preds (Tensor): Shape (N,) or (N, C)
        relevances (Tensor): Shape (N,) or (N, C)
        t (float): Relevance threshold.

    Returns:
        Tensor: A 1D Tensor of SER values with shape (C,).
    """
    # Convert all inputs to PyTorch tensors
    labels = torch.as_tensor(labels).float()
    preds = torch.as_tensor(preds).float()
    relevances = torch.as_tensor(relevances).float()
    
    # Ensure (N, C)
    if labels.ndim == 1: labels = labels.view(-1, 1)
    if preds.ndim == 1: preds = preds.view(-1, 1)
    if relevances.ndim == 1: relevances = relevances.view(-1, 1)

    num_channels = labels.shape[1]
    
    # Initialize output vector
    ser_values = torch.zeros(num_channels, dtype=labels.dtype, device=labels.device)

    for i in range(num_channels):
        # Masking for the specific channel
        mask = relevances[:, i] >= t
        if mask.any():
            l_filt = labels[mask, i]
            p_filt = preds[mask, i]
            ser_values[i] = torch.sum((l_filt - p_filt) ** 2)
        else:
            ser_values[i] = 0.0

    return ser_values


def calc_sera(labels: torch.Tensor, preds: torch.Tensor, relevances: torch.Tensor, sampling: int = 50, t: float = 0.8) -> torch.Tensor:
    """
    Calculates Sum of Squared Errors Area (SERA), per channel.

    Args:
        labels (Tensor): Shape (N,) or (N, C)
        preds (Tensor): Shape (N,) or (N, C)
        relevances (Tensor): Shape (N,) or (N, C)
        sampling (int): Number of steps for integration.
        t (float): Relevance threshold.

    Returns:
        Tensor: A 1D Tensor of SERA values with shape (C,).
    """
    labels = torch.as_tensor(labels).float()
    preds = torch.as_tensor(preds).float()
    relevances = torch.as_tensor(relevances).float()
    
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
        preds = preds.unsqueeze(1)
        relevances = relevances.unsqueeze(1)

    num_channels = labels.shape[1]
    
    # Create thresholds
    t_s = torch.linspace(t, 1, sampling, device=labels.device)
    
    # Create matrix to store SER values: Shape (Sampling_Steps, Num_Channels)
    ser_matrix = torch.zeros((sampling, num_channels), dtype=labels.dtype, device=labels.device)

    # Iterate over sampling steps
    for j, t_val in enumerate(t_s):
        # Vectorized call for all channels at this threshold
        ser_matrix[j] = calc_ser_nd(labels, preds, relevances, t_val)
        
    # Integrate using Trapezoidal rule along the sampling dimension (dim=0)
    sera_values = torch.trapezoid(ser_matrix, x=t_s, dim=0)
        
    return sera_values

def naiive_calc_alpha(labels: torch.Tensor, preds: torch.Tensor, densities: torch.Tensor) -> torch.Tensor:
    """
    Calculates the alpha metric (1 - |PCC(1/density, MAE)|) for all channels simultaneously.
    
    This implementation vectorizes the Pearson Correlation calculation to avoid loops.

    Args:
        labels (Tensor): Shape (N,) or (N, C)
        preds (Tensor): Shape (N,) or (N, C)
        densities (Tensor): Shape (N,) or (N, C)

    Returns:
        Tensor: A 1D Tensor of alpha values with shape (C,).
    """
    # 1. Standardize inputs
    labels = torch.as_tensor(labels).float()
    preds = torch.as_tensor(preds).float()
    densities = torch.as_tensor(densities).float()
    
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
        preds = preds.unsqueeze(1)
        densities = densities.unsqueeze(1)

    # N = Number of samples, C = Number of channels
    N, C = labels.shape
    # Edge Case: If N <= 1, correlation is undefined. 
    # Return 0.0 (default neutral value) for all channels.
    if N <= 1:
        return torch.zeros(C, dtype=labels.dtype, device=labels.device)
    
    maes = torch.abs(labels - preds)
    
    # 2. Prepare Variables (Shape: N, C)
    maes = torch.abs(labels - preds)
    inv_densities = 1 / torch.clamp(densities, min=1e-8)
    
    # 3. Vectorized Pearson Correlation Calculation
    # Formula: Cov(X, Y) / (Std(X) * Std(Y))
    # Equivalent to: Sum((X - meanX)*(Y - meanY)) / Sqrt(Sum((X-meanX)^2) * Sum((Y-meanY)^2))
    
    # A. Calculate Means along the sample dimension (dim=0)
    # Shape: (1, C)
    mae_mean = maes.mean(dim=0, keepdim=True)
    dens_mean = inv_densities.mean(dim=0, keepdim=True)

    # B. Center the data (X - mean)
    # Shape: (N, C) via broadcasting
    mae_centered = maes - mae_mean
    dens_centered = inv_densities - dens_mean

    # C. Calculate Numerator (Covariance unnormalized)
    # Shape: (C,)
    numerator = torch.sum(mae_centered * dens_centered, dim=0)

    # D. Calculate Denominator (Product of unnormalized Stds)
    # Shape: (C,)
    denom_mae = torch.sqrt(torch.sum(mae_centered ** 2, dim=0))
    denom_dens = torch.sqrt(torch.sum(dens_centered ** 2, dim=0))
    denominator = denom_mae * denom_dens

    # 4. Compute Correlation
    # Add epsilon to denominator to prevent division by zero (if constant values exist)
    correlations = numerator / (denominator + 1e-8)
    
    # Explicitly zero out correlations where denominator was effectively 0 (constant input)
    # This handles cases where a channel has identical values for all samples
    mask_undefined = denominator < 1e-8
    correlations[mask_undefined] = 0.0

    # 5. Calculate Alpha
    # alpha = 1 - |PCC|
    alpha_values = 1 - torch.abs(correlations)
    
    return alpha_values

def calc_alpha(labels: torch.Tensor, preds: torch.Tensor, densities: torch.Tensor) -> torch.Tensor:
    """
    Calculates awareness using dCor with Log-L1 Error.
    Normalized to ensure valid range [0, 1].
    """
    labels = torch.as_tensor(labels).float()
    preds = torch.as_tensor(preds).float()
    densities = torch.as_tensor(densities).float()
    
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
        preds = preds.unsqueeze(1)
        densities = densities.unsqueeze(1)

    N, C = labels.shape
    if N <= 3:
        return torch.zeros(C, dtype=labels.dtype, device=labels.device)

    # Log-L1 Error
    error_metric = torch.log1p(torch.abs(labels - preds))
    
    inv_densities = 1.0 / torch.clamp(densities, min=1e-6)

    # --- Z-SCORE NORMALIZATION (Critical for stability) ---
    # We detach std to avoid division instability in gradients if std=0
    error_metric = (error_metric - error_metric.mean(dim=0, keepdim=True)) / (error_metric.std(dim=0, keepdim=True) + 1e-6)
    inv_densities = (inv_densities - inv_densities.mean(dim=0, keepdim=True)) / (inv_densities.std(dim=0, keepdim=True) + 1e-6)
    # -----------------------------------------------------

    # Vectorized dCor Calculation
    X = inv_densities.T.unsqueeze(-1)  # (C, N, 1)
    Y = error_metric.T.unsqueeze(-1)   # (C, N, 1)
    
    a = torch.cdist(X, X, p=2)
    b = torch.cdist(Y, Y, p=2)
    
    a_mean_row = a.mean(dim=2, keepdim=True)
    a_mean_col = a.mean(dim=1, keepdim=True)
    a_mean_grand = a.mean(dim=(1, 2), keepdim=True)
    A_centered = a - a_mean_row - a_mean_col + a_mean_grand
    
    b_mean_row = b.mean(dim=2, keepdim=True)
    b_mean_col = b.mean(dim=1, keepdim=True)
    b_mean_grand = b.mean(dim=(1, 2), keepdim=True)
    B_centered = b - b_mean_row - b_mean_col + b_mean_grand
    
    # Statistics
    dcov2 = (A_centered * B_centered).mean(dim=(1, 2))
    dvar_x2 = (A_centered * A_centered).mean(dim=(1, 2))
    dvar_y2 = (B_centered * B_centered).mean(dim=(1, 2))
    
    # Safe Sqrt
    dcov2 = torch.clamp(dcov2, min=0.0)
    dvar_x2 = torch.clamp(dvar_x2, min=1e-12)
    dvar_y2 = torch.clamp(dvar_y2, min=1e-12)
    
    dcor = torch.sqrt(dcov2 + 1e-12) / (torch.sqrt(torch.sqrt(dvar_x2 * dvar_y2)) + 1e-8)
    
    return 1.0 - dcor


# def calc_ser(labels, preds, relevances, t):
#     keep_ind = np.where(relevances >= t)
#     labels, preds, relevances = labels[keep_ind], preds[keep_ind], relevances[keep_ind]
#     ser = np.sum(np.square(labels - preds))
#     return ser

# def calc_sera(labels, preds, relevances, sampling = 50, t = 0.6):
#     t_s = np.linspace(t, 1, sampling)
#     sers = np.zeros(sampling)
#     for i, t in enumerate(t_s):
#         sers[i] = calc_ser(labels, preds, relevances, t)
#     sera = np.trapz(sers, t_s)
#     return sera

# def calc_alpha(labels, preds, densities):
#     maes = np.abs(labels - preds)
#     return 1 - np.abs(np.corrcoef(1 / densities, maes)[0][-1])