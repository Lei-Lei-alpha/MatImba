import numpy as np
import pandas as pd
from typing import Union
import statsmodels.api as sm
from scipy.special import erf
from scipy.signal.windows import triang
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.stats import entropy, gaussian_kde, wasserstein_distance
from typing import Dict, List, Optional, Tuple, Union

# Define Types
ArrayLike = Union[np.ndarray, List[float]]

def get_kernel_window(
    kernel: str = "gaussian", 
    kernel_size: int = 7, 
    sigma: int = 3
) -> np.ndarray:
    """
    Generate a normalised kernel window for smoothing.

    Parameters:
    - kernel: Type of kernel ('gaussian', 'triang', 'laplace').
    - kernel_size: Size of the kernel window (odd integer recommended).
    - sigma: Standard deviation for gaussian/laplace kernels.

    Returns:
    - Normalised kernel window as a numpy array.
    """
    if kernel not in ["gaussian", "triang", "laplace"]:
        raise ValueError(f"Unknown kernel '{kernel}'. Supported: 'gaussian', 'triang', 'laplace'.")
    
    half_ks = (kernel_size - 1) // 2
    
    if kernel == "gaussian":
        base_kernel = np.zeros(kernel_size)
        base_kernel[half_ks] = 1.0
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
        kernel_window /= kernel_window.max()
        
    elif kernel == "triang":
        kernel_window = triang(kernel_size)
        
    else:  # laplace
        x = np.arange(-half_ks, half_ks + 1)
        kernel_window = np.exp(-np.abs(x) / sigma) / (2.0 * sigma)
        kernel_window /= kernel_window.max()
        
    return kernel_window


def calc_density(counts: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Calculate normalized densities from histogram counts and bin edges.

    Parameters:
    - counts: Histogram counts.
    - bin_edges: Bin edges from np.histogram.

    Returns:
    - Normalized densities (integral over bins approximates to 1).
    """
    db = np.diff(bin_edges).astype(float)
    densities = counts / db
    return densities / counts.sum()


def estimate_density(
    labels: np.ndarray, 
    bins: Optional[Union[int, str]] = None, 
    smooth: Optional[str] = "kde", 
    **convolve_params
) -> np.ndarray:
    """
    Estimate density at each label point using KDE, convolution smoothing, or raw histogram.

    Handles multi-channel (N, C) arrays by calculating density per channel.

    Parameters:
    - labels: Array of labels/values. Shape (N,) or (N, C).
    - bins: Number of bins for histogram (auto if None using 'fd').
    - smooth: Smoothing method ('kde', 'convolve', None).
    - **convolve_params: Parameters for get_kernel_window if 'convolve'.

    Returns:
    - Density estimates at each label. Shape (N,) or (N, C).
    """
    # Normalize input to (N, C)
    original_ndim = labels.ndim
    if original_ndim == 1:
        labels = labels.reshape(-1, 1)
    elif original_ndim > 2:
        raise ValueError(f"Input must be 1D (N,) or 2D (N, C), but got {original_ndim} dimensions.")

    n_samples, n_channels = labels.shape
    all_densities = np.zeros((n_samples, n_channels))
    
    if smooth not in {"kde", "convolve", None}:
        raise ValueError(f"Unknown smooth method '{smooth}'. Supported: 'kde', 'convolve', None.")
    
    for c in range(n_channels):
        labels_c = labels[:, c]
        
        if smooth == "kde":
            try:
                kde = gaussian_kde(labels_c)
                densities_c = kde.evaluate(labels_c)
            except np.linalg.LinAlgError:
                # Fallback for singular matrix (e.g., all values identical)
                densities_c = np.ones_like(labels_c)
        else:
            # Histogram based methods
            # Calculate bin edges if not provided or 'fd'
            if bins is None or isinstance(bins, str):
                # Use numpy's histogram to determine optimal bins
                _, bin_edges = np.histogram(labels_c, bins="fd" if bins is None else bins)
            else:
                _, bin_edges = np.histogram(labels_c, bins=bins)
            
            # Get counts with fixed bins
            counts, _ = np.histogram(labels_c, bins=bin_edges)

            if smooth == "convolve":
                kernel_window = get_kernel_window(**convolve_params)
                counts = convolve1d(counts, weights=kernel_window, mode="constant")
            
            # Calculate density from counts
            density_vals = calc_density(counts, bin_edges)
            
            # Interpolate back to original label points
            x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Handle edge case where only 1 bin exists
            if len(x_centers) < 2:
                 densities_c = np.full_like(labels_c, density_vals[0] if len(density_vals)>0 else 0.0)
            else:
                density_func = PchipInterpolator(x_centers, density_vals, extrapolate=True)
                densities_c = density_func(labels_c)
                
            # Clean up numerical artifacts
            densities_c = np.nan_to_num(densities_c, nan=0.0, posinf=0.0, neginf=0.0)
            densities_c = np.maximum(densities_c, 0.0) # Density cannot be negative

        all_densities[:, c] = densities_c

    # Return to original shape if input was 1D
    if original_ndim == 1:
        return all_densities.flatten()
        
    return all_densities

def focus_func(
    Y: np.ndarray, 
    mu: float, 
    sigma: float, 
    alpha: Optional[float] = None, 
    amp: Optional[float] = None
) -> np.ndarray:
    """
    Compute a focus function (gaussian or skewed gaussian).

    Parameters:
    - Y: Input values.
    - mu: Mean.
    - sigma: Standard deviation.
    - alpha: Skewness parameter (if None, standard gaussian).
    - amp: Amplitude (default 1).

    Returns:
    - Focus values for each Y.
    """
    amp = amp if amp is not None else 1.0
    
    if alpha is None:
        return amp * np.exp(- (Y - mu)**2 / (2 * sigma**2))
    else:
        normpdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((Y - mu)**2 / (2 * sigma**2)))
        normcdf = 0.5 * (1 + erf(alpha * (Y - mu) / (sigma * np.sqrt(2))))
        return amp * normpdf * normcdf

def get_weights(
    density: np.ndarray, 
    method: str = "log_inv", 
    eps: float = 0.1, 
    focus: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Compute target weights from densities.

    Parameters:
    - density: Density estimates.
    - method: Weighting method ('log_inv', 'sqrt_inv', 'exp_decay', 'inverse').
    - eps: Epsilon for minimum weight adjustment (if >0, scale to [eps,1]; if 0, normalize to [0,1]).
    - focus: Optional dict with 'mu', 'sigma', 'alpha', 'amp' for focus multiplication.

    Returns:
    - Weights array.
    """
    valid_methods = {"log_inv", "sqrt_inv", "exp_decay", "inverse"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")
    
    # Calculate raw weights
    if method == "log_inv":
        weights = 1.0 / np.log(density + np.e)
    elif method == "sqrt_inv":
        weights = 1.0 / (0.1 + density)
    elif method == "inverse":
        weights = 1.0 / np.clip(density, 5e-3, None)
    else:  # exp_decay
        weights = np.exp(-density)

    # Apply focus function if provided
    if focus:
        # density passed to focus_func seems conceptually odd (usually focus is on target values Y)
        # Assuming the intent is to boost weights where density is specific value OR 
        # maybe focus was meant for Y values. Keeping logic as per original script but noting ambiguity.
        # Based on original: focus_func(density, **focus)
        f_vals = focus_func(density, **focus)
        weights += f_vals

    # Normalize
    max_w = weights.max()
    if max_w > 0:
        weights /= max_w
            
    # Scale
    if eps > 0:
        # Log-scaling to range [eps, 1]
        # Avoid log(0)
        weights = np.clip(weights, 1e-8, 1.0)
        ln_weights = np.log(weights)
        min_ln = ln_weights.min()
        
        if min_ln != 0: # If weights aren't all 1.0
            ln_weights *= np.log(eps) / min_ln
            weights = np.exp(ln_weights)
    else:
        # Min-Max scaling to range [0, 1]
        min_w = weights.min()
        range_w = weights.max() - min_w
        if range_w > 0:
            weights = (weights - min_w) / range_w
        else:
            weights = np.ones_like(weights)
        
    return weights


def adjusted_boxplot_params(data: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute adjusted boxplot parameters for skewed data using medcouple.

    Parameters:
    - data: Input data array.

    Returns:
    - lower_bound: Lower outlier bound.
    - upper_bound: Upper outlier bound.
    - whis: (low_whis, high_whis) percentages for boxplot.
    """
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1
    
    # Calculate medcouple (robust skewness measure)
    y = np.asarray(data, dtype=np.double) 
    mc = sm.stats.stattools.medcouple(y) 

    # Define the outlier range based on skewness
    if mc > 0:
        lower_bound = q1 - 1.5 * np.exp(-4 * mc) * iqr
        upper_bound = q3 + 1.5 * np.exp(3 * mc) * iqr
    else:
        lower_bound = q1 - 1.5 * np.exp(-3 * mc) * iqr
        upper_bound = q3 + 1.5 * np.exp(4 * mc) * iqr
    
    # Calculate whisker percentiles for plotting
    sorted_data = np.sort(data)
    n = len(data)
    # Use np.searchsorted for faster lookup than interp if exact mapping isn't needed, 
    # but interp is smoother for percentiles.
    whis_low = np.interp(lower_bound, sorted_data, np.linspace(0, 1, n)) * 100
    whis_high = np.interp(upper_bound, sorted_data, np.linspace(0, 1, n)) * 100
    
    return lower_bound, upper_bound, (whis_low, whis_high)

def select_outliers_adjusted_boxplot(
    df: pd.DataFrame, 
    col_name: str, 
    show_plot: bool = False
) -> pd.DataFrame:
    """
    Select outlier rows using adjusted boxplot.

    Parameters:
    - df: DataFrame.
    - col_name: Column to check for outliers.
    - show_plot: If True, display boxplot.

    Returns:
    - DataFrame with outlier rows.
    """
    data = df[col_name].values
    lower_bound, upper_bound, whis = adjusted_boxplot_params(data)
    
    outliers = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]

    if show_plot:
        # Lazy import to avoid dependency if not plotting
        import seaborn as sns
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col_name], whis=whis)
        plt.title(f"Adjusted Boxplot for {col_name}")
        plt.show()

    return outliers


def plot_relevance(
    labels: np.ndarray, 
    relevances: np.ndarray, 
    whis: Tuple[float, float], 
    fig_name: Optional[str] = None
) -> None:
    """
    Plot boxplot and relevance curve.

    Parameters:
    - labels: Sorted labels.
    - relevances: Corresponding relevances.
    - whis: Whisker percentages.
    - fig_name: Optional save path.
    """
    fig = plt.figure(figsize=(3.2, 2.8), layout="constrained")
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1:])
    
    medianprops = dict(color="black", linewidth=1.5)
    ax1.boxplot(
        labels, 
        vert=False, 
        whis=whis,
        medianprops=medianprops, 
        patch_artist=True,
        boxprops={"facecolor": "#d1e1ee"}, 
        flierprops={
            "marker": "o", "markersize": 4,
            "markerfacecolor": "#d1e1ee",
            "linewidth": 0.25
        }
    )
    ax1.set_ylim(0.8, 1.2)
    ax1.set_axis_off()
    
    ax2.plot(labels, relevances, lw=2, c="#a1c3d8")
    ax2.hlines(1, labels.min(), labels.max(), colors="#a4a4a4", lw=1.5, ls="-.", alpha=0.65)
    
    ax2.set_ylabel(r"$\phi (y)$")
    ax2.set_xlabel(r"$y$")
    
    if fig_name:
        plt.savefig(fig_name, dpi=600)
    plt.show()


def calc_relevance(
    labels: np.ndarray, 
    adjusted_box_plot: bool = True, 
    eps: float = 1e-4,
    sort: bool = False, 
    plot: bool = True, 
    fig_name: Optional[str] = None
) -> np.ndarray:
    """
    Calculate target relevance scores based on boxplot bounds.

    Parameters:
    - labels: Label values.
    - adjusted_box_plot: Use medcouple adjustment if True.
    - eps: Minimum relevance at median.
    - sort: Return sorted relevances if True.
    - plot: Plot if True.
    - fig_name: Save plot if provided.

    Returns:
    - Relevance array (unsorted unless sort=True).
    """
    if adjusted_box_plot:
        lower_bound, upper_bound, whis = adjusted_boxplot_params(labels)
    else:
        q1, q3 = np.quantile(labels, [0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Calculate whiskers for standard boxplot (1.5 IQR)
        # We map these back to percentiles for plotting consistency
        sorted_labels = np.sort(labels)
        n = len(labels)
        whis_low = np.interp(lower_bound, sorted_labels, np.linspace(0, 1, n)) * 100
        whis_high = np.interp(upper_bound, sorted_labels, np.linspace(0, 1, n)) * 100
        whis = (whis_low, whis_high)
    
    median = np.median(labels)

    # Define control points for relevance interpolation
    # Structure: [outlier_low, ..., lower_bound, median, upper_bound, ..., outlier_high]
    # Relevance: [1,           ..., 1,           eps,    1,           ..., 1]
    
    # Identify outliers
    outliers_low = labels[labels < lower_bound]
    outliers_high = labels[labels > upper_bound]
    
    # Sort outliers for interpolation stability
    rf_x_l = np.sort(outliers_low)
    rf_y_l = np.ones_like(rf_x_l)
    
    rf_x_h = np.sort(outliers_high)
    rf_y_h = np.ones_like(rf_x_h)
    
    # Core control points
    rf_x_core = np.array([lower_bound, median, upper_bound])
    rf_y_core = np.array([1.0, eps, 1.0])
    
    # Combine all points
    rf_x = np.concatenate([rf_x_l, rf_x_core, rf_x_h])
    rf_y = np.concatenate([rf_y_l, rf_y_core, rf_y_h])
    
    # Ensure x is strictly increasing for PchipInterpolator
    # 1. Sort unique values
    # 2. If duplicates exist (e.g., median == lower_bound), drop duplicates
    #    but preserve the logical 'shape' (e.g. if median=bound, relevance goes 1->eps instantly)
    unique_x, unique_indices = np.unique(rf_x, return_index=True)
    
    # Since unique sorts, we need to ensure y follows.
    # However, unique_indices returns the *first* occurrence. 
    # If we have x=[10, 10], y=[1, 0.001], unique keeps index 0 (y=1).
    # We likely want the 'inner' value (eps) if bounds collide.
    # For robustness, let's just use the unique x and corresponding y.
    
    rf_x_sorted = rf_x[np.argsort(rf_x)]
    rf_y_sorted = rf_y[np.argsort(rf_x)]
    
    # Remove duplicates
    valid_mask = np.concatenate(([True], np.diff(rf_x_sorted) > 1e-8))
    rf_x_clean = rf_x_sorted[valid_mask]
    rf_y_clean = rf_y_sorted[valid_mask]
    
    # Interpolate
    if len(rf_x_clean) < 2:
        relevance = np.ones_like(labels)
    else:
        relevant_func = PchipInterpolator(rf_x_clean, rf_y_clean, extrapolate=True)
        
        # Apply to sorted labels for plotting/calculation
        sorted_labels = np.sort(labels)
        relevance_sorted = relevant_func(sorted_labels)
        relevance_sorted = np.clip(relevance_sorted, 0.0, 1.0)
        
        if plot:
            plot_relevance(sorted_labels, relevance_sorted, whis, fig_name)
            
        # Map back to original order
        # We can't just use argsort reverse because multiple values might be identical.
        # We must re-evaluate the function on the original labels.
        relevance = relevant_func(labels)
        relevance = np.clip(relevance, 0.0, 1.0)
    
    if sort:
        # Return sorted by label value
        sort_idx = np.argsort(labels)
        return relevance[sort_idx]

    return relevance


def calc_dil(
    dataset: np.ndarray, 
    bins: str = 'fd', 
    method: str = 'pietra'
) -> float:
    """
    Calculates a [0,1] bounded distribution imbalance level.
    
    Parameters:
    - dataset: Target values.
    - bins: Binning strategy.
    - method: 'pietra' (L1-norm, recommended) or 'bounded_cv' (L2-norm soft bound).
    
    Returns:
    - Imbalance score in [0, 1].
    """
    dataset = np.asarray(dataset).flatten()
    
    # 1. Get Smoothed Density (using LDS concepts from your paper)
    # We use counts directly as the metric is ratio-invariant
    counts, _ = np.histogram(dataset, bins=bins)
    
    # Optional: Apply minimal smoothing to remove binning noise (as in previous step)
    kernel = triang(3)
    counts = convolve1d(counts, weights=kernel/kernel.sum(), mode="mirror")
    
    # Mean density
    mu = counts.mean()
    if mu == 0: return 0.0
    
    if method == 'pietra':
        # --- Method A: Pietra Ratio (L1 Norm) ---
        # Physically interpretation: Fraction of data that needs to be moved 
        # to achieve uniformity.
        # Max value is (N-1)/N, which -> 1.0 for large N.
        abs_dev = np.abs(counts - mu).sum()
        total_mass = counts.sum() # = mu * N
        return 0.5 * (abs_dev / total_mass) * (len(counts) / (len(counts)-1))
        # Note: The factor (N / N-1) strictly normalizes the max to 1.0 for finite N
        
    elif method == 'bounded_cv':
        # --- Method B: Soft Bounded CV (L2 Norm) ---
        # Maps [0, inf) -> [0, 1] using x / (x+1)
        # Preserves the "variance" sensitivity of original DIL
        sigma = counts.std()
        cv = sigma / mu
        return cv / (cv + 1)
    
    else:
        raise ValueError("Unknown method")


def calc_comprehensive_imbalance(
    dataset: np.ndarray, 
    bins: str = 'fd'
) -> dict:
    """
    Calculates the Robust DIL (Pietra) alongside Gini, KL, and Wasserstein metrics.
    All metrics in this function are effectively bounded or normalized for 
    easier comparison across different datasets.
    """
    dataset = np.asarray(dataset).flatten()
    
    # 1. Get Density (using Freedman-Diaconis rule by default)
    counts, bin_edges = np.histogram(dataset, bins=bins)
    
    # Convert to float and handle potential zeros for stability
    # (Though raw counts are preferred for Pietra to avoid float errors)
    n_bins = len(counts)
    total_mass = counts.sum()
    
    # Handle edge case: Empty or single-value dataset
    if total_mass == 0 or n_bins <= 1:
        return {
            "DIL_Robust": 0.0,
            "Gini": 0.0,
            "KL_Div": 0.0,
            "Wasserstein": 0.0
        }

    # --- Metric 1: Robust DIL (Pietra Ratio) ---
    # Formula: 0.5 * (Sum of Absolute Deviations) / Total Mass
    # Interpretation: The fraction of mass that must be moved to match the mean.
    mu = counts.mean()
    abs_dev = np.abs(counts - mu).sum()
    
    # Standard Pietra is bounded by [0, 1 - 1/N]. 
    # We apply the correction factor (N / N-1) to map strictly to [0, 1].
    correction_factor = n_bins / (n_bins - 1)
    dil_robust = 0.5 * (abs_dev / total_mass) * correction_factor
    
    # --- Metric 2: Gini Coefficient of Density ---
    # Measures inequality of the histogram heights
    density = counts.astype(float) + 1e-9  # Epsilon for stability
    sorted_p = np.sort(density)
    index = np.arange(1, n_bins + 1)
    gini = ((2 * index - n_bins - 1) * sorted_p).sum() / (n_bins * sorted_p.sum())
    
    # --- Metric 3: KL Divergence (from Uniform) ---
    # Information theoretic "surprise" relative to uniform
    density_norm = density / density.sum() # Probability Mass Function
    uniform_dist = np.ones(n_bins) / n_bins
    kl_div = entropy(density_norm, uniform_dist)
    
    # --- Metric 4: Wasserstein Distance (to Uniform) ---
    # Geometric transport cost
    # Normalize data to [0,1] for scale-invariant distance
    if dataset.max() == dataset.min():
        ws_dist = 0.0
    else:
        data_scaled = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        uniform_samples = np.linspace(0, 1, len(dataset))
        ws_dist = wasserstein_distance(data_scaled, uniform_samples)
    
    return {
        "DIL": dil_robust,   # [0,1] Bounded, replacing CV
        "Gini": gini,               # [0,1] Bounded
        "KL_Div": kl_div,           # Unbounded (0 to inf)
        "Wasserstein": ws_dist      # ~[0, 0.5] for normalized data
    }
