"""Stage 3a of the MatImba workflow: out-of-distribution error analysis.

Functions here decompose a model's test error along the label distribution:
head/tail split parity (with marginal KDEs), label-binned MAE against the
test-set histogram, SER curves over the full relevance range, and a head/tail
metric table.  They work identically on standard cross-validation predictions
(in-distribution) and MatFold structure-disjoint predictions (OOD) — the
comparison between the two is the OOD analysis.

All functions accept either a single :class:`PredictionSet` or a list of runs
sharing the same test set (plotted as median with interquartile band).
"""

from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .collect import filter_bad_runs
from .predictions import EnsemblePrediction, PredictionSet, TAIL_PHI

PredGroup = Union[PredictionSet, Sequence[PredictionSet]]

_COLORS = ["#4d4d4d", "#2166ac", "#b2182b", "#35978f", "#ff7f0e"]
_FACE_COLORS = ["#e0e0e0", "#d1e5f0", "#fff5eb", "#c7eae5", "#ffbb78"]
_MARKERS = ["s", "o", "^", "v", "<", ">", "d", "p", "*", "h"]
_LINESTYLES = ["-", (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (5, 1)), (0, (1, 1))]


def _as_runs(group: PredGroup, drop_bad: bool = False) -> List[PredictionSet]:
    # drop_bad defaults to False: runs loaded via analysis.collect are already
    # filtered per fold, and re-filtering here would wrongly compare R2 across
    # folds with different test sets.
    runs = list(group) if isinstance(group, (list, tuple, np.ndarray)) else [group]
    return filter_bad_runs(runs) if drop_bad else runs


def _as_combined(group: PredGroup, drop_bad: bool = False):
    runs = _as_runs(group, drop_bad)
    return runs[0] if len(runs) == 1 else EnsemblePrediction(runs)


# ----------------------------------------------------------------------
# Head/tail split
# ----------------------------------------------------------------------
def get_tail_mask(pred, config: Dict[str, float]):
    """Tail mask from a one-entry config dict.

    Supported keys: ``relevance`` (phi threshold), ``quantile`` (target
    quantile), or ``targets``/``absolute``/``labels`` (absolute target value).
    Returns ``(mask, threshold_value, threshold_column)``.
    """
    key, val = next(iter(config.items()))
    if key == "quantile":
        threshold = np.quantile(pred.targets, val)
        return pred.targets > threshold, threshold, "targets"
    if key == "relevance":
        if pred.relevances is None:
            raise ValueError("Prediction object has no relevance values.")
        return pred.relevances > val, val, "relevance"
    if key in ("labels", "absolute", "targets"):
        return pred.targets > val, val, "targets"
    raise ValueError(f"Unknown threshold type: {key}")


def split_data(pred, threshold_config: Optional[Dict[str, float]] = None) -> dict:
    """Splits predictions into head and tail partitions.

    Default partition: relevance > TAIL_PHI (0.8), the paper's tail-MAE
    convention.
    """
    if threshold_config is None:
        threshold_config = {"relevance": TAIL_PHI}
    tail_mask, threshold_val, threshold_col = get_tail_mask(pred, threshold_config)

    def sl(arr, mask):
        return arr[mask] if arr is not None else np.array([])

    return {
        "tail": {"preds": sl(pred.preds, tail_mask), "gt": sl(pred.targets, tail_mask),
                 "relevance": sl(pred.relevances, tail_mask)},
        "head": {"preds": sl(pred.preds, ~tail_mask), "gt": sl(pred.targets, ~tail_mask),
                 "relevance": sl(pred.relevances, ~tail_mask)},
        "metadata": {"threshold_val": threshold_val, "threshold_col": threshold_col,
                     "config": threshold_config, "total_count": len(pred.targets)},
    }


def head_tail_table(groups: Dict[str, PredGroup], phi: float = TAIL_PHI) -> pd.DataFrame:
    """Head vs tail MAE (mean +/- sample std over runs, ddof=1) per method.

    ``groups`` maps method label -> PredictionSet or list of runs.
    """
    rows = []
    for label, group in groups.items():
        runs = _as_runs(group)
        head = np.array([p.head_mae(phi) for p in runs])
        tail = np.array([p.tail_mae(phi) for p in runs])
        mae = np.array([p.mae for p in runs])
        ddof = 1 if len(runs) > 1 else 0
        rows.append({
            "method": label,
            "n_runs": len(runs),
            "mae_mean": mae.mean(), "mae_std": mae.std(ddof=ddof),
            "head_mae_mean": np.nanmean(head), "head_mae_std": np.nanstd(head, ddof=ddof),
            "tail_mae_mean": np.nanmean(tail), "tail_mae_std": np.nanstd(tail, ddof=ddof),
        })
    return pd.DataFrame(rows).set_index("method")


# ----------------------------------------------------------------------
# Parity plots
# ----------------------------------------------------------------------
def plot_split_parity(ax, split: dict, target_name: str = ""):
    """Parity plot with head (grey) and low/high tail (blue/red) partitions,
    per-partition MAE in the legend, and head-range guide lines."""
    head_gt, head_preds = split["head"]["gt"], split["head"]["preds"]
    tail_gt, tail_preds = split["tail"]["gt"], split["tail"]["preds"]

    head_min, head_max = np.min(head_gt), np.max(head_gt)
    mae_head = np.mean(np.abs(head_gt - head_preds))
    ax.scatter(head_gt, head_preds, c="gray", alpha=0.3, s=20, label=f"Head: {mae_head:.2f}")

    low_mask = tail_gt < head_min
    high_mask = tail_gt > head_max
    if np.any(low_mask):
        mae_low = np.mean(np.abs(tail_gt[low_mask] - tail_preds[low_mask]))
        ax.scatter(tail_gt[low_mask], tail_preds[low_mask], c="#1f77b4", alpha=0.6, s=20,
                   label=f"Low Tail: {mae_low:.2f}")
        ax.axvline(head_min, color="k", ls=":", lw=1.5)
        ax.axhline(head_min, color="k", ls=":", lw=1.5)
    if np.any(high_mask):
        mae_high = np.mean(np.abs(tail_gt[high_mask] - tail_preds[high_mask]))
        ax.scatter(tail_gt[high_mask], tail_preds[high_mask], c="#d62728", alpha=0.6, s=20,
                   label=f"High Tail: {mae_high:.2f}")
        ax.axvline(head_max, color="k", ls=":", lw=1.5)
        ax.axhline(head_max, color="k", ls=":", lw=1.5)
    other_mask = ~(low_mask | high_mask)
    if np.any(other_mask):
        mae_other = np.mean(np.abs(tail_gt[other_mask] - tail_preds[other_mask]))
        ax.scatter(tail_gt[other_mask], tail_preds[other_mask], c="orange", alpha=0.6, s=20,
                   label=f"Other Tail: {mae_other:.2f}")

    all_vals = np.concatenate([head_gt, tail_gt, head_preds, tail_preds])
    min_v, max_v = np.min(all_vals), np.max(all_vals)
    ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=2)
    ax.legend(fontsize=8, loc="upper left", handletextpad=0.5, borderpad=0.2,
              borderaxespad=0.3, labelspacing=0.6)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel(f"True {target_name}", fontsize=10)
    ax.set_ylabel(f"Pred {target_name}", fontsize=10)


def plot_marginal_kde(ax, gt, preds, n_total, subset_label, color,
                      orientation="horizontal", head_bounds=None):
    """Marginal Gaussian-KDE curves of ground truth (dashed) vs predictions
    (filled), weighted by subset size.  Note: KDEs have infinite support, so
    smooth tails naturally extend past the head/tail boundary."""
    if len(gt) < 5:
        return

    def kde_curve(data, weight, label, c, linestyle="-", alpha=1.0, filled=False):
        try:
            kde = gaussian_kde(data)
            buffer = (data.max() - data.min()) * 0.1 or 1e-6
            grid = np.linspace(data.min() - buffer, data.max() + buffer, 500)
            density = kde(grid) * weight
            if orientation == "horizontal":
                if filled:
                    ax.fill_between(grid, density, color=c, alpha=alpha, label=label)
                else:
                    ax.plot(grid, density, color=c, linestyle=linestyle, lw=1.5, label=label)
            else:
                if filled:
                    ax.fill_betweenx(grid, density, color=c, alpha=alpha, label=label)
                else:
                    ax.plot(density, grid, color=c, linestyle=linestyle, lw=1.5, label=label)
        except np.linalg.LinAlgError:
            pass

    two_sided = False
    if head_bounds is not None and subset_label == "Tail":
        head_min, head_max = head_bounds
        low_mask, high_mask = gt < head_min, gt > head_max
        if low_mask.sum() > 2 and high_mask.sum() > 2:
            two_sided = True
            w_low = low_mask.sum() / n_total
            kde_curve(gt[low_mask], w_low, "Low Tail GT", "#1f77b4", linestyle="--")
            kde_curve(preds[low_mask], w_low, "Low Tail Pred", "#1f77b4", alpha=0.3, filled=True)
            w_high = high_mask.sum() / n_total
            kde_curve(gt[high_mask], w_high, "High Tail GT", "#d62728", linestyle="--")
            kde_curve(preds[high_mask], w_high, "High Tail Pred", "#d62728", alpha=0.3, filled=True)

    if not two_sided:
        weight = len(gt) / n_total
        kde_curve(gt, weight, f"{subset_label} Truth", color, linestyle="--")
        kde_curve(preds, weight, f"{subset_label} Pred", color, alpha=0.3, filled=True)

    ax.grid(True, alpha=0.2)
    if orientation == "horizontal":
        ax.set_yticks([])
        ax.text(0.125, 0.5, "Tail", va="center", ha="center", transform=ax.transAxes)
    else:
        ax.set_xticks([])
        ax.set_xlabel(r"$\rho$", labelpad=9, fontsize=10)
        ax.xaxis.set_label_position("top")
        ax.text(0.5, 0.825, "Head", va="center", ha="center", transform=ax.transAxes,
                rotation=90)


def composite_parity(group: PredGroup, target_name: str = "",
                     threshold_config: Optional[Dict[str, float]] = None,
                     axes=None, title: Optional[str] = None):
    """Split parity plot with marginal head/tail KDEs (paper Fig. 3 panel).

    ``axes`` = (ax_main, ax_top, ax_right) or None to create a standalone
    figure.  Returns the axes used.
    """
    import matplotlib.gridspec as gridspec

    combined = _as_combined(group)
    split = split_data(combined, threshold_config)
    n_total = split["metadata"]["total_count"]
    head_gt = split["head"]["gt"]
    head_bounds = (
        (head_gt.min(), head_gt.max()) if len(head_gt) > 0
        else (split["tail"]["gt"].min(), split["tail"]["gt"].max())
    )

    if axes is None:
        fig = plt.figure(figsize=(3.5, 3))
        gs = gridspec.GridSpec(2, 2, width_ratios=[7, 1], height_ratios=[1, 7],
                               wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    else:
        ax_main, ax_top, ax_right = axes

    plot_split_parity(ax_main, split, target_name)
    plot_marginal_kde(ax_top, split["tail"]["gt"], split["tail"]["preds"], n_total,
                      "Tail", "#d62728", orientation="horizontal", head_bounds=head_bounds)
    plot_marginal_kde(ax_right, split["head"]["gt"], split["head"]["preds"], n_total,
                      "Head", "gray", orientation="vertical")
    if title:
        ax_top.set_title(title, loc="center", fontsize=10)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    return ax_main, ax_top, ax_right


# ----------------------------------------------------------------------
# Binned MAE and SER curves
# ----------------------------------------------------------------------
def _rebin_on_edges(pred: PredictionSet, bin_edges: np.ndarray) -> np.ndarray:
    """Mean |error| of one run per bin of a common edge grid (runs from
    different folds have different test sets, so their own FD bins differ)."""
    nbins = len(bin_edges) - 1
    locs = np.fmin(np.digitize(pred.targets, bin_edges), nbins)
    out = np.full(nbins, np.nan)
    for j in range(nbins):
        mask = locs == j + 1
        if mask.any():
            out[j] = pred.abs_errors[mask].mean()
    return out


def compare_binned_mae(groups: Dict[str, PredGroup], target_name: str = "", ax=None):
    """Median label-binned MAE per method (markers with IQR error bars for
    ensembles) over the test-set histogram.  Bins are computed on the pooled
    targets of all runs so cross-fold groups share a common grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 2.8), layout="compressed")
    axtwin = ax.twinx()

    # Common bin grid over all methods' pooled targets
    all_runs = [p for group in groups.values() for p in _as_runs(group)]
    pooled = np.concatenate([p.targets for p in all_runs])
    hist, bin_edges = np.histogram(pooled, bins="fd")
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (bin_edges[-1] - bin_edges[0]) / len(x)
    # Average counts per run (equals the raw test-set histogram when all
    # runs share one test set)
    axtwin.bar(x, hist / max(len(all_runs), 1), color="#4d4d4d", alpha=0.15,
               width=bin_width, linewidth=0)

    global_max_mae = 0
    for i, (label, group) in enumerate(groups.items()):
        runs = _as_runs(group)
        all_binned = np.vstack([_rebin_on_edges(p, bin_edges) for p in runs])
        valid_mask = ~np.isnan(all_binned).all(axis=0)
        all_binned = all_binned[:, valid_mask]
        x_vals = x[valid_mask]
        median_binned = np.nanmedian(all_binned, axis=0)
        safe_max = np.nanpercentile(median_binned, 95) * 1.5
        global_max_mae = max(global_max_mae, safe_max)
        c = i % len(_COLORS)

        if len(runs) > 1:
            lower = np.nanpercentile(all_binned, 25, axis=0)
            upper = np.clip(np.nanpercentile(all_binned, 75, axis=0), None, safe_max * 2)
            ax.errorbar(x_vals, median_binned,
                        yerr=[np.maximum(median_binned - lower, 0),
                              np.maximum(upper - median_binned, 0)],
                        c=_COLORS[c], marker=_MARKERS[c], ms=6,
                        markerfacecolor=_FACE_COLORS[c], alpha=0.8, label=label,
                        linestyle="none", capsize=0, elinewidth=1.5)
        else:
            ax.plot(x_vals, median_binned, c=_COLORS[c], marker=_MARKERS[c], ms=6,
                    markerfacecolor=_FACE_COLORS[c], alpha=0.8, label=label,
                    linestyle="none")

    axtwin.set_ylabel("Testset Counts", fontsize=10)
    ax.set_xlabel(target_name if target_name else r"$y$", fontsize=10)
    ax.set_ylabel("Median MAE (Test)", fontsize=10)
    ax.set_zorder(axtwin.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.grid(False)
    axtwin.grid(False)
    if global_max_mae > 0:
        ax.set_ylim(-0.05 * global_max_mae, global_max_mae)
    ax.legend(fontsize=8, loc="upper center", handletextpad=0.5, borderpad=0.2,
              borderaxespad=0.3, labelspacing=0.6)
    return ax


def compare_ser_curves(groups: Dict[str, PredGroup], target_name: str = "", ax=None):
    """Median SER(t) curve per method over relevance thresholds t in [0, 1],
    with IQR band for ensembles.  The area over [SERA_T0, 1] is SERA."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 2.8), layout="compressed")
    global_max_ser = 0

    for i, (label, group) in enumerate(groups.items()):
        runs = _as_runs(group)
        t_s = runs[0].t_s
        all_sers = np.vstack([p.sers for p in runs])
        median_sers = np.nanmedian(all_sers, axis=0)
        safe_max = np.nanmax(median_sers[:-int(len(median_sers) * 0.05)]) * 1.5
        if not np.isnan(safe_max):
            global_max_ser = max(global_max_ser, safe_max)
        c = i % len(_COLORS)
        ax.plot(t_s, median_sers, ls=_LINESTYLES[c % len(_LINESTYLES)], color=_COLORS[c],
                lw=2, alpha=0.8, label=label)
        if len(runs) > 1:
            ax.fill_between(t_s, np.nanpercentile(all_sers, 25, axis=0),
                            np.nanpercentile(all_sers, 75, axis=0),
                            color=_COLORS[c], alpha=0.15, linewidth=0)

    if target_name:
        clean = target_name.replace("$", "").replace("{{", "{").replace("}}", "}")
        ax.set_xlabel(rf"Relevance $\phi({clean})$")
    else:
        ax.set_xlabel(r"Relevance $\phi(y)$", fontsize=10)
    ax.set_ylabel("Median SER", fontsize=10)
    if global_max_ser > 0:
        ax.set_ylim(-0.05 * global_max_ser, global_max_ser)
    ax.legend(fontsize=8, loc="upper right", handletextpad=0.5, borderpad=0.2,
              borderaxespad=0.3, labelspacing=0.6)
    return ax


def error_vs_density(group: PredGroup, ax=None, n_bins: int = 12):
    """Mean absolute error vs local label density (log-spaced density bins) —
    the raw relationship that awareness alpha summarises."""
    combined = _as_combined(group)
    if combined.densities is None:
        raise ValueError("Predictions carry no density values.")
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 2.8), layout="compressed")

    dens = combined.densities
    errs = combined.abs_errors
    edges = np.quantile(dens, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (dens >= lo) & (dens <= hi)
        if mask.sum() > 1:
            centers.append(dens[mask].mean())
            means.append(errs[mask].mean())
            stds.append(errs[mask].std(ddof=1))
    ax.errorbar(centers, means, yerr=stds, marker="o", ms=5, color="#2166ac",
                linestyle="-", lw=1.2, capsize=2, alpha=0.85)
    ax.set_xlabel(r"local label density $\rho$", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    return ax


def summary_plot(groups: Dict[str, PredGroup], target_name: str = "",
                 file_name: Optional[str] = None):
    """Composite figure: one split-parity panel per method + SER curves +
    binned MAE (paper Fig. 3 layout)."""
    import matplotlib.gridspec as gridspec

    labels = list(groups)
    num_plots = len(labels) + 2
    ncols = 3 if num_plots != 4 else 2
    nrows = int(np.ceil(num_plots / ncols))
    fig = plt.figure(figsize=((2.8 + 0.15) * ncols + 0.15, (2.75 + 0.1) * nrows + 0.1))
    gs_main = gridspec.GridSpec(nrows, ncols, figure=fig)
    label_axes = []

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        if i < len(labels):
            inner = gridspec.GridSpecFromSubplotSpec(8, 8, subplot_spec=gs_main[i])
            ax_top = fig.add_subplot(inner[0, 0:7])
            ax_main = fig.add_subplot(inner[1:8, 0:7])
            ax_right = fig.add_subplot(inner[1:8, 7])
            ax_top.sharex(ax_main)
            ax_right.sharey(ax_main)
            composite_parity(groups[labels[i]], target_name=target_name,
                             axes=(ax_main, ax_top, ax_right), title=labels[i])
            label_axes.append(ax_top)
        elif i == len(labels):
            ax_ser = fig.add_subplot(gs_main[r, c])
            compare_ser_curves(groups, target_name=target_name, ax=ax_ser)
            label_axes.append(ax_ser)
        elif i == len(labels) + 1:
            ax_mae = fig.add_subplot(gs_main[r, c])
            compare_binned_mae(groups, target_name=target_name, ax=ax_mae)
            label_axes.append(ax_mae)

    for i, ax in enumerate(label_axes):
        ax.set_title(chr(ord("a") + i), loc="left", fontsize=10, y=1.05)
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
    return fig
