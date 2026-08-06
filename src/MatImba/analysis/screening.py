"""Stage 3b of the MatImba workflow: real screening performance.

Translates regression accuracy into virtual high-throughput screening (vHTS)
outcomes: if the model ranks candidates and a budget of experiments is spent
on the top-ranked ones, how many true tail materials are found?

Metrics: extrapolative precision, tail recall and tail MAE at a given budget
(:func:`discovery_metrics`, ported from the paper's adaptive discovery
analysis), plus precision/recall/enrichment as a function of budget
(:func:`budget_curve`).
"""

from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ood import PredGroup, _as_combined
from .predictions import SCREEN_PHI


def discovery_metrics(group: PredGroup, rel_threshold: float = 0.5,
                      budget_ratio: float = 1.0, discover_mode: str = "auto") -> dict:
    """Adaptive discovery metrics for one method.

    Samples with relevance > ``rel_threshold`` are screening targets; the
    head median splits them into a high tail (searched by descending
    prediction) and a low tail (ascending).  ``discover_mode`` = 'auto'
    activates each direction when it holds > 5% of relevant samples.
    ``budget_ratio`` scales the number of candidates tested relative to the
    true tail size.

    Returns a dict with Discovery Mode, Tail Size (GT), Budget Ratio,
    Extrap. Precision, Tail Recall, Tail MAE and Breakdown.
    """
    pred = _as_combined(group)
    targets = np.ravel(pred.targets)
    preds = np.ravel(pred.preds)
    if pred.relevances is None:
        return {"Error": "No relevance scores found in predictions."}
    relevances = np.ravel(pred.relevances)

    is_relevant = relevances > rel_threshold
    if not np.any(is_relevant):
        return {"Error": f"No samples found with relevance > {rel_threshold}"}

    head_mask = ~is_relevant
    head_median = np.median(targets[head_mask]) if np.any(head_mask) else np.median(targets)
    gt_high_mask = is_relevant & (targets >= head_median)
    gt_low_mask = is_relevant & (targets < head_median)
    n_high, n_low = int(gt_high_mask.sum()), int(gt_low_mask.sum())

    mode = discover_mode.lower()
    total_relevant = is_relevant.sum()
    if mode == "auto":
        has_high = n_high > 0.05 * total_relevant
        has_low = n_low > 0.05 * total_relevant
        mode = "both" if (has_high and has_low) else ("low" if has_low else "high")

    all_indices = np.arange(len(targets))
    candidate_indices: set = set()
    target_gt_indices: set = set()

    def select(gt_mask, ascending):
        n_target = gt_mask.sum()
        if n_target == 0:
            return 0
        k = min(int(np.ceil(n_target * budget_ratio)), len(targets))
        selection = np.argsort(preds)[:k] if ascending else np.argsort(preds)[-k:]
        candidate_indices.update(selection)
        target_gt_indices.update(all_indices[gt_mask])
        return k

    k_used = 0
    if mode in ("both", "high"):
        k_used += select(gt_high_mask, ascending=False)
    if mode in ("both", "low"):
        k_used += select(gt_low_mask, ascending=True)

    if not target_gt_indices:
        return {"Error": f'No relevant samples found in mode "{mode}"'}

    hits = len(candidate_indices & target_gt_indices)
    precision = hits / k_used if k_used > 0 else 0.0
    recall = hits / len(target_gt_indices)
    tail_mask = (gt_high_mask | gt_low_mask) if mode == "both" else (
        gt_high_mask if mode == "high" else gt_low_mask)
    tail_mae = float(np.mean(np.abs(targets[tail_mask] - preds[tail_mask])))

    return {
        "Discovery Mode": mode,
        "Tail Size (GT)": len(target_gt_indices),
        "Budget Ratio": f"{budget_ratio}x ({k_used} tests)",
        "Extrap. Precision": round(precision, 4),
        "Tail Recall": round(recall, 4),
        "Tail MAE": round(tail_mae, 4),
        "Breakdown": f"High={n_high}, Low={n_low}",
    }


def budget_curve(group: PredGroup, rel_threshold: float = 0.5,
                 budget_ratios: Optional[Sequence[float]] = None,
                 discover_mode: str = "auto") -> pd.DataFrame:
    """Precision, recall and enrichment factor as a function of budget.

    Enrichment factor = precision / base rate (fraction of relevant samples
    in the test set); EF > 1 means the model beats random selection.
    """
    if budget_ratios is None:
        budget_ratios = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    pred = _as_combined(group)
    base_rate = float((np.ravel(pred.relevances) > rel_threshold).mean())

    rows = []
    for ratio in budget_ratios:
        m = discovery_metrics(group, rel_threshold=rel_threshold,
                              budget_ratio=ratio, discover_mode=discover_mode)
        if "Error" in m:
            continue
        precision = m["Extrap. Precision"]
        rows.append({
            "budget_ratio": ratio,
            "precision": precision,
            "recall": m["Tail Recall"],
            "enrichment": precision / base_rate if base_rate > 0 else np.nan,
            "tail_mae": m["Tail MAE"],
        })
    return pd.DataFrame(rows)


def screening_table(groups: Dict[str, PredGroup], rel_threshold: float = 0.5,
                    budget_ratio: float = 1.0) -> pd.DataFrame:
    """Discovery metrics for several methods, one row per method label."""
    rows = []
    for label, group in groups.items():
        m = discovery_metrics(group, rel_threshold=rel_threshold, budget_ratio=budget_ratio)
        m["method"] = label
        rows.append(m)
    return pd.DataFrame(rows).set_index("method")


def plot_budget_curves(groups: Dict[str, PredGroup], rel_threshold: float = 0.5,
                       metric: str = "recall", ax=None,
                       budget_ratios: Optional[Sequence[float]] = None):
    """Screening metric vs budget per method (paper Fig. 4 style; legend is
    placed outside the axes to avoid overlap — Referee 1)."""
    colors = ["#4d4d4d", "#2166ac", "#b2182b", "#35978f", "#ff7f0e"]
    markers = ["s", "o", "^", "v", "d"]
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 2.8), layout="compressed")
    for i, (label, group) in enumerate(groups.items()):
        df = budget_curve(group, rel_threshold=rel_threshold, budget_ratios=budget_ratios)
        if df.empty:
            continue
        c = i % len(colors)
        ax.plot(df["budget_ratio"], df[metric], marker=markers[c % len(markers)], ms=5,
                color=colors[c], lw=1.5, alpha=0.85, label=label)
    ax.set_xlabel("screening budget (x tail size)", fontsize=10)
    ax.set_ylabel({"recall": "Tail recall", "precision": "Extrap. precision",
                   "enrichment": "Enrichment factor",
                   "tail_mae": "Tail MAE"}.get(metric, metric), fontsize=10)
    if metric == "enrichment":
        ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
              handletextpad=0.5, borderpad=0.2, labelspacing=0.6)
    return ax
