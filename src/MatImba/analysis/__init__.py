"""MatImba analysis layer: describe datasets, diagnose OOD error, evaluate
screening performance, and test whether tail error is coupled to awareness.

Model-agnostic: everything here consumes arrays, ``*_test_predictions.csv``
and ``*_val_log.csv`` files (see :mod:`MatImba.analysis.collect` for
discovery).  Checkpoint re-evaluation, the only model-coupled part, lives in
:mod:`MatImba.analysis.checkpoint` and imports torch lazily.

Typical use::

    from MatImba.analysis import collect_experiments, DatasetProfile
    from MatImba.analysis import load_trajectories, coupling_table

    index, preds = collect_experiments("experiments/final")
    coupling = coupling_table(load_trajectories(index))

or from the command line::

    matimba-analyse --experiments experiments/final --out figs/
"""

from .awareness import (
    alpha_sera_correlation,
    classify_regime,
    coupling_table,
    coupling_vs_imbalance,
    final_coupling_table,
    load_trajectories,
    plot_alpha_sera,
    plot_trajectory_phase,
)
from .collect import (
    collect_experiments,
    discover_runs,
    filter_bad_runs,
    load_predictions,
    metrics_table,
)
from .dataset_profile import DatasetProfile, compare_profiles, profiles_table
from .ood import (
    compare_binned_mae,
    compare_ser_curves,
    composite_parity,
    error_vs_density,
    head_tail_table,
    plot_split_parity,
    split_data,
    summary_plot,
)
from .predictions import (
    SCREEN_PHI,
    SERA_T0,
    TAIL_PHI,
    EnsemblePrediction,
    PredictionSet,
)
from .report import aggregate, format_pm, formatted_table, run_report, use_matimba_style
from .screening import budget_curve, discovery_metrics, plot_budget_curves, screening_table

__all__ = [
    # predictions
    "PredictionSet", "EnsemblePrediction", "SERA_T0", "TAIL_PHI", "SCREEN_PHI",
    # collect
    "collect_experiments", "discover_runs", "load_predictions", "metrics_table",
    "filter_bad_runs",
    # dataset profile
    "DatasetProfile", "compare_profiles", "profiles_table",
    # ood
    "split_data", "plot_split_parity", "composite_parity", "compare_binned_mae",
    "compare_ser_curves", "error_vs_density", "head_tail_table", "summary_plot",
    # screening
    "discovery_metrics", "budget_curve", "screening_table", "plot_budget_curves",
    # awareness
    "alpha_sera_correlation", "classify_regime", "coupling_table",
    "final_coupling_table", "coupling_vs_imbalance", "load_trajectories",
    "plot_alpha_sera", "plot_trajectory_phase",
    # report
    "aggregate", "format_pm", "formatted_table", "run_report", "use_matimba_style",
]


def evaluate_ckpt(*args, **kwargs):
    """Lazy wrapper for :func:`MatImba.analysis.checkpoint.evaluate_ckpt`
    (keeps torch/model imports out of the base analysis import)."""
    from .checkpoint import evaluate_ckpt as _evaluate_ckpt
    return _evaluate_ckpt(*args, **kwargs)
