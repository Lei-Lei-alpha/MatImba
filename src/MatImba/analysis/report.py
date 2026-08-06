"""Reporting: aggregated metric tables, formatting hygiene and the
``matimba-analyse`` CLI.

Statistics conventions (fixing the referee-flagged issues):

- All mean +/- std values use the **sample** standard deviation (ddof=1,
  Bessel-corrected) — appropriate for n=5 folds.
- :func:`format_pm` rounds the uncertainty to one significant figure and the
  mean to the matching decimal place (``4.108 +/- 1.739`` -> ``4.1 +/- 1.7``),
  switching to scientific notation for very small values.
- LaTeX tables mark the best method per dataset/metric in bold.
"""

import argparse
import logging
import math
import os
from importlib import resources
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .awareness import coupling_table, coupling_vs_imbalance, load_trajectories
from .collect import collect_experiments, metrics_table
from .ood import summary_plot
from .screening import screening_table

logger = logging.getLogger(__name__)

METRICS = ["mae", "r2", "sera", "alpha", "tail_mae", "head_mae"]
LOWER_IS_BETTER = {"mae": True, "r2": False, "sera": True, "alpha": False,
                   "tail_mae": True, "head_mae": True}


def use_matimba_style():
    """Activates the packaged ggplot_bw matplotlib style (portable
    replacement for the old hard-coded style path)."""
    try:
        style_path = resources.files("MatImba.utils").joinpath("ggplot_bw.mplstyle")
        with resources.as_file(style_path) as p:
            plt.style.use(str(p))
    except Exception as e:  # style is cosmetic; never fail an analysis on it
        logger.warning("Could not load packaged mplstyle (%s); using defaults.", e)


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def _std_digits(std: float) -> int:
    """Decimal places so the uncertainty keeps one significant figure — two
    when its leading digit is 1 (standard convention, e.g. 1.739 -> 1.7)."""
    exp = int(math.floor(math.log10(std)))
    leading = int(std / 10.0 ** exp)
    return -exp + (1 if leading == 1 else 0)


def format_pm(mean: float, std: float, sci_threshold: float = 1e-2) -> str:
    """``mean +/- std`` with the uncertainty rounded to one significant
    figure (two when its leading digit is 1) and the mean to the matching
    decimal place: ``4.108 +/- 1.739`` -> ``4.1 ± 1.7``.

    Values whose magnitude is below ``sci_threshold`` are rendered in
    scientific notation with a shared exponent.
    """
    if not (np.isfinite(mean) and np.isfinite(std)):
        return "--"
    if std == 0:
        return f"{mean:.3g} ± 0"

    if max(abs(mean), std) < sci_threshold:
        exp = int(math.floor(math.log10(max(abs(mean), std))))
        scale = 10.0 ** exp
        m, s = mean / scale, std / scale
        digits = max(0, _std_digits(s)) if s > 0 else 1
        return f"({m:.{digits}f} ± {s:.{digits}f})e{exp:+03d}"

    digits = _std_digits(std)
    std_r = round(std, digits)
    # Rounding can bump the magnitude (0.096 -> 0.1): recompute the digits
    if std_r > 0:
        digits = _std_digits(std_r)
    # digits < 0 (std >= 10): round the mean to the same power of ten
    mean_r = round(mean, digits)
    digits = max(digits, 0)
    return f"{mean_r:.{digits}f} ± {std_r:.{digits}f}"


def aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std (ddof=1) per dataset x method for every metric in
    a per-run table from :func:`~MatImba.analysis.collect.metrics_table`."""
    metrics = [m for m in METRICS if m in per_run.columns]
    agg = per_run.groupby(["dataset", "method"])[metrics].agg(
        ["mean", lambda s: s.std(ddof=1), "count"]
    )
    agg.columns = [
        f"{metric}_{'std' if name == '<lambda_0>' else name}"
        for metric, name in agg.columns
    ]
    return agg.reset_index()


def formatted_table(agg: pd.DataFrame, metrics: Optional[List[str]] = None
                    ) -> pd.DataFrame:
    """Human-readable table with :func:`format_pm` strings, one row per
    dataset x method."""
    metrics = metrics or [m for m in METRICS if f"{m}_mean" in agg.columns]
    out = agg[["dataset", "method"]].copy()
    for m in metrics:
        out[m] = [
            format_pm(row[f"{m}_mean"], row[f"{m}_std"]) for _, row in agg.iterrows()
        ]
    return out


def to_latex_bold_best(agg: pd.DataFrame, metric: str) -> str:
    """LaTeX table of one metric (datasets x methods), best method per
    dataset in bold (Referee 1 request)."""
    lower_better = LOWER_IS_BETTER.get(metric, True)
    datasets = agg["dataset"].unique()
    methods = agg["method"].unique()

    lines = [
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(m.replace("_", "\\_") for m in methods) + " \\\\",
        "\\midrule",
    ]
    for ds in datasets:
        sub = agg[agg["dataset"] == ds].set_index("method")
        means = sub[f"{metric}_mean"]
        best = means.idxmin() if lower_better else means.idxmax()
        cells = []
        for m in methods:
            if m not in sub.index:
                cells.append("--")
                continue
            cell = format_pm(sub.loc[m, f"{metric}_mean"], sub.loc[m, f"{metric}_std"])
            cell = cell.replace("±", "$\\pm$")
            cells.append(f"\\textbf{{{cell}}}" if m == best else cell)
        lines.append(ds.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def run_report(experiments_dir: str, outdir: str,
               datasets: Optional[List[str]] = None,
               methods: Optional[List[str]] = None,
               make_figures: bool = True) -> Dict[str, pd.DataFrame]:
    """Full standard analysis of one experiment set.

    Writes to ``outdir``: per-run and aggregated metric CSVs, formatted
    markdown/LaTeX tables, alpha-SERA coupling table, screening table, and
    (optionally) a summary figure per dataset.  Returns the tables as a dict.
    """
    os.makedirs(outdir, exist_ok=True)
    use_matimba_style()

    index, predictions = collect_experiments(experiments_dir, datasets, methods)
    if index.empty:
        raise SystemExit(f"No prediction CSVs found under {experiments_dir}")
    logger.info("Found %d runs: %s", len(index),
                index.groupby(["dataset", "method"]).size().to_dict())

    per_run = metrics_table(predictions)
    agg = aggregate(per_run)
    pretty = formatted_table(agg)

    per_run.to_csv(os.path.join(outdir, "metrics_per_run.csv"), index=False)
    agg.to_csv(os.path.join(outdir, "metrics_aggregated.csv"), index=False)
    pretty.to_csv(os.path.join(outdir, "metrics_formatted.csv"), index=False)
    with open(os.path.join(outdir, "metrics_formatted.md"), "w") as f:
        f.write(pretty.to_markdown(index=False))
    for metric in ("mae", "sera", "alpha"):
        if f"{metric}_mean" in agg.columns:
            with open(os.path.join(outdir, f"table_{metric}.tex"), "w") as f:
                f.write(to_latex_bold_best(agg, metric))

    tables = {"per_run": per_run, "aggregated": agg, "formatted": pretty}

    # Alpha-SERA coupling from training trajectories
    trajectories = load_trajectories(index)
    if not trajectories.empty:
        coupling = coupling_table(trajectories)
        coupling.to_csv(os.path.join(outdir, "alpha_sera_coupling.csv"), index=False)
        tables["coupling"] = coupling
        tables["trajectories"] = trajectories

    # Screening metrics per dataset
    screen_rows = []
    for dataset, by_method in predictions.items():
        t = screening_table(by_method)
        t.insert(0, "dataset", dataset)
        screen_rows.append(t.reset_index())
    if screen_rows:
        screening = pd.concat(screen_rows, ignore_index=True)
        screening.to_csv(os.path.join(outdir, "screening_metrics.csv"), index=False)
        tables["screening"] = screening

    if make_figures:
        from .awareness import plot_alpha_sera
        for dataset, by_method in predictions.items():
            fig = summary_plot(by_method, target_name=dataset)
            fig.savefig(os.path.join(outdir, f"{dataset}_summary.jpg"), dpi=600,
                        bbox_inches="tight")
            plt.close(fig)
            if not trajectories.empty and (trajectories["dataset"] == dataset).any():
                ax = plot_alpha_sera(trajectories, dataset)
                ax.figure.savefig(os.path.join(outdir, f"{dataset}_alpha_sera.jpg"),
                                  dpi=600, bbox_inches="tight")
                plt.close(ax.figure)

    logger.info("Report written to %s", outdir)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="matimba-analyse",
        description="Analyse MatImba experiment outputs: metric tables "
                    "(ddof=1, significant-figure formatting), alpha-SERA "
                    "coupling diagnosis, screening metrics and figures.",
    )
    parser.add_argument("--experiments", required=True,
                        help="Experiment set directory, e.g. experiments/final")
    parser.add_argument("--out", default="analysis_out",
                        help="Output directory (default: analysis_out)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict to these datasets (default: all found)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Restrict to these methods (default: all found)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Tables only, skip figure generation")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_report(args.experiments, args.out, datasets=args.datasets,
               methods=args.methods, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
