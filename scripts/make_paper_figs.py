"""Regenerates every figure and table in the MatImba paper from archived data.

Deterministic, top-to-bottom script mirroring Final_figs.ipynb (each section
below is a verbatim port of the notebook cells noted in its docstring/comments):

    python scripts/make_paper_figs.py --out figs/paper

Produces
    fig1_distributions.jpg / table1_imbalance.csv|tex
        dataset imbalance profiles, correlation heatmap, hand-annotated PCA
        biplot (Fig. 1, Table 1)
    {dataset}_unified_analysis.jpg / {dataset}_matfold_unified_analysis.jpg,
    tables/combined_flatness_table.txt / matfold_combined_flatness_table.txt
        6-panel training-dynamics analysis + LaTeX flatness table (Fig. 2)
    fig3_{expt}_{dataset}_fold{N}_aggregated.jpg
        per-fold split parity + SER + binned MAE composites (Fig. 3)
    fig4_discov_performance_{expt}.jpg, tables/discov_metrics_{dataset}_{expt}.csv
        4x3 OOD discovery grid at rel_threshold=0.8 (Fig. 4)
    fig5_quantitative_alpha_sera_sensitivity.jpg
        CV+MatFold SERA/alpha PCA-of-imbalance sensitivity grid (Fig. 5)
    fig_methods_profile.jpg
        runtime/GPU-memory boxplots per dataset x method
    alpha_sera_transfer_analysis.jpg
        12-panel Alpha-SERA Transfer Mechanism figure (MatFold)
    tables/alpha_sera_correlation.csv, alpha_sera_correlation_{cv,matfold}.txt,
    tables/coupling_vs_imbalance.csv, coupling_vs_imbalance_table.txt
        SI3/SI4/SI5: alpha-SERA Pearson/Spearman/Kendall per dataset x method,
        and coupling strength vs imbalance descriptors (needs both experiment sets)
    tables/: metrics_{expt}.csv|md, table_{metric}_{expt}.tex (bold-best)

Every statistic uses ddof=1 and significant-figure formatting via
MatImba.analysis.report.
"""

import argparse
import glob
import logging
import math
import os
import re
import string
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import palettable.colorbrewer as ptcb
import palettable.colorbrewer.diverging as cbd
import seaborn as sns
import torch
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyArrowPatch
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde, kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from adjustText import adjust_text
except ImportError:  # optional dependency, only used for the PCA label layout
    adjust_text = None

from MatImba.analysis import (
    PredictionSet,
    aggregate,
    discover_runs,
    discovery_metrics,
    filter_bad_runs,
    formatted_table,
    load_predictions,
    metrics_table,
    summary_plot,
    use_matimba_style,
)
from MatImba.analysis.report import to_latex_bold_best
from MatImba.dataset import calc_comprehensive_imbalance
from MatImba.utils.losses import calc_alpha, calc_sera

logger = logging.getLogger("make_paper_figs")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELLED_DATASETS = ["log_kvrh", "log_gvrh", "perovskites", "phonons"]

# Fig 1 dataset roster and classification exclusion, ported verbatim from
# Final_figs.ipynb cells 9-10.
FIG1_DATASETS = [
    "matbench_dielectric", "matbench_jdft2d", "matbench_steels",
    "matbench_expt_gap", "matbench_phonons", "matbench_log_gvrh",
    "matbench_log_kvrh", "matbench_glass", "matbench_expt_is_metal",
    "matbench_perovskites", "matbench_mp_e_form", "matbench_mp_gap",
    "matbench_mp_is_metal",
]
FIG1_CLASSIFICATION = {"matbench_expt_is_metal", "matbench_mp_is_metal", "matbench_glass"}
# Hand-tuned label offsets for the 8 background PCA points, in scatter order
# after the two highlighted datasets (steels, mp_e_form) are pulled out —
# ported verbatim from Final_figs.ipynb cell 13.
FIG1_PCA_OPTIMISED_COORDS = (
    (3.22, -0.72), (3.0, 0.2), (2.4, 0.6), (-0.4, 0.8),
    (-1.05, -0.8), (-1.0, 0.5), (-0.9, -0.5), (1.55, 1.0),
)


def add_panel_label(ax, index, loc=(-0.1, 1.1), alphabet=string.ascii_lowercase):
    ax.text(*loc, f"{alphabet[index]}", transform=ax.transAxes,
            fontsize=10, va="top", ha="right")


# ----------------------------------------------------------------------
# Fig 1 + Table 1: dataset imbalance profiles
#
# Ported verbatim (metrics, histogram/KDE panel, correlation heatmap, hand-
# annotated PCA biplot) from Final_figs.ipynb cells 9-13, so the combined
# panel a/b/c figure matches matbench_dil.jpg exactly rather than a generic
# per-dataset grid.
# ----------------------------------------------------------------------
def _plot_distribution_with_metrics(data, ax, label=None, show_kde=True):
    raw_data = np.asarray(data).flatten()
    clean_data = pd.to_numeric(raw_data, errors="coerce")
    data = clean_data[~np.isnan(clean_data)]
    if len(data) == 0:
        ax.text(0.5, 0.5, "No Valid Data", ha="center", va="center")
        return

    ax.hist(data, bins="fd", density=False, color="#4393c3", alpha=0.65,
            edgecolor="None", linewidth=0.5)

    if show_kde and len(data) > 1 and np.var(data) > 0:
        try:
            density = gaussian_kde(data)
            xs = np.linspace(min(data), max(data), 200)
            ax.plot(xs, density(xs), color="#1f4e79", lw=1.5, label="KDE")
        except Exception:
            pass

    metrics = calc_comprehensive_imbalance(data)
    stats_text = (f"{label}\n"
                  f"$h$: {metrics['DIL']: .3f}\n"
                  f"$W_1$: {metrics['Wasserstein']: .3f}")
    at = AnchoredText(stats_text, prop=dict(size=8, family="monospace"),
                      frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.1,rounding_size=0.2")
    at.patch.set_alpha(0.25)
    at.patch.set_edgecolor("#d3d3d3")
    ax.add_artist(at)


def make_dataset_profiles(data_dir: str, outdir: str):
    labels, regressions, data_by_label = [], [], {}
    for dataset in FIG1_DATASETS:
        if dataset in FIG1_CLASSIFICATION:
            continue
        label = dataset.split("_", 1)[-1]
        path = os.path.join(data_dir, f"{label}.csv")
        if not os.path.exists(path):
            logger.warning("Missing %s — skipping from Fig 1.", path)
            continue
        df = pd.read_csv(path)
        target_col = list(df)[-1]
        raw = np.asarray(df[[target_col]]).flatten()
        clean = pd.to_numeric(raw, errors="coerce")
        data = clean[~np.isnan(clean)]
        labels.append(label)
        regressions.append(dataset)
        data_by_label[label] = data

    if not labels:
        logger.warning("No dataset CSVs found in %s — skipping Fig 1.", data_dir)
        return None

    rows = []
    for label in labels:
        data = data_by_label[label]
        metrics = calc_comprehensive_imbalance(data)
        metrics.update({"dataset": label, "size": len(data)})
        rows.append(metrics)
    result_df = pd.DataFrame(rows)[
        ["dataset", "size", "DIL", "Gini", "KL_Div", "Wasserstein"]]

    fig = plt.figure(figsize=(8.8, 6.2), layout="compressed")
    subfigs = fig.subfigures(1, 2, width_ratios=[3.2, 2])

    ncols_hist = 2
    nrows_hist = int(np.ceil(len(labels) / ncols_hist))
    gs_left = subfigs[0].add_gridspec(nrows_hist, ncols_hist)
    hst_axes = []
    for col in range(ncols_hist):
        for row in range(nrows_hist):
            hst_axes.append(subfigs[0].add_subplot(gs_left[row, col]))
    for label, ax in zip(labels, hst_axes):
        _plot_distribution_with_metrics(data_by_label[label], ax=ax, label=label)
    for ax in hst_axes[len(labels):]:
        ax.axis("off")
    subfigs[0].supxlabel("Target values", fontsize=10)
    subfigs[0].supylabel("Counts", fontsize=10)
    hst_axes[0].set_title("a", loc="left", fontsize=10)

    gs_right = subfigs[1].add_gridspec(2, 1)
    df = result_df.rename(columns={"DIL": r"$h$", "Gini": r"$G$",
                                   "KL_Div": r"$D_{KL}$", "Wasserstein": r"$W_1$"})
    metrics_cols = [r"$h$", r"$G$", r"$D_{KL}$", r"$W_1$"]

    cor_ax = subfigs[1].add_subplot(gs_right[0, 0])
    corr_matrix = df[metrics_cols].corr(method="pearson")
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=0.6, vmax=1, square=True, linewidths=0.5, cbar=True,
                ax=cor_ax, annot_kws={"size": 9},
                cbar_kws={"shrink": 1.0, "pad": -0.09, "aspect": 40,
                          "location": "left"})
    cor_ax.set_title("b", loc="left", fontsize=10)

    pca_ax = subfigs[1].add_subplot(gs_right[1, 0])
    X = df[metrics_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    loadings = pca.components_.T

    texts, x_coords, y_coords = [], [], []
    for i, row in df.iterrows():
        name = row["dataset"]
        x, y = coords[i, 0], coords[i, 1]
        if name == "steels":
            pca_ax.scatter(x, y, s=120, c="#92c5de", edgecolors="#4d4d4d", zorder=10)
            pca_ax.annotate(name, xy=(x, y), xytext=(x + 1.28, y + 1.3), fontsize=9,
                            va="center", ha="center",
                            arrowprops=dict(arrowstyle="-|>",
                                            connectionstyle="angle,angleA=0,angleB=90,rad=15"),
                            bbox=dict(boxstyle="round", fc="#dceaf6", ec="#000000", alpha=0.55))
            pca_ax.plot([x, x], [0, y], "k:", alpha=0.3)
        elif name == "mp_e_form":
            pca_ax.scatter(x, y, s=120, c="#f4a582", edgecolors="#4d4d4d", zorder=10)
            pca_ax.annotate(name, xy=(x, y), xytext=(x + 1.25, y - 1.6), fontsize=9,
                            va="center", ha="center",
                            bbox=dict(boxstyle="round", fc="#fcbba1", ec="#000000", alpha=0.55),
                            arrowprops=dict(arrowstyle="-|>",
                                            connectionstyle="angle,angleA=0,angleB=90,rad=15"))
            pca_ax.plot([x, x], [0, y], "k:", alpha=0.3)
        else:
            pca_ax.scatter(x, y, s=60, c="silver", edgecolors="tab:gray")
            t = pca_ax.text(x, y, name, fontsize=8, alpha=0.5)
            texts.append(t)
            x_coords.append(x)
            y_coords.append(y)

    for t, new_coord, orig_x, orig_y in zip(texts, FIG1_PCA_OPTIMISED_COORDS,
                                             x_coords, y_coords):
        new_x, new_y = new_coord
        t.set_horizontalalignment("right" if new_x < orig_x else "left")
        t.set_verticalalignment("center")
        t.set_position((new_x, new_y))
        pca_ax.annotate("", xy=(orig_x, orig_y), xytext=(new_x, new_y),
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
                        va="center", ha="center", zorder=1)

    pca_ax.axhline(0, color="k", lw=1, ls="--")
    pca_ax.axvline(0, color="k", lw=1, ls="--")
    pca_ax.set_xlabel(
        f"Statistical Imbalance Magnitude\n PC1: ({pca.explained_variance_ratio_[0]:.1%} var)",
        fontsize=10)
    pca_ax.set_ylabel(
        f"PC2: ({pca.explained_variance_ratio_[1]:.1%} var)\n Geometric Transport Cost",
        fontsize=10)
    pca_ax.grid(True, linestyle=":", alpha=0.4)

    scale = 3.5
    for i, metric in enumerate(metrics_cols):
        alpha = 1.0 if metric in [r"$h$", r"$W_1$"] else 0.75
        weight = "bold" if metric in [r"$h$", r"$W_1$"] else "normal"
        if metric == r"$W_1$":
            color = "#2166ac"
            tx, ty = loadings[i, 0] * scale * 1.25, loadings[i, 1] * scale * 1.1
        elif metric == r"$h$":
            color = "#b2182b"
            tx, ty = loadings[i, 0] * scale * 1.25, loadings[i, 1] * scale * 1.1
        else:
            color = "k"
            tx, ty = loadings[i, 0] * scale * 1.25, loadings[i, 1] * scale * 1.35
        pca_ax.text(tx, ty, metric, color=color, ha="center", va="center",
                    weight=weight, fontsize=10,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.2),
                    zorder=5)
        pca_ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                     color=color, alpha=alpha, head_width=0.12, linewidth=1.5, zorder=5)
    pca_ax.set_title("c", loc="left", fontsize=10)

    fig.savefig(os.path.join(outdir, "fig1_distributions.jpg"), dpi=600)
    plt.close(fig)

    # Table 1 (notebook-native columns)
    result_df.to_csv(os.path.join(outdir, "table1_imbalance.csv"), index=False)
    with open(os.path.join(outdir, "table1_imbalance.tex"), "w") as f:
        f.write(result_df.set_index("dataset").round(3).to_latex())
    logger.info("PCA explained variance: %s", np.round(pca.explained_variance_ratio_, 3))

    # profiles_table-shaped view (h/Gini/D_KL/W1, indexed by dataset) for
    # MatImba.analysis.coupling_vs_imbalance downstream.
    table = result_df.set_index("dataset").rename(
        columns={"DIL": "h", "KL_Div": "D_KL", "Wasserstein": "W1"})[
        ["h", "Gini", "D_KL", "W1"]]
    return {"result_df": result_df, "table": table,
            "pca": {"explained_variance_ratio": pca.explained_variance_ratio_}}


# ----------------------------------------------------------------------
# Fig 2: training-trajectory dynamics (6-panel unified analysis + flatness
# tables), ported verbatim from Final_figs.ipynb cell 15 (functions) and
# cells 17/19 (CV / MatFold invocations with per-dataset filter overrides).
# ----------------------------------------------------------------------
def filter_outlier_runs(dfs, method_label, fold_id, metric="mae",
                        explode_ratio=100.0, conv_ratio=5.25, warmup_epochs=20,
                        explosion_warmup=35, sera_explode_ratio=250.0):
    """Robustly filters runs ON A PER-FOLD BASIS using Relative metrics.

    The explosion checks start at ``explosion_warmup`` (later than
    ``warmup_epochs``) so ordinary early-training transients don't disqualify
    otherwise-healthy runs. SERA uses its own multiplier (250x vs 100x for
    MAE) because SERA is intrinsically far more volatile."""
    total_runs = len(dfs)
    if total_runs == 0:
        return []

    valid_final_scores, valid_final_seras = [], []
    for df in dfs:
        if metric in df.columns and np.isfinite(df[metric]).all():
            valid_final_scores.append(df[metric].iloc[-5:].mean())
        if "sera" in df.columns and np.isfinite(df["sera"]).all():
            valid_final_seras.append(df["sera"].iloc[-5:].mean())

    if not valid_final_scores:
        return []

    median_final_metric = np.median(valid_final_scores)
    best_final_metric = np.min(valid_final_scores)
    median_final_sera = np.median(valid_final_seras) if valid_final_seras else None

    surviving = []
    for df in dfs:
        if not np.isfinite(df[metric]).all():
            continue
        if "r2_score" in df.columns:
            final_r2 = df["r2_score"].iloc[-5:].mean()
            if final_r2 < 0.65:
                continue
        max_spike = (df[metric].iloc[explosion_warmup:].max()
                     if len(df) > explosion_warmup else df[metric].max())
        if max_spike > median_final_metric * explode_ratio:
            continue
        if median_final_sera is not None and "sera" in df.columns:
            max_sera = (df["sera"].iloc[explosion_warmup:].max()
                        if len(df) > explosion_warmup else df["sera"].max())
            if max_sera > median_final_sera * sera_explode_ratio:
                continue
        final_val = df[metric].iloc[-5:].mean()
        if final_val > best_final_metric * conv_ratio:
            continue
        surviving.append(df)

    return surviving


def load_all_folds(method_label, search_pattern, metric, warmup_epochs=20,
                   filter_runs=False, **filter_kwargs):
    """Loads all runs and applies the filter independently to each fold."""
    files = glob.glob(search_pattern)
    if not files:
        return []

    fold_files = defaultdict(list)
    for f in files:
        filename = os.path.basename(f)
        try:
            fold_id = "fold_" + filename.split("fold_")[1].split("_")[0]
            fold_files[fold_id].append(f)
        except Exception:
            fold_files["unknown"].append(f)

    all_valid_dfs = []
    for fold_id, f_list in fold_files.items():
        dfs = []
        for f in f_list:
            try:
                df = pd.read_csv(f)
                df["run_id"] = os.path.basename(f)
                if len(df) > warmup_epochs + 5:
                    dfs.append(df)
            except Exception:
                pass
        if filter_runs:
            valid_dfs = filter_outlier_runs(
                dfs, method_label, fold_id, metric=metric, warmup_epochs=warmup_epochs,
                **filter_kwargs)
        else:
            valid_dfs = dfs
        all_valid_dfs.extend(valid_dfs)

    return all_valid_dfs


def plot_universal_dynamics_on_axes(axes_row, methods_config, metric="sera",
                                    filter_runs=False, **filter_kwargs):
    """Plots Loss, Awareness, and Phase Space on the top row."""
    ax_loss, ax_aware, ax_phase = axes_row
    DISCARD_EPOCHS = 20
    SMOOTH_SPAN = 20

    for config in methods_config:
        search_pattern = config["search_pattern"]
        base_label = config["label"]
        color = config["color"]

        dfs = load_all_folds(base_label, search_pattern, metric=metric,
                             warmup_epochs=DISCARD_EPOCHS, filter_runs=True,
                             **filter_kwargs)
        if not dfs:
            continue

        final_label = f"{base_label} (n={len(dfs)})"

        processed_dfs = []
        for df in dfs:
            sub_df = df.sort_values("epoch").copy()
            sub_df = sub_df[sub_df["epoch"] >= DISCARD_EPOCHS].copy()
            sub_df["metric_smooth"] = sub_df[metric].ewm(span=SMOOTH_SPAN, adjust=False).mean()
            sub_df["aware_smooth"] = sub_df["awareness"].ewm(span=SMOOTH_SPAN, adjust=False).mean()
            processed_dfs.append(sub_df)

        clean_df = pd.concat(processed_dfs)
        cols_to_agg = ["metric_smooth", "aware_smooth"]

        agg_df = clean_df.groupby("epoch")[cols_to_agg].agg(["mean", "std"]).reset_index()
        agg_df.columns = ["_".join(col).strip() if col[1] else col[0]
                          for col in agg_df.columns.values]

        x_epochs = agg_df["epoch"]

        y_mean = agg_df["metric_smooth_mean"]
        y_std = agg_df["metric_smooth_std"]
        ax_loss.plot(x_epochs, y_mean, color=color, label=final_label, linewidth=1.5)
        ax_loss.fill_between(x_epochs, y_mean - y_std, y_mean + y_std, color=color,
                             alpha=0.15, linewidth=0)

        y_mean_aw = agg_df["aware_smooth_mean"]
        y_std_aw = agg_df["aware_smooth_std"]
        ax_aware.plot(x_epochs, y_mean_aw, color=color, label=final_label, linewidth=1.5)
        ax_aware.fill_between(x_epochs, y_mean_aw - y_std_aw, y_mean_aw + y_std_aw,
                              color=color, alpha=0.15, linewidth=0)

        ax_phase.plot(y_mean_aw, y_mean, color=color, linewidth=1.5, label=final_label,
                     alpha=0.8, zorder=5)

        points_idx = np.linspace(0, len(agg_df) - 1, 5, dtype=int)
        for idx in points_idx:
            x_pt = y_mean_aw.iloc[idx]
            y_pt = y_mean.iloc[idx]
            x_err = y_std_aw.iloc[idx]
            y_err = y_std.iloc[idx]
            ax_phase.errorbar(x_pt, y_pt, xerr=x_err, yerr=y_err, color=color,
                              ecolor=color, capsize=0, alpha=0.6, fmt="none", zorder=10)

        arrow_indices = np.linspace(0, len(agg_df) - 1, 5, dtype=int)
        points = np.array([y_mean_aw.values, y_mean.values]).T
        for i in range(len(arrow_indices) - 1):
            idx_start, idx_end = arrow_indices[i], arrow_indices[i + 1]
            start_pt = points[idx_start]
            end_pt = points[int(idx_start + (idx_end - idx_start) * 0.5)]
            arrow = FancyArrowPatch(
                posA=start_pt, posB=end_pt,
                arrowstyle="simple,head_length=6,head_width=6,tail_width=2",
                mutation_scale=0.7, facecolor=color, edgecolor="black",
                linewidth=0.75, alpha=0.9, zorder=15)
            ax_phase.add_patch(arrow)

        ax_phase.scatter(points[-1, 0], points[-1, 1], color=color, marker="o", s=60,
                         edgecolors="k", zorder=25)

    y_axis_name = metric.upper()
    ax_loss.set_ylabel(f"Validation {y_axis_name}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend(fontsize=9)

    ax_aware.set_ylabel(r"$\alpha$")
    ax_aware.set_xlabel("Epoch")
    ax_aware.legend(fontsize=9)

    ax_phase.set_xlabel(r"$\alpha$")
    ax_phase.set_ylabel(y_axis_name)
    ax_phase.legend(fontsize=9)

    if metric == "sera":
        ax_loss.set_yscale("log")
        ax_phase.set_yscale("log")

        plain_formatter = ticker.FuncFormatter(lambda x, pos: f"{x:.1f}")
        sci_formatter = ticker.LogFormatterSciNotation()

        for ax in [ax_loss, ax_phase]:
            plt.draw()
            ymin, ymax = ax.get_ylim()
            if ymin > 100:
                ax.yaxis.set_major_formatter(sci_formatter)
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            else:
                ax.yaxis.set_major_formatter(plain_formatter)
                ax.yaxis.set_major_locator(
                    ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def plot_basin_on_ax(ax, methods_config, metric="mae"):
    """Plots the aggregated Basin KDE on a specific Matplotlib axis."""
    ANALYSIS_WINDOW = 75
    WARMUP_EPOCHS = 25 if metric == "sera" else 5
    is_log_y = metric == "sera"
    legend_handles = []
    flatness_dict = {}

    for config in methods_config:
        label = config["label"]
        color = config["color"]
        search_pattern = config["search_pattern"]

        valid_dfs = load_all_folds(label, search_pattern, metric=metric,
                                   warmup_epochs=WARMUP_EPOCHS, filter_runs=True)
        if not valid_dfs:
            continue

        basin_data_y, basin_data_x = [], []
        for df in valid_dfs:
            recent_df = df.iloc[-ANALYSIS_WINDOW:]
            y_vals = np.clip(recent_df[metric].values, a_min=1e-6, a_max=None)
            basin_data_y.extend(y_vals)
            basin_data_x.extend(recent_df["awareness"].values)

        y_std = np.std(basin_data_y, ddof=1)
        flatness_score = 1.0 / (y_std + 1e-8)
        flatness_dict[label] = flatness_score

        formatted_score = (f"{flatness_score:.1e}"
                           if flatness_score > 10000 or flatness_score < 0.01
                           else f"{flatness_score:.2f}")
        display_label = f"{label} (Flatness: {formatted_score})"

        try:
            sns.kdeplot(x=basin_data_x, y=basin_data_y, ax=ax, color=color, fill=True,
                       alpha=0.3, levels=5, thresh=0.01, log_scale=(False, is_log_y))
            sns.kdeplot(x=basin_data_x, y=basin_data_y, ax=ax, color=color, levels=[0.5],
                       linewidths=1, log_scale=(False, is_log_y), legend=False)
        except Exception:
            ax.scatter(basin_data_x, basin_data_y, color=color, alpha=0.1, s=5, label=label)

        ax.scatter(np.mean(basin_data_x), np.mean(basin_data_y), color=color, marker="o",
                  s=60, edgecolors="k", zorder=10)

        proxy = mlines.Line2D([], [], color=color, marker="o", linestyle="None", mec="k",
                              markersize=60 ** 0.5, label=display_label)
        legend_handles.append(proxy)

    y_label = "Loss (SERA)" if metric == "sera" else "Loss (MAE)"
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(y_label)

    if is_log_y:
        plt.draw()
        ymin, ymax = ax.get_ylim()
        if ymin > 100:
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:.1f}"))
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    return flatness_dict


def alpha_sera_cor(ax, fold, folder, iqr_multiplier=3.0):
    """Plots alpha-SERA correlation for a specific fold, using all runs and
    relying purely on IQR to drop numerical explosions."""
    runs = [0, 1, 2, 3, 4]
    filenames = [f"fold_{fold}_run{r}_val_log.csv" for r in runs]
    if not os.path.exists(os.path.join(folder, filenames[0])):
        filenames = [f"fold_{fold}_run{r}_val_log (1).csv" for r in runs]

    log_files = [os.path.join(folder, f) for f in filenames]

    df_list = []
    for i, f in enumerate(log_files):
        try:
            temp_df = pd.read_csv(f)
            temp_df["run_id"] = f"Run {runs[i]}"
            df_list.append(temp_df)
        except FileNotFoundError:
            pass

    if not df_list:
        return

    all_data = pd.concat(df_list, ignore_index=True)

    Q1 = all_data["sera"].quantile(0.25)
    Q3 = all_data["sera"].quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + (iqr_multiplier * IQR)
    clean_data = all_data[all_data["sera"] <= threshold].copy()

    x = clean_data["awareness"].values
    y = clean_data["sera"].values

    p_corr, _ = pearsonr(x, y)
    s_corr, _ = spearmanr(x, y)
    k_corr, _ = kendalltau(x, y)
    colormap = cbd.RdGy_8.mpl_colors

    sns.scatterplot(data=clean_data, x="awareness", y="sera", hue="run_id", alpha=0.5,
                    palette=colormap, edgecolor=None, s=15, legend=False, ax=ax)
    sns.regplot(data=clean_data, x="awareness", y="sera", scatter=False, color="#4d4d4d",
               line_kws={"linestyle": "--", "linewidth": 2}, ax=ax)

    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_ylabel("SERA", fontsize=10)
    ax.set_yscale("log")

    plt.draw()
    ymin, ymax = ax.get_ylim()
    if ymin > 100:
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    else:
        plain_formatter = ticker.FuncFormatter(lambda val, pos: f"{val:.1f}")
        ax.yaxis.set_major_formatter(plain_formatter)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    stats_text = (f"Pearson ($r$): {p_corr:.3f}\n"
                 f"Spearman ($\\rho$): {s_corr:.3f}\n"
                 f"Kendall ($\\tau$): {k_corr:.3f}")
    ax.text(0.05, 0.10, stats_text, transform=ax.transAxes, fontsize=9, va="bottom", ha="left")


def generate_analysis_figure(dataset="phonons", methods_config=None,
                             result_dir="experiments/final", primary_metric="mae",
                             filter_runs=False, filename=None, outdir="figs/paper",
                             **filter_kwargs):
    sns.set_context("paper")

    if methods_config is None:
        methods_config = [
            {"label": "DILA", "color": "#ff7f0e",
             "search_pattern": f"{result_dir}/{dataset}/smooth_dila/fold_*_run*_val_log*.csv"},
            {"label": "Control", "color": "#4d4d4d",
             "search_pattern": f"{result_dir}/{dataset}/control/fold_*_run*_val_log*.csv"},
        ]

    fig, (axes_row1, axes_row2) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6),
                                               layout="compressed")

    plot_universal_dynamics_on_axes(axes_row1, methods_config, metric=primary_metric,
                                    filter_runs=True, **filter_kwargs)

    mae_flatness = plot_basin_on_ax(axes_row2[0], methods_config, metric="mae")
    sera_flatness = plot_basin_on_ax(axes_row2[1], methods_config, metric="sera")

    dila_folder = f"{result_dir}/{dataset}/smooth_dila"
    alpha_sera_cor(axes_row2[2], fold=0, folder=dila_folder)

    start = ord("a")
    for i, ax in enumerate(axes_row1):
        ax.set_title(chr(start + i), loc="left", fontsize=9)
    for i, ax in enumerate(axes_row2):
        ax.set_title(chr(start + i + 3), loc="left", fontsize=9)

    filename = (os.path.join(outdir, f"{dataset}_unified_analysis.jpg")
               if filename is None else filename)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig.savefig(filename, dpi=600)
    plt.close(fig)
    logger.info("Saved unified analysis figure to %s", filename)
    return mae_flatness, sera_flatness


def generate_latex_flatness_table(all_flatness_data, table_filename):
    """Compiles a LaTeX table from flatness data: scientific notation,
    bold-best, Control before DILA."""
    def fmt_latex(val, is_best=False):
        if pd.isna(val) or np.isnan(val):
            return "-"

        def b(val_str):
            return f"\\mathbf{{{val_str}}}" if is_best else val_str

        if abs(val) > 10000 or (abs(val) < 0.01 and val != 0):
            base, exp = f"{val:.1e}".split("e")
            return f"${b(base)} \\times 10^{{{int(exp)}}}$"
        return f"${b(f'{val:.2f}')}$"

    def check_is_best(val, best_val):
        if pd.isna(val) or best_val is None:
            return False
        return math.isclose(val, best_val, rel_tol=1e-9, abs_tol=1e-12)

    def method_priority(m):
        if m == "Control":
            return 0
        if m == "DILA":
            return 1
        return 2

    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{llcc}\n\\toprule\n"
    latex_str += "Dataset & Method & MAE Flatness & SERA Flatness \\\\\n\\midrule\n"

    for dataset, data in all_flatness_data.items():
        dataset_formatted = dataset.replace("_", "\\_")
        raw_methods = list(data["mae"].keys())
        methods = sorted(raw_methods, key=method_priority)

        mae_vals = [data["mae"].get(m, np.nan) for m in methods]
        sera_vals = [data["sera"].get(m, np.nan) for m in methods]

        valid_mae = [v for v in mae_vals if not np.isnan(v)]
        valid_sera = [v for v in sera_vals if not np.isnan(v)]

        best_mae = max(valid_mae) if valid_mae else None
        best_sera = max(valid_sera) if valid_sera else None

        for i, method in enumerate(methods):
            m_flat = mae_vals[i]
            s_flat = sera_vals[i]
            m_str = fmt_latex(m_flat, is_best=check_is_best(m_flat, best_mae))
            s_str = fmt_latex(s_flat, is_best=check_is_best(s_flat, best_sera))
            ds_col = f"\\texttt{{{dataset_formatted}}}" if i == 0 else ""
            latex_str += f"{ds_col} & {method} & {m_str} & {s_str} \\\\\n"

        latex_str += "\\midrule\n"

    latex_str = latex_str.rsplit("\\midrule\n", 1)[0] + "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += ("\\caption{Quantitative analysis of generalisation basin geometric "
                  "stability across all evaluated materials datasets.}\n")
    latex_str += "\\label{tab:combined_flatness}\n\\end{table}\n"

    os.makedirs(os.path.dirname(table_filename) or ".", exist_ok=True)
    with open(table_filename, "w") as text_file:
        text_file.write(latex_str)
    logger.info("Saved combined LaTeX flatness table to %s", table_filename)


# Per-dataset filter_kwargs overrides, ported verbatim from Final_figs.ipynb
# cell 19 (MatFold log_gvrh: wider spike/convergence tolerance for the
# 300x-1e9x fold-1 Control divergence).
FIG2_FILTER_OVERRIDES = {
    ("matfold", "log_gvrh"): {"explode_ratio": 50.0, "conv_ratio": 3.25},
}


def make_fig2_dynamics(expt_dir: str, expt_name: str, outdir: str, tables_dir: str):
    """Runs generate_analysis_figure for each modelled dataset and writes the
    combined flatness LaTeX table, replicating Final_figs.ipynb cells 17/19."""
    flatness_data = {}
    for dataset in MODELLED_DATASETS:
        overrides = FIG2_FILTER_OVERRIDES.get((expt_name, dataset), {})
        filename = (os.path.join(outdir, f"{dataset}_matfold_unified_analysis.jpg")
                   if expt_name == "matfold" else None)
        m_flat, s_flat = generate_analysis_figure(
            dataset=dataset, primary_metric="sera", result_dir=expt_dir,
            filename=filename, outdir=outdir, filter_runs=True, **overrides)
        flatness_data[dataset] = {"mae": m_flat, "sera": s_flat}

    table_name = ("matfold_combined_flatness_table.txt" if expt_name == "matfold"
                 else "combined_flatness_table.txt")
    generate_latex_flatness_table(flatness_data, os.path.join(tables_dir, table_name))
    return flatness_data


# ----------------------------------------------------------------------
# Fig 3 data layer: converged-health run screening + legacy-npz merge,
# ported verbatim from Final_figs.ipynb cell 21 ("New data layer" — replaces
# the deprecated analyse_dataset/imba_analyser stack).
# ----------------------------------------------------------------------
METHODS_ORDERED = {"control": "Control", "dir": "DIR", "bsam": "BSAM",
                   "smooth_dila": "DILA"}
MATH_NAMES = {
    "log_kvrh": r"log($K_{\mathrm{{vrh}}}$)",
    "log_gvrh": r"log($G_{\mathrm{{vrh}}}$)",
    "perovskites": r"$E_{\mathrm{{F}}}$ [eV cell$^{-1}$]",
    "phonons": r"$\nu _{\mathrm{{last}}}$ [cm$^{-1}$]",
}


def _masked_final_stats(df, warmup_epochs=20, explode_ratio=100.0,
                        sera_explode_ratio=250.0):
    """Converged (last-5-epoch) validation MAE and R2 with exploded epochs
    masked out. An epoch is 'exploded' when its MAE (SERA) exceeds the run's
    own post-warmup median by explode_ratio (sera_explode_ratio)."""
    post = df.iloc[warmup_epochs:]
    healthy = post
    if "mae" in healthy.columns:
        healthy = healthy[healthy["mae"] <= post["mae"].median() * explode_ratio]
    if "sera" in healthy.columns:
        healthy = healthy[healthy["sera"] <= post["sera"].median() * sera_explode_ratio]
    tail = healthy.iloc[-5:]
    r2 = tail["r2_score"].mean() if "r2_score" in tail.columns else np.nan
    return tail["mae"].mean(), r2


def filter_index_runs(index, warmup_epochs=20, conv_ratio=5.25, r2_floor=0.65):
    """Converged-health screen for prediction-level products (parity
    composites, metrics tables, screening analysers). Unlike the Fig 2
    trajectory filter (filter_outlier_runs), a transient mid-training
    explosion does NOT disqualify a run here: these products are computed
    from the best checkpoint, so a run that diverged mid-training but
    recovered is a valid sample. A run is dropped only if its converged
    quality is poor: final R2 < r2_floor, or final MAE above conv_ratio x the
    fold-best final MAE. Rows without a val_log are kept (screened later by
    filter_bad_runs)."""
    keep = []
    for (ds, meth, fold), g in index.groupby(["dataset", "method", "fold"]):
        stats, rows = [], []
        for i, row in g.iterrows():
            if row["val_log"] is None:
                keep.append(i)
                continue
            df = pd.read_csv(row["val_log"])
            if len(df) > warmup_epochs + 5:
                stats.append(_masked_final_stats(df, warmup_epochs))
                rows.append(i)
        finite = [m for m, _ in stats if np.isfinite(m)]
        if not finite:
            continue
        best = np.min(finite)
        for (mae, r2), i in zip(stats, rows):
            if not np.isfinite(mae) or mae > best * conv_ratio:
                continue
            if np.isfinite(r2) and r2 < r2_floor:
                continue
            keep.append(i)
    kept = index.loc[sorted(keep)].reset_index(drop=True)
    logger.info("converged-health filter: kept %d/%d runs", len(kept), len(index))
    return kept


def load_npz_runs(pattern):
    """Legacy npz runs, metrics recomputed from raw arrays (SERA_T0=0.5),
    filtered per fold with the prediction-level screen."""
    by_fold = {}
    for path in sorted(glob.glob(pattern)):
        raw = PredictionSet.load(path)
        name = os.path.basename(path).replace("_ml_pred.npz", "")
        ps = PredictionSet(raw.targets, raw.preds, relevances=raw.relevances,
                           densities=raw.densities, name=name)
        m = re.search(r"fold_(\d+)", name)
        by_fold.setdefault(m.group(1) if m else "0", []).append(ps)
    return [ps for runs in by_fold.values() for ps in filter_bad_runs(runs)]


def collect_filtered(base, merge_npz=False):
    """{dataset: {method: [PredictionSet]}} from the filtered run index."""
    index = filter_index_runs(discover_runs(base))
    preds = load_predictions(index) if not index.empty else {}
    if merge_npz:
        for dataset in MODELLED_DATASETS:
            for method in METHODS_ORDERED:
                runs = load_npz_runs(os.path.join(
                    base, dataset, method, "fold_*_run*_ml_pred.npz"))
                if runs:
                    preds.setdefault(dataset, {})[method] = runs
    return index, preds


class FoldAnalyser:
    """Adapter exposing the old imba_analyser interface over analysis objects."""

    def __init__(self, labels, all_preds):
        self.labels = labels
        self.all_preds = all_preds

    def _filter_bad_runs(self, runs):
        return filter_bad_runs(runs)

    def calculate_adaptive_discovery_metrics(self, pred_group, budget_ratio=1.0,
                                             rel_threshold=0.5, discover_mode="auto"):
        try:
            return discovery_metrics(pred_group, rel_threshold=rel_threshold,
                                     budget_ratio=budget_ratio, discover_mode=discover_mode)
        except Exception as e:
            return {"Error": str(e)}


def build_analysers(preds_by_method, folds=range(5)):
    analysers = []
    for fold in folds:
        labels, all_preds = [], []
        for method, label in METHODS_ORDERED.items():
            runs = [p for p in preds_by_method.get(method, [])
                    if p.name and p.name.startswith(f"fold_{fold}_")]
            if runs:
                labels.append(label)
                all_preds.append(runs)
        analysers.append(FoldAnalyser(labels, all_preds))
    return analysers


def fold_groups(preds_by_method, fold):
    groups = {}
    for method, label in METHODS_ORDERED.items():
        runs = [p for p in preds_by_method.get(method, [])
                if p.name and p.name.startswith(f"fold_{fold}_")]
        if runs:
            groups[label] = runs
    return groups


# ----------------------------------------------------------------------
# Fig 4: OOD discovery metrics, ported verbatim from Final_figs.ipynb
# cells 28/30-34 (calc_imba_metrics / plot_discovery_analysis are the
# original code, unchanged — they consume the FoldAnalyser adapters above).
# ----------------------------------------------------------------------
# discover_mode per dataset, ported from cells 30/33: perovskites screens the
# LOW tail (below-target discovery), the other three screen the HIGH tail.
FIG4_DISCOVER_MODE = {"log_kvrh": "high", "log_gvrh": "high",
                      "phonons": "high", "perovskites": "low"}


def calc_imba_metrics(imba_analysers, rel_thresholds=np.linspace(0.5, 0.9, num=9),
                      budget_ratios=np.linspace(0.5, 3, num=11), discover_mode="high"):
    all_results = []
    for fold, imba_analyser in enumerate(imba_analysers):
        for rel_threshold in rel_thresholds:
            for budget_ratio in budget_ratios:
                for i, method in enumerate(imba_analyser.labels):
                    results = imba_analyser.calculate_adaptive_discovery_metrics(
                        imba_analyser.all_preds[i], budget_ratio=budget_ratio,
                        rel_threshold=rel_threshold, discover_mode=discover_mode)
                    if "Error" in results:
                        continue
                    filtered_results = {key: results[key] for key in [
                        "Discovery Mode", "Tail Size (GT)", "Extrap. Precision",
                        "Tail Recall", "Tail MAE", "Breakdown"]}
                    filtered_results.update({
                        "Method": method, "Rel Threshold": rel_threshold,
                        "Budget Ratio": budget_ratio, "Fold": fold})
                    all_results.append(filtered_results)
    return pd.DataFrame(all_results)


def plot_discovery_analysis(df, axes=None, rel_threshold=0.8, save_path=None):
    """Generates a Discovery Performance analysis figure: Efficiency
    (Precision), Completeness (Recall), and Error (MAE)."""
    subset = df[df["Rel Threshold"] == rel_threshold].copy()
    if subset.empty:
        logger.warning("No data found for Rel Threshold = %s", rel_threshold)
        return

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), constrained_layout=True)

    e_colors = ["#4d4d4d", "#2166ac", "#b2182b", "#35978f", "#ff7f0e"]
    f_colors = ["#e0e0e0", "#d1e5f0", "#fff5eb", "#c7eae5", "#ffbb78"]
    markers = ["s", "o", "^", "v", "D"]

    def plot_metric(ax, metric_col, ylabel, show_legend=False):
        for i, method in enumerate(["Control", "DIR", "BSAM", "DILA"]):
            c_idx = i % len(e_colors)
            sns.lineplot(
                data=subset[subset["Method"] == method], x="Budget Ratio", y=metric_col,
                ax=ax, color=e_colors[c_idx], linewidth=1.5, errorbar=("ci", 95),
                err_style="band", label=method if show_legend else None,
                marker=markers[c_idx], markersize=7, markeredgecolor=e_colors[c_idx],
                markerfacecolor=f_colors[c_idx])
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Budget Ratio (x Tail Size)", fontsize=10)
        ax.axvline(1.0, color="#555555", linestyle="--", alpha=0.6)
        if show_legend:
            ax.legend(fontsize=8, loc="best")

    plot_metric(axes[0], "Extrap. Precision", "Success Rate (Precision)")
    plot_metric(axes[1], "Tail Recall", "Fraction Found (Recall)", show_legend=True)
    plot_metric(axes[2], "Tail MAE", "Tail MAE (eV)")

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        logger.info("Plot saved to %s", save_path)


def make_fig4_discovery(predictions, expt_name, outdir, tables_dir):
    """Runs calc_imba_metrics for each modelled dataset, writes the raw
    metric CSVs, and composes the 4x3 CV/MatFold discovery grid at
    rel_threshold=0.8, ported from Final_figs.ipynb cells 30-31/33-34."""
    results = {}
    for dataset in MODELLED_DATASETS:
        analysers = build_analysers(predictions.get(dataset, {}))
        if not any(a.labels for a in analysers):
            continue
        df = calc_imba_metrics(analysers, discover_mode=FIG4_DISCOVER_MODE[dataset])
        df.to_csv(os.path.join(tables_dir, f"discov_metrics_{dataset}_{expt_name}.csv"),
                  index=False)
        results[dataset] = df

    if not results:
        return

    fig, axes_grid = plt.subplots(len(MODELLED_DATASETS), 3, figsize=(9.5, 9),
                                  sharex="col", layout="compressed")
    axes_grid = np.atleast_2d(axes_grid)
    ds_math = {"log_kvrh": r"log($K_{\mathrm{vrh}}$)", "log_gvrh": r"log($G_{\mathrm{vrh}}$)",
              "perovskites": "Perovskites", "phonons": "Phonons"}
    for row, dataset in enumerate(MODELLED_DATASETS):
        if dataset not in results:
            for ax in axes_grid[row]:
                ax.axis("off")
            continue
        plot_discovery_analysis(results[dataset], axes=axes_grid[row], rel_threshold=0.8)
        axes_grid[row][-1].text(1.075, 0.5, ds_math[dataset], va="center", ha="center",
                                rotation=90, transform=axes_grid[row][-1].transAxes)

    fig.savefig(os.path.join(outdir, f"fig4_discov_performance_{expt_name}.jpg"), dpi=600)
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 5: SERA/alpha sensitivity to dataset imbalance (h, W1) via a PCA of
# the two Stage-1 descriptors, CV + MatFold 2x4 grid. Ported verbatim from
# Final_figs.ipynb cells 36-38.
# ----------------------------------------------------------------------
# Hardcoded (h, w1) knowledge base — ported verbatim from cell 36. These
# match make_dataset_profiles' own computed DIL/Wasserstein for the same
# four datasets (kept as a literal constant here, matching the notebook,
# rather than re-deriving from the Fig 1 result_df).
FIG5_DATASET_META = {
    "log_kvrh":    {"h": 0.516, "w1": 0.207},
    "log_gvrh":    {"h": 0.498, "w1": 0.153},
    "phonons":     {"h": 0.631, "w1": 0.354},
    "perovskites": {"h": 0.521, "w1": 0.186},
}


def _fig5_pca_meta():
    meta_df = pd.DataFrame.from_dict(FIG5_DATASET_META, orient="index")
    scaler = StandardScaler()
    coords = PCA(n_components=2).fit_transform(scaler.fit_transform(meta_df[["h", "w1"]]))
    meta = {k: dict(v) for k, v in FIG5_DATASET_META.items()}
    for i, ds_name in enumerate(meta_df.index):
        meta[ds_name]["PC1"] = coords[i, 0]
        meta[ds_name]["PC2"] = coords[i, 1]
    pca = PCA(n_components=2).fit(scaler.fit_transform(meta_df[["h", "w1"]]))
    return meta, pca


def extract_robust_metrics(dataset, analysers, sera_t_threshold=0.8):
    """Extracts paired per-fold SERA/alpha deltas vs Control from a list of
    FoldAnalyser objects, using each analyser's own _filter_bad_runs."""
    if not isinstance(analysers, list):
        analysers = [analysers]
    metrics_dict = {}
    if not analysers:
        return pd.DataFrame()

    methods = analysers[0].labels
    for i, method in enumerate(methods):
        valid_folds = []
        for fold, analyser in enumerate(analysers):
            if i >= len(analyser.all_preds):
                continue
            raw_runs = analyser.all_preds[i]
            valid_runs = analyser._filter_bad_runs(raw_runs)
            if not valid_runs:
                continue

            labels = np.concatenate([np.ravel(p.targets) for p in valid_runs])
            preds = np.concatenate([np.ravel(p.preds) for p in valid_runs])

            if hasattr(valid_runs[0], "relevances") and valid_runs[0].relevances is not None:
                relevances = np.concatenate([np.ravel(p.relevances) for p in valid_runs])
                sera_val = calc_sera(labels, preds, relevances, t=sera_t_threshold)
                sera_val = float(sera_val.item()) if hasattr(sera_val, "item") else float(sera_val)
            else:
                sera_val = np.nan

            if hasattr(valid_runs[0], "densities") and valid_runs[0].densities is not None:
                densities = np.concatenate([np.ravel(p.densities) for p in valid_runs])
                if len(densities) > 3:
                    alpha_val = float(calc_alpha(
                        torch.as_tensor(labels, dtype=torch.float32),
                        torch.as_tensor(preds, dtype=torch.float32),
                        torch.as_tensor(densities, dtype=torch.float32)).item())
                else:
                    alpha_val = np.nan
            else:
                alpha_val = np.nan

            valid_folds.append({"fold": fold, "test_sera": sera_val, "test_awareness": alpha_val})

        if valid_folds:
            metrics_dict[method] = pd.DataFrame(valid_folds)

    ctrl_key = next((k for k in metrics_dict.keys() if k.lower() == "control"), None)
    if not ctrl_key:
        logger.warning("No valid control data found for %s", dataset)
        return pd.DataFrame()

    ctrl_df = metrics_dict[ctrl_key].set_index("fold")
    ctrl_sera_by_fold = ctrl_df["test_sera"].to_dict()
    ctrl_alpha_by_fold = ctrl_df["test_awareness"].to_dict()

    rows = []
    for method, method_df in metrics_dict.items():
        if method == ctrl_key:
            continue
        prefix = "DILA" if "dila" in method.lower() else method.upper()
        for _, fold_row in method_df.iterrows():
            fold = fold_row["fold"]
            sera_val = fold_row["test_sera"]
            alpha_val = fold_row["test_awareness"]
            ctrl_sera_fold = ctrl_sera_by_fold.get(fold, np.nan)
            ctrl_alpha_fold = ctrl_alpha_by_fold.get(fold, np.nan)

            if pd.notnull(ctrl_sera_fold) and ctrl_sera_fold > 0:
                sera_delta = ((ctrl_sera_fold - sera_val) / ctrl_sera_fold) * 100
            else:
                sera_delta = np.nan
            alpha_gain = (alpha_val - ctrl_alpha_fold) if pd.notnull(ctrl_alpha_fold) else np.nan

            rows.append({"dataset": dataset, "method": prefix, "fold": fold,
                        "sera_delta": float(sera_delta) if pd.notnull(sera_delta) else np.nan,
                        "alpha_gain": float(alpha_gain) if pd.notnull(alpha_gain) else np.nan})

    return pd.DataFrame(rows)


def aggregate_datasets(results_map, meta_data):
    dfs = []
    for ds_name, df in results_map.items():
        if df.empty:
            continue
        df_copy = df.copy()
        if ds_name in meta_data:
            df_copy["h_index"] = meta_data[ds_name]["h"]
            df_copy["w1_dist"] = meta_data[ds_name]["w1"]
            df_copy["PC1"] = meta_data[ds_name]["PC1"]
            df_copy["PC2"] = meta_data[ds_name]["PC2"]
        dfs.append(df_copy)
    return pd.concat(dfs, ignore_index=True)


def analyse_method_sensitivity(df, ax, pca, x_metric="h_index", performance="sera",
                               is_top_row=False, estimator="median", error_type="sem"):
    """Plots performance metric against dataset difficulty, using robust
    (median + SEM) estimators so heavy-tailed SERA doesn't blow up error bars."""
    y_metric = "sera_delta" if performance == "sera" else "alpha_gain"
    e_colors = ["#2166ac", "#b2182b", "#35978f"]
    f_colors = ["#d1e5f0", "#fff5eb", "#c7eae5"]
    markers = ["o", "^", "v"]
    method_names = ["DIR", "BSAM", "DILA"]

    x_min, x_max = df[x_metric].min(), df[x_metric].max()
    offset_step = (x_max - x_min) * 0.015 if x_max > x_min else 0.01

    for idx, name in enumerate(method_names):
        method_df = df[df["method"] == name]
        if method_df.empty:
            continue

        if estimator == "median":
            grouped_center = method_df.groupby(x_metric)[y_metric].median().reset_index(name="center")
        else:
            grouped_center = method_df.groupby(x_metric)[y_metric].mean().reset_index(name="center")

        grouped_stats = method_df.groupby(x_metric)[y_metric].agg(["count", "std"]).reset_index()
        grouped = pd.merge(grouped_center, grouped_stats, on=x_metric)

        if error_type == "sem":
            grouped["error"] = grouped["std"] / np.sqrt(grouped["count"])
        elif error_type == "ci95":
            grouped["error"] = grouped.apply(
                lambda row: scipy_stats.t.ppf(0.975, df=row["count"] - 1) * (row["std"] / np.sqrt(row["count"]))
                if row["count"] > 1 and pd.notnull(row["std"]) else 0, axis=1)
        elif error_type == "iqr" and estimator == "median":
            q1 = method_df.groupby(x_metric)[y_metric].quantile(0.25).reset_index(name="q1")
            q3 = method_df.groupby(x_metric)[y_metric].quantile(0.75).reset_index(name="q3")
            grouped = pd.merge(grouped, q1, on=x_metric)
            grouped = pd.merge(grouped, q3, on=x_metric)
            grouped["err_lower"] = grouped["center"] - grouped["q1"]
            grouped["err_upper"] = grouped["q3"] - grouped["center"]

        x_vals = grouped[x_metric] + ((idx - 1) * offset_step)
        if error_type == "iqr" and estimator == "median":
            yerr = [grouped["err_lower"], grouped["err_upper"]]
        else:
            yerr = grouped["error"]

        ax.errorbar(x_vals, grouped["center"], yerr=yerr, c=e_colors[idx], marker=markers[idx],
                    ms=7, markerfacecolor=f_colors[idx], alpha=0.8, label=name,
                    linestyle="none", capsize=0, elinewidth=1.5)

    ax.axhline(0, color="black", lw=1.5, alpha=0.4, linestyle=":")

    if x_metric == "h_index":
        x_label = "Distribution Imbalance ($h$)"
    elif x_metric == "w1_dist":
        x_label = r"Wasserstein Distance ($W_1$)"
    elif x_metric == "PC1":
        x_label = f"PC1: ({pca.explained_variance_ratio_[0]:.1%} var) Statistical\n Imbalance Magnitude"
    elif x_metric == "PC2":
        x_label = f"PC2: ({pca.explained_variance_ratio_[1]:.1%} var)\n Geometric Transport Cost"
    else:
        x_label = x_metric
    ax.set_xlabel(x_label, fontsize=10)

    if not is_top_row:
        ax.legend(loc="best", fontsize=8)

    if is_top_row:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        unique_ds = df[[x_metric, "dataset"]].drop_duplicates().sort_values(by=x_metric)

        def format_ds_name(name):
            if name == "log_kvrh":
                return r"$\log(K_{\mathrm{vrh}})$"
            if name == "log_gvrh":
                return r"$\log(G_{\mathrm{vrh}})$"
            if name == "phonons":
                return "Phonons"
            if name == "perovskites":
                return "Perovskites"
            return name

        ax_top.set_xticks(unique_ds[x_metric])
        ax_top.set_xticklabels([format_ds_name(ds) for ds in unique_ds["dataset"]],
                               fontsize=9, rotation=90)
        ax_top.grid(False)


def make_fig5_sensitivity(preds_by_expt, outdir):
    """Builds the CV/MatFold SERA-alpha PCA-sensitivity grid, ported from
    Final_figs.ipynb cells 36-38. Needs both 'final' and 'matfold' predictions."""
    if "final" not in preds_by_expt or "matfold" not in preds_by_expt:
        logger.warning("Fig 5 needs both 'final' and 'matfold' experiment sets — skipping.")
        return

    meta, pca = _fig5_pca_meta()

    def build_master_df(preds):
        results_map = {}
        for dataset in MODELLED_DATASETS:
            analysers = build_analysers(preds.get(dataset, {}))
            results_map[dataset] = extract_robust_metrics(dataset, analysers)
        return aggregate_datasets(results_map, meta_data=meta)

    master_df = build_master_df(preds_by_expt["final"])
    matfold_df = build_master_df(preds_by_expt["matfold"])

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
        2, 4, figsize=(9.5, 5.5), layout="constrained")

    analyse_method_sensitivity(master_df, ax1, pca, x_metric="PC1", performance="sera", is_top_row=True)
    analyse_method_sensitivity(master_df, ax2, pca, x_metric="PC2", performance="sera", is_top_row=True)
    analyse_method_sensitivity(master_df, ax3, pca, x_metric="PC1", performance="alpha", is_top_row=True)
    analyse_method_sensitivity(master_df, ax4, pca, x_metric="PC2", performance="alpha", is_top_row=True)
    ax1.set_ylabel("Relative SERA\n Reduction (%)", fontsize=10)
    ax2.set_ylabel("")
    ax4.set_ylabel("")

    analyse_method_sensitivity(matfold_df, ax5, pca, x_metric="PC1", performance="sera", is_top_row=False)
    analyse_method_sensitivity(matfold_df, ax6, pca, x_metric="PC2", performance="sera", is_top_row=False)
    analyse_method_sensitivity(matfold_df, ax7, pca, x_metric="PC1", performance="alpha", is_top_row=False)
    analyse_method_sensitivity(matfold_df, ax8, pca, x_metric="PC2", performance="alpha", is_top_row=False)
    ax5.set_ylabel("Relative SERA\n Reduction (%)", fontsize=10)
    ax7.set_ylabel(r"Awareness Gain ($\Delta \alpha$)", fontsize=10)

    ax2.sharey(ax1)
    ax4.sharey(ax3)
    ax1.sharex(ax5)
    ax2.sharex(ax6)
    ax6.sharey(ax5)
    ax8.sharey(ax7)
    ax3.sharex(ax7)
    ax4.sharex(ax8)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.label_outer()
    for ax in [ax3, ax7]:
        ax.tick_params(labelleft=True)
        ax.yaxis.label.set_visible(True)
        ax.set_ylabel(r"Awareness Gain ($\Delta \alpha$)", fontsize=10)

    fig.canvas.draw()
    dataset_keywords = ["vrh", "Perov", "perov", "Phonon", "phonon"]
    for ax in fig.axes:
        for label in ax.get_xticklabels():
            text_str = label.get_text()
            if any(keyword in text_str for keyword in dataset_keywords):
                x_val = label.get_position()[0]
                label.set_visible(False)
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                new_x = x_val
                if "Perovskites" in text_str or "perovskites" in text_str:
                    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
                    if ax.get_xlim() == ax1.get_xlim() or ax.get_xlim() == ax3.get_xlim():
                        new_x = x_val - (x_span * 0.05)
                    else:
                        new_x = x_val + (x_span * 0.05)
                    ax.plot([x_val, new_x], [1.0, 1.02], transform=trans, color="gray",
                           lw=0.8, clip_on=False)
                ax.text(new_x, 1.02, text_str, transform=trans, rotation=90,
                       va="bottom", ha="center", fontsize=10, color="#4d4d4d")

    ax4.text(1.05, 0.5, "Cross-Validation (CV)", va="center", ha="center", rotation=90,
             transform=ax4.transAxes, fontsize=10)
    ax8.text(1.05, 0.5, "MatFold (OOD Split)", va="center", ha="center", rotation=90,
             transform=ax8.transAxes, fontsize=10)
    fig.align_ylabels(((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)))

    fig.savefig(os.path.join(outdir, "fig5_quantitative_alpha_sera_sensitivity.jpg"),
               dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Fig 5 sensitivity grid")


# ----------------------------------------------------------------------
# Method profiling (runtime / GPU memory), ported verbatim from
# Final_figs.ipynb cells 40-42.
#
# Note: the notebook's cell 41 redefines plot_epoch_time with a simpler body
# (no cmap, no axis labels) that shadows cell 40's version — the final
# figure (cell 42) runs under the last definition, and the axis labels it
# drops are set explicitly by the caller anyway (axes1[0].set_ylabel(...)).
# Ported with the same last-definition-wins behaviour, not cell 40's.
# ----------------------------------------------------------------------
PROFILE_METHOD_DIRS = {"Control": "control", "LDS": "lds", "DIR": "dir",
                       "DILA": "smooth_dila", "BSAM": "bsam", "ASAM": "asam"}


def load_and_process_profiles(dataset, methods, basedir):
    method_patterns = {
        method: os.path.join(basedir, dataset, PROFILE_METHOD_DIRS[method],
                             "fold_0_*_runtime_profile.csv")
        for method in methods
    }
    all_data = []
    for method_name, pattern in method_patterns.items():
        files = glob.glob(pattern)
        if not files:
            logger.warning("No files found for %s with pattern %s", method_name, pattern)
            continue
        for file in files:
            try:
                df = pd.read_csv(file)
                clean_df = df[~(df["epoch"] == 0)].copy()
                df_group = clean_df.groupby("epoch").agg(
                    {"time_ms": "sum", "peak_mem_mb": "max"}).reset_index()
                df_group.columns = ["epoch", "time_ms", "peak_mem_mb"]
                df_group["Method"] = method_name
                df_group["Run"] = os.path.basename(file)
                all_data.append(df_group)
            except Exception as e:
                logger.warning("Error reading %s: %s", file, e)

    if not all_data:
        raise ValueError(f"No profiling data loaded for {dataset} under {basedir}.")
    return pd.concat(all_data, ignore_index=True)


def plot_epoch_time(df, ax):
    sns.boxplot(data=df, x="Method", y="time_ms", hue="Method", palette="viridis",
               ax=ax, showfliers=False)


def plot_gpu_usage(df, ax):
    cmap = ptcb.diverging.RdBu_4.mpl_colors
    sns.boxplot(data=df, x="Method", y="peak_mem_mb", hue="Method", palette=cmap,
               ax=ax, showfliers=False)
    ax.set_ylabel("Peak GPU Memory Reserved (MB)")
    ax.set_xlabel("")


def make_fig_profiling(profile_dir, outdir):
    if not os.path.isdir(profile_dir):
        logger.warning("No profile dir at %s — skipping method-profiling figure.", profile_dir)
        return

    fig, (axes1, axes2) = plt.subplots(2, 4, figsize=(9.5, 4), sharex=True,
                                       layout="compressed")
    axes1, axes2 = axes1.flatten(), axes2.flatten()
    any_plotted = False
    for dataset, ax1, ax2 in zip(MODELLED_DATASETS, axes1, axes2):
        try:
            combined_df = load_and_process_profiles(
                dataset, methods=["Control", "DIR", "BSAM", "DILA"], basedir=profile_dir)
        except ValueError as e:
            logger.warning("%s", e)
            continue
        plot_epoch_time(combined_df, ax1)
        plot_gpu_usage(combined_df, ax2)
        ax1.set_title(dataset, loc="center", fontsize=10)
        ax1.set_ylabel("")
        ax2.set_ylabel("")
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    axes1[0].set_ylabel("Epoch runtime (ms)")
    axes2[0].set_ylabel("Peak GPU Memory\n Reserved (MB)")
    fig.savefig(os.path.join(outdir, "fig_methods_profile.jpg"), dpi=600)
    plt.close(fig)
    logger.info("Saved method-profiling figure")


# ----------------------------------------------------------------------
# Alpha-SERA Transfer Mechanism: 12-panel figure investigating why alpha
# gains don't always yield commensurate SERA reductions (saturation ceiling
# on perovskites; flat alpha-SERA slope at high alpha). Ported verbatim from
# Final_figs.ipynb cells 44-45, reusing filter_outlier_runs from Fig 2.
# ----------------------------------------------------------------------
TRANSFER_DS_LABELS = {
    "log_kvrh":    r"$\log(K_{\mathrm{vrh}})$",
    "log_gvrh":    r"$\log(G_{\mathrm{vrh}})$",
    "perovskites": "Perovskites",
    "phonons":     "Phonons",
}
TRANSFER_DS_COLORS = {
    "log_kvrh": "#e41a1c", "log_gvrh": "#377eb8",
    "perovskites": "#4daf4a", "phonons": "#984ea3",
}
TRANSFER_METHODS = list(METHODS_ORDERED.keys())  # control, dir, bsam, smooth_dila
# Mapped 1:1 from the manuscript theme: color -> line/marker edge, fill -> light marker face.
TRANSFER_MSTYLE = {
    "control":     {"label": "Control", "color": "#4d4d4d", "fill": "#e0e0e0", "marker": "s"},
    "dir":         {"label": "DIR",     "color": "#2166ac", "fill": "#d1e5f0", "marker": "o"},
    "bsam":        {"label": "BSAM",    "color": "#b2182b", "fill": "#fff5eb", "marker": "^"},
    "smooth_dila": {"label": "DILA",    "color": "#35978f", "fill": "#c7eae5", "marker": "v"},
}
TRANSFER_ALPHA_BINS = np.linspace(0.30, 0.99, 16)
TRANSFER_TAIL_N = 75  # epochs per run for converged-basin scatter


def _load_method_transfer(dataset, method, base_dir, warmup=20):
    """Load and filter val_log CSVs for one dataset/method combination."""
    pattern = os.path.join(base_dir, dataset, method, "fold_*_run*_val_log.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    fold_files = defaultdict(list)
    for f in files:
        fn = os.path.basename(f)
        try:
            fid = fn.split("fold_")[1].split("_")[0]
        except Exception:
            fid = "unk"
        fold_files[fid].append(f)

    all_dfs = []
    for fid, flist in fold_files.items():
        raw = []
        for f in flist:
            try:
                df = pd.read_csv(f)
                df["_fold"] = fid
                df["_run"] = os.path.basename(f)
                if len(df) > warmup + 5:
                    raw.append(df)
            except Exception:
                pass
        valid = filter_outlier_runs(raw, method, fid, metric="mae", warmup_epochs=warmup)
        all_dfs.extend(valid)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["dataset"] = dataset
    combined["method"] = method
    combined["method_label"] = TRANSFER_MSTYLE.get(method, {}).get("label", method)
    combined["log_sera"] = np.log10(combined["sera"].clip(lower=1e-6))
    return combined


def _transfer_spearman(df, x="awareness", y="sera"):
    """Spearman rho between alpha and raw SERA over finite rows."""
    if df.empty or x not in df.columns or y not in df.columns:
        return np.nan
    df2 = df.dropna(subset=[x, y])
    df2 = df2[np.isfinite(df2[y]) & np.isfinite(df2[x])]
    if len(df2) < 10:
        return np.nan
    rho, _ = spearmanr(df2[x], df2[y])
    return rho


def _ds_label_short(ds):
    return {"log_kvrh": r"$\log K$", "log_gvrh": r"$\log G$",
           "perovskites": "Perov.", "phonons": "Phonons"}.get(ds, ds)


def _load_transfer_data(base_dir):
    logger.info("Loading filtered data from %s/ ...", base_dir)
    data = {}
    for ds in MODELLED_DATASETS:
        data[ds] = {}
        for m in TRANSFER_METHODS:
            df = _load_method_transfer(ds, m, base_dir)
            data[ds][m] = df
            n_runs = df["_run"].nunique() if not df.empty else 0
            logger.info("  %-12s / %-20s: %2d valid runs, %6d epoch-rows", ds, m, n_runs, len(df))
    return data


def _log_transfer_dispersion_diagnostic(data, base_dir):
    """Reproduces the notebook's dispersion-artefact printout: raw
    epoch-pooled CV vs robust run-level measures, all methods pooled."""
    final_sera = {}
    for ds in MODELLED_DATASETS:
        vals = []
        for m in TRANSFER_METHODS:
            df = data[ds][m]
            if df.empty:
                continue
            fin = df.sort_values("epoch").groupby(["_fold", "_run"]).last().reset_index()
            fin_s = fin["sera"].dropna()
            fin_s = fin_s[fin_s > 0]
            vals.extend(fin_s.values)
        final_sera[ds] = np.array(vals)

    logger.info("SERA dispersion: raw epoch-pooled CV vs robust run-level measures "
               "(all methods pooled)")
    for ds in MODELLED_DATASETS:
        total_raw, total_filt = 0, 0
        for m in TRANSFER_METHODS:
            pattern = os.path.join(base_dir, ds, m, "fold_*_run*_val_log.csv")
            files = glob.glob(pattern)
            total_raw += len(files)
            total_filt += data[ds][m]["_run"].nunique() if not data[ds][m].empty else 0
        pct_filtered = (total_raw - total_filt) / total_raw * 100 if total_raw > 0 else 0

        all_s = pd.concat([data[ds][m]["sera"].dropna()
                           for m in TRANSFER_METHODS if not data[ds][m].empty])
        all_s = all_s[all_s > 0].values
        cv_all = all_s.std() / all_s.mean()
        q1, med, q3 = np.percentile(all_s, [25, 50, 75])
        iqr_all = (q3 - q1) / med

        s = final_sera[ds]
        cv_fin = s.std() / s.mean()
        q1f, medf, q3f = np.percentile(s, [25, 50, 75])
        iqr_fin = (q3f - q1f) / medf
        log10_std = np.log10(s).std()

        logger.info("  %-12s %15.1f%%  %14.2f  %19.3f  %15.3f  %14.3f  %16.3f",
                   ds, pct_filtered, cv_all, iqr_all, cv_fin, iqr_fin, log10_std)
    return final_sera


def make_fig_transfer_mechanism(outdir, base_dir=None):
    base_dir = base_dir or os.path.join(REPO, "experiments", "matfold")
    if not os.path.isdir(base_dir):
        logger.warning("No dir at %s — skipping Alpha-SERA transfer figure.", base_dir)
        return

    data = _load_transfer_data(base_dir)
    if not any(not df.empty for by_method in data.values() for df in by_method.values()):
        logger.warning("No val_log trajectories under %s — skipping transfer figure.", base_dir)
        return
    final_sera = _log_transfer_dispersion_diagnostic(data, base_dir)

    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(13, 10))
    gs_top = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.30,
                               top=0.88, bottom=0.38, left=0.07, right=0.98)
    gs_bot = gridspec.GridSpec(1, 4, figure=fig, hspace=0.30, wspace=0.38,
                               top=0.30, bottom=0.07, left=0.07, right=0.98)

    ax_scatter = [fig.add_subplot(gs_top[0, i]) for i in range(4)]
    ax_cond = [fig.add_subplot(gs_top[1, i]) for i in range(4)]
    ax_transfer = fig.add_subplot(gs_bot[0, 0])
    ax_cv_compare = fig.add_subplot(gs_bot[0, 1])
    ax_coupling = fig.add_subplot(gs_bot[0, 2])
    ax_saturation = fig.add_subplot(gs_bot[0, 3])

    # ROWS 1-2: per-dataset panels
    for col, ds in enumerate(MODELLED_DATASETS):
        ax = ax_scatter[col]
        for m in TRANSFER_METHODS:
            df = data[ds][m]
            if df.empty:
                continue
            tail = df.sort_values("epoch").groupby(["_fold", "_run"], group_keys=False).tail(
                TRANSFER_TAIL_N)
            tail2 = tail.dropna(subset=["awareness", "log_sera"])
            tail2 = tail2[np.isfinite(tail2["log_sera"])]
            st = TRANSFER_MSTYLE[m]
            ax.scatter(tail2["awareness"], tail2["log_sera"], color=st["color"], s=4,
                      alpha=0.12, rasterized=True)

        pool = pd.concat([data[ds][m] for m in TRANSFER_METHODS if not data[ds][m].empty],
                         ignore_index=True)
        rho_pool = _transfer_spearman(pool)
        ax.text(0.97, 0.96, f"$\\rho = {rho_pool:.2f}$", transform=ax.transAxes, ha="right",
               va="top", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", alpha=0.9))
        ax.set_title(TRANSFER_DS_LABELS[ds], fontsize=10, loc="center")
        ax.set_xlabel(r"$\alpha$", fontsize=9)
        if col == 0:
            ax.set_ylabel(r"$\log_{10}(\mathrm{SERA})$", fontsize=9)
        ax.tick_params(labelsize=8)
        add_panel_label(ax, col, loc=(-0.14, 1.14))

        ax2 = ax_cond[col]
        bin_centers = (TRANSFER_ALPHA_BINS[:-1] + TRANSFER_ALPHA_BINS[1:]) / 2
        for m in TRANSFER_METHODS:
            df = data[ds][m]
            if df.empty:
                continue
            df2 = df.dropna(subset=["awareness", "log_sera"])
            df2 = df2[np.isfinite(df2["log_sera"])]
            st = TRANSFER_MSTYLE[m]
            meds, q1s, q3s, xs = [], [], [], []
            for lo, hi, cx in zip(TRANSFER_ALPHA_BINS[:-1], TRANSFER_ALPHA_BINS[1:], bin_centers):
                sub = df2[(df2["awareness"] >= lo) & (df2["awareness"] < hi)]
                if len(sub) >= 8:
                    meds.append(sub["log_sera"].median())
                    q1s.append(sub["log_sera"].quantile(0.25))
                    q3s.append(sub["log_sera"].quantile(0.75))
                    xs.append(cx)
            if len(xs) > 1:
                ax2.plot(xs, meds, color=st["color"], lw=1.8, marker=st["marker"], ms=7.0,
                        mfc=st.get("fill", st["color"]), mec=st["color"], mew=0.9)
                ax2.fill_between(xs, q1s, q3s, color=st["color"], alpha=0.12)

        all_alpha = pd.concat([data[ds][m]["awareness"].dropna()
                              for m in TRANSFER_METHODS if not data[ds][m].empty])
        ax2.axvspan(all_alpha.quantile(0.10), all_alpha.quantile(0.90), alpha=0.07,
                   color="#888888", lw=0, zorder=0)
        ax2.set_xlabel(r"$\alpha$", fontsize=9)
        if col == 0:
            ax2.set_ylabel(r"Median $\log_{10}(\mathrm{SERA})$", fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.set_title(TRANSFER_DS_LABELS[ds], fontsize=10, loc="center")
        add_panel_label(ax2, col + 4, loc=(-0.14, 1.14))

    all_ls = pd.concat([data[ds][m]["log_sera"].dropna()
                        for ds in MODELLED_DATASETS for m in TRANSFER_METHODS
                        if not data[ds][m].empty])
    y1_lo, y1_hi = all_ls.quantile(0.01), all_ls.quantile(0.99)
    for ax in ax_scatter:
        ax.set_ylim(y1_lo - 0.3, y1_hi + 0.3)

    # ROW 3a (i): DILA alpha-gain vs SERA reduction
    for ds in MODELLED_DATASETS:
        ctrl_df = data[ds]["control"]
        dila_df = data[ds]["smooth_dila"]
        if ctrl_df.empty or dila_df.empty:
            continue
        ctrl_fin = ctrl_df.groupby(["_fold", "_run"]).last().groupby("_fold").agg(
            {"awareness": "mean", "sera": "mean"})
        dila_fin = dila_df.groupby(["_fold", "_run"]).last().groupby("_fold").agg(
            {"awareness": "mean", "sera": "mean"})
        common = ctrl_fin.index.intersection(dila_fin.index)
        if common.empty:
            continue
        alpha_gains = dila_fin.loc[common, "awareness"] - ctrl_fin.loc[common, "awareness"]
        sera_reds = ((ctrl_fin.loc[common, "sera"] - dila_fin.loc[common, "sera"])
                    / ctrl_fin.loc[common, "sera"] * 100)

        c = TRANSFER_DS_COLORS[ds]
        ax_transfer.scatter(alpha_gains, sera_reds, color=c, s=45, alpha=0.70,
                           edgecolors="white", lw=0.5, zorder=5)
        ag_mu, sr_mu = alpha_gains.mean(), sera_reds.mean()
        ax_transfer.scatter(ag_mu, sr_mu, color=c, s=60, marker="*", edgecolors="black",
                           lw=0.8, zorder=10)

    ax_transfer.axhline(0, color="k", lw=1.0, ls=":", alpha=0.5)
    ax_transfer.axvline(0, color="k", lw=1.0, ls=":", alpha=0.5)
    ax_transfer.set_xlabel(r"DILA $\Delta\alpha$ (gain over Control)", fontsize=9)
    ax_transfer.set_ylabel("Relative SERA Reduction (%)\n(DILA vs Control, per fold)", fontsize=9)
    ax_transfer.set_title(r"Alpha Gain $\rightarrow$ SERA Transfer", fontsize=9, loc="center")
    ax_transfer.tick_params(labelsize=8)
    add_panel_label(ax_transfer, 8, loc=(-0.18, 1.12))

    ds_handles_bot = [Line2D([0], [0], marker="*", color="w",
                             markerfacecolor=TRANSFER_DS_COLORS[ds], markersize=10,
                             label=TRANSFER_DS_LABELS[ds])
                      for ds in MODELLED_DATASETS]
    ax_transfer.legend(handles=ds_handles_bot, fontsize=7.5, loc="lower right", frameon=True)

    # ROW 3b (j): CV artifact panel
    x_pos = np.arange(len(MODELLED_DATASETS))
    xticks = [_ds_label_short(ds) for ds in MODELLED_DATASETS]
    bar_w = 0.35

    cv_pooled = []
    for ds in MODELLED_DATASETS:
        all_s = pd.concat([data[ds][m]["sera"].dropna()
                           for m in TRANSFER_METHODS if not data[ds][m].empty])
        all_s = all_s[all_s > 0].values
        cv_pooled.append(all_s.std() / all_s.mean())

    iqr_final = []
    for ds in MODELLED_DATASETS:
        s = final_sera[ds]
        s = s[s > 0]
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        iqr_final.append((q3 - q1) / med)

    ax_cv_compare.twinx()  # (unused right axis, kept for layout parity with the notebook)
    ax_cv_compare.bar(x_pos - bar_w / 2, cv_pooled, width=bar_w, facecolor="#8c8c8c",
                      alpha=0.25, hatch="//", edgecolor="#666666", lw=1.0,
                      label="CV (all epochs)")
    ax_cv_compare.bar(x_pos + bar_w / 2, iqr_final, width=bar_w,
                      color=[TRANSFER_DS_COLORS[ds] for ds in MODELLED_DATASETS],
                      alpha=0.85, label="IQR/median (final epoch)", edgecolor="white", lw=0.4)
    ax_cv_compare.set_yscale("log")
    ax_cv_compare.set_xticks(x_pos)
    ax_cv_compare.set_xticklabels(xticks, fontsize=9)
    ax_cv_compare.set_ylabel("CV (all-epoch, log scale)", fontsize=8.5, color="#555555")
    ax_cv_compare.tick_params(axis="y", labelcolor="#555555", labelsize=7.5)
    ax_cv_compare.tick_params(axis="x", labelsize=9)
    ax_cv_compare.grid(False)

    for xi, iqr in zip(x_pos, iqr_final):
        ax_cv_compare.text(xi + bar_w / 2, iqr * 1.15, f"{iqr:.2f}", ha="center",
                          va="bottom", fontsize=7.5, fontweight="bold")
    ax_cv_compare.set_title("CV Artifact vs Robust Dispersion", fontsize=9, loc="center")
    legend_handles_cv = [
        mpatches.Patch(facecolor="#8c8c8c", alpha=0.25, edgecolor="#666666", hatch="//",
                      label="CV (all epochs, log scale)"),
        mpatches.Patch(color="#666666", alpha=0.85, label="IQR/median (final epoch)"),
    ]
    ax_cv_compare.legend(handles=legend_handles_cv, fontsize=7, loc="upper left")
    add_panel_label(ax_cv_compare, 9, loc=(-0.18, 1.12))

    # ROW 3c (k): Spearman rho per dataset/method
    n_ds, n_mth, w = len(MODELLED_DATASETS), len(TRANSFER_METHODS), 0.18
    x_ds = np.arange(n_ds)
    for mi, m in enumerate(TRANSFER_METHODS):
        st = TRANSFER_MSTYLE[m]
        rhos = [_transfer_spearman(data[ds][m]) for ds in MODELLED_DATASETS]
        offs = (mi - (n_mth - 1) / 2) * w
        ax_coupling.bar(x_ds + offs, rhos, width=w, color=st["color"], label=st["label"],
                       edgecolor="white", lw=0.4, alpha=0.85)
    ax_coupling.axhline(0, color="k", lw=0.8)
    ax_coupling.set_xticks(x_ds)
    ax_coupling.set_xticklabels(xticks, fontsize=9)
    ax_coupling.set_ylabel(r"Spearman $\rho$($\alpha$, SERA)", fontsize=9)
    ax_coupling.set_title(r"$\alpha$-SERA Coupling Strength", fontsize=9, loc="center")
    ax_coupling.legend(fontsize=7, bbox_to_anchor=(0.625, 0), loc="lower center", ncol=1)
    ax_coupling.tick_params(labelsize=8)
    add_panel_label(ax_coupling, 10, loc=(-0.20, 1.12))

    # ROW 3d (l): saturation effect
    for ds in MODELLED_DATASETS:
        ctrl_df = data[ds]["control"]
        if ctrl_df.empty:
            continue
        baseline_alpha = ctrl_df["awareness"].mean()
        for m in TRANSFER_METHODS:
            df = data[ds][m]
            rho = _transfer_spearman(df)
            st = TRANSFER_MSTYLE[m]
            ax_saturation.scatter(baseline_alpha, rho, color=TRANSFER_DS_COLORS[ds],
                                 marker=st["marker"], s=60, alpha=0.85,
                                 edgecolors=st["color"], lw=1.75, zorder=5)

    if not data["perovskites"]["control"].empty:
        pv_alpha = data["perovskites"]["control"]["awareness"].mean()
        pv_rho = _transfer_spearman(data["perovskites"]["control"])
        ax_saturation.annotate(
            "Perovskites\n(saturation)", xy=(pv_alpha, pv_rho),
            xytext=(pv_alpha - 0.08, pv_rho - 0.24), fontsize=7.5, color="#4d4d4d",
            arrowprops=dict(arrowstyle="-|>",
                            connectionstyle="angle,angleA=-90,angleB=180,rad=5",
                            lw=1.5, fc="#e0e0e0"))

    ax_saturation.axhline(0, color="k", lw=0.8, ls="--", alpha=0.3)
    ax_saturation.set_xlabel(r"Control baseline $\bar{\alpha}$", fontsize=9)
    ax_saturation.set_ylabel(r"Spearman $\rho$($\alpha$, SERA)", fontsize=9)
    ax_saturation.set_title("Saturation Effect", fontsize=9, loc="center")
    ax_saturation.tick_params(labelsize=8)
    add_panel_label(ax_saturation, 11, loc=(-0.20, 1.12))

    ds_leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=TRANSFER_DS_COLORS[ds],
                     markersize=7, label=TRANSFER_DS_LABELS[ds])
             for ds in MODELLED_DATASETS]
    m_leg = [Line2D([0], [0], marker=TRANSFER_MSTYLE[m]["marker"], color="grey",
                    markeredgecolor=TRANSFER_MSTYLE[m]["color"], markersize=6, lw=0,
                    label=TRANSFER_MSTYLE[m]["label"])
            for m in TRANSFER_METHODS]
    ax_saturation.legend(handles=ds_leg + m_leg, fontsize=6.5, loc="upper left", ncol=2,
                        frameon=True)

    method_handles_global = [
        Line2D([0], [0], marker=TRANSFER_MSTYLE[m]["marker"], color=TRANSFER_MSTYLE[m]["color"],
              mfc=TRANSFER_MSTYLE[m].get("fill", TRANSFER_MSTYLE[m]["color"]),
              mec=TRANSFER_MSTYLE[m]["color"], ms=7, lw=1.5, label=TRANSFER_MSTYLE[m]["label"])
        for m in TRANSFER_METHODS
    ]
    fig.legend(handles=method_handles_global, loc="upper center", ncols=4, fontsize=8,
              bbox_to_anchor=(0.5, 0.965), frameon=True, columnspacing=1.6, handletextpad=0.5)

    fig.savefig(os.path.join(outdir, "alpha_sera_transfer_analysis.jpg"), dpi=300,
               bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Alpha-SERA transfer mechanism figure")


# ----------------------------------------------------------------------
# SI tables: alpha-SERA coupling (Tables SI3/SI4) and coupling vs Stage-1
# descriptors (Table SI5), ported verbatim from Final_figs.ipynb cell 47.
# Same estimator as the transfer figure's panel (k): Pearson on
# (awareness, log10(SERA)), Spearman/Kendall on (awareness, raw SERA), over
# the full filtered trajectories (filter_outlier_runs) — CV loads from
# experiments/final, MatFold from experiments/matfold, independently of the
# converged-health `predictions` used by Figs 2-4.
# ----------------------------------------------------------------------
def _corr3(df, x="awareness", y="sera"):
    """(Pearson-on-log10(SERA), Spearman, Kendall) between alpha and SERA;
    NaN triple if <10 rows."""
    if df.empty or x not in df.columns or y not in df.columns:
        return (np.nan,) * 3
    d = df.dropna(subset=[x, y])
    d = d[np.isfinite(d[x]) & np.isfinite(d[y]) & (d[y] > 0)]
    if len(d) < 10:
        return (np.nan,) * 3
    return (pearsonr(d[x], np.log10(d[y]))[0], spearmanr(d[x], d[y])[0],
            kendalltau(d[x], d[y])[0])


def make_si_coupling_tables(profile_result, tables_dir):
    cv_dir = os.path.join(REPO, "experiments", "final")
    mf_dir = os.path.join(REPO, "experiments", "matfold")
    if not (os.path.isdir(cv_dir) and os.path.isdir(mf_dir)):
        logger.warning("Need both experiments/final and experiments/matfold — "
                       "skipping SI coupling tables.")
        return

    logger.info("Loading filtered CV trajectory data from %s ...", cv_dir)
    data_cv = _load_transfer_data(cv_dir)
    logger.info("Loading filtered MatFold trajectory data from %s ...", mf_dir)
    data_mf = _load_transfer_data(mf_dir)
    split_data = {"cv": data_cv, "matfold": data_mf}
    ds_tex = {"log_kvrh": r"\texttt{log\_kvrh}", "log_gvrh": r"\texttt{log\_gvrh}",
             "perovskites": r"\texttt{perovskites}", "phonons": r"\texttt{phonons}"}

    rows = []
    for split, data in split_data.items():
        for ds in MODELLED_DATASETS:
            for m in TRANSFER_METHODS:
                p, s, k = _corr3(data[ds][m])
                rows.append(dict(experiment=split, dataset=ds,
                                 method=TRANSFER_MSTYLE[m]["label"],
                                 n_rows=len(data[ds][m]), pearson=p, spearman=s, kendall=k))
            nonempty = [data[ds][m] for m in TRANSFER_METHODS if not data[ds][m].empty]
            pooled = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
            p, s, k = _corr3(pooled)
            rows.append(dict(experiment=split, dataset=ds, method="pooled",
                             n_rows=len(pooled), pearson=p, spearman=s, kendall=k))
    coupling_df = pd.DataFrame(rows)
    coupling_df.to_csv(os.path.join(tables_dir, "alpha_sera_correlation.csv"), index=False)

    def fmt(v):
        return "--" if not np.isfinite(v) else f"${v:.3f}$"

    for split in ("cv", "matfold"):
        lines = []
        for ds in MODELLED_DATASETS:
            sub = coupling_df[(coupling_df.experiment == split) & (coupling_df.dataset == ds)
                              & (coupling_df.method != "pooled")]
            for i, (_, r) in enumerate(sub.iterrows()):
                lead = ds_tex[ds] if i == 0 else " "
                lines.append(f"{lead} & {r.method} & {fmt(r.pearson)} & "
                             f"{fmt(r.spearman)} & {fmt(r.kendall)} \\\\")
            if ds != MODELLED_DATASETS[-1]:
                lines.append(r"\midrule")
        out = os.path.join(tables_dir, f"alpha_sera_correlation_{split}.txt")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Saved -> %s", out)

    logger.info("Method-pooled Spearman rho (quoted in main text Section 4):\n%s",
               coupling_df[coupling_df.method == "pooled"]
               [["experiment", "dataset", "n_rows", "spearman"]].round(3).to_string(index=False))

    # Table SI5: coupling strength vs Stage-1 descriptors. Coupling strength
    # per dataset = mean |Spearman| over methods and both split types;
    # descriptors from the Fig 1 result_df.
    if profile_result is None:
        return
    per_method = coupling_df[coupling_df.method != "pooled"].dropna(subset=["spearman"])
    strength = per_method.groupby("dataset")["spearman"].apply(lambda s: np.abs(s).mean())

    desc_cols = {"$h$": "DIL", "$G$": "Gini", "$D_{KL}$": "KL_Div", "$W_1$": "Wasserstein"}
    prof = profile_result["result_df"].set_index("dataset")
    cvi_rows, tex_lines = [], []
    for tex_name, col in desc_cols.items():
        x = prof.loc[strength.index, col].astype(float)
        r, pval = pearsonr(x, strength.values)
        cvi_rows.append(dict(descriptor=col, pearson_r=r, p_value=pval, n_datasets=len(strength)))
        tex_lines.append(f"{tex_name:9s} & ${r:.3f}$ & ${pval:.2f}$ & {len(strength)} \\\\")
    cvi_df = pd.DataFrame(cvi_rows)
    cvi_df.to_csv(os.path.join(tables_dir, "coupling_vs_imbalance.csv"), index=False)
    with open(os.path.join(tables_dir, "coupling_vs_imbalance_table.txt"), "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    logger.info("Saved -> coupling_vs_imbalance_table.txt")


# ----------------------------------------------------------------------
# Figs 2-4 + tables per experiment set
# ----------------------------------------------------------------------
def make_experiment_figs(expt_dir: str, expt_name: str, outdir: str, tables_dir: str):
    index, predictions = collect_filtered(expt_dir, merge_npz=(expt_name == "final"))
    if index.empty and not predictions:
        logger.warning("No runs under %s — skipping.", expt_dir)
        return None

    # Metric tables (ddof=1, sig-figs)
    per_run = metrics_table(predictions)
    agg = aggregate(per_run)
    formatted_table(agg).to_csv(
        os.path.join(tables_dir, f"metrics_{expt_name}.csv"), index=False)
    with open(os.path.join(tables_dir, f"metrics_{expt_name}.md"), "w") as f:
        f.write(formatted_table(agg).to_markdown(index=False))
    for metric in ("mae", "sera", "alpha"):
        if f"{metric}_mean" in agg.columns and agg["method"].nunique() > 1:
            with open(os.path.join(tables_dir, f"table_{metric}_{expt_name}.tex"), "w") as f:
                f.write(to_latex_bold_best(agg, metric))

    # Fig 2: 6-panel unified training-dynamics analysis + flatness tables
    # (per-dataset basin/phase/correlation figure + LaTeX flatness table)
    make_fig2_dynamics(expt_dir, expt_name, outdir, tables_dir)

    # Fig 3: per-fold aggregated composites (parity + SER + binned MAE), one
    # figure per fold, ported from Final_figs.ipynb cell 24.
    for dataset in MODELLED_DATASETS:
        math_name = MATH_NAMES.get(dataset, dataset)
        by_method = predictions.get(dataset, {})
        n_saved = 0
        for fold in range(5):
            groups = fold_groups(by_method, fold)
            if not groups:
                continue
            fig = summary_plot(groups, target_name=math_name)
            fig.savefig(os.path.join(
                outdir, f"fig3_{expt_name}_{dataset}_fold{fold}_aggregated.jpg"),
                dpi=600, bbox_inches="tight")
            plt.close(fig)
            n_saved += 1
        if n_saved:
            logger.info("%s/%s: %d per-fold composites saved", expt_name, dataset, n_saved)

    # Fig 4: discovery screening (see make_fig4_discovery)
    make_fig4_discovery(predictions, expt_name, outdir, tables_dir)

    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(REPO, "figs", "paper"),
                        help="Output directory (default: figs/paper)")
    parser.add_argument("--experiments", nargs="+",
                        default=["final", "matfold"],
                        help="Experiment sets under experiments/ (default: final matfold)")
    parser.add_argument("--data-dir", default=os.path.join(REPO, "matbench_data"),
                        help="Directory of raw target CSVs for Fig 1")
    parser.add_argument("--profile-dir", default=os.path.join(REPO, "experiments", "profile"),
                        help="Directory of runtime_profile.csv archives for method profiling")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outdir = args.out
    tables_dir = os.path.join(outdir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    use_matimba_style()

    profile_result = make_dataset_profiles(args.data_dir, outdir)

    preds_by_expt = {}
    for expt in args.experiments:
        expt_dir = os.path.join(REPO, "experiments", expt)
        logger.info("=== Experiment set: %s ===", expt)
        predictions = make_experiment_figs(expt_dir, expt, outdir, tables_dir)
        if predictions is not None:
            preds_by_expt[expt] = predictions

    # Fig 5: SERA/alpha sensitivity grid (needs both CV and MatFold)
    make_fig5_sensitivity(preds_by_expt, outdir)

    # Method profiling (runtime / GPU memory)
    make_fig_profiling(args.profile_dir, outdir)

    # Alpha-SERA Transfer Mechanism (12-panel figure, MatFold trajectories)
    if "matfold" in preds_by_expt:
        make_fig_transfer_mechanism(outdir, os.path.join(REPO, "experiments", "matfold"))

    # SI tables: alpha-SERA coupling (SI3/SI4) + coupling vs imbalance (SI5)
    if "final" in preds_by_expt and "matfold" in preds_by_expt:
        make_si_coupling_tables(profile_result, tables_dir)

    logger.info("All outputs written to %s", outdir)


if __name__ == "__main__":
    main()
