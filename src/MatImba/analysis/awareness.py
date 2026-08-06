"""Stage 3c of the MatImba workflow: SERA-alpha coupling diagnosis.

The central diagnostic of the package: is tail error (SERA) reducible by
raising distribution awareness (alpha = 1 - dCor(error, 1/density)), or is it
fixed by the dataset itself?

For each dataset x method x split regime this module quantifies the coupling
between per-epoch awareness and SERA along training trajectories
(``*_val_log.csv``) and between final per-run values, classifies the coupling
regime, and relates coupling strength to the dataset imbalance descriptors
(h, Gini, D_KL, W1) from :mod:`~MatImba.analysis.dataset_profile`:

- ``linear``      — SERA decreases monotonically with alpha (e.g. log_kvrh,
                    log_gvrh: Spearman rho ~ -0.75 .. -0.82).  Awareness buys
                    tail accuracy; DILA is an effective, cheap mitigation.
- ``thresholded`` — coupling only below a breakpoint (phonons: alpha < 0.8).
                    Gains saturate once awareness is high.
- ``decoupled``   — no significant dependence (perovskites under MatFold
                    OOD: rho ~ -0.03).  The tail-error floor is set by the
                    dataset geometry (transport cost W1), not by training;
                    no awareness-based method can lower it.
"""

import logging
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .predictions import PredictionSet

logger = logging.getLogger(__name__)

# Epochs to skip at the start of each trajectory (optimisation burn-in, both
# metrics still dominated by the initial transient).
TRAJECTORY_BURN_IN = 25
# |Spearman rho| below which (or p-value above 0.05) coupling is 'decoupled'.
DECOUPLED_RHO = 0.2
# Minimum R2 gain of the hinge fit over the linear fit to call 'thresholded'.
HINGE_R2_GAIN = 0.1


# ----------------------------------------------------------------------
# Correlation with bootstrap CIs
# ----------------------------------------------------------------------
def alpha_sera_correlation(alpha: np.ndarray, sera: np.ndarray,
                           n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman and Pearson correlation between alpha and SERA with
    bootstrap 95% confidence intervals.

    SERA spans orders of magnitude, so the Pearson correlation is computed on
    log10(SERA); Spearman is rank-based and unaffected.
    """
    alpha = np.ravel(np.asarray(alpha, dtype=float))
    sera = np.ravel(np.asarray(sera, dtype=float))
    mask = np.isfinite(alpha) & np.isfinite(sera) & (sera > 0)
    alpha, sera = alpha[mask], sera[mask]
    n = len(alpha)
    if n < 5:
        return {"n": n, "spearman": np.nan, "spearman_p": np.nan,
                "spearman_ci": (np.nan, np.nan), "pearson": np.nan,
                "pearson_p": np.nan, "pearson_ci": (np.nan, np.nan)}

    log_sera = np.log10(sera)
    rho, rho_p = stats.spearmanr(alpha, sera)
    r, r_p = stats.pearsonr(alpha, log_sera)

    rng = np.random.default_rng(seed)
    boot_rho, boot_r = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, s, ls = alpha[idx], sera[idx], log_sera[idx]
        if np.ptp(a) == 0 or np.ptp(s) == 0:
            continue
        boot_rho.append(stats.spearmanr(a, s)[0])
        boot_r.append(stats.pearsonr(a, ls)[0])

    def ci(vals):
        return ((float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
                if vals else (np.nan, np.nan))

    return {
        "n": n,
        "spearman": float(rho), "spearman_p": float(rho_p), "spearman_ci": ci(boot_rho),
        "pearson": float(r), "pearson_p": float(r_p), "pearson_ci": ci(boot_r),
    }


# ----------------------------------------------------------------------
# Regime classification
# ----------------------------------------------------------------------
def _linear_r2(x, y):
    if np.ptp(x) == 0:
        return 0.0, (0.0, y.mean())
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return (1 - ss_res / ss_tot if ss_tot > 0 else 0.0), (slope, intercept)


def _hinge_r2(x, y, breakpoints):
    """Best hinge fit y = a + b*max(0, c - x) over a grid of breakpoints c:
    coupling below the breakpoint, flat above it."""
    best = (0.0, None)
    for c in breakpoints:
        feat = np.maximum(0.0, c - x)
        if np.ptp(feat) == 0:
            continue
        b, a = np.polyfit(feat, y, 1)
        pred = a + b * feat
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 > best[0]:
            best = (r2, {"breakpoint": float(c), "slope": float(b), "intercept": float(a)})
    return best


def classify_regime(alpha: np.ndarray, sera: np.ndarray) -> dict:
    """Classifies the alpha-SERA coupling as linear / thresholded / decoupled.

    Fits log10(SERA) against alpha with (i) a straight line and (ii) a hinge
    (active only below a breakpoint, grid over the alpha range).  Decoupled if
    the Spearman correlation is weak or insignificant; thresholded if the
    hinge fit beats the linear fit by more than ``HINGE_R2_GAIN``.
    """
    alpha = np.ravel(np.asarray(alpha, dtype=float))
    sera = np.ravel(np.asarray(sera, dtype=float))
    mask = np.isfinite(alpha) & np.isfinite(sera) & (sera > 0)
    alpha, sera = alpha[mask], sera[mask]
    corr = alpha_sera_correlation(alpha, sera)

    if corr["n"] < 10:
        return {"regime": "insufficient-data", **corr}

    log_sera = np.log10(sera)
    lin_r2, (slope, intercept) = _linear_r2(alpha, log_sera)
    grid = np.quantile(alpha, np.linspace(0.2, 0.95, 16))
    hinge_r2, hinge_fit = _hinge_r2(alpha, log_sera, np.unique(grid))

    if abs(corr["spearman"]) < DECOUPLED_RHO or corr["spearman_p"] > 0.05:
        regime = "decoupled"
    elif hinge_fit is not None and hinge_r2 - lin_r2 > HINGE_R2_GAIN:
        regime = "thresholded"
    else:
        regime = "linear"

    return {
        "regime": regime,
        "linear_r2": float(lin_r2), "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "hinge_r2": float(hinge_r2), "hinge_fit": hinge_fit,
        **corr,
    }


# ----------------------------------------------------------------------
# Data sources: trajectories and final runs
# ----------------------------------------------------------------------
def load_trajectories(index: pd.DataFrame, burn_in: int = TRAJECTORY_BURN_IN
                      ) -> pd.DataFrame:
    """Stacks per-epoch (awareness, sera, mae, r2_score) from every val_log in
    a :func:`~MatImba.analysis.collect.discover_runs` index, skipping the
    first ``burn_in`` epochs of each run."""
    frames = []
    for _, row in index.iterrows():
        if row["val_log"] is None:
            continue
        try:
            df = pd.read_csv(row["val_log"])
        except Exception as e:
            logger.warning("Unreadable val_log %s: %s", row["val_log"], e)
            continue
        df = df.iloc[burn_in:].copy()
        df["dataset"] = row["dataset"]
        df["method"] = row["method"]
        df["fold"] = row["fold"]
        df["run"] = row["run"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def coupling_table(trajectories: pd.DataFrame,
                   by=("dataset", "method")) -> pd.DataFrame:
    """Alpha-SERA coupling per group of trajectories (default: dataset x
    method), regenerating the SI rank-correlation tables.

    Returns one row per group with Spearman/Pearson correlations, bootstrap
    CIs and the classified regime.
    """
    rows = []
    for keys, g in trajectories.groupby(list(by)):
        res = classify_regime(g["awareness"].values, g["sera"].values)
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n": res["n"], "regime": res["regime"],
            "spearman": res["spearman"], "spearman_p": res["spearman_p"],
            "spearman_lo": res["spearman_ci"][0], "spearman_hi": res["spearman_ci"][1],
            "pearson": res["pearson"], "pearson_p": res["pearson_p"],
            "pearson_lo": res["pearson_ci"][0], "pearson_hi": res["pearson_ci"][1],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def final_coupling_table(predictions: Dict[str, Dict[str, List[PredictionSet]]]
                         ) -> pd.DataFrame:
    """Alpha-SERA correlation over final per-run values (pooled across
    methods within each dataset).  Complements :func:`coupling_table`; needs
    enough runs per dataset to be meaningful."""
    rows = []
    for dataset, by_method in predictions.items():
        alpha = np.array([p.alpha for runs in by_method.values() for p in runs])
        sera = np.array([p.sera for runs in by_method.values() for p in runs])
        res = alpha_sera_correlation(alpha, sera)
        rows.append({"dataset": dataset, "n": res["n"],
                     "spearman": res["spearman"], "spearman_p": res["spearman_p"],
                     "spearman_lo": res["spearman_ci"][0],
                     "spearman_hi": res["spearman_ci"][1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Cross-dataset: coupling strength vs imbalance descriptors
# ----------------------------------------------------------------------
def coupling_vs_imbalance(coupling: pd.DataFrame, profile_table: pd.DataFrame
                          ) -> pd.DataFrame:
    """Relates coupling strength |Spearman rho| to the dataset imbalance
    descriptors — the quantitative evidence that the challenge is
    dataset-intrinsic.

    ``coupling``: output of :func:`coupling_table` (uses the mean |rho| per
    dataset).  ``profile_table``: output of
    :func:`~MatImba.analysis.dataset_profile.profiles_table` indexed by the
    same dataset names.  Returns one row per descriptor with the Pearson
    correlation between descriptor and coupling strength across datasets.
    """
    strength = coupling.groupby("dataset")["spearman"].apply(
        lambda s: float(np.nanmean(np.abs(s)))
    )
    joined = profile_table.join(strength.rename("coupling_strength"), how="inner")
    if len(joined) < 3:
        logger.warning("Only %d datasets in common — correlation is indicative only.",
                       len(joined))
    rows = []
    for col in ["h", "Gini", "D_KL", "W1"]:
        if col not in joined.columns:
            continue
        x = joined[col].values.astype(float)
        y = joined["coupling_strength"].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3 and np.ptp(x[mask]) > 0:
            r, p = stats.pearsonr(x[mask], y[mask])
        else:
            r, p = np.nan, np.nan
        rows.append({"descriptor": col, "pearson_r": r, "p_value": p,
                     "n_datasets": int(mask.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_alpha_sera(trajectories: pd.DataFrame, dataset: str, ax=None,
                    color_by: str = "method"):
    """Alpha-SERA scatter for one dataset with the classified regime fit
    (producer of the paper's quantitative alpha-SERA figure)."""
    g = trajectories[trajectories["dataset"] == dataset]
    if g.empty:
        raise ValueError(f"No trajectories for dataset {dataset!r}.")
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 2.8), layout="compressed")

    colors = ["#2166ac", "#b2182b", "#35978f", "#ff7f0e", "#4d4d4d"]
    for i, (label, sub) in enumerate(g.groupby(color_by)):
        ax.scatter(sub["awareness"], sub["sera"], s=8, alpha=0.35,
                   color=colors[i % len(colors)], label=str(label), linewidths=0)

    res = classify_regime(g["awareness"].values, g["sera"].values)
    if res["regime"] not in ("insufficient-data",):
        xs = np.linspace(g["awareness"].min(), g["awareness"].max(), 100)
        if res["regime"] == "thresholded" and res.get("hinge_fit"):
            hf = res["hinge_fit"]
            ys = 10 ** (hf["intercept"] + hf["slope"] * np.maximum(0, hf["breakpoint"] - xs))
            fit_label = f"hinge (bp={hf['breakpoint']:.2f})"
        else:
            ys = 10 ** (res["linear_intercept"] + res["linear_slope"] * xs)
            fit_label = "linear fit"
        if res["regime"] != "decoupled":
            ax.plot(xs, ys, "k--", lw=1.5, label=fit_label)

    ax.set_yscale("log")
    ax.set_xlabel(r"awareness $\alpha$", fontsize=10)
    ax.set_ylabel("SERA", fontsize=10)
    ax.set_title(
        rf"{dataset}: $\rho_s$={res['spearman']:.2f} ({res['regime']})", fontsize=9)
    ax.legend(fontsize=7, loc="best", handletextpad=0.4, borderpad=0.2,
              labelspacing=0.4)
    return ax


def plot_trajectory_phase(val_log: str, skip: int = TRAJECTORY_BURN_IN, ax=None,
                          x: str = "awareness", y: str = "r2_score"):
    """Training-trajectory phase plot (e.g. awareness vs R2, coloured by
    epoch) from one val_log CSV — paper Fig. 2c/f style."""
    df = pd.read_csv(val_log)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 2.8), layout="compressed")
    sc = ax.scatter(df[x][skip:], df[y][skip:], c=df["epoch"][skip:], cmap="RdYlBu", s=12)
    plt.colorbar(sc, ax=ax, label="Epoch")
    labels = {"awareness": r"awareness $\alpha$", "r2_score": "$R^2$", "sera": "SERA",
              "mae": "MAE"}
    ax.set_xlabel(labels.get(x, x), fontsize=10)
    ax.set_ylabel(labels.get(y, y), fontsize=10)
    return ax
