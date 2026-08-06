"""Stage 1 of the MatImba workflow: describe the dataset.

:class:`DatasetProfile` characterises a regression target's distribution
imbalance before any model is trained: local label density rho, tail
relevance phi, and four bounded imbalance metrics —

- ``h`` (normalised Pietra ratio, the paper's Distribution Imbalance Level),
- Gini coefficient of the label histogram,
- ``D_KL`` divergence from the uniform distribution,
- ``W1`` Wasserstein-1 transport cost to the uniform distribution.

:func:`compare_profiles` places several datasets in the common metric space
(table, correlation matrix, PCA biplot), reproducing the paper's Fig. 1
analysis: PC1 aggregates statistical imbalance magnitude, while PC2 contrasts
W1 (sample scarcity / transport cost) against h (manifold fragmentation).
"""

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..dataset.imba import calc_comprehensive_imbalance, calc_relevance, estimate_density
from .predictions import TAIL_PHI


class DatasetProfile:
    """Imbalance profile of one regression target.

    Args:
        y: Target values, shape (N,).
        name: Dataset identifier used in tables and plots.
        bins: Histogram rule for the imbalance metrics (default Freedman–Diaconis).
    """

    def __init__(self, y, name: str = "dataset", bins: str = "fd"):
        self.y = np.ravel(np.asarray(y, dtype=float))
        self.name = name
        self.density = estimate_density(self.y, smooth="kde")
        self.relevance = calc_relevance(self.y, plot=False)
        imba = calc_comprehensive_imbalance(self.y, bins=bins)
        self.h = float(imba["DIL"])
        self.gini = float(imba["Gini"])
        self.kl_div = float(imba["KL_Div"])
        self.wasserstein = float(imba["Wasserstein"])

    def tail_fraction(self, phi: float = TAIL_PHI) -> float:
        """Fraction of samples in the tail (relevance > phi)."""
        return float((self.relevance > phi).mean())

    def metrics(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "n": len(self.y),
            "h": self.h,
            "Gini": self.gini,
            "D_KL": self.kl_div,
            "W1": self.wasserstein,
            "tail_fraction": self.tail_fraction(),
        }

    def plot(self, ax=None, bins: int = 60):
        """Label distribution with relevance phi and density rho overlays."""
        if ax is None:
            _, ax = plt.subplots(figsize=(3.4, 2.8), layout="compressed")
        ax.hist(self.y, bins=bins, color="#bdbdbd", edgecolor="none", density=True,
                label="label density")
        order = np.argsort(self.y)
        ax_r = ax.twinx()
        ax_r.plot(self.y[order], self.relevance[order], color="#b2182b", lw=1.5,
                  label=r"relevance $\phi$")
        ax_r.set_ylabel(r"$\phi(y)$", fontsize=10)
        ax_r.set_ylim(-0.02, 1.05)
        ax.set_xlabel(self.name, fontsize=10)
        ax.set_ylabel("density", fontsize=10)
        ax.set_title(
            f"h={self.h:.2f}  G={self.gini:.2f}  "
            f"$D_{{KL}}$={self.kl_div:.2f}  $W_1$={self.wasserstein:.2f}",
            fontsize=9,
        )
        return ax


def profiles_table(profiles: Sequence[DatasetProfile]) -> pd.DataFrame:
    """Tidy table of imbalance metrics, one row per dataset."""
    return pd.DataFrame([p.metrics() for p in profiles]).set_index("name")


def compare_profiles(profiles: Sequence[DatasetProfile],
                     pca_ax=None,
                     annotate: bool = True):
    """Cross-dataset imbalance comparison.

    Returns a dict with:
        ``table`` — :func:`profiles_table` DataFrame,
        ``correlation`` — Pearson correlation matrix of the four metrics,
        ``pca`` — dict with loadings, explained variance ratios and scores.

    When ``pca_ax`` is given (or created), draws a PCA biplot of the datasets
    in imbalance-metric space with metric loading arrows.
    """
    table = profiles_table(profiles)
    metric_cols = ["h", "Gini", "D_KL", "W1"]
    X = table[metric_cols].to_numpy(dtype=float)
    corr = table[metric_cols].corr()

    # PCA on standardised metrics
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    scores = U * S
    explained = S**2 / np.sum(S**2)
    loadings = Vt.T  # (metrics, components)

    # SVD signs are arbitrary: orient each PC so h loads positively
    # (higher imbalance -> positive score), for stable interpretation
    for k in range(loadings.shape[1]):
        if loadings[0, k] < 0:
            loadings[:, k] *= -1
            scores[:, k] *= -1

    pca = {
        "scores": scores[:, :2],
        "loadings": pd.DataFrame(loadings[:, :2], index=metric_cols, columns=["PC1", "PC2"]),
        "explained_variance_ratio": explained[:2],
        "datasets": list(table.index),
    }

    if pca_ax is not None:
        _draw_pca_biplot(pca, pca_ax, annotate=annotate)

    return {"table": table, "correlation": corr, "pca": pca}


def _draw_pca_biplot(pca: dict, ax, annotate: bool = True):
    scores = pca["scores"]
    loadings = pca["loadings"]
    evr = pca["explained_variance_ratio"]

    ax.scatter(scores[:, 0], scores[:, 1], s=45, color="#4575b4",
               edgecolor="#2d2d2d", linewidth=0.5, zorder=3)
    if annotate:
        # Small offsets reduce label overlap (Referee 1 / Referee 3 issue 7)
        for (x, y), label in zip(scores, pca["datasets"]):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=8)

    scale = 0.9 * np.abs(scores[:, :2]).max() / (np.abs(loadings.values).max() + 1e-12)
    for metric, (lx, ly) in loadings.iterrows():
        ax.annotate(
            "", xy=(lx * scale, ly * scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.4),
        )
        label = {"h": "$h$", "Gini": "$G$", "D_KL": "$D_{KL}$", "W1": "$W_1$"}[metric]
        ax.annotate(label, (lx * scale, ly * scale), xytext=(3, 3),
                    textcoords="offset points", fontsize=9, color="#b2182b")

    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var.)", fontsize=10)
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var.)", fontsize=10)
    return ax
