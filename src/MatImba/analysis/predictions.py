"""Model-agnostic prediction containers.

The analysis layer consumes plain arrays — targets, predictions and the two
imbalance descriptors relevance phi and density rho — regardless of what model
produced them.  :class:`PredictionSet` wraps one evaluation of one model on one
test set; :class:`EnsemblePrediction` concatenates repeated runs that share a
test set.

Threshold conventions (single source of truth for the whole package):

``SERA_T0 = 0.5``
    Lower bound of the SERA integration interval [t0, 1].  Captures tail
    samples with relevance > 0.5 while excluding near-zero contributions from
    the dense head.  Used for both training-time monitoring and reported
    values.
``TAIL_PHI = 0.8``
    Relevance threshold defining the head/tail partition for tail-MAE tables
    and split parity plots.
``SCREEN_PHI = 0.75``
    Relevance threshold defining screening candidates in virtual
    high-throughput screening (discovery precision/recall).
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from ..utils.losses import calc_alpha, calc_ser_nd, calc_sera

SERA_T0 = 0.5
TAIL_PHI = 0.8
SCREEN_PHI = 0.75


class PredictionSet:
    """Predictions of one model run on one test set, with derived metrics.

    Computes on construction: per-sample absolute errors, MAE, R2, awareness
    alpha (1 - dCor(log-L1 error, 1/density)), SERA integrated over
    [SERA_T0, 1], the SER curve over 50 thresholds, and label-binned MAE
    (Freedman–Diaconis bins).

    Args:
        targets: Ground-truth values, shape (N,).
        preds: Model predictions, shape (N,).
        relevances: Tail relevance phi(y) in [0, 1], or None.
        densities: Local label density rho(y), or None.
        train_log: Optional path to the run's ``*_val_log.csv``.
        name: Optional identifier (e.g. "fold_0_run1").
    """

    def __init__(self, targets, preds, relevances=None, densities=None,
                 train_log: Optional[str] = None, name: Optional[str] = None):
        self.targets = np.ravel(np.asarray(targets, dtype=float))
        self.preds = np.ravel(np.asarray(preds, dtype=float))
        self.relevances = None if relevances is None else np.ravel(np.asarray(relevances, dtype=float))
        self.densities = None if densities is None else np.ravel(np.asarray(densities, dtype=float))
        self.train_log = train_log
        self.name = name

        self.abs_errors = np.abs(self.targets - self.preds)
        self.mae = float(self.abs_errors.mean())
        self.r2_score = float(r2_score(self.targets, self.preds))
        self.alpha = (
            float(calc_alpha(self.targets, self.preds, self.densities).mean().item())
            if self.densities is not None else float("nan")
        )
        self.sera = (
            float(calc_sera(self.targets, self.preds, self.relevances, t=SERA_T0).mean().item())
            if self.relevances is not None else float("nan")
        )
        self._compute_binned_mae()
        self._compute_ser_curve()

    # Backwards-compatible alias used throughout the old analyser code
    @property
    def maes(self):
        return self.abs_errors

    def _compute_binned_mae(self, bins="fd"):
        """Label-binned MAE: histogram of targets + mean |error| per bin."""
        self.hist, self.bin_edges = np.histogram(self.targets, bins=bins)
        self.x = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.nbins = len(self.x)
        self.bin_width = (self.bin_edges[-1] - self.bin_edges[0]) / self.nbins
        label_locs = np.fmin(np.digitize(self.targets, self.bin_edges), self.nbins)
        self.binned_AEs = np.full(self.nbins, np.nan)
        for j in range(self.nbins):
            locs = label_locs == j + 1
            if locs.any():
                self.binned_AEs[j] = self.abs_errors[locs].mean()

    def _compute_ser_curve(self, sampling=50):
        """SER(t) over t in [0, 1]; the area over [SERA_T0, 1] is SERA."""
        self.t_s = np.linspace(0, 1, sampling)
        self.sers = np.full(sampling, np.nan)
        if self.relevances is None:
            return
        for j, t in enumerate(self.t_s):
            self.sers[j] = calc_ser_nd(self.targets, self.preds, self.relevances, t)

    def tail_mask(self, phi: float = TAIL_PHI) -> np.ndarray:
        """Boolean mask of tail samples (relevance > phi)."""
        if self.relevances is None:
            raise ValueError("PredictionSet has no relevance values.")
        return self.relevances > phi

    def tail_mae(self, phi: float = TAIL_PHI) -> float:
        """MAE restricted to tail samples (relevance > phi)."""
        mask = self.tail_mask(phi)
        return float(self.abs_errors[mask].mean()) if mask.any() else float("nan")

    def head_mae(self, phi: float = TAIL_PHI) -> float:
        """MAE restricted to head samples (relevance <= phi)."""
        mask = ~self.tail_mask(phi)
        return float(self.abs_errors[mask].mean()) if mask.any() else float("nan")

    def metrics(self) -> dict:
        """Scalar metric summary of this run."""
        out = {"mae": self.mae, "r2": self.r2_score, "sera": self.sera, "alpha": self.alpha}
        if self.relevances is not None:
            out["tail_mae"] = self.tail_mae()
            out["head_mae"] = self.head_mae()
        return out

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, filepath, train_log=None, name=None) -> "PredictionSet":
        """Load from a trainer ``*_test_predictions.csv``
        (columns: labels, predictions, relevance, density)."""
        df = pd.read_csv(filepath)
        return cls(
            df["labels"].values,
            df["predictions"].values,
            df["relevance"].values if "relevance" in df.columns else None,
            df["density"].values if "density" in df.columns else None,
            train_log=train_log,
            name=name or os.path.basename(filepath).replace("_test_predictions.csv", ""),
        )

    def save(self, filepath):
        """Save arrays and derived metrics to a compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            targets=self.targets,
            preds=self.preds,
            relevances=self.relevances if self.relevances is not None else np.array([np.nan]),
            densities=self.densities if self.densities is not None else np.array([np.nan]),
            has_relevances=np.array(self.relevances is not None),
            has_densities=np.array(self.densities is not None),
            train_log=np.array(self.train_log if self.train_log else "NONE"),
            name=np.array(self.name if self.name else "NONE"),
            r2_score=np.array(self.r2_score),
            alpha=np.array(self.alpha),
            sera=np.array(self.sera),
            hist=self.hist,
            bin_edges=self.bin_edges,
            x=self.x,
            binned_AEs=self.binned_AEs,
            t_s=self.t_s,
            sers=self.sers,
        )

    @classmethod
    def load(cls, filepath) -> "PredictionSet":
        """Load from a .npz produced by :meth:`save` (or the legacy
        ``ml_pred.save`` format), recomputing nothing."""
        data = np.load(filepath, allow_pickle=True)
        inst = cls.__new__(cls)
        inst.targets = data["targets"]
        inst.preds = data["preds"]
        inst.abs_errors = data["maes"] if "maes" in data else np.abs(inst.targets - inst.preds)

        def opt_field(key, has_key):
            if has_key in data:
                return data[key] if bool(data[has_key]) else None
            # legacy format: None stored as object array [None]
            arr = data[key]
            if arr.dtype == object and arr.size == 1 and arr[0] is None:
                return None
            return arr

        inst.relevances = opt_field("relevances", "has_relevances")
        inst.densities = opt_field("densities", "has_densities")
        log_val = str(data["train_log"])
        inst.train_log = None if log_val == "NONE" else log_val
        name_val = str(data["name"]) if "name" in data else "NONE"
        inst.name = None if name_val == "NONE" else name_val

        def scalar(key):
            # legacy files store some scalars with shape (1,), which float()
            # rejects on numpy >= 2
            return float(np.ravel(data[key])[0])

        inst.r2_score = scalar("r2_score")
        inst.alpha = scalar("alpha")
        inst.sera = scalar("sera")
        inst.mae = float(inst.abs_errors.mean())
        inst.hist = data["hist"]
        inst.bin_edges = data["bin_edges"]
        inst.x = data["x"]
        inst.nbins = len(inst.x)
        inst.bin_width = float(data["bin_width"]) if "bin_width" in data else float(
            (inst.bin_edges[-1] - inst.bin_edges[0]) / inst.nbins
        )
        inst.binned_AEs = data["binned_AEs"]
        inst.t_s = data["t_s"]
        inst.sers = data["sers"]
        return inst


class EnsemblePrediction:
    """Concatenation of multiple runs sharing the same test set.

    The binning structure (x, hist, bin_width) is taken from the first run,
    assuming an identical test set across runs.
    """

    def __init__(self, runs: List[PredictionSet]):
        if not runs:
            raise ValueError("EnsemblePrediction needs at least one run.")
        self.runs = list(runs)
        self.targets = np.concatenate([p.targets for p in runs])
        self.preds = np.concatenate([p.preds for p in runs])
        self.abs_errors = np.abs(self.targets - self.preds)
        self.relevances = (
            np.concatenate([p.relevances for p in runs])
            if runs[0].relevances is not None else None
        )
        self.densities = (
            np.concatenate([p.densities for p in runs])
            if runs[0].densities is not None else None
        )
        self.x = runs[0].x
        self.hist = runs[0].hist
        self.bin_width = runs[0].bin_width

    @property
    def maes(self):
        return self.abs_errors

    def per_run(self, attr: str) -> np.ndarray:
        """Array of a scalar metric across runs (e.g. 'sera', 'alpha', 'mae')."""
        return np.array([getattr(p, attr) for p in self.runs], dtype=float)
