"""Discovery and loading of experiment outputs.

Walks the ``experiments/{set}/{dataset}/{method}/`` layout produced by the
trainer, pairing every ``fold_{f}_run{r}_test_predictions.csv`` with its
``fold_{f}_run{r}_val_log.csv``, and returns tidy metadata plus loaded
:class:`~MatImba.analysis.predictions.PredictionSet` objects.  All downstream
analyses go through this module so nothing else needs to know the directory
layout.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .predictions import PredictionSet

logger = logging.getLogger(__name__)

_PRED_RE = re.compile(r"fold_(\d+)_run(\d+)_test_predictions\.csv$")


def filter_bad_runs(runs: Sequence[PredictionSet], r2_drop_threshold: float = 0.05
                    ) -> List[PredictionSet]:
    """Drops exploded runs whose R2 falls more than ``r2_drop_threshold``
    below the best run in the group.  Returns the original list if fewer
    than two runs or if everything would be dropped."""
    runs = list(runs)
    if len(runs) < 2:
        return runs
    r2s = [
        p.r2_score if getattr(p, "r2_score", None) is not None
        else r2_score(np.ravel(p.targets), np.ravel(p.preds))
        for p in runs
    ]
    best = max(r2s)
    kept = [p for p, r2 in zip(runs, r2s) if r2 >= best - r2_drop_threshold]
    for p, r2 in zip(runs, r2s):
        if r2 < best - r2_drop_threshold:
            logger.info("Dropped exploded run %s: R2=%.3f (best=%.3f)",
                        getattr(p, "name", "?"), r2, best)
    return kept if kept else runs


def discover_runs(experiments_dir: str,
                  datasets: Optional[Sequence[str]] = None,
                  methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Enumerates prediction CSVs under ``experiments_dir/{dataset}/{method}/``.

    Returns a DataFrame with columns
    ``dataset, method, fold, run, pred_csv, val_log`` (val_log may be None),
    sorted by dataset/method/fold/run.
    """
    rows = []
    if not os.path.isdir(experiments_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiments_dir}")

    for dataset in sorted(os.listdir(experiments_dir)):
        ds_dir = os.path.join(experiments_dir, dataset)
        if not os.path.isdir(ds_dir) or (datasets and dataset not in datasets):
            continue
        for method in sorted(os.listdir(ds_dir)):
            m_dir = os.path.join(ds_dir, method)
            if not os.path.isdir(m_dir) or (methods and method not in methods):
                continue
            for fname in sorted(os.listdir(m_dir)):
                m = _PRED_RE.match(fname)
                if not m:
                    continue
                fold, run = int(m.group(1)), int(m.group(2))
                val_log = os.path.join(m_dir, f"fold_{fold}_run{run}_val_log.csv")
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "fold": fold,
                    "run": run,
                    "pred_csv": os.path.join(m_dir, fname),
                    "val_log": val_log if os.path.exists(val_log) else None,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "method", "fold", "run"]).reset_index(drop=True)
    return df


def load_predictions(index: pd.DataFrame, drop_bad_runs: bool = True
                     ) -> Dict[str, Dict[str, List[PredictionSet]]]:
    """Loads every run in a :func:`discover_runs` index.

    Returns ``{dataset: {method: [PredictionSet, ...]}}``.  When
    ``drop_bad_runs`` is set, exploded runs are filtered via
    :func:`filter_bad_runs` **within each fold** (runs of different folds see
    different test sets, so their R2 values are not comparable).
    """
    out: Dict[str, Dict[str, List[PredictionSet]]] = {}
    for (dataset, method), group in index.groupby(["dataset", "method"]):
        runs = []
        for fold, fold_group in group.groupby("fold"):
            fold_runs = [
                PredictionSet.from_csv(
                    row["pred_csv"],
                    train_log=row["val_log"],
                    name=f"fold_{row['fold']}_run{row['run']}",
                )
                for _, row in fold_group.iterrows()
            ]
            if drop_bad_runs:
                fold_runs = filter_bad_runs(fold_runs)
            runs.extend(fold_runs)
        out.setdefault(dataset, {})[method] = runs
    return out


def collect_experiments(experiments_dir: str,
                        datasets: Optional[Sequence[str]] = None,
                        methods: Optional[Sequence[str]] = None,
                        drop_bad_runs: bool = True):
    """One-call convenience: discover + load.

    Returns ``(index_df, predictions)`` where ``index_df`` is the
    :func:`discover_runs` DataFrame and ``predictions`` the nested dict from
    :func:`load_predictions`.
    """
    index = discover_runs(experiments_dir, datasets, methods)
    if index.empty:
        logger.warning("No prediction CSVs found under %s", experiments_dir)
        return index, {}
    return index, load_predictions(index, drop_bad_runs=drop_bad_runs)


def metrics_table(predictions: Dict[str, Dict[str, List[PredictionSet]]]) -> pd.DataFrame:
    """Per-run scalar metrics as a tidy DataFrame
    (columns: dataset, method, run, mae, r2, sera, alpha, tail_mae, head_mae)."""
    rows = []
    for dataset, by_method in predictions.items():
        for method, runs in by_method.items():
            for p in runs:
                row = {"dataset": dataset, "method": method, "run": p.name}
                row.update(p.metrics())
                rows.append(row)
    return pd.DataFrame(rows)
