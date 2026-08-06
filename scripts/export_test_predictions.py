"""Backfill trainer-style ``fold_{f}_run{r}_test_predictions.csv`` files.

The analysis data layer (``MatImba.analysis.collect.discover_runs``) indexes
runs by their ``*_test_predictions.csv``; method directories without them are
invisible to it. For every (fold, run) found in a method directory this script
applies a three-tier fallback:

1. ``fold_{f}_run{r}_test_predictions.csv`` already exists  -> skip (never overwritten)
2. ``fold_{f}_run{r}_ml_pred.npz`` exists                   -> convert npz -> csv
3. a checkpoint exists                                       -> run predictions on the
   test set (``evaluate_ckpt``, as in create_ml_pred.py), cache the npz, write the csv

Directories may be method dirs (``experiments/matfold/log_kvrh/bsam``) or
dataset dirs (``experiments/matfold/log_gvrh``) — dataset dirs recurse into
their method subdirectories.

Usage:
    python scripts/export_test_predictions.py experiments/matfold/log_kvrh/bsam
    python scripts/export_test_predictions.py experiments/matfold/log_gvrh
    python scripts/export_test_predictions.py --metric dil experiments/matfold

Tier 3 needs the training environment (torch, configs, matbench data); tiers
1–2 run anywhere with numpy/pandas. The checkpoint metric (default ``dil``,
the robust checkpoint) matches the create_ml_pred.py default that produced
the existing npz caches.
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "MatImba", "src"))
from MatImba.analysis.predictions import PredictionSet  # noqa: E402

# Checkpoint metric suffix, as in create_ml_pred.py:
# '' -> best (MAE), 'dil_' -> robust, 'sera_' -> SERA, 'r2_score_' -> R2
METRIC_SUFFIX = {"mae": "", "dil": "dil_", "sera": "sera_", "r2": "r2_score_"}

FOLD_RUN_RE = re.compile(r"fold_(\d+)_run(\d+)")


def _fold_runs(m_dir):
    """All (fold, run) pairs present in a method dir, from any artefact."""
    pairs = set()
    for path in glob.glob(os.path.join(m_dir, "fold_*_run*")):
        m = FOLD_RUN_RE.search(os.path.basename(path))
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
    return sorted(pairs)


def _config_for(m_dir, config_root):
    """Infer the experiment YAML from experiments/{expt}/{dataset}/{method}."""
    parts = os.path.normpath(os.path.abspath(m_dir)).split(os.sep)
    try:
        i = len(parts) - 1 - parts[::-1].index("experiments")
    except ValueError:
        return None
    if len(parts) < i + 4:
        return None
    expt, dataset, method = parts[i + 1], parts[i + 2], parts[i + 3]
    name = f"{dataset}.yaml" if method == "control" else f"{dataset}_{method}.yaml"
    cfg = os.path.join(config_root, expt, name)
    return cfg if os.path.exists(cfg) else None


def _write_csv(ps, out):
    cols = {"labels": ps.targets.ravel(), "predictions": ps.preds.ravel()}
    if ps.relevances is not None:
        cols["relevance"] = ps.relevances.ravel()
    else:
        print(f"  WARNING {os.path.basename(out)}: no relevances (SERA unavailable)")
    if ps.densities is not None:
        cols["density"] = ps.densities.ravel()
    else:
        print(f"  WARNING {os.path.basename(out)}: no densities (alpha unavailable)")
    pd.DataFrame(cols).to_csv(out, index=False)


def _predict_from_ckpt(m_dir, fold, run, metric, config_root):
    """Tier 3: evaluate the checkpoint and return a PredictionSet (or None)."""
    prefix = f"fold_{fold}_run{run}"
    ckpt = os.path.join(m_dir, f"{prefix}.ckpt.{METRIC_SUFFIX[metric]}best.pth.tar")
    if not os.path.exists(ckpt):
        print(f"  [MISS] {prefix}: no npz and no {metric} checkpoint — skipping")
        return None
    cfg = _config_for(m_dir, config_root)
    if cfg is None:
        print(f"  [MISS] {prefix}: checkpoint found but no config YAML inferred — skipping")
        return None

    # Deterministic evaluation, as in create_ml_pred.py (before torch import).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    from MatImba.analysis import evaluate_ckpt

    print(f"  [EVAL] {prefix} (metric={metric}, config={cfg}) ...")
    trainer = evaluate_ckpt(ckpt_path=ckpt, config_file=cfg, fold=fold, run_id=run)
    targets, preds, relevances, densities = trainer.predict(trainer.test_loader)
    ps = PredictionSet(targets, preds, relevances, densities, name=prefix)
    npz = os.path.join(m_dir, f"{prefix}_ml_pred.npz")
    if not os.path.exists(npz):
        ps.save(npz)  # cache alongside, matching create_ml_pred.py
    return ps


def export_dir(m_dir, metric="dil", config_root="expt_configs"):
    pairs = _fold_runs(m_dir)
    if not pairs:
        print(f"{m_dir}: no run artefacts found")
        return
    written = skipped = predicted = missed = 0
    for fold, run in pairs:
        prefix = f"fold_{fold}_run{run}"
        out = os.path.join(m_dir, f"{prefix}_test_predictions.csv")
        if os.path.exists(out):
            skipped += 1
            continue
        npz = os.path.join(m_dir, f"{prefix}_ml_pred.npz")
        if os.path.exists(npz):
            ps = PredictionSet.load(npz)
        else:
            ps = _predict_from_ckpt(m_dir, fold, run, metric, config_root)
            if ps is None:
                missed += 1
                continue
            predicted += 1
        _write_csv(ps, out)
        written += 1
    print(f"{m_dir}: wrote {written} CSVs ({predicted} via checkpoint prediction), "
          f"skipped {skipped} existing, {missed} unrecoverable")


_SKIP_DIRS = {"hpo_trials", ".ipynb_checkpoints", "__pycache__"}


def _method_dirs(path):
    """Expand a dataset-level dir into its method subdirs; pass method dirs through."""
    if _fold_runs(path):
        return [path]
    subs = [os.path.join(path, d) for d in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, d)) and d not in _SKIP_DIRS
            and not d.startswith(".")]
    found = [s for s in subs if _fold_runs(s)]
    if found:
        return found
    # two levels down (an experiments/{expt} root)
    deeper = []
    for s in subs:
        deeper.extend(_method_dirs(s))
    return deeper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill fold_*_run*_test_predictions.csv from npz caches "
                    "or, failing that, checkpoint predictions.")
    parser.add_argument("dirs", nargs="+",
                        help="Method dirs, dataset dirs, or an experiments/{expt} root")
    parser.add_argument("--metric", default="dil", choices=list(METRIC_SUFFIX),
                        help="Checkpoint selection metric for tier-3 prediction "
                             "(default: dil, the robust checkpoint)")
    parser.add_argument("--config-root", default="expt_configs",
                        help="Root of the experiment YAML configs (default: expt_configs)")
    args = parser.parse_args()

    targets = []
    for d in args.dirs:
        if not os.path.isdir(d):
            sys.exit(f"Not a directory: {d}")
        targets.extend(_method_dirs(d))
    if not targets:
        sys.exit("No run artefacts found under the given directories.")
    for m_dir in targets:
        export_dir(m_dir, metric=args.metric, config_root=args.config_root)
