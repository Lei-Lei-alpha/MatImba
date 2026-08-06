"""Analysis-layer tests: PredictionSet round-trips, ddof=1 statistics,
sig-fig formatting, run collection and alpha-SERA regime classification."""

import os

import numpy as np
import pandas as pd
import pytest

from MatImba.analysis import (
    DatasetProfile,
    PredictionSet,
    aggregate,
    alpha_sera_correlation,
    classify_regime,
    collect_experiments,
    filter_bad_runs,
    format_pm,
    metrics_table,
)

RNG = np.random.default_rng(0)


def make_pred(n=400, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.lognormal(0, 0.8, n)
    preds = y + noise * rng.normal(size=n)
    dens = np.exp(-0.5 * ((np.log(y)) / 0.8) ** 2) + 0.05
    phi = (y - y.min()) / (y.max() - y.min())
    return PredictionSet(y, preds, phi, dens, name="synthetic")


# ----------------------------------------------------------------------
# PredictionSet
# ----------------------------------------------------------------------
def test_prediction_set_metrics_finite():
    p = make_pred()
    m = p.metrics()
    for key in ("mae", "r2", "sera", "alpha", "tail_mae", "head_mae"):
        assert np.isfinite(m[key]), key
    assert 0 <= m["alpha"] <= 1


def test_prediction_set_csv_roundtrip(tmp_path):
    p = make_pred()
    csv = tmp_path / "fold_0_run0_test_predictions.csv"
    pd.DataFrame({
        "labels": p.targets, "predictions": p.preds,
        "relevance": p.relevances, "density": p.densities,
    }).to_csv(csv, index=False)
    q = PredictionSet.from_csv(str(csv))
    assert q.mae == pytest.approx(p.mae)
    assert q.sera == pytest.approx(p.sera)
    assert q.alpha == pytest.approx(p.alpha)
    assert q.name == "fold_0_run0"


def test_prediction_set_npz_roundtrip(tmp_path):
    p = make_pred()
    path = str(tmp_path / "pred.npz")
    p.save(path)
    q = PredictionSet.load(path)
    np.testing.assert_allclose(q.targets, p.targets)
    assert q.sera == pytest.approx(p.sera)
    assert q.alpha == pytest.approx(p.alpha)
    np.testing.assert_allclose(q.sers, p.sers)


# ----------------------------------------------------------------------
# Statistics hygiene
# ----------------------------------------------------------------------
def test_aggregate_uses_sample_std():
    per_run = pd.DataFrame({
        "dataset": ["d"] * 5, "method": ["m"] * 5,
        "mae": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    agg = aggregate(per_run)
    assert agg.loc[0, "mae_mean"] == pytest.approx(3.0)
    # ddof=1 for n=5: std = sqrt(10/4), NOT the population sqrt(10/5)
    assert agg.loc[0, "mae_std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert agg.loc[0, "mae_count"] == 5


@pytest.mark.parametrize("mean,std,expected", [
    (4.108, 1.739, "4.1 ± 1.7"),      # leading digit 1 -> two sig figs
    (8.557, 10.807, "9 ± 11"),
    (5.618, 3.021, "6 ± 3"),
    (0.912, 0.0123, "0.912 ± 0.012"),
    (708099, 300000, "700000 ± 300000"),
    (0.037, 0.0004, "0.0370 ± 0.0004"),
])
def test_format_pm(mean, std, expected):
    assert format_pm(mean, std) == expected


def test_format_pm_scientific_for_tiny_values():
    out = format_pm(3.7e-4, 4e-6)
    assert "e" in out and "±" in out


# ----------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------
def test_filter_bad_runs_drops_exploded():
    good = [make_pred(noise=0.3, seed=s) for s in range(3)]
    bad = make_pred(noise=3.0, seed=9)
    kept = filter_bad_runs(good + [bad])
    assert bad not in kept
    assert all(g in kept for g in good)


def test_collect_experiments_walks_layout(tmp_path):
    expdir = tmp_path / "expt"
    for fold in (0, 1):
        d = expdir / "mydata" / "mymethod"
        os.makedirs(d, exist_ok=True)
        p = make_pred(seed=fold)
        pd.DataFrame({
            "labels": p.targets, "predictions": p.preds,
            "relevance": p.relevances, "density": p.densities,
        }).to_csv(d / f"fold_{fold}_run0_test_predictions.csv", index=False)
        pd.DataFrame({
            "epoch": range(30), "mae": np.linspace(1, 0.5, 30),
            "sera": np.linspace(10, 5, 30), "scaled_error": 0.5,
            "r2_score": np.linspace(0, 0.8, 30),
            "awareness": np.linspace(0.5, 0.9, 30),
            "robust_score": 1.0, "penalty": 0.0,
        }).to_csv(d / f"fold_{fold}_run0_val_log.csv", index=False)

    index, preds = collect_experiments(str(expdir))
    assert len(index) == 2
    assert index["val_log"].notna().all()
    table = metrics_table(preds)
    assert set(table["dataset"]) == {"mydata"}
    assert len(table) == 2


# ----------------------------------------------------------------------
# Alpha-SERA coupling
# ----------------------------------------------------------------------
def test_alpha_sera_correlation_sign():
    alpha = RNG.uniform(0.3, 0.95, 300)
    sera = 10 ** (2 - 2 * alpha) * np.exp(0.1 * RNG.normal(size=300))
    res = alpha_sera_correlation(alpha, sera)
    assert res["spearman"] < -0.9
    assert res["spearman_ci"][0] <= res["spearman"] <= res["spearman_ci"][1]


def test_classify_regime_linear():
    alpha = RNG.uniform(0.3, 0.95, 400)
    sera = 10 ** (2 - 2 * alpha) * np.exp(0.05 * RNG.normal(size=400))
    assert classify_regime(alpha, sera)["regime"] == "linear"


def test_classify_regime_decoupled():
    alpha = RNG.uniform(0.3, 0.95, 400)
    sera = 10 ** RNG.normal(size=400)
    assert classify_regime(alpha, sera)["regime"] == "decoupled"


def test_classify_regime_thresholded():
    alpha = RNG.uniform(0.3, 1.0, 600)
    # Strong coupling below 0.7, flat above
    log_sera = np.where(alpha < 0.7, 3 * (0.7 - alpha), 0.0) + 0.03 * RNG.normal(size=600)
    res = classify_regime(alpha, 10 ** log_sera)
    assert res["regime"] == "thresholded"
    assert res["hinge_fit"]["breakpoint"] == pytest.approx(0.7, abs=0.15)


# ----------------------------------------------------------------------
# Dataset profile
# ----------------------------------------------------------------------
def test_dataset_profile_ordering():
    uniform = DatasetProfile(RNG.uniform(0, 1, 4000), name="uniform")
    skewed = DatasetProfile(RNG.lognormal(0, 1, 4000), name="skewed")
    assert skewed.h > uniform.h
    assert skewed.wasserstein > uniform.wasserstein
    assert 0 <= skewed.tail_fraction() < 0.5
