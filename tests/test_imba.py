"""Imbalance metric sanity tests: bounds and ordering on synthetic samples."""

import numpy as np
import pytest

from MatImba.dataset.imba import (
    calc_comprehensive_imbalance,
    calc_dil,
    calc_relevance,
    estimate_density,
    get_weights,
)

RNG = np.random.default_rng(0)
UNIFORM = RNG.uniform(0, 1, 5000)
SKEWED = RNG.lognormal(0, 1, 5000)


def test_dil_bounds_and_ordering():
    h_uniform = calc_dil(UNIFORM)
    h_skewed = calc_dil(SKEWED)
    assert 0 <= h_uniform <= 1
    assert 0 <= h_skewed <= 1
    assert h_skewed > h_uniform


def test_comprehensive_metrics_ordering():
    m_uniform = calc_comprehensive_imbalance(UNIFORM)
    m_skewed = calc_comprehensive_imbalance(SKEWED)
    assert set(m_uniform) == {"DIL", "Gini", "KL_Div", "Wasserstein"}
    for key in ("DIL", "Gini", "KL_Div", "Wasserstein"):
        assert m_skewed[key] > m_uniform[key], key
    assert 0 <= m_uniform["Gini"] <= 1
    assert 0 <= m_skewed["Gini"] <= 1


def test_estimate_density_shape_and_ordering():
    dens = estimate_density(SKEWED, smooth="kde")
    assert dens.shape == SKEWED.shape
    assert (dens > 0).all()
    # The lognormal bulk (near the median) must be denser than the far tail
    bulk = np.abs(SKEWED - np.median(SKEWED)) < 0.2
    tail = SKEWED > np.quantile(SKEWED, 0.99)
    assert dens[bulk].mean() > dens[tail].mean()


def test_weights_inverse_to_density():
    dens = estimate_density(SKEWED, smooth="kde")
    w = get_weights(dens, method="log_inv", eps=0.1)
    assert w.shape == SKEWED.shape
    assert w.min() >= 0.1 - 1e-9 and w.max() <= 1.0 + 1e-9
    # Rare samples get larger weights
    assert w[dens < np.quantile(dens, 0.1)].mean() > w[dens > np.quantile(dens, 0.9)].mean()


def test_relevance_bounds_and_tail():
    phi = calc_relevance(SKEWED, plot=False)
    assert phi.shape == SKEWED.shape
    assert phi.min() >= 0 and phi.max() <= 1 + 1e-9
    # Extreme values are more relevant than the median region
    assert phi[SKEWED > np.quantile(SKEWED, 0.995)].mean() > \
        phi[np.abs(SKEWED - np.median(SKEWED)) < 0.1].mean()
