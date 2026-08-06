"""Metric and loss sanity tests: calc_sera, calc_alpha, SmoothDILALoss.

Includes the regression test for the dCor clamp bug (Priority 1): gradients
must keep flowing when the model reaches the desired anti-correlated state.
"""

import numpy as np
import pytest
import torch

from MatImba.utils.losses import SmoothDILALoss, calc_alpha, calc_sera, calc_ser_nd


def test_calc_ser_zero_for_perfect_predictions():
    y = torch.linspace(0, 10, 100)
    phi = torch.linspace(0, 1, 100)
    assert calc_ser_nd(y, y, phi, t=0.5).item() == pytest.approx(0.0)
    assert calc_sera(y, y, phi, t=0.5).item() == pytest.approx(0.0)


def test_calc_sera_monotone_in_error():
    torch.manual_seed(0)
    y = torch.linspace(0, 10, 200)
    phi = torch.linspace(0, 1, 200)
    noise = torch.randn(200)
    small = calc_sera(y, y + 0.1 * noise, phi, t=0.5).item()
    large = calc_sera(y, y + 1.0 * noise, phi, t=0.5).item()
    assert 0 < small < large


def test_calc_sera_counts_only_relevant_samples():
    """Errors on zero-relevance (head) samples must not contribute."""
    y = torch.zeros(100)
    preds = torch.zeros(100)
    phi = torch.zeros(100)
    phi[:10] = 1.0
    preds[50:] = 5.0  # large error, but only on phi=0 samples
    assert calc_sera(y, preds, phi, t=0.5).item() == pytest.approx(0.0)


def test_calc_alpha_high_for_density_independent_error():
    """Error independent of density -> alpha near 1."""
    torch.manual_seed(0)
    n = 500
    y = torch.randn(n)
    preds = y + 0.5 * torch.randn(n)  # noise uncorrelated with density
    dens = torch.rand(n) + 0.1
    alpha = calc_alpha(y, preds, dens).item()
    assert alpha > 0.85


def test_calc_alpha_low_for_density_coupled_error():
    """Error that grows exactly with 1/density -> alpha near 0."""
    torch.manual_seed(0)
    n = 500
    y = torch.randn(n)
    dens = torch.rand(n) * 0.9 + 0.1
    preds = y + 1.0 / dens  # error fully determined by inverse density
    alpha = calc_alpha(y, preds, dens).item()
    assert alpha < 0.2


def test_smooth_dila_forward_returns_loss_and_penalty():
    torch.manual_seed(0)
    loss_fn = SmoothDILALoss(base_metric="huber", lambda_dcor=1.0)
    preds = torch.randn(64, 1, requires_grad=True)
    y = torch.randn(64, 1)
    dens = torch.rand(64, 1) + 0.1
    loss, penalty = loss_fn(preds, y, dens)
    assert torch.isfinite(loss)
    loss.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_smooth_dila_gradient_flows_in_anticorrelated_state():
    """Regression test for the P1 clamp bug: when error is anti-correlated
    with inverse density (the desired state, dcov2 driven to its floor), the
    dCor penalty must still propagate a non-degenerate gradient."""
    torch.manual_seed(0)
    n = 128
    dens = torch.rand(n, 1) * 0.9 + 0.1
    y = torch.randn(n, 1)
    # Error largest where density is high -> anti-correlated with 1/density
    preds = (y + dens).clone().requires_grad_(True)
    loss_fn = SmoothDILALoss(base_metric="l1", lambda_dcor=5.0)
    loss, _ = loss_fn(preds, y, dens)
    loss.backward()
    assert torch.isfinite(preds.grad).all()
    assert preds.grad.abs().sum() > 0
