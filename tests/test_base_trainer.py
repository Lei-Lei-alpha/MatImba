"""BaseRobustTrainer works with a non-graph model and plain tuple batches.

This is the proof that the robust training method is usable beyond MEGNet:
a 3-epoch fit of an MLP on tuple batches must produce the same artefacts
(val_log CSV with the full column set, checkpoint flavors, test predictions)
as the production graph trainer.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from MatImba.base_trainer import BaseRobustTrainer, BatchFields
from MatImba.trainer import BSAM, CgcnnTrainer, LossExplosionError

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
sys.path.insert(0, EXAMPLES_DIR)
from custom_trainer_example import MLPTrainer, build_loader, make_skewed_dataset  # noqa: E402

from MatImba.dataset.imba import calc_relevance, estimate_density, get_weights  # noqa: E402


@pytest.fixture(scope="module")
def fitted_trainer(tmp_path_factory):
    torch.manual_seed(0)
    outdir = str(tmp_path_factory.mktemp("trainer_out"))

    x, y = make_skewed_dataset(n=400, seed=0)
    density = estimate_density(y, smooth="kde")
    weights = get_weights(density, method="log_inv", eps=0.1)
    relevance = calc_relevance(y, plot=False)

    def loader(sl, shuffle=False):
        return build_loader(x[sl], y[sl], density[sl], weights[sl], relevance[sl],
                            batch_size=64, shuffle=shuffle)

    model = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1))
    trainer = MLPTrainer(
        model=model,
        train_loader=loader(slice(0, 280), shuffle=True),
        val_loader=loader(slice(280, 340)),
        test_loader=loader(slice(340, 400)),
        optimiser=torch.optim.AdamW(model.parameters(), lr=1e-2),
        scheduler_type="CosineAnnealingLR",
        epoch_range=40,
        dil_inform=True,
        dil_config={"lambda": 1.0, "base_metric": "huber", "warmup_epochs": 2},
        name="mlp_test",
        outdir=outdir,
    )
    results = trainer.fit()
    return trainer, results, outdir


def test_fit_returns_finite_test_metrics(fitted_trainer):
    _, results, _ = fitted_trainer
    expected = {"test_mse", "test_mae", "test_esr", "test_sera",
                "test_scaled_error", "test_r2", "test_awareness"}
    assert expected == set(results)
    for key, value in results.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"


def test_val_log_written_with_full_column_set(fitted_trainer):
    trainer, _, outdir = fitted_trainer
    log = pd.read_csv(os.path.join(outdir, f"{trainer.name}_val_log.csv"))
    assert list(log.columns) == [
        "epoch", "mae", "sera", "scaled_error", "r2_score",
        "awareness", "robust_score", "penalty",
    ]
    assert len(log) == 40
    assert np.isfinite(log["robust_score"]).all()


def test_checkpoints_and_predictions_saved(fitted_trainer):
    trainer, _, outdir = fitted_trainer
    files = os.listdir(outdir)
    # save_checkpoint writes {name}.ckpt.pth.tar plus best-flavor copies
    assert f"{trainer.name}.ckpt.pth.tar" in files
    assert f"{trainer.name}.ckpt.best.pth.tar" in files
    pred_path = os.path.join(outdir, f"{trainer.name}_test_predictions.csv")
    assert os.path.exists(pred_path)
    preds = pd.read_csv(pred_path)
    assert list(preds.columns) == ["labels", "predictions", "relevance", "density"]
    assert len(preds) == 60


def test_robust_state_selected(fitted_trainer):
    trainer, _, _ = fitted_trainer
    assert trainer.best_robust_state is not None
    assert np.isfinite(trainer.best_robust_score)


def test_unpack_batch_guards():
    """weighted_loss/dil_inform without the matching field raise a clear error."""

    class NoFieldsTrainer(BaseRobustTrainer):
        def move_batch(self, batch):
            return [t for t in batch]

        def forward_batch(self, batch):
            return self.model(batch[0])

        def unpack_batch(self, batch):
            return BatchFields(y=batch[1])

    x = torch.randn(32, 8)
    y = torch.randn(32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=16
    )
    model = nn.Linear(8, 1)
    trainer = NoFieldsTrainer(
        model=model, train_loader=loader, val_loader=loader,
        epoch_range=1, dil_inform=True, save_checkpoints=False,
        name="guard_test",
    )
    with pytest.raises(ValueError, match="density"):
        trainer.train(epoch=0)


def test_cgcnn_trainer_is_subclass():
    assert issubclass(CgcnnTrainer, BaseRobustTrainer)
    assert issubclass(BSAM, torch.optim.Optimizer)
    assert issubclass(LossExplosionError, Exception)
