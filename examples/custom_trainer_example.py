"""Use MatImba's robust training with your own PyTorch model.

This example trains a plain MLP on a synthetic, heavily right-skewed
regression target — the typical shape of materials property distributions.
It shows the complete recipe for reusing MatImba's robust training method
(DILA loss, EMA-smoothed robust model selection, tail-error resistant
checkpointing) with any model and dataloader:

1. Compute the imbalance statistics on the training labels:
   density rho (``estimate_density``), sample weights omega (``get_weights``)
   and tail relevance phi (``calc_relevance``).
2. Put them in your batches (here: extra tensors in a ``TensorDataset``).
3. Subclass ``BaseRobustTrainer`` and implement ``forward_batch`` and
   ``unpack_batch``.

Run on CPU in under a minute:

    python examples/custom_trainer_example.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from MatImba.base_trainer import BaseRobustTrainer, BatchFields
from MatImba.dataset.imba import calc_relevance, estimate_density, get_weights


# ----------------------------------------------------------------------
# 1. Synthetic imbalanced regression problem
# ----------------------------------------------------------------------
def make_skewed_dataset(n: int = 2000, seed: int = 0):
    """y = ||x||-driven log-normal target: dense head, sparse high-value tail."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)).astype(np.float32)
    y = np.exp(0.8 * x[:, 0] + 0.3 * rng.normal(size=n)).astype(np.float32)
    return x, y


def build_loader(x, y, density, weights, relevance, batch_size=128, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(weights.astype(np.float32)),
        torch.from_numpy(density.astype(np.float32)),
        torch.from_numpy(relevance.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ----------------------------------------------------------------------
# 2. Trainer subclass: the only MatImba-specific code you need to write
# ----------------------------------------------------------------------
class MLPTrainer(BaseRobustTrainer):
    def move_batch(self, batch):
        # TensorDataset batches are tuples, not objects with .to()
        return [t.to(self.device, non_blocking=True) for t in batch]

    def forward_batch(self, batch):
        x = batch[0]
        return self.model(x)

    def unpack_batch(self, batch):
        _, y, omega, rou, phi = batch
        return BatchFields(y=y, weights=omega, density=rou, relevance=phi)


def main():
    torch.manual_seed(0)

    x, y = make_skewed_dataset()
    n_train, n_val = 1400, 300

    # Imbalance statistics computed on the full label set (as in the paper's
    # pipeline); in a strict OOD protocol compute them on the training labels only.
    density = estimate_density(y, smooth="kde")
    weights = get_weights(density, method="log_inv", eps=0.1)
    relevance = calc_relevance(y, plot=False)

    train_loader = build_loader(x[:n_train], y[:n_train], density[:n_train],
                                weights[:n_train], relevance[:n_train], shuffle=True)
    val_loader = build_loader(x[n_train:n_train + n_val], y[n_train:n_train + n_val],
                              density[n_train:n_train + n_val],
                              weights[n_train:n_train + n_val],
                              relevance[n_train:n_train + n_val])
    test_loader = build_loader(x[n_train + n_val:], y[n_train + n_val:],
                               density[n_train + n_val:], weights[n_train + n_val:],
                               relevance[n_train + n_val:])

    model = nn.Sequential(
        nn.Linear(8, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 1),
    )

    trainer = MLPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimiser=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler_type="CosineAnnealingLR",
        epoch_range=30,
        dil_inform=True,                      # enable the DILA objective
        dil_config={"lambda": 1.0, "base_metric": "huber", "warmup_epochs": 5},
        name="mlp_dila_example",
        outdir="mlp_dila_example_out",
    )

    results = trainer.fit()
    print("\nTest metrics (best robust checkpoint):")
    for k, v in results.items():
        print(f"  {k:>20s} = {v:.4f}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
