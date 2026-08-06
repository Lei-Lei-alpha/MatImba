"""Graph-network trainer for MatImba (MEGNet / CGCNN-style PyG batches).

All generic robust-training logic (DILA loss wiring, EMA-smoothed robust
score, tail-error resistant checkpointing, CSV logging) lives in
:class:`MatImba.base_trainer.BaseRobustTrainer`.  This module keeps only the
MEGNet-specific batch handling and re-exports the public names that existing
scripts import from ``MatImba.trainer`` (``CgcnnTrainer``, ``BSAM``,
``LossExplosionError``).
"""

import torch

from .base_trainer import (
    BSAM,
    BaseRobustTrainer,
    BatchFields,
    LossExplosionError,
)

__all__ = ["CgcnnTrainer", "BaseRobustTrainer", "BatchFields", "BSAM", "LossExplosionError"]


class CgcnnTrainer(BaseRobustTrainer):
    """
    Robust trainer for graph neural networks on PyTorch Geometric batches.

    Expects MEGNet-style ``Data`` batches carrying:
        - ``x, edge_index, edge_attr, state, batch, bond_batch`` — model inputs
        - ``y`` — regression targets
        - ``omega`` (optional) — sample weights for weighted losses
        - ``rou`` (optional) — local label density (DILA / awareness)
        - ``phi`` (optional) — tail relevance in [0, 1] (SERA)

    The constructor signature and all training behaviour are inherited
    unchanged from :class:`BaseRobustTrainer`; see its docstring for the
    full argument list.
    """

    def forward_batch(self, batch) -> torch.Tensor:
        return self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.state,
            batch.batch,
            batch.bond_batch,
        )

    def unpack_batch(self, batch) -> BatchFields:
        return BatchFields(
            y=batch.y,
            weights=getattr(batch, "omega", None),
            density=getattr(batch, "rou", None),
            relevance=getattr(batch, "phi", None),
        )
