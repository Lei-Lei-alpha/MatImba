"""Deprecated: use :mod:`MatImba.analysis` instead.

This package is a thin compatibility shim.  The prediction containers and
checkpoint evaluation moved to the model-agnostic :mod:`MatImba.analysis`
subpackage; the plotting methods of the old ``imba_analyser`` class are now
plain functions there (``summary_plot``, ``composite_parity``,
``compare_binned_mae``, ``compare_ser_curves``, ``discovery_metrics``, ...).

Mapping:
    ml_pred       -> MatImba.analysis.PredictionSet
    CombinedPred  -> MatImba.analysis.EnsemblePrediction
    evaluate_ckpt -> MatImba.analysis.checkpoint.evaluate_ckpt
    imba_analyser -> module-level functions in MatImba.analysis
"""

import warnings

from ..analysis import EnsemblePrediction as CombinedPred
from ..analysis import PredictionSet as ml_pred
from ..analysis import evaluate_ckpt

warnings.warn(
    "MatImba.analyser is deprecated; use MatImba.analysis instead "
    "(ml_pred -> PredictionSet, CombinedPred -> EnsemblePrediction, "
    "imba_analyser methods -> analysis module functions).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ml_pred", "CombinedPred", "evaluate_ckpt", "imba_analyser"]


def __getattr__(name):
    # The legacy imba_analyser class (and anything else from the old module)
    # is loaded lazily: it drags in torch/model imports.
    if name == "imba_analyser":
        from .analyser import imba_analyser
        return imba_analyser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
