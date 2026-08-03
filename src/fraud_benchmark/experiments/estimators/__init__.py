"""One module per model, behind one interface.

An estimator is a function `(Prepared, seed) -> dict`. It receives a design matrix,
labels and split indices, and returns one record per seed:

    {"seed": int, "fit_seconds": float,
     "scores": {"val": {...}, "test": {...}},
     ...whatever else that model can usefully report}

Everything upstream -- reading the parquet, the artifact group, the history window,
the label regime, the split -- has already happened in `experiments.experiment`, so
an estimator sees none of it and cannot disagree with another about it. That is the
point of the split: a gap between two rows of a results table is attributable to the
model, because nothing else differed.

Each module owns its own threshold choice only in the sense of applying the shared
rule: the threshold maximising F1 on validation, applied unchanged to test. Fixed
hyperparameters, no per-condition tuning -- otherwise a gap measures tuning effort.

    xgboost   gradient-boosted trees; indifferent to the ordinal codes
    mlp       3-layer net, one-hot expanding the codes per batch
    logistic  L2 logistic regression, the linear reference and no more
"""

from __future__ import annotations

from collections.abc import Callable

#: Name on the CLI -> the module path that implements it. Imported lazily: the MLP
#: pulls in torch, which costs seconds and is pointless for an XGBoost run.
ESTIMATORS = {
    "xgboost": "fraud_benchmark.experiments.estimators.xgboost",
    "mlp": "fraud_benchmark.experiments.estimators.mlp",
    "logistic": "fraud_benchmark.experiments.estimators.logistic",
}

MODELS = tuple(ESTIMATORS)


def get_estimator(name: str) -> Callable:
    """The `fit_and_score` of one model, imported on demand."""
    from importlib import import_module

    if name not in ESTIMATORS:
        raise ValueError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    return import_module(ESTIMATORS[name]).fit_and_score
