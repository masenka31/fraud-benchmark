"""L2 logistic regression: the linear reference.

Here to answer "is the boosted model doing anything a line could not?", and nothing
more. Deterministic, so every seed returns the same numbers -- run it with one.

Two things it needs that the tree models do not, both because a linear model has no
notion of either:

* **Standardisation.** The matrix mixes ordinal codes with amounts, and on IBM CCF a
  capped vocabulary still reaches 256 while `hour` reaches 23; lbfgs does not
  converge on that, and a non-converged linear reference is worse than none.
* **A value for NaN.** A feature parquet uses NaN to mean "this history does not
  exist". Imputing the train mean is the least-committal choice available, and it is
  fitted on train like everything else here.

Both make this model a weaker reference than it looks: the ordinal codes imply an
ordering that does not exist (merchant state 5 is not "more" than 4), so a low score
here is partly an encoding artifact rather than evidence that the problem is
non-linear. `estimators/mlp.py` one-hot expands for exactly this reason.
"""

from __future__ import annotations

import time

import numpy as np

from fraud_benchmark.experiments.experiment import Prepared
from fraud_benchmark.experiments.metrics import best_f1_threshold
from fraud_benchmark.experiments.metrics import score
from fraud_benchmark.experiments.models import fit_logistic


def fit_and_score(prepared: Prepared, seed: int) -> dict:
    """Fit on train, choose a threshold on val, report val and test.

    `seed` is recorded but unused: lbfgs on a fixed matrix is deterministic.
    """
    started = time.monotonic()

    train = prepared.x[prepared.train_rows]
    mean = np.nanmean(train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype('float32')
    std = np.nanstd(train, axis=0)
    # A constant column has no spread; dividing by 1 leaves it at 0.
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype('float32')

    def matrix(rows: np.ndarray) -> np.ndarray:
        block = prepared.x[rows]
        block = np.where(np.isnan(block), mean, block)
        return (block - mean) / std

    model = fit_logistic(matrix(prepared.train_rows), prepared.y[prepared.train_rows])
    val_scores = model.predict_proba(matrix(prepared.val_rows))[:, 1]
    test_scores = model.predict_proba(matrix(prepared.test_rows))[:, 1]
    threshold = best_f1_threshold(prepared.y_true[prepared.val_rows], val_scores)

    return {
        'seed': seed,
        'fit_seconds': round(time.monotonic() - started, 1),
        'deterministic': True,
        'n_iterations': int(np.max(model.n_iter_)),
        'converged': bool(np.max(model.n_iter_) < model.max_iter),
        'scores': {
            'val': score(prepared.y_true[prepared.val_rows], val_scores, threshold),
            'test': score(prepared.y_true[prepared.test_rows], test_scores, threshold),
        },
    }
