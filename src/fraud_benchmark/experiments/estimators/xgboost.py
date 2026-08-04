"""Gradient-boosted trees.

The default model, and the right default: trees are indifferent to the ordinal codes
the pipeline produces, they handle the NaN that a feature parquet uses to mean "this
history does not exist" natively, and they need no standardisation. An MLP needs all
three handled and its number is only interpretable once they are.

Hyperparameters come from `experiments.models.XGB_PARAMS` and are fixed across every
condition. `scale_pos_weight` is the one thing that varies, and it varies because the
data does: it is set from the training positives actually present, so a censored
label regime is reweighted to the labels it can see rather than to the labels it
cannot.

Note that this module is named `xgboost` and does not import xgboost. It reads
`XGB_PARAMS` and `fit_xgboost` from `experiments.models`, which owns them because the
trivial-rule floor is reported next to them. Absolute imports mean the shadowing is
harmless either way.
"""

from __future__ import annotations

import time

import numpy as np

from fraud_benchmark.experiments.experiment import Prepared
from fraud_benchmark.experiments.metrics import best_f1_threshold
from fraud_benchmark.experiments.metrics import score
from fraud_benchmark.experiments.models import fit_xgboost

#: Feature-importance entries kept in the record. The full list is one entry per
#: column, which at 200+ lagged columns would dominate a JSONL line.
TOP_FEATURES = 20


def fit_and_score(prepared: Prepared, seed: int) -> dict:
    """Fit on train, choose a threshold on val, report val and test."""
    started = time.monotonic()

    model = fit_xgboost(
        prepared.x[prepared.train_rows],
        prepared.y[prepared.train_rows],
        seed,
    )

    val_scores = model.predict_proba(prepared.x[prepared.val_rows])[:, 1]
    test_scores = model.predict_proba(prepared.x[prepared.test_rows])[:, 1]

    # Threshold from validation, applied unchanged to test. Choosing it on test would
    # be fitting to the thing being reported.
    threshold = best_f1_threshold(prepared.y_true[prepared.val_rows], val_scores)

    return {
        'seed': seed,
        'fit_seconds': round(time.monotonic() - started, 1),
        'n_trees': int(model.n_estimators),
        'top_features': _top_features(model, prepared.names),
        'scores': {
            'val': score(prepared.y_true[prepared.val_rows], val_scores, threshold),
            'test': score(prepared.y_true[prepared.test_rows], test_scores, threshold),
        },
    }


def _top_features(model, names: list[str]) -> list[dict]:
    """The highest-gain columns, as a share of total gain.

    Reported because it is how the retired flattened-window experiment found that
    lagged columns earned only 13% of total gain -- which is a fact about whether the
    history is worth its memory, and is invisible in an average precision alone.
    """
    booster = model.get_booster()
    gains = booster.get_score(importance_type='total_gain')
    if not gains:
        return []
    total = sum(gains.values())
    # XGBoost names columns f0, f1, ... when fitted on a numpy array.
    ranked = sorted(gains.items(), key=lambda item: item[1], reverse=True)
    out = []
    for key, gain in ranked[:TOP_FEATURES]:
        index = int(key[1:]) if key.startswith('f') and key[1:].isdigit() else None
        out.append(
            {
                'feature': names[index] if index is not None and index < len(names) else key,
                'gain_share': round(float(gain) / total, 5),
            }
        )
    return out


def lagged_gain_share(top_features: list[dict]) -> float:
    """Share of reported gain earned by lagged columns. 0.0 with no history."""
    if not top_features:
        return 0.0
    return float(np.sum([f['gain_share'] for f in top_features if '_lag' in f['feature']]))
