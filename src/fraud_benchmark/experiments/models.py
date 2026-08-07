"""Fixed XGBoost configuration shared by the two public paper protocols."""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'n_jobs': 4,
    'eval_metric': 'aucpr',
}


def fit_xgboost(x: np.ndarray, y: np.ndarray, seed: int) -> XGBClassifier:
    """Fit the fixed paper model, weighting from labels visible in training."""
    positive = int(np.sum(y))
    negative = int(len(y) - positive)
    scale = negative / positive if positive else 1.0
    model = XGBClassifier(random_state=seed, scale_pos_weight=scale, **XGB_PARAMS)
    model.fit(x, y)
    return model
