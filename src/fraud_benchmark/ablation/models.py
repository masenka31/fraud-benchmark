"""The three models.

Hyperparameters are fixed across every condition. Tuning per condition would
confound the ablation with tuning effort: a gap between `leaky` and `clean` would
no longer be attributable to the columns.

The trivial rule is the floor and matters more than it looks. An earlier probe
found a boosted model scoring *below* the one-line rule on the IBM CCF subsample;
without the floor in the table that result is invisible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# (column, value) pairs that alone identify most of a dataset's frauds.
TRIVIAL_RULES: dict[str, tuple[str, str]] = {
    "ibm_ccf": ("Merchant State", "Italy"),
    "ibm_ccf_subsample_fast": ("Merchant State", "Italy"),
    "ibm_ccf_subsample_slow": ("Merchant State", "Italy"),
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 4,
    "eval_metric": "aucpr",
}


def fit_logistic(x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    """L2 logistic regression. Deterministic, so it needs no seeds.

    No n_jobs: lbfgs ignores it for a binary problem, and sklearn 1.9 deprecates
    passing it at all.
    """
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    model.fit(x, y)
    return model


def fit_xgboost(x: pd.DataFrame, y: np.ndarray, seed: int) -> XGBClassifier:
    positive = int(np.sum(y))
    negative = int(len(y) - positive)
    scale = (negative / positive) if positive else 1.0
    model = XGBClassifier(random_state=seed, scale_pos_weight=scale, **XGB_PARAMS)
    model.fit(x, y)
    return model


def trivial_rule_scores(df: pd.DataFrame, dataset: str) -> np.ndarray | None:
    """1.0 where the rule fires, 0.0 elsewhere. None if the dataset has no rule.

    Reads the raw frame, not a feature matrix, so it is unaffected by the feature
    set and label regime -- which is why it is evaluated once per dataset.
    """
    rule = TRIVIAL_RULES.get(dataset)
    if rule is None:
        return None
    column, value = rule
    return (df[column].astype("string") == value).to_numpy().astype("float64")
