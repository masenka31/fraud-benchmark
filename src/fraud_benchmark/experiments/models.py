"""The three models.

Hyperparameters are fixed across every condition. Tuning per condition would
confound a comparison with tuning effort: a gap between two feature sets, or
between two label-delay regimes, would no longer be attributable to what changed.

The trivial rule is the floor and matters more than it looks. An earlier probe
found a boosted model scoring *below* the one-line rule on IBM CCF (0.041 against
0.764); without the floor in the table that result is invisible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# (column, value) pairs that alone identify most of a dataset's frauds. The column
# is named as it appears in a *feature parquet*, which is what an experiment reads;
# the raw `Merchant State` survives there as `artifact_merchant_state`.
TRIVIAL_RULES: dict[str, tuple[str, str]] = {
    # Not a baseline a train-only model could reach: train holds zero Italy
    # frauds, so this rule imports knowledge from outside the training data. It
    # bounds what the labels encode, and is reported as a ceiling, not a floor.
    "ibm_ccf": ("artifact_merchant_state", "Italy"),
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
    # fillna(False): the comparison yields pd.NA where the column is null, and
    # IBM CCF's Merchant State is null on every online transaction. A null is
    # not Italy, so False is the right reading -- without this the conversion
    # raises on NAType and takes the whole cell down before a single fit.
    matches = (df[column].astype("string") == value).fillna(False)
    return matches.to_numpy().astype("float64")
