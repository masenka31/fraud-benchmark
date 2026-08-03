import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraud_benchmark.ablation.models import (
    TRIVIAL_RULES,
    fit_logistic,
    fit_xgboost,
    trivial_rule_scores,
)


def xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (x["a"] + rng.normal(scale=0.1, size=n) > 0).astype(int).to_numpy()
    return x, y


def test_logistic_learns_a_separable_signal():
    x, y = xy()
    model = fit_logistic(x, y)
    scores = model.predict_proba(x)[:, 1]
    assert scores.shape == (len(x),)
    assert average_precision_score(y, scores) > 0.9


def test_xgboost_learns_a_separable_signal():
    x, y = xy()
    model = fit_xgboost(x, y, seed=0)
    assert average_precision_score(y, model.predict_proba(x)[:, 1]) > 0.9


def test_xgboost_is_deterministic_for_a_fixed_seed():
    x, y = xy()
    a = fit_xgboost(x, y, seed=3).predict_proba(x)[:, 1]
    b = fit_xgboost(x, y, seed=3).predict_proba(x)[:, 1]
    assert np.allclose(a, b)


def test_xgboost_differs_across_seeds():
    """If seeds do not move the model, the seed-noise floor is meaningless."""
    x, y = xy()
    a = fit_xgboost(x, y, seed=0).predict_proba(x)[:, 1]
    b = fit_xgboost(x, y, seed=1).predict_proba(x)[:, 1]
    assert not np.allclose(a, b)


def test_the_trivial_rule_scores_one_for_matching_rows():
    df = pd.DataFrame({"Merchant State": pd.Series(["Italy", "CA"], dtype="string")})
    scores = trivial_rule_scores(df, dataset="ibm_ccf")
    assert list(scores) == [1.0, 0.0]


def test_datasets_without_a_rule_return_none():
    for dataset in ["saml_d", "sparkov", "sparkov_slow"]:
        assert dataset not in TRIVIAL_RULES
        assert trivial_rule_scores(pd.DataFrame({"x": [1]}), dataset=dataset) is None


def test_the_rule_needs_no_fitting_and_ignores_labels():
    """It is a fixed rule, which is why it is evaluated once per dataset."""
    df = pd.DataFrame({"Merchant State": pd.Series(["Italy"], dtype="string")})
    assert list(trivial_rule_scores(df, dataset="ibm_ccf")) == [1.0]


def test_the_rule_treats_a_null_as_not_matching():
    """IBM CCF's Merchant State is null on every online transaction. Without
    this the comparison yields pd.NA and the float conversion raises, taking
    down the whole cell before a single model is fitted."""
    df = pd.DataFrame({"Merchant State": pd.Series(["Italy", None, "CA"], dtype="string")})
    scores = trivial_rule_scores(df, dataset="ibm_ccf")
    assert list(scores) == [1.0, 0.0, 0.0]
