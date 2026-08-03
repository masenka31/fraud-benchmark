import numpy as np
import pytest

from fraud_benchmark.ablation.metrics import best_f1_threshold, score


def test_a_perfect_ranking_gets_average_precision_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert score(y, s, threshold=0.5)["average_precision"] == pytest.approx(1.0)


def test_average_precision_of_a_random_scorer_approaches_the_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.01).astype(int)
    s = rng.random(20_000)
    assert score(y, s, threshold=0.5)["average_precision"] == pytest.approx(0.01, abs=0.005)


def test_roc_auc_is_not_reported():
    """Base rates here are 0.10-0.52%; ROC AUC would flatter a useless model."""
    y = np.array([0, 1])
    s = np.array([0.1, 0.9])
    assert "roc_auc" not in score(y, s, threshold=0.5)


def test_precision_recall_and_f1_come_from_the_given_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.6, 0.9])
    out = score(y, s, threshold=0.5)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(0.8)


def test_best_f1_threshold_finds_a_separating_cut():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = best_f1_threshold(y, s)
    assert score(y, s, threshold=threshold)["f1"] == pytest.approx(1.0)


def test_score_reports_the_positive_count_it_was_given():
    y = np.array([0, 0, 1])
    s = np.array([0.1, 0.2, 0.9])
    out = score(y, s, threshold=0.5)
    assert out["n_rows"] == 3
    assert out["n_positive"] == 1


def test_a_column_with_no_positives_does_not_crash():
    y = np.zeros(5, dtype=int)
    s = np.linspace(0, 1, 5)
    out = score(y, s, threshold=0.5)
    assert out["n_positive"] == 0
    assert np.isnan(out["average_precision"])
