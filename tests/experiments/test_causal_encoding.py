import numpy as np
import pandas as pd

from fraud_benchmark.experiments.causal_encoding import causal_count
from fraud_benchmark.experiments.causal_encoding import causal_target_rate


def test_causal_count_only_uses_earlier_training_rows():
    values = pd.Series(['a', 'a', 'a', 'a'])
    train = np.array([True, False, True, False])
    np.testing.assert_array_equal(causal_count(values, train), [0, 1, 1, 2])


def test_future_training_label_cannot_change_an_earlier_feature():
    values = pd.Series(['a', 'a', 'a', 'a'])
    train = np.array([True, True, True, False])
    first = causal_target_rate(values, np.array([0, 0, 1, 1]), train)
    changed = causal_target_rate(values, np.array([0, 0, 0, 1]), train)
    np.testing.assert_array_equal(first[:3], changed[:3])
    assert first[3] > changed[3]


def test_validation_label_never_updates_the_stream():
    values = pd.Series(['a', 'a', 'a'])
    train = np.array([True, False, False])
    first = causal_target_rate(values, np.array([0, 1, 0]), train)
    changed = causal_target_rate(values, np.array([0, 0, 0]), train)
    np.testing.assert_array_equal(first, changed)
