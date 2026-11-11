import numpy as np
import pandas as pd

from fraud_benchmark.experiments.causal_encoding import causal_count


def test_causal_count_only_uses_earlier_training_rows():
    values = pd.Series(['a', 'a', 'a', 'a'])
    train = np.array([True, False, True, False])
    np.testing.assert_array_equal(causal_count(values, train), [0, 1, 1, 2])
