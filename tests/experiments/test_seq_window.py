"""The lag window must look backwards only.

A forward shift would raise the score and raise no error, so this is the one
property worth a dedicated test.
"""
import numpy as np

from fraud_benchmark.experiments.seq_window import lag_matrix


def test_lag_one_returns_the_previous_row_of_the_same_user():
    values = np.array([[10.0], [20.0], [30.0]], dtype="float32")
    user = np.array([1, 1, 1])
    out = lag_matrix(values, user, n_lags=1)
    assert list(out[:, 0]) == [0.0, 10.0, 20.0]


def test_a_users_first_row_is_zero_filled_not_borrowed():
    """Row 2 starts a new user; its lag must be 0, never user 1's transaction."""
    values = np.array([[10.0], [20.0], [30.0]], dtype="float32")
    user = np.array([1, 1, 2])
    out = lag_matrix(values, user, n_lags=1)
    assert list(out[:, 0]) == [0.0, 10.0, 0.0]


def test_deeper_lags_zero_fill_until_history_exists():
    values = np.array([[1.0], [2.0], [3.0]], dtype="float32")
    user = np.array([7, 7, 7])
    out = lag_matrix(values, user, n_lags=3)
    # columns are lag1, lag2, lag3
    assert list(out[0]) == [0.0, 0.0, 0.0]
    assert list(out[1]) == [1.0, 0.0, 0.0]
    assert list(out[2]) == [2.0, 1.0, 0.0]


def test_no_future_value_ever_appears():
    """The direct causality statement: nothing in row i's window exceeds
    the values available strictly before i."""
    rng = np.random.default_rng(0)
    n = 300
    user = np.sort(rng.integers(0, 5, n))
    values = np.arange(n, dtype="float32").reshape(-1, 1)  # strictly increasing
    out = lag_matrix(values, user, n_lags=9)
    for i in range(n):
        window = out[i]
        assert (window < values[i, 0]).all() or (window == 0).all() or \
               all(w == 0 or w < values[i, 0] for w in window), \
               f"row {i} window {window} contains a value >= its own {values[i,0]}"


def test_width_is_features_times_lags():
    values = np.zeros((5, 4), dtype="float32")
    out = lag_matrix(values, np.zeros(5, dtype=int), n_lags=9)
    assert out.shape == (5, 36)
