"""The lagged history window.

The property that matters: lag *k* of a row is that row's own entity's *k*th
previous transaction, or nothing. A lag that reaches a later row, or another
entity's row, is lookahead that would improve every score it touched.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments.history import MISSING
from fraud_benchmark.experiments.history import lag_columns
from fraud_benchmark.experiments.history import lag_matrix
from fraud_benchmark.experiments.history import previous_positions

BASE = pd.Timestamp('2020-05-01 08:00:00')


def frame(entities, hours):
    return (
        pd.Series(entities, dtype='string'),
        pd.Series([BASE + pd.Timedelta(hours=h) for h in hours]),
    )


def test_previous_positions_walks_back_within_one_entity():
    entity, time = frame(['a', 'a', 'a'], [0, 1, 2])
    previous = previous_positions(entity, time, 2)
    assert previous[0].tolist() == [-1, -1]  # no history at all
    assert previous[1].tolist() == [0, -1]  # one prior transaction
    assert previous[2].tolist() == [1, 0]


def test_previous_positions_never_crosses_an_entity():
    entity, time = frame(['a', 'b', 'a', 'b'], [0, 1, 2, 3])
    previous = previous_positions(entity, time, 1)
    assert previous[:, 0].tolist() == [-1, -1, 0, 1]


def test_previous_positions_follows_time_not_row_order():
    # Row 0 happens last, so it is the one with history.
    entity, time = frame(['a', 'a', 'a'], [10, 0, 5])
    previous = previous_positions(entity, time, 2)
    assert previous[0].tolist() == [2, 1]
    assert previous[1].tolist() == [-1, -1]
    assert previous[2].tolist() == [1, -1]


def test_previous_positions_only_ever_points_backwards_in_time():
    rng = np.random.default_rng(3)
    n = 400
    entity, time = frame(rng.choice(['a', 'b', 'c'], n), rng.integers(0, 300, n))
    previous = previous_positions(entity, time, 5)
    times = time.to_numpy()
    for lag in range(5):
        found = previous[:, lag] >= 0
        source = previous[found, lag]
        assert (times[source] <= times[found]).all()
        assert (entity.to_numpy()[source] == entity.to_numpy()[found]).all()


def test_previous_positions_are_strictly_ordered_between_lags():
    """Lag 2 is always older than lag 1, wherever both exist."""
    rng = np.random.default_rng(9)
    n = 300
    entity, time = frame(rng.choice(['a', 'b'], n), np.sort(rng.integers(0, 200, n)))
    previous = previous_positions(entity, time, 3)
    times = time.to_numpy()
    both = (previous[:, 0] >= 0) & (previous[:, 1] >= 0)
    assert (times[previous[both, 1]] <= times[previous[both, 0]]).all()


def test_lag_matrix_puts_the_target_block_first():
    entity, time = frame(['a', 'a'], [0, 1])
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
    out = lag_matrix(values, entity, time, 1)
    assert out.shape == (2, 4)
    assert out[:, :2].tolist() == values.tolist()


def test_lag_matrix_copies_the_previous_rows_values():
    entity, time = frame(['a', 'a', 'a'], [0, 1, 2])
    values = np.array([[10.0], [20.0], [30.0]], dtype='float32')
    out = lag_matrix(values, entity, time, 2)
    # columns: target, lag1, lag2
    assert out[2].tolist() == [30.0, 20.0, 10.0]
    assert out[1].tolist() == [20.0, 10.0, MISSING]
    assert out[0].tolist() == [10.0, MISSING, MISSING]


def test_lag_matrix_marks_absent_history_rather_than_zeroing_it():
    """0.0 would read as a real measurement of zero. MISSING is learnable."""
    entity, time = frame(['a'], [0])
    out = lag_matrix(np.array([[0.0]], dtype='float32'), entity, time, 1)
    assert out[0, 0] == 0.0  # a genuine zero survives
    assert out[0, 1] == MISSING  # and is distinguishable from no history


def test_lag_matrix_is_a_no_op_at_zero_lags():
    entity, time = frame(['a', 'a'], [0, 1])
    values = np.array([[1.0], [2.0]], dtype='float32')
    assert lag_matrix(values, entity, time, 0).tolist() == values.tolist()


def test_lag_matrix_does_not_leak_a_later_row_into_an_earlier_one():
    entity, time = frame(['a'] * 5, [0, 1, 2, 3, 4])
    values = np.arange(5, dtype='float32').reshape(5, 1)
    before = lag_matrix(values, entity, time, 2)

    tampered = values.copy()
    tampered[-1] = 10_000.0
    after = lag_matrix(tampered, entity, time, 2)

    assert before[:-1].tolist() == after[:-1].tolist()


def test_lag_matrix_keeps_entities_apart():
    entity, time = frame(['a', 'b', 'a'], [0, 1, 2])
    values = np.array([[1.0], [99.0], [3.0]], dtype='float32')
    out = lag_matrix(values, entity, time, 1)
    # b's row must not appear as a's history.
    assert out[2].tolist() == [3.0, 1.0]
    assert out[1].tolist() == [99.0, MISSING]


def test_lag_matrix_handles_tied_timestamps_without_duplicating_a_row():
    entity, time = frame(['a', 'a'], [0, 0])
    values = np.array([[1.0], [2.0]], dtype='float32')
    out = lag_matrix(values, entity, time, 1)
    # Ties break on input order, so row 1's history is row 0 and not itself.
    assert out[0].tolist() == [1.0, MISSING]
    assert out[1].tolist() == [2.0, 1.0]


def test_lag_columns_names_every_block_in_matrix_order():
    assert lag_columns(['a', 'b'], 2) == ['a', 'b', 'a_lag1', 'b_lag1', 'a_lag2', 'b_lag2']


def test_lag_columns_at_zero_lags_is_the_input():
    assert lag_columns(['a', 'b'], 0) == ['a', 'b']


@pytest.mark.parametrize('n_lags', [1, 3, 7])
def test_lag_matrix_width_matches_its_column_names(n_lags):
    entity, time = frame(['a'] * 4, [0, 1, 2, 3])
    values = np.ones((4, 3), dtype='float32')
    out = lag_matrix(values, entity, time, n_lags)
    assert out.shape[1] == len(lag_columns(['x', 'y', 'z'], n_lags))
