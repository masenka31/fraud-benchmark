"""Censoring is the consumption side of the label delay, so it is tested here.

The property that matters: an unreported fraud stays in the frame labelled 0. It is
not dropped. Dropping would model a system that knows which rows to distrust, which
is exactly the knowledge label delay denies it.
"""

import numpy as np
import pandas as pd

from fraud_benchmark.data.censoring import censored_labels


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'event_time': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
            'is_fraud': [True, True, False, False],
            'reported_at': pd.to_datetime(['2024-01-05', '2024-02-20', None, None]),
        }
    )


def test_fraud_reported_before_the_cutoff_is_known():
    labels = censored_labels(_frame(), cutoff=pd.Timestamp('2024-01-10'))
    assert labels[0] == 1


def test_fraud_reported_after_the_cutoff_is_labelled_zero_not_dropped():
    frame = _frame()
    labels = censored_labels(frame, cutoff=pd.Timestamp('2024-01-10'))
    assert len(labels) == len(frame)
    assert labels[1] == 0


def test_non_fraud_is_zero():
    labels = censored_labels(_frame(), cutoff=pd.Timestamp('2024-03-01'))
    assert labels[2] == 0 and labels[3] == 0


def test_returns_an_integer_array():
    labels = censored_labels(_frame(), cutoff=pd.Timestamp('2024-01-10'))
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.dtype('int64')


def test_a_cutoff_past_every_report_recovers_the_true_labels():
    frame = _frame()
    labels = censored_labels(frame, cutoff=pd.Timestamp('2030-01-01'))
    assert labels.tolist() == frame['is_fraud'].astype(int).tolist()


def test_a_non_fraud_reported_before_the_cutoff_is_still_zero():
    # Guards the `is_fraud` gate: a genuine, early reported_at must not flip a
    # non-fraud row to 1.
    frame = pd.DataFrame(
        {
            'event_time': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'is_fraud': [False, False],
            'reported_at': pd.to_datetime(['2023-01-01', None]),
        }
    )
    labels = censored_labels(frame, cutoff=pd.Timestamp('2024-06-01'))
    assert labels.tolist() == [0, 0]


def test_a_fraud_never_reported_is_zero():
    frame = pd.DataFrame(
        {
            'event_time': pd.to_datetime(['2024-01-01']),
            'is_fraud': [True],
            'reported_at': pd.to_datetime([None]),
        }
    )
    labels = censored_labels(frame, cutoff=pd.Timestamp('2030-01-01'))
    assert labels.tolist() == [0]
