"""Both post-preparation splits must stay strictly temporal.

A split that lets a later row train against an earlier one is lookahead: it raises
the score and raises no error.
"""

import pandas as pd
import pytest

from fraud_benchmark.experiments.splits import FIRST_ITALY_FRAUD
from fraud_benchmark.experiments.splits import LAST_LABELLED_FRAUD
from fraud_benchmark.experiments.splits import italy_holdout_split
from fraud_benchmark.experiments.splits import standard_split


def _frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'event_time': pd.date_range('2020-01-01', periods=n, freq='D'),
            'is_fraud': [False] * n,
            'amount': [1.0] * n,
        }
    )


def test_standard_split_labels_every_row():
    out = standard_split(_frame())
    assert set(out['split']) == {'train', 'val', 'test'}
    assert out['split'].notna().all()


def test_standard_split_is_ordered_in_time():
    out = standard_split(_frame())
    ends = {s: out.loc[out['split'] == s, 'event_time'].max() for s in ('train', 'val')}
    starts = {s: out.loc[out['split'] == s, 'event_time'].min() for s in ('val', 'test')}
    assert ends['train'] < starts['val']
    assert ends['val'] < starts['test']


def test_standard_split_is_roughly_80_10_10():
    out = standard_split(_frame(1000))
    shares = out['split'].value_counts(normalize=True)
    assert shares['train'] == pytest.approx(0.8, abs=0.02)
    assert shares['val'] == pytest.approx(0.1, abs=0.02)


def test_italy_holdout_train_predates_the_first_italy_fraud():
    frame = pd.DataFrame(
        {
            'event_time': pd.date_range('2017-01-01', periods=4000, freq='6h'),
            'is_fraud': [False] * 4000,
            'amount': [1.0] * 4000,
        }
    )
    out = italy_holdout_split(frame)
    train_end = out.loc[out['split'] == 'train', 'event_time'].max()
    assert train_end < FIRST_ITALY_FRAUD


def test_italy_holdout_drops_rows_after_the_last_labelled_fraud():
    # Rows after that timestamp are unlabelled, so keeping them would score a model
    # against absent labels.
    frame = pd.DataFrame(
        {
            'event_time': pd.date_range('2017-01-01', periods=5000, freq='6h'),
            'is_fraud': [False] * 5000,
            'amount': [1.0] * 5000,
        }
    )
    # Guard against a vacuous pass: the fixture must genuinely span past the cut.
    assert (frame['event_time'] > LAST_LABELLED_FRAUD).any()

    out = italy_holdout_split(frame)
    assert out['event_time'].max() <= LAST_LABELLED_FRAUD
