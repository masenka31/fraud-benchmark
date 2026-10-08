"""Both post-preparation splits must stay strictly temporal.

A split that lets a later row train against an earlier one is lookahead: it raises
the score and raises no error.
"""

import pandas as pd
import pytest

from fraud_benchmark.experiments.splits import FIRST_ITALY_FRAUD
from fraud_benchmark.experiments.splits import LAST_LABELLED_FRAUD
from fraud_benchmark.experiments.splits import iid_customer_split
from fraud_benchmark.experiments.splits import iid_row_split
from fraud_benchmark.experiments.splits import italy_holdout_split
from fraud_benchmark.experiments.splits import pre_italy_68_16_16_split
from fraud_benchmark.experiments.splits import pre_italy_iid_customer_split
from fraud_benchmark.experiments.splits import pre_italy_iid_row_split
from fraud_benchmark.experiments.splits import pre_italy_split
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


def test_pre_italy_split_right_crops_and_resplits():
    frame = pd.DataFrame(
        {
            'event_time': pd.date_range('2017-01-01', periods=4000, freq='6h'),
            'is_fraud': [False] * 4000,
            'amount': [1.0] * 4000,
            'artifact_merchant_state': ['Italy' if i == 100 else 'CA' for i in range(4000)],
        }
    )
    out = pre_italy_split(frame)
    assert out['event_time'].max() < FIRST_ITALY_FRAUD
    assert out['artifact_merchant_state'].eq('Italy').sum() == 1
    assert out['split'].value_counts(normalize=True).to_dict() == pytest.approx(
        {'train': 0.8, 'val': 0.1, 'test': 0.1}, abs=0.001
    )


def test_pre_italy_68_16_16_split_keeps_time_ties_and_population():
    frame = _frame(100).assign(
        event_time=pd.to_datetime(
            ['2010-01-01'] * 68 + ['2011-01-01'] * 16 + ['2012-01-01'] * 16
        )
    )
    out = pre_italy_68_16_16_split(frame.sample(frac=1, random_state=4))
    assert out['split'].value_counts().to_dict() == {'train': 68, 'val': 16, 'test': 16}
    assert out.groupby('event_time')['split'].nunique().eq(1).all()
    assert out['event_time'].max() < FIRST_ITALY_FRAUD


def test_pre_italy_regimes_use_the_same_right_cropped_rows():
    frame = pd.DataFrame(
        {
            'event_time': pd.date_range('2017-01-01', periods=4000, freq='6h'),
            'entity_id': [f'user-{i % 40}' for i in range(4000)],
            'is_fraud': [False] * 4000,
            'amount': [1.0] * 4000,
            'artifact_merchant_state': ['Italy' if i == 100 else 'CA' for i in range(4000)],
        }
    )
    regimes = (
        pre_italy_split(frame),
        pre_italy_iid_row_split(frame),
        pre_italy_iid_customer_split(frame),
    )
    expected_times = regimes[0]['event_time'].tolist()
    for regime in regimes:
        assert regime['event_time'].tolist() == expected_times
        assert regime['artifact_merchant_state'].eq('Italy').sum() == 1


def test_iid_row_split_is_exact_disjoint_and_deterministic():
    frame = _frame(1000).assign(entity_id=[f'user-{i % 37}' for i in range(1000)])
    first = iid_row_split(frame)
    second = iid_row_split(frame.sample(frac=1, random_state=9))
    assert first['split'].value_counts().to_dict() == {'train': 800, 'val': 100, 'test': 100}
    pd.testing.assert_series_equal(first['split'], second['split'])


def test_iid_customer_split_keeps_customers_whole_and_balances_rows():
    sizes = [240, 180, 130, 100, 80, 70, 60, 50, 40, 30, 20, 10, 10, 10, 10, 10]
    entity = [f'user-{i}' for i, size in enumerate(sizes) for _ in range(size)]
    frame = _frame(len(entity)).assign(entity_id=entity)
    out = iid_customer_split(frame)

    assert out.groupby('entity_id')['split'].nunique().eq(1).all()
    assert out.groupby('split')['entity_id'].nunique().to_dict() == {
        'train': 12,
        'val': 1,
        'test': 3,
    }
    shares = out['split'].value_counts(normalize=True)
    # With indivisible groups the largest customer bounds the possible error.
    max_share = max(sizes) / len(entity)
    assert shares['train'] == pytest.approx(0.8, abs=max_share)
    assert shares['val'] == pytest.approx(0.1, abs=max_share)
    assert shares['test'] == pytest.approx(0.1, abs=max_share)
    pd.testing.assert_series_equal(out['split'], iid_customer_split(frame)['split'])
