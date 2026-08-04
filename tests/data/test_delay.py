import pandas as pd
import pytest

from fraud_benchmark.data.delay import DelayParams
from fraud_benchmark.data.delay import assign_reported_at


def frame(rows):
    """rows: list of (entity_id, 'YYYY-MM-DD', is_fraud, campaign_id)."""
    return pd.DataFrame(
        {
            'entity_id': pd.Series([r[0] for r in rows], dtype='string'),
            'event_time': pd.to_datetime([r[1] for r in rows]),
            'is_fraud': [r[2] for r in rows],
            'campaign_id': pd.Series([r[3] for r in rows], dtype='Int64'),
        }
    )


PARAMS = DelayParams(median_days=7.0, sigma=1.0, seed=0)


def test_non_fraud_rows_have_no_reported_at():
    df = frame([('a', '2023-01-01', False, pd.NA)])
    out = assign_reported_at(df, PARAMS)
    assert out.isna().all()


def test_every_fraud_gets_a_reported_at():
    df = frame([('a', '2023-01-01', True, 0)])
    out = assign_reported_at(df, PARAMS)
    assert out.notna().all()


def test_reported_at_is_never_before_the_transaction():
    df = frame([('a', f'2023-01-{d:02d}', True, d) for d in range(1, 20)])
    out = assign_reported_at(df, PARAMS)
    assert (out >= df['event_time']).all()


def test_one_campaign_shares_a_single_reported_at():
    df = frame(
        [
            ('a', '2023-01-01', True, 0),
            ('a', '2023-01-02', True, 0),
            ('a', '2023-01-03', True, 0),
        ]
    )
    out = assign_reported_at(df, PARAMS)
    assert out.nunique() == 1


def test_the_shared_timestamp_follows_the_last_transaction_in_the_campaign():
    """A campaign cannot be reported before its final fraud has happened."""
    df = frame(
        [
            ('a', '2023-01-01', True, 0),
            ('a', '2023-06-01', True, 0),
        ]
    )
    out = assign_reported_at(df, PARAMS)
    assert (out >= pd.Timestamp('2023-06-01')).all()


def test_separate_campaigns_get_independent_timestamps():
    df = frame(
        [
            ('a', '2023-01-01', True, 0),
            ('b', '2023-01-01', True, 1),
        ]
    )
    out = assign_reported_at(df, PARAMS)
    assert out.iloc[0] != out.iloc[1]


def test_the_same_seed_reproduces_the_same_timestamps():
    df = frame([('a', f'2023-01-{d:02d}', True, d) for d in range(1, 15)])
    first = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=42))
    second = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=42))
    assert first.equals(second)


def test_a_different_seed_gives_different_timestamps():
    df = frame([('a', f'2023-01-{d:02d}', True, d) for d in range(1, 15)])
    first = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=1))
    second = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=2))
    assert not first.equals(second)


def test_row_order_does_not_change_the_result():
    """Reproducibility must not depend on how the frame happens to be sorted."""
    df = frame([('a', f'2023-01-{d:02d}', True, d) for d in range(1, 15)])
    straight = assign_reported_at(df, PARAMS)
    reversed_ = assign_reported_at(df.iloc[::-1], PARAMS)
    assert straight.equals(reversed_.iloc[::-1])


def test_the_median_delay_matches_the_configured_median():
    """Lognormal's median is exp(mu), so median_days should come out directly."""
    df = frame([('a', '2023-01-01', True, i) for i in range(20_000)])
    out = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=0))
    delays = (out - df['event_time']).dt.total_seconds() / 86_400
    assert 6.6 < delays.median() < 7.4


def test_the_distribution_is_right_skewed():
    """Mean well above median is the property that makes lognormal the right choice."""
    df = frame([('a', '2023-01-01', True, i) for i in range(20_000)])
    out = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=0))
    delays = (out - df['event_time']).dt.total_seconds() / 86_400
    assert delays.mean() > delays.median() * 1.3


def test_max_delay_days_truncates_the_tail():
    df = frame([('a', '2023-01-01', True, i) for i in range(5_000)])
    params = DelayParams(median_days=7.0, sigma=2.0, seed=0, max_delay_days=30.0)
    out = assign_reported_at(df, params)
    delays = (out - df['event_time']).dt.total_seconds() / 86_400
    assert delays.max() <= 30.0 + 1e-9


def test_invalid_params_are_rejected():
    for kwargs in (
        {'median_days': 0.0, 'sigma': 1.0},
        {'median_days': -1.0, 'sigma': 1.0},
        {'median_days': 7.0, 'sigma': 0.0},
        {'median_days': 7.0, 'sigma': -1.0},
    ):
        with pytest.raises(ValueError, match='must be positive'):
            DelayParams(seed=0, **kwargs)


def test_result_is_aligned_to_the_input_index():
    df = frame([('a', '2023-01-01', True, 0), ('a', '2023-01-02', True, 0)])
    shuffled = df.iloc[::-1]
    out = assign_reported_at(shuffled, PARAMS)
    assert out.index.equals(shuffled.index)


def test_a_heavy_tail_does_not_overflow():
    """sigma=2 with no cap produced ~90,000-day draws and overflowed [ns]."""
    df = frame([('a', '2023-01-01', True, i) for i in range(100_000)])
    out = assign_reported_at(df, DelayParams(median_days=7.0, sigma=2.0, seed=0))
    assert out.notna().all()
    assert pd.api.types.is_datetime64_dtype(out)


def test_is_fraud_is_not_required():
    """The contract is campaign_id, not is_fraud: a null id means not reportable."""
    df = frame([('a', '2023-01-01', True, 0), ('a', '2023-01-02', False, pd.NA)])
    out = assign_reported_at(df.drop(columns=['is_fraud']), PARAMS)
    assert out.notna().sum() == 1
    assert pd.isna(out.iloc[1])
