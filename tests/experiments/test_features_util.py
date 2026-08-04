"""The feature primitives, and the one property that matters: no lookahead.

A feature that sees the row it describes makes every measurement *better*, not
noisier, so nothing downstream fails when it happens. These tests are the only
place that catches it.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments.features.util import ARTIFACT_PREFIX
from fraud_benchmark.experiments.features.util import KEY_COLUMNS
from fraud_benchmark.experiments.features.util import EntityHistory
from fraud_benchmark.experiments.features.util import FeatureContractError
from fraud_benchmark.experiments.features.util import amount_shape
from fraud_benchmark.experiments.features.util import artifact_columns
from fraud_benchmark.experiments.features.util import clock_features
from fraud_benchmark.experiments.features.util import days_between
from fraud_benchmark.experiments.features.util import feature_columns
from fraud_benchmark.experiments.features.util import haversine_km
from fraud_benchmark.experiments.features.util import parse_money
from fraud_benchmark.experiments.features.util import safe_ratio
from fraud_benchmark.experiments.features.util import signed_log1p
from fraud_benchmark.experiments.features.util import write_features

BASE = pd.Timestamp('2020-03-01 12:00:00')


def frame(entities, hours, amounts, values=None):
    """A minimal (entity, time, amount) frame with an optional context column."""
    data = {
        'entity_id': pd.Series(entities, dtype='string'),
        'event_time': [BASE + pd.Timedelta(hours=h) for h in hours],
        'amount': pd.Series(amounts, dtype='float64'),
    }
    if values is not None:
        data['value'] = pd.Series(values, dtype='string')
    return pd.DataFrame(data)


def history_of(df):
    return EntityHistory(df['entity_id'], df['event_time'])


# --- windows exclude the row they describe ---------------------------------


def test_rolling_count_excludes_the_current_row():
    df = frame(['a', 'a', 'a'], [0, 1, 2], [10.0, 10.0, 10.0])
    assert list(history_of(df).rolling_count('24h')) == [0.0, 1.0, 2.0]


def test_rolling_sum_excludes_the_current_row():
    df = frame(['a', 'a', 'a'], [0, 1, 2], [1.0, 2.0, 4.0])
    sums = history_of(df).rolling_sum(df['amount'], '24h')
    assert list(sums) == [0.0, 1.0, 3.0]


def test_rolling_sum_is_zero_not_nan_without_history():
    df = frame(['a'], [0], [5.0])
    assert history_of(df).rolling_sum(df['amount'], '7d')[0] == 0.0


def test_window_drops_rows_that_have_aged_out():
    # 0h, 1h, then 48h later: the third row's 24h window is empty again.
    df = frame(['a', 'a', 'a'], [0, 1, 49], [1.0, 1.0, 1.0])
    assert list(history_of(df).rolling_count('24h')) == [0.0, 1.0, 0.0]


def test_windows_do_not_cross_entities():
    df = frame(['a', 'b', 'a'], [0, 1, 2], [1.0, 1.0, 1.0])
    assert list(history_of(df).rolling_count('24h')) == [0.0, 0.0, 1.0]


def test_zscore_is_zero_when_history_is_flat_or_empty():
    df = frame(['a'] * 4, [0, 1, 2, 3], [5.0, 5.0, 7.0, 100.0])
    z = history_of(df).rolling_zscore(df['amount'], '7d')
    assert z[0] == 0.0  # no history
    assert z[1] == 0.0  # one prior row, so no standard deviation
    assert z[2] == 0.0  # prior history is flat, so no deviation to measure
    assert z[3] > 0.0


# --- expanding statistics are shifted -------------------------------------


def test_prior_mean_and_max_exclude_this_row():
    df = frame(['a', 'a', 'a'], [0, 1, 2], [10.0, 20.0, 900.0])
    history = history_of(df)
    assert list(history.prior_mean(df['amount'])) == [0.0, 10.0, 15.0]
    prior_max = history.prior_max(df['amount'])
    assert np.isnan(prior_max[0])
    assert list(prior_max[1:]) == [10.0, 20.0]


def test_ordinal_counts_only_prior_rows():
    df = frame(['a', 'b', 'a', 'a'], [0, 0, 1, 2], [1.0] * 4)
    assert list(history_of(df).ordinal()) == [0.0, 0.0, 1.0, 2.0]


def test_prior_distinct_excludes_this_row():
    df = frame(['a'] * 4, [0, 1, 2, 3], [1.0] * 4, ['x', 'x', 'y', 'z'])
    assert list(history_of(df).prior_distinct(df['value'])) == [0.0, 1.0, 1.0, 2.0]


# --- context: seen before, and when ---------------------------------------


def test_first_occurrence_fires_once_per_value():
    df = frame(['a'] * 4, [0, 1, 2, 3], [1.0] * 4, ['x', 'y', 'x', 'y'])
    assert list(history_of(df).first_occurrence(df['value'])) == [1.0, 1.0, 0.0, 0.0]


def test_first_occurrence_is_per_entity():
    df = frame(['a', 'b'], [0, 1], [1.0, 1.0], ['x', 'x'])
    assert list(history_of(df).first_occurrence(df['value'])) == [1.0, 1.0]


def test_missing_context_values_group_together_rather_than_vanishing():
    df = frame(['a'] * 3, [0, 1, 2], [1.0] * 3, [None, None, 'x'])
    assert list(history_of(df).first_occurrence(df['value'])) == [1.0, 0.0, 1.0]


def test_gap_seconds_is_nan_on_an_entity_first_row():
    df = frame(['a', 'a'], [0, 2], [1.0, 1.0])
    gaps = history_of(df).gap_seconds()
    assert np.isnan(gaps[0])
    assert gaps[1] == 2 * 3600


def test_gap_since_same_measures_the_matching_value_only():
    df = frame(['a'] * 3, [0, 1, 5], [1.0] * 3, ['x', 'y', 'x'])
    gaps = history_of(df).gap_since_same(df['value'])
    assert np.isnan(gaps[0])
    assert np.isnan(gaps[1])
    assert gaps[2] == 5 * 3600


def test_gap_since_same_is_nan_exactly_where_first_occurrence_is_one():
    rng = np.random.default_rng(4)
    n = 300
    df = frame(
        rng.choice(['a', 'b', 'c'], n),
        np.sort(rng.integers(0, 500, n)),
        rng.random(n) * 100,
        rng.choice(['p', 'q', 'r', 's'], n),
    )
    history = history_of(df)
    firsts = history.first_occurrence(df['value']) == 1.0
    assert np.array_equal(firsts, np.isnan(history.gap_since_same(df['value'])))


# --- rolling_distinct against brute force ---------------------------------


def brute_force_distinct(df, window):
    """Distinct values in each row's trailing window, computed the obvious way."""
    span = pd.Timedelta(window)
    out = []
    for _, row in df.iterrows():
        past = df[
            (df['entity_id'] == row['entity_id'])
            & (df['event_time'] < row['event_time'])
            & (df['event_time'] >= row['event_time'] - span)
        ]
        out.append(float(past['value'].nunique()))
    return out


def test_rolling_distinct_matches_brute_force():
    rng = np.random.default_rng(11)
    n = 400
    df = frame(
        rng.choice(['a', 'b', 'c'], n),
        np.sort(rng.integers(0, 200, n)),
        rng.random(n),
        rng.choice(['p', 'q', 'r', 's', 't'], n),
    )
    computed = history_of(df).rolling_distinct(df['value'], '24h')
    assert list(computed) == brute_force_distinct(df, '24h')


def test_rolling_distinct_excludes_the_current_row():
    df = frame(['a', 'a'], [0, 1], [1.0, 1.0], ['x', 'x'])
    assert list(history_of(df).rolling_distinct(df['value'], '24h')) == [0.0, 1.0]


def test_rolling_distinct_excludes_rows_sharing_this_row_timestamp():
    """Coarse timestamps make ties the common case, not an edge case.

    IBM CCF has minute resolution and no seconds. A tied row must not count as this
    row's past, and the pandas-backed windows agree.
    """
    df = frame(['a'] * 3, [0, 0, 0], [1.0] * 3, ['x', 'y', 'z'])
    history = history_of(df)
    assert list(history.rolling_distinct(df['value'], '24h')) == [0.0, 0.0, 0.0]
    assert list(history.rolling_count('24h')) == [0.0, 0.0, 0.0]


def test_rolling_distinct_agrees_with_rolling_count_when_values_are_unique():
    rng = np.random.default_rng(7)
    n = 250
    df = frame(
        rng.choice(['a', 'b'], n),
        np.sort(rng.integers(0, 100, n)),
        rng.random(n),
        [f'v{i}' for i in range(n)],
    )
    history = history_of(df)
    assert list(history.rolling_distinct(df['value'], '24h')) == list(history.rolling_count('24h'))


def test_rolling_distinct_forgets_values_outside_the_window():
    df = frame(['a'] * 3, [0, 1, 49], [1.0] * 3, ['x', 'y', 'z'])
    assert list(history_of(df).rolling_distinct(df['value'], '24h')) == [0.0, 1.0, 0.0]


# --- results land on the right rows --------------------------------------


def test_results_follow_the_input_order_not_the_sorted_order():
    ordered = frame(['a', 'b', 'a', 'b'], [0, 0, 1, 2], [1.0, 2.0, 3.0, 4.0])
    shuffled = ordered.iloc[[3, 0, 2, 1]].reset_index(drop=True)

    on_ordered = history_of(ordered).rolling_count('24h')
    on_shuffled = history_of(shuffled).rolling_count('24h')
    assert list(on_shuffled) == [on_ordered[i] for i in [3, 0, 2, 1]]


def test_tied_timestamps_keep_input_order():
    # Every row shares one timestamp, so only the input order can break the tie.
    df = frame(['a'] * 3, [0, 0, 0], [1.0, 2.0, 3.0])
    assert list(history_of(df).ordinal()) == [0.0, 1.0, 2.0]


def test_a_future_row_cannot_change_an_earlier_one():
    df = frame(['a'] * 4, [0, 1, 2, 3], [10.0, 20.0, 30.0, 40.0])
    history = history_of(df)
    before = history.rolling_sum(df['amount'], '24h')

    tampered = df.copy()
    tampered.loc[3, 'amount'] = 10_000_000.0
    after = history_of(tampered).rolling_sum(tampered['amount'], '24h')

    assert list(before[:3]) == list(after[:3])


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match='same length'):
        EntityHistory(pd.Series(['a', 'b'], dtype='string'), pd.Series([BASE]))


def test_unknown_window_names_the_known_ones():
    df = frame(['a'], [0], [1.0])
    with pytest.raises(ValueError, match="unknown window '3h'"):
        history_of(df).rolling_count('3h')


# --- arithmetic ---------------------------------------------------------


def test_safe_ratio_returns_zero_where_the_result_is_not_finite():
    assert list(safe_ratio([1.0, 1.0, np.nan], [2.0, 0.0, 1.0])) == [0.5, 0.0, 0.0]


def test_signed_log1p_keeps_the_sign_of_a_refund():
    assert signed_log1p([-1.0])[0] == pytest.approx(-np.log(2))


def test_haversine_km_matches_a_known_distance():
    # Prague to Vienna, about 250 km.
    computed = haversine_km([50.0755], [14.4378], [48.2082], [16.3738])
    assert computed[0] == pytest.approx(250.0, abs=10.0)


def test_haversine_km_is_zero_at_the_same_point():
    assert haversine_km([10.0], [20.0], [10.0], [20.0])[0] == pytest.approx(0.0)


def test_days_between_is_signed_and_fractional():
    later = pd.Series([BASE + pd.Timedelta(hours=36)])
    assert days_between(later, pd.Series([BASE]))[0] == pytest.approx(1.5)


def test_clock_features_are_cyclical_and_carry_no_year():
    clock = clock_features(pd.Series([pd.Timestamp('2020-03-07 23:30:00')]))
    assert 'year' not in clock.columns
    assert clock.loc[0, 'hour'] == 23
    assert clock.loc[0, 'is_weekend'] == 1.0
    assert clock.loc[0, 'hour_cos'] == pytest.approx(np.cos(2 * np.pi * 23 / 24))


def test_amount_shape_reads_the_number_not_the_magnitude():
    shape = amount_shape(pd.Series([100.0, 0.05, -20.0, 12.34]))
    assert list(shape['amount_is_round_100']) == [1.0, 0.0, 0.0, 0.0]
    assert list(shape['amount_is_micro']) == [0.0, 1.0, 0.0, 0.0]
    assert list(shape['amount_is_refund']) == [0.0, 0.0, 1.0, 0.0]
    assert list(shape['amount_cents']) == [0.0, 5.0, 0.0, 34.0]


def test_parse_money_handles_dollar_strings_and_passes_numbers_through():
    assert list(parse_money(pd.Series(['$1,234.50', '$-25.00']))) == [1234.5, -25.0]
    assert list(parse_money(pd.Series([1.5, 2.5]))) == [1.5, 2.5]


# --- the parquet contract ----------------------------------------------


def keys_frame(n=3):
    return pd.DataFrame(
        {
            'entity_id': pd.Series(['a'] * n, dtype='string'),
            'event_time': [BASE + pd.Timedelta(hours=i) for i in range(n)],
            'reported_at': [pd.NaT] * n,
            'is_fraud': [False] * n,
            'split': ['train'] * n,
        }
    )


def test_write_features_writes_float32_and_category(tmp_path):
    features = pd.DataFrame({'count_24h': [1, 2, 3], 'artifact_city': ['x', 'y', None]})
    written = pd.read_parquet(write_features('demo', keys_frame(), features, features_dir=tmp_path))

    assert written['count_24h'].dtype == 'float32'
    assert isinstance(written['artifact_city'].dtype, pd.CategoricalDtype)
    # Unfitted: the levels are named, not coded or capped.
    assert set(written['artifact_city'].cat.categories) == {'x', 'y', '~na'}


def test_write_features_puts_the_keys_first(tmp_path):
    features = pd.DataFrame({'z_feature': [1.0, 2.0, 3.0]})
    written = pd.read_parquet(write_features('demo', keys_frame(), features, features_dir=tmp_path))
    assert list(written.columns) == [*KEY_COLUMNS, 'z_feature']


def test_write_features_rejects_a_label_descriptive_column(tmp_path):
    # SAML-D's typology restates the label for the positive rows.
    features = pd.DataFrame({'count_24h': [1.0, 2.0, 3.0], 'Laundering_type': list('abc')})
    with pytest.raises(Exception, match='must never reach a model'):
        write_features('demo', keys_frame(), features, features_dir=tmp_path)


def test_write_features_rejects_a_missing_key(tmp_path):
    keys = keys_frame().drop(columns=['reported_at'])
    with pytest.raises(FeatureContractError, match='missing'):
        write_features('demo', keys, pd.DataFrame({'a': [1.0] * 3}), features_dir=tmp_path)


def test_write_features_rejects_a_raw_column_smuggled_in_as_a_key(tmp_path):
    keys = keys_frame().assign(merchant=['x', 'y', 'z'])
    with pytest.raises(FeatureContractError, match='non-key column'):
        write_features('demo', keys, pd.DataFrame({'a': [1.0] * 3}), features_dir=tmp_path)


def test_write_features_rejects_a_column_that_is_both_key_and_feature(tmp_path):
    features = pd.DataFrame({'entity_id': ['a'] * 3})
    with pytest.raises(FeatureContractError, match='both a key and a feature'):
        write_features('demo', keys_frame(), features, features_dir=tmp_path)


def test_write_features_rejects_infinities(tmp_path):
    features = pd.DataFrame({'ratio': [1.0, np.inf, 3.0]})
    with pytest.raises(FeatureContractError, match='infinities'):
        write_features('demo', keys_frame(), features, features_dir=tmp_path)


def test_write_features_rejects_a_row_count_mismatch(tmp_path):
    with pytest.raises(FeatureContractError, match='key rows against'):
        write_features(
            'demo', keys_frame(3), pd.DataFrame({'a': [1.0, 2.0]}), features_dir=tmp_path
        )


def test_write_features_accepts_an_extra_key(tmp_path):
    keys = keys_frame().assign(reported_at_slow=[pd.NaT] * 3)
    written = pd.read_parquet(
        write_features(
            'demo',
            keys,
            pd.DataFrame({'a': [1.0] * 3}),
            features_dir=tmp_path,
            extra_keys=('reported_at_slow',),
        )
    )
    assert 'reported_at_slow' in written.columns
    assert feature_columns(written) == ['a']


def test_feature_and_artifact_columns_partition_the_features(tmp_path):
    features = pd.DataFrame(
        {'count_24h': [1.0] * 3, 'artifact_city': list('xyz'), 'same_state': [1.0] * 3}
    )
    written = pd.read_parquet(write_features('demo', keys_frame(), features, features_dir=tmp_path))
    assert artifact_columns(written) == ['artifact_city']
    assert set(feature_columns(written)) == set(features.columns)
    assert not any(c.startswith(ARTIFACT_PREFIX) for c in KEY_COLUMNS)


def test_write_features_leaves_no_temporary_file_behind(tmp_path):
    written = write_features(
        'demo', keys_frame(), pd.DataFrame({'a': [1.0] * 3}), features_dir=tmp_path
    )
    assert written.exists()
    assert not list(tmp_path.glob('*.tmp'))


def test_a_failed_write_leaves_neither_a_temporary_nor_a_truncated_parquet(tmp_path):
    """A walltime kill mid-write must not leave a reader a half-built file."""
    good = write_features(
        'demo', keys_frame(), pd.DataFrame({'a': [1.0] * 3}), features_dir=tmp_path
    )
    before = good.read_bytes()

    class Exploding(pd.DataFrame):
        def to_parquet(self, *args, **kwargs):
            raise KeyboardInterrupt('killed mid-write')

    import fraud_benchmark.experiments.features.util as util

    original = util.pd.concat
    util.pd.concat = lambda *a, **k: Exploding(original(*a, **k))
    try:
        with pytest.raises(KeyboardInterrupt):
            write_features(
                'demo',
                keys_frame(),
                pd.DataFrame({'a': [2.0] * 3}),
                features_dir=tmp_path,
            )
    finally:
        util.pd.concat = original

    # The previous build survives untouched, and no .tmp is left to be mistaken
    # for a build in progress.
    assert good.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
