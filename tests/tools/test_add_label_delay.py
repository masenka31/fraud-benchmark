"""The standalone `tools/label_delay/add_label_delay.py` against the package.

The script is a copy of `campaigns.assign_campaigns` and `delay.assign_reported_at`
so that it can be handed to someone without this repository. A copy can drift, so
the parity tests assert *identical* output to the package on the same inputs; the
rest cover what only the script does: reading, column mapping, regimes and the
sidecar.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.data.campaigns import assign_campaigns
from fraud_benchmark.data.delay import DelayParams
from fraud_benchmark.data.delay import assign_reported_at

SCRIPT = Path(__file__).parents[2] / 'tools' / 'label_delay' / 'add_label_delay.py'


def load_script():
    spec = importlib.util.spec_from_file_location('_add_label_delay', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look their module up in sys.modules while the class is built.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = load_script()


def transactions(n=4_000, seed=1):
    """Random transactions with bursty frauds, integer entities and timestamp ties."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2020-01-01')
    minutes = rng.integers(0, 60 * 24 * 365, size=n)
    return pd.DataFrame(
        {
            'card': rng.integers(0, 120, size=n),
            'ts': (start + pd.to_timedelta(minutes, unit='min')).astype('datetime64[us]'),
            'label': (rng.random(n) < 0.08).astype(int),
            'amount': rng.random(n),
        }
    )


def canonical(df):
    """The frame the package sees for `transactions()`: the pipeline's dtypes."""
    return pd.DataFrame(
        {
            'entity_id': df['card'].astype(str).astype('string'),
            'event_time': df['ts'].astype('datetime64[us]'),
            'is_fraud': df['label'].astype(bool),
        },
        index=df.index,
    )


# --- parity with the package -------------------------------------------------


@pytest.mark.parametrize('gap', ['1D', '6h', '0s'])
def test_campaigns_match_the_package(gap):
    df = transactions()
    expected = assign_campaigns(canonical(df), gap=pd.Timedelta(gap))
    actual = tool.assign_campaigns(canonical(df), gap=pd.Timedelta(gap))
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    ('median', 'sigma', 'cap', 'seed'),
    [(7.0, 1.0, None, 0), (15.0, 1.665, 365.0, 0), (30.0, 1.0, None, 3)],
)
def test_reported_at_matches_the_package(median, sigma, cap, seed):
    df = canonical(transactions())
    df['campaign_id'] = assign_campaigns(df, gap=pd.Timedelta('1D'))
    expected = assign_reported_at(df, DelayParams(median, sigma, seed, cap))
    actual = tool.assign_reported_at(df, tool.Regime('', median, sigma, cap), seed)
    pd.testing.assert_series_equal(actual, expected)


def test_end_to_end_matches_the_package(tmp_path):
    df = transactions()
    src = tmp_path / 'in.parquet'
    df.to_parquet(src)
    out = tmp_path / 'out.parquet'
    tool.main([str(src), '-o', str(out), '--entity', 'card', '--time', 'ts', '--fraud', 'label'])

    expected = canonical(df)
    expected['campaign_id'] = assign_campaigns(expected, gap=pd.Timedelta('1D'))
    expected_reported = assign_reported_at(expected, DelayParams(7.0, 1.0, 0))

    written = pd.read_parquet(out)
    pd.testing.assert_series_equal(written['campaign_id'], expected['campaign_id'])
    pd.testing.assert_series_equal(written['reported_at'], expected_reported)


# --- what only the script does -----------------------------------------------


def run(tmp_path, df, *args, name='in.parquet'):
    src = tmp_path / name
    if name.endswith('.csv'):
        df.to_csv(src, index=False)
    else:
        df.to_parquet(src)
    tool.main([str(src), '--entity', 'card', '--time', 'ts', '--fraud', 'label', *args])
    return src


def test_input_columns_and_row_order_pass_through(tmp_path):
    df = transactions(n=500)
    src = run(tmp_path, df)
    written = pd.read_parquet(src.with_name('in.label_delay.parquet'))
    pd.testing.assert_frame_equal(written[df.columns], df)
    assert list(written.columns) == [*df.columns, 'campaign_id', 'reported_at']


def test_csv_round_trip(tmp_path):
    df = transactions(n=500)
    df['ts'] = df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    src = run(tmp_path, df, name='in.csv')
    written = pd.read_csv(src.with_name('in.label_delay.csv'))
    assert written['reported_at'].notna().sum() == df['label'].sum()


def test_numeric_time_needs_an_origin(tmp_path):
    df = transactions(n=200)
    df['ts'] = (df['ts'] - pd.Timestamp('2020-01-01')).dt.total_seconds()
    with pytest.raises(SystemExit):
        run(tmp_path, df)


def test_numeric_time_with_origin_matches_datetimes(tmp_path):
    df = transactions(n=500)
    as_dates = run(tmp_path, df, name='dates.parquet')
    numeric = df.assign(ts=(df['ts'] - pd.Timestamp('2020-01-01')).dt.total_seconds())
    as_seconds = run(tmp_path, numeric, '--time-origin', '2020-01-01', name='secs.parquet')
    a = pd.read_parquet(as_dates.with_name('dates.label_delay.parquet'))
    b = pd.read_parquet(as_seconds.with_name('secs.label_delay.parquet'))
    pd.testing.assert_series_equal(a['reported_at'], b['reported_at'])


def test_existing_output_column_is_refused(tmp_path):
    df = transactions(n=200).assign(reported_at=pd.NaT)
    with pytest.raises(SystemExit):
        run(tmp_path, df)


def test_explicit_flag_overrides_preset():
    args = tool.parse_args([
        'x.parquet', '--entity', 'e', '--time', 't', '--fraud', 'f',
        '--preset', 'card-slow', '--median-days', '20',
    ])  # fmt: skip
    regime = tool.base_regime(args)
    assert (regime.median_days, regime.sigma, regime.max_delay_days) == (20.0, 1.665, 365.0)


def test_extra_regimes_share_campaigns_and_rows(tmp_path):
    df = transactions(n=2_000)
    src = run(tmp_path, df, '--regime', 'slow:15,1.665,365', '--regime', 'aml:30,1.0')
    written = pd.read_parquet(src.with_name('in.label_delay.parquet'))
    fraud = written['label'] == 1
    for column in ('reported_at', 'reported_at_slow', 'reported_at_aml'):
        assert written.loc[fraud, column].notna().all()
        assert written.loc[~fraud, column].isna().all()
    expected = canonical(df)
    expected['campaign_id'] = written['campaign_id']
    slow = assign_reported_at(expected, DelayParams(15.0, 1.665, 0, 365.0))
    pd.testing.assert_series_equal(written['reported_at_slow'], slow.rename('reported_at_slow'))


def test_sidecar_records_parameters_and_summary(tmp_path):
    df = transactions(n=1_000)
    src = run(tmp_path, df, '--cutoff', '2020-09-01', '--seed', '4')
    card = json.loads(src.with_name('in.label_delay.parquet.json').read_text())
    assert card['seed'] == 4
    assert card['campaign_gap'] == '1 days 00:00:00'
    (regime,) = card['regimes']
    assert regime['column'] == 'reported_at'
    assert regime['median_days'] == 7.0
    assert 0.0 <= regime['censored_at_cutoff']['share'] <= 1.0
    assert card['n_campaigns'] > 0


# --- edge cases from the review of PR #7 ------------------------------------------


def main_with(src, out, *args):
    tool.main(
        [str(src), '-o', str(out), '--entity', 'card', '--time', 'ts', '--fraud', 'label', *args]
    )


def test_csv_entity_ids_keep_leading_zeros(tmp_path):
    """'0007' and '7' are different cards, and CSV must agree with Parquet."""
    df = transactions(n=1_000)
    df['card'] = df['card'].map(lambda c: f'{c:04d}' if c % 2 else str(c))
    a = run(tmp_path, df, name='p.parquet')
    b = run(tmp_path, df, name='c.csv')
    from_parquet = pd.read_parquet(a.with_name('p.label_delay.parquet'))
    from_csv = pd.read_csv(b.with_name('c.label_delay.csv'), dtype={'card': str})
    assert from_csv['card'].equals(df['card'])
    assert from_csv['campaign_id'].equals(from_parquet['campaign_id'].astype('float64'))
    reported = pd.to_datetime(from_csv['reported_at']).astype('datetime64[us]')
    assert reported.equals(from_parquet['reported_at'])


def test_mixed_utc_offsets_in_csv_are_read(tmp_path):
    df = transactions(n=500)
    df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').astype(str)
    src = run(tmp_path, df, name='in.csv')
    written = pd.read_csv(src.with_name('in.label_delay.csv'))
    assert written['reported_at'].notna().sum() == df['label'].sum()


def test_tz_aware_input_matches_utc_and_writes_utc_aware(tmp_path):
    df = transactions(n=1_000)
    naive = run(tmp_path, df, name='naive.parquet')
    aware = df.assign(ts=df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Prague'))
    src = run(tmp_path, aware, '--cutoff', '2020-07-01T00:00+02:00', name='aware.parquet')
    a = pd.read_parquet(naive.with_name('naive.label_delay.parquet'))['reported_at']
    b = pd.read_parquet(src.with_name('aware.label_delay.parquet'))['reported_at']
    assert str(b.dt.tz) == 'UTC'
    pd.testing.assert_series_equal(b.dt.tz_localize(None), a)
    # The written column compares directly with the input's own tz-aware times.
    assert (b.dropna() >= aware.loc[b.notna(), 'ts']).all()


def test_cutoff_timezone_must_match_the_data(tmp_path):
    aware = transactions(n=200)
    aware['ts'] = aware['ts'].dt.tz_localize('UTC')
    with pytest.raises(SystemExit):
        run(tmp_path, aware, '--cutoff', '2020-07-01', name='a.parquet')
    with pytest.raises(SystemExit):
        run(tmp_path, transactions(n=200), '--cutoff', '2020-07-01T00:00Z', name='n.parquet')


@pytest.mark.parametrize('gap', ['-1D', 'NaT'])
def test_invalid_campaign_gap_is_refused(tmp_path, gap):
    with pytest.raises(SystemExit):
        run(tmp_path, transactions(n=200), f'--campaign-gap={gap}')


def test_sidecar_does_not_clobber_a_sibling_json(tmp_path):
    (tmp_path / 'meta.json').write_text('keep')
    src = tmp_path / 'in.parquet'
    transactions(n=200).to_parquet(src)
    main_with(src, tmp_path / 'meta.parquet')
    assert (tmp_path / 'meta.json').read_text() == 'keep'
    assert (tmp_path / 'meta.parquet.json').exists()


def test_output_may_not_overwrite_the_input(tmp_path):
    src = tmp_path / 'in.parquet'
    transactions(n=200).to_parquet(src)
    with pytest.raises(SystemExit):
        main_with(src, src)


def test_bad_output_format_is_refused_before_reading(tmp_path):
    with pytest.raises(SystemExit):
        main_with(tmp_path / 'does_not_exist.parquet', tmp_path / 'out.xlsx')


def test_csv_output_from_parquet_input(tmp_path):
    src = tmp_path / 'in.parquet'
    df = transactions(n=300)
    df.to_parquet(src)
    main_with(src, tmp_path / 'out.csv')
    assert pd.read_csv(tmp_path / 'out.csv')['reported_at'].notna().sum() == df['label'].sum()


def test_time_origin_on_a_datetime_column_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, transactions(n=200), '--time-origin', '2020-01-01')


@pytest.mark.parametrize(
    'labels', [['true', 'False'], ['1', '0'], [True, False], [1.0, 0.0]], ids=str
)
def test_fraud_label_forms_are_accepted(tmp_path, labels):
    df = transactions(n=200)
    df['label'] = [labels[0] if y else labels[1] for y in df['label']]
    src = run(tmp_path, df)
    written = pd.read_parquet(src.with_name('in.label_delay.parquet'))
    assert written['reported_at'].notna().sum() == (df['label'] == labels[0]).sum()


@pytest.mark.parametrize('labels', [[2, 0], ['yes', 'no']], ids=str)
def test_other_fraud_labels_are_refused(tmp_path, labels):
    df = transactions(n=200)
    df['label'] = [labels[0] if y else labels[1] for y in df['label']]
    with pytest.raises(SystemExit):
        run(tmp_path, df)


def test_missing_fraud_label_is_refused(tmp_path):
    df = transactions(n=200).astype({'label': 'float64'})
    df.loc[0, 'label'] = np.nan
    with pytest.raises(SystemExit):
        run(tmp_path, df)


def test_no_frauds_writes_empty_columns(tmp_path):
    src = run(tmp_path, transactions(n=200).assign(label=0))
    written = pd.read_parquet(src.with_name('in.label_delay.parquet'))
    assert written['campaign_id'].isna().all()
    assert written['reported_at'].isna().all()
