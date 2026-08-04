import json
import os
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.data.config import Config
from fraud_benchmark.data.delay import DelayParams
from fraud_benchmark.data.pipeline import _drop_source_label
from fraud_benchmark.data.pipeline import prepare

FIXTURES = Path(__file__).parent.parent / 'fixtures'
FIXTURE = FIXTURES / 'paysim'


@pytest.fixture
def config(tmp_path):
    return Config(
        raw_dir=tmp_path / 'raw',
        processed_dir=tmp_path / 'processed',
        split_ratios=(0.6, 0.2, 0.2),
        delay=DelayParams(median_days=7.0, sigma=1.0, seed=0),
        campaign_gap=pd.Timedelta(days=1),
        datasets={'paysim': {'start_date': '2023-01-01'}},
    )


@pytest.fixture
def no_download(monkeypatch):
    """Pretend the raw files are already downloaded, using the test fixture."""

    def fake_fetch(source, dest, *, force=False):
        return FIXTURE

    monkeypatch.setattr('fraud_benchmark.data.pipeline.fetch', fake_fetch)


def test_prepare_writes_parquet(config, no_download):
    out = prepare('paysim', config)
    assert (out / 'data.parquet').exists()
    assert (out / 'dataset_card.json').exists()


def test_output_has_core_columns_first(config, no_download):
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')
    assert list(df.columns)[:5] == [
        'event_time',
        'entity_id',
        'amount',
        'is_fraud',
        'split',
    ]


def test_output_row_count_matches_fixture(config, no_download):
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')
    assert len(df) == 5


def test_every_row_has_a_split(config, no_download):
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')
    assert df['split'].notna().all()
    assert set(df['split']) <= {'train', 'val', 'test'}


def test_output_is_sorted_by_event_time(config, no_download):
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')
    assert df['event_time'].is_monotonic_increasing


def test_dataset_card_records_counts_and_provenance(config, no_download):
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['name'] == 'paysim'
    assert card['n_rows'] == 5
    assert card['n_fraud'] == 2
    assert card['source']['handle'] == 'ealaxi/paysim1'
    assert card['column_mapping']['entity_id'] == 'nameOrig'
    assert card['caveats']
    assert card['split']['ratios'] == [0.6, 0.2, 0.2]
    assert 'train' in card['split']['counts']


def test_rerun_replaces_previous_output(config, no_download):
    first = prepare('paysim', config)
    stale = first / 'stale.txt'
    stale.write_text('left over from a previous run')
    prepare('paysim', config)
    assert not stale.exists()
    assert (first / 'data.parquet').exists()


def test_reported_at_is_written(config, no_download):
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')
    assert 'reported_at' in df.columns
    assert df.loc[df.is_fraud, 'reported_at'].notna().all()
    assert df.loc[~df.is_fraud, 'reported_at'].isna().all()


def test_campaign_id_is_written(config, no_download):
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')
    assert 'campaign_id' in df.columns
    assert df.loc[~df.is_fraud, 'campaign_id'].isna().all()


def test_reported_at_never_precedes_the_transaction(config, no_download):
    out = prepare('paysim', config)
    df = pd.read_parquet(out / 'data.parquet')
    fraud = df[df.is_fraud]
    assert (fraud['reported_at'] >= fraud['event_time']).all()


def test_the_run_is_reproducible(config, no_download):
    """Same seed, same output — otherwise the benchmark is not comparable."""
    first = pd.read_parquet(prepare('paysim', config) / 'data.parquet')
    second = pd.read_parquet(prepare('paysim', config) / 'data.parquet')
    assert first['reported_at'].equals(second['reported_at'])


def test_card_records_the_delay_parameters(config, no_download):
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['label_delay']['distribution'] == 'lognormal'
    assert card['label_delay']['median_days'] == 7.0
    assert card['label_delay']['seed'] == 0
    assert card['label_delay']['campaign_gap'] == '1 days 00:00:00'
    assert card['label_delay']['n_campaigns'] >= 1
    assert card['label_delay']['largest_campaign'] >= 1


def test_no_temp_directory_is_left_behind(config, no_download):
    prepare('paysim', config)
    leftovers = list(config.processed_dir.glob('*.tmp*'))
    assert leftovers == []


def test_validation_failure_writes_nothing(config, monkeypatch):
    def fake_fetch(source, dest, *, force=False):
        return FIXTURE

    monkeypatch.setattr('fraud_benchmark.data.pipeline.fetch', fake_fetch)
    monkeypatch.setattr(
        'fraud_benchmark.data.pipeline.validate_canonical',
        lambda df: (_ for _ in ()).throw(ValueError('boom')),
    )
    with pytest.raises(ValueError, match='boom'):
        prepare('paysim', config)
    assert not (config.processed_dir / 'paysim').exists()


def test_failed_swap_preserves_previous_output(config, no_download, monkeypatch):
    """The old output must survive a rewrite that dies during the swap itself.

    The failure is injected at os.replace, i.e. after the current code has
    already removed dest. Patching an earlier step (to_parquet) would not
    exercise this: it raises before dest is ever touched.
    """
    out = prepare('paysim', config)
    marker = out / 'marker.txt'
    marker.write_text('previous good run')
    original = (out / 'data.parquet').read_bytes()

    def boom(src, dst):
        raise OSError('swap interrupted')

    monkeypatch.setattr('fraud_benchmark.data.pipeline.os.replace', boom)

    with pytest.raises(OSError, match='swap interrupted'):
        prepare('paysim', config)

    assert marker.exists(), 'previous good output was destroyed by a failed swap'
    assert (out / 'data.parquet').read_bytes() == original


def test_concurrent_swap_does_not_raise(config, no_download, monkeypatch):
    """A racing process may move dest aside first; that must not be an error."""
    prepare('paysim', config)

    real_rename = os.rename
    calls = []

    def racing_rename(src, dst):
        # Simulate another process having already moved dest away: the racer's
        # rename actually happens (src is gone by the time we look), and our
        # own rename call observes that as FileNotFoundError.
        if not calls:
            calls.append((src, dst))
            real_rename(src, config.processed_dir / 'stolen-by-racer')
            raise FileNotFoundError(2, 'No such file or directory')
        return real_rename(src, dst)

    monkeypatch.setattr('fraud_benchmark.data.pipeline.os.rename', racing_rename)

    out = prepare('paysim', config)
    assert (out / 'data.parquet').exists()


def test_no_backup_directory_is_left_behind(config, no_download):
    prepare('paysim', config)
    prepare('paysim', config)
    leftovers = sorted(p.name for p in config.processed_dir.iterdir())
    assert leftovers == ['paysim'], f'leftover directories: {leftovers}'


def test_non_serializable_option_does_not_break_the_card(config, no_download, tmp_path):
    config.datasets['paysim']['scratch'] = tmp_path / 'somewhere'
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert 'scratch' in card['options']


def test_dataset_card_records_the_data_license(config, no_download):
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['data_license'] == 'CC BY-SA 4.0'


def test_dataset_card_records_commercial_use(config, no_download):
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['commercial_use'] is True


def test_card_records_the_shared_split_strategy(config, no_download):
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['split']['strategy'] == 'temporal, cut on timestamp values'
    assert card['split']['ratios'] == [0.6, 0.2, 0.2]


def test_auxiliary_frames_are_written_alongside(config, no_download, monkeypatch):
    import pandas as pd

    from fraud_benchmark.data.adapters.base import get_adapter

    adapter = get_adapter('paysim')
    monkeypatch.setattr(
        type(adapter),
        'auxiliary_frames',
        lambda self, raw_dir, options: {'extra': pd.DataFrame({'a': [1, 2, 3]})},
    )
    out = prepare('paysim', config)
    assert (out / 'extra.parquet').exists()
    assert len(pd.read_parquet(out / 'extra.parquet')) == 3


def test_auxiliary_frames_are_recorded_in_the_card(config, no_download, monkeypatch):
    import pandas as pd

    from fraud_benchmark.data.adapters.base import get_adapter

    adapter = get_adapter('paysim')
    monkeypatch.setattr(
        type(adapter),
        'auxiliary_frames',
        lambda self, raw_dir, options: {'extra': pd.DataFrame({'a': [1, 2, 3]})},
    )
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['auxiliary'] == {'extra': 3}


def test_no_auxiliary_key_when_there_are_none(config, no_download):
    out = prepare('paysim', config)
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['auxiliary'] == {}
    assert not list(out.glob('extra*.parquet'))


@pytest.mark.parametrize('bad_key', ['../escape', 'sub/dir', '', '.', '..'])
def test_auxiliary_keys_that_are_not_plain_filenames_are_rejected(
    config, no_download, monkeypatch, bad_key
):
    import pandas as pd

    from fraud_benchmark.data.adapters.base import get_adapter

    monkeypatch.setattr(
        type(get_adapter('paysim')),
        'auxiliary_frames',
        lambda self, raw_dir, options: {bad_key: pd.DataFrame({'a': [1]})},
    )
    with pytest.raises(ValueError, match='not a valid filename'):
        prepare('paysim', config)


def test_a_failing_auxiliary_write_preserves_previous_output(config, no_download, monkeypatch):
    """The highest-value case: an aux write that dies must not cost the good run."""
    out = prepare('paysim', config)
    marker = out / 'marker.txt'
    marker.write_text('previous good run')
    original = (out / 'data.parquet').read_bytes()

    class _Exploding(pd.DataFrame):
        @property
        def _constructor(self):
            return _Exploding

        def to_parquet(self, *args, **kwargs):
            raise OSError('aux write failed')

    from fraud_benchmark.data.adapters.base import get_adapter

    monkeypatch.setattr(
        type(get_adapter('paysim')),
        'auxiliary_frames',
        lambda self, raw_dir, options: {'boom': _Exploding({'a': [1]})},
    )

    with pytest.raises(OSError, match='aux write failed'):
        prepare('paysim', config)

    assert marker.exists(), 'previous good output was destroyed by a failed aux write'
    assert (out / 'data.parquet').read_bytes() == original
    assert not list(config.processed_dir.glob('*.tmp*'))
    assert not list(config.processed_dir.glob('*.old*'))


@pytest.fixture
def config_with_override(tmp_path):
    return Config(
        raw_dir=tmp_path / 'raw',
        processed_dir=tmp_path / 'processed',
        split_ratios=(0.6, 0.2, 0.2),
        delay=DelayParams(median_days=7.0, sigma=1.0, seed=0),
        campaign_gap=pd.Timedelta(days=1),
        datasets={
            'paysim': {
                'start_date': '2023-01-01',
                'delay': {'median_days': 1.0},
            }
        },
    )


def test_the_override_changes_the_timestamps(config, config_with_override, no_download):
    """The same seed must still produce different delays under a different median."""
    base = pd.read_parquet(prepare('paysim', config) / 'data.parquet')
    other = pd.read_parquet(prepare('paysim', config_with_override) / 'data.parquet')
    assert not base['reported_at'].equals(other['reported_at'])


def test_the_card_records_the_resolved_delay(config_with_override, no_download):
    out = prepare('paysim', config_with_override)
    card = json.loads((out / 'dataset_card.json').read_text())
    # 1.0, not the global 7.0 — a card must never misreport what produced it.
    assert card['label_delay']['median_days'] == 1.0
    assert card['label_delay']['sigma'] == 1.0


@pytest.fixture
def run_prepare(tmp_path, monkeypatch):
    """Run `prepare` for any dataset against its committed fixture directory.

    The `no_download`/`config` pair above is paysim-only; the label-drop rule has
    to be checked on datasets whose source column is kept or already canonical.
    """

    def run(name, options=None):
        monkeypatch.setattr(
            'fraud_benchmark.data.pipeline.fetch',
            lambda source, dest, *, force=False: FIXTURES / name,
        )
        config = Config(
            raw_dir=tmp_path / 'raw',
            processed_dir=tmp_path / 'processed',
            split_ratios=(0.6, 0.2, 0.2),
            delay=DelayParams(median_days=7.0, sigma=1.0, seed=0),
            campaign_gap=pd.Timedelta(days=1),
            datasets={name: dict(options or {})},
        )
        return prepare(name, config)

    return run


def test_prepare_drops_the_source_label_column(run_prepare):
    """The output carries one binary label. A second one is an answer key."""
    out = run_prepare('paysim', {'start_date': '2023-01-01'})
    frame = pd.read_parquet(out / 'data.parquet')
    assert 'isFraud' not in frame.columns
    assert 'is_fraud' in frame.columns


def test_prepare_keeps_a_label_descriptive_column(run_prepare):
    """saml_d's typology is kept: is_fraud cannot express which typology it was."""
    out = run_prepare('saml_d')
    frame = pd.read_parquet(out / 'data.parquet')
    assert 'Laundering_type' in frame.columns
    assert 'Is_laundering' not in frame.columns


def test_the_card_records_the_dropped_label_column(run_prepare):
    """The output must say what was removed, or the drop is invisible."""
    out = run_prepare('paysim', {'start_date': '2023-01-01'})
    card = json.loads((out / 'dataset_card.json').read_text())
    assert card['dropped_source_label'] == 'isFraud'


def test_a_dataset_whose_source_label_is_already_canonical_keeps_it(run_prepare):
    """sparkov's source column IS is_fraud -- the drop must not delete the label."""
    out = run_prepare('sparkov')
    frame = pd.read_parquet(out / 'data.parquet')
    assert 'is_fraud' in frame.columns
    assert frame['is_fraud'].dtype == 'bool'


@pytest.mark.parametrize(
    ('name', 'source_label', 'options'),
    [
        ('paysim', 'isFraud', {'start_date': '2023-01-01'}),
        ('banksim', 'fraud', {'start_date': '2023-01-01'}),
        ('ibm_ccf', 'Is Fraud?', {'entity_key': 'user'}),
        ('ieee_cis', 'isFraud', {'start_date': '2017-12-01'}),
        ('saml_d', 'Is_laundering', {}),
    ],
)
def test_both_halves_of_the_drop_contract_hold_for_every_dropping_adapter(
    run_prepare, name, source_label, options
):
    """The raw label is gone AND is_fraud survived carrying the same rows.

    Only `prepare` can be observed for this: `to_canonical` does not drop, so the
    adapter-level tests assert the declaration instead. Each fixture has 5 rows,
    2 of them fraudulent.
    """
    frame = pd.read_parquet(run_prepare(name, options) / 'data.parquet')
    assert source_label not in frame.columns
    assert frame['is_fraud'].dtype == 'bool'
    assert int(frame['is_fraud'].sum()) == 2


def test_a_stale_source_label_declaration_fails_loudly():
    """The guard against the one failure mode this design introduces.

    An adapter that renames or mistypes its `source_label_column` would otherwise
    keep passing its raw label through under a name nobody is watching -- the
    silent version of exactly the bug the drop exists to prevent.
    """

    class _Stale:
        name = 'stale_probe'
        source_label_column = 'NotAColumn'
        label_descriptive_columns = ()

    frame = pd.DataFrame({'is_fraud': [True], 'amount': [1.0]})
    with pytest.raises(ValueError, match='NotAColumn'):
        _drop_source_label(frame, _Stale())
