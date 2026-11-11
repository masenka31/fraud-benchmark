#!/usr/bin/env python
"""Run the IBM endpoint and Sparkov full-row LSTM protocols."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.data.config import load_config
from fraud_benchmark.experiments.features.util import FEATURE_DIR
from fraud_benchmark.experiments.lstm import ARCHIVED_LSTM_CONFIGS
from fraud_benchmark.experiments.lstm import LSTM_CONFIGS
from fraud_benchmark.experiments.sequence import complete_chunks
from fraud_benchmark.experiments.sequence_experiment import IBM_ARCHIVED_REGIMES
from fraud_benchmark.experiments.sequence_experiment import IBM_REGIMES
from fraud_benchmark.experiments.sequence_experiment import IBM_TABLE3_REGIMES
from fraud_benchmark.experiments.sequence_experiment import SPARKOV_REGIMES
from fraud_benchmark.experiments.sequence_experiment import prepare_ibm
from fraud_benchmark.experiments.sequence_experiment import prepare_sparkov
from fraud_benchmark.experiments.sequence_experiment import run_ibm_xgboost
from fraud_benchmark.experiments.sequence_experiment import run_lstm
from fraud_benchmark.experiments.splits import pre_italy_split
from fraud_benchmark.experiments.splits import standard_split

PAPER = Path('results/paper')
ARCHIVE = Path('results/archive')
IBM_TABLE = ARCHIVE / 'ibm_sequence' / 'ibm_lstm_protocol.md'
SPARKOV_TABLE = ARCHIVE / 'sparkov_lstm' / 'sparkov_lstm_delay.md'
PAPER_TABLE = PAPER / 'ibm_lstm_split_comparison.md'
PAPER_CONFIG = 'sqrt_weight_slow'
SPARKOV_PAPER_DIR = PAPER / 'sparkov_lstm_delay'
SPARKOV_PAPER_TABLE = PAPER / 'sparkov_label_delay.md'
SPARKOV_PAPER_CONFIG = 'fixed15'
SPARKOV_PAPER_LENGTH = 10


def _is_paper_cell(
    dataset: str,
    model: str,
    regime: str,
    window_length: int,
    window_stride: int | None,
    config: str,
    feature_set: str = 'full',
) -> bool:
    """The paper's two LSTM tables.

    IBM split table: 68/16/16, length 10, stride 5, selected preset.
    Sparkov label-delay table: length 10, full features, `fixed15` preset.
    """
    if dataset == 'sparkov':
        return (
            model == 'lstm'
            and window_length == SPARKOV_PAPER_LENGTH
            and feature_set == 'full'
            and config == SPARKOV_PAPER_CONFIG
        )
    return (
        dataset == 'ibm'
        and model == 'lstm'
        and regime in IBM_TABLE3_REGIMES
        and window_length == 10
        and window_stride == 5
        and config == PAPER_CONFIG
    )


def _default_dir(
    dataset: str,
    model: str,
    regime: str,
    window_length: int,
    window_stride: int | None,
    config: str,
    feature_set: str = 'full',
) -> Path:
    if _is_paper_cell(dataset, model, regime, window_length, window_stride, config, feature_set):
        return SPARKOV_PAPER_DIR if dataset == 'sparkov' else PAPER
    if config in ARCHIVED_LSTM_CONFIGS:
        return ARCHIVE / 'lstm_presets'
    return ARCHIVE / ('ibm_sequence' if dataset == 'ibm' else 'sparkov_lstm')


def _path(
    dataset: str,
    model: str,
    regime: str,
    results_dir: Path | None,
    window_length: int = 30,
    feature_set: str = 'full',
    window_stride: int | None = None,
    config: str = 'default',
) -> Path:
    if results_dir is None:
        results_dir = _default_dir(
            dataset, model, regime, window_length, window_stride, config, feature_set
        )
    if feature_set == 'rawish':
        suffix = f'_rawish_len{window_length}'
    else:
        suffix = '' if window_length == 30 else f'_len{window_length}'
    if dataset == 'ibm' and window_stride is not None and window_stride != window_length:
        suffix += f'_stride{window_stride}'
    if config != 'default':
        suffix += f'_cfg-{config}'
    return results_dir / f'{dataset}_{model}_{regime}{suffix}.json'


def _read(
    dataset: str,
    model: str,
    regimes: tuple[str, ...],
    results_dir: Path | None,
    window_length: int = 30,
    feature_set: str = 'full',
) -> dict:
    return {
        regime: json.loads(
            _path(dataset, model, regime, results_dir, window_length, feature_set).read_text()
        )
        for regime in regimes
    }


def _values(record: dict, partition: str, metric: str) -> np.ndarray:
    if record['model'] == 'lstm':
        values = [row['scores'][partition][metric] for row in record['seeds']]
    else:
        name = 'validation' if partition == 'val' else partition
        values = [row[name][metric] for row in record['seeds']]
    return np.asarray(values)


def _mean_sd(record: dict, partition: str, metric: str) -> str:
    values = _values(record, partition, metric)
    return f'{np.mean(values):.4f} ± {np.std(values):.4f}'


def _test_minus_validation_ap(record: dict) -> str:
    difference = _values(record, 'test', 'average_precision') - _values(
        record, 'val', 'average_precision'
    )
    return f'{np.mean(difference):+.4f}'


def render_ibm(lstm: dict, xgboost: dict) -> str:
    rows = []
    labels = {
        'pre_italy': 'Temporal',
        'pre_italy_iid_chunks': 'Chunk IID',
        'pre_italy_iid_customers': 'Customer IID',
    }
    for regime in IBM_ARCHIVED_REGIMES:
        for model, records in (('LSTM', lstm), ('XGBoost', xgboost)):
            record = records[regime]
            rows.append(
                f'| {labels[regime]} | {model} | '
                f'{_mean_sd(record, "val", "average_precision")} | '
                f'{_mean_sd(record, "test", "average_precision")} | '
                f'{_test_minus_validation_ap(record)} | '
                f'{record["rows"]["test"]:,} | {record["positives"]["test"]:,} |'
            )
    population = lstm['pre_italy']['source_rows']
    targets = lstm['pre_italy']['target_rows']
    length = lstm['pre_italy']['chunk_length']
    return '\n'.join(
        [
            f'# IBM CCF disjoint {length}-transaction endpoint protocol',
            '',
            'Generated by `scripts/lstm_protocol.py`; do not edit by hand.',
            '',
            f'The source has {population:,} pre-Italy transactions. Exactly {targets:,} '
            'complete entity chunks contribute one final-transaction target each.',
            'All cells use the same endpoint targets and 77 label-free feature names.',
            f'XGBoost receives only the endpoint feature row; the LSTM receives all {length} rows.',
            'Chunk IID is a different split from the original transaction-IID protocol.',
            'These AP values are not comparable to the original full-row IBM table.',
            'Entries are mean ± population standard deviation over seeds 0–4.',
            'Test − validation AP is the mean within-seed difference; '
            'negative means lower test AP.',
            '',
            '| Split | Model | Validation AP | Test AP | Test − validation AP | '
            'Test targets | Test fraud targets |',
            '|---|---|---:|---:|---:|---:|---:|',
            *rows,
        ]
    )


def render_ibm_68_16_16_default_preset(lstm: dict[str, dict], xgboost: dict[str, dict]) -> str:
    """The Table 3 cells with the default LSTM preset and endpoint XGBoost (archived)."""
    temporal = lstm['pre_italy_68_16_16']
    length = temporal['chunk_length']
    stride = temporal['window_stride']
    labels = {
        'pre_italy_68_16_16': 'Temporal',
        'pre_italy_iid_chunks_68_16_16': 'Window IID',
        'pre_italy_iid_customers_68_16_16': 'Customer IID',
    }
    counts = []
    rows = []
    for regime in IBM_TABLE3_REGIMES:
        record = lstm[regime]
        for partition, label in (('train', 'Train'), ('val', 'Validation'), ('test', 'Test')):
            positives = record['positives'][partition]
            n_rows = record['rows'][partition]
            counts.append(
                f'| {labels[regime]} | {label} | {n_rows:,} | {positives:,} | '
                f'{100 * positives / n_rows:.6f}% |'
            )
        for model, records in (('LSTM', lstm), ('XGBoost', xgboost)):
            result = records[regime]
            rows.append(
                f'| {labels[regime]} | {model} | '
                f'{_mean_sd(result, "val", "average_precision")} | '
                f'{_mean_sd(result, "test", "average_precision")} | '
                f'{_test_minus_validation_ap(result)} |'
            )
    return '\n'.join(
        [
            f'# IBM CCF exploratory 68/16/16, length {length}, stride {stride}',
            '',
            'Generated by `scripts/lstm_protocol.py`; do not edit by hand.',
            '',
            f'The source has {temporal["source_rows"]:,} pre-Italy transactions and '
            f'{temporal["target_rows"]:,} complete same-entity windows.',
            'All cells score the same endpoint transactions with true labels and '
            'use 68/16/16 split shares.',
            f'XGBoost receives only the endpoint feature row; the LSTM receives {length} rows.',
            'The time cutoffs were selected after inspecting labels, so this is exploratory.',
            'Window IID assigns overlapping windows independently, so context rows can '
            'occur in multiple partitions; labels are not sequence inputs.',
            'Entries are mean ± population standard deviation over seeds 0–4.',
            '',
            '| Split | Partition | Endpoints | Fraud endpoints | Fraud rate |',
            '|---|---|---:|---:|---:|',
            *counts,
            '',
            '| Split | Model | Validation AP | Test AP | Test − validation AP |',
            '|---|---|---:|---:|---:|',
            *rows,
        ]
    )


def render_sparkov(records: dict) -> str:
    rows = []
    labels = {'off': 'No delay', 'on': 'Default delay', 'slow': 'Slow delay'}
    baseline = np.mean(
        [row['scores']['test']['average_precision'] for row in records['off']['seeds']]
    )
    length = records['off']['window_length']
    feature_set = records['off'].get('feature_set', 'full')
    for regime in SPARKOV_REGIMES:
        record = records[regime]
        ap = np.mean([row['scores']['test']['average_precision'] for row in record['seeds']])
        rows.append(
            f'| {labels[regime]} | {record["censored_train_labels"]:,} | '
            f'{_mean_sd(record, "val", "average_precision")} | '
            f'{_mean_sd(record, "test", "average_precision")} | '
            f'{_test_minus_validation_ap(record)} | {ap - baseline:+.4f} |'
        )
    return '\n'.join(
        [
            '# Sparkov LSTM synthetic label-delay protocol',
            '',
            'Generated by `scripts/lstm_protocol.py`; do not edit by hand.',
            '',
            'All three regimes score every Sparkov row with true evaluation labels.',
            f'The {length}-transaction windows, split, features, settings, and seeds are fixed;',
            'only synthetic training-label availability changes.',
            *(
                [
                    'Inputs are 21 raw-ish current-event/static features; '
                    'no entity-history statistics are model inputs.'
                ]
                if feature_set == 'rawish'
                else []
            ),
            'Entries are mean ± population standard deviation over seeds 0–4.',
            'Test − validation AP is the mean within-seed difference; '
            'negative means lower test AP.',
            '',
            '| Training labels | Hidden train frauds | Validation AP | Test AP | '
            'Test − validation AP | Δ test AP vs off |',
            '|---|---:|---:|---:|---:|---:|',
            *rows,
        ]
    )


def render_ibm_paper_split(lstm: dict[str, dict]) -> str:
    """The paper's temporal-versus-IID table: LSTM only, with a test − validation row."""
    temporal = lstm['pre_italy_68_16_16']
    columns = {
        'pre_italy_68_16_16': 'Temporal',
        'pre_italy_iid_chunks_68_16_16': 'Window IID',
        'pre_italy_iid_customers_68_16_16': 'Customer IID',
    }

    def gap(record: dict) -> str:
        difference = _values(record, 'test', 'average_precision') - _values(
            record, 'val', 'average_precision'
        )
        return f'{np.mean(difference):+.4f} ± {np.std(difference):.4f}'

    def row(label: str, cell) -> str:
        return f'| {label} | ' + ' | '.join(cell(lstm[regime]) for regime in columns) + ' |'

    return '\n'.join(
        [
            '# IBM CCF temporal versus IID splits (paper)',
            '',
            'Generated by `scripts/lstm_protocol.py`; do not edit by hand.',
            '',
            f'LSTM preset `{temporal["model_settings"]["config"]}`, selected on temporal '
            'validation AP. Exploratory 68/16/16 split of the pre-Italy rows; the cutoffs',
            'were chosen after inspecting labels so validation and test have similar fraud rates.',
            f'Complete windows of {temporal["chunk_length"]} transactions every '
            f'{temporal["window_stride"]} transactions; each window is scored on its final '
            'transaction.',
            'Window IID can share context rows across partitions.',
            'Entries are mean ± population standard deviation over seeds 0–4; the last row is '
            'the within-seed test − validation AP.',
            '',
            '| | ' + ' | '.join(columns.values()) + ' |',
            '|---|' + '---:|' * len(columns),
            row('Validation AP', lambda r: _mean_sd(r, 'val', 'average_precision')),
            row('Test AP', lambda r: _mean_sd(r, 'test', 'average_precision')),
            row('Test − validation AP', gap),
        ]
    )


def sparkov_paper_records(directory: Path = SPARKOV_PAPER_DIR) -> dict[str, dict]:
    """Merge the Sparkov label-delay records into one record per regime.

    The records are kept as run: seeds 0–4 in one file per regime, and each later seed
    in its own `seedN/` folder. Every field except `seeds` must agree between files of a
    regime, no seed may repeat, and all regimes must share everything except the
    training labels.
    """
    merged = {}
    for regime in SPARKOV_REGIMES:
        name = _path(
            'sparkov',
            'lstm',
            regime,
            directory,
            SPARKOV_PAPER_LENGTH,
            config=SPARKOV_PAPER_CONFIG,
        ).name
        paths = sorted(directory.rglob(name))
        if not paths:
            raise FileNotFoundError(f'no {name} under {directory}')
        records = [json.loads(path.read_text()) for path in paths]
        base = {key: value for key, value in records[0].items() if key != 'seeds'}
        seeds = []
        for path, record in zip(paths, records, strict=True):
            if {key: value for key, value in record.items() if key != 'seeds'} != base:
                raise ValueError(f'{path} differs from {paths[0]} outside its seeds')
            seeds.extend(record['seeds'])
        numbers = [row['seed'] for row in seeds]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f'Sparkov {regime}: repeated seeds in {sorted(numbers)}')
        merged[regime] = {**base, 'seeds': sorted(seeds, key=lambda row: row['seed'])}

    per_regime = {'regime', 'censored_train_labels', 'seeds'}
    reference = merged['off']
    for regime, record in merged.items():
        for key in reference.keys() - per_regime:
            if record.get(key) != reference[key]:
                raise ValueError(f'Sparkov {regime} differs from off in {key!r}')
        if [row['seed'] for row in record['seeds']] != [row['seed'] for row in reference['seeds']]:
            raise ValueError(f'Sparkov {regime} does not have the same seeds as off')
    return merged


def sparkov_delays(features_dir: Path = FEATURE_DIR) -> dict[str, tuple[float, float]]:
    """Median and mean synthetic transaction-to-report delay in days, over all frauds."""
    df = pd.read_parquet(
        Path(features_dir) / 'sparkov.parquet',
        columns=['event_time', 'is_fraud', 'reported_at', 'reported_at_slow'],
    )
    fraud = df[df['is_fraud'].astype(bool)]
    delays = {}
    for regime, column in (('on', 'reported_at'), ('slow', 'reported_at_slow')):
        days = (fraud[column] - fraud['event_time']).dt.total_seconds() / 86_400
        delays[regime] = (float(days.median()), float(days.mean()))
    return delays


def sparkov_delay_settings() -> str:
    """The configured delay parameters of the two Sparkov regimes, as one sentence."""
    config = load_config()
    gap = config.campaign_gap_for('sparkov')
    parts = []
    for label, dataset in (('default', 'sparkov'), ('slow', 'sparkov_slow')):
        params = config.delay_for(dataset)
        cap = (
            f'capped at {params.max_delay_days:g} days'
            if params.max_delay_days is not None
            else 'no cap'
        )
        parts.append(f'{label} median {params.median_days:g} days, σ {params.sigma:g}, {cap}')
    days = gap / pd.Timedelta(days=1)
    return (
        'Synthetic delays: frauds on one card form a campaign while each follows the previous '
        f'within {days:g} day{"" if days == 1 else "s"}; one lognormal delay per campaign, '
        'from its last fraud (' + '; '.join(parts) + ').'
    )


def _seed_range(seeds: list[int]) -> str:
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f'{seeds[0]}–{seeds[-1]}'
    return ', '.join(str(seed) for seed in seeds)


def render_sparkov_paper(
    records: dict[str, dict], delays: dict[str, tuple[float, float]], delay_settings: str
) -> str:
    """The paper's Sparkov label-delay table, with paired test-AP differences."""
    off = records['off']
    seeds = [row['seed'] for row in off['seeds']]
    test = {
        regime: _values(record, 'test', 'average_precision') for regime, record in records.items()
    }
    train_frauds = off['positives']['train']

    def hidden(regime: str) -> str:
        count = records[regime]['censored_train_labels']
        if regime == 'off':
            return f'{count:,}'
        return f'{count:,} ({100 * count / train_frauds:.1f}%)'

    def delta(regime: str) -> str:
        difference = test[regime] - test['off']
        error = np.std(difference, ddof=1) / np.sqrt(len(difference))
        return f'{np.mean(difference):+.4f} ({error:.4f})'

    def lower(regime: str) -> str:
        difference = test[regime] - test['off']
        return f'{int(np.sum(difference < 0))}/{len(difference)}'

    def ap(partition: str):
        return lambda regime: _mean_sd(records[regime], partition, 'average_precision')

    def row(label: str, cell, *, delayed_only: bool = True) -> str:
        cells = [
            '–' if delayed_only and regime == 'off' else cell(regime) for regime in SPARKOV_REGIMES
        ]
        return f'| {label} | ' + ' | '.join(cells) + ' |'

    settings = off['model_settings']
    return '\n'.join(
        [
            '# Sparkov synthetic label delay (paper)',
            '',
            'Generated by `scripts/lstm_protocol.py`; do not edit by hand.',
            '',
            f'LSTM preset `{settings["config"]}`: {settings["max_epochs"]} epochs without early '
            'stopping; the final weights are scored.',
            f'Chronological 80/10/10 split of all {off["source_rows"]:,} Sparkov rows. Every '
            f'transaction is a target, scored with up to {off["window_length"] - 1} earlier '
            'transactions of the same card. Earlier means a strictly earlier timestamp, so '
            'transactions at the same instant are excluded. Shorter histories are padded after '
            'the target, and the score is read at the target, so padding has no effect.',
            'All regimes use the same rows, split, features, model settings, and seeds; only '
            'the training labels available at the cutoff differ.',
            'Validation and test are scored against the true labels. The reporting delays are '
            'synthetic, and the slow regime is a stress test.',
            delay_settings,
            'Delays are realised transaction-to-report delays over all Sparkov frauds. Hidden '
            f'train frauds are the training frauds (of {train_frauds:,}) reported after the '
            'training cutoff; they stay in training as negatives.',
            f'AP entries are mean ± population standard deviation over seeds {_seed_range(seeds)}. '
            'Δ is the mean within-seed difference in test AP to the no-delay run, with its '
            'standard error in parentheses.',
            '',
            '| | No delay | Default delay | Slow delay |',
            '|---|---:|---:|---:|',
            row('Median delay', lambda regime: f'{delays[regime][0]:.1f} days'),
            row('Mean delay', lambda regime: f'{delays[regime][1]:.1f} days'),
            row('Hidden train frauds', hidden, delayed_only=False),
            row('Validation AP', ap('val'), delayed_only=False),
            row('Test AP', ap('test'), delayed_only=False),
            row('Δ test AP', delta),
            row('Seeds with lower test AP than no delay', lower),
        ]
    )


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print(f'Wrote {path}', flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ibm', 'sparkov'), required=True)
    parser.add_argument('--regime', choices=(*IBM_REGIMES, *SPARKOV_REGIMES))
    parser.add_argument('--features-dir', type=Path, default=FEATURE_DIR)
    parser.add_argument(
        '--results-dir',
        type=Path,
        help='write records here; by default paper cells go to results/paper '
        'and others to results/archive',
    )
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--model', choices=('lstm', 'xgboost', 'both'))
    parser.add_argument('--window-length', type=int, default=30)
    parser.add_argument('--window-stride', type=int, help='IBM window step; defaults to length')
    parser.add_argument('--feature-set', choices=('full', 'rawish'), default='full')
    parser.add_argument(
        '--lstm-config',
        choices=tuple(LSTM_CONFIGS),
        default='default',
        help=f'LSTM training preset ({PAPER_CONFIG} for the paper IBM split table, '
        f'{SPARKOV_PAPER_CONFIG} for the paper Sparkov label-delay table; archived presets '
        f'are {", ".join(ARCHIVED_LSTM_CONFIGS)}); '
        'non-default presets write separate records',
    )
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='skip result files already present')
    args = parser.parse_args(argv)
    args.model = args.model or ('both' if args.dataset == 'ibm' else 'lstm')
    if args.window_length < 2:
        parser.error('window length must be at least 2')
    if args.window_stride is not None:
        if args.dataset != 'ibm':
            parser.error('--window-stride is defined only for IBM')
        if not 1 <= args.window_stride <= args.window_length:
            parser.error('window stride must be between 1 and window length')
        if args.window_stride < args.window_length and args.regime == 'pre_italy_iid_chunks':
            parser.error('chunk-IID requires disjoint windows')
        if args.window_stride < args.window_length and args.regime is None:
            parser.error('overlapping IBM windows require an explicit --regime')
    seeds = tuple(args.seeds)
    if args.lstm_config != 'default' and args.model != 'lstm':
        parser.error('--lstm-config applies to --model lstm only')
    if args.dataset == 'sparkov' and args.model != 'lstm':
        parser.error('Sparkov XGBoost has its own runner, sparkov_delay_protocol.py')
    if args.dataset == 'ibm' and args.feature_set != 'full':
        parser.error('the raw-ish feature set is currently defined for Sparkov only')
    if args.regime and args.regime not in (
        IBM_REGIMES if args.dataset == 'ibm' else SPARKOV_REGIMES
    ):
        parser.error(f'{args.regime!r} is not a {args.dataset} regime')
    if not args.render_only and seeds != (0, 1, 2, 3, 4) and args.results_dir is None:
        parser.error('noncanonical seeds require a separate --results-dir for diagnostics')

    if not args.render_only:
        if args.dataset == 'ibm':
            source = pre_italy_split(pd.read_parquet(args.features_dir / 'ibm_ccf.parquet'))
            chunks = complete_chunks(
                source['entity_id'],
                source['event_time'],
                args.window_length,
                args.window_stride,
            )
            print(f'IBM: {len(chunks.rows):,} complete endpoint windows', flush=True)
            for regime in (args.regime,) if args.regime else IBM_ARCHIVED_REGIMES:
                wanted = ('lstm', 'xgboost') if args.model == 'both' else (args.model,)
                if args.resume and all(
                    _path(
                        'ibm',
                        model,
                        regime,
                        args.results_dir,
                        args.window_length,
                        window_stride=args.window_stride,
                        config=args.lstm_config,
                    ).exists()
                    for model in wanted
                ):
                    print(f'Skipping completed IBM {regime}', flush=True)
                    continue
                prepared = prepare_ibm(
                    regime,
                    args.features_dir,
                    source=source,
                    chunks=chunks,
                    source_prepared=True,
                    window_length=args.window_length,
                    window_stride=args.window_stride,
                )
                print(f'IBM {regime}: {prepared.metadata["positives"]}', flush=True)
                if args.prepare_only:
                    continue
                if args.model in ('lstm', 'both'):
                    path = _path(
                        'ibm',
                        'lstm',
                        regime,
                        args.results_dir,
                        args.window_length,
                        window_stride=args.window_stride,
                        config=args.lstm_config,
                    )
                    if not (args.resume and path.exists()):
                        _write(path, run_lstm(prepared, seeds, args.device, args.lstm_config))
                if args.model in ('xgboost', 'both'):
                    path = _path(
                        'ibm',
                        'xgboost',
                        regime,
                        args.results_dir,
                        args.window_length,
                        window_stride=args.window_stride,
                    )
                    if not (args.resume and path.exists()):
                        _write(path, run_ibm_xgboost(prepared, seeds))
        else:
            source = standard_split(pd.read_parquet(args.features_dir / 'sparkov.parquet'))
            for regime in (args.regime,) if args.regime else SPARKOV_REGIMES:
                path = _path(
                    'sparkov',
                    'lstm',
                    regime,
                    args.results_dir,
                    args.window_length,
                    args.feature_set,
                    config=args.lstm_config,
                )
                if args.resume and path.exists():
                    print(f'Skipping completed Sparkov {regime}', flush=True)
                    continue
                prepared = prepare_sparkov(
                    regime,
                    args.features_dir,
                    source=source,
                    source_prepared=True,
                    window_length=args.window_length,
                    feature_set=args.feature_set,
                )
                print(f'Sparkov {regime}: {prepared.metadata["positives"]}', flush=True)
                if not args.prepare_only:
                    _write(path, run_lstm(prepared, seeds, args.device, args.lstm_config))

    if args.prepare_only or args.results_dir is not None:
        return 0
    if args.dataset == 'sparkov' and _is_paper_cell(
        'sparkov', 'lstm', 'off', args.window_length, None, args.lstm_config, args.feature_set
    ):
        if all(
            _path(
                'sparkov',
                'lstm',
                regime,
                None,
                args.window_length,
                args.feature_set,
                config=args.lstm_config,
            ).exists()
            for regime in SPARKOV_REGIMES
        ):
            table = render_sparkov_paper(
                sparkov_paper_records(),
                sparkov_delays(args.features_dir),
                sparkov_delay_settings(),
            )
            SPARKOV_PAPER_TABLE.write_text(table + '\n')
            print(f'Wrote {SPARKOV_PAPER_TABLE}', flush=True)
        return 0
    if args.lstm_config != 'default':
        paper = {
            regime: _path(
                'ibm',
                'lstm',
                regime,
                None,
                args.window_length,
                window_stride=args.window_stride,
                config=args.lstm_config,
            )
            for regime in IBM_TABLE3_REGIMES
        }
        if (
            args.dataset == 'ibm'
            and args.regime in IBM_TABLE3_REGIMES
            and all(path.parent == PAPER and path.exists() for path in paper.values())
        ):
            records = {regime: json.loads(path.read_text()) for regime, path in paper.items()}
            PAPER_TABLE.write_text(render_ibm_paper_split(records) + '\n')
            print(f'Wrote {PAPER_TABLE}', flush=True)
        return 0
    if args.dataset == 'ibm':
        if args.regime in IBM_TABLE3_REGIMES:
            paths = {
                (model, regime): _path(
                    'ibm',
                    model,
                    regime,
                    args.results_dir,
                    args.window_length,
                    window_stride=args.window_stride,
                )
                for model in ('lstm', 'xgboost')
                for regime in IBM_TABLE3_REGIMES
            }
            if all(path.exists() for path in paths.values()):
                table = (
                    ARCHIVE
                    / 'ibm_sequence'
                    / (
                        f'ibm_lstm_pre_italy_68_16_16_len{args.window_length}'
                        f'_stride{args.window_stride or args.window_length}.md'
                    )
                )
                records = {
                    model: {
                        regime: json.loads(paths[model, regime].read_text())
                        for regime in IBM_TABLE3_REGIMES
                    }
                    for model in ('lstm', 'xgboost')
                }
                table.write_text(
                    render_ibm_68_16_16_default_preset(records['lstm'], records['xgboost']) + '\n'
                )
                print(f'Wrote {table}', flush=True)
            return 0
        if args.window_stride is not None and args.window_stride != args.window_length:
            return 0
        if all(
            _path('ibm', model, regime, args.results_dir, args.window_length).exists()
            for model in ('lstm', 'xgboost')
            for regime in IBM_ARCHIVED_REGIMES
        ):
            table = (
                IBM_TABLE
                if args.window_length == 30
                else IBM_TABLE.with_stem(f'{IBM_TABLE.stem}_len{args.window_length}')
            )
            table.write_text(
                render_ibm(
                    _read(
                        'ibm', 'lstm', IBM_ARCHIVED_REGIMES, args.results_dir, args.window_length
                    ),
                    _read(
                        'ibm',
                        'xgboost',
                        IBM_ARCHIVED_REGIMES,
                        args.results_dir,
                        args.window_length,
                    ),
                )
                + '\n'
            )
            print(f'Wrote {table}', flush=True)
    elif all(
        _path(
            'sparkov', 'lstm', regime, args.results_dir, args.window_length, args.feature_set
        ).exists()
        for regime in SPARKOV_REGIMES
    ):
        if args.feature_set == 'rawish':
            table = SPARKOV_TABLE.with_stem(f'{SPARKOV_TABLE.stem}_rawish_len{args.window_length}')
        elif args.window_length == 30:
            table = SPARKOV_TABLE
        else:
            table = SPARKOV_TABLE.with_stem(f'{SPARKOV_TABLE.stem}_len{args.window_length}')
        table.write_text(
            render_sparkov(
                _read(
                    'sparkov',
                    'lstm',
                    SPARKOV_REGIMES,
                    args.results_dir,
                    args.window_length,
                    args.feature_set,
                )
            )
            + '\n'
        )
        print(f'Wrote {table}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
