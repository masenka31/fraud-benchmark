#!/usr/bin/env python
"""Run and render the paper's five-seed pre-Italy IBM split comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from fraud_benchmark.experiments.causal_encoding import causal_count
from fraud_benchmark.experiments.causal_encoding import causal_target_rate
from fraud_benchmark.experiments.experiment import ExperimentConfig
from fraud_benchmark.experiments.experiment import prepare
from fraud_benchmark.experiments.features.util import FEATURE_DIR
from fraud_benchmark.experiments.models import fit_xgboost
from fraud_benchmark.experiments.splits import FIRST_ITALY_FRAUD
from fraud_benchmark.experiments.splits import pre_italy_iid_customer_split
from fraud_benchmark.experiments.splits import pre_italy_iid_row_split
from fraud_benchmark.experiments.splits import pre_italy_split

REGIMES = ('pre_italy', 'pre_italy_iid_rows', 'pre_italy_iid_customers')
LABELS = {
    'pre_italy': 'Temporal',
    'pre_italy_iid_rows': 'Transaction IID',
    'pre_italy_iid_customers': 'Customer IID',
}
RESULTS = {
    'pre_italy': Path('results/paper/ibm_pre_italy_temporal.json'),
    'pre_italy_iid_rows': Path('results/paper/ibm_pre_italy_iid_rows.json'),
    'pre_italy_iid_customers': Path('results/paper/ibm_pre_italy_iid_customers.json'),
}
DEFAULT_TABLE = Path('results/ibm_split_protocol.md')
CATEGORICAL = (
    'artifact_merchant_state',
    'artifact_merchant_city',
    'artifact_merchant_name',
    'artifact_mcc',
    'artifact_error_type',
)


def _split(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == 'pre_italy':
        return pre_italy_split(df)
    if regime == 'pre_italy_iid_rows':
        return pre_italy_iid_row_split(df)
    return pre_italy_iid_customer_split(df)


def _engineered(regime: str, features_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Build chronological features without propagating future labels backward."""
    columns = ['event_time', 'entity_id', 'is_fraud', 'merchant_is_foreign', *CATEGORICAL]
    df = pd.read_parquet(features_dir / 'ibm_ccf.parquet', columns=columns)
    df = _split(df, regime)
    train = df['split'].eq('train').to_numpy()
    labels = df['is_fraud'].to_numpy(dtype='int64')
    state_count = causal_count(df['artifact_merchant_state'], train)
    rarity = 1.0 / np.sqrt(state_count + 1.0)
    foreign = df['merchant_is_foreign'].to_numpy(dtype='float64')
    blocks = [rarity, state_count == 0, rarity * foreign]
    names = ['state_prior_rarity', 'state_not_seen_previously', 'foreign_state_rarity']
    for column in CATEGORICAL:
        blocks.append(causal_target_rate(df[column], labels, train))
        names.append(f'causal_target_rate_{column}')
    return np.column_stack(blocks).astype('float32'), names


def _fit(x: np.ndarray, y: np.ndarray, rows: np.ndarray, seed: int):
    started = time.monotonic()
    model = fit_xgboost(x[rows], y[rows], seed)
    return model, round(time.monotonic() - started, 1)


def run_regime(regime: str, features_dir: Path) -> dict:
    prepared = prepare(
        ExperimentConfig(
            dataset='ibm_ccf',
            split=regime,
            label_delay='off',
        ),
        features_dir,
    )
    engineered, names = _engineered(regime, features_dir)
    if len(engineered) != len(prepared.x):
        raise RuntimeError('engineered features do not align with prepared rows')
    x = np.column_stack((prepared.x, engineered))
    seed_records = []
    for seed in range(5):
        model, fit_seconds = _fit(x, prepared.y, prepared.train_rows, seed)
        record = {'seed': seed, 'fit_seconds': fit_seconds}
        for partition, rows in (
            ('validation', prepared.val_rows),
            ('test', prepared.test_rows),
        ):
            scores = model.predict_proba(x[rows])[:, 1]
            record[partition] = {
                'average_precision': float(
                    average_precision_score(prepared.y_true[rows], scores)
                ),
                'roc_auc': float(roc_auc_score(prepared.y_true[rows], scores)),
            }
        seed_records.append(record)
        print(json.dumps(record), flush=True)
    return {
        'dataset': 'ibm_ccf',
        'population': f'event_time < {FIRST_ITALY_FRAUD.isoformat()}',
        'split': regime,
        'label_delay': 'off',
        'artifacts': 'drop',
        'feature_semantics': 'chronological; score before training-label update',
        'engineered_features': names,
        'rows': {
            'train': len(prepared.train_rows),
            'validation': len(prepared.val_rows),
            'test': len(prepared.test_rows),
        },
        'seeds': seed_records,
    }


def _metric(record: dict, partition: str, metric: str) -> str:
    values = [seed[partition][metric] for seed in record['seeds']]
    return f'{np.mean(values):.4f} ± {np.std(values):.4f}'


def render_markdown(records: dict[str, dict]) -> str:
    rows = []
    for metric_label, partition, metric in (
        ('Validation AP', 'validation', 'average_precision'),
        ('Validation ROC-AUC', 'validation', 'roc_auc'),
        ('Test AP', 'test', 'average_precision'),
        ('Test ROC-AUC', 'test', 'roc_auc'),
    ):
        rows.append(
            f'| {metric_label} | '
            + ' | '.join(_metric(records[regime], partition, metric) for regime in REGIMES)
            + ' |'
        )
    population = sum(records['pre_italy']['rows'].values())
    return '\n'.join(
        [
            '# IBM CCF pre-Italy temporal versus IID protocol',
            '',
            'Generated by `scripts/ibm_split_protocol.py`; do not edit by hand.',
            '',
            f'All regimes use the same **{population:,} transactions strictly before the first',
            f'Italy fraud ({FIRST_ITALY_FRAUD.isoformat()})**. The population, features,',
            'XGBoost settings, and model seeds are fixed; only split assignment changes.',
            'Features derived from labels are chronological, use training labels only, and',
            'score a training row before incorporating its own label.',
            '',
            'Entries are mean ± population standard deviation over seeds 0–4.',
            '',
            '| Metric | Temporal | Transaction IID | Customer IID |',
            '|---|---:|---:|---:|',
            *rows,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, default=FEATURE_DIR)
    parser.add_argument('--table-out', type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        '--render-only',
        action='store_true',
        help='regenerate the Markdown table from the recorded JSON without fitting',
    )
    args = parser.parse_args(argv)

    if args.render_only:
        records = {regime: json.loads(path.read_text()) for regime, path in RESULTS.items()}
    else:
        records = {}
        for regime in REGIMES:
            print(f'Running IBM pre-Italy split regime: {regime}', flush=True)
            record = run_regime(regime, args.features_dir)
            RESULTS[regime].parent.mkdir(parents=True, exist_ok=True)
            RESULTS[regime].write_text(json.dumps(record, indent=2) + '\n')
            records[regime] = record

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(render_markdown(records) + '\n')
    print(f'Wrote {args.table_out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
