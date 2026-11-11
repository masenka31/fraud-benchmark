"""Data preparation and scoring for the LSTM sequence protocols.

The paper's Table 3 uses the IBM `IBM_TABLE3_REGIMES` and Table 4 the Sparkov
`SPARKOV_REGIMES`.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.data.censoring import censored_labels
from fraud_benchmark.experiments.causal_encoding import causal_count
from fraud_benchmark.experiments.features.util import ARTIFACT_PREFIX
from fraud_benchmark.experiments.features.util import FEATURE_DIR
from fraud_benchmark.experiments.features.util import feature_columns
from fraud_benchmark.experiments.lstm import LSTM_CONFIGS
from fraud_benchmark.experiments.lstm import EncodedFeatures
from fraud_benchmark.experiments.lstm import encode_features
from fraud_benchmark.experiments.lstm import model_settings
from fraud_benchmark.experiments.lstm import train_and_predict
from fraud_benchmark.experiments.metrics import best_f1_threshold
from fraud_benchmark.experiments.metrics import score
from fraud_benchmark.experiments.sequence import Chunks
from fraud_benchmark.experiments.sequence import WindowIndex
from fraud_benchmark.experiments.sequence import complete_chunks
from fraud_benchmark.experiments.sequence import iid_chunk_split
from fraud_benchmark.experiments.sequence import window_index
from fraud_benchmark.experiments.splits import FIRST_ITALY_FRAUD
from fraud_benchmark.experiments.splits import IID_SPLIT_SEED
from fraud_benchmark.experiments.splits import iid_customer_split
from fraud_benchmark.experiments.splits import pre_italy_68_16_16_boundaries
from fraud_benchmark.experiments.splits import pre_italy_split
from fraud_benchmark.experiments.splits import standard_split


class ExperimentError(ValueError):
    """A protocol's inputs are missing or cannot support it."""


#: The paper's Table 3: temporal, window iid and customer iid on 68/16/16.
IBM_TABLE3_REGIMES = (
    'pre_italy_68_16_16',
    'pre_italy_iid_chunks_68_16_16',
    'pre_italy_iid_customers_68_16_16',
)
IBM_REGIMES = IBM_TABLE3_REGIMES
IBM_TABLE3_RATIOS = (0.68, 0.16, 0.16)
SPARKOV_REGIMES = ('off', 'on', 'slow')
SPLIT_CODE = {'train': 0, 'val': 1, 'test': 2}
RARITY_NAMES = ['state_prior_rarity', 'state_not_seen_previously', 'foreign_state_rarity']


@dataclass
class PreparedSequence:
    features: EncodedFeatures
    positions_for: object
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    y_fit: np.ndarray
    y_true: np.ndarray
    metadata: dict


def _source(features_dir: Path, dataset: str) -> pd.DataFrame:
    path = features_dir / f'{dataset}.parquet'
    if not path.exists():
        raise ExperimentError(f'{path} does not exist; build it with scripts/features.py')
    return pd.read_parquet(path)


def _partition_ids(split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = tuple(np.flatnonzero(split == code).astype('int32') for code in range(3))
    if any(len(part) == 0 for part in rows):
        raise ExperimentError('each train, validation, and test partition needs target rows')
    return rows


def _ibm_split(df: pd.DataFrame, chunks: Chunks, regime: str) -> tuple[np.ndarray, np.ndarray]:
    """Return target and source split codes. Context rows have no targets."""
    if regime == 'pre_italy_68_16_16':
        times = df['event_time']
        val_start, test_start = pre_italy_68_16_16_boundaries(times)
        source = np.where(times < val_start, 0, np.where(times < test_start, 1, 2))
        source = source.astype('int8')
        return source[chunks.targets], source
    if regime == 'pre_italy_iid_customers_68_16_16':
        endpoints = df[['entity_id', 'event_time']].iloc[chunks.targets]
        assignment = iid_customer_split(endpoints, seed=IID_SPLIT_SEED, ratios=IBM_TABLE3_RATIOS)
        mapping = assignment.drop_duplicates('entity_id').set_index('entity_id')['split']
        source = df['entity_id'].map(mapping).map(SPLIT_CODE).fillna(-1).to_numpy(dtype='int8')
        return source[chunks.targets], source
    if regime == 'pre_italy_iid_chunks_68_16_16':
        target = iid_chunk_split(len(chunks.rows), IID_SPLIT_SEED, IBM_TABLE3_RATIOS)
        source = np.full(len(df), -1, dtype='int8')
        source[chunks.targets] = target
        return target, source
    raise ValueError(f'unknown IBM regime {regime!r}')


def _ibm_rarity(df: pd.DataFrame, train_mask: np.ndarray) -> None:
    """Only label-free chronological features from the old IBM extra block."""
    count = causal_count(df['artifact_merchant_state'], train_mask)
    rarity = (1.0 / np.sqrt(count + 1.0)).astype('float32')
    df[RARITY_NAMES[0]] = rarity
    df[RARITY_NAMES[1]] = (count == 0).astype('float32')
    df[RARITY_NAMES[2]] = rarity * df['merchant_is_foreign'].to_numpy(dtype='float32')


def prepare_ibm(
    regime: str,
    features_dir: Path | str = FEATURE_DIR,
    *,
    source: pd.DataFrame | None = None,
    chunks: Chunks | None = None,
    source_prepared: bool = False,
    window_length: int = 10,
    window_stride: int | None = 5,
) -> PreparedSequence:
    """One fixed IBM endpoint population, with split-specific train encoding."""
    if regime not in IBM_REGIMES:
        raise ValueError(f'unknown IBM regime {regime!r}')
    if source_prepared and source is None:
        raise ValueError('source_prepared requires source')
    df = (
        source.copy(deep=False)
        if source_prepared
        else pre_italy_split(_source(Path(features_dir), 'ibm_ccf') if source is None else source)
    )
    if chunks is None:
        chunks = complete_chunks(df['entity_id'], df['event_time'], window_length, window_stride)
    elif chunks.rows.shape[1] != window_length:
        raise ValueError('chunk width does not match window_length')
    elif window_stride is not None and chunks.stride != window_stride:
        raise ValueError('chunk stride does not match window_stride')
    target_split, source_split = _ibm_split(df, chunks, regime)
    train_mask = source_split == 0
    _ibm_rarity(df, train_mask)
    names = [name for name in feature_columns(df) if not name.startswith(ARTIFACT_PREFIX)]
    if len(names) != 77:
        raise ExperimentError(f'expected 77 IBM LSTM features, got {len(names)}')
    features = encode_features(df, names, train_mask)
    targets = chunks.targets
    y_true = df['is_fraud'].to_numpy(dtype='int64')[targets]
    train, validation, test = _partition_ids(target_split)
    if any(y_true[part].sum() == 0 for part in (train, validation, test)):
        raise ExperimentError('an IBM endpoint partition contains no fraud targets')

    def positions_for(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return chunks.rows[ids], np.full(len(ids), window_length, dtype='int32')

    return PreparedSequence(
        features,
        positions_for,
        train,
        validation,
        test,
        y_true.copy(),
        y_true,
        {
            'dataset': 'ibm_ccf',
            'regime': regime,
            'population': f'event_time < {FIRST_ITALY_FRAUD.isoformat()}',
            'source_rows': len(df),
            'target_rows': len(targets),
            'dropped_incomplete_rows': chunks.n_dropped,
            'chunk_length': window_length,
            'window_stride': chunks.stride,
            'target': 'last transaction of each complete entity chunk',
            'tie_policy': 'source row order within equal entity timestamps',
            'split_seed': IID_SPLIT_SEED if 'iid' in regime else None,
            'split_unit': (
                'overlapping windows'
                if regime == 'pre_italy_iid_chunks_68_16_16'
                else 'customers'
                if regime == 'pre_italy_iid_customers_68_16_16'
                else 'source transactions'
            ),
            'features': names,
            'rows': {
                name: len(rows)
                for name, rows in zip(('train', 'val', 'test'), (train, validation, test))
            },
            'positives': {
                name: int(y_true[rows].sum())
                for name, rows in zip(('train', 'val', 'test'), (train, validation, test))
            },
        },
    )


def prepare_sparkov(
    regime: str,
    features_dir: Path | str = FEATURE_DIR,
    *,
    source: pd.DataFrame | None = None,
    encoded: EncodedFeatures | None = None,
    index: WindowIndex | None = None,
    source_prepared: bool = False,
    window_length: int = 10,
) -> PreparedSequence:
    """Same rows, split, and inputs in all three synthetic delay regimes."""
    if regime not in SPARKOV_REGIMES:
        raise ValueError(f'unknown Sparkov regime {regime!r}')
    if source_prepared and source is None:
        raise ValueError('source_prepared requires source')
    df = (
        source.copy(deep=False)
        if source_prepared
        else standard_split(_source(Path(features_dir), 'sparkov') if source is None else source)
    )
    split = df['split'].map(SPLIT_CODE).to_numpy(dtype='int8')
    train_mask = split == 0
    available = [name for name in feature_columns(df) if not name.startswith(ARTIFACT_PREFIX)]
    names = available
    if len(names) != 46:
        raise ExperimentError(f'expected 46 Sparkov LSTM features, got {len(names)}')
    if index is None:
        index = window_index(df['entity_id'], df['event_time'], window_length)
    elif index.length != window_length:
        raise ValueError('index width does not match window_length')
    if encoded is None:
        encoded = encode_features(df, names, train_mask).reordered(index.order)
    train, validation, test = _partition_ids(split)
    y_true = df['is_fraud'].to_numpy(dtype='int64')
    y_fit = y_true.copy()
    if regime != 'off':
        column = 'reported_at' if regime == 'on' else 'reported_at_slow'
        if column not in df:
            raise ExperimentError(f'sparkov: no {column!r} column')
        cutoff = df.loc[train_mask, 'event_time'].max()
        visible = censored_labels(
            pd.DataFrame({'is_fraud': df['is_fraud'], 'reported_at': df[column]}), cutoff
        )
        y_fit[train_mask] = visible[train_mask]
    if any(y_true[part].sum() == 0 for part in (validation, test)):
        raise ExperimentError('a Sparkov evaluation partition contains no fraud targets')
    return PreparedSequence(
        encoded,
        index.batch,
        train,
        validation,
        test,
        y_fit,
        y_true,
        {
            'dataset': 'sparkov',
            'regime': regime,
            'source_rows': len(df),
            'target_rows': len(df),
            'window_length': window_length,
            # Kept so new records merge with the committed Table 4 records.
            'feature_set': 'full',
            'target': 'each transaction',
            'tie_policy': 'exclude equal-timestamp predecessor rows',
            'censored_train_labels': int(y_true[train].sum() - y_fit[train].sum()),
            'features': names,
            'rows': {
                name: len(rows)
                for name, rows in zip(('train', 'val', 'test'), (train, validation, test))
            },
            'positives': {
                name: int(y_true[rows].sum())
                for name, rows in zip(('train', 'val', 'test'), (train, validation, test))
            },
        },
    )


def run_lstm(
    prepared: PreparedSequence, seeds: tuple[int, ...], device: str, config: str
) -> dict:
    """Train and score one preset over `seeds` on a prepared Sparkov or IBM protocol."""
    seed_records = []
    for seed in seeds:
        started = time.monotonic()
        val_scores, test_scores, details = train_and_predict(
            prepared.features,
            prepared.positions_for,
            prepared.train,
            prepared.validation,
            prepared.test,
            prepared.y_fit,
            prepared.y_true,
            seed,
            device,
            LSTM_CONFIGS[config],
        )
        threshold = best_f1_threshold(prepared.y_true[prepared.validation], val_scores)
        seed_records.append(
            {
                'seed': seed,
                'fit_seconds': round(time.monotonic() - started, 1),
                'training': details,
                'scores': {
                    'val': score(prepared.y_true[prepared.validation], val_scores, threshold),
                    'test': score(prepared.y_true[prepared.test], test_scores, threshold),
                },
            }
        )
    return {
        **prepared.metadata,
        'model': 'lstm',
        'model_settings': model_settings(config),
        'categorical_cardinalities': dict(
            zip(prepared.features.categorical_names, prepared.features.cardinalities)
        ),
        'seeds': seed_records,
        'python': platform.python_version(),
    }
