"""Shared preparation for the two public paper protocols.

This module reads a feature parquet, applies the protocol's split, fits categorical
encoding on training rows only, and constructs training labels at the train cutoff.
Model fitting and result rendering live in the two dedicated protocol runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.data.censoring import censored_labels
from fraud_benchmark.experiments.encoding import CappedOrdinalEncoder
from fraud_benchmark.experiments.features.util import ARTIFACT_PREFIX
from fraud_benchmark.experiments.features.util import FEATURE_DIR
from fraud_benchmark.experiments.features.util import feature_columns
from fraud_benchmark.experiments.splits import pre_italy_iid_customer_split
from fraud_benchmark.experiments.splits import pre_italy_iid_row_split
from fraud_benchmark.experiments.splits import pre_italy_split
from fraud_benchmark.experiments.splits import standard_split

DATASETS = ('ibm_ccf', 'sparkov')
LABEL_DELAYS = ('off', 'on', 'slow')
SPLITS = ('standard', 'pre_italy', 'pre_italy_iid_rows', 'pre_italy_iid_customers')
SLOW_DELAY_COLUMN = 'reported_at_slow'


class ExperimentError(ValueError):
    """Raised when a paper-protocol configuration cannot be run as requested."""


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    label_delay: str = 'off'
    split: str = 'standard'

    def validate(self) -> None:
        if self.dataset not in DATASETS:
            raise ExperimentError(f'unknown paper dataset {self.dataset!r}; known: {DATASETS}')
        if self.label_delay not in LABEL_DELAYS:
            raise ExperimentError(
                f'label_delay must be one of {LABEL_DELAYS}, got {self.label_delay!r}'
            )
        if self.split not in SPLITS:
            raise ExperimentError(f'split must be one of {SPLITS}, got {self.split!r}')
        if self.label_delay == 'slow' and self.dataset != 'sparkov':
            raise ExperimentError('the slow synthetic delay exists only for Sparkov')
        if self.split.startswith('pre_italy') and self.dataset != 'ibm_ccf':
            raise ExperimentError('pre-Italy splits are defined only for IBM CCF')
        if self.dataset == 'ibm_ccf' and not self.split.startswith('pre_italy'):
            raise ExperimentError('the public IBM paper protocol uses pre-Italy rows only')
        if self.dataset == 'sparkov' and self.split != 'standard':
            raise ExperimentError('the public Sparkov paper protocol uses the standard split')


@dataclass
class Prepared:
    x: np.ndarray
    y: np.ndarray
    y_true: np.ndarray
    train_rows: np.ndarray
    val_rows: np.ndarray
    test_rows: np.ndarray
    names: list[str]
    censored_train_labels: int
    notes: list[str] = field(default_factory=list)


def _apply_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == 'standard':
        return standard_split(df)
    if split == 'pre_italy':
        return pre_italy_split(df)
    if split == 'pre_italy_iid_rows':
        return pre_italy_iid_row_split(df)
    return pre_italy_iid_customer_split(df)


def _labels(
    df: pd.DataFrame,
    config: ExperimentConfig,
    train_mask: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, int]:
    if config.label_delay == 'off':
        return y_true.copy(), 0
    column = 'reported_at' if config.label_delay == 'on' else SLOW_DELAY_COLUMN
    if column not in df.columns:
        raise ExperimentError(f'{config.dataset}: no {column!r} column')
    cutoff = df.loc[train_mask, 'event_time'].max()
    known = censored_labels(
        pd.DataFrame({'is_fraud': df['is_fraud'], 'reported_at': df[column]}), cutoff
    )
    y = y_true.copy()
    y[train_mask] = known[train_mask]
    hidden = int(y_true[train_mask].sum() - y[train_mask].sum())
    return y, hidden


def prepare(
    config: ExperimentConfig, features_dir: Path | str = FEATURE_DIR
) -> Prepared:
    """Build the encoded matrix and cutoff-aware labels for one paper regime."""
    config.validate()
    source = Path(features_dir) / f'{config.dataset}.parquet'
    if not source.exists():
        raise ExperimentError(
            f'{source} does not exist; build it with `scripts/features.py '
            f'--dataset {config.dataset}`'
        )
    df = _apply_split(pd.read_parquet(source), config.split).reset_index(drop=True)
    columns = [c for c in feature_columns(df) if not c.startswith(ARTIFACT_PREFIX)]
    categorical = [c for c in columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    train_mask = df['split'].eq('train').to_numpy()
    encoder = CappedOrdinalEncoder().fit(df.loc[train_mask], categorical)
    x = np.empty((len(df), len(columns)), dtype='float32')
    for index, column in enumerate(columns):
        if column in categorical:
            x[:, index] = encoder.transform_column(df[column], column)
        else:
            x[:, index] = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype='float32')

    y_true = df['is_fraud'].to_numpy(dtype='int64')
    y, hidden = _labels(df, config, train_mask, y_true)
    rows = {
        name: np.flatnonzero(df['split'].eq(name).to_numpy())
        for name in ('train', 'val', 'test')
    }
    for name, index in rows.items():
        if len(index) == 0:
            raise ExperimentError(f'the {name} split is empty')
    for name in ('val', 'test'):
        if not y_true[rows[name]].sum():
            raise ExperimentError(f'the {name} split contains no frauds')
    if not y[rows['train']].sum():
        raise ExperimentError('no fraud is labeled in train')

    notes = [f'dropped {len(feature_columns(df)) - len(columns)} artifact column(s)']
    if hidden:
        notes.append(f'{hidden:,} train fraud label(s) unknown at the cutoff')
    return Prepared(
        x=x,
        y=y,
        y_true=y_true,
        train_rows=rows['train'],
        val_rows=rows['val'],
        test_rows=rows['test'],
        names=columns,
        censored_train_labels=hidden,
        notes=notes,
    )


__all__ = ['ExperimentConfig', 'ExperimentError', 'Prepared', 'prepare']
