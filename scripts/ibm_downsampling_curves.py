#!/usr/bin/env python
"""Fit and plot the paper's Figure 2: ROC-AUC hides what 1:1 downsampling costs.

Two XGBoost models on the pre-Italy IBM CCF 68/16/16 temporal split, with the same
features, hyperparameters, and model seed:

- Model X trains on every training fraud plus an equal number of random training
  negatives (1:1, sample seed 1729);
- Model Y trains on all training rows.

`scale_pos_weight` follows one rule for both, negatives/positives of the rows the model
is fitted on (1.0 for X). Validation and test are complete, with true labels.

The feature matrix is the one used in the paper. Unlike the repository's standard
preparation, it keeps the `artifact_*` columns and adds merchant-state rarity and
chronological fraud rates for five artifact categories. Every count and rate is fitted on
training rows only. A training row sees only the labels of training rows before it in
time order; rows with the same timestamp are taken in the order of a stable sort.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier

from fraud_benchmark.experiments.encoding import CappedOrdinalEncoder
from fraud_benchmark.experiments.features.util import FEATURE_DIR
from fraud_benchmark.experiments.features.util import feature_columns
from fraud_benchmark.experiments.models import XGB_PARAMS
from fraud_benchmark.experiments.splits import pre_italy_68_16_16_split

OUTPUT = Path('results/paper')
STEM = 'ibm_downsampling_roc_pr'
SEED = 0
SAMPLE_SEED = 1729
MODEL_X = 'Model X: random negatives 1:1'
MODEL_Y = 'Model Y: all training rows'
RATE_COLUMNS = (
    'artifact_merchant_state',
    'artifact_merchant_city',
    'artifact_merchant_name',
    'artifact_mcc',
    'artifact_error_type',
)
#: Smoothing of the chronological fraud rates: category weight and global-prior weight.
RATE_ALPHA = 20.0
PRIOR_STRENGTH = 1000.0
PRIOR_FRAUDS = 1.0


def encoded_features(df: pd.DataFrame, train: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Every feature column, artifact columns included, with train-fitted vocabularies."""
    columns = feature_columns(df)
    categorical = [c for c in columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    encoder = CappedOrdinalEncoder().fit(df.loc[train], categorical)
    x = np.empty((len(df), len(columns)), dtype='float32')
    for index, column in enumerate(columns):
        if column in categorical:
            x[:, index] = encoder.transform_column(df[column], column)
        else:
            x[:, index] = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype='float32')
    return x, list(columns)


def engineered_features(df: pd.DataFrame, train: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Merchant-state rarity and chronological artifact-category fraud rates.

    Training rows get the smoothed fraud rate of their category over earlier
    training rows; validation and test rows get the rate over all training rows.
    """
    counts = df.loc[train, 'artifact_merchant_state'].value_counts(dropna=False)
    count = df['artifact_merchant_state'].map(counts).fillna(0).to_numpy(dtype='float64')
    rarity = 1.0 / np.sqrt(count + 1.0)
    foreign = df['merchant_is_foreign'].to_numpy(dtype='float64')
    blocks = [rarity, count == 0, rarity * foreign]
    names = ['state_train_rarity', 'state_unseen_in_train', 'foreign_state_rarity']

    labels = df['is_fraud'].to_numpy(dtype='float64')
    train_labels = labels[train]
    prior_before = (np.cumsum(train_labels) - train_labels + PRIOR_FRAUDS) / (
        np.arange(len(train_labels)) + PRIOR_STRENGTH
    )
    frozen_prior = float((train_labels.sum() + PRIOR_FRAUDS) / (len(train_labels) + PRIOR_STRENGTH))
    later = ~train
    for column in RATE_COLUMNS:
        train_frame = df.loc[train, [column, 'is_fraud']]
        grouped = train_frame.groupby(column, observed=True)['is_fraud']
        previous_count = grouped.cumcount().to_numpy()
        previous_fraud = grouped.cumsum().to_numpy() - train_labels
        encoded = np.full(len(df), frozen_prior, dtype='float64')
        encoded[train] = (previous_fraud + RATE_ALPHA * prior_before) / (
            previous_count + RATE_ALPHA
        )
        totals = grouped.agg(['sum', 'count'])
        sums = df.loc[later, column].map(totals['sum']).fillna(0).to_numpy(dtype='float64')
        seen = df.loc[later, column].map(totals['count']).fillna(0).to_numpy(dtype='float64')
        encoded[later] = (sums + RATE_ALPHA * frozen_prior) / (seen + RATE_ALPHA)
        blocks.append(encoded)
        names.append(f'causal_target_rate_{column}')
    return np.column_stack(blocks).astype('float32'), names


def fit(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> tuple[XGBClassifier, float]:
    positive = int(y[rows].sum())
    negative = len(rows) - positive
    model = XGBClassifier(random_state=SEED, scale_pos_weight=negative / positive, **XGB_PARAMS)
    started = time.monotonic()
    model.fit(x[rows], y[rows])
    return model, round(time.monotonic() - started, 1)


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        'average_precision': float(average_precision_score(y, score)),
        'roc_auc': float(roc_auc_score(y, score)),
    }


def _thin_monotone_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep real step vertices on a grid dense near x=0 and uniform elsewhere."""
    targets = np.unique(
        np.concatenate(([0.0], np.geomspace(1e-8, 1e-2, 800), np.linspace(1e-2, 1.0, 1200)))
    )
    right = np.searchsorted(x, targets, side='left')
    indices = np.unique(np.clip(np.concatenate((right - 1, right)), 0, len(x) - 1))
    return x[indices], y[indices]


def plot(labels: np.ndarray, scores: dict[str, np.ndarray], test: dict, destination: Path):
    """The paper's two-panel figure: ROC and precision-recall on the test split."""
    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
            'font.size': 7,
            'axes.titlesize': 6.5,
            'axes.labelsize': 6,
            'legend.fontsize': 5.5,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'axes.linewidth': 0.7,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    colors = ('#0072B2', '#D55E00')
    styles = ('-', '--')
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for axis in (ax_roc, ax_pr):
        axis.set_box_aspect(1)
        axis.spines[['top', 'right']].set_visible(False)
    for i, (name, values) in enumerate(scores.items()):
        display = name.split(':')[0]
        fpr, tpr, _ = roc_curve(labels, values)
        fpr, tpr = _thin_monotone_curve(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labels, values, drop_intermediate=True)
        metric = test[name]
        ax_roc.plot(
            fpr,
            tpr,
            color=colors[i],
            linestyle=styles[i],
            linewidth=0.8,
            label=f'{display} (AUROC {metric["roc_auc"]:.4f})',
            zorder=3,
        )
        ax_pr.plot(
            recall,
            precision,
            color=colors[i],
            linestyle=styles[i],
            linewidth=0.8,
            label=f'{display} (AP {metric["average_precision"]:.3f})',
            zorder=3,
        )
        ax_roc.fill_between(fpr, tpr, color=colors[i], alpha=0.08, linewidth=0)
        ax_pr.fill_between(recall, precision, color=colors[i], alpha=0.08, linewidth=0)
    ax_roc.plot([0, 1], [0, 1], color='0.6', linestyle=':', linewidth=0.5)
    ax_pr.axhline(labels.mean(), color='0.6', linestyle=':', linewidth=0.5)
    ax_roc.set(
        xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.005)
    )
    ax_pr.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.005))
    ax_roc.legend(loc='lower right', frameon=False, handlelength=1.8)
    ax_pr.legend(loc='lower left', frameon=False, handlelength=1.8)
    for axis in (ax_roc, ax_pr):
        axis.grid(visible=True, linewidth=0.4, alpha=0.5, color='0.85')
    fig.tight_layout(pad=0.25, w_pad=0.5)
    fig.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(destination.with_suffix('.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--features-dir', type=Path, default=FEATURE_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    df = pre_italy_68_16_16_split(pd.read_parquet(args.features_dir / 'ibm_ccf.parquet'))
    train = df['split'].eq('train').to_numpy()
    base, base_names = encoded_features(df, train)
    engineered, engineered_names = engineered_features(df, train)
    x = np.column_stack((base, engineered))
    names = [*base_names, *engineered_names]
    y = df['is_fraud'].to_numpy(dtype='int64')
    rows = {
        name: np.flatnonzero(df['split'].eq(name).to_numpy()) for name in ('train', 'val', 'test')
    }
    del df, base, engineered

    train_rows = rows['train']
    positives = train_rows[y[train_rows] == 1]
    negatives = train_rows[y[train_rows] == 0]
    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = np.concatenate((positives, rng.permutation(negatives)[: len(positives)]))
    rng.shuffle(sampled)

    records, test_scores = {}, {}
    for name, fit_rows in ((MODEL_Y, train_rows), (MODEL_X, sampled)):
        model, seconds = fit(x, y, fit_rows)
        positive = int(y[fit_rows].sum())
        record = {
            'training_rows': len(fit_rows),
            'training_positives': positive,
            'training_negatives': len(fit_rows) - positive,
            'scale_pos_weight': (len(fit_rows) - positive) / positive,
            'fit_seconds': seconds,
        }
        for partition, label in (('val', 'validation'), ('test', 'test')):
            score = model.predict_proba(x[rows[partition]])[:, 1]
            record[label] = metrics(y[rows[partition]], score)
            if partition == 'test':
                test_scores[name] = score
        records[name] = record
        print(name, json.dumps(record), flush=True)

    output = {
        'figure': 'paper Figure 2',
        'dataset': 'ibm_ccf',
        'split': 'pre_italy_68_16_16',
        'label_delay': 'off',
        'evaluation': 'complete validation and test rows with true labels',
        'model_seed': SEED,
        'sample_seed': SAMPLE_SEED,
        'xgboost': XGB_PARAMS,
        'features': names,
        'rows': {name: len(values) for name, values in rows.items()},
        'positives': {name: int(y[values].sum()) for name, values in rows.items()},
        'models': records,
        'python': platform.python_version(),
        'packages': {
            name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost')
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / STEM
    temporary = destination.with_suffix(f'.json.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination.with_suffix('.json'))
    ordered = {name: test_scores[name] for name in (MODEL_X, MODEL_Y)}
    plot(y[rows['test']], ordered, {name: records[name]['test'] for name in ordered}, destination)
    print(f'Wrote {destination}.json, .pdf and .png', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
