"""Chronological label-rate features with no backward label propagation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_count(values: pd.Series, train_mask: np.ndarray) -> np.ndarray:
    """Count earlier training occurrences for every temporally ordered row."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(values) != len(train_mask):
        raise ValueError('values and train_mask must have equal length')
    codes, _ = pd.factorize(values.astype('string').fillna('__null__'), sort=False)
    updates = pd.Series(train_mask.astype('int64'))
    cumulative = updates.groupby(codes, sort=False).cumsum().to_numpy()
    return cumulative - train_mask.astype('int64')


def causal_target_rate(
    values: pd.Series,
    labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    alpha: float = 20.0,
    prior_frauds: float = 1.0,
    prior_strength: float = 1000.0,
) -> np.ndarray:
    """Score each row from earlier labelled training rows, then update on train.

    Rows must already be in temporal order. Validation/test labels never update the
    stream. A training row is scored before its own label is added.
    """
    if len(values) != len(labels) or len(labels) != len(train_mask):
        raise ValueError('values, labels and train_mask must have equal length')
    labels = np.asarray(labels, dtype='float64')
    train_mask = np.asarray(train_mask, dtype=bool)
    codes, _ = pd.factorize(values.astype('string').fillna('__null__'), sort=False)
    count_updates = train_mask.astype('int64')
    fraud_updates = labels * train_mask
    previous_count = (
        pd.Series(count_updates).groupby(codes, sort=False).cumsum().to_numpy() - count_updates
    )
    previous_fraud = (
        pd.Series(fraud_updates).groupby(codes, sort=False).cumsum().to_numpy() - fraud_updates
    )
    previous_train = np.cumsum(count_updates) - count_updates
    previous_global_fraud = np.cumsum(fraud_updates) - fraud_updates
    prior = (previous_global_fraud + prior_frauds) / (previous_train + prior_strength)
    return ((previous_fraud + alpha * prior) / (previous_count + alpha)).astype('float32')
