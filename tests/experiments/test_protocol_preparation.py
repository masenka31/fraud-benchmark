"""Preparation invariants shared by the two public paper protocols."""

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments.experiment import ExperimentConfig
from fraud_benchmark.experiments.experiment import ExperimentError
from fraud_benchmark.experiments.experiment import prepare
from fraud_benchmark.experiments.features.util import write_features


@pytest.fixture
def sparkov_features(tmp_path):
    n = 100
    event_time = pd.date_range('2020-01-01', periods=n, freq='1D')
    is_fraud = np.arange(n) % 10 == 0
    keys = pd.DataFrame(
        {
            'entity_id': pd.Series([f'card-{i % 5}' for i in range(n)], dtype='string'),
            'event_time': event_time,
            'reported_at': [
                time + pd.Timedelta(days=5 if i % 20 else 100) if fraud else pd.NaT
                for i, (time, fraud) in enumerate(zip(event_time, is_fraud))
            ],
            'reported_at_slow': [
                time + pd.Timedelta(days=20 if i % 20 else 200) if fraud else pd.NaT
                for i, (time, fraud) in enumerate(zip(event_time, is_fraud))
            ],
            'is_fraud': is_fraud,
            'split': pd.Series(['train'] * n, dtype='string'),
        }
    )
    features = pd.DataFrame(
        {
            'amount': np.linspace(1.0, 100.0, n),
            'category': pd.Series([f'cat-{i % 3}' for i in range(n)]).astype('category'),
            'artifact_merchant': pd.Series([f'shop-{i % 4}' for i in range(n)]).astype(
                'category'
            ),
        }
    )
    write_features(
        'sparkov',
        keys,
        features,
        features_dir=tmp_path,
        extra_keys=('reported_at_slow',),
    )
    return tmp_path


def test_sparkov_regimes_keep_rows_and_true_evaluation_labels(sparkov_features):
    prepared = {
        regime: prepare(
            ExperimentConfig(dataset='sparkov', label_delay=regime), sparkov_features
        )
        for regime in ('off', 'on', 'slow')
    }
    oracle = prepared['off']
    assert oracle.x.shape == (100, 2)
    assert 'artifact_merchant' not in oracle.names
    for regime in prepared.values():
        np.testing.assert_array_equal(regime.y_true, oracle.y_true)
        assert len(regime.train_rows) == len(oracle.train_rows)
        np.testing.assert_array_equal(
            regime.y[regime.val_rows], regime.y_true[regime.val_rows]
        )
        np.testing.assert_array_equal(
            regime.y[regime.test_rows], regime.y_true[regime.test_rows]
        )
    assert prepared['on'].censored_train_labels > 0
    assert prepared['slow'].censored_train_labels >= prepared['on'].censored_train_labels


def test_public_config_rejects_non_protocol_combinations():
    ExperimentConfig(dataset='ibm_ccf', split='pre_italy_68_16_16').validate()
    with pytest.raises(ExperimentError, match='pre-Italy'):
        ExperimentConfig(dataset='ibm_ccf').validate()
    with pytest.raises(ExperimentError, match='only for IBM'):
        ExperimentConfig(dataset='sparkov', split='pre_italy').validate()
