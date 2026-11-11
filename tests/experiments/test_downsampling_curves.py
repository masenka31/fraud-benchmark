"""The Figure 2 features use training labels only, and only earlier ones for training rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'ibm_downsampling_curves.py'


@pytest.fixture(scope='module')
def runner():
    pytest.importorskip('matplotlib')
    pytest.importorskip('sklearn')
    pytest.importorskip('xgboost')
    spec = importlib.util.spec_from_file_location('ibm_downsampling_curves', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(labels, states):
    data = {
        column: pd.Categorical(['a'] * len(labels))
        for column in (
            'artifact_merchant_city',
            'artifact_merchant_name',
            'artifact_mcc',
            'artifact_error_type',
        )
    }
    return pd.DataFrame(
        {
            'artifact_merchant_state': pd.Categorical(states),
            'merchant_is_foreign': np.zeros(len(labels)),
            'is_fraud': labels,
            **data,
        }
    )


def test_training_rates_see_only_earlier_training_labels(runner):
    train = np.array([True, True, True, False, False])
    before = runner.engineered_features(_frame([0, 0, 1, 0, 1], list('xxxxy')), train)[0]
    # Flip the last training label and every evaluation label.
    after = runner.engineered_features(_frame([0, 0, 0, 1, 0], list('xxxxy')), train)[0]
    rate = 3 + runner.RATE_COLUMNS.index('artifact_merchant_state')
    np.testing.assert_array_equal(before[:3, rate], after[:3, rate])
    assert before[3, rate] != after[3, rate]  # evaluation rows use all training labels


def test_evaluation_labels_never_change_any_feature(runner):
    train = np.array([True, True, True, False, False])
    before = runner.engineered_features(_frame([0, 1, 0, 0, 0], list('xxyxz')), train)[0]
    after = runner.engineered_features(_frame([0, 1, 0, 1, 1], list('xxyxz')), train)[0]
    np.testing.assert_array_equal(before, after)
    names = runner.engineered_features(_frame([0, 1, 0, 0, 0], list('xxyxz')), train)[1]
    unseen = before[:, names.index('state_unseen_in_train')]
    np.testing.assert_array_equal(unseen, [0, 0, 0, 0, 1])
