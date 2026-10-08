"""Causal sequence boundaries and label-free LSTM inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments import lstm
from fraud_benchmark.experiments.lstm import encode_features
from fraud_benchmark.experiments.sequence import complete_chunks
from fraud_benchmark.experiments.sequence import iid_chunk_split
from fraud_benchmark.experiments.sequence import window_index
from fraud_benchmark.experiments.sequence_experiment import IBM_PAPER_REGIMES
from fraud_benchmark.experiments.sequence_experiment import SPARKOV_RAWISH_FEATURES
from fraud_benchmark.experiments.sequence_experiment import _ibm_split
from fraud_benchmark.experiments.sequence_experiment import prepare_ibm
from fraud_benchmark.experiments.sequence_experiment import prepare_sparkov
from fraud_benchmark.experiments.sequence_experiment import run_ibm_xgboost


def test_complete_chunks_use_only_final_targets_and_never_mix_entities():
    entity = pd.Series(np.tile(['a', 'b'], 65))
    times = pd.Series(
        pd.Timestamp('2024-01-01') + pd.to_timedelta(np.repeat(np.arange(65), 2), unit='min')
    )
    chunks = complete_chunks(entity, times)
    assert chunks.rows.shape == (4, 30)
    assert chunks.n_dropped == 10
    assert len(np.unique(chunks.rows)) == 120
    assert np.array_equal(np.sort(chunks.targets), [58, 59, 118, 119])
    for rows in chunks.rows:
        assert entity.iloc[rows].nunique() == 1
        assert times.iloc[rows].is_monotonic_increasing
        assert times.iloc[rows[:-1]].max() <= times.iloc[rows[-1]]


def test_ten_transaction_chunks_triple_complete_targets_and_bound_windows():
    entity = pd.Series(np.repeat(['a', 'b'], 31))
    times = pd.Series(
        pd.Timestamp('2024-01-01') + pd.to_timedelta(np.tile(np.arange(31), 2), unit='min')
    )
    long = complete_chunks(entity, times, length=30)
    short = complete_chunks(entity, times, length=10)
    assert long.rows.shape == (2, 30)
    assert short.rows.shape == (6, 10)
    assert short.n_dropped == 2
    assert np.array_equal(short.targets, [9, 19, 29, 40, 50, 60])
    index = window_index(entity, times, length=10)
    positions, lengths = index.batch(np.array([30, 61]))
    assert positions.shape == (2, 10)
    assert lengths.tolist() == [10, 10]
    assert np.array_equal(index.order[positions[:, -1]], [30, 61])
    with pytest.raises(ValueError, match='at least 2'):
        complete_chunks(entity, times, length=1)
    with pytest.raises(ValueError, match='at least 2'):
        window_index(entity, times, length=1)


def test_ibm_68_16_16_assigns_chunk_targets_by_endpoint_time():
    frame = pd.DataFrame(
        {
            'entity_id': ['a'] * 100,
            'event_time': pd.date_range('2010-01-01', periods=100, freq='min'),
            'split': ['train'] * 100,
        }
    )
    chunks = complete_chunks(frame['entity_id'], frame['event_time'], length=10)
    targets, source = _ibm_split(frame, chunks, 'pre_italy_68_16_16')
    assert np.bincount(source).tolist() == [68, 16, 16]
    assert np.bincount(targets).tolist() == [6, 2, 2]


def test_chunk_iid_assignment_is_fixed_and_whole_chunk():
    first = iid_chunk_split(101, seed=20260806)
    assert np.array_equal(first, iid_chunk_split(101, seed=20260806))
    assert np.bincount(first).tolist() == [80, 10, 11]
    assert not np.array_equal(first, iid_chunk_split(101, seed=1))


def test_window_index_excludes_same_time_and_other_entities():
    entity = pd.Series(['a', 'b', 'a', 'a', 'b', 'a'])
    times = pd.Series(
        pd.to_datetime(
            [
                '2024-01-01 00:00',
                '2024-01-01 00:00',
                '2024-01-01 00:00',
                '2024-01-01 00:01',
                '2024-01-01 00:02',
                '2024-01-01 00:03',
            ]
        )
    )
    index = window_index(entity, times)
    positions, lengths = index.batch(np.array([2, 3, 4, 5]))
    histories = [index.order[row[:length]].tolist() for row, length in zip(positions, lengths)]
    assert histories == [[2], [0, 2, 3], [1, 4], [0, 2, 3, 5]]


def test_encoder_fits_train_only_and_keeps_missing_mask():
    frame = pd.DataFrame(
        {
            'amount': [1.0, 3.0, np.nan, 1000.0],
            'category': pd.Categorical(['known', 'known', 'known', 'future']),
        }
    )
    encoded = encode_features(frame, ['amount', 'category'], np.array([True, True, True, False]))
    assert encoded.numeric[3, 0] > 100
    assert encoded.missing[:, 0].tolist() == [0, 0, 1, 0]
    assert encoded.categorical[3, 0] == encoded.cardinalities[0] - 1
    assert encoded.tree_matrix(np.array([3])).shape == (1, 2)


def test_sparkov_delay_changes_training_targets_but_not_windows():
    n = 100
    times = pd.Series(pd.Timestamp('2024-01-01') + pd.to_timedelta(np.arange(n), unit='h'))
    frame = pd.DataFrame(
        {
            'entity_id': np.repeat('card', n),
            'event_time': times,
            'is_fraud': np.isin(np.arange(n), [0, 70, 85, 95]).astype('int8'),
            'reported_at': pd.NaT,
            'reported_at_slow': pd.NaT,
            'split': 'train',
        }
    )
    frame.loc[[0, 70, 85, 95], 'reported_at'] = times.iloc[[1, 90, 86, 96]].to_numpy()
    frame.loc[[0, 70, 85, 95], 'reported_at_slow'] = times.iloc[[90, 91, 86, 96]].to_numpy()
    for index in range(46):
        frame[f'feature_{index}'] = float(index)
    off = prepare_sparkov('off', source=frame)
    on = prepare_sparkov('on', source=frame)
    slow = prepare_sparkov('slow', source=frame)
    assert np.array_equal(off.y_true, on.y_true)
    assert np.array_equal(off.y_true, slow.y_true)
    assert on.metadata['censored_train_labels'] == 1
    assert slow.metadata['censored_train_labels'] == 2
    assert np.array_equal(on.y_fit[on.validation], on.y_true[on.validation])
    assert len(on.features.names) == 46
    assert not {'is_fraud', 'reported_at', 'reported_at_slow'} & set(on.features.names)
    targets = np.array([0, 85, 95])
    assert np.array_equal(off.positions_for(targets)[0], on.positions_for(targets)[0])
    short = prepare_sparkov('off', source=frame, window_length=10)
    assert short.metadata['window_length'] == 10
    assert short.positions_for(targets)[0].shape == (3, 10)
    rawish_source = frame.copy()
    for name in SPARKOV_RAWISH_FEATURES:
        rawish_source[name] = (
            pd.Categorical(['x'] * n) if name in ('category', 'job', 'gender') else 1.0
        )
    rawish = prepare_sparkov('on', source=rawish_source, window_length=10, feature_set='rawish')
    assert rawish.metadata['feature_set'] == 'rawish'
    assert rawish.metadata['censored_train_labels'] == on.metadata['censored_train_labels']
    assert rawish.features.names == list(SPARKOV_RAWISH_FEATURES)
    assert rawish.positions_for(targets)[0].shape == (3, 10)


def test_ibm_regimes_keep_the_same_endpoints_and_no_label_derived_inputs():
    entities = np.tile(np.arange(20), 300)
    ordinal = np.repeat(np.arange(300), 20)
    frame = pd.DataFrame(
        {
            'entity_id': entities,
            'event_time': pd.Timestamp('2016-01-01') + pd.to_timedelta(ordinal, unit='min'),
            'reported_at': pd.NaT,
            'is_fraud': (
                (ordinal % 30 == 29)
                & np.isin(ordinal // 30, [0, 4, 8, 9])
                & ((ordinal // 30 < 8) | (entities % 2 == 0))
            ).astype('int8'),
            'split': 'train',
            'merchant_is_foreign': np.zeros(len(ordinal), dtype='float32'),
            'artifact_merchant_state': pd.Categorical(np.where(entities % 2, 'NY', 'CA')),
        }
    )
    for index in range(73):
        frame[f'feature_{index}'] = np.ones(len(frame), dtype='float32')
    prepared = [prepare_ibm(regime, source=frame) for regime in IBM_PAPER_REGIMES]
    endpoints = [
        item.positions_for(np.arange(len(item.y_true), dtype='int32'))[0][:, -1]
        for item in prepared
    ]
    assert all(np.array_equal(endpoints[0], other) for other in endpoints[1:])
    assert all(item.metadata['target_rows'] == 200 for item in prepared)
    assert all(item.metadata['dropped_incomplete_rows'] == 0 for item in prepared)
    assert all(item.metadata['positives']['val'] > 0 for item in prepared)
    assert all(item.metadata['positives']['test'] > 0 for item in prepared)
    assert all(len(item.features.names) == 77 for item in prepared)
    assert all(
        not any(name.startswith('causal_target_rate_') for name in item.features.names)
        for item in prepared
    )
    comparator = run_ibm_xgboost(prepared[0], seeds=(0,))
    assert comparator['model'] == 'xgboost'
    assert np.isfinite(comparator['seeds'][0]['test']['average_precision'])
    short = prepare_ibm('pre_italy', source=frame, window_length=10)
    assert short.metadata['target_rows'] == 600
    assert short.metadata['chunk_length'] == 10
    assert short.positions_for(np.array([0]))[0].shape == (1, 10)


def test_lstm_training_scores_the_requested_targets(monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.setattr(lstm, 'MAX_EPOCHS', 1)
    monkeypatch.setattr(lstm, 'BATCH_SIZE', 8)
    n = 60
    frame = pd.DataFrame(
        {
            'amount': np.arange(n, dtype='float32'),
            'category': pd.Categorical(np.where(np.arange(n) % 2, 'a', 'b')),
        }
    )
    features = encode_features(frame, ['amount', 'category'], np.arange(n) < 40)
    labels = (np.arange(n) % 7 == 0).astype('int64')

    def pairs(ids):
        previous = np.where(ids % 10 == 0, ids, ids - 1)
        return np.column_stack((previous, ids)).astype('int32'), np.full(len(ids), 2)

    validation, test, details = lstm.train_and_predict(
        features,
        pairs,
        np.arange(40),
        np.arange(40, 50),
        np.arange(50, 60),
        labels,
        labels,
        seed=0,
        device_name='cpu',
    )
    assert validation.shape == (10,)
    assert test.shape == (10,)
    assert np.isfinite(validation).all()
    assert np.isfinite(test).all()
    assert details['best_epoch'] == 1
