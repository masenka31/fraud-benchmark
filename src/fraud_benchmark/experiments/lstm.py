"""Small fixed LSTM classifier for the two sequence protocols."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_benchmark.experiments.encoding import CappedOrdinalEncoder

HIDDEN_SIZE = 64
EMBEDDING_SIZE = 8
DROPOUT = 0.2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 1024
MAX_EPOCHS = 8
PATIENCE = 2
GRADIENT_CLIP = 1.0


@dataclass
class EncodedFeatures:
    numeric: np.ndarray
    missing: np.ndarray
    categorical: np.ndarray
    names: list[str]
    numeric_names: list[str]
    categorical_names: list[str]
    cardinalities: list[int]

    def reordered(self, order: np.ndarray) -> EncodedFeatures:
        return EncodedFeatures(
            self.numeric[order],
            self.missing[order],
            self.categorical[order],
            self.names,
            self.numeric_names,
            self.categorical_names,
            self.cardinalities,
        )

    def tree_matrix(self) -> np.ndarray:
        """The same information as the LSTM receives, one row per transaction."""
        values: dict[str, np.ndarray] = {}
        for index, name in enumerate(self.numeric_names):
            column = self.numeric[:, index].copy()
            column[self.missing[:, index].astype(bool)] = np.nan
            values[name] = column
        for index, name in enumerate(self.categorical_names):
            values[name] = self.categorical[:, index].astype('float32')
        return np.column_stack([values[name] for name in self.names]).astype('float32')


def encode_features(
    df: pd.DataFrame, names: list[str], train_mask: np.ndarray
) -> EncodedFeatures:
    """Fit vocabulary and numeric scaling on train source rows only."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(train_mask) != len(df) or not train_mask.any():
        raise ValueError('train_mask must cover the frame and contain training rows')
    categorical_names = [name for name in names if isinstance(df[name].dtype, pd.CategoricalDtype)]
    numeric_names = [name for name in names if name not in categorical_names]
    encoder = CappedOrdinalEncoder().fit(df.loc[train_mask], categorical_names)
    numeric = np.empty((len(df), len(numeric_names)), dtype='float32')
    missing = np.empty((len(df), len(numeric_names)), dtype='uint8')
    for index, name in enumerate(numeric_names):
        values = pd.to_numeric(df[name], errors='coerce').to_numpy(dtype='float32')
        finite = np.isfinite(values)
        observed = values[train_mask & finite].astype('float64')
        mean = float(observed.mean()) if len(observed) else 0.0
        std = float(observed.std()) if len(observed) else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        numeric[:, index] = np.where(finite, (values - mean) / std, 0.0)
        missing[:, index] = ~finite
    categorical = np.empty((len(df), len(categorical_names)), dtype='int16')
    cardinalities = []
    for index, name in enumerate(categorical_names):
        cardinality = encoder.cardinality(name)
        if cardinality > np.iinfo('int16').max:
            raise ValueError(f'{name}: categorical vocabulary exceeds int16')
        categorical[:, index] = encoder.transform_column(df[name], name).astype('int16')
        cardinalities.append(cardinality)
    return EncodedFeatures(
        numeric, missing, categorical, names, numeric_names, categorical_names, cardinalities
    )


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('LSTM protocols require the optional `lstm` dependencies') from exc
    return torch


def _model(n_numeric: int, cardinalities: list[int]):
    torch = _torch()
    from torch import nn

    class SequenceClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                nn.Embedding(size, EMBEDDING_SIZE) for size in cardinalities
            )
            width = 2 * n_numeric + EMBEDDING_SIZE * len(cardinalities)
            self.project = nn.Linear(width, HIDDEN_SIZE)
            self.lstm = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
            self.dropout = nn.Dropout(DROPOUT)
            self.head = nn.Linear(HIDDEN_SIZE, 1)

        def forward(self, numeric, missing, categorical, lengths):
            pieces = [numeric, missing]
            pieces.extend(
                embedding(categorical[:, :, index])
                for index, embedding in enumerate(self.embeddings)
            )
            events = torch.relu(self.project(torch.cat(pieces, dim=-1)))
            outputs, _ = self.lstm(events)
            last = outputs[torch.arange(len(lengths), device=outputs.device), lengths - 1]
            return self.head(self.dropout(last)).squeeze(-1)

    return SequenceClassifier()


def _tensor_batch(
    features: EncodedFeatures,
    positions: np.ndarray,
    lengths: np.ndarray,
    device,
):
    torch = _torch()
    numeric = torch.from_numpy(features.numeric[positions]).to(device)
    missing = torch.from_numpy(features.missing[positions].astype('float32')).to(device)
    categorical = torch.from_numpy(features.categorical[positions].astype('int64')).to(device)
    lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
    return numeric, missing, categorical, lengths_tensor


def _predict(
    model,
    features: EncodedFeatures,
    targets: np.ndarray,
    positions_for: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    device,
) -> np.ndarray:
    torch = _torch()
    result = np.empty(len(targets), dtype='float32')
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(targets), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(targets))
            positions, lengths = positions_for(targets[start:stop])
            batch = _tensor_batch(features, positions, lengths, device)
            result[start:stop] = torch.sigmoid(model(*batch)).cpu().numpy()
    return result


def train_and_predict(
    features: EncodedFeatures,
    positions_for: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    y_fit: np.ndarray,
    y_true: np.ndarray,
    seed: int,
    device_name: str = 'auto',
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit on visible train labels; checkpoint by validation AP only."""
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(
        'cuda' if device_name == 'auto' and torch.cuda.is_available() else
        'cpu' if device_name == 'auto' else device_name
    )
    model = _model(features.numeric.shape[1], features.cardinalities).to(device)
    positives = int(y_fit[train].sum())
    if positives == 0:
        raise ValueError('training targets have no visible fraud labels')
    weight = (len(train) - positives) / positives
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    best_ap = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_gain = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_ids in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train) / BATCH_SIZE)))):
            positions, lengths = positions_for(batch_ids)
            batch = _tensor_batch(features, positions, lengths, device)
            labels = torch.as_tensor(y_fit[batch_ids], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*batch), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_ids)
        val_scores = _predict(model, features, validation, positions_for, device)
        val_ap = float(average_precision_score(y_true[validation], val_scores))
        history.append({'epoch': epoch, 'train_loss': total_loss / len(train), 'val_ap': val_ap})
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    validation_scores = _predict(model, features, validation, positions_for, device)
    test_scores = _predict(model, features, test, positions_for, device)
    details = {
        'device': str(device),
        'torch': torch.__version__,
        'positive_weight': float(weight),
        'best_epoch': best_epoch,
        'history': history,
    }
    return validation_scores, test_scores, details
