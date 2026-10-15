"""Small fixed LSTM classifier for the two sequence protocols."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass

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
MODEL_SETTINGS = {
    'architecture': 'one-layer unidirectional sequence-to-one LSTM',
    'hidden_size': HIDDEN_SIZE,
    'embedding_size': EMBEDDING_SIZE,
    'dropout': DROPOUT,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'batch_size': BATCH_SIZE,
    'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE,
    'gradient_clip': GRADIENT_CLIP,
    'loss': 'binary cross entropy with visible-label negative/positive weight',
}


@dataclass(frozen=True)
class LSTMConfig:
    """Training settings; the default reproduces the recorded protocol exactly."""

    learning_rate: float = LEARNING_RATE
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    positive_weight: str = 'ratio'  # 'ratio' = negatives/positives, 'sqrt' = its square root
    warmup_cosine: bool = False  # per-step linear warmup over one epoch, then cosine decay
    input_layer_norm: bool = False


DEFAULT_CONFIG = LSTMConfig()
SLOW = {'learning_rate': 0.0003, 'max_epochs': 15, 'patience': 4, 'warmup_cosine': True}
LSTM_CONFIGS = {
    'default': DEFAULT_CONFIG,
    'sqrt_weight': LSTMConfig(positive_weight='sqrt'),
    'full_weight_slow': LSTMConfig(**SLOW),
    'sqrt_weight_slow': LSTMConfig(positive_weight='sqrt', **SLOW),
    'sqrt_weight_slow_ln': LSTMConfig(positive_weight='sqrt', input_layer_norm=True, **SLOW),
}


def model_settings(name: str = 'default') -> dict:
    """Recorded settings; the default record keeps its original form."""
    if name == 'default':
        return MODEL_SETTINGS
    config = LSTM_CONFIGS[name]
    return {
        **MODEL_SETTINGS,
        'config': name,
        **asdict(config),
        'loss': 'binary cross entropy with visible-label positive weight '
        + ('negatives/positives' if config.positive_weight == 'ratio'
           else 'sqrt(negatives/positives)'),
    }


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

    def tree_matrix(self, rows: np.ndarray) -> np.ndarray:
        """The same current-event information as the LSTM, at selected targets."""
        values: dict[str, np.ndarray] = {}
        for index, name in enumerate(self.numeric_names):
            column = self.numeric[rows, index].copy()
            column[self.missing[rows, index].astype(bool)] = np.nan
            values[name] = column
        for index, name in enumerate(self.categorical_names):
            values[name] = self.categorical[rows, index].astype('float32')
        return np.column_stack([values[name] for name in self.names]).astype('float32')


def encode_features(df: pd.DataFrame, names: list[str], train_mask: np.ndarray) -> EncodedFeatures:
    """Fit vocabulary and numeric scaling on train source rows only."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(train_mask) != len(df) or not train_mask.any():
        raise ValueError('train_mask must cover the frame and contain training rows')
    categorical_names = [name for name in names if isinstance(df[name].dtype, pd.CategoricalDtype)]
    numeric_names = [name for name in names if name not in categorical_names]
    encoder = CappedOrdinalEncoder().fit(df.loc[train_mask, categorical_names], categorical_names)
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


def _model(n_numeric: int, cardinalities: list[int], input_layer_norm: bool = False):
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
            self.norm = nn.LayerNorm(HIDDEN_SIZE) if input_layer_norm else nn.Identity()
            self.lstm = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
            self.dropout = nn.Dropout(DROPOUT)
            self.head = nn.Linear(HIDDEN_SIZE, 1)

        def forward(self, numeric, missing, categorical, lengths):
            pieces = [numeric, missing]
            pieces.extend(
                embedding(categorical[:, :, index])
                for index, embedding in enumerate(self.embeddings)
            )
            events = torch.relu(self.norm(self.project(torch.cat(pieces, dim=-1))))
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
    config: LSTMConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit on visible train labels; checkpoint by validation AP only."""
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(
        'cuda'
        if device_name == 'auto' and torch.cuda.is_available()
        else 'cpu'
        if device_name == 'auto'
        else device_name
    )
    model = _model(
        features.numeric.shape[1], features.cardinalities, config.input_layer_norm
    ).to(device)
    positives = int(y_fit[train].sum())
    if positives == 0:
        raise ValueError('training targets have no visible fraud labels')
    weight = (len(train) - positives) / positives
    if config.positive_weight == 'sqrt':
        weight = math.sqrt(weight)
    elif config.positive_weight != 'ratio':
        raise ValueError(f'unknown positive weight {config.positive_weight!r}')
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = max(1, int(np.ceil(len(train) / BATCH_SIZE)))
    scheduler = None
    if config.warmup_cosine:
        total = steps_per_epoch * config.max_epochs

        def factor(step: int) -> float:
            if step < steps_per_epoch:
                return (step + 1) / steps_per_epoch
            progress = (step - steps_per_epoch) / max(1, total - steps_per_epoch)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    rng = np.random.default_rng(seed)
    best_ap = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_gain = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_ids in np.array_split(rng.permutation(train), steps_per_epoch):
            positions, lengths = positions_for(batch_ids)
            batch = _tensor_batch(features, positions, lengths, device)
            labels = torch.as_tensor(y_fit[batch_ids], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*batch), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
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
            if epochs_without_gain >= config.patience:
                break
    if best_state is None:
        raise RuntimeError('LSTM training did not produce a validation checkpoint')
    model.load_state_dict(best_state)
    validation_scores = _predict(model, features, validation, positions_for, device)
    test_scores = _predict(model, features, test, positions_for, device)
    details = {
        'device': str(device),
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch': torch.__version__,
        'positive_weight': float(weight),
        'best_epoch': best_epoch,
        'history': history,
    }
    return validation_scores, test_scores, details
