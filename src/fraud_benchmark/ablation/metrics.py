"""Scoring.

Average precision throughout. ROC AUC is deliberately absent: base rates in this
suite run from 0.10% to 0.52%, where ROC AUC is dominated by the negative class
and stays high for a model with no useful precision -- exactly the differences
this study exists to measure.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def score(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Average precision, plus precision/recall/F1 at `threshold`."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype="float64")
    n_positive = int(y_true.sum())

    if n_positive == 0:
        ap = float("nan")
    else:
        ap = float(average_precision_score(y_true, y_score))

    predicted = y_score >= threshold
    tp = int((predicted & (y_true == 1)).sum())
    fp = int((predicted & (y_true == 0)).sum())
    fn = int((~predicted & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "n_rows": int(len(y_true)),
        "n_positive": n_positive,
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The threshold maximising F1. Chosen on validation, applied unchanged to test."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])
