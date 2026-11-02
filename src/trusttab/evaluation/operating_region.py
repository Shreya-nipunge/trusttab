"""Operating-region analysis across binary classification thresholds."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class OperatingPoint:
    """Classification metrics at a particular decision threshold."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    predicted_positive_rate: float


def analyze_operating_region(y_true, y_prob, thresholds=None):
    """Evaluate binary classification performance at decision thresholds.

    Probabilities are for positive class label 1. Predictions use
    ``y_prob >= threshold`` as the positive-class decision rule.

    If thresholds are omitted, evaluate thresholds from 0.0 to 1.0
    in increments of 0.05.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("Inputs must be non-empty and have equal lengths.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities must be finite.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array.")

    if not np.isfinite(thresholds).all():
        raise ValueError("Thresholds must be finite.")

    if ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Thresholds must be between 0 and 1.")

    y_true = y_true.astype(int)
    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        results.append(
            OperatingPoint(
                threshold=float(threshold),
                accuracy=float(accuracy_score(y_true, y_pred)),
                precision=float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                predicted_positive_rate=float(y_pred.mean()),
            )
        )

    return results