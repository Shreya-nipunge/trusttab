"""Classification and probability reliability metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def _validate_inputs(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("All inputs must be one-dimensional.")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("All inputs must have equal lengths.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities must be finite.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if np.unique(y_true).size != 2:
        raise ValueError("ROC-AUC requires both classes in y_true.")

    return y_true.astype(int), y_pred.astype(int), y_prob


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate binary top-label ECE using equal-width confidence bins."""
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

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    predicted = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1 - y_prob)
    correct = (predicted == y_true).astype(float)
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)

    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if mask.any():
            ece += mask.mean() * abs(
                correct[mask].mean() - confidence[mask].mean()
            )

    return float(ece)


def evaluate_classifier(y_true, y_pred, y_prob):
    """Return six binary classification and reliability metrics.

    y_prob must contain probabilities for positive class label 1.
    """
    y_true, y_pred, y_prob = _validate_inputs(y_true, y_pred, y_prob)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
