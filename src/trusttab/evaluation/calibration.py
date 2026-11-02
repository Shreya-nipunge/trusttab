"""Probability calibration utilities for TrustTab."""

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.utils.validation import check_is_fitted


def calibrate_classifier(
    estimator: Any,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Create a classifier with probability calibration.

    For an unfitted estimator, fit the returned object on training data.
    For an already-fitted estimator, use calibrate_fitted_classifier.
    """
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'.")

    if cv < 2:
        raise ValueError("cv must be at least 2.")

    return CalibratedClassifierCV(estimator, method=method, cv=cv)


def calibrate_fitted_classifier(
    estimator: Any,
    X_calibration: Any,
    y_calibration: Any,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrate a fitted classifier using separate calibration data."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'.")

    check_is_fitted(estimator)

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(estimator),
        method=method,
    )
    calibrated.fit(X_calibration, y_calibration)
    return calibrated


def get_calibration_curve(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean predicted probability and observed frequency per bin."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim != 1:
        raise ValueError("y_prob must be a one-dimensional array.")

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob must contain only finite values.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob values must be between 0 and 1.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain binary labels (0 and 1).")

    observed, predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    return predicted, observed
