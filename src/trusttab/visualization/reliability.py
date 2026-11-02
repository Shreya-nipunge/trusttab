"""Reliability visualization utilities for TrustTab."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve


def _validate_binary_inputs(y_true: Any, y_prob: Any) -> None:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("Inputs must be non-empty and have equal lengths.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities must be finite.")
    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")


def plot_reliability_diagram(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Any = None,
):
    """Plot observed positive frequency against predicted probability."""
    _validate_binary_inputs(y_true, y_prob)

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    observed, predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(predicted, observed, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive frequency",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Reliability diagram",
    )
    ax.legend()
    return ax


def plot_probability_distribution(y_prob: Any, bins: int = 10, ax: Any = None):
    """Plot the distribution of predicted positive-class probabilities."""
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.ndim != 1 or len(y_prob) == 0:
        raise ValueError("y_prob must be a non-empty one-dimensional array.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(y_prob, bins=bins, range=(0, 1), edgecolor="black")
    ax.set(
        xlabel="Predicted probability for positive class",
        ylabel="Count",
        xlim=(0, 1),
        title="Predicted probability distribution",
    )
    return ax


def plot_roc_curve(y_true: Any, y_prob: Any, ax: Any = None):
    """Plot the binary receiver operating characteristic curve."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    _validate_binary_inputs(y_true, y_prob)

    if np.unique(y_true).size != 2:
        raise ValueError("ROC curve requires both classes in y_true.")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="ROC curve",
    )
    ax.legend()
    return ax


def save_figure(figure: Any, path: str | Path, dpi: int = 150) -> Path:
    """Save a Matplotlib figure and return its path."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
