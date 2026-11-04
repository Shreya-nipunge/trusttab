"""Reliability visualization and diagnostic utilities for TrustTab."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from trusttab.evaluation import analyze_operating_region

_RELIABILITY_METRICS = {
    "ece": "ece",
    "adaptive_ece": "adaptive_ece",
    "maximum_calibration_error": "maximum_calibration_error",
    "mce": "maximum_calibration_error",
    "brier_score": "brier_score",
    "log_loss": "log_loss",
}


def _validate_binary_inputs(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
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
    return y_true.astype(int), y_prob


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _figure_axes(ax: Any = None) -> tuple[Figure, Axes]:
    if ax is None:
        return plt.subplots()
    return ax.figure, ax


def _save_if_requested(figure: Any, output_path: str | Path | None) -> None:
    if output_path is not None:
        save_figure(figure, output_path)


def plot_reliability_diagram(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Any = None,
    *,
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot observed positive frequency against mean predicted probability.

    Empty bins are omitted by the calibration calculation. ``strategy`` may
    be ``"uniform"`` or ``"quantile"``. Returns the figure and axes.
    """
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    _positive_integer(n_bins, "n_bins")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    observed, predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    figure, axes = _figure_axes(ax)
    axes.plot(predicted, observed, marker="o", label="Model")
    axes.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axes.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive frequency",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Reliability diagram",
    )
    axes.legend()
    _save_if_requested(figure, output_path)
    return figure, axes


def plot_probability_distribution(
    y_prob: Any,
    bins: int = 10,
    ax: Any = None,
    *,
    y_true: Any = None,
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot predicted probabilities, optionally separated by actual class.

    ``y_prob`` contains positive-class probabilities. When ``y_true`` is
    supplied, positive and negative examples are drawn as separate histograms.
    The probability-only form remains available for compatibility.
    """
    _positive_integer(bins, "bins")
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 1 or len(y_prob) == 0:
        raise ValueError("y_prob must be a non-empty one-dimensional array.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    if y_true is not None:
        y_true = np.asarray(y_true)
        if y_true.ndim != 1 or len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must be one-dimensional and equal length.")
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("y_true must contain only binary labels 0 and 1.")

    figure, axes = _figure_axes(ax)
    if y_true is None:
        axes.hist(y_prob, bins=bins, range=(0, 1), edgecolor="black", label="All")
    else:
        axes.hist(
            [y_prob[y_true == 0], y_prob[y_true == 1]],
            bins=bins,
            range=(0, 1),
            edgecolor="black",
            label=["Actual negative", "Actual positive"],
            alpha=0.7,
        )
    axes.legend()
    axes.set(
        xlabel="Predicted probability for positive class",
        ylabel="Count",
        xlim=(0, 1),
        title="Predicted probability distribution",
    )
    _save_if_requested(figure, output_path)
    return figure, axes


def plot_operating_region(
    y_true: Any,
    y_prob: Any,
    thresholds: Any = None,
    *,
    output_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot precision, recall, and F1 against classification threshold.

    Returns one figure and an array of three axes, one per metric.
    """
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    points = analyze_operating_region(y_true, y_prob, thresholds=thresholds)
    threshold_values = [point.threshold for point in points]
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
    for axis, name, values in zip(
        axes,
        ("Precision", "Recall", "F1"),
        (
            [point.precision for point in points],
            [point.recall for point in points],
            [point.f1 for point in points],
        ),
    ):
        axis.plot(threshold_values, values, marker="o", label=name)
        axis.axvline(0.5, color="black", linestyle="--", label="Default threshold (0.5)")
        axis.set(ylabel=name, ylim=(0, 1))
        axis.legend()
    axes[-1].set_xlabel("Decision threshold")
    figure.suptitle("Operating-region metrics")
    figure.tight_layout()
    _save_if_requested(figure, output_path)
    return figure, axes


def plot_metric_comparison(
    results: pd.DataFrame,
    metric: str = "ece",
    ax: Any = None,
    *,
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare a reliability metric across models and noise levels.

    Values are averaged across seeds for each model/noise combination, with
    standard deviation shown as error bars when repeated seeds are available.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    normalized_metric = (
        metric.strip().lower().replace(" ", "_") if isinstance(metric, str) else ""
    )
    if normalized_metric not in _RELIABILITY_METRICS:
        choices = ", ".join(_RELIABILITY_METRICS)
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {choices}.")
    metric_column = _RELIABILITY_METRICS[normalized_metric]
    required = {"model", "noise_rate", metric_column}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("results must contain at least one row.")

    data = results[["model", "noise_rate", metric_column]].copy()
    data["noise_rate"] = pd.to_numeric(data["noise_rate"], errors="coerce")
    data[metric_column] = pd.to_numeric(data[metric_column], errors="coerce")
    if (
        data["model"].isna().any()
        or not np.isfinite(data["noise_rate"]).all()
        or not np.isfinite(data[metric_column]).all()
    ):
        raise ValueError("Model names, noise rates, and metric values must be valid and finite.")

    summary = (
        data.groupby(["model", "noise_rate"], sort=True)[metric_column]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure, axes = _figure_axes(ax)
    for model, group in summary.groupby("model", sort=True):
        yerr = group["std"].fillna(0).to_numpy()
        axes.errorbar(
            group["noise_rate"],
            group["mean"],
            yerr=yerr,
            marker="o",
            capsize=3,
            label=str(model),
        )
    axes.set(
        xlabel="Training label-noise rate",
        ylabel=metric,
        title=f"{metric} across label-noise levels",
    )
    axes.legend(title="Model")
    _save_if_requested(figure, output_path)
    return figure, axes


def plot_roc_curve(y_true: Any, y_prob: Any, ax: Any = None) -> Axes:
    """Plot the binary receiver operating characteristic curve."""
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    if np.unique(y_true).size != 2:
        raise ValueError("ROC curve requires both classes in y_true.")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    _figure, axes = _figure_axes(ax)
    axes.plot(fpr, tpr, label="Model")
    axes.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    axes.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="ROC curve",
    )
    axes.legend()
    return axes


def save_figure(figure: Any, path: str | Path, dpi: int = 150) -> Path:
    """Save a Matplotlib figure and return its path."""
    _positive_integer(dpi, "dpi")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
