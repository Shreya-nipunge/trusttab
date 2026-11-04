"""Tests for reliability visualizations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from trusttab.visualization import (
    plot_metric_comparison,
    plot_operating_region,
    plot_probability_distribution,
    plot_reliability_diagram,
    plot_roc_curve,
    save_figure,
)


@pytest.fixture
def sample_data():
    return (
        np.array([0, 0, 1, 1, 0, 1]),
        np.array([0.1, 0.3, 0.8, 0.9, 0.4, 0.7]),
    )


def test_reliability_diagram_returns_figure_and_axes(sample_data, tmp_path):
    y, p = sample_data
    output = tmp_path / "reliability.png"
    fig, ax = plot_reliability_diagram(
        y, p, n_bins=3, strategy="quantile", output_path=output
    )
    assert ax.get_title() == "Reliability diagram"
    assert len(ax.get_legend().get_texts()) == 2
    assert fig is ax.figure
    assert output.exists()
    plt.close(fig)


def test_probability_distribution_and_actual_class_legend(sample_data):
    y, p = sample_data
    fig, ax = plot_probability_distribution(p, bins=5, y_true=y)
    assert ax.get_title() == "Predicted probability distribution"
    assert len(ax.patches) > 0
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Actual negative",
        "Actual positive",
    ]
    plt.close(fig)


def test_probability_distribution_retains_probability_only_form(sample_data):
    _, p = sample_data
    fig, ax = plot_probability_distribution(p)
    assert ax.get_legend().get_texts()[0].get_text() == "All"
    plt.close(fig)


def test_operating_region_plot_has_metrics_and_default_threshold(sample_data):
    y, p = sample_data
    fig, axes = plot_operating_region(y, p, thresholds=[0.25, 0.5, 0.75])
    assert len(axes) == 3
    assert [ax.get_ylabel() for ax in axes] == ["Precision", "Recall", "F1"]
    assert all(
        any(np.all(np.asarray(line.get_xdata()) == 0.5) for line in ax.lines)
        for ax in axes
    )
    plt.close(fig)


def test_metric_comparison_aggregates_seeds_compares_models_and_saves(tmp_path):
    results = pd.DataFrame(
        {
            "model": ["LR", "LR", "RF", "RF"],
            "noise_rate": [0.0, 0.0, 0.1, 0.1],
            "seed": [1, 2, 1, 2],
            "adaptive_ece": [0.1, 0.3, 0.2, 0.4],
        }
    )
    output = tmp_path / "nested" / "metric.png"
    fig, ax = plot_metric_comparison(
        results, metric="adaptive_ece", output_path=output
    )
    assert len(ax.get_legend().get_texts()) == 2
    assert ax.get_ylabel() == "adaptive_ece"
    assert output.exists()
    plt.close(fig)


def test_metric_comparison_accepts_mce_alias():
    results = pd.DataFrame(
        {"model": ["LR"], "noise_rate": [0.0], "maximum_calibration_error": [0.2]}
    )
    fig, ax = plot_metric_comparison(results, metric="mce")
    assert ax.get_ylabel() == "mce"
    plt.close(fig)


def test_roc_curve_returns_figure_and_axes(sample_data):
    y, p = sample_data
    ax = plot_roc_curve(y, p)
    assert ax.get_title() == "ROC curve"
    assert len(ax.lines) == 2
    plt.close(ax.figure)


def test_save_figure_creates_parent_directory(tmp_path):
    fig, _ = plt.subplots()
    output = save_figure(fig, tmp_path / "nested" / "plot.png")
    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(fig)


@pytest.mark.parametrize(
    "function,args",
    [
        (plot_reliability_diagram, ([0, 1], [0.2, 0.8], 0)),
        (plot_reliability_diagram, ([0, 1], [0.2, 0.8], 2, "invalid")),
        (plot_probability_distribution, ([0.2, 0.8], 0)),
        (plot_roc_curve, ([0, 0], [0.2, 0.8])),
        (plot_operating_region, ([0, 1], [0.2, 0.8], [1.5])),
    ],
)
def test_invalid_plot_inputs_raise(function, args):
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize(
    "y,p",
    [
        ([0, 1], [0.2]),
        ([0, 2], [0.2, 0.8]),
        ([0, 1], [0.2, float("nan")]),
        ([0, 1], [0.2, 1.2]),
    ],
)
def test_invalid_binary_inputs_raise(y, p):
    with pytest.raises(ValueError):
        plot_reliability_diagram(y, p)


def test_probability_plot_validates_actual_labels():
    with pytest.raises(ValueError, match="y_true"):
        plot_probability_distribution([0.2, 0.8], y_true=[0, 2])


@pytest.mark.parametrize(
    "results,metric,message",
    [
        (pd.DataFrame({"model": ["LR"], "noise_rate": [0.1]}), "ece", "missing"),
        (
            pd.DataFrame({"model": ["LR"], "noise_rate": [0.1], "ece": [0.2]}),
            "unknown",
            "Unsupported metric",
        ),
        (pd.DataFrame(), "ece", "missing"),
    ],
)
def test_metric_comparison_validates_columns_and_metric(results, metric, message):
    with pytest.raises(ValueError, match=message):
        plot_metric_comparison(results, metric=metric)


def test_save_figure_rejects_invalid_dpi(tmp_path):
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="dpi"):
        save_figure(fig, tmp_path / "plot.png", dpi=0)
    plt.close(fig)
