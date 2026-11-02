"""Tests for reliability visualizations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trusttab.visualization import (
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


def test_reliability_diagram(sample_data):
    y, p = sample_data
    ax = plot_reliability_diagram(y, p, n_bins=3)
    assert ax.get_title() == "Reliability diagram"
    assert len(ax.lines) == 2
    plt.close(ax.figure)


def test_probability_distribution(sample_data):
    _, p = sample_data
    ax = plot_probability_distribution(p)
    assert ax.get_title() == "Predicted probability distribution"
    assert len(ax.patches) > 0
    plt.close(ax.figure)


def test_roc_curve(sample_data):
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
        (plot_probability_distribution, ([0.2, 0.8], 0)),
        (plot_roc_curve, ([0, 0], [0.2, 0.8])),
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


def test_save_figure_rejects_invalid_dpi(tmp_path):
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="dpi"):
        save_figure(fig, tmp_path / "plot.png", dpi=0)
    plt.close(fig)