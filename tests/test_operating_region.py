"""Tests for operating-region analysis."""

import pytest

from trusttab.evaluation.operating_region import (
    OperatingPoint,
    analyze_operating_region,
)


def test_analyze_operating_region_at_custom_threshold():
    results = analyze_operating_region(
        [0, 1, 1],
        [0.2, 0.8, 0.4],
        thresholds=[0.5],
    )

    assert len(results) == 1
    assert isinstance(results[0], OperatingPoint)
    assert results[0].threshold == 0.5
    assert results[0].accuracy == pytest.approx(2 / 3)
    assert results[0].precision == pytest.approx(1.0)
    assert results[0].recall == pytest.approx(0.5)
    assert results[0].f1 == pytest.approx(2 / 3)
    assert results[0].predicted_positive_rate == pytest.approx(1 / 3)


def test_default_thresholds_cover_zero_to_one():
    results = analyze_operating_region([0, 1], [0.2, 0.8])

    assert len(results) == 21
    assert results[0].threshold == pytest.approx(0.0)
    assert results[-1].threshold == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("y_true", "y_prob"),
    [
        ([0, 1], [0.5]),
        ([], []),
        ([[0, 1]], [[0.2, 0.8]]),
        ([0, 2], [0.2, 0.8]),
        ([0, 1], [0.2, float("nan")]),
        ([0, 1], [-0.1, 0.8]),
        ([0, 1], [0.2, 1.1]),
    ],
)
def test_invalid_inputs_raise_value_error(y_true, y_prob):
    with pytest.raises(ValueError):
        analyze_operating_region(y_true, y_prob)


@pytest.mark.parametrize(
    "thresholds",
    [
        [],
        [[0.5]],
        [-0.1, 0.5],
        [0.5, 1.1],
        [float("nan")],
    ],
)
def test_invalid_thresholds_raise_value_error(thresholds):
    with pytest.raises(ValueError):
        analyze_operating_region([0, 1], [0.2, 0.8], thresholds)