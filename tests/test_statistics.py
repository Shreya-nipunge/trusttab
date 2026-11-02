import numpy as np
import pytest

from trusttab.statistics import ComparisonResult, compare_paired_scores


def test_compare_paired_scores_returns_result():
    result = compare_paired_scores(
        [0.9, 0.85, 0.88, 0.92],
        [0.8, 0.82, 0.81, 0.86],
    )

    assert isinstance(result, ComparisonResult)
    assert result.mean_difference > 0
    assert 0 <= result.p_value <= 1
    assert result.confidence_interval[0] <= result.mean_difference
    assert result.mean_difference <= result.confidence_interval[1]


def test_difference_sign_is_correct():
    result = compare_paired_scores([0.9, 0.8, 0.7], [0.7, 0.6, 0.5])
    assert result.mean_difference == pytest.approx(0.2)


def test_confidence_interval_changes_with_confidence_level():
    a = [0.9, 0.8, 0.7, 0.95]
    b = [0.7, 0.65, 0.6, 0.8]

    narrow = compare_paired_scores(a, b, confidence_level=0.90)
    wide = compare_paired_scores(a, b, confidence_level=0.99)

    narrow_width = narrow.confidence_interval[1] - narrow.confidence_interval[0]
    wide_width = wide.confidence_interval[1] - wide.confidence_interval[0]
    assert wide_width > narrow_width


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([0.8, np.nan], [0.7, 0.6]),
        ([0.8, np.inf], [0.7, 0.6]),
        ([[0.8, 0.9]], [[0.7, 0.6]]),
        ([0.8], [0.7]),
        ([0.8, 0.9], [0.7]),
    ],
)
def test_invalid_scores_raise_value_error(a, b):
    with pytest.raises(ValueError):
        compare_paired_scores(a, b)


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.1])
def test_invalid_confidence_level_raises_value_error(level):
    with pytest.raises(ValueError):
        compare_paired_scores([0.8, 0.9], [0.7, 0.8], confidence_level=level)