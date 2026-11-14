"""Tests for symmetric label-noise injection."""

import numpy as np
import pandas as pd
import pytest

from trusttab.noise import inject_symmetric_label_noise


def test_zero_noise_preserves_labels():
    y = np.array([0, 1, 0, 1, 1, 0])

    result = inject_symmetric_label_noise(y, 0.0, seed=42)

    np.testing.assert_array_equal(result, y)


def test_full_noise_changes_every_label():
    y = np.array([0, 1, 0, 1, 1, 0])

    result = inject_symmetric_label_noise(y, 1.0, seed=42)

    assert np.all(result != y)


def test_noise_count_is_random_around_the_expected_value():
    y = np.tile([0, 1], 10_000)

    result = inject_symmetric_label_noise(y, 0.2, seed=42)

    flipped = np.sum(result != y)
    assert abs(flipped / len(y) - 0.2) < 0.02


def test_noise_is_reproducible():
    y = np.array([0, 1] * 50)

    first = inject_symmetric_label_noise(y, 0.3, seed=42)
    second = inject_symmetric_label_noise(y, 0.3, seed=42)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_can_produce_different_results():
    y = np.array([0, 1] * 50)

    first = inject_symmetric_label_noise(y, 0.3, seed=42)
    second = inject_symmetric_label_noise(y, 0.3, seed=99)

    assert not np.array_equal(first, second)


def test_original_labels_are_not_modified():
    y = np.array([0, 1, 0, 1])

    original = y.copy()

    inject_symmetric_label_noise(y, 0.5, seed=42)

    np.testing.assert_array_equal(y, original)


def test_multiclass_noise_uses_different_class():
    y = np.array([0, 1, 2] * 20)

    result = inject_symmetric_label_noise(y, 1.0, seed=42)

    assert np.all(result != y)
    assert set(result).issubset(set(y))


def test_pandas_series_preserves_index_and_name():
    y = pd.Series(
        [0, 1, 0, 1],
        index=["a", "b", "c", "d"],
        name="target",
    )

    result = inject_symmetric_label_noise(y, 0.5, seed=42)

    assert isinstance(result, pd.Series)
    assert result.index.equals(y.index)
    assert result.name == y.name


@pytest.mark.parametrize("noise_rate", [-0.1, 1.1])
def test_invalid_noise_rate_is_rejected(noise_rate):
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        inject_symmetric_label_noise(y, noise_rate, seed=42)


def test_single_class_rejects_positive_noise():
    y = np.array([1, 1, 1, 1])

    with pytest.raises(ValueError):
        inject_symmetric_label_noise(y, 0.5, seed=42)


def test_empty_labels_are_rejected():
    with pytest.raises(ValueError):
        inject_symmetric_label_noise(np.array([]), 0.1, seed=42)


def test_missing_labels_are_rejected():
    y = np.array([0.0, 1.0, np.nan])

    with pytest.raises(ValueError):
        inject_symmetric_label_noise(y, 0.1, seed=42)


def test_multidimensional_labels_are_rejected():
    y = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="one-dimensional"):
        inject_symmetric_label_noise(y, 0.1, seed=42)


@pytest.mark.parametrize("noise_rate", [np.nan, np.inf, -np.inf])
def test_non_finite_noise_rates_are_rejected(noise_rate):
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        inject_symmetric_label_noise(y, noise_rate, seed=42)
