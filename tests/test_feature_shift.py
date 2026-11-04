"""Tests for numerical and categorical feature-shift utilities."""

import numpy as np
import pandas as pd
import pytest

from trusttab.noise import (
    apply_categorical_feature_shift,
    apply_feature_shifts,
    apply_numerical_feature_shift,
)


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "amount": np.r_[np.linspace(0.0, 3.0, 99), np.nan],
            "color": ["red"] * 90 + ["blue"] * 10,
        },
        index=range(100),
    )


def test_gaussian_noise_is_reproducible_and_preserves_missing_values(sample_frame):
    first = apply_numerical_feature_shift(
        sample_frame, "amount", "gaussian_noise", 0.2, seed=12
    )
    second = apply_numerical_feature_shift(
        sample_frame, "amount", "gaussian_noise", 0.2, seed=12
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["amount"].isna().sum() == 1
    assert first.index.equals(sample_frame.index)


def test_mean_shift_and_variance_scaling():
    data = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0]})
    shifted = apply_numerical_feature_shift(data, "x", "mean_shift", 0.5)
    np.testing.assert_allclose(shifted["x"], data["x"] + 0.5)
    scaled = apply_numerical_feature_shift(data, "x", "variance_scaling", 4.0)
    assert np.var(scaled["x"]) == pytest.approx(np.var(data["x"]) * 4.0)


def test_feature_shifts_do_not_modify_original_and_preserve_structure(sample_frame):
    original = sample_frame.copy(deep=True)
    shifted = apply_feature_shifts(
        sample_frame,
        [
            {"type": "mean_shift", "features": ["amount"], "magnitude": 1.0},
            {
                "type": "category_replacement",
                "features": ["color"],
                "magnitude": 0.1,
                "replacement": "blue",
            },
        ],
        seed=8,
    )
    pd.testing.assert_frame_equal(sample_frame, original)
    assert shifted.shape == sample_frame.shape
    assert shifted.columns.equals(sample_frame.columns)
    assert shifted.index.equals(sample_frame.index)
    assert shifted["amount"].isna().sum() == 1


def test_probability_redistribution_moves_toward_target_distribution(sample_frame):
    shifted = apply_categorical_feature_shift(
        sample_frame,
        "color",
        "category_probability_redistribution",
        1.0,
        seed=42,
        probabilities={"red": 0.0, "blue": 1.0},
    )
    assert (shifted["color"] == "blue").all()
    assert (sample_frame["color"] == "red").sum() == 90


def test_probability_redistribution_without_target_uses_uniform_categories(sample_frame):
    spec = {
        "type": "category_probability_redistribution",
        "features": ["color"],
        "magnitude": 1.0,
    }
    shifted = apply_feature_shifts(sample_frame, [spec], seed=7)
    assert shifted["color"].value_counts(normalize=True)["blue"] == pytest.approx(
        0.5, abs=0.08
    )
    repeated = apply_feature_shifts(sample_frame, [spec], seed=7)
    pd.testing.assert_frame_equal(shifted, repeated)


def test_controlled_category_replacement_changes_expected_count(sample_frame):
    shifted = apply_categorical_feature_shift(
        sample_frame, "color", "category_replacement", 0.2, seed=9
    )
    assert (shifted["color"] != sample_frame["color"]).sum() == 20


def test_categorical_shift_preserves_missing_values_and_category_dtype():
    frame = pd.DataFrame(
        {"kind": pd.Categorical(["a", "b", None, "a"], categories=["a", "b"])}
    )
    shifted = apply_categorical_feature_shift(
        frame, "kind", "category_replacement", 0.5, seed=4
    )
    assert shifted["kind"].isna().sum() == 1
    assert isinstance(shifted["kind"].dtype, pd.CategoricalDtype)
    assert list(shifted["kind"].cat.categories) == ["a", "b"]


def test_category_replacement_validates_available_category(sample_frame):
    with pytest.raises(ValueError, match="not available"):
        apply_feature_shifts(
            sample_frame,
            [
                {
                    "type": "category_replacement",
                    "features": ["color"],
                    "magnitude": 0.2,
                    "replacement": "green",
                }
            ],
            seed=2,
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "unknown", "features": ["amount"], "magnitude": 1.0},
        {"type": "mean_shift", "features": ["missing"], "magnitude": 1.0},
        {"type": "gaussian_noise", "features": ["amount"], "magnitude": -1.0},
        {"type": "variance_scaling", "features": ["amount"], "magnitude": -1.0},
        {"type": "category_replacement", "features": ["color"], "magnitude": 1.1},
        {"type": "mean_shift", "features": ["amount"], "magnitude": np.nan},
    ],
)
def test_general_feature_shift_rejects_invalid_specs(sample_frame, spec):
    with pytest.raises((TypeError, ValueError)):
        apply_feature_shifts(sample_frame, [spec], seed=3)


def test_numeric_shift_rejects_categorical_feature(sample_frame):
    with pytest.raises(TypeError, match="numeric"):
        apply_numerical_feature_shift(sample_frame, "color", "mean_shift", 1.0)


def test_categorical_shift_probability_validation(sample_frame):
    with pytest.raises(ValueError, match="Unknown category"):
        apply_categorical_feature_shift(
            sample_frame,
            "color",
            "category_probability_redistribution",
            1.0,
            probabilities={"green": 1.0},
        )
    with pytest.raises(ValueError, match="sum to > 0"):
        apply_categorical_feature_shift(
            sample_frame,
            "color",
            "category_probability_redistribution",
            1.0,
            probabilities={"red": 0.0, "blue": 0.0},
        )
