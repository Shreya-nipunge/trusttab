
"""Tests for the Adult Census Income dataset loader."""

import pandas as pd
import pytest

from trusttab.data.adult import (
    TARGET_MAPPING,
    validate_adult_data,
)


def make_valid_adult_data():
    features = pd.DataFrame(
        {
            f"feature_{i}": [1, 2, 3]
            for i in range(14)
        }
    )

    target = pd.Series(
        ["<=50K", ">50K", "<=50K"],
        name="class",
    )

    return features, target


def test_validate_accepts_valid_adult_data():
    features = pd.DataFrame(
        {
            f"feature_{i}": [1] * 48842
            for i in range(14)
        }
    )

    target = pd.Series(
        ["<=50K"] * 37155 + [">50K"] * 11687
    )

    validate_adult_data(features, target)


def test_validate_accepts_missing_feature_values():
    features = pd.DataFrame(
        {
            f"feature_{i}": [1.0, float("nan")]
            for i in range(14)
        }
    )

    features = pd.concat(
        [features] * 24421,
        ignore_index=True,
    )

    target = pd.Series(
        ["<=50K"] * 48841 + [">50K"]
    )

    validate_adult_data(features, target)


def test_validate_rejects_non_dataframe_features():
    with pytest.raises(TypeError, match="DataFrame"):
        validate_adult_data(
            [[1, 2]],
            pd.Series(["<=50K"]),
        )


def test_validate_rejects_wrong_feature_count():
    features = pd.DataFrame(
        {"feature": [1] * 48842}
    )

    target = pd.Series(["<=50K"] * 48842)

    with pytest.raises(ValueError, match="features"):
        validate_adult_data(features, target)


def test_validate_rejects_missing_target():
    features = pd.DataFrame(
        {
            f"feature_{i}": [1] * 48842
            for i in range(14)
        }
    )

    target = pd.Series(["<=50K"] * 48841 + [None])

    with pytest.raises(ValueError, match="Target contains missing"):
        validate_adult_data(features, target)


def test_target_mapping_is_binary():
    assert TARGET_MAPPING == {
        "<=50k": 0,
        ">50k": 1,
    }
