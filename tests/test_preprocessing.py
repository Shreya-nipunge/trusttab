"""Tests for leakage-safe tabular preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from trusttab.data.preprocessing import build_preprocessor


@pytest.fixture
def sample_features():
    return pd.DataFrame(
        {
            "age": [25, 35, np.nan, 45],
            "income": [30000, 50000, 40000, np.nan],
            "job": ["student", "manager", "worker", "manager"],
            "housing": ["rent", "own", "rent", None],
        }
    )


def test_build_preprocessor_returns_unfitted_transformer(
    sample_features,
):
    preprocessor = build_preprocessor(sample_features)

    assert preprocessor is not None

    with pytest.raises(Exception):
        check_is_fitted(preprocessor)


def test_preprocessor_handles_missing_values(sample_features):
    preprocessor = build_preprocessor(sample_features)

    transformed = preprocessor.fit_transform(sample_features)

    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_categories(sample_features):
    preprocessor = build_preprocessor(sample_features)

    preprocessor.fit(sample_features)

    new_features = pd.DataFrame(
        {
            "age": [30],
            "income": [45000],
            "job": ["astronaut"],
            "housing": ["rent"],
        }
    )

    transformed = preprocessor.transform(new_features)

    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_preprocessor_rejects_non_dataframe():
    with pytest.raises(TypeError):
        build_preprocessor([[1, 2], [3, 4]])


def test_preprocessor_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        build_preprocessor(pd.DataFrame())


def test_preprocessor_rejects_duplicate_columns():
    features = pd.DataFrame(
        [[1, 2]],
        columns=["age", "age"],
    )

    with pytest.raises(ValueError):
        build_preprocessor(features)