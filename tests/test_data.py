
import pandas as pd
import pytest

from trusttab.data.german_credit import (
    EXPECTED_FEATURES,
    EXPECTED_ROWS,
    load_german_credit,
    validate_german_credit_data,
)


def make_valid_dataset():
    """Create a small, predictable test fixture with the expected shape."""

    features = pd.DataFrame(
        {
            f"feature_{i}": [float(j) for j in range(EXPECTED_ROWS)]
            for i in range(EXPECTED_FEATURES)
        }
    )

    target = pd.Series(
        ["good", "bad"] * (EXPECTED_ROWS // 2),
        name="class",
    )

    return features, target


def test_validate_german_credit_data_accepts_valid_data():
    features, target = make_valid_dataset()

    validate_german_credit_data(features, target)


def test_validate_rejects_wrong_number_of_rows():
    features, target = make_valid_dataset()

    with pytest.raises(ValueError, match="Expected 1000 rows"):
        validate_german_credit_data(features.iloc[:-1], target.iloc[:-1])


def test_validate_rejects_wrong_number_of_features():
    features, target = make_valid_dataset()

    with pytest.raises(ValueError, match="Expected 20 features"):
        validate_german_credit_data(features.drop(columns="feature_0"), target)


def test_validate_rejects_missing_feature_values():
    features, target = make_valid_dataset()
    features.iloc[0, 0] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_german_credit_data(features, target)


def test_validate_rejects_unexpected_target_labels():
    features, target = make_valid_dataset()
    target.iloc[0] = "unknown"

    with pytest.raises(ValueError, match="Unexpected target labels"):
        validate_german_credit_data(features, target)


def test_load_german_credit_maps_target_labels(monkeypatch):
    features, raw_target = make_valid_dataset()

    class MockDataset:
        data = features
        target = raw_target

    def mock_fetch_openml(**kwargs):
        assert kwargs["data_id"] == 31
        assert kwargs["as_frame"] is True
        return MockDataset()

    monkeypatch.setattr(
        "trusttab.data.german_credit.fetch_openml",
        mock_fetch_openml,
    )

    X, y = load_german_credit()

    assert X.shape == (EXPECTED_ROWS, EXPECTED_FEATURES)
    assert y.shape == (EXPECTED_ROWS,)
    assert set(y.unique()) == {0, 1}
    assert y.iloc[0] == 0
    assert y.iloc[1] == 1
    assert y.name == "target"
