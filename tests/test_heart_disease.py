"""Offline tests for the UCI processed Cleveland dataset loader."""

from collections import Counter
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from trusttab.data.heart_disease import (
    CACHE_FILENAME,
    CATEGORICAL_FEATURES,
    EXPECTED_FEATURES,
    EXPECTED_ROWS,
    EXPECTED_NUM_COUNTS,
    FEATURE_COLUMNS,
    MISSING_VALUE_FEATURES,
    TARGET_MAPPING,
    _read_processed_cleveland_archive,
    load_heart_disease,
    validate_heart_disease_data,
)
from trusttab.data.preprocessing import build_preprocessor


@pytest.fixture
def processed_cleveland():
    rows = EXPECTED_ROWS
    features = pd.DataFrame(
        {
            "age": np.linspace(29, 77, rows),
            "sex": np.arange(rows) % 2,
            "cp": np.arange(rows) % 4 + 1,
            "trestbps": np.full(rows, 130.0),
            "chol": np.full(rows, 240.0),
            "fbs": np.arange(rows) % 2,
            "restecg": np.arange(rows) % 3,
            "thalach": np.full(rows, 150.0),
            "exang": np.arange(rows) % 2,
            "oldpeak": np.zeros(rows),
            "slope": np.arange(rows) % 3 + 1,
            "ca": np.arange(rows) % 4,
            "thal": np.resize([3, 6, 7], rows),
        },
        columns=FEATURE_COLUMNS,
    )
    # The processed Cleveland source has missing values only in ca and thal.
    features.loc[[10, 20, 30, 40], "ca"] = np.nan
    features.loc[[50, 60], "thal"] = np.nan
    target = pd.Series(
        np.repeat([0, 1, 2, 3, 4], [164, 55, 36, 35, 13]),
        name="num",
    )
    return features, target


def _archive_with_cleveland_and_other_database(features, target):
    frame = features.copy()
    frame["num"] = target
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "heart-disease/processed.cleveland.data",
            frame.to_csv(index=False, header=False, na_rep="?"),
        )
        archive.writestr(
            "heart-disease/processed.hungarian.data",
            "999,999,999,999,999,999,999,999,999,999,999,999,999,4\n",
        )
    return output.getvalue()


def test_archive_reader_selects_only_processed_cleveland(processed_cleveland):
    features, target = processed_cleveland
    archive = _archive_with_cleveland_and_other_database(features, target)

    loaded = _read_processed_cleveland_archive(archive)

    assert loaded.shape == (EXPECTED_ROWS, EXPECTED_FEATURES + 1)
    assert tuple(loaded.columns[:-1]) == FEATURE_COLUMNS
    assert loaded["num"].value_counts().sort_index().to_dict() == {
        0: 164,
        1: 55,
        2: 36,
        3: 35,
        4: 13,
    }
    assert loaded.loc[0, "age"] != 999


def test_validate_accepts_source_missing_values(processed_cleveland):
    features, target = processed_cleveland

    validate_heart_disease_data(features, target)


def test_validate_rejects_unexpected_missing_feature(processed_cleveland):
    features, target = processed_cleveland
    features.loc[0, "age"] = np.nan

    with pytest.raises(ValueError, match="Unexpected missing feature values"):
        validate_heart_disease_data(features, target)


def test_validate_rejects_invalid_original_num(processed_cleveland):
    features, target = processed_cleveland
    target.iloc[0] = 5

    with pytest.raises(ValueError, match="Unexpected original num"):
        validate_heart_disease_data(features, target)


def test_validate_checks_original_class_distribution(processed_cleveland):
    features, target = processed_cleveland
    assert target.value_counts().sort_index().to_dict() == EXPECTED_NUM_COUNTS

    target.iloc[0] = 1
    with pytest.raises(ValueError, match="class distribution"):
        validate_heart_disease_data(features, target)


def test_validate_rejects_wrong_shape(processed_cleveland):
    features, target = processed_cleveland

    with pytest.raises(ValueError, match="Expected 303 rows"):
        validate_heart_disease_data(features.iloc[:-1], target.iloc[:-1])

    with pytest.raises(ValueError, match="Expected 13 features"):
        validate_heart_disease_data(features.drop(columns="age"), target)


def test_loader_maps_target_preserves_num_and_missingness(
    monkeypatch, tmp_path, processed_cleveland
):
    features, original_num = processed_cleveland
    archive = _archive_with_cleveland_and_other_database(features, original_num)
    requested = []

    def mock_urlopen(url, timeout):
        requested.append((url, timeout))
        return BytesIO(archive)

    monkeypatch.setattr(
        "trusttab.data.heart_disease.urlopen",
        mock_urlopen,
    )

    X, y = load_heart_disease(data_home=tmp_path)

    assert requested == [
        ("https://archive.ics.uci.edu/static/public/45/heart+disease.zip", 30)
    ]
    assert X.shape == (EXPECTED_ROWS, EXPECTED_FEATURES)
    assert tuple(X.columns) == FEATURE_COLUMNS
    assert y.shape == (EXPECTED_ROWS,)
    assert y.name == "target"
    assert Counter(y) == {0: 164, 1: 139}
    pd.testing.assert_series_equal(y.attrs["original_num"], original_num)
    assert set(y.attrs["original_num"].unique()) == set(TARGET_MAPPING)
    assert X[list(MISSING_VALUE_FEATURES)].isna().sum().to_dict() == {
        "ca": 4,
        "thal": 2,
    }
    assert all(
        pd.api.types.is_object_dtype(X[column].dtype) for column in CATEGORICAL_FEATURES
    )
    assert pd.api.types.is_numeric_dtype(X["age"])
    transformed = build_preprocessor(X).fit_transform(X)
    assert np.isfinite(transformed).all()

    cache_path = tmp_path / CACHE_FILENAME
    assert cache_path.exists()
    cached = pd.read_csv(cache_path)
    assert "num" in cached
    assert "target" not in cached
    assert len(cached) == EXPECTED_ROWS


def test_loader_uses_cache_without_network(monkeypatch, tmp_path, processed_cleveland):
    features, target = processed_cleveland
    archive = _archive_with_cleveland_and_other_database(features, target)
    monkeypatch.setattr(
        "trusttab.data.heart_disease.urlopen",
        lambda url, timeout: BytesIO(archive),
    )
    first_X, first_y = load_heart_disease(data_home=tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("cached loading should not access the network")

    monkeypatch.setattr("trusttab.data.heart_disease.urlopen", no_network)
    second_X, second_y = load_heart_disease(data_home=tmp_path)

    pd.testing.assert_frame_equal(first_X, second_X)
    pd.testing.assert_series_equal(first_y, second_y)


def test_archive_must_contain_processed_cleveland_only():
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("processed.hungarian.data", "not Cleveland\n")

    with pytest.raises(ValueError, match="processed.cleveland.data"):
        _read_processed_cleveland_archive(output.getvalue())
