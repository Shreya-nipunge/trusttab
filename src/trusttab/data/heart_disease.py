"""UCI Heart Disease processed Cleveland dataset loader.

Source: UCI Machine Learning Repository, dataset 45, processed Cleveland
subset only. Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano,
R. (1989). *Heart Disease*. https://doi.org/10.24432/C52P4X. The dataset is
licensed CC BY 4.0; see https://creativecommons.org/licenses/by/4.0/.

The source ``num`` target is retained as ``y.attrs['original_num']`` before
binary mapping (0 -> 0, 1-4 -> 1). Feature missing values are preserved.
"""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

UCI_DATA_ID = 45
DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
EXPECTED_ROWS = 303
EXPECTED_FEATURES = 13
CACHE_FILENAME = "heart_disease_uci45_cleveland.csv"

FEATURE_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)
TARGET_COLUMN = "num"
CATEGORICAL_FEATURES = (
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)
MISSING_VALUE_FEATURES = ("ca", "thal")
TARGET_MAPPING = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
EXPECTED_NUM_COUNTS = {0: 164, 1: 55, 2: 36, 3: 35, 4: 13}


def _read_processed_cleveland_archive(archive: bytes) -> pd.DataFrame:
    """Read only UCI's processed Cleveland member from its dataset archive."""

    with ZipFile(BytesIO(archive)) as zipped:
        matches = [
            name
            for name in zipped.namelist()
            if Path(name).name.lower() == "processed.cleveland.data"
        ]
        if len(matches) != 1:
            raise ValueError(
                "UCI dataset archive must contain exactly one "
                "processed.cleveland.data file."
            )
        with zipped.open(matches[0]) as source_file:
            frame = pd.read_csv(
                source_file,
                header=None,
                names=[*FEATURE_COLUMNS, TARGET_COLUMN],
                na_values=["?"],
                skipinitialspace=True,
            )
    return frame


def _download_processed_cleveland() -> pd.DataFrame:
    """Download UCI dataset 45 and parse only the processed Cleveland file."""

    with urlopen(DATASET_URL, timeout=30) as response:
        return _read_processed_cleveland_archive(response.read())


def validate_heart_disease_data(
    features: pd.DataFrame,
    original_num: pd.Series,
) -> None:
    """Validate processed Cleveland rows, schema, missingness, and ``num``.

    The UCI source has missing values in ``ca`` and ``thal``. They are
    intentionally preserved for downstream train-only imputation; rows are
    never dropped and no imputation occurs in this loader.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")
    if not isinstance(original_num, pd.Series):
        raise TypeError("Original num target must be a pandas Series.")
    if features.empty:
        raise ValueError("The feature dataset is empty.")
    if len(features) != len(original_num):
        raise ValueError("Features and target have different row counts.")
    if features.shape[0] != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {features.shape[0]}.")
    if features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, got {features.shape[1]}."
        )
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the processed Cleveland schema.")
    if features.columns.has_duplicates:
        raise ValueError("Feature names must be unique.")
    if original_num.isna().any():
        raise ValueError("Original num target contains missing values.")

    raw_values = pd.to_numeric(original_num, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw_values).all():
        raise ValueError("Original num target must contain finite numeric values.")
    if not np.equal(raw_values, np.floor(raw_values)).all():
        raise ValueError("Original num target values must be integers 0 through 4.")
    observed_targets = set(raw_values.astype(int))
    if not observed_targets.issubset(TARGET_MAPPING):
        raise ValueError(
            "Unexpected original num target values. Expected a subset of "
            f"{set(TARGET_MAPPING)}, got {observed_targets}."
        )
    observed_counts = pd.Series(raw_values.astype(int)).value_counts().sort_index()
    if observed_counts.to_dict() != EXPECTED_NUM_COUNTS:
        raise ValueError(
            "Unexpected processed Cleveland num class distribution. Expected "
            f"{EXPECTED_NUM_COUNTS}, got {observed_counts.to_dict()}."
        )

    allowed_missing = set(MISSING_VALUE_FEATURES)
    missing_columns = set(features.columns[features.isna().any()])
    if missing_columns.difference(allowed_missing):
        raise ValueError(
            "Unexpected missing feature values in columns: "
            f"{sorted(missing_columns.difference(allowed_missing))}."
        )

    for column in features.columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid_values = features[column].notna() & converted.isna()
        if invalid_values.any():
            raise ValueError(f"Feature '{column}' contains non-numeric values.")
        if not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"Feature '{column}' contains non-finite values.")


def _normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Set stable numeric and categorical dtypes without filling missing data."""

    normalized = features.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in FEATURE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        codes = normalized[column].astype("Int64").astype(object)
        normalized[column] = codes.where(codes.notna(), np.nan)
    return normalized


def load_heart_disease(
    data_home: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI dataset 45's processed Cleveland subset.

    Only ``processed.cleveland.data`` is read from the UCI archive; the
    Cleveland, Hungarian, Switzerland, and VA datasets are never combined.
    With ``data_home``, the original 13 feature columns and multiclass ``num``
    are cached as CSV and reused on later calls. ``?`` source markers become
    ``NaN``. The 303 rows are retained and feature missingness is left for the
    experiment pipeline to impute after splitting.

    The returned target is binary: original ``num`` 0 maps to 0 and values
    1, 2, 3, or 4 map to 1. The unmodified numeric target is also preserved in
    ``y.attrs['original_num']``. UCI citation and attribution: Janosi et al.
    (1989), *Heart Disease*, DOI: https://doi.org/10.24432/C52P4X, CC BY 4.0.

    Parameters
    ----------
    data_home:
        Optional cache directory. If omitted, the UCI archive is fetched for
        each call and no local dataset file is written.

    Returns
    -------
    X:
        DataFrame with the 13 processed Cleveland features in documented
        order. Categorical codes use object dtype with integer values; numeric
        measurements remain numeric. Missing values are preserved as NaN.
    y:
        Integer binary target Series named ``target`` with original ``num``
        available in its ``attrs`` metadata.
    """

    cache_path = Path(data_home) / CACHE_FILENAME if data_home is not None else None
    if cache_path is not None and cache_path.exists():
        raw_data = pd.read_csv(cache_path, na_values=["?"])
        missing_columns = set([*FEATURE_COLUMNS, TARGET_COLUMN]).difference(
            raw_data.columns
        )
        if missing_columns:
            raise ValueError(
                f"Cached Cleveland data is missing columns: {sorted(missing_columns)}."
            )
        features = raw_data.loc[:, list(FEATURE_COLUMNS)].copy()
        original_num = raw_data[TARGET_COLUMN].copy()
    else:
        raw_data = _download_processed_cleveland()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            raw_data.to_csv(cache_path, index=False)
        features = raw_data.loc[:, list(FEATURE_COLUMNS)].copy()
        original_num = raw_data[TARGET_COLUMN].copy()

    validate_heart_disease_data(features, original_num)
    normalized_features = _normalize_features(features)
    numeric_num = pd.to_numeric(original_num, errors="raise").astype("int64")
    target = numeric_num.map(TARGET_MAPPING).astype("int64")
    target.name = "target"
    target.attrs["original_num"] = numeric_num.copy()

    return normalized_features, target
