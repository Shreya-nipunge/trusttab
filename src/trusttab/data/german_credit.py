
"""German Credit dataset loading and validation."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


OPENML_DATA_ID = 31
EXPECTED_ROWS = 1000
EXPECTED_FEATURES = 20

TARGET_MAPPING = {
    "good": 0,
    "bad": 1,
}


def validate_german_credit_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Validate the structure and contents of the German Credit dataset."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")

    if not isinstance(target, pd.Series):
        raise TypeError("Target must be a pandas Series.")

    if features.empty:
        raise ValueError("The feature dataset is empty.")

    if len(features) != len(target):
        raise ValueError("Features and target have different row counts.")

    if features.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, got {features.shape[0]}."
        )

    if features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {features.shape[1]}."
        )

    if features.columns.has_duplicates:
        raise ValueError("Feature names must be unique.")

    if features.isnull().any().any():
        raise ValueError("Features contain missing values.")

    if target.isnull().any():
        raise ValueError("Target contains missing values.")

    observed_labels = set(target.astype(str).str.strip().str.lower().unique())

    if observed_labels != set(TARGET_MAPPING):
        raise ValueError(
            "Unexpected target labels. "
            f"Expected {set(TARGET_MAPPING)}, got {observed_labels}."
        )


def load_german_credit(
    data_home: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download and load the German Credit dataset from OpenML.

    Returns
    -------
    X:
        DataFrame containing 20 input features.
    y:
        Integer target Series: good=0, bad=1.
    """

    dataset = fetch_openml(
        data_id=OPENML_DATA_ID,
        as_frame=True,
        data_home=data_home,
    )

    features = dataset.data.copy()
    raw_target = dataset.target.copy()

    validate_german_credit_data(features, raw_target)

    target = (
        raw_target.astype(str)
        .str.strip()
        .str.lower()
        .map(TARGET_MAPPING)
        .astype("int64")
    )

    target.name = "target"

    return features, target
