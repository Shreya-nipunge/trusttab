"""Leakage-safe preprocessing utilities for tabular datasets."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    features: pd.DataFrame, *, scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build an unfitted preprocessing transformer for tabular features.

    Numeric columns:
        Median imputation, optionally followed by standard scaling.

    Categorical columns:
        Most-frequent imputation followed by one-hot encoding.

    The returned transformer must be fitted only on training data.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")

    if features.empty:
        raise ValueError("Features must not be empty.")

    if features.columns.has_duplicates:
        raise ValueError("Feature names must be unique.")

    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()

    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns found.")

    transformers = []

    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)

        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
