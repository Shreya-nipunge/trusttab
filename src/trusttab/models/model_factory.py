"""Model factory for TrustTab classical machine learning models."""

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from trusttab.data.preprocessing import build_preprocessor


SUPPORTED_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)


def build_model(
    model_name: str,
    features: pd.DataFrame,
    random_state: int = 42,
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Build a complete preprocessing and classification pipeline.

    Parameters
    ----------
    model_name:
        One of the supported model names.
    features:
        Training feature DataFrame, used to configure preprocessing.
        The returned pipeline remains unfitted.
    random_state:
        Random seed for reproducibility.
    model_params:
        Optional dictionary of model-specific hyperparameters.

    Returns
    -------
    An unfitted sklearn Pipeline containing preprocessing and classifier.
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {SUPPORTED_MODELS}"
        )

    params = model_params.copy() if model_params is not None else {}

    if model_name == "logistic_regression":
        defaults = {
            "max_iter": 1000,
            "random_state": random_state,
        }
        defaults.update(params)
        classifier = LogisticRegression(**defaults)

    elif model_name == "random_forest":
        defaults = {
            "n_estimators": 300,
            "random_state": random_state,
            "n_jobs": -1,
        }
        defaults.update(params)
        classifier = RandomForestClassifier(**defaults)

    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is optional. Install it with "
                "'pip install trusttab[xgboost]' or "
                "'pip install xgboost'."
            ) from exc

        defaults = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }
        defaults.update(params)
        classifier = XGBClassifier(**defaults)

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", classifier),
        ]
    )