"""Tests for TrustTab model integration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from trusttab.models import SUPPORTED_MODELS, build_model


@pytest.fixture
def sample_data():
    features = pd.DataFrame(
        {
            "age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "income": [
                20000,
                25000,
                30000,
                35000,
                40000,
                45000,
                50000,
                55000,
                60000,
                65000,
            ],
            "job": [
                "student",
                "student",
                "worker",
                "worker",
                "manager",
                "manager",
                "worker",
                "manager",
                "worker",
                "manager",
            ],
        }
    )

    target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    return features, target


@pytest.mark.parametrize(
    "model_name",
    ["logistic_regression", "random_forest"],
)
def test_classical_model_creation(model_name, sample_data):
    features, _ = sample_data

    model = build_model(
        model_name,
        features,
        random_state=42,
    )

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == [
        "preprocessor",
        "classifier",
    ]


@pytest.mark.parametrize(
    "model_name",
    ["logistic_regression", "random_forest"],
)
def test_classical_model_can_fit_and_predict(model_name, sample_data):
    features, target = sample_data

    model = build_model(
        model_name,
        features,
        random_state=42,
    )

    model.fit(features, target)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    assert len(predictions) == len(target)
    assert probabilities.shape == (len(target), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_pipeline_starts_unfitted(sample_data):
    features, _ = sample_data

    model = build_model("logistic_regression", features)

    assert not hasattr(model, "classes_")


def test_model_uses_requested_random_state(sample_data):
    features, _ = sample_data

    model = build_model(
        "random_forest",
        features,
        random_state=123,
    )

    assert model.named_steps["classifier"].random_state == 123


def test_custom_model_parameters_are_supported(sample_data):
    features, _ = sample_data

    model = build_model(
        "random_forest",
        features,
        model_params={"n_estimators": 50, "max_depth": 5},
    )

    classifier = model.named_steps["classifier"]

    assert classifier.n_estimators == 50
    assert classifier.max_depth == 5


def test_unknown_model_is_rejected(sample_data):
    features, _ = sample_data

    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unsupported_model", features)


def test_supported_models_are_declared():
    assert "logistic_regression" in SUPPORTED_MODELS
    assert "random_forest" in SUPPORTED_MODELS
    assert "xgboost" in SUPPORTED_MODELS


def test_tabpfn_is_a_supported_model():
    assert "tabpfn" in SUPPORTED_MODELS


def test_tabpfn_model_creation_does_not_require_tabpfn(sample_data):
    features, _ = sample_data

    model = build_model(
        "tabpfn",
        features,
        model_params={"device": "cpu"},
    )

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == [
        "preprocessor",
        "classifier",
    ]
    assert model.named_steps["classifier"].model_params == {
        "device": "cpu",
        "random_state": 42,
    }
    assert not hasattr(model, "classes_")
    configured_transformers = dict(
        (name, transformer)
        for name, transformer, _columns in model.named_steps[
            "preprocessor"
        ].transformers
    )
    numeric_steps = configured_transformers["numeric"].named_steps
    assert set(numeric_steps) == {"imputer"}
    categorical_steps = configured_transformers["categorical"].named_steps
    assert set(categorical_steps) == {"imputer", "onehot"}


def test_managed_random_state_cannot_be_overridden_by_model_parameters(sample_data):
    features, _ = sample_data

    model = build_model(
        "random_forest",
        features,
        random_state=123,
        model_params={"random_state": 999, "n_estimators": 10},
    )

    assert model.named_steps["classifier"].random_state == 123


def test_tabpfn_missing_dependency_has_install_instructions(sample_data, monkeypatch):
    import builtins

    features, target = sample_data
    original_import = builtins.__import__

    def missing_tabpfn(name, *args, **kwargs):
        if name == "tabpfn":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_tabpfn)

    model = build_model("tabpfn", features)

    with pytest.raises(ImportError, match=r"trusttab\[tabpfn\]"):
        model.fit(features, target)


def test_xgboost_integration_when_installed(sample_data):
    pytest.importorskip("xgboost")

    features, target = sample_data

    model = build_model(
        "xgboost",
        features,
        random_state=42,
        model_params={"n_estimators": 10},
    )

    model.fit(features, target)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (len(target), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
