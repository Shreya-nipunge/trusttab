"""Tests for experiments evaluated under feature distribution shift."""

import numpy as np
import pandas as pd
import pytest

from trusttab.experiments import feature_shift_runner
from trusttab.experiments import run as experiment_cli


@pytest.fixture
def fake_dataset():
    features = pd.DataFrame(
        {
            "age": np.linspace(18, 70, 100),
            "job": ["worker", "manager"] * 50,
        }
    )
    labels = pd.Series([0, 1] * 50, name="target")
    return features, labels


def fake_evaluator(y_true, y_pred, y_prob):
    return {
        "accuracy": 0.7,
        "balanced_accuracy": 0.6,
        "macro_f1": 0.65,
        "roc_auc": 0.75,
        "brier_score": 0.2,
        "ece": 0.08,
        "maximum_calibration_error": 0.1,
        "adaptive_ece": 0.09,
        "log_loss": 0.6,
    }


def fake_model_factory(model_name, features, random_state, model_params=None):
    class FakeModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y):
            self.n_fit = len(X)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.7, 0.3], (len(X), 1))

    return FakeModel()


def test_runner_evaluates_clean_and_shifted_for_models_and_seeds(
    monkeypatch, fake_dataset
):
    fitted_maxima = []

    def tracking_factory(model_name, features, random_state, model_params=None):
        model = fake_model_factory(model_name, features, random_state, model_params)
        original_fit = model.fit

        def fit_and_track(X, y):
            fitted_maxima.append(X["age"].max())
            return original_fit(X, y)

        model.fit = fit_and_track
        return model

    monkeypatch.setitem(
        feature_shift_runner.DATASET_LOADERS,
        "german_credit",
        lambda: fake_dataset,
    )
    monkeypatch.setattr(feature_shift_runner, "build_model", tracking_factory)
    results = feature_shift_runner.run_feature_shift_test(
        feature_shifts=[
            {"type": "mean_shift", "features": ["age"], "magnitude": 4.0},
            {"type": "category_replacement", "features": ["job"], "magnitude": 0.1},
        ],
        seeds=[1, 2],
        model_names=["logistic_regression", "random_forest"],
    )
    assert len(results) == 12
    assert set(results["scenario"]) == {"clean", "shifted"}
    assert results.groupby(["seed", "model"])["scenario"].count().eq(3).all()
    assert {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "brier_score",
        "ece",
        "maximum_calibration_error",
        "adaptive_ece",
        "log_loss",
    }.issubset(results.columns)
    clean = results[results.scenario == "clean"]
    assert len(clean) == 4
    assert (clean["shift_type"] == "none").all()
    assert (clean["shift_magnitude"] == 0).all()
    assert all(value <= fake_dataset[0]["age"].max() for value in fitted_maxima)


def test_runner_writes_csv_and_records_shift_metadata(
    monkeypatch, fake_dataset, tmp_path
):
    monkeypatch.setitem(
        feature_shift_runner.DATASET_LOADERS,
        "adult",
        lambda: fake_dataset,
    )
    monkeypatch.setattr(feature_shift_runner, "build_model", fake_model_factory)
    path = tmp_path / "output" / "feature-shifts.csv"
    results = feature_shift_runner.run_feature_shift_test(
        feature_shifts=[
            {"type": "gaussian_noise", "features": ["age"], "magnitude": 0.2}
        ],
        seeds=[5],
        model_names=["logistic_regression"],
        dataset_name="adult",
        metric_evaluator=fake_evaluator,
        output_path=path,
    )
    saved = pd.read_csv(path)
    assert len(saved) == len(results) == 2
    shifted = saved[saved.scenario == "shifted"].iloc[0]
    assert shifted["shift_type"] == "gaussian_noise"
    assert shifted["affected_feature"] == "age"
    assert shifted["shift_magnitude"] == 0.2
    assert shifted["seed"] == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": []},
        {"seeds": [True]},
        {"seeds": [-1]},
        {"seeds": [1, 1]},
        {"model_names": []},
        {"model_names": ["lr", "lr"]},
        {"dataset_name": "unknown"},
    ],
)
def test_runner_rejects_invalid_configuration(kwargs):
    base = {
        "feature_shifts": [
            {"type": "mean_shift", "features": ["age"], "magnitude": 1.0}
        ]
    }
    with pytest.raises((TypeError, ValueError)):
        feature_shift_runner.run_feature_shift_test(**(base | kwargs))


def test_runner_rejects_unknown_feature(monkeypatch, fake_dataset):
    monkeypatch.setitem(
        feature_shift_runner.DATASET_LOADERS,
        "german_credit",
        lambda: fake_dataset,
    )
    with pytest.raises(ValueError, match="Unknown feature"):
        feature_shift_runner.run_feature_shift_test(
            feature_shifts=[
                {"type": "mean_shift", "features": ["unknown"], "magnitude": 1.0}
            ],
            seeds=[1],
            model_names=["logistic_regression"],
        )


def test_cli_dispatches_feature_shift_configuration(monkeypatch, tmp_path):
    config = {
        "dataset": {"name": "german_credit"},
        "experiments": {"seeds": [1], "test_size": 0.2},
        "models": ["logistic_regression"],
        "output": {"directory": str(tmp_path), "filename": "shift.csv"},
        "feature_shifts": [
            {"type": "mean_shift", "features": ["age"], "magnitude": 1.0}
        ],
    }
    calls = []
    monkeypatch.setattr(experiment_cli, "load_experiment_config", lambda _: config)
    monkeypatch.setattr(
        experiment_cli,
        "run_feature_shift_test",
        lambda **kwargs: calls.append(kwargs) or pd.DataFrame({"accuracy": [1.0]}),
    )
    monkeypatch.setattr("sys.argv", ["trusttab-run", "--config", "unused.yaml"])
    experiment_cli.main()
    assert calls[0]["feature_shifts"] == config["feature_shifts"]
    assert calls[0]["output_path"] == tmp_path / "shift.csv"
