
"""Tests for the TrustTab experiment runner."""

import numpy as np
import pandas as pd
import pytest

import trusttab.experiments.runner as runner
def patch_german_credit_loader(monkeypatch, fake_dataset):
    monkeypatch.setitem(
        runner.DATASET_LOADERS,
        "german_credit",
        lambda: fake_dataset,
    )


@pytest.fixture
def fake_dataset():
    X = pd.DataFrame(
        {
            "age": np.arange(100),
            "job": ["worker", "manager"] * 50,
        }
    )
    y = pd.Series([0, 1] * 50, name="target")
    return X, y


def fake_model_factory(model_name, features, random_state, model_params=None):
    class FakeModel:
        classes_ = np.array([0, 1])

        def fit(self, X, y):
            self.fitted_rows = len(X)
            self.fitted_labels = np.asarray(y).copy()
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.7, 0.3], (len(X), 1))

    return FakeModel()


def fake_evaluator(y_true, y_pred, y_prob):
    return {
        "accuracy": 0.7,
        "balanced_accuracy": 0.6,
        "macro_f1": 0.65,
        "roc_auc": 0.75,
        "ece": 0.08,
        "brier_score": 0.2,
    }


def test_runner_creates_expected_number_of_results(
    monkeypatch, fake_dataset
):
    patch_german_credit_loader(monkeypatch, fake_dataset)
    monkeypatch.setattr(runner, "build_model", fake_model_factory)

    results = runner.run_stress_test(
        noise_levels=[0.0, 0.2],
        seeds=[1, 2],
        model_names=["logistic_regression"],
        metric_evaluator=fake_evaluator,
    )

    assert len(results) == 4


def test_runner_records_experiment_metadata(
    monkeypatch, fake_dataset
):
    patch_german_credit_loader(monkeypatch, fake_dataset)
    monkeypatch.setattr(runner, "build_model", fake_model_factory)

    results = runner.run_stress_test(
        noise_levels=[0.2],
        seeds=[42],
        model_names=["logistic_regression"],
        metric_evaluator=fake_evaluator,
    )

    row = results.iloc[0]

    assert row["dataset"] == "german_credit"
    assert row["seed"] == 42
    assert row["noise_rate"] == 0.2
    assert row["n_train"] == 80
    assert row["n_test"] == 20
    assert row["n_corrupted"] == 16


def test_runner_uses_supplied_metric_evaluator(
    monkeypatch, fake_dataset
):
    patch_german_credit_loader(monkeypatch, fake_dataset)
    monkeypatch.setattr(runner, "build_model", fake_model_factory)

    results = runner.run_stress_test(
        noise_levels=[0.0],
        seeds=[42],
        model_names=["logistic_regression"],
        metric_evaluator=fake_evaluator,
    )

    assert results.iloc[0]["ece"] == 0.08
    assert results.iloc[0]["brier_score"] == 0.2


def test_runner_saves_csv(
    monkeypatch, fake_dataset, tmp_path
):
    patch_german_credit_loader(monkeypatch, fake_dataset)
    monkeypatch.setattr(runner, "build_model", fake_model_factory)

    output_file = tmp_path / "results" / "test.csv"

    results = runner.run_stress_test(
        noise_levels=[0.0],
        seeds=[42],
        model_names=["logistic_regression"],
        metric_evaluator=fake_evaluator,
        output_path=output_file,
    )

    assert output_file.exists()

    saved = pd.read_csv(output_file)

    assert len(saved) == len(results)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0},
        {"test_size": 1},
        {"noise_levels": []},
        {"seeds": []},
        {"model_names": []},
        {"noise_levels": [-0.1]},
        {"noise_levels": [1.1]},
        {"seeds": [42, 42]},
        {"model_names": ["random_forest", "random_forest"]},
    ],
)
def test_runner_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        runner.run_stress_test(**kwargs)


def test_runner_uses_adult_dataset(monkeypatch, fake_dataset):
    adult_loader_called = []

    def fake_adult_loader():
        adult_loader_called.append(True)
        return fake_dataset

    monkeypatch.setitem(
        runner.DATASET_LOADERS,
        "adult",
        fake_adult_loader,
    )
    monkeypatch.setattr(runner, "build_model", fake_model_factory)

    results = runner.run_stress_test(
        dataset_name="adult",
        noise_levels=[0.0],
        seeds=[42],
        model_names=["logistic_regression"],
        metric_evaluator=fake_evaluator,
    )

    assert adult_loader_called == [True]
    assert results.iloc[0]["dataset"] == "adult"


def test_runner_rejects_unsupported_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        runner.run_stress_test(dataset_name="unknown_dataset")