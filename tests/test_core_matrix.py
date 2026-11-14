"""Bounded fake-model verification of the configured core matrix."""

from itertools import product

import numpy as np
import pandas as pd

from trusttab.experiments import benchmark
from trusttab.experiments.config import (
    expand_experiment_conditions,
    load_experiment_config,
)


class FakeModel:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.fit_y = np.asarray(y).copy()
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.7, 0.3], (len(X), 1))


def _fixture_dataset(offset):
    features = pd.DataFrame(
        {
            "measurement": np.arange(40, dtype=float) + offset,
            "category": ["a", "b"] * 20,
        }
    )
    target = pd.Series([0, 1] * 20, name="target")
    return features, target


def test_runner_visits_every_core_condition_with_fake_models(monkeypatch):
    config = load_experiment_config("configs/experiments.yaml")
    assert config["tracking"] == {
        "enabled": True,
        "directory": "experiments/runs",
        "experiment_name": "trusttab-core-matrix",
    }
    conditions = expand_experiment_conditions(config)
    expected = set(
        product(
            config["dataset"]["names"],
            config["models"],
            config["benchmark_settings"]["label_noise_levels"],
            config["experiments"]["seeds"],
        )
    )
    originals = {
        "german_credit": _fixture_dataset(0),
        "heart_disease": _fixture_dataset(100),
    }
    snapshots = {
        name: (features.copy(deep=True), target.copy(deep=True))
        for name, (features, target) in originals.items()
    }
    for name, (features, target) in originals.items():
        monkeypatch.setitem(
            benchmark.DATASET_LOADERS,
            name,
            lambda data_home=None, features=features, target=target: (
                features,
                target,
            ),
        )

    model_builds = []
    evaluated_test_pairs = []
    original_evaluate = benchmark._evaluate

    def track_evaluation(model, X_test, y_test, metric_evaluator):
        evaluated_test_pairs.append(
            (X_test["measurement"].to_numpy(copy=True), np.asarray(y_test).copy())
        )
        return original_evaluate(model, X_test, y_test, metric_evaluator)

    def build_fake_model(model_name, features, random_state, model_params=None):
        model_builds.append((model_name, random_state, model_params))
        return FakeModel()

    monkeypatch.setattr(benchmark, "build_model", build_fake_model)
    monkeypatch.setattr(benchmark, "_evaluate", track_evaluation)
    results = benchmark.run_robustness_benchmark(
        dataset_names=config["dataset"]["names"],
        label_noise_levels=config["benchmark_settings"]["label_noise_levels"],
        seeds=config["experiments"]["seeds"],
        model_names=config["models"],
        test_size=config["experiments"]["test_size"],
        model_params=config["model_params"],
        dataset_data_home=config["dataset"].get("data_home"),
    )

    label_rows = results.loc[results["experiment_type"] == "label_noise"]
    actual = set(
        zip(
            label_rows["dataset"],
            label_rows["model"],
            label_rows["corruption_level"],
            label_rows["seed"],
        )
    )
    assert len(conditions) == 200
    assert len(label_rows) == 200
    assert actual == expected
    assert len(model_builds) == 200
    assert all(
        params == config["model_params"][model] for model, _, params in model_builds
    )
    assert label_rows[["split_seed", "noise_seed", "model_seed"]].notna().all().all()
    assert (label_rows["split_seed"] != label_rows["noise_seed"]).all()
    assert (label_rows["noise_seed"] != label_rows["model_seed"]).all()

    for measured_rows, observed_test_labels in evaluated_test_pairs:
        dataset_name = "german_credit" if measured_rows.min() < 50 else "heart_disease"
        features, target = originals[dataset_name]
        offset = int(features["measurement"].iloc[0])
        row_ids = measured_rows.astype(int) - offset
        expected_test_labels = target.iloc[row_ids].to_numpy()
        np.testing.assert_array_equal(observed_test_labels, expected_test_labels)

    for name, (features, target) in originals.items():
        pd.testing.assert_frame_equal(features, snapshots[name][0])
        pd.testing.assert_series_equal(target, snapshots[name][1])
