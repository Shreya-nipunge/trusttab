"""Tests for the unified robustness benchmark runner."""

import numpy as np
import pandas as pd
import pytest

from trusttab.experiments import benchmark
from trusttab.experiments import run as experiment_cli


def _dataset(offset=0.0):
    X = pd.DataFrame(
        {
            "age": np.linspace(20 + offset, 70 + offset, 120),
            "job": ["worker", "manager"] * 60,
        }
    )
    y = pd.Series([0, 1] * 60, name="target")
    return X, y


class FakeModel:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.fit_y = np.asarray(y).copy()
        self.fit_rows = len(X)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.7, 0.3], (len(X), 1))


def test_unified_benchmark_combines_scenarios_datasets_models_and_seeds(
    monkeypatch,
):
    originals = {"german_credit": _dataset(), "adult": _dataset(3.0)}
    snapshots = {
        name: (X.copy(deep=True), y.copy(deep=True))
        for name, (X, y) in originals.items()
    }
    for name, (X, y) in originals.items():
        monkeypatch.setitem(
            benchmark.DATASET_LOADERS,
            name,
            lambda X=X, y=y: (X, y),
        )

    fit_calls = []

    def build_model(model_name, features, random_state, model_params=None):
        model = FakeModel()
        original_fit = model.fit

        def track_fit(X, y):
            fit_calls.append((model_name, random_state, np.asarray(y).copy()))
            return original_fit(X, y)

        model.fit = track_fit
        return model

    monkeypatch.setattr(benchmark, "build_model", build_model)
    result = benchmark.run_robustness_benchmark(
        dataset_names=["german_credit", "adult"],
        label_noise_levels=[0.0, 0.2],
        feature_shifts={
            "enabled": True,
            "types": ["mean_shift", "category_replacement"],
            "magnitudes": [0.1, 0.2],
            "features": {
                "mean_shift": ["age"],
                "category_replacement": ["job"],
            },
        },
        seeds=[4, 9],
        model_names=["logistic_regression", "random_forest"],
    )

    # 2 datasets x 2 seeds x 2 models x (baseline + 2 label + 4 shifts).
    assert len(result) == 56
    assert set(result["experiment_type"]) == {
        "clean_baseline",
        "label_noise",
        "feature_shift",
    }
    assert set(result["corruption_type"]) == {
        "none",
        "symmetric_label_noise",
        "mean_shift",
        "category_replacement",
    }
    assert set(benchmark.BENCHMARK_RESULT_COLUMNS) == set(result.columns)
    assert result["baseline_id"].nunique() == 8
    assert {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "brier_score",
        "ece",
        "adaptive_ece",
        "mce",
        "log_loss",
    }.issubset(result.columns)
    assert len(fit_calls) == 16  # clean fit once, plus one fit per positive noise level

    baseline = result[result.experiment_type == "clean_baseline"].set_index("baseline_id")
    zero_noise = result[
        (result.experiment_type == "label_noise") & (result.corruption_level == 0)
    ].set_index("baseline_id")
    pd.testing.assert_series_equal(
        baseline["accuracy"], zero_noise["accuracy"], check_names=False
    )
    pd.testing.assert_series_equal(
        baseline["mce"], zero_noise["mce"], check_names=False
    )
    for name, (X, y) in originals.items():
        pd.testing.assert_frame_equal(X, snapshots[name][0])
        pd.testing.assert_series_equal(y, snapshots[name][1])


def test_benchmark_is_deterministic_and_exports_csv(monkeypatch, tmp_path):
    X, y = _dataset()
    monkeypatch.setitem(benchmark.DATASET_LOADERS, "german_credit", lambda: (X, y))
    monkeypatch.setattr(
        benchmark,
        "build_model",
        lambda **kwargs: FakeModel(),
    )
    kwargs = {
        "dataset_name": "german_credit",
        "label_noise_levels": [0.0, 0.25],
        "feature_shifts": [
            {"type": "gaussian_noise", "features": ["age"], "magnitude": 0.2}
        ],
        "seeds": [12],
        "model_names": ["logistic_regression"],
    }
    first = benchmark.run_robustness_benchmark(**kwargs)
    output_path = tmp_path / "nested" / "benchmark.csv"
    second = benchmark.run_robustness_benchmark(**kwargs, output_path=output_path)
    pd.testing.assert_frame_equal(first, second)
    saved = pd.read_csv(output_path)
    assert len(saved) == len(second)
    assert list(saved.columns) == list(second.columns)
    assert saved["experiment_type"].tolist() == second["experiment_type"].tolist()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_names": []},
        {"dataset_names": ["unsupported"]},
        {"seeds": []},
        {"seeds": [True]},
        {"label_noise_levels": [-0.1]},
        {"label_noise_levels": [0.2, 0.2]},
        {"model_names": []},
        {"test_size": 1.0},
        {"dataset_names": ["adult"], "dataset_name": "german_credit"},
    ],
)
def test_unified_benchmark_rejects_invalid_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        benchmark.run_robustness_benchmark(**kwargs)


def test_metric_evaluator_missing_metrics_are_filled_with_nan(monkeypatch):
    X, y = _dataset()
    monkeypatch.setitem(benchmark.DATASET_LOADERS, "german_credit", lambda: (X, y))
    monkeypatch.setattr(benchmark, "build_model", lambda **kwargs: FakeModel())
    result = benchmark.run_robustness_benchmark(
        dataset_name="german_credit",
        seeds=[1],
        model_names=["logistic_regression"],
        metric_evaluator=lambda *_: {"accuracy": 0.5},
    )
    assert result.loc[0, "accuracy"] == 0.5
    assert np.isnan(result.loc[0, "mce"])


def test_roc_auc_is_nan_when_test_split_contains_one_class():
    values = benchmark._evaluate(
        FakeModel(),
        pd.DataFrame({"age": [20.0, 30.0]}),
        pd.Series([0, 0]),
        benchmark.evaluate_classifier,
    )
    assert np.isnan(values["roc_auc"])
    assert np.isfinite(values["accuracy"])
    assert np.isfinite(values["log_loss"])


def test_cli_uses_unified_runner_for_combined_configuration(monkeypatch, tmp_path):
    config = {
        "benchmark_mode": "unified",
        "dataset": {"names": ["german_credit", "adult"]},
        "experiments": {"seeds": [1], "test_size": 0.2},
        "models": ["logistic_regression"],
        "benchmark_settings": {
            "label_noise_levels": [0.0, 0.2],
            "feature_shifts": {"enabled": False, "types": [], "magnitudes": []},
        },
        "output": {"directory": str(tmp_path), "filename": "unified.csv"},
    }
    captured = {}
    monkeypatch.setattr(experiment_cli, "load_experiment_config", lambda _: config)
    monkeypatch.setattr(
        experiment_cli,
        "run_robustness_benchmark",
        lambda **kwargs: captured.update(kwargs) or pd.DataFrame({"accuracy": [1.0]}),
    )
    monkeypatch.setattr("sys.argv", ["trusttab-run", "--config", "unused.yaml"])
    experiment_cli.main()
    assert captured["dataset_names"] == ["german_credit", "adult"]
    assert captured["label_noise_levels"] == [0.0, 0.2]
