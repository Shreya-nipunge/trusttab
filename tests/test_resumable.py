"""Interruption and integrity tests for atomic core-matrix scheduling."""

import json

import numpy as np
import pandas as pd
import pytest

from trusttab.experiments import resumable
from trusttab.experiments import benchmark
from trusttab.experiments.config import load_experiment_config


def _small_config():
    config = load_experiment_config("configs/experiments.yaml")
    config["dataset"]["names"] = ["german_credit"]
    config["models"] = ["logistic_regression"]
    config["benchmark_settings"]["label_noise_levels"] = [0.0, 0.1]
    config["experiments"]["seeds"] = [0, 1]
    return config


def _fixture_loader(data_home=None):
    return (
        pd.DataFrame({"x": np.arange(20, dtype=float)}),
        pd.Series([0, 1] * 10, name="target"),
    )


def _emit_condition(callback, dataset, model, level, seed):
    callback(
        {
            "dataset": dataset,
            "model": model,
            "experiment_type": "label_noise",
            "corruption_level": level,
            "seed": seed,
            "condition_id": (f"{dataset}:{model}:{float(level).hex()}:{int(seed)}"),
        }
    )


def test_interruption_checkpoints_and_resume_skips_completed_conditions(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(resumable.DATASET_LOADERS, "german_credit", _fixture_loader)
    config = _small_config()
    run_dir = tmp_path / "run"
    emitted = []

    def interrupting_runner(**kwargs):
        callback = kwargs["_result_callback"]
        levels = kwargs["label_noise_levels"]
        seed = kwargs["seeds"][0]
        for level in levels:
            _emit_condition(
                callback, "german_credit", "logistic_regression", level, seed
            )
            emitted.append((level, seed))
            if len(emitted) == 2:
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        resumable.run_resumable_experiment(
            config, run_dir, benchmark_runner=interrupting_runner
        )

    checkpoint = pd.read_csv(run_dir / "raw_results.csv")
    assert len(checkpoint) == 2
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert (run_dir / ".run.lock").exists()

    calls = []

    def completing_runner(**kwargs):
        seed = kwargs["seeds"][0]
        for level in kwargs["label_noise_levels"]:
            calls.append((level, seed))
            _emit_condition(
                kwargs["_result_callback"],
                "german_credit",
                "logistic_regression",
                level,
                seed,
            )

    results = resumable.run_resumable_experiment(
        config, run_dir, benchmark_runner=completing_runner
    )
    label_rows = results[results["experiment_type"] == "label_noise"]
    assert len(label_rows) == 4
    assert label_rows["condition_id"].is_unique
    assert len(calls) == 2
    assert set(calls).isdisjoint(set(emitted))
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["integrity"]["complete"] is True
    assert manifest["integrity"]["completed"] == 4


def test_failed_chunk_is_explicit_and_can_be_retried(tmp_path, monkeypatch):
    monkeypatch.setitem(resumable.DATASET_LOADERS, "german_credit", _fixture_loader)
    config = _small_config()
    run_dir = tmp_path / "run"

    def failed_runner(**kwargs):
        raise RuntimeError("controlled fixture failure")

    partial = resumable.run_resumable_experiment(
        config, run_dir, benchmark_runner=failed_runner
    )
    assert partial.empty
    assert partial.attrs["integrity"]["complete"] is False
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert len(manifest["failures"]) == 2
    assert all(
        "controlled fixture failure" in error for error in manifest["failures"].values()
    )
    assert (
        sum(state["status"] == "pending" for state in manifest["conditions"].values())
        == 2
    )

    def successful_runner(**kwargs):
        seed = kwargs["seeds"][0]
        for level in kwargs["label_noise_levels"]:
            _emit_condition(
                kwargs["_result_callback"],
                "german_credit",
                "logistic_regression",
                level,
                seed,
            )

    completed = resumable.run_resumable_experiment(
        config, run_dir, benchmark_runner=successful_runner
    )
    assert len(completed[completed["experiment_type"] == "label_noise"]) == 4
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["failures"] == {}


def test_integrity_rejects_duplicates_and_reports_missing_conditions():
    expected = ["a", "b"]
    rows = pd.DataFrame(
        {
            "experiment_type": ["label_noise", "label_noise"],
            "condition_id": ["a", "a"],
        }
    )
    report = resumable.validate_resumable_results(expected, rows, {})
    assert report["duplicates"] == ["a"]
    assert report["missing"] == ["b"]
    assert report["complete"] is False


def test_actual_benchmark_persists_each_completed_condition(tmp_path, monkeypatch):
    monkeypatch.setitem(resumable.DATASET_LOADERS, "german_credit", _fixture_loader)
    monkeypatch.setitem(benchmark.DATASET_LOADERS, "german_credit", _fixture_loader)
    monkeypatch.setattr(benchmark, "build_model", lambda **kwargs: _TinyEstimator())
    config = _small_config()
    config["experiments"]["seeds"] = [0]
    run_dir = tmp_path / "actual-runner"

    results = resumable.run_resumable_experiment(config, run_dir)
    labels = results[results["experiment_type"] == "label_noise"]
    assert len(labels) == 2
    assert labels["condition_id"].is_unique
    assert labels["run_id"].notna().all()
    assert labels["completed_at_utc"].notna().all()
    assert labels["train_time_seconds"].notna().all()
    assert pd.read_csv(run_dir / "raw_results.csv")["condition_id"].notna().all()


class _TinyEstimator:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.rate = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), int(self.rate >= 0.5), dtype=int)

    def predict_proba(self, X):
        return np.tile([1 - self.rate, self.rate], (len(X), 1))


def test_resume_refuses_modified_configuration(tmp_path, monkeypatch):
    monkeypatch.setitem(resumable.DATASET_LOADERS, "german_credit", _fixture_loader)
    config = _small_config()
    run_dir = tmp_path / "run"

    def failed_runner(**kwargs):
        raise RuntimeError("leave an incomplete manifest")

    resumable.run_resumable_experiment(config, run_dir, benchmark_runner=failed_runner)
    changed = _small_config()
    changed["model_params"]["logistic_regression"]["max_iter"] = 50
    with pytest.raises(ValueError, match="configuration has changed"):
        resumable.run_resumable_experiment(
            changed, run_dir, benchmark_runner=failed_runner
        )
