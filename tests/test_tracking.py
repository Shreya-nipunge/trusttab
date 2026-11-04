"""Tests for offline experiment tracking and reproducibility utilities."""

import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from trusttab.experiments import benchmark, tracking
from trusttab.experiments import run as experiment_cli
from trusttab.experiments.config import load_experiment_config
from trusttab.experiments.tracking import (
    capture_environment,
    compare_runs,
    configuration_hash,
    dataset_fingerprint,
    get_run,
    inspect_run,
    list_runs,
    verify_reproducibility,
)


def _dataset():
    features = pd.DataFrame(
        {"age": np.linspace(20.0, 70.0, 120), "group": ["a", "b"] * 60},
        index=np.arange(1000, 1120),
    )
    target = pd.Series([0, 1] * 60, name="target")
    return features, target


class FakeModel:
    classes_ = np.array([0, 1])

    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features):
        return np.tile([0.7, 0.3], (len(features), 1))


def _tracked_benchmark(monkeypatch, runs_dir, *, seed=4):
    features, target = _dataset()
    monkeypatch.setitem(
        benchmark.DATASET_LOADERS, "german_credit", lambda: (features, target)
    )
    monkeypatch.setattr(benchmark, "build_model", lambda **kwargs: FakeModel())
    return benchmark.run_robustness_benchmark(
        dataset_name="german_credit",
        label_noise_levels=[0.0, 0.2],
        seeds=[seed],
        model_names=["logistic_regression"],
        tracking_dir=runs_dir,
        experiment_name="synthetic run",
        original_config={
            "seed": seed,
            "path": Path("relative/config.yaml"),
            "vector": np.array([2, 4]),
            "unavailable": np.nan,
        },
        normalized_config={"seed": seed},
    )


def test_run_ids_are_unique_and_complete_records_include_provenance(
    monkeypatch, tmp_path
):
    first = _tracked_benchmark(monkeypatch, tmp_path / "runs")
    second = _tracked_benchmark(monkeypatch, tmp_path / "runs")
    assert first.attrs["run_id"] != second.attrs["run_id"]

    run = get_run(tmp_path / "runs", first.attrs["run_id"])
    manifest = run["manifest"]
    assert manifest["status"] == "completed"
    assert manifest["created_at_utc"].endswith("Z")
    assert manifest["completed_at_utc"].endswith("Z")
    assert manifest["experiment_name"] == "synthetic run"
    assert manifest["dataset_names"] == ["german_credit"]
    assert manifest["models"] == ["logistic_regression"]
    assert manifest["random_seeds"] == [4]
    assert manifest["configuration_sha256"] == configuration_hash(
        {
            "seed": 4,
            "path": Path("relative/config.yaml"),
            "vector": np.array([2, 4]),
            "unavailable": np.nan,
        }
    )
    assert run["config"]["original"]["path"] == str(Path("relative/config.yaml"))
    assert run["config"]["original"]["vector"] == [2, 4]
    assert run["config"]["original"]["unavailable"] is None
    assert run["dataset"]["german_credit"]["n_samples"] == 120
    assert run["dataset"]["german_credit"]["n_features"] == 2
    assert run["dataset"]["german_credit"]["fingerprint_sha256"]
    for filename in (
        "manifest.json",
        "config.json",
        "environment.json",
        "dataset.json",
        "results.csv",
        "report.md",
    ):
        assert (run["run_dir"] / filename).is_file()
    assert list_runs(
        tmp_path / "runs",
        dataset="german_credit",
        model="logistic_regression",
        status="completed",
    )
    assert inspect_run(tmp_path / "runs", first.attrs["run_id"])["manifest"] == manifest


def test_configuration_hash_and_dataset_fingerprint_are_deterministic():
    assert configuration_hash({"b": 2, "a": 1}) == configuration_hash({"a": 1, "b": 2})
    features, target = _dataset()
    reset_features = features.reset_index(drop=True)
    reset_target = target.reset_index(drop=True)
    assert dataset_fingerprint(features, target) == dataset_fingerprint(
        reset_features, reset_target
    )
    changed = target.copy()
    changed.iloc[0] = 1
    assert dataset_fingerprint(features, target) != dataset_fingerprint(
        features, changed
    )


def test_incomplete_run_remains_distinguishable_as_running(tmp_path):
    tracker = tracking._RunTracker(
        tmp_path / "runs",
        experiment_name=None,
        original_config={"seed": 8},
        normalized_config={"seed": 8},
        dataset_names=["german_credit"],
        models=["logistic_regression"],
        seeds=[8],
    )
    assert list_runs(tmp_path / "runs", status="running")[0]["run_id"] == tracker.run_id
    assert get_run(tmp_path / "runs", tracker.run_id)["manifest"]["status"] == "running"


def test_failed_benchmark_updates_manifest_and_registry_filters(monkeypatch, tmp_path):
    monkeypatch.setitem(
        benchmark.DATASET_LOADERS,
        "german_credit",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        benchmark.run_robustness_benchmark(
            dataset_name="german_credit",
            seeds=[1],
            model_names=["logistic_regression"],
            tracking_dir=tmp_path / "runs",
        )
    runs = list_runs(tmp_path / "runs", status="failed")
    assert len(runs) == 1
    assert runs[0]["failure"]["type"] == "RuntimeError"
    assert runs[0]["failure"]["message"] == "synthetic failure"
    assert list_runs(tmp_path / "runs", status="completed") == []


def test_verification_compares_metadata_and_numerical_metrics(monkeypatch, tmp_path):
    first = _tracked_benchmark(monkeypatch, tmp_path / "runs")
    second = _tracked_benchmark(monkeypatch, tmp_path / "runs")
    run_dir_a = first.attrs["run_dir"]
    run_dir_b = second.attrs["run_dir"]
    verified = verify_reproducibility(
        tmp_path / "runs", first.attrs["run_id"], second.attrs["run_id"]
    )
    assert verified["status"] == "numerically_equivalent"
    assert (
        compare_runs(tmp_path / "runs", first.attrs["run_id"], second.attrs["run_id"])[
            "status"
        ]
        == "numerically_equivalent"
    )
    assert verified["checks"]["configuration_hash"]["match"]
    assert verified["checks"]["dataset_fingerprints"]["match"]
    assert verified["checks"]["result_metrics"]["match"]

    changed_results = pd.read_csv(f"{run_dir_b}/results.csv")
    changed_results.loc[0, "accuracy"] += 1e-5
    changed_results.to_csv(f"{run_dir_b}/results.csv", index=False)
    strict = verify_reproducibility(
        tmp_path / "runs", first.attrs["run_id"], second.attrs["run_id"]
    )
    tolerant = verify_reproducibility(
        tmp_path / "runs",
        first.attrs["run_id"],
        second.attrs["run_id"],
        atol=1e-4,
    )
    assert strict["checks"]["result_metrics"]["match"] is False
    assert tolerant["checks"]["result_metrics"]["match"] is True
    assert run_dir_a != run_dir_b


def test_missing_optional_dependency_is_recorded_as_unavailable(monkeypatch):
    original_version = tracking.metadata.version

    def version(package):
        if package == "xgboost":
            raise tracking.metadata.PackageNotFoundError(package)
        return original_version(package)

    monkeypatch.setattr(tracking.metadata, "version", version)
    assert capture_environment()["dependencies"]["xgboost"] == "unavailable"


def test_tracking_disabled_keeps_benchmark_without_run_files(monkeypatch, tmp_path):
    features, target = _dataset()
    monkeypatch.setitem(
        benchmark.DATASET_LOADERS, "german_credit", lambda: (features, target)
    )
    monkeypatch.setattr(benchmark, "build_model", lambda **kwargs: FakeModel())
    result = benchmark.run_robustness_benchmark(
        dataset_name="german_credit",
        seeds=[3],
        model_names=["logistic_regression"],
    )
    assert "run_id" not in result.attrs
    assert list(tmp_path.iterdir()) == []


def test_cli_tracked_smoke_run_and_registry_commands(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "tracked.yaml"
    config_path.write_text(
        dedent(
            """\
            project:
              name: TrustTab
            dataset:
              name: german_credit
            experiments:
              seeds: [1]
              noise_levels: [0.0, 0.2]
            models: [logistic_regression]
            output:
              directory: results
              filename: smoke.csv
            tracking:
              enabled: true
              directory: runs
              experiment_name: smoke
            """
        ),
        encoding="utf-8",
    )
    features, target = _dataset()
    monkeypatch.setitem(
        benchmark.DATASET_LOADERS, "german_credit", lambda: (features, target)
    )
    monkeypatch.setattr(benchmark, "build_model", lambda **kwargs: FakeModel())
    monkeypatch.setattr(
        "sys.argv",
        [
            "trusttab-run",
            "--config",
            str(config_path),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    experiment_cli.main()
    created = list_runs(tmp_path / "runs", status="completed")
    assert len(created) == 1
    run_id = created[0]["run_id"]
    assert (tmp_path / "runs" / run_id / "results.csv").is_file()

    monkeypatch.setattr(
        "sys.argv",
        ["trusttab-run", "--list-runs", "--runs-dir", str(tmp_path / "runs")],
    )
    experiment_cli.main()
    assert run_id in capsys.readouterr().out

    monkeypatch.setattr(
        "sys.argv",
        ["trusttab-run", "--verify-run", run_id, "--runs-dir", str(tmp_path / "runs")],
    )
    experiment_cli.main()
    verified_runs = list_runs(tmp_path / "runs", status="completed")
    assert len(verified_runs) == 2
    new_run_id = next(run["run_id"] for run in verified_runs if run["run_id"] != run_id)
    verification = json.loads(
        (tmp_path / "runs" / new_run_id / "verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert verification["status"] == "numerically_equivalent"

    loaded = load_experiment_config(config_path)
    assert loaded["tracking"]["enabled"] is True
