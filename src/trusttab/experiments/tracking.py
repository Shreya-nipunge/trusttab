"""Offline experiment tracking, run registry, and reproducibility checks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from datetime import UTC, date, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

_RESULT_KEYS = [
    "dataset",
    "model",
    "experiment_type",
    "corruption_type",
    "corruption_level",
    "seed",
]
_RESULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "brier_score",
    "ece",
    "adaptive_ece",
    "mce",
    "log_loss",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Convert common Python and NumPy objects to strict JSON-compatible data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (datetime,)):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return normalized.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def configuration_hash(configuration: Any) -> str:
    """Return SHA-256 of canonical JSON for a configuration object."""
    serialized = json.dumps(
        _json_safe(configuration),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def dataset_fingerprint(features: pd.DataFrame, target: pd.Series) -> str:
    """Hash dataset schema and ordered values without storing raw observations.

    The index and source path are excluded. Row order, column order, names,
    dtypes, and target values are included because they can affect training.
    """
    if not isinstance(features, pd.DataFrame) or not isinstance(target, pd.Series):
        raise TypeError("features must be a DataFrame and target must be a Series.")
    if len(features) != len(target):
        raise ValueError("features and target must have equal lengths.")
    schema = {
        "columns": [str(column) for column in features.columns],
        "dtypes": [str(dtype) for dtype in features.dtypes],
        "target_dtype": str(target.dtype),
        "n_samples": len(features),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    feature_hashes = (
        hash_pandas_object(
            features.reset_index(drop=True), index=False, categorize=True
        )
        .to_numpy(dtype=np.uint64)
        .astype("<u8", copy=False)
    )
    target_hashes = (
        hash_pandas_object(target.reset_index(drop=True), index=False, categorize=True)
        .to_numpy(dtype=np.uint64)
        .astype("<u8", copy=False)
    )
    digest.update(feature_hashes.tobytes())
    digest.update(target_hashes.tobytes())
    return digest.hexdigest()


def dataset_metadata(
    name: str, features: pd.DataFrame, target: pd.Series
) -> dict[str, Any]:
    """Describe a loaded dataset without copying its raw values to JSON."""
    counts = target.value_counts(dropna=False, sort=False)
    return {
        "name": name,
        "n_samples": len(features),
        "n_features": features.shape[1],
        "target_column_or_description": str(target.name or "target"),
        "class_distribution": {
            str(label): int(count) for label, count in counts.items()
        },
        "feature_types": {
            str(column): str(dtype) for column, dtype in features.dtypes.items()
        },
        "fingerprint_sha256": dataset_fingerprint(features, target),
    }


def _git_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout
        return {"commit": commit, "working_tree_clean": not bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_clean": None}


def capture_environment() -> dict[str, Any]:
    """Capture selected runtime/dependency details without reading env vars."""
    from trusttab import __version__

    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "xgboost": "xgboost",
    }
    versions = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return {
        "python_version": platform.python_version(),
        "trusttab_version": __version__,
        "operating_system": platform.system(),
        "platform": platform.platform(),
        "dependencies": versions,
        "git": _git_metadata(),
    }


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


class _RunTracker:
    """Manage files for one tracked benchmark execution."""

    def __init__(
        self,
        runs_dir: str | Path,
        *,
        experiment_name: str | None,
        original_config: Any,
        normalized_config: Any,
        dataset_names: list[str],
        models: list[str],
        seeds: list[int],
    ) -> None:
        if isinstance(runs_dir, str) and not runs_dir.strip():
            raise ValueError("runs_dir must not be empty.")
        if experiment_name is not None and (
            not isinstance(experiment_name, str) or not experiment_name.strip()
        ):
            raise ValueError(
                "experiment_name must be a non-empty string when provided."
            )
        root = Path(runs_dir)
        root.mkdir(parents=True, exist_ok=True)
        for _attempt in range(10):
            run_id = _new_run_id()
            run_dir = root / run_id
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("Could not allocate a unique experiment run ID.")
        self.run_id = run_id
        self.run_dir = run_dir
        self.environment = capture_environment()
        self.config_hash = configuration_hash(original_config)
        self.datasets: dict[str, Any] = {}
        self.manifest = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "created_at_utc": _utc_now(),
            "completed_at_utc": None,
            "status": "running",
            "trusttab_version": self.environment["trusttab_version"],
            "git_commit": self.environment["git"]["commit"],
            "git_working_tree_clean": self.environment["git"]["working_tree_clean"],
            "dataset_names": dataset_names,
            "models": models,
            "random_seeds": seeds,
            "configuration_sha256": self.config_hash,
            "result_rows": None,
            "failure": None,
        }
        _write_json(
            run_dir / "config.json",
            {
                "original": original_config,
                "normalized": normalized_config,
                "sha256": self.config_hash,
            },
        )
        _write_json(run_dir / "environment.json", self.environment)
        _write_json(run_dir / "dataset.json", self.datasets)
        self._save_manifest()

    def _save_manifest(self) -> None:
        _write_json(self.run_dir / "manifest.json", self.manifest)

    def record_dataset(
        self, name: str, features: pd.DataFrame, target: pd.Series
    ) -> None:
        self.datasets[name] = dataset_metadata(name, features, target)
        _write_json(self.run_dir / "dataset.json", self.datasets)
        self.manifest["dataset_fingerprints"] = {
            key: value["fingerprint_sha256"] for key, value in self.datasets.items()
        }
        self._save_manifest()

    def complete(self, results: pd.DataFrame) -> None:
        results.to_csv(self.run_dir / "results.csv", index=False)
        from trusttab.reporting import generate_robustness_report

        generate_robustness_report(results, self.run_dir)
        self.manifest.update(
            status="completed",
            completed_at_utc=_utc_now(),
            result_rows=len(results),
        )
        self._save_manifest()

    def fail(self, error: BaseException) -> None:
        self.manifest.update(
            status="failed",
            completed_at_utc=_utc_now(),
            failure={
                "type": type(error).__name__,
                "message": str(error)[:2000],
            },
        )
        self._save_manifest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_runs(
    runs_dir: str | Path,
    *,
    dataset: str | None = None,
    model: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List local runs, optionally filtered by dataset, model, or status."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    if status is not None and status not in {"running", "completed", "failed"}:
        raise ValueError("status must be 'running', 'completed', or 'failed'.")
    runs = []
    for manifest_path in root.glob("*/manifest.json"):
        manifest = _read_json(manifest_path)
        if dataset is not None and dataset not in manifest.get("dataset_names", []):
            continue
        if model is not None and model not in manifest.get("models", []):
            continue
        if status is not None and manifest.get("status") != status:
            continue
        runs.append(manifest)
    return sorted(runs, key=lambda run: run.get("created_at_utc", ""), reverse=True)


def get_run(runs_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Retrieve a run manifest and its saved configuration/environment metadata."""
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a simple non-empty run identifier.")
    run_dir = Path(runs_dir) / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return {
        "run_dir": run_dir,
        "manifest": _read_json(run_dir / "manifest.json"),
        "config": _read_json(run_dir / "config.json"),
        "environment": _read_json(run_dir / "environment.json"),
        "dataset": _read_json(run_dir / "dataset.json"),
    }


def inspect_run(runs_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Alias for retrieving all inspectable metadata for one run."""
    return get_run(runs_dir, run_id)


def _compare_result_files(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_path = reference_dir / "results.csv"
    candidate_path = candidate_dir / "results.csv"
    if not reference_path.exists() or not candidate_path.exists():
        return {"match": False, "status": "missing_results_file"}
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    keys = [key for key in _RESULT_KEYS if key in reference and key in candidate]
    if len(keys) != len(_RESULT_KEYS):
        return {"match": False, "status": "missing_scenario_keys"}
    if reference.duplicated(keys).any() or candidate.duplicated(keys).any():
        return {"match": False, "status": "duplicate_scenario_keys"}
    joined = reference.merge(
        candidate,
        on=keys,
        how="outer",
        suffixes=("_reference", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = int((joined["_merge"] != "both").sum())
    metrics = {}
    all_match = unmatched == 0
    reference_metrics = [name for name in _RESULT_METRICS if name in reference.columns]
    candidate_metrics = [name for name in _RESULT_METRICS if name in candidate.columns]
    metric_schema_match = reference_metrics == candidate_metrics
    for metric in _RESULT_METRICS:
        left_name, right_name = f"{metric}_reference", f"{metric}_candidate"
        if left_name not in joined or right_name not in joined:
            continue
        left = pd.to_numeric(joined[left_name], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(joined[right_name], errors="coerce").to_numpy(dtype=float)
        matched_rows = joined["_merge"].to_numpy() == "both"
        equal = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
        equal &= matched_rows
        mismatches = int((~equal & matched_rows).sum())
        finite = np.isfinite(left[matched_rows]) & np.isfinite(right[matched_rows])
        max_difference = (
            float(
                np.max(np.abs(left[matched_rows][finite] - right[matched_rows][finite]))
            )
            if finite.any()
            else 0.0
        )
        metrics[metric] = {
            "match": mismatches == 0,
            "compared_rows": int(matched_rows.sum()),
            "mismatches": mismatches,
            "max_absolute_difference": max_difference,
        }
        all_match &= mismatches == 0
    return {
        "match": bool(all_match and metric_schema_match),
        "status": "compared",
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
        "unmatched_scenarios": unmatched,
        "metric_schema_match": metric_schema_match,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "metrics": metrics,
        "tolerance": {"rtol": rtol, "atol": atol},
    }


def verify_reproducibility(
    runs_dir: str | Path,
    reference_run_id: str,
    candidate_run_id: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Compare configuration, data, environment, seeds, and metrics of two runs.

    A successful numeric comparison means results are equivalent within the
    supplied tolerances; it does not claim bitwise reproducibility.
    """
    for value, name in ((rtol, "rtol"), (atol, "atol")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative number.")
    reference = get_run(runs_dir, reference_run_id)
    candidate = get_run(runs_dir, candidate_run_id)
    ref_manifest, cand_manifest = reference["manifest"], candidate["manifest"]
    ref_env, cand_env = reference["environment"], candidate["environment"]
    ref_git, cand_git = ref_env.get("git", {}), cand_env.get("git", {})
    ref_fingerprints = ref_manifest.get("dataset_fingerprints", {})
    cand_fingerprints = cand_manifest.get("dataset_fingerprints", {})
    git_available = bool(ref_git.get("commit") and cand_git.get("commit"))
    checks = {
        "configuration_hash": {
            "match": ref_manifest.get("configuration_sha256")
            == cand_manifest.get("configuration_sha256"),
            "reference": ref_manifest.get("configuration_sha256"),
            "candidate": cand_manifest.get("configuration_sha256"),
        },
        "dataset_fingerprints": {
            "match": ref_fingerprints == cand_fingerprints,
            "reference": ref_fingerprints,
            "candidate": cand_fingerprints,
        },
        "git_commit": {
            "match": ref_git.get("commit") == cand_git.get("commit")
            if git_available
            else None,
            "status": "compared" if git_available else "unavailable",
            "reference": ref_git.get("commit"),
            "candidate": cand_git.get("commit"),
        },
        "git_working_tree_clean": {
            "match": (
                ref_git.get("working_tree_clean") == cand_git.get("working_tree_clean")
                if isinstance(ref_git.get("working_tree_clean"), bool)
                and isinstance(cand_git.get("working_tree_clean"), bool)
                else None
            ),
            "reference": ref_git.get("working_tree_clean"),
            "candidate": cand_git.get("working_tree_clean"),
        },
        "dependency_versions": {
            "match": ref_env.get("dependencies") == cand_env.get("dependencies"),
            "reference": ref_env.get("dependencies"),
            "candidate": cand_env.get("dependencies"),
        },
        "runtime_versions": {
            "match": {
                "python_version": ref_env.get("python_version"),
                "trusttab_version": ref_env.get("trusttab_version"),
            }
            == {
                "python_version": cand_env.get("python_version"),
                "trusttab_version": cand_env.get("trusttab_version"),
            },
            "reference": {
                "python_version": ref_env.get("python_version"),
                "trusttab_version": ref_env.get("trusttab_version"),
            },
            "candidate": {
                "python_version": cand_env.get("python_version"),
                "trusttab_version": cand_env.get("trusttab_version"),
            },
        },
        "random_seeds": {
            "match": ref_manifest.get("random_seeds")
            == cand_manifest.get("random_seeds"),
            "reference": ref_manifest.get("random_seeds"),
            "candidate": cand_manifest.get("random_seeds"),
        },
        "run_status": {
            "match": ref_manifest.get("status")
            == cand_manifest.get("status")
            == "completed",
            "reference": ref_manifest.get("status"),
            "candidate": cand_manifest.get("status"),
        },
    }
    result_comparison = _compare_result_files(
        reference["run_dir"], candidate["run_dir"], rtol=rtol, atol=atol
    )
    checks["result_metrics"] = result_comparison
    considered = [
        check["match"] for check in checks.values() if check["match"] is not None
    ]
    return {
        "reference_run_id": reference_run_id,
        "candidate_run_id": candidate_run_id,
        "status": "numerically_equivalent"
        if considered and all(considered)
        else "differences_found",
        "checks": checks,
    }


def compare_runs(
    runs_dir: str | Path,
    first_run_id: str,
    second_run_id: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Compare two runs using their provenance metadata and saved metrics."""
    return verify_reproducibility(
        runs_dir, first_run_id, second_run_id, rtol=rtol, atol=atol
    )
