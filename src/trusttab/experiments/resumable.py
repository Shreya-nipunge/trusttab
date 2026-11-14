"""Atomic, resumable execution for the configured label-noise matrix."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import uuid
from importlib import metadata
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

import pandas as pd

from trusttab.experiments.benchmark import run_robustness_benchmark
from trusttab.experiments.config import expand_experiment_conditions
from trusttab.experiments.runner import DATASET_LOADERS
from trusttab.experiments.tracking import configuration_hash, dataset_fingerprint


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _versions() -> dict[str, str | None]:
    names = (
        "trusttab",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "xgboost",
        "tabpfn",
        "torch",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _acquire_lock(path: Path):
    """Acquire an OS-managed lock that is released even after process death."""
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError(f"Another process holds the run lock: {path}") from exc
    return stream


def _release_lock(stream) -> None:
    with suppress(OSError):
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream.close()


def _dataset_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    dataset_config = config["dataset"]
    names = dataset_config.get("names") or [dataset_config["name"]]
    data_home = dataset_config.get("data_home")
    fingerprints = {}
    for name in names:
        loader = DATASET_LOADERS[name]
        home = data_home.get(name) if isinstance(data_home, Mapping) else data_home
        features, target = loader() if home is None else loader(data_home=home)
        fingerprints[name] = dataset_fingerprint(features, target)
    return fingerprints


def _condition_id(item: Mapping[str, Any]) -> str:
    return (
        f"{item['dataset']}:{item['model']}:"
        f"{float(item['noise_level']).hex()}:{int(item['seed'])}"
    )


def validate_resumable_results(
    expected_ids: list[str], result_rows: pd.DataFrame, failed: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact condition uniqueness and completeness without filling gaps."""
    if result_rows.empty:
        actual_ids = pd.Series(dtype=str)
    elif "condition_id" not in result_rows:
        raise ValueError("Results are missing condition_id.")
    else:
        actual_ids = result_rows.loc[
            result_rows["experiment_type"] == "label_noise", "condition_id"
        ].astype(str)
    duplicates = sorted(actual_ids[actual_ids.duplicated(keep=False)].unique())
    expected = set(expected_ids)
    actual = set(actual_ids)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    return {
        "expected": len(expected),
        "completed": len(actual),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "failed": sorted(set(failed).intersection(expected)),
        "complete": not (missing or unexpected or duplicates),
    }


def run_resumable_experiment(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    benchmark_runner: Callable[..., pd.DataFrame] = run_robustness_benchmark,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run one dataset/model/seed chunk at a time and checkpoint every result.

    Reuse ``run_dir`` to resume. The normalized configuration and dataset
    fingerprints must match the original run. Failed conditions remain in the
    manifest and are eligible for retry on the next invocation.
    """
    conditions = expand_experiment_conditions(config)
    expected_ids = [_condition_id(item) for item in conditions]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Configuration expands to duplicate experiment conditions.")
    if config.get("benchmark_settings", {}).get("feature_shifts"):
        raise ValueError(
            "Resumable core execution supports label-noise conditions only."
        )

    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".run.lock"
    lock_stream = _acquire_lock(lock)

    manifest_path = directory / "run_manifest.json"
    results_path = directory / "raw_results.csv"
    try:
        config_hash = configuration_hash(config)
        fingerprints = _dataset_fingerprints(config)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_hash") != config_hash:
                raise ValueError("Cannot resume: normalized configuration has changed.")
            if manifest.get("expected_condition_ids") != expected_ids:
                raise ValueError("Cannot resume: expected condition set has changed.")
            if manifest.get("dataset_fingerprints") != fingerprints:
                raise ValueError(
                    "Cannot resume: one or more dataset fingerprints changed."
                )
            if manifest.get("packages") != _versions():
                raise ValueError(
                    "Cannot resume: installed package versions have changed."
                )
            if manifest.get("python") != platform.python_version():
                raise ValueError("Cannot resume: Python version has changed.")
        else:
            other_files = [path.name for path in directory.iterdir() if path != lock]
            if other_files:
                raise FileExistsError(
                    "Run directory is non-empty but has no run manifest: "
                    f"{other_files}"
                )
            manifest = {
                "schema_version": 1,
                "run_id": f"trusttab-{uuid.uuid4().hex}",
                "status": "running",
                "created_at": _now(),
                "updated_at": _now(),
                "config_hash": config_hash,
                "normalized_config": config,
                "expected_condition_ids": expected_ids,
                "dataset_fingerprints": fingerprints,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _versions(),
                "conditions": {
                    key: {"status": "pending", "attempts": 0} for key in expected_ids
                },
                "failures": {},
            }
            _atomic_json(manifest_path, manifest)

        if results_path.exists():
            rows = pd.read_csv(results_path)
            if rows["condition_id"].duplicated().any():
                raise ValueError("Checkpoint results contain duplicate condition IDs.")
        else:
            rows = pd.DataFrame()
        completed = (
            set(
                rows.loc[
                    rows.get("experiment_type", pd.Series(dtype=str)) == "label_noise",
                    "condition_id",
                ].astype(str)
            )
            if not rows.empty
            else set()
        )
        if completed.difference(expected_ids):
            raise ValueError(
                "Checkpoint contains conditions outside this configuration."
            )
        for condition_id in completed:
            manifest["conditions"][condition_id]["status"] = "succeeded"
            manifest["conditions"][condition_id].pop("error", None)
            manifest["failures"].pop(condition_id, None)

        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for condition in conditions:
            grouped.setdefault(
                (condition["dataset"], condition["model"], condition["seed"]), []
            ).append(condition)

        for (dataset, model, seed), group in grouped.items():
            pending = [item for item in group if _condition_id(item) not in completed]
            if not pending:
                continue
            levels = [item["noise_level"] for item in pending]
            for item in pending:
                key = _condition_id(item)
                state = manifest["conditions"][key]
                state["status"] = "running"
                state["attempts"] += 1
                state.pop("error", None)
                state.pop("last_error", None)
                manifest["failures"].pop(key, None)
            manifest["status"] = "running"
            manifest["updated_at"] = _now()
            _atomic_json(manifest_path, manifest)
            if progress_callback:
                progress_callback(
                    f"Starting {dataset}/{model}/seed={seed}; "
                    f"pending noise levels={levels}"
                )

            def checkpoint(record: dict[str, Any]) -> None:
                nonlocal rows, completed
                row = dict(record)
                row["run_id"] = manifest["run_id"]
                row["scheduler_run_id"] = manifest["run_id"]
                row["completed_at_utc"] = _now()
                key = str(row.get("condition_id", ""))
                is_condition = row.get("experiment_type") == "label_noise"
                # Baseline rows have stable IDs too; ignore them on a resumed
                # group instead of duplicating their ancillary result.
                if not key or (is_condition and key in completed):
                    return
                if "condition_id" in rows and key in set(
                    rows["condition_id"].astype(str)
                ):
                    return
                rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
                _atomic_csv(results_path, rows)
                if is_condition:
                    completed.add(key)
                    manifest["conditions"][key]["status"] = "succeeded"
                    manifest["conditions"][key]["completed_at"] = row[
                        "completed_at_utc"
                    ]
                    manifest["failures"].pop(key, None)
                manifest["updated_at"] = _now()
                _atomic_json(manifest_path, manifest)
                if progress_callback and is_condition:
                    progress_callback(
                        f"Completed {len(completed)}/{len(expected_ids)}: {key}"
                    )

            try:
                benchmark_runner(
                    dataset_names=[dataset],
                    label_noise_levels=levels,
                    seeds=[seed],
                    model_names=[model],
                    test_size=config["experiments"].get("test_size", 0.2),
                    model_params=config.get("model_params", {}),
                    calibration_settings=config.get("calibration_settings"),
                    dataset_data_home=config["dataset"].get("data_home"),
                    output_path=None,
                    _result_callback=checkpoint,
                )
            except (KeyboardInterrupt, SystemExit):
                manifest["status"] = "interrupted"
                manifest["updated_at"] = _now()
                _atomic_json(manifest_path, manifest)
                raise
            except Exception as exc:  # retain this chunk's partial successes
                error = f"{type(exc).__name__}: {exc}"
                unresolved = [
                    _condition_id(item)
                    for item in pending
                    if _condition_id(item) not in completed
                ]
                if unresolved:
                    failed_id = unresolved[0]
                    manifest["conditions"][failed_id]["status"] = "failed"
                    manifest["conditions"][failed_id]["error"] = error
                    manifest["failures"][failed_id] = error
                    for not_attempted_id in unresolved[1:]:
                        manifest["conditions"][not_attempted_id]["status"] = "pending"
                        manifest["conditions"][not_attempted_id]["last_error"] = (
                            f"Not attempted after chunk failure at {failed_id}."
                        )
                manifest["updated_at"] = _now()
                _atomic_json(manifest_path, manifest)
                if progress_callback:
                    progress_callback(f"Failed {dataset}/{model}/seed={seed}: {error}")

        integrity = validate_resumable_results(expected_ids, rows, manifest["failures"])
        manifest["integrity"] = integrity
        manifest["status"] = "completed" if integrity["complete"] else "incomplete"
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)
        if integrity["complete"]:
            return rows
        rows.attrs["integrity"] = integrity
        return rows
    finally:
        _release_lock(lock_stream)
