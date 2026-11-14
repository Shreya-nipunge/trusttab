"""Unified robustness benchmark orchestration."""

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from trusttab.evaluation import (
    adaptive_expected_calibration_error,
    calibrate_fitted_classifier,
    evaluate_classifier,
    expected_calibration_error,
    log_loss_score,
    maximum_calibration_error,
)
from trusttab.experiments.runner import DATASET_LOADERS, MetricEvaluator
from trusttab.experiments.calibration_protocol import (
    validate_calibration_settings,
)
from trusttab.experiments.seeding import derive_seed
from trusttab.experiments.splitting import split_train_calibration_test
from trusttab.models import build_model
from trusttab.noise import (
    apply_feature_shifts,
    inject_symmetric_label_noise,
    validate_feature_shifts,
)

BENCHMARK_METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "brier_score",
    "ece",
    "adaptive_ece",
    "mce",
    "log_loss",
)
CALIBRATED_METRIC_COLUMNS = tuple(
    f"calibrated_{metric}" for metric in BENCHMARK_METRIC_COLUMNS
)

BENCHMARK_RESULT_COLUMNS = (
    "dataset",
    "model",
    "experiment_type",
    "corruption_type",
    "corruption_level",
    "seed",
    "split_seed",
    "noise_seed",
    "model_seed",
    *BENCHMARK_METRIC_COLUMNS,
    "n_train",
    "n_test",
    "baseline_id",
    "affected_features",
    "n_corrupted",
    "train_time_seconds",
    "shift_seed",
    "condition_id",
    "calibration_method",
    "calibration_status",
    "calibration_error",
    "calibration_split_seed",
    "n_calibration",
    "split_train_fraction",
    "split_calibration_fraction",
    "split_test_fraction",
    "model_params",
    *CALIBRATED_METRIC_COLUMNS,
)


def _validate_values(
    *,
    dataset_names: Sequence[str],
    seeds: Sequence[int],
    model_names: Sequence[str],
    label_noise_levels: Sequence[float],
    test_size: float,
) -> None:
    if (
        not dataset_names
        or any(not isinstance(name, str) or not name for name in dataset_names)
        or len(set(dataset_names)) != len(dataset_names)
    ):
        raise ValueError("dataset_names must be non-empty and unique.")
    unsupported = set(dataset_names).difference(DATASET_LOADERS)
    if unsupported:
        raise ValueError(
            f"Unsupported datasets: {sorted(unsupported)}. "
            f"Choose from: {list(DATASET_LOADERS)}."
        )
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= np.iinfo(np.uint32).max
        for seed in seeds
    ):
        raise ValueError("Seeds must be integers between 0 and 2**32 - 1.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    if (
        not model_names
        or any(not isinstance(name, str) or not name for name in model_names)
        or len(set(model_names)) != len(model_names)
    ):
        raise ValueError("Model names must be non-empty and unique.")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (float, int))
        or not np.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be between 0 and 1.")
    for level in label_noise_levels:
        if (
            isinstance(level, bool)
            or not isinstance(level, (int, float, np.number))
            or not np.isfinite(level)
            or not 0 <= level <= 1
        ):
            raise ValueError(
                "Label-noise levels must be finite numbers between 0 and 1."
            )
    if len(set(label_noise_levels)) != len(label_noise_levels):
        raise ValueError("Label-noise levels must be unique.")


def _feature_specs_for_data(
    data: pd.DataFrame,
    feature_shifts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if feature_shifts is None:
        return []
    if not isinstance(feature_shifts, Mapping):
        return validate_feature_shifts(feature_shifts)
    if not feature_shifts.get("enabled", True):
        return []

    types = feature_shifts.get("types", [])
    magnitudes = feature_shifts.get("magnitudes", [])
    features_by_type = feature_shifts.get("features")
    if not types or not magnitudes:
        raise ValueError(
            "Enabled feature shifts require non-empty types and magnitudes."
        )

    specs = []
    for shift_type in types:
        selected_features = (
            features_by_type.get(shift_type)
            if isinstance(features_by_type, Mapping)
            else features_by_type
        )
        if selected_features is None:
            if shift_type in {"gaussian_noise", "mean_shift", "variance_scaling"}:
                selected_features = [
                    column
                    for column in data.columns
                    if is_numeric_dtype(data[column].dtype)
                    and not is_bool_dtype(data[column].dtype)
                ]
            else:
                selected_features = [
                    column
                    for column in data.columns
                    if not is_numeric_dtype(data[column].dtype)
                    or is_bool_dtype(data[column].dtype)
                ]
        for magnitude in magnitudes:
            specs.append(
                {
                    "type": shift_type,
                    "features": list(selected_features),
                    "magnitude": magnitude,
                }
            )
    return validate_feature_shifts(specs)


def _evaluate(
    model, X_test, y_test, metric_evaluator: MetricEvaluator
) -> dict[str, float]:
    predictions = model.predict(X_test)
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError(
            f"{type(model).__name__} must expose predict_proba for raw ECE and "
            "Brier Score. CalibratedClassifierCV can fit from decision_function "
            "when available, but raw probability metrics require probabilities."
        )
    probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
    classes = list(model.classes_)
    if probabilities.ndim != 2 or probabilities.shape != (len(y_test), len(classes)):
        raise ValueError(
            "predict_proba must return one probability column per class and "
            "one row per evaluation observation."
        )
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Calibration evaluation requires classes 0 and 1.")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("predict_proba must return finite, normalized probabilities.")
    positive_index = classes.index(1)
    positive_probabilities = probabilities[:, positive_index]
    if metric_evaluator is evaluate_classifier and np.unique(y_test).size < 2:
        # ROC-AUC is undefined for one target class; preserve the remaining
        # shared metric definitions and record only AUC as unavailable.
        y_array = np.asarray(y_test)
        values = {
            "accuracy": accuracy_score(y_array, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_array, predictions),
            "macro_f1": f1_score(
                y_array,
                predictions,
                average="macro",
                labels=[0, 1],
                zero_division=0,
            ),
            "roc_auc": np.nan,
            "brier_score": np.mean((positive_probabilities - y_array) ** 2),
            "ece": expected_calibration_error(y_array, positive_probabilities),
            "adaptive_ece": adaptive_expected_calibration_error(
                y_array, positive_probabilities
            ),
            "maximum_calibration_error": maximum_calibration_error(
                y_array, positive_probabilities
            ),
            "log_loss": log_loss_score(y_array, positive_probabilities),
        }
    else:
        values = metric_evaluator(y_test, predictions, positive_probabilities)
    return {
        "accuracy": values.get("accuracy", np.nan),
        "balanced_accuracy": values.get("balanced_accuracy", np.nan),
        "macro_f1": values.get("macro_f1", np.nan),
        "roc_auc": values.get("roc_auc", np.nan),
        "brier_score": values.get("brier_score", np.nan),
        "ece": values.get("ece", np.nan),
        "adaptive_ece": values.get("adaptive_ece", np.nan),
        "mce": values.get("mce", values.get("maximum_calibration_error", np.nan)),
        "log_loss": values.get("log_loss", np.nan),
    }


def _run_robustness_benchmark(
    *,
    dataset_names: Sequence[str] | None = None,
    dataset_name: str | None = None,
    label_noise_levels: Sequence[float] = (),
    feature_shifts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    model_names: Sequence[str] = ("logistic_regression", "random_forest"),
    test_size: float = 0.2,
    output_path: str | Path | None = None,
    metric_evaluator: MetricEvaluator | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    calibration_settings: Mapping[str, Any] | None = None,
    dataset_data_home: str | Path | Mapping[str, str | Path] | None = None,
    _dataset_callback: Callable[[str, pd.DataFrame, pd.Series], None] | None = None,
    _result_callback: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Run clean, label-noise, and feature-shift evaluations in one result table.

    A common stratified split is created for every dataset/seed pair. A clean
    model is fit once per dataset/seed/model and reused for the baseline and
    all shifted-test evaluations. Positive label-noise levels fit separate
    models using corrupted copies of training labels only.

    ``feature_shifts`` accepts either validated per-shift mappings with
    ``type``, ``features``, and ``magnitude`` keys, or a configuration mapping
    with ``enabled``, ``types``, ``magnitudes``, and optional ``features``.
    Missing feature selections are inferred from data dtypes per shift type.
    """
    if dataset_names is not None and dataset_name is not None:
        raise ValueError("Pass dataset_names or dataset_name, not both.")
    datasets = (
        list(dataset_names)
        if dataset_names is not None
        else [dataset_name or "german_credit"]
    )
    levels = list(label_noise_levels)
    calibration = validate_calibration_settings(calibration_settings)
    _validate_values(
        dataset_names=datasets,
        seeds=seeds,
        model_names=model_names,
        label_noise_levels=levels,
        test_size=test_size,
    )

    evaluator = evaluate_classifier if metric_evaluator is None else metric_evaluator
    records: list[dict[str, Any]] = []

    for name in datasets:
        loader = DATASET_LOADERS[name]
        loader_data_home = (
            dataset_data_home.get(name)
            if isinstance(dataset_data_home, Mapping)
            else dataset_data_home
        )
        if loader_data_home is None:
            X, y = loader()
        else:
            X, y = loader(data_home=loader_data_home)
        if _dataset_callback is not None:
            _dataset_callback(name, X, y)
        for seed in seeds:
            split_seed = derive_seed(seed, "split", name)
            if calibration["enabled"]:
                calibration_split_seed = derive_seed(seed, "calibration_split", name)
                (
                    X_train,
                    X_calibration,
                    X_test,
                    y_train,
                    y_calibration,
                    y_test,
                ) = split_train_calibration_test(
                    X,
                    y,
                    split_seed=split_seed,
                    calibration_split_seed=calibration_split_seed,
                    train_size=calibration["train_size"],
                    calibration_size=calibration["calibration_size"],
                    test_size=calibration["test_size"],
                )
                n_calibration = len(X_calibration)
            else:
                calibration_split_seed = None
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=split_seed,
                    stratify=y,
                )
                X_calibration = y_calibration = None
                n_calibration = 0
            specs = _feature_specs_for_data(X_test, feature_shifts)
            shifted_tests = []
            for shift_index, spec in enumerate(specs):
                shift_seed = derive_seed(
                    seed,
                    "feature_shift",
                    name,
                    shift_index,
                    spec["type"],
                    float(spec["magnitude"]).hex(),
                )
                shifted_tests.append(
                    {
                        "spec": spec,
                        "X_test": apply_feature_shifts(X_test, [spec], seed=shift_seed),
                        "shift_seed": shift_seed,
                    }
                )

            for model_name in model_names:
                model_seed = derive_seed(seed, "model", name, model_name)
                params = model_params.get(model_name, {}) if model_params else {}
                baseline_started = perf_counter()
                baseline_model = build_model(
                    model_name=model_name,
                    features=X_train,
                    random_state=model_seed,
                    model_params=params,
                )
                baseline_model.fit(X_train, y_train)
                baseline_train_seconds = perf_counter() - baseline_started
                baseline_id = f"{name}:{model_name}:{seed}"

                baseline_calibrated_model = None
                baseline_calibrated_metrics = None
                baseline_calibration_error = None
                if calibration["enabled"]:
                    try:
                        baseline_calibrated_model = calibrate_fitted_classifier(
                            baseline_model,
                            X_calibration,
                            y_calibration,
                            method=calibration["method"],
                        )
                        baseline_calibrated_metrics = _evaluate(
                            baseline_calibrated_model,
                            X_test,
                            y_test,
                            evaluator,
                        )
                    except Exception as exc:
                        baseline_calibration_error = f"{type(exc).__name__}: {exc}"
                        baseline_calibrated_model = None
                        baseline_calibrated_metrics = {
                            metric: np.nan for metric in BENCHMARK_METRIC_COLUMNS
                        }

                # In calibration mode, the calibrator is fitted before either
                # raw or calibrated test metrics are evaluated.
                baseline_metrics = _evaluate(baseline_model, X_test, y_test, evaluator)

                def add_record(
                    experiment_type: str,
                    corruption_type: str,
                    corruption_level: float,
                    metrics: dict[str, float],
                    *,
                    affected_features: str = "",
                    n_corrupted: int = 0,
                    train_time_seconds: float | None = None,
                    shift_seed: int | None = None,
                    noise_seed: int | None = None,
                    model_seed_value: int = model_seed,
                    split_seed_value: int = split_seed,
                    dataset_value: str = name,
                    model_value: str = model_name,
                    seed_value: int = int(seed),
                    train_count: int = len(X_train),
                    test_count: int = len(y_test),
                    baseline_value: str = baseline_id,
                    condition_id_value: str = "",
                    calibrated_metrics_value: dict[str, float] | None = None,
                    calibration_method_value: str | None = (
                        calibration["method"] if calibration["enabled"] else None
                    ),
                    calibration_status_value: str = (
                        "succeeded" if calibration["enabled"] else "disabled"
                    ),
                    calibration_error_value: str | None = None,
                    calibration_split_seed_value: int | None = calibration_split_seed,
                    calibration_count: int = n_calibration,
                    train_fraction: float | None = (
                        calibration["train_size"] if calibration["enabled"] else None
                    ),
                    calibration_fraction: float | None = (
                        calibration["calibration_size"]
                        if calibration["enabled"]
                        else None
                    ),
                    test_fraction: float | None = (
                        calibration["test_size"] if calibration["enabled"] else None
                    ),
                ) -> None:
                    record = {
                        "dataset": dataset_value,
                        "model": model_value,
                        "experiment_type": experiment_type,
                        "corruption_type": corruption_type,
                        "corruption_level": float(corruption_level),
                        "seed": seed_value,
                        "split_seed": split_seed_value,
                        "noise_seed": noise_seed,
                        "model_seed": model_seed_value,
                        **metrics,
                        "n_train": train_count,
                        "n_test": test_count,
                        "baseline_id": baseline_value,
                        "affected_features": affected_features,
                        "n_corrupted": n_corrupted,
                        "train_time_seconds": train_time_seconds,
                        "shift_seed": shift_seed,
                        "condition_id": condition_id_value,
                        "calibration_method": calibration_method_value,
                        "calibration_status": calibration_status_value,
                        "calibration_error": calibration_error_value,
                        "calibration_split_seed": calibration_split_seed_value,
                        "n_calibration": calibration_count,
                        "split_train_fraction": train_fraction,
                        "split_calibration_fraction": calibration_fraction,
                        "split_test_fraction": test_fraction,
                        "model_params": json.dumps(params, sort_keys=True, default=str),
                        **{
                            f"calibrated_{metric}": (
                                calibrated_metrics_value.get(metric, np.nan)
                                if calibrated_metrics_value is not None
                                else np.nan
                            )
                            for metric in BENCHMARK_METRIC_COLUMNS
                        },
                    }
                    records.append(record)
                    if _result_callback is not None:
                        _result_callback(record.copy())

                add_record(
                    "clean_baseline",
                    "none",
                    0.0,
                    baseline_metrics,
                    train_time_seconds=baseline_train_seconds,
                    model_seed_value=model_seed,
                    condition_id_value=f"baseline:{baseline_id}",
                    calibrated_metrics_value=baseline_calibrated_metrics,
                    calibration_status_value=(
                        "failed"
                        if baseline_calibration_error is not None
                        else ("succeeded" if calibration["enabled"] else "disabled")
                    ),
                    calibration_error_value=baseline_calibration_error,
                )

                for level in levels:
                    noise_seed = derive_seed(
                        seed, "label_noise", name, float(level).hex()
                    )
                    if level == 0:
                        model = baseline_model
                        corrupted_count = 0
                        model_train_seconds = baseline_train_seconds
                        calibrated_model = baseline_calibrated_model
                        calibrated_metrics = baseline_calibrated_metrics
                        calibration_error = baseline_calibration_error
                    else:
                        noisy_y = inject_symmetric_label_noise(
                            y_train, float(level), seed=noise_seed
                        )
                        corrupted_count = int(
                            np.sum(np.asarray(noisy_y) != np.asarray(y_train))
                        )
                        model_started = perf_counter()
                        model = build_model(
                            model_name=model_name,
                            features=X_train,
                            random_state=model_seed,
                            model_params=params,
                        )
                        model.fit(X_train, noisy_y)
                        model_train_seconds = perf_counter() - model_started
                        calibrated_model = None
                        calibrated_metrics = None
                        calibration_error = None
                        if calibration["enabled"]:
                            try:
                                calibrated_model = calibrate_fitted_classifier(
                                    model,
                                    X_calibration,
                                    y_calibration,
                                    method=calibration["method"],
                                )
                                calibrated_metrics = _evaluate(
                                    calibrated_model,
                                    X_test,
                                    y_test,
                                    evaluator,
                                )
                            except Exception as exc:
                                calibration_error = f"{type(exc).__name__}: {exc}"
                                calibrated_model = None
                                calibrated_metrics = {
                                    metric: np.nan
                                    for metric in BENCHMARK_METRIC_COLUMNS
                                }
                    metrics = (
                        baseline_metrics
                        if level == 0
                        else _evaluate(model, X_test, y_test, evaluator)
                    )
                    condition_id = (
                        f"{name}:{model_name}:{float(level).hex()}:{int(seed)}"
                    )
                    add_record(
                        "label_noise",
                        "symmetric_label_noise",
                        float(level),
                        metrics,
                        n_corrupted=corrupted_count,
                        train_time_seconds=model_train_seconds,
                        noise_seed=noise_seed,
                        model_seed_value=model_seed,
                        condition_id_value=condition_id,
                        calibrated_metrics_value=calibrated_metrics,
                        calibration_status_value=(
                            "failed"
                            if calibration_error is not None
                            else ("succeeded" if calibration["enabled"] else "disabled")
                        ),
                        calibration_error_value=calibration_error,
                    )

                for shifted in shifted_tests:
                    spec = shifted["spec"]
                    metrics = _evaluate(
                        baseline_model, shifted["X_test"], y_test, evaluator
                    )
                    add_record(
                        "feature_shift",
                        spec["type"],
                        spec["magnitude"],
                        metrics,
                        affected_features=",".join(spec["features"]),
                        shift_seed=shifted["shift_seed"],
                        model_seed_value=model_seed,
                        calibration_method_value=None,
                        calibration_status_value=(
                            "not_applicable" if calibration["enabled"] else "disabled"
                        ),
                    )

    results = pd.DataFrame.from_records(records, columns=BENCHMARK_RESULT_COLUMNS)
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
    return results


def run_robustness_benchmark(
    *,
    dataset_names: Sequence[str] | None = None,
    dataset_name: str | None = None,
    label_noise_levels: Sequence[float] = (),
    feature_shifts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    model_names: Sequence[str] = ("logistic_regression", "random_forest"),
    test_size: float = 0.2,
    output_path: str | Path | None = None,
    metric_evaluator: MetricEvaluator | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    calibration_settings: Mapping[str, Any] | None = None,
    dataset_data_home: str | Path | Mapping[str, str | Path] | None = None,
    tracking_dir: str | Path | None = None,
    experiment_name: str | None = None,
    original_config: Any = None,
    normalized_config: Any = None,
    _result_callback: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Run the robustness benchmark, optionally recording a local experiment.

    Tracking is disabled when ``tracking_dir`` is omitted. Tracked runs save
    the configuration, environment and dataset provenance, results, a report,
    and a manifest containing completion or failure status.
    """
    datasets = (
        list(dataset_names)
        if dataset_names is not None
        else [dataset_name or "german_credit"]
    )
    if original_config is None:
        original_config = {
            "dataset_names": datasets,
            "label_noise_levels": list(label_noise_levels),
            "feature_shifts": feature_shifts,
            "seeds": list(seeds),
            "models": list(model_names),
            "model_params": model_params or {},
            "calibration_settings": calibration_settings,
            "dataset_data_home": dataset_data_home,
            "test_size": test_size,
            "output_path": output_path,
            "metric_evaluator": (
                f"{metric_evaluator.__module__}.{metric_evaluator.__qualname__}"
                if metric_evaluator is not None
                and hasattr(metric_evaluator, "__module__")
                and hasattr(metric_evaluator, "__qualname__")
                else None
            ),
        }
    if normalized_config is None:
        normalized_config = original_config

    tracker = None
    callback = None
    if tracking_dir is not None:
        from trusttab.experiments.tracking import _RunTracker

        tracker = _RunTracker(
            tracking_dir,
            experiment_name=experiment_name,
            original_config=original_config,
            normalized_config=normalized_config,
            dataset_names=datasets,
            models=list(model_names),
            seeds=list(seeds),
        )
        callback = tracker.record_dataset

    try:
        results = _run_robustness_benchmark(
            dataset_names=dataset_names,
            dataset_name=dataset_name,
            label_noise_levels=label_noise_levels,
            feature_shifts=feature_shifts,
            seeds=seeds,
            model_names=model_names,
            test_size=test_size,
            output_path=output_path,
            metric_evaluator=metric_evaluator,
            model_params=model_params,
            calibration_settings=calibration_settings,
            dataset_data_home=dataset_data_home,
            _dataset_callback=callback,
            _result_callback=_result_callback,
        )
        if tracker is not None:
            tracker.complete(results)
            results.attrs["run_id"] = tracker.run_id
            results.attrs["run_dir"] = str(tracker.run_dir)
        return results
    except Exception as exc:
        if tracker is not None:
            tracker.fail(exc)
        raise
