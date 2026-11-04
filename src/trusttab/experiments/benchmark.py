"""Unified robustness benchmark orchestration."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from trusttab.evaluation import (
    adaptive_expected_calibration_error,
    evaluate_classifier,
    expected_calibration_error,
    log_loss_score,
    maximum_calibration_error,
)
from trusttab.experiments.runner import DATASET_LOADERS, MetricEvaluator
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

BENCHMARK_RESULT_COLUMNS = (
    "dataset",
    "model",
    "experiment_type",
    "corruption_type",
    "corruption_level",
    "seed",
    *BENCHMARK_METRIC_COLUMNS,
    "n_train",
    "n_test",
    "baseline_id",
    "affected_features",
    "n_corrupted",
    "shift_seed",
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
    probabilities = model.predict_proba(X_test)
    positive_index = list(model.classes_).index(1)
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
    _dataset_callback: Callable[[str, pd.DataFrame, pd.Series], None] | None = None,
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
        X, y = DATASET_LOADERS[name]()
        if _dataset_callback is not None:
            _dataset_callback(name, X, y)
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=int(seed),
                stratify=y,
            )
            specs = _feature_specs_for_data(X_test, feature_shifts)
            shifted_tests = []
            for shift_index, spec in enumerate(specs):
                shift_seed = int(
                    np.random.SeedSequence([int(seed), shift_index]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                shifted_tests.append(
                    {
                        "spec": spec,
                        "X_test": apply_feature_shifts(X_test, [spec], seed=shift_seed),
                        "shift_seed": shift_seed,
                    }
                )

            for model_name in model_names:
                params = model_params.get(model_name, {}) if model_params else {}
                baseline_model = build_model(
                    model_name=model_name,
                    features=X_train,
                    random_state=int(seed),
                    model_params=params,
                )
                baseline_model.fit(X_train, y_train)
                baseline_metrics = _evaluate(baseline_model, X_test, y_test, evaluator)
                baseline_id = f"{name}:{model_name}:{seed}"

                def add_record(
                    experiment_type: str,
                    corruption_type: str,
                    corruption_level: float,
                    metrics: dict[str, float],
                    *,
                    affected_features: str = "",
                    n_corrupted: int = 0,
                    shift_seed: int | None = None,
                    dataset_value: str = name,
                    model_value: str = model_name,
                    seed_value: int = int(seed),
                    train_count: int = len(X_train),
                    test_count: int = len(y_test),
                    baseline_value: str = baseline_id,
                ) -> None:
                    records.append(
                        {
                            "dataset": dataset_value,
                            "model": model_value,
                            "experiment_type": experiment_type,
                            "corruption_type": corruption_type,
                            "corruption_level": float(corruption_level),
                            "seed": seed_value,
                            **metrics,
                            "n_train": train_count,
                            "n_test": test_count,
                            "baseline_id": baseline_value,
                            "affected_features": affected_features,
                            "n_corrupted": n_corrupted,
                            "shift_seed": shift_seed,
                        }
                    )

                add_record("clean_baseline", "none", 0.0, baseline_metrics)

                for level in levels:
                    if level == 0:
                        model = baseline_model
                        corrupted_count = 0
                    else:
                        noisy_y = inject_symmetric_label_noise(
                            y_train, float(level), seed=int(seed)
                        )
                        corrupted_count = int(
                            np.sum(np.asarray(noisy_y) != np.asarray(y_train))
                        )
                        model = build_model(
                            model_name=model_name,
                            features=X_train,
                            random_state=int(seed),
                            model_params=params,
                        )
                        model.fit(X_train, noisy_y)
                    metrics = (
                        baseline_metrics
                        if level == 0
                        else _evaluate(model, X_test, y_test, evaluator)
                    )
                    add_record(
                        "label_noise",
                        "symmetric_label_noise",
                        float(level),
                        metrics,
                        n_corrupted=corrupted_count,
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
    tracking_dir: str | Path | None = None,
    experiment_name: str | None = None,
    original_config: Any = None,
    normalized_config: Any = None,
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
            _dataset_callback=callback,
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
