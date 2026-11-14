"""Experiment runner for feature-distribution shifts on held-out data."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trusttab.evaluation import evaluate_classifier
from trusttab.experiments.runner import DATASET_LOADERS, MetricEvaluator
from trusttab.experiments.seeding import derive_seed
from trusttab.models import build_model
from trusttab.noise import apply_feature_shifts, validate_feature_shifts


def run_feature_shift_test(
    *,
    feature_shifts: Sequence[dict[str, Any]],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    model_names: Sequence[str] = ("logistic_regression", "random_forest"),
    test_size: float = 0.2,
    dataset_name: str = "german_credit",
    output_path: str | Path | None = None,
    metric_evaluator: MetricEvaluator | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Train on original data and evaluate clean and feature-shifted test sets.

    Each feature-shift specification is applied independently to the clean
    held-out split. One clean baseline row and one row per shift are recorded
    for every model and seed.
    """
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not np.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be between 0 and 1.")
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= np.iinfo(np.uint32).max
        for seed in seeds
    ):
        raise ValueError("Every seed must be an integer between 0 and 2**32 - 1.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    if (
        not model_names
        or any(not isinstance(name, str) or not name for name in model_names)
        or len(set(model_names)) != len(model_names)
    ):
        raise ValueError("Model names must be non-empty and unique.")
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_LOADERS)}."
        )
    shifts = validate_feature_shifts(feature_shifts)
    X, y = DATASET_LOADERS[dataset_name]()
    evaluator = evaluate_classifier if metric_evaluator is None else metric_evaluator
    records = []

    for seed in seeds:
        split_seed = derive_seed(seed, "split", dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=split_seed,
            stratify=y,
        )
        scenarios = [
            {
                "scenario": "clean",
                "shift_index": -1,
                "shift_type": "none",
                "affected_feature": "",
                "shift_magnitude": 0.0,
                "shift_seed": None,
                "X_test": X_test,
            }
        ]
        for index, spec in enumerate(shifts):
            shift_seed = derive_seed(
                seed,
                "feature_shift",
                dataset_name,
                index,
                spec["type"],
                float(spec["magnitude"]).hex(),
            )
            scenarios.append(
                {
                    "scenario": "shifted",
                    "shift_index": index,
                    "shift_type": spec["type"],
                    "affected_feature": ",".join(spec["features"]),
                    "shift_magnitude": spec["magnitude"],
                    "shift_seed": shift_seed,
                    "X_test": apply_feature_shifts(X_test, [spec], seed=shift_seed),
                }
            )

        for model_name in model_names:
            model_seed = derive_seed(seed, "model", dataset_name, model_name)
            params = model_params.get(model_name, {}) if model_params else {}
            model = build_model(
                model_name=model_name,
                features=X_train,
                random_state=model_seed,
                model_params=params,
            )
            model.fit(X_train, y_train)
            positive_class_index = list(model.classes_).index(1)

            for scenario in scenarios:
                predictions = model.predict(scenario["X_test"])
                probabilities = model.predict_proba(scenario["X_test"])
                positive_probabilities = probabilities[:, positive_class_index]
                metrics = evaluator(y_test, predictions, positive_probabilities)
                records.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": int(seed),
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "scenario": scenario["scenario"],
                        "shift_index": scenario["shift_index"],
                        "shift_type": scenario["shift_type"],
                        "affected_feature": scenario["affected_feature"],
                        "shift_magnitude": scenario["shift_magnitude"],
                        "shift_seed": scenario["shift_seed"],
                        "n_train": len(X_train),
                        "n_test": len(y_test),
                        **metrics,
                    }
                )

    results = pd.DataFrame.from_records(records)
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
    return results
