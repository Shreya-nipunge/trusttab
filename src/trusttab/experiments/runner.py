
"""Reproducible label-noise stress-test experiment runner."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trusttab.data import load_adult, load_german_credit
from trusttab.models import build_model
from trusttab.noise import inject_symmetric_label_noise


MetricEvaluator = Callable[
    [pd.Series | np.ndarray, np.ndarray, np.ndarray],
    dict[str, float],
]


DATASET_LOADERS = {
    "german_credit": load_german_credit,
    "adult": load_adult,
}


def run_stress_test(
    *,
    noise_levels: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    model_names: Sequence[str] = (
        "logistic_regression",
        "random_forest",
    ),
    test_size: float = 0.2,
    dataset_name: str = "german_credit",
    output_path: str | Path | None = None,
    metric_evaluator: MetricEvaluator | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Run a multi-seed label-noise stress test on a supported dataset.

    The test split is clean. Only training labels are corrupted.

    If metric_evaluator is provided, it receives:
        y_test, predictions, positive-class probabilities.

    If no evaluator is provided, basic metrics are calculated locally
    as a temporary fallback until the shared B0 evaluator is integrated.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not noise_levels:
        raise ValueError("At least one noise level is required.")

    if not seeds:
        raise ValueError("At least one random seed is required.")

    if not model_names:
        raise ValueError("At least one model must be selected.")

    if any(not 0 <= rate <= 1 for rate in noise_levels):
        raise ValueError("Noise levels must be between 0 and 1.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    if len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be unique.")

    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_LOADERS)}."
        )

    X, y = DATASET_LOADERS[dataset_name]()

    results = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        for noise_rate in noise_levels:
            noisy_y_train = inject_symmetric_label_noise(
                y_train,
                noise_rate=noise_rate,
                seed=seed,
            )

            corrupted_count = int(
                np.sum(
                    np.asarray(noisy_y_train) != np.asarray(y_train)
                )
            )

            for model_name in model_names:
                params = (
                    model_params.get(model_name, {})
                    if model_params is not None
                    else {}
                )

                model = build_model(
                    model_name=model_name,
                    features=X_train,
                    random_state=seed,
                    model_params=params,
                )

                model.fit(X_train, noisy_y_train)

                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)

                positive_class_index = list(model.classes_).index(1)
                positive_probabilities = probabilities[
                    :, positive_class_index
                ]

                if metric_evaluator is not None:
                    metrics = metric_evaluator(
                        y_test,
                        predictions,
                        positive_probabilities,
                    )
                else:
                    from sklearn.metrics import (
                        accuracy_score,
                        balanced_accuracy_score,
                        f1_score,
                        roc_auc_score,
                    )

                    metrics = {
                        "accuracy": accuracy_score(y_test, predictions),
                        "balanced_accuracy": balanced_accuracy_score(
                            y_test, predictions
                        ),
                        "macro_f1": f1_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0,
                        ),
                        "roc_auc": roc_auc_score(
                            y_test,
                            positive_probabilities,
                        ),
                    }

                results.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "noise_rate": float(noise_rate),
                        "seed": int(seed),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "n_corrupted": corrupted_count,
                        **metrics,
                    }
                )

    results_df = pd.DataFrame(results)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)

    return results_df
