"""Leakage-aware train/calibration/test splitting for binary experiments."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trusttab.experiments.calibration_protocol import validate_calibration_settings


def split_train_calibration_test(
    X: Any,
    y: Any,
    *,
    split_seed: int,
    calibration_split_seed: int,
    train_size: float = 0.70,
    calibration_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Create disjoint, stratified train, calibration, and test partitions.

    Returns ``X_train, X_calibration, X_test, y_train, y_calibration,
    y_test``. The split uses two independently derived seed streams so both
    stages can be replayed and audited separately.
    """

    proportions = validate_calibration_settings(
        {
            "enabled": True,
            "train_size": train_size,
            "calibration_size": calibration_size,
            "test_size": test_size,
        }
    )
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("Calibration splitting requires one-dimensional labels.")
    if len(X) != len(labels):
        raise ValueError("X and y must contain the same number of observations.")
    if len(labels) == 0 or pd.isna(labels).any():
        raise ValueError(
            "Calibration splitting requires non-empty labels without NaNs."
        )
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Calibration splitting requires both binary classes 0 and 1.")
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("Calibration splitting requires binary labels 0 and 1.")
    if np.min(counts) < 3:
        raise ValueError(
            "Each binary class needs at least three observations for stratified "
            "train/calibration/test partitions."
        )

    holdout_size = proportions["calibration_size"] + proportions["test_size"]
    try:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            train_size=proportions["train_size"],
            test_size=holdout_size,
            random_state=split_seed,
            stratify=y,
        )
        calibration_share = proportions["calibration_size"] / holdout_size
        test_share = proportions["test_size"] / holdout_size
        X_calibration, X_test, y_calibration, y_test = train_test_split(
            X_holdout,
            y_holdout,
            train_size=calibration_share,
            test_size=test_share,
            random_state=calibration_split_seed,
            stratify=y_holdout,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create stratified train/calibration/test partitions; "
            "increase the per-class sample counts or adjust the split proportions."
        ) from exc

    for split_name, split_labels in (
        ("training", y_train),
        ("calibration", y_calibration),
        ("test", y_test),
    ):
        if set(np.unique(split_labels).tolist()) != {0, 1}:
            raise ValueError(
                f"The {split_name} partition must contain both binary classes."
            )

    return X_train, X_calibration, X_test, y_train, y_calibration, y_test
