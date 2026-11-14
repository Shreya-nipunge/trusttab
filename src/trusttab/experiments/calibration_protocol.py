"""Configuration and data-splitting helpers for calibration experiments."""

from collections.abc import Mapping
from typing import Any

import numpy as np

DEFAULT_CALIBRATION_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "method": "sigmoid",
    "train_size": 0.70,
    "calibration_size": 0.15,
    "test_size": 0.15,
}


def validate_calibration_settings(settings: Any = None) -> dict[str, Any]:
    """Validate and normalize Phase 4 calibration settings.

    Calibration is opt-in. When enabled, the blueprint's stratified
    train/calibration/test proportions (70/15/15) are used by default.
    """

    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise TypeError("experiments.calibration must be a YAML mapping.")

    unexpected = set(settings).difference(DEFAULT_CALIBRATION_SETTINGS)
    if unexpected:
        raise ValueError(
            "Unsupported calibration settings: "
            f"{sorted(unexpected)}. Supported keys: "
            f"{sorted(DEFAULT_CALIBRATION_SETTINGS)}."
        )

    normalized = {**DEFAULT_CALIBRATION_SETTINGS, **settings}
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("experiments.calibration.enabled must be a boolean.")
    if normalized["method"] not in {"sigmoid", "platt"}:
        raise ValueError(
            "experiments.calibration.method must be 'sigmoid' (Platt scaling)."
        )
    normalized["method"] = "sigmoid"

    proportion_names = ("train_size", "calibration_size", "test_size")
    proportions = []
    for name in proportion_names:
        value = normalized[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(
                f"experiments.calibration.{name} must be finite and between 0 and 1."
            )
        normalized[name] = float(value)
        proportions.append(float(value))

    if not np.isclose(sum(proportions), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            "experiments.calibration train_size, calibration_size, and "
            "test_size must sum to 1."
        )
    return normalized
