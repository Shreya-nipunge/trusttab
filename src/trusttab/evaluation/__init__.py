"""Evaluation utilities for TrustTab."""

from trusttab.evaluation.calibration import (
    calibrate_classifier,
    calibrate_fitted_classifier,
    get_calibration_curve,
)
from trusttab.evaluation.metrics import (
    evaluate_classifier,
    expected_calibration_error,
)
from trusttab.evaluation.operating_region import (
    OperatingPoint,
    analyze_operating_region,
)

__all__ = [
    "OperatingPoint",
    "analyze_operating_region",
    "calibrate_classifier",
    "calibrate_fitted_classifier",
    "evaluate_classifier",
    "expected_calibration_error",
    "get_calibration_curve",
]