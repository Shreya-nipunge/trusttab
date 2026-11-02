"""Evaluation utilities for TrustTab."""

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
    "evaluate_classifier",
    "expected_calibration_error",
]