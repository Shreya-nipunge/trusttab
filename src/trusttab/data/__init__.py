"""Dataset loading utilities for TrustTab."""

from trusttab.data.preprocessing import build_preprocessor

from trusttab.data.german_credit import (
    load_german_credit,
    validate_german_credit_data,
)

from trusttab.data.adult import (
    load_adult,
    validate_adult_data,
)

__all__ = [
    "load_german_credit",
    "validate_german_credit_data",
    "load_adult",
    "validate_adult_data",
    "build_preprocessor",
]