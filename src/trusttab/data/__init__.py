"""Dataset loading utilities for TrustTab."""

from trusttab.data.german_credit import (
    load_german_credit,
    validate_german_credit_data,
)

__all__ = [
    "load_german_credit",
    "validate_german_credit_data",
]