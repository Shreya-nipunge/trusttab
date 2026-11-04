"""Statistical comparison utilities for TrustTab."""

from trusttab.statistics.comparison import (
    ComparisonResult,
    MultipleComparisonResult,
    PairedTestResult,
    adjust_p_values,
    analyze_noise_levels,
    compare_models,
    compare_paired_scores,
    compare_paired_scores_wilcoxon,
)

__all__ = [
    "ComparisonResult",
    "MultipleComparisonResult",
    "PairedTestResult",
    "adjust_p_values",
    "analyze_noise_levels",
    "compare_models",
    "compare_paired_scores",
    "compare_paired_scores_wilcoxon",
]
