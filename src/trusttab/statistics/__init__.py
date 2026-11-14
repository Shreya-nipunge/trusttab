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
from trusttab.statistics.research import (
    analyze_friedman_nemenyi,
    calibration_wilcoxon_results,
    compare_degradation_slopes,
    degradation_slope_results,
    descriptive_summary,
    rank_biserial_correlation,
    run_research_analysis,
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
    "analyze_friedman_nemenyi",
    "calibration_wilcoxon_results",
    "compare_degradation_slopes",
    "degradation_slope_results",
    "descriptive_summary",
    "rank_biserial_correlation",
    "run_research_analysis",
]
