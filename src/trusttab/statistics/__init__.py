"""Statistical comparison utilities for TrustTab."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a paired statistical comparison."""

    statistic: float
    p_value: float
    mean_difference: float
    confidence_interval: tuple[float, float]


def compare_paired_scores(
    scores_a,
    scores_b,
    confidence_level: float = 0.95,
) -> ComparisonResult:
    """Compare paired model scores using a paired t-test.

    A positive mean_difference means scores_a are higher than scores_b.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Score arrays must be one-dimensional.")
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Score arrays must have equal lengths of at least 2.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    differences = a - b
    result = stats.ttest_rel(a, b)
    mean_difference = float(np.mean(differences))
    standard_error = float(stats.sem(differences))
    critical_value = float(
        stats.t.ppf(
            (1 + confidence_level) / 2,
            df=len(differences) - 1,
        )
    )
    margin = critical_value * standard_error

    return ComparisonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        mean_difference=mean_difference,
        confidence_interval=(
            mean_difference - margin,
            mean_difference + margin,
        ),
    )