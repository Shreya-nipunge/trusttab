"""Paired statistical tests and model comparison for experiment results."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ComparisonResult:
    """Paired t-test result with a confidence interval for mean difference."""

    statistic: float
    p_value: float
    mean_difference: float
    confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired test; differences are always scores_a minus scores_b."""

    statistic: float
    p_value: float
    mean_difference: float
    n_observations: int
    test_name: str


@dataclass(frozen=True)
class MultipleComparisonResult:
    """Adjusted p-values and decisions from a family-wise correction."""

    adjusted_p_values: np.ndarray
    reject_null: np.ndarray
    method: str
    alpha: float


def _validate_paired_scores(scores_a, scores_b) -> tuple[np.ndarray, np.ndarray]:
    try:
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scores must be numeric one-dimensional arrays.") from exc
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Score arrays must be one-dimensional.")
    if len(a) != len(b):
        raise ValueError("Score arrays must have equal lengths.")
    if len(a) < 2:
        raise ValueError("At least 2 paired observations are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite.")
    return a, b


def compare_paired_scores(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> ComparisonResult:
    """Compare paired scores with a two-sided paired t-test.

    A positive mean difference means ``scores_a`` is higher. The test assumes
    independent pairs and approximately normally distributed pairwise
    differences. At least two pairs are required.
    """
    a, b = _validate_paired_scores(scores_a, scores_b)
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be between 0 and 1.")

    differences = a - b
    mean_difference = float(np.mean(differences))
    constant_differences = np.allclose(
        differences, differences[0], rtol=1e-12, atol=1e-15
    )
    if constant_differences:
        if np.isclose(differences[0], 0.0, rtol=0.0, atol=1e-15):
            statistic, p_value = 0.0, 1.0
        else:
            statistic = float(np.copysign(np.inf, differences[0]))
            p_value = 0.0
        interval = (mean_difference, mean_difference)
    else:
        result = stats.ttest_rel(a, b)
        standard_error = float(stats.sem(differences))
        critical_value = float(
            stats.t.ppf(
                (1 + confidence_level) / 2,
                df=len(differences) - 1,
            )
        )
        margin = critical_value * standard_error
        statistic, p_value = float(result.statistic), float(result.pvalue)
        interval = (mean_difference - margin, mean_difference + margin)

    return ComparisonResult(
        statistic=statistic,
        p_value=p_value,
        mean_difference=mean_difference,
        confidence_interval=interval,
    )


def compare_paired_scores_wilcoxon(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
) -> PairedTestResult:
    """Compare paired scores with a two-sided Wilcoxon signed-rank test.

    This non-parametric test works on the ranks of non-zero paired differences
    and assumes independent pairs with differences symmetric about zero. At
    least two paired observations are required. Identical pairs return p=1.
    """
    a, b = _validate_paired_scores(scores_a, scores_b)
    differences = a - b
    if np.all(differences == 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(
            a,
            b,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return PairedTestResult(
        statistic=statistic,
        p_value=p_value,
        mean_difference=float(np.mean(differences)),
        n_observations=len(a),
        test_name="wilcoxon",
    )


def adjust_p_values(
    p_values: Sequence[float],
    method: Literal[
        "bonferroni",
        "holm",
        "holm-bonferroni",
        "fdr_bh",
        "benjamini-hochberg",
    ] = "holm",
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Adjust p-values by Bonferroni, Holm, or Benjamini-Hochberg.

    ``fdr_bh`` controls the expected false discovery rate; Bonferroni and Holm
    control family-wise error. Returned decisions reject when adjusted p is at
    most ``alpha``.
    """
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must be finite and between 0 and 1.")
    aliases = {
        "bonferroni": "bonferroni",
        "holm": "holm",
        "holm_bonferroni": "holm",
        "fdr_bh": "fdr_bh",
        "benjamini_hochberg": "fdr_bh",
    }
    normalized_method = (
        method.lower().replace("-", "_") if isinstance(method, str) else ""
    )
    if normalized_method not in aliases:
        raise ValueError(
            "method must be 'bonferroni', 'holm', or 'benjamini-hochberg'."
        )
    method = aliases[normalized_method]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (float, int))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("alpha must be between 0 and 1.")

    count = len(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    if method == "bonferroni":
        adjusted_sorted = np.minimum(sorted_values * count, 1.0)
    elif method == "holm":
        scaled = sorted_values * np.arange(count, 0, -1)
        adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    else:
        scaled = sorted_values * count / np.arange(1, count + 1)
        adjusted_sorted = np.minimum(
            np.minimum.accumulate(scaled[::-1])[::-1], 1.0
        )

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return MultipleComparisonResult(
        adjusted_p_values=adjusted,
        reject_null=adjusted <= alpha,
        method=method,
        alpha=float(alpha),
    )


def _validate_result_table(
    results: pd.DataFrame,
    metric: str,
    models: Sequence[str] | None,
    *,
    require_two_models: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty column name.")
    required = {"model", "noise_rate", "seed", metric}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("results must contain at least one row.")

    if results["model"].isna().any():
        raise ValueError("model values must not be missing.")
    selected = (
        sorted(pd.unique(results["model"]).tolist())
        if models is None
        else list(models)
    )
    if len(selected) < (2 if require_two_models else 1):
        raise ValueError("At least two distinct models are required." if require_two_models else "At least one model is required.")
    if len(set(selected)) != len(selected):
        raise ValueError("models must contain unique model names.")
    available = set(results["model"].dropna())
    absent = set(selected).difference(available)
    if absent:
        raise ValueError(f"Unknown model(s): {sorted(absent)}")

    data = results.loc[results["model"].isin(selected)].copy()
    data["noise_rate"] = pd.to_numeric(data["noise_rate"], errors="coerce")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    if (
        data["model"].isna().any()
        or data["seed"].isna().any()
        or not np.isfinite(data["noise_rate"]).all()
        or not np.isfinite(data[metric]).all()
    ):
        raise ValueError("Model, seed, noise-rate, and metric values must be valid and finite.")
    keys = (["dataset"] if "dataset" in data.columns else []) + ["seed"]
    if data.duplicated(["model", "noise_rate", *keys]).any():
        raise ValueError("Each model/configuration must have at most one metric value.")
    return data, selected, keys


def _paired_test(a: np.ndarray, b: np.ndarray, test: str) -> PairedTestResult:
    if test == "ttest":
        result = compare_paired_scores(a, b)
        return PairedTestResult(
            statistic=result.statistic,
            p_value=result.p_value,
            mean_difference=result.mean_difference,
            n_observations=len(a),
            test_name="paired_ttest",
        )
    if test == "wilcoxon":
        return compare_paired_scores_wilcoxon(a, b)
    raise ValueError("test must be 'ttest' or 'wilcoxon'.")


def compare_models(
    results: pd.DataFrame,
    metric: str,
    *,
    models: Sequence[str] | None = None,
    test: Literal["ttest", "wilcoxon"] = "ttest",
    unmatched: Literal["raise", "drop"] = "raise",
    correction: Literal["bonferroni", "holm", "fdr_bh"] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare model pairs on matched seeds within each noise level.

    By default all available models are compared pairwise. Pair keys are seed
    and, when present, dataset. Unequal configurations raise by default;
    ``unmatched="drop"`` permits an inner join and reports unmatched counts.
    Differences are ``model_a - model_b``. Corrected p-values, if requested,
    are adjusted across all valid pair/noise tests in this result.
    """
    if test not in {"ttest", "wilcoxon"}:
        raise ValueError("test must be 'ttest' or 'wilcoxon'.")
    if unmatched not in {"raise", "drop"}:
        raise ValueError("unmatched must be 'raise' or 'drop'.")
    if correction is not None:
        # Validate method and alpha even when no test can be computed.
        adjust_p_values([0.5], correction, alpha)
    data, selected, keys = _validate_result_table(results, metric, models)
    model_pairs = list(combinations(selected, 2))
    records = []

    for noise_rate in sorted(data["noise_rate"].unique()):
        at_noise = data.loc[data["noise_rate"] == noise_rate]
        for model_a, model_b in model_pairs:
            left = at_noise.loc[at_noise["model"] == model_a, [*keys, metric]]
            right = at_noise.loc[at_noise["model"] == model_b, [*keys, metric]]
            paired = left.merge(
                right,
                on=keys,
                how="inner",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            unmatched_a = len(left) - len(paired)
            unmatched_b = len(right) - len(paired)
            if unmatched == "raise" and (unmatched_a or unmatched_b):
                raise ValueError(
                    f"Unmatched configurations at noise_rate={noise_rate} for "
                    f"{model_a} vs {model_b}: {unmatched_a} and {unmatched_b}. "
                    "Use unmatched='drop' to compare only shared configurations."
                )

            values_a = paired[f"{metric}_a"].to_numpy(dtype=float)
            values_b = paired[f"{metric}_b"].to_numpy(dtype=float)
            if len(paired) >= 2:
                outcome = _paired_test(values_a, values_b, test)
                statistic, p_value = outcome.statistic, outcome.p_value
                mean_difference = outcome.mean_difference
                test_name = outcome.test_name
            else:
                statistic = p_value = mean_difference = np.nan
                test_name = test
            records.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "noise_level": float(noise_rate),
                    "n_paired": len(paired),
                    "n_unmatched_a": unmatched_a,
                    "n_unmatched_b": unmatched_b,
                    "mean_a": float(np.mean(values_a)) if len(values_a) else np.nan,
                    "std_a": float(np.std(values_a, ddof=1)) if len(values_a) > 1 else np.nan,
                    "mean_b": float(np.mean(values_b)) if len(values_b) else np.nan,
                    "std_b": float(np.std(values_b, ddof=1)) if len(values_b) > 1 else np.nan,
                    "mean_difference": mean_difference,
                    "difference_definition": "model_a - model_b",
                    "test": test_name,
                    "status": "ok" if len(paired) >= 2 else "insufficient_observations",
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

    comparison = pd.DataFrame.from_records(records)
    comparison["adjusted_p_value"] = np.nan
    comparison["reject_null"] = pd.Series(pd.NA, index=comparison.index, dtype="boolean")
    valid = comparison["p_value"].notna()
    if correction is not None and valid.any():
        adjusted = adjust_p_values(comparison.loc[valid, "p_value"], correction, alpha)
        comparison.loc[valid, "adjusted_p_value"] = adjusted.adjusted_p_values
        comparison.loc[valid, "reject_null"] = adjusted.reject_null
    return comparison


def analyze_noise_levels(
    results: pd.DataFrame,
    metric: str,
    *,
    reference_model: str | None = None,
    test: Literal["ttest", "wilcoxon"] = "ttest",
    unmatched: Literal["raise", "drop"] = "raise",
) -> pd.DataFrame:
    """Summarize performance by model and noise, optionally against a reference.

    With ``reference_model``, inferential columns compare each model's matched
    scores against that reference at the same noise level. Without a reference,
    ``n_paired`` is the observed replicate count and test columns are null.
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    if test not in {"ttest", "wilcoxon"}:
        raise ValueError("test must be 'ttest' or 'wilcoxon'.")
    if unmatched not in {"raise", "drop"}:
        raise ValueError("unmatched must be 'raise' or 'drop'.")
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty column name.")
    required = {"model", "noise_rate", "seed", metric}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")
    if results["model"].isna().any():
        raise ValueError("model values must not be missing.")
    selected = sorted(pd.unique(results["model"]).tolist())
    if reference_model is not None and reference_model not in selected:
        raise ValueError(f"Unknown reference_model: {reference_model}")
    data, selected, _keys = _validate_result_table(
        results, metric, selected, require_two_models=False
    )

    records = []
    for (model, noise_rate), group in data.groupby(["model", "noise_rate"], sort=True):
        reference_n = len(group)
        statistic = p_value = mean_difference = np.nan
        test_name = test if reference_model is not None else None
        if reference_model is not None and model != reference_model:
            paired = compare_models(
                results,
                metric,
                models=[model, reference_model],
                test=test,
                unmatched=unmatched,
            )
            match = paired.loc[paired["noise_level"] == noise_rate].iloc[0]
            reference_n = int(match["n_paired"])
            statistic, p_value = match["statistic"], match["p_value"]
            mean_difference = match["mean_difference"]
            test_name = match["test"]
        records.append(
            {
                "model": model,
                "metric": metric,
                "noise_level": float(noise_rate),
                "n_paired": reference_n,
                "mean_performance": float(group[metric].mean()),
                "std_performance": float(group[metric].std()),
                "reference_model": reference_model,
                "mean_difference": mean_difference,
                "difference_definition": (
                    f"{model} - {reference_model}" if reference_model else None
                ),
                "test": test_name,
                "statistic": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame.from_records(records)
