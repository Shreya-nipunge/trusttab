"""Seed-blocked inferential analyses for the TrustTab research matrix.

All analyses keep datasets separate. Seeds are the repeated-measures unit;
individual test observations are never treated as independent replicates.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
import json
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from trusttab.statistics.comparison import adjust_p_values

CORE_MODELS = ("logistic_regression", "random_forest", "xgboost", "tabpfn")
CORE_NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
CORE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "ece",
    "brier_score",
)
HIGHER_IS_BETTER = frozenset(CORE_METRICS[:4])
LOWER_IS_BETTER = frozenset(CORE_METRICS[4:])
BOOTSTRAP_RESAMPLES = 10_000


def bootstrap_seed(*keys: object) -> int:
    """Derive a stable seed for a report interval from its condition keys."""
    payload = json.dumps([str(key) for key in keys], separators=(",", ":"))
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little"
    )


def rank_biserial_correlation(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial effect; positive means differences > 0."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite one-dimensional array")
    nonzero = values[values != 0]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def _noise_rows(results: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "seed", "corruption_level"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")
    data = results.copy()
    if "experiment_type" in data:
        data = data.loc[data["experiment_type"] == "label_noise"].copy()
    data["corruption_level"] = pd.to_numeric(data["corruption_level"], errors="coerce")
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
    return data


def analyze_friedman_nemenyi(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = CORE_METRICS,
    models: Sequence[str] = CORE_MODELS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    noise_levels: Sequence[float] = CORE_NOISE_LEVELS,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman tests within dataset/noise/metric, then gated Nemenyi pairs.

    Holm families are the five noise-level omnibus tests within each
    dataset/metric. Nemenyi p-values are already adjusted across model pairs.
    Incomplete or non-finite blocks are reported and never silently paired.
    """
    data = _noise_rows(results)
    model_names, seed_ids = tuple(models), tuple(seeds)
    if len(model_names) < 3 or len(set(model_names)) != len(model_names):
        raise ValueError("models must contain at least three unique names")
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("seeds must be unique")
    records = []
    for dataset in sorted(data["dataset"].dropna().unique()):
        for metric in metrics:
            if metric not in data:
                continue
            family = []
            for level in noise_levels:
                subset = data.loc[
                    data["dataset"].eq(dataset)
                    & np.isclose(data["corruption_level"], level)
                    & data["model"].isin(model_names)
                ]
                if subset.duplicated(["model", "seed"]).any():
                    raise ValueError("duplicate model/seed result in Friedman block")
                matrix, missing = [], []
                for seed in seed_ids:
                    block = subset.loc[subset["seed"].eq(seed)]
                    values = []
                    for model in model_names:
                        found = block.loc[block["model"].eq(model), metric]
                        values.append(float(found.iloc[0]) if len(found) else np.nan)
                    if np.isfinite(values).all():
                        matrix.append(values)
                    else:
                        missing.append(seed)
                ranks_avg = {model: np.nan for model in model_names}
                status, statistic, p_value = (
                    "insufficient_complete_blocks",
                    np.nan,
                    np.nan,
                )
                df = len(model_names) - 1
                kendall_w = np.nan
                if len(matrix) == len(seed_ids) and len(matrix) >= 2:
                    values = np.asarray(matrix)
                    oriented = -values if metric in HIGHER_IS_BETTER else values
                    rank_matrix = np.vstack([stats.rankdata(row) for row in oriented])
                    means = rank_matrix.mean(axis=0)
                    ranks_avg = dict(zip(model_names, means.tolist(), strict=True))
                    n, k = rank_matrix.shape
                    q = 12.0 * n / (k * (k + 1)) * np.square(means).sum() - 3 * n * (
                        k + 1
                    )
                    tie_sum = 0.0
                    for row in oriented:
                        _, counts = np.unique(row, return_counts=True)
                        tie_sum += float(np.sum(counts**3 - counts))
                    tie_correction = 1.0 - tie_sum / (n * k * (k * k - 1))
                    if tie_correction <= 1e-15:
                        statistic, p_value, kendall_w, status = (
                            0.0,
                            1.0,
                            0.0,
                            "constant_ties",
                        )
                    else:
                        statistic = max(0.0, q / tie_correction)
                        p_value = float(stats.chi2.sf(statistic, df))
                        kendall_w = statistic / (n * (k - 1))
                        status = "ok"
                family.append(
                    {
                        "dataset": dataset,
                        "noise_level": float(level),
                        "metric": metric,
                        "n_blocks": len(matrix),
                        "expected_blocks": len(seed_ids),
                        "missing_seeds": ",".join(map(str, missing)),
                        "status": status,
                        "statistic": statistic,
                        "degrees_of_freedom": df,
                        "p_value": p_value,
                        "kendall_w": kendall_w,
                        "average_ranks": json.dumps(
                            {
                                name: (float(value) if np.isfinite(value) else None)
                                for name, value in ranks_avg.items()
                            }
                        ),
                    }
                )
            valid_idx = [
                i for i, row in enumerate(family) if np.isfinite(row["p_value"])
            ]
            if valid_idx:
                adj = adjust_p_values(
                    [
                        row["p_value"] if np.isfinite(row["p_value"]) else 1.0
                        for row in family
                    ],
                    "holm",
                    alpha,
                )
                for i in valid_idx:
                    family[i]["holm_adjusted_p_value"] = float(adj.adjusted_p_values[i])
                    family[i]["omnibus_significant"] = bool(adj.reject_null[i])
            for row in family:
                row.setdefault("holm_adjusted_p_value", np.nan)
                row.setdefault("omnibus_significant", False)
            records.extend(family)

    friedman = pd.DataFrame.from_records(records)
    nemenyi_records = []
    for row in records:
        for a, b in combinations(model_names, 2):
            run = (
                row["status"] in {"ok", "constant_ties"} and row["omnibus_significant"]
            )
            average_ranks = json.loads(row["average_ranks"])
            rank_a = average_ranks.get(a, np.nan)
            rank_b = average_ranks.get(b, np.nan)
            rank_a = np.nan if rank_a is None else rank_a
            rank_b = np.nan if rank_b is None else rank_b
            n, k = row["n_blocks"], len(model_names)
            se = np.sqrt(k * (k + 1) / (6 * n)) if n else np.nan
            cd = (
                float(
                    stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * se
                )
                if n
                else np.nan
            )
            difference = (
                abs(rank_a - rank_b)
                if np.isfinite(rank_a) and np.isfinite(rank_b)
                else np.nan
            )
            p = (
                float(
                    stats.studentized_range.sf(
                        (difference / se) * np.sqrt(2), k, np.inf
                    )
                )
                if run and se
                else np.nan
            )
            nemenyi_records.append(
                {
                    "dataset": row["dataset"],
                    "noise_level": row["noise_level"],
                    "metric": row["metric"],
                    "model_a": a,
                    "model_b": b,
                    "n_blocks": n,
                    "average_rank_a": rank_a,
                    "average_rank_b": rank_b,
                    "absolute_average_rank_difference": difference,
                    "critical_difference": cd,
                    "adjusted_p_value": p,
                    "reject_null": bool(p <= alpha) if np.isfinite(p) else pd.NA,
                    "status": "ok"
                    if run
                    else (
                        "not_run_omnibus_not_significant"
                        if row["status"] in {"ok", "constant_ties"}
                        else "not_run_insufficient_blocks"
                    ),
                }
            )
    return friedman, pd.DataFrame.from_records(nemenyi_records)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean over independent seed units.

    The resampling unit is each seed-level result, never an individual test
    observation. With five seeds, this is a small-sample descriptive interval;
    it does not provide high-precision coverage guarantees.
    """
    if not len(values):
        return np.nan, np.nan, np.nan
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("values must be a finite one-dimensional array")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 1
    ):
        raise ValueError("n_resamples must be a positive integer")
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(n_resamples), len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high)


def calibration_wilcoxon_results(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("ece", "brier_score"),
    models: Sequence[str] = CORE_MODELS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    noise_levels: Sequence[float] = CORE_NOISE_LEVELS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare raw minus calibrated metric by matched seed; positive improves."""
    data = _noise_rows(results)
    rows = []
    for dataset in sorted(data["dataset"].dropna().unique()):
        for metric in metrics:
            calibrated = f"calibrated_{metric}"
            if metric not in data or calibrated not in data:
                continue
            family = []
            for model in models:
                for level in noise_levels:
                    block = data.loc[
                        data["dataset"].eq(dataset)
                        & data["model"].eq(model)
                        & np.isclose(data["corruption_level"], level)
                    ]
                    if block.duplicated("seed").any():
                        raise ValueError("duplicate seed in calibration comparison")
                    diffs, excluded = [], []
                    for seed in seeds:
                        item = block.loc[block["seed"].eq(seed)]
                        valid_status = "calibration_status" not in item or (
                            len(item)
                            and item.iloc[0]["calibration_status"] == "succeeded"
                        )
                        if (
                            len(item)
                            and valid_status
                            and np.isfinite(
                                item.iloc[0][[metric, calibrated]].astype(float)
                            ).all()
                        ):
                            diffs.append(
                                float(item.iloc[0][metric] - item.iloc[0][calibrated])
                            )
                        else:
                            excluded.append(seed)
                    d = np.asarray(diffs, dtype=float)
                    ci_seed = bootstrap_seed(
                        "calibration", dataset, model, metric, float(level).hex()
                    )
                    mean, low, high = bootstrap_mean_ci(d, seed=ci_seed)
                    status = (
                        "ok"
                        if len(d) == len(seeds)
                        else "insufficient_pairs_descriptive"
                    )
                    statistic = p_value = effect = np.nan
                    if len(d) >= len(seeds):
                        if np.all(d == 0):
                            statistic, p_value, effect = 0.0, 1.0, 0.0
                        else:
                            test = stats.wilcoxon(
                                d,
                                zero_method="wilcox",
                                alternative="two-sided",
                                method="auto",
                            )
                            statistic, p_value = (
                                float(test.statistic),
                                float(test.pvalue),
                            )
                            effect = rank_biserial_correlation(d)
                    family.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "noise_level": float(level),
                            "metric": metric,
                            "n_pairs": len(d),
                            "expected_pairs": len(seeds),
                            "excluded_seeds": ",".join(map(str, excluded)),
                            "mean_raw_minus_calibrated": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "interval_method": "percentile_bootstrap_seed_resampling",
                            "ci_seed": ci_seed,
                            "ci_resamples": BOOTSTRAP_RESAMPLES,
                            "direction": "calibrated_lower"
                            if mean > 0
                            else "raw_lower"
                            if mean < 0
                            else "no_mean_difference"
                            if np.isfinite(mean)
                            else "unavailable",
                            "statistic": statistic,
                            "p_value": p_value,
                            "rank_biserial": effect,
                            "status": status,
                        }
                    )
            valid = [i for i, item in enumerate(family) if np.isfinite(item["p_value"])]
            if valid:
                adjusted = adjust_p_values(
                    [
                        item["p_value"] if np.isfinite(item["p_value"]) else 1.0
                        for item in family
                    ],
                    "holm",
                    alpha,
                )
                for i in valid:
                    family[i]["holm_adjusted_p_value"] = float(
                        adjusted.adjusted_p_values[i]
                    )
                    family[i]["reject_null"] = bool(adjusted.reject_null[i])
            for item in family:
                item.setdefault("holm_adjusted_p_value", np.nan)
                item.setdefault("reject_null", pd.NA)
            rows.extend(family)
    columns = [
        "dataset",
        "model",
        "noise_level",
        "metric",
        "n_pairs",
        "expected_pairs",
        "excluded_seeds",
        "mean_raw_minus_calibrated",
        "ci95_low",
        "ci95_high",
        "direction",
        "statistic",
        "p_value",
        "rank_biserial",
        "status",
        "holm_adjusted_p_value",
        "reject_null",
    ]
    return pd.DataFrame.from_records(rows).reindex(columns=columns)


def degradation_slope_results(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = CORE_METRICS,
    models: Sequence[str] = CORE_MODELS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    noise_levels: Sequence[float] = CORE_NOISE_LEVELS,
) -> pd.DataFrame:
    """Per-seed OLS slopes, normalized to worsening per +10pp noise.

    All required metrics have theoretical range [0, 1]. Normalized slope is
    ``direction * raw_slope * 0.10 / range``; positive means degradation.
    This range normalization avoids unstable division by near-zero baselines,
    but equal ranges do not imply equal practical importance.
    """
    data = _noise_rows(results)
    rows = []
    for dataset in sorted(data["dataset"].dropna().unique()):
        for model in models:
            for seed in seeds:
                for metric in metrics:
                    if metric not in data:
                        continue
                    block = data.loc[
                        data["dataset"].eq(dataset)
                        & data["model"].eq(model)
                        & data["seed"].eq(seed)
                    ]
                    if block.duplicated("corruption_level").any():
                        raise ValueError("duplicate noise level in degradation slope")
                    selected = block.loc[
                        block["corruption_level"].isin(noise_levels),
                        ["corruption_level", metric],
                    ].dropna()
                    observed = set(selected["corruption_level"].astype(float))
                    missing = [
                        level for level in noise_levels if float(level) not in observed
                    ]
                    raw_slope = normalized = np.nan
                    status = "incomplete_noise_levels"
                    if not missing and len(selected) == len(noise_levels):
                        raw_slope = float(
                            np.polyfit(
                                selected["corruption_level"],
                                selected[metric].astype(float),
                                1,
                            )[0]
                        )
                        sign = -1.0 if metric in HIGHER_IS_BETTER else 1.0
                        normalized = sign * raw_slope * 0.10
                        status = "ok"
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "metric": metric,
                            "n_noise_levels": len(selected),
                            "expected_noise_levels": len(noise_levels),
                            "missing_noise_levels": ",".join(map(str, missing)),
                            "raw_slope_per_unit_noise": raw_slope,
                            "normalized_degradation_per_10pp": normalized,
                            "status": status,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def compare_degradation_slopes(
    slopes: pd.DataFrame, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Paired H1 accuracy/F1-vs-reliability slope contrasts by dataset/model."""
    rows = []
    performance = ("accuracy", "macro_f1")
    reliability = ("ece", "brier_score")
    for dataset in sorted(slopes["dataset"].dropna().unique()):
        family = []
        subset = slopes.loc[slopes["dataset"].eq(dataset)]
        for model in CORE_MODELS:
            for perf_metric in performance:
                for metric in reliability:
                    part = subset.loc[subset["model"].eq(model)]
                    perf = part.loc[
                        part["metric"].eq(perf_metric),
                        ["seed", "normalized_degradation_per_10pp"],
                    ]
                    rel = part.loc[
                        part["metric"].eq(metric),
                        ["seed", "normalized_degradation_per_10pp"],
                    ]
                    paired = perf.merge(
                        rel,
                        on="seed",
                        suffixes=("_performance", "_reliability"),
                        validate="one_to_one",
                    ).dropna()
                    delta = (
                        paired["normalized_degradation_per_10pp_reliability"].to_numpy()
                        - paired[
                            "normalized_degradation_per_10pp_performance"
                        ].to_numpy()
                    )
                    ci_seed = bootstrap_seed(
                        "degradation_slope",
                        dataset,
                        model,
                        perf_metric,
                        metric,
                    )
                    mean, low, high = bootstrap_mean_ci(delta, seed=ci_seed)
                    statistic = p_value = effect = np.nan
                    status = "insufficient_pairs_descriptive"
                    if len(delta) >= 5:
                        if np.all(delta == 0):
                            statistic, p_value, effect, status = 0.0, 1.0, 0.0, "ok"
                        else:
                            outcome = stats.wilcoxon(
                                delta,
                                zero_method="wilcox",
                                alternative="two-sided",
                                method="auto",
                            )
                            statistic, p_value, effect, status = (
                                float(outcome.statistic),
                                float(outcome.pvalue),
                                rank_biserial_correlation(delta),
                                "ok",
                            )
                    family.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "performance_metric": perf_metric,
                            "reliability_metric": metric,
                            "n_pairs": len(delta),
                            "mean_reliability_minus_performance_degradation": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "interval_method": "percentile_bootstrap_seed_resampling",
                            "ci_seed": ci_seed,
                            "ci_resamples": BOOTSTRAP_RESAMPLES,
                            "direction": "reliability_faster"
                            if mean > 0
                            else "performance_faster"
                            if mean < 0
                            else "equal_mean_slopes"
                            if np.isfinite(mean)
                            else "unavailable",
                            "statistic": statistic,
                            "p_value": p_value,
                            "rank_biserial": effect,
                            "status": status,
                        }
                    )
        valid = [i for i, row in enumerate(family) if np.isfinite(row["p_value"])]
        if valid:
            adj = adjust_p_values(
                [
                    row["p_value"] if np.isfinite(row["p_value"]) else 1.0
                    for row in family
                ],
                "holm",
                alpha,
            )
            for i in valid:
                family[i]["holm_adjusted_p_value"] = float(adj.adjusted_p_values[i])
                family[i]["reject_null"] = bool(adj.reject_null[i])
        for row in family:
            row.setdefault("holm_adjusted_p_value", np.nan)
            row.setdefault("reject_null", pd.NA)
        rows.extend(family)
    return pd.DataFrame.from_records(rows)


def descriptive_summary(
    results: pd.DataFrame, *, metrics: Sequence[str] = CORE_METRICS
) -> pd.DataFrame:
    """Means, SDs, and seed-level 95% t intervals; small-n estimates."""
    data = _noise_rows(results)
    rows = []
    for metric in metrics:
        if metric not in data:
            continue
        for key, group in data.groupby(
            ["dataset", "model", "corruption_level"], dropna=False, sort=True
        ):
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            ci_seed = bootstrap_seed(
                "descriptive_summary",
                key[0],
                key[1],
                float(key[2]).hex(),
                metric,
            )
            mean, low, high = bootstrap_mean_ci(values, seed=ci_seed)
            rows.append(
                {
                    "dataset": key[0],
                    "model": key[1],
                    "noise_level": key[2],
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean,
                    "standard_deviation": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "interval_method": "percentile_bootstrap_seed_resampling",
                    "ci_seed": ci_seed,
                    "ci_resamples": BOOTSTRAP_RESAMPLES,
                    "status": "ok" if len(values) else "missing",
                }
            )
    return pd.DataFrame.from_records(rows)


def run_research_analysis(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all Phase 7 tabular outputs from existing result rows."""
    friedman, nemenyi = analyze_friedman_nemenyi(results)
    calibration = calibration_wilcoxon_results(results)
    slopes = degradation_slope_results(results)
    slope_tests = compare_degradation_slopes(slopes)
    effects = pd.concat(
        [
            friedman.assign(
                effect_size_type="kendall_w", effect_size=friedman["kendall_w"]
            )[
                [
                    "dataset",
                    "noise_level",
                    "metric",
                    "n_blocks",
                    "effect_size_type",
                    "effect_size",
                    "status",
                ]
            ],
            nemenyi.assign(
                effect_size_type="average_rank_difference",
                effect_size=nemenyi["absolute_average_rank_difference"],
            )[
                [
                    "dataset",
                    "noise_level",
                    "metric",
                    "n_blocks",
                    "model_a",
                    "model_b",
                    "effect_size_type",
                    "effect_size",
                    "status",
                ]
            ],
            calibration.assign(
                effect_size_type="rank_biserial",
                effect_size=calibration["rank_biserial"],
            )[
                [
                    "dataset",
                    "noise_level",
                    "metric",
                    "n_pairs",
                    "effect_size_type",
                    "effect_size",
                    "status",
                ]
            ],
            slope_tests.assign(
                effect_size_type="rank_biserial",
                effect_size=slope_tests["rank_biserial"],
            )[
                [
                    "dataset",
                    "reliability_metric",
                    "performance_metric",
                    "n_pairs",
                    "effect_size_type",
                    "effect_size",
                    "status",
                ]
            ].rename(
                columns={
                    "reliability_metric": "metric",
                    "performance_metric": "comparison",
                }
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    return {
        "friedman_results": friedman,
        "nemenyi_results": nemenyi,
        "wilcoxon_calibration_results": calibration,
        "degradation_slope_results": slopes,
        "degradation_slope_comparisons": slope_tests,
        "effect_sizes": effects,
        "statistical_summary": descriptive_summary(results),
    }
