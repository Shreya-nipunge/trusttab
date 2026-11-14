"""Independent numerical checks for seed-blocked research statistics."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from trusttab.statistics.research import (
    analyze_friedman_nemenyi,
    bootstrap_mean_ci,
    bootstrap_seed,
    calibration_wilcoxon_results,
    compare_degradation_slopes,
    degradation_slope_results,
    rank_biserial_correlation,
)

MODELS = ("logistic_regression", "random_forest", "xgboost", "tabpfn")
LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)


def test_percentile_bootstrap_ci_is_reproducible_and_seed_level():
    values = np.array([0.1, 0.2, 0.4, 0.5, 0.8])
    seed = bootstrap_seed("fixture", "german_credit", "accuracy")
    observed = bootstrap_mean_ci(values, seed=seed, n_resamples=2_000)
    repeated = bootstrap_mean_ci(values, seed=seed, n_resamples=2_000)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(2_000, len(values)))
    means = values[indices].mean(axis=1)
    expected = (values.mean(), *np.quantile(means, [0.025, 0.975]))
    assert observed == pytest.approx(expected)
    assert repeated == observed
    assert bootstrap_seed("fixture", "german_credit", "accuracy") == seed
    assert bootstrap_seed("fixture", "heart_disease", "accuracy") != seed


def test_bootstrap_ci_keeps_single_seed_interval_undefined():
    mean, low, high = bootstrap_mean_ci([0.4], seed=17)
    assert mean == pytest.approx(0.4)
    assert np.isnan(low)
    assert np.isnan(high)


def _rows():
    rows = []
    offsets = dict(zip(MODELS, (0.0, 0.1, 0.2, 0.3), strict=True))
    for model in MODELS:
        for level in LEVELS:
            for seed in range(5):
                acc = 0.9 - offsets[model] - level * (0.4 + seed * 0.01)
                ece = 0.05 + offsets[model] + level * (0.1 + seed * 0.002)
                rows.append(
                    {
                        "dataset": "cleveland",
                        "model": model,
                        "seed": seed,
                        "experiment_type": "label_noise",
                        "corruption_level": level,
                        "accuracy": acc,
                        "ece": ece,
                        "brier_score": ece + 0.1,
                        "balanced_accuracy": acc,
                        "macro_f1": acc,
                        "roc_auc": acc,
                        "calibrated_ece": ece - 0.01,
                        "calibrated_brier_score": ece + 0.08,
                        "calibration_status": "succeeded",
                    }
                )
    return pd.DataFrame(rows)


def test_friedman_uses_seed_blocks_and_matches_scipy():
    data = _rows()
    friedman, pairs = analyze_friedman_nemenyi(
        data, metrics=("accuracy",), noise_levels=(0.0,)
    )
    observed = friedman.iloc[0]
    pivot = data.query("corruption_level == 0").pivot(
        index="seed", columns="model", values="accuracy"
    )
    expected = stats.friedmanchisquare(*(pivot[model].to_numpy() for model in MODELS))
    assert observed["statistic"] == pytest.approx(expected.statistic)
    assert observed["p_value"] == pytest.approx(expected.pvalue)
    assert observed["n_blocks"] == 5
    assert len(pairs) == 6


def test_friedman_reports_missing_seed_blocks_and_rejects_duplicates():
    data = _rows().query("seed != 4")
    friedman, _ = analyze_friedman_nemenyi(
        data, metrics=("accuracy",), noise_levels=(0.0,)
    )
    assert friedman.iloc[0]["status"] == "insufficient_complete_blocks"
    assert friedman.iloc[0]["missing_seeds"] == "4"
    duplicate = pd.concat([_rows(), _rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate model/seed"):
        analyze_friedman_nemenyi(duplicate, metrics=("accuracy",), noise_levels=(0.0,))


def test_tie_correction_handles_constant_identical_block():
    data = _rows()
    selected = data["corruption_level"].eq(0.0)
    data.loc[selected, "accuracy"] = 0.5
    friedman, pairs = analyze_friedman_nemenyi(
        data, metrics=("accuracy",), noise_levels=(0.0,)
    )
    assert friedman.iloc[0]["status"] == "constant_ties"
    assert friedman.iloc[0]["statistic"] == 0
    assert friedman.iloc[0]["p_value"] == 1
    assert set(pairs["status"]) == {"not_run_omnibus_not_significant"}


def test_calibration_wilcoxon_matches_known_pairs_and_excludes_failed_rows():
    data = _rows()
    target = data["model"].eq(MODELS[0]) & data["corruption_level"].eq(0.0)
    data.loc[target, "calibrated_ece"] = data.loc[target, "ece"] - 0.01
    data.loc[target, "calibration_status"] = "succeeded"
    selected = (
        data["model"].eq(MODELS[1])
        & data["corruption_level"].eq(0.0)
        & data["seed"].eq(0)
    )
    data.loc[selected, "calibration_status"] = "failed"
    result = calibration_wilcoxon_results(
        data, metrics=("ece",), models=MODELS[:2], noise_levels=(0.0,)
    )
    first = result.query("model == @MODELS[0]").iloc[0]
    assert first["n_pairs"] == 5
    expected = stats.wilcoxon(np.full(5, 0.01))
    assert first["statistic"] == pytest.approx(expected.statistic)
    assert first["p_value"] == pytest.approx(expected.pvalue)
    second = result.query("model == @MODELS[1]").iloc[0]
    assert second["n_pairs"] == 4
    assert second["status"] == "insufficient_pairs_descriptive"
    assert np.isnan(second["p_value"])
    assert first["holm_adjusted_p_value"] == pytest.approx(
        min(1.0, first["p_value"] * 2)
    )


def test_rank_biserial_and_degradation_direction_are_correct():
    assert rank_biserial_correlation([1, 2, -1]) == pytest.approx(0.5)
    assert rank_biserial_correlation([0, 0]) == 0
    slopes = degradation_slope_results(
        _rows(), metrics=("accuracy", "ece"), models=MODELS[:1]
    )
    accuracy = slopes.query("metric == 'accuracy'")
    ece = slopes.query("metric == 'ece'")
    assert (accuracy["normalized_degradation_per_10pp"] > 0).all()
    assert (ece["normalized_degradation_per_10pp"] > 0).all()
    assert accuracy.iloc[0]["normalized_degradation_per_10pp"] == pytest.approx(0.04)
    contrasts = compare_degradation_slopes(slopes)
    assert len(contrasts) == 16  # four models x two performance x two reliability
    assert set(contrasts["performance_metric"]) == {"accuracy", "macro_f1"}


def test_nemenyi_is_gated_until_adjusted_omnibus_is_significant():
    data = _rows()
    # Destroy model ordering at all levels to make omnibus ties and prevent post-hoc.
    data["accuracy"] = 0.5
    friedman, pairs = analyze_friedman_nemenyi(data, metrics=("accuracy",))
    assert not friedman["omnibus_significant"].any()
    assert pairs["adjusted_p_value"].isna().all()
    assert pairs["status"].str.startswith("not_run").all()


def test_nemenyi_adjusted_pair_p_matches_studentized_range():
    data = _rows()
    friedman, pairs = analyze_friedman_nemenyi(
        data, metrics=("accuracy",), noise_levels=LEVELS
    )
    assert friedman["omnibus_significant"].all()
    pair = pairs.query("model_a == 'logistic_regression' and model_b == 'tabpfn'").iloc[
        0
    ]
    k, n = len(MODELS), 5
    se = np.sqrt(k * (k + 1) / (6 * n))
    expected_p = stats.studentized_range.sf(
        pair["absolute_average_rank_difference"] / se * np.sqrt(2), k, np.inf
    )
    assert pair["adjusted_p_value"] == pytest.approx(expected_p)
    assert pair["critical_difference"] == pytest.approx(
        stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * se
    )
