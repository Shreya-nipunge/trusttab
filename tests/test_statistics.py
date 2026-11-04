import numpy as np
import pandas as pd
import pytest
from scipy import stats

from trusttab.statistics import (
    ComparisonResult,
    adjust_p_values,
    analyze_noise_levels,
    compare_models,
    compare_paired_scores,
    compare_paired_scores_wilcoxon,
)


def test_compare_paired_scores_returns_result():
    result = compare_paired_scores(
        [0.9, 0.85, 0.88, 0.92],
        [0.8, 0.82, 0.81, 0.86],
    )

    assert isinstance(result, ComparisonResult)
    assert result.mean_difference > 0
    assert 0 <= result.p_value <= 1
    assert result.confidence_interval[0] <= result.mean_difference
    assert result.mean_difference <= result.confidence_interval[1]


def test_difference_sign_is_correct():
    result = compare_paired_scores([0.9, 0.8, 0.7], [0.7, 0.6, 0.5])
    assert result.mean_difference == pytest.approx(0.2)


def test_paired_ttest_matches_scipy_and_reports_direction():
    a = np.array([0.72, 0.81, 0.75, 0.9])
    b = np.array([0.7, 0.79, 0.73, 0.8])
    result = compare_paired_scores(a, b)
    expected = stats.ttest_rel(a, b)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.mean_difference == pytest.approx(np.mean(a - b))


def test_paired_ttest_handles_constant_and_identical_pairs():
    constant = compare_paired_scores([0.8, 0.9], [0.7, 0.8])
    assert np.isinf(constant.statistic)
    assert constant.p_value == 0.0
    same = compare_paired_scores([0.8, 0.9], [0.8, 0.9])
    assert same.statistic == 0.0
    assert same.p_value == 1.0


def test_wilcoxon_paired_test_matches_scipy():
    a = np.array([0.8, 0.9, 0.7, 0.85, 0.93])
    b = np.array([0.7, 0.88, 0.75, 0.8, 0.91])
    result = compare_paired_scores_wilcoxon(a, b)
    expected = stats.wilcoxon(a, b)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.mean_difference == pytest.approx(np.mean(a - b))
    assert result.n_observations == len(a)


def test_wilcoxon_identical_pairs_return_non_significant_result():
    result = compare_paired_scores_wilcoxon([0.8, 0.9], [0.8, 0.9])
    assert result.statistic == 0
    assert result.p_value == 1


def test_multiple_comparison_adjustments():
    p_values = [0.01, 0.04, 0.03, 0.2]
    bonferroni = adjust_p_values(p_values, "bonferroni")
    holm = adjust_p_values(p_values, "holm-bonferroni")
    fdr = adjust_p_values(p_values, "benjamini-hochberg")
    assert bonferroni.adjusted_p_values == pytest.approx([0.04, 0.16, 0.12, 0.8])
    assert holm.adjusted_p_values == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert fdr.adjusted_p_values == pytest.approx(
        [0.04, 0.0533333333, 0.0533333333, 0.2]
    )
    assert bonferroni.reject_null.tolist() == [True, False, False, False]


def _experiment_results():
    rows = []
    for model, offset in (("A", 0.05), ("B", 0.0)):
        for noise_rate in (0.0, 0.2):
            for seed, score in ((1, 0.7), (2, 0.8), (3, 0.75)):
                rows.append(
                    {
                        "dataset": "synthetic",
                        "model": model,
                        "noise_rate": noise_rate,
                        "seed": seed,
                        "accuracy": score + offset - noise_rate / 2,
                    }
                )
    return pd.DataFrame(rows)


def test_compare_models_pairs_by_seed_and_noise_and_is_deterministic():
    results = _experiment_results()
    first = compare_models(results, "accuracy", correction="holm")
    second = compare_models(results.sample(frac=1, random_state=7), "accuracy", correction="holm")
    assert len(first) == 2
    assert first["n_paired"].tolist() == [3, 3]
    assert first["mean_difference"].tolist() == pytest.approx([0.05, 0.05])
    assert set(first["difference_definition"]) == {"model_a - model_b"}
    pd.testing.assert_frame_equal(first, second)


def test_compare_models_supports_wilcoxon_and_custom_model_subset():
    result = compare_models(
        _experiment_results(),
        "accuracy",
        models=["A", "B"],
        test="wilcoxon",
    )
    assert set(result["test"]) == {"wilcoxon"}
    assert result["p_value"].between(0, 1).all()


def test_compare_models_compares_all_model_pairs():
    results = _experiment_results()
    third = results[results["model"] == "A"].copy()
    third["model"] = "C"
    third["accuracy"] -= 0.1
    all_models = pd.concat([results, third], ignore_index=True)
    compared = compare_models(all_models, "accuracy")
    assert len(compared) == 6
    assert set(zip(compared["model_a"], compared["model_b"])) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    }


def test_compare_models_reports_or_rejects_unmatched_configurations():
    results = _experiment_results()
    results = results.drop(results[(results.model == "B") & (results.seed == 3)].index)
    with pytest.raises(ValueError, match="Unmatched configurations"):
        compare_models(results, "accuracy")

    result = compare_models(results, "accuracy", unmatched="drop")
    assert (result["n_paired"] == 2).all()
    assert (result["n_unmatched_a"] == 1).all()
    assert (result["n_unmatched_b"] == 0).all()


def test_compare_models_reports_insufficient_paired_observations():
    results = _experiment_results().query("seed == 1")
    result = compare_models(results, "accuracy")
    assert (result["n_paired"] == 1).all()
    assert result["p_value"].isna().all()
    assert set(result["status"]) == {"insufficient_observations"}

    adjusted = compare_models(results, "accuracy", correction="holm")
    assert adjusted["adjusted_p_value"].isna().all()
    assert adjusted["reject_null"].isna().all()


def test_analyze_noise_levels_returns_descriptive_and_paired_results():
    summary = analyze_noise_levels(
        _experiment_results(), "accuracy", reference_model="B"
    )
    model_a = summary[summary.model == "A"]
    assert len(model_a) == 2
    assert (model_a["n_paired"] == 3).all()
    assert np.allclose(model_a["mean_difference"], 0.05)
    assert model_a["p_value"].between(0, 1).all()


def test_confidence_interval_changes_with_confidence_level():
    a = [0.9, 0.8, 0.7, 0.95]
    b = [0.7, 0.65, 0.6, 0.8]

    narrow = compare_paired_scores(a, b, confidence_level=0.90)
    wide = compare_paired_scores(a, b, confidence_level=0.99)

    narrow_width = narrow.confidence_interval[1] - narrow.confidence_interval[0]
    wide_width = wide.confidence_interval[1] - wide.confidence_interval[0]
    assert wide_width > narrow_width


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([0.8, np.nan], [0.7, 0.6]),
        ([0.8, np.inf], [0.7, 0.6]),
        ([[0.8, 0.9]], [[0.7, 0.6]]),
        ([0.8], [0.7]),
        ([0.8, 0.9], [0.7]),
    ],
)
def test_invalid_scores_raise_value_error(a, b):
    with pytest.raises(ValueError):
        compare_paired_scores(a, b)


@pytest.mark.parametrize(
    "function",
    [compare_paired_scores, compare_paired_scores_wilcoxon],
)
def test_paired_tests_reject_insufficient_samples(function):
    with pytest.raises(ValueError, match="2 paired"):
        function([0.8], [0.7])


@pytest.mark.parametrize(
    "p_values,method",
    [([0.1, np.nan], "holm"), ([0.1, 1.1], "holm"), ([[0.1]], "holm"), ([0.1], "unknown")],
)
def test_adjust_p_values_rejects_invalid_inputs(p_values, method):
    with pytest.raises(ValueError):
        adjust_p_values(p_values, method)


def test_compare_models_rejects_missing_columns_and_duplicates():
    with pytest.raises(ValueError, match="missing required columns"):
        compare_models(pd.DataFrame({"model": ["A"]}), "accuracy")
    duplicated = pd.concat([_experiment_results(), _experiment_results().iloc[[0]]])
    with pytest.raises(ValueError, match="at most one"):
        compare_models(duplicated, "accuracy")


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.1])
def test_invalid_confidence_level_raises_value_error(level):
    with pytest.raises(ValueError):
        compare_paired_scores([0.8, 0.9], [0.7, 0.8], confidence_level=level)
