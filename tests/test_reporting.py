"""Tests for automated robustness reporting."""

import numpy as np
import pandas as pd
import pytest

from trusttab.reporting import generate_robustness_report
from trusttab.reporting.report import _write_critical_difference_diagrams


def _results(include_optional=True):
    rows = []
    for dataset in ("german_credit", "adult"):
        for model, offset in (("logistic_regression", 0.0), ("random_forest", 0.03)):
            for seed in (1, 2, 3):
                baseline = {
                    "dataset": dataset,
                    "model": model,
                    "experiment_type": "clean_baseline",
                    "corruption_type": "none",
                    "corruption_level": 0.0,
                    "seed": seed,
                    "accuracy": 0.9 - offset + seed * 0.001,
                    "balanced_accuracy": 0.88 - offset,
                    "macro_f1": 0.87 - offset,
                    "brier_score": 0.12 + offset,
                    "ece": 0.04 + offset,
                    "baseline_id": f"{dataset}:{model}:{seed}",
                }
                if include_optional:
                    baseline.update(
                        roc_auc=0.92 - offset,
                        adaptive_ece=0.05 + offset,
                        mce=0.08 + offset,
                        log_loss=0.3 + offset,
                    )
                rows.append(baseline)
                for exp_type, corruption, level in (
                    ("label_noise", "symmetric_label_noise", 0.2),
                    ("label_noise", "symmetric_label_noise", 0.4),
                    ("feature_shift", "mean_shift", 0.5),
                ):
                    stress = {
                        **baseline,
                        "experiment_type": exp_type,
                        "corruption_type": corruption,
                        "corruption_level": level,
                        "accuracy": baseline["accuracy"] - 0.03 * level,
                        "balanced_accuracy": baseline["balanced_accuracy"]
                        - 0.04 * level,
                        "macro_f1": baseline["macro_f1"] - 0.05 * level,
                        "brier_score": baseline["brier_score"] + 0.02 * level,
                        "ece": baseline["ece"] + 0.01 * level,
                    }
                    if include_optional:
                        stress.update(
                            roc_auc=baseline["roc_auc"] - 0.02 * level,
                            adaptive_ece=baseline["adaptive_ece"] + 0.01 * level,
                            mce=baseline["mce"] + 0.02 * level,
                            log_loss=baseline["log_loss"] + 0.03 * level,
                        )
                    rows.append(stress)
    return pd.DataFrame(rows)


def test_report_creates_summary_tables_markdown_and_figures(tmp_path):
    result = generate_robustness_report(
        _results(), tmp_path / "new" / "robustness_report", title="Synthetic"
    )
    output = tmp_path / "new" / "robustness_report"
    assert result["summary"].set_index("item").loc["datasets_evaluated", "value"] == 2
    assert result["summary"].set_index("item").loc["models_evaluated", "value"] == 2
    assert (
        result["summary"].set_index("item").loc["best_calibrated_model", "value"]
        == "unavailable"
    )
    assert len(result["model_comparison"]) == 2 * 2 * 4
    assert result["statistical_comparisons"]["dataset"].nunique() == 2
    for filename in (
        "friedman_results.csv",
        "nemenyi_results.csv",
        "wilcoxon_calibration_results.csv",
        "degradation_slope_results.csv",
        "effect_sizes.csv",
        "statistical_summary.csv",
    ):
        assert (output / filename).is_file()
    for filename in (
        "noise_robustness.png",
        "ece_vs_noise.png",
        "brier_score_vs_noise.png",
        "macro_f1_vs_noise.png",
        "model_ranking_heatmap.png",
        "cross_dataset_comparison.png",
    ):
        assert (output / "figures" / filename).is_file()
    assert (output / "figures" / "feature_shift_robustness.png").is_file()
    assert not (output / "figures" / "calibration_comparison.png").exists()
    assert "dataset" in pd.read_csv(output / "friedman_results.csv").columns
    assert "n_blocks" in pd.read_csv(output / "friedman_results.csv").columns
    assert "adjusted_p_value" in pd.read_csv(output / "nemenyi_results.csv").columns
    assert (
        "rank_biserial"
        in pd.read_csv(output / "wilcoxon_calibration_results.csv").columns
    )
    assert not result["statistical_comparisons"].empty
    assert "mean_relative_degradation" in result["robustness_degradation"].columns
    for artifact in result["artifacts"]:
        assert (output / artifact).is_file()
    assert "# Synthetic" in (output / "report.md").read_text(encoding="utf-8")
    assert pd.read_csv(output / "model_comparison.csv").shape[0] == len(
        result["model_comparison"]
    )
    assert pd.read_csv(output / "robustness_degradation.csv").shape[0] == len(
        result["robustness_degradation"]
    )


def test_report_handles_missing_optional_metrics_and_nan_values(tmp_path):
    data = _results(include_optional=False)
    data.loc[data["seed"] == 1, "accuracy"] = np.nan
    data["ece"] = data["ece"].astype(float)
    data.loc[data["seed"] == 2, "ece"] = np.nan
    report = generate_robustness_report(data, tmp_path)
    assert "mean_baseline_accuracy" in report["summary"]["item"].tolist()
    assert "roc_auc_mean" not in report["model_comparison"].columns
    assert report["robustness_degradation"]["absolute_degradation"].isna().any()
    assert not (tmp_path / "figures" / "calibration_comparison.png").exists()


def test_report_preserves_calibrated_metrics_and_failed_condition_status(tmp_path):
    data = _results()
    data["condition_id"] = [f"condition-{index}" for index in range(len(data))]
    data["calibration_method"] = "sigmoid"
    data["calibration_status"] = "succeeded"
    data["calibration_error"] = None
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "brier_score",
        "ece",
        "adaptive_ece",
        "mce",
        "log_loss",
    ):
        data[f"calibrated_{metric}"] = data[metric]

    failed_index = data.index[
        data["dataset"].eq("german_credit")
        & data["model"].eq("logistic_regression")
        & data["experiment_type"].eq("label_noise")
        & data["corruption_level"].eq(0.4)
        & data["seed"].eq(2)
    ][0]
    data.loc[failed_index, "calibration_status"] = "failed"
    data.loc[failed_index, "calibration_error"] = "RuntimeError: fixture failure"
    calibrated_columns = [
        f"calibrated_{metric}"
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "brier_score",
            "ece",
            "adaptive_ece",
            "mce",
            "log_loss",
        )
    ]
    data.loc[failed_index, calibrated_columns] = np.nan

    report = generate_robustness_report(data, tmp_path)

    assert "calibrated_ece_mean" in report["model_comparison"].columns
    assert (tmp_path / "figures" / "calibration_comparison.png").is_file()
    assert (
        not report["statistical_comparisons"]["metric"]
        .str.startswith("calibrated_")
        .any()
    )
    failed_count = (
        report["calibration_status"]
        .query("calibration_status == 'failed'")["n_conditions"]
        .sum()
    )
    assert failed_count == 1
    failed_comparison = (
        report["calibration_comparison"]
        .loc[
            report["calibration_comparison"]["condition_id"]
            == data.loc[failed_index, "condition_id"]
        ]
        .iloc[0]
    )
    assert failed_comparison["calibration_status"] == "failed"
    assert failed_comparison["calibration_error"] == "RuntimeError: fixture failure"
    assert np.isnan(failed_comparison["calibrated_ece"])
    assert np.isnan(failed_comparison["calibrated_minus_raw_ece"])
    assert "calibration_status.csv" in report["artifacts"]
    assert "calibration_comparison.csv" in report["artifacts"]
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Calibration outcomes" in markdown
    assert "failed" in markdown


def test_relative_degradation_is_undefined_for_zero_baseline(tmp_path):
    data = _results(include_optional=False)
    data.loc[data["experiment_type"] == "clean_baseline", "ece"] = 0.0
    report = generate_robustness_report(data, tmp_path)
    ece_rows = report["robustness_degradation"].query("metric == 'ece'")
    assert ece_rows["relative_degradation"].isna().all()


@pytest.mark.parametrize(
    "data, error, message",
    [
        (pd.DataFrame(), ValueError, "at least one row"),
        (_results().drop(columns="seed"), ValueError, "seed"),
        (_results().drop(columns="dataset"), ValueError, "dataset"),
        (
            _results().drop(
                columns=[
                    "accuracy",
                    "ece",
                    "brier_score",
                    "balanced_accuracy",
                    "macro_f1",
                    "roc_auc",
                    "adaptive_ece",
                    "mce",
                    "log_loss",
                ]
            ),
            ValueError,
            "at least one metric",
        ),
        (_results().drop(index=0), ValueError, "matching clean baseline"),
    ],
)
def test_report_rejects_invalid_benchmark_results(tmp_path, data, error, message):
    with pytest.raises(error, match=message):
        generate_robustness_report(data, tmp_path)


def test_report_input_is_not_mutated(tmp_path):
    data = _results()
    original = data.copy(deep=True)
    generate_robustness_report(data, tmp_path)
    pd.testing.assert_frame_equal(data, original)


def test_critical_difference_figure_requires_gated_valid_pairs(tmp_path):
    rows = []
    for left, right in (
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ):
        rows.append(
            {
                "dataset": "fixture",
                "metric": "accuracy",
                "noise_level": 0.2,
                "model_a": left,
                "model_b": right,
                "average_rank_a": 1.0,
                "average_rank_b": 4.0,
                "critical_difference": 2.5,
                "status": "ok",
                "adjusted_p_value": 0.01,
                "reject_null": False,
            }
        )
    paths = _write_critical_difference_diagrams(
        pd.DataFrame(rows), tmp_path / "figures"
    )
    assert len(paths) == 1
    assert (tmp_path / paths[0]).is_file()
    skipped = pd.DataFrame(rows).assign(status="not_run_omnibus_not_significant")
    assert _write_critical_difference_diagrams(skipped, tmp_path / "skipped") == []
