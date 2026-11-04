"""Tests for automated robustness reporting."""

import numpy as np
import pandas as pd
import pytest

from trusttab.reporting import generate_robustness_report


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
    assert len(result["model_comparison"]) == 2 * 2 * 4
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
    assert (tmp_path / "figures" / "calibration_comparison.png").is_file()


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
