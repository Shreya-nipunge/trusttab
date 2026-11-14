"""Automated analysis and artifact generation for robustness benchmarks."""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from trusttab.statistics import (
    compare_models,
    run_research_analysis,
)
from trusttab.statistics.research import (
    BOOTSTRAP_RESAMPLES,
    bootstrap_mean_ci,
    bootstrap_seed,
)
from trusttab.visualization import save_figure

_METRICS = (
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
_HIGHER_IS_BETTER = {"accuracy", "balanced_accuracy", "macro_f1", "roc_auc"}
_RELIABILITY_METRICS = {"brier_score", "ece", "adaptive_ece", "mce", "log_loss"}
_CALIBRATION_COMPARISON_METRICS = (
    "brier_score",
    "ece",
    "adaptive_ece",
    "mce",
    "log_loss",
)
_REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "experiment_type",
    "corruption_type",
    "corruption_level",
    "seed",
}


def _validate_results(results: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    if results.empty:
        raise ValueError("results must contain at least one row.")
    missing = _REQUIRED_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")
    metrics = [metric for metric in _METRICS if metric in results.columns]
    metrics.extend(
        f"calibrated_{metric}"
        for metric in _METRICS
        if f"calibrated_{metric}" in results.columns
    )
    if not metrics:
        raise ValueError(f"results must include at least one metric from {_METRICS}.")

    data = results.copy()
    for column in ("dataset", "model", "experiment_type", "corruption_type"):
        if data[column].isna().any():
            raise ValueError(
                f"results column '{column}' must not contain missing values."
            )
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
    data["corruption_level"] = pd.to_numeric(data["corruption_level"], errors="coerce")
    if not np.isfinite(data["seed"]).all():
        raise ValueError("results seed values must be finite numbers.")
    if (data["seed"] % 1 != 0).any():
        raise ValueError("results seed values must be integers.")
    if not np.isfinite(data["corruption_level"]).all():
        raise ValueError("results corruption_level values must be finite numbers.")
    if (data["experiment_type"] == "clean_baseline").sum() == 0:
        raise ValueError("results must contain clean_baseline rows.")
    for metric in metrics:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data.loc[~np.isfinite(data[metric]), metric] = np.nan
    if data.duplicated(
        [
            "dataset",
            "model",
            "experiment_type",
            "corruption_type",
            "corruption_level",
            "seed",
        ]
    ).any():
        raise ValueError("results contains duplicate dataset/model/scenario/seed rows.")
    return data, metrics


def _metric_comparison(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = [
        "dataset",
        "experiment_type",
        "corruption_type",
        "corruption_level",
        "model",
    ]
    pieces = []
    for metric in metrics:
        summary = (
            data.groupby(keys, dropna=False, sort=True)[metric]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
        summary = summary.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "n": f"{metric}_n",
            }
        )
        pieces.append(summary)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=keys, how="outer", validate="one_to_one")
    return result


def _degradation(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    baseline = data.loc[data["experiment_type"] == "clean_baseline"]
    stress = data.loc[data["experiment_type"] != "clean_baseline"]
    keys = ["dataset", "model", "seed"]
    baseline_columns = [*keys, *metrics]
    joined = stress.merge(
        baseline[baseline_columns],
        on=keys,
        how="left",
        suffixes=("_stress", "_baseline"),
        validate="many_to_one",
        indicator=True,
    )
    if (joined["_merge"] != "both").any():
        missing_rows = int((joined["_merge"] != "both").sum())
        raise ValueError(
            f"{missing_rows} stress result row(s) do not have a matching clean baseline."
        )
    records: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        values = row._asdict()
        base = {
            "dataset": values["dataset"],
            "model": values["model"],
            "experiment_type": values["experiment_type"],
            "corruption_type": values["corruption_type"],
            "corruption_level": values["corruption_level"],
            "seed": values["seed"],
        }
        for metric in metrics:
            baseline_value = values[f"{metric}_baseline"]
            stress_value = values[f"{metric}_stress"]
            if pd.isna(baseline_value) or pd.isna(stress_value):
                absolute = relative = np.nan
            else:
                delta = (
                    baseline_value - stress_value
                    if metric in _HIGHER_IS_BETTER
                    else stress_value - baseline_value
                )
                absolute = float(delta)
                relative = (
                    float(delta / abs(baseline_value))
                    if baseline_value != 0
                    else np.nan
                )
            records.append(
                {
                    **base,
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "stress_value": stress_value,
                    "absolute_degradation": absolute,
                    "relative_degradation": relative,
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "experiment_type",
                "corruption_type",
                "corruption_level",
                "metric",
                "baseline_value",
                "stress_value",
                "absolute_degradation",
                "relative_degradation",
                "mean_degradation",
                "std_degradation",
                "n_seeds",
                "mean_relative_degradation",
                "std_relative_degradation",
            ]
        )
    group_keys = [
        "dataset",
        "model",
        "experiment_type",
        "corruption_type",
        "corruption_level",
        "metric",
    ]
    grouped = (
        result.groupby(group_keys, dropna=False)
        .agg(
            mean_degradation=("absolute_degradation", "mean"),
            std_degradation=("absolute_degradation", "std"),
            n_seeds=("absolute_degradation", "count"),
            mean_relative_degradation=("relative_degradation", "mean"),
            std_relative_degradation=("relative_degradation", "std"),
        )
        .reset_index()
    )
    return result.merge(grouped, on=group_keys, how="left", validate="many_to_one")


def _statistical_comparisons(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    scenario_keys = [
        "dataset",
        "experiment_type",
        "corruption_type",
        "corruption_level",
    ]
    rows = []
    for scenario, group in data.groupby(scenario_keys, sort=True, dropna=False):
        for metric in metrics:
            valid = group.loc[group[metric].notna(), ["model", "seed", metric]].copy()
            if valid["model"].nunique() < 2:
                continue
            # Adapt the unified benchmark scenario to the existing statistics API.
            valid["noise_rate"] = float(scenario[3])
            comparisons = compare_models(
                valid,
                metric,
                unmatched="drop",
                correction="holm",
            )
            for comparison in comparisons.to_dict("records"):
                left = valid.loc[valid["model"] == comparison["model_a"]]
                right = valid.loc[valid["model"] == comparison["model_b"]]
                paired = left.merge(
                    right,
                    on=["seed"],
                    how="inner",
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )
                ci = (np.nan, np.nan)
                ci_seed = bootstrap_seed(
                    "supplementary_model_difference",
                    scenario[0],
                    scenario[1],
                    scenario[2],
                    float(scenario[3]).hex(),
                    metric,
                    comparison["model_a"],
                    comparison["model_b"],
                )
                if len(paired) >= 2:
                    delta = paired[f"{metric}_a"].to_numpy(dtype=float) - paired[
                        f"{metric}_b"
                    ].to_numpy(dtype=float)
                    _, ci_low, ci_high = bootstrap_mean_ci(delta, seed=ci_seed)
                    ci = (ci_low, ci_high)
                rows.append(
                    {
                        "dataset": scenario[0],
                        "experiment_type": scenario[1],
                        "corruption_type": scenario[2],
                        "corruption_level": scenario[3],
                        **comparison,
                        "confidence_interval_low": ci[0],
                        "confidence_interval_high": ci[1],
                        "interval_method": "percentile_bootstrap_seed_resampling",
                        "ci_seed": ci_seed,
                        "ci_resamples": BOOTSTRAP_RESAMPLES,
                        "analysis_role": "supplementary_exploratory_paired_t_test",
                    }
                )
    return pd.DataFrame.from_records(rows)


def _summary_rows(
    data: pd.DataFrame,
    metrics: list[str],
    comparison: pd.DataFrame,
    degradation: pd.DataFrame,
    title: str,
) -> pd.DataFrame:
    baseline = data.loc[data["experiment_type"] == "clean_baseline"]
    performance_metrics = [
        metric
        for metric in metrics
        if metric.removeprefix("calibrated_") in _HIGHER_IS_BETTER
    ]
    calibration_metrics = [
        metric
        for metric in metrics
        if metric.removeprefix("calibrated_") in _RELIABILITY_METRICS
    ]
    calibrated_reliability_metrics = [
        metric
        for metric in calibration_metrics
        if metric.startswith("calibrated_") and baseline[metric].notna().any()
    ]
    stress = degradation.loc[degradation["absolute_degradation"].notna()]

    def best_model(values: pd.Series, direction: str) -> str:
        values = values.dropna()
        if values.empty:
            return "unavailable"
        ordered = values.sort_values(ascending=direction == "lower")
        return str(ordered.index[0])

    performance = (
        baseline.groupby("model")[performance_metrics].mean().mean(axis=1)
        if performance_metrics
        else pd.Series(dtype=float)
    )
    if calibrated_reliability_metrics:
        # Rank within each dataset/metric to avoid mixing incompatible units.
        calibration = baseline.groupby(["dataset", "model"])[
            calibrated_reliability_metrics
        ].mean()
        ranks = calibration.groupby(level="dataset").rank(ascending=True).mean(axis=1)
        calibration_score = ranks.groupby(level="model").mean()
    else:
        calibration_score = pd.Series(dtype=float)
    if not stress.empty:
        robustness = (
            stress.loc[stress["metric"].isin(performance_metrics)]
            .groupby("model")["absolute_degradation"]
            .mean()
        )
    else:
        robustness = pd.Series(dtype=float)
    performance_stress = stress.loc[stress["metric"].isin(performance_metrics)]
    worst = (
        performance_stress.groupby(
            ["experiment_type", "corruption_type", "corruption_level", "metric"]
        )["absolute_degradation"]
        .mean()
        .sort_values(ascending=False)
        if not performance_stress.empty
        else pd.Series(dtype=float)
    )
    rows = [
        ("title", title),
        ("datasets_evaluated", int(data["dataset"].nunique())),
        ("models_evaluated", int(data["model"].nunique())),
        ("random_seeds", int(data["seed"].nunique())),
        (
            "stress_scenarios",
            int(
                data.loc[
                    data["experiment_type"] != "clean_baseline",
                    ["experiment_type", "corruption_type", "corruption_level"],
                ]
                .drop_duplicates()
                .shape[0]
            ),
        ),
        ("overall_best_performing_model", best_model(performance, "higher")),
        ("most_robust_model", best_model(robustness, "lower")),
        ("best_calibrated_model", best_model(calibration_score, "lower")),
        (
            "largest_mean_performance_degradation",
            str(worst.index[0]) if not worst.empty else "unavailable",
        ),
        (
            "largest_degradation_value",
            float(worst.iloc[0]) if not worst.empty else np.nan,
        ),
        (
            "paired_model_comparisons",
            int(comparison["p_value"].notna().sum()) if not comparison.empty else 0,
        ),
        (
            "significant_after_holm",
            int(comparison["reject_null"].fillna(False).sum())
            if not comparison.empty
            else 0,
        ),
    ]
    rows.extend(
        (f"mean_baseline_{metric}", float(baseline[metric].mean()))
        for metric in metrics
    )
    return pd.DataFrame(rows, columns=["item", "value"])


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value) if pd.notna(value) else "—" for value in row]
        lines.append(
            "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
        )
    return "\n".join(lines)


def _calibration_status_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count condition outcomes without dropping failed calibration rows."""
    if "calibration_status" not in data.columns:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "experiment_type",
                "corruption_type",
                "corruption_level",
                "calibration_method",
                "calibration_status",
                "n_conditions",
            ]
        )
    keys = [
        column
        for column in (
            "dataset",
            "model",
            "experiment_type",
            "corruption_type",
            "corruption_level",
            "calibration_method",
            "calibration_status",
        )
        if column in data.columns
    ]
    return (
        data.groupby(keys, dropna=False, sort=True)
        .size()
        .rename("n_conditions")
        .reset_index()
    )


def _calibration_condition_results(data: pd.DataFrame) -> pd.DataFrame:
    """Keep raw/calibrated probability metrics paired by original condition.

    This table is descriptive. It retains failed/disabled rows and their NaNs
    rather than replacing missing calibrated values or silently dropping them.
    """
    metric_pairs = [
        (metric, f"calibrated_{metric}")
        for metric in _CALIBRATION_COMPARISON_METRICS
        if metric in data.columns and f"calibrated_{metric}" in data.columns
    ]
    identifiers = [
        column
        for column in (
            "dataset",
            "model",
            "experiment_type",
            "corruption_type",
            "corruption_level",
            "seed",
            "condition_id",
            "calibration_method",
            "calibration_status",
            "calibration_error",
        )
        if column in data.columns
    ]
    output_columns = [*identifiers]
    for raw, calibrated in metric_pairs:
        output_columns.extend([f"raw_{raw}", calibrated, f"calibrated_minus_raw_{raw}"])
    if "experiment_type" in data.columns:
        selected = data.loc[data["experiment_type"] == "label_noise"]
    else:
        selected = data
    result = selected.loc[:, identifiers].copy()
    for raw, calibrated in metric_pairs:
        result[f"raw_{raw}"] = selected[raw]
        result[calibrated] = selected[calibrated]
        result[f"calibrated_minus_raw_{raw}"] = selected[calibrated] - selected[raw]
    return result.reindex(columns=output_columns)


def _plot_metric(
    data: pd.DataFrame,
    metric: str,
    path: Path,
    *,
    experiment_type: str | None = None,
    title: str,
    degradation: pd.DataFrame | None = None,
) -> None:
    figure = Figure(figsize=(9, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    if degradation is not None:
        selected = degradation.loc[
            (degradation["metric"] == metric)
            & (degradation["experiment_type"] != "clean_baseline")
        ]
        if selected.empty:
            axes.text(
                0.5, 0.5, "No stress comparisons available", ha="center", va="center"
            )
        else:
            for (
                dataset,
                model,
                scenario_type,
                corruption_type,
            ), group in selected.groupby(
                ["dataset", "model", "experiment_type", "corruption_type"]
            ):
                summary = group.groupby("corruption_level")["absolute_degradation"].agg(
                    ["mean", "std"]
                )
                axes.errorbar(
                    summary.index,
                    summary["mean"],
                    yerr=summary["std"].fillna(0),
                    marker="o",
                    capsize=3,
                    label=f"{model} ({dataset}, {scenario_type}: {corruption_type})",
                )
            axes.axhline(0, color="black", linestyle="--", linewidth=1)
            axes.legend()
        axes.set(
            xlabel="Corruption level / shift magnitude", ylabel=f"{metric} degradation"
        )
    else:
        selected = data.loc[data[metric].notna()]
        if experiment_type == "clean_baseline":
            selected = selected.loc[selected["experiment_type"] == "clean_baseline"]
            if selected.empty:
                axes.text(
                    0.5,
                    0.5,
                    "No baseline observations available",
                    ha="center",
                    va="center",
                )
            else:
                grouped = selected.groupby(["dataset", "model"])[metric].agg(
                    ["mean", "std"]
                )
                labels = [f"{model}\n({dataset})" for dataset, model in grouped.index]
                axes.bar(
                    np.arange(len(grouped)),
                    grouped["mean"],
                    yerr=grouped["std"].fillna(0),
                    capsize=3,
                    tick_label=labels,
                )
                axes.tick_params(axis="x", labelrotation=25)
            axes.set_ylabel(metric)
        elif experiment_type is not None:
            selected = selected.loc[selected["experiment_type"] == experiment_type]
            if selected.empty:
                axes.text(
                    0.5, 0.5, "No observations available", ha="center", va="center"
                )
            else:
                group_keys = ["dataset", "model"]
                if experiment_type == "feature_shift":
                    group_keys.append("corruption_type")
                for group_key, group in selected.groupby(group_keys, sort=True):
                    summary = group.groupby("corruption_level")[metric].agg(
                        ["mean", "std"]
                    )
                    axes.errorbar(
                        summary.index,
                        summary["mean"],
                        yerr=summary["std"].fillna(0),
                        marker="o",
                        capsize=3,
                        label=" / ".join(map(str, group_key)),
                    )
                axes.legend()
            axes.set(xlabel="Level / magnitude", ylabel=metric)
        elif selected.empty:
            axes.text(0.5, 0.5, "No observations available", ha="center", va="center")
        else:
            for (dataset, model), group in selected.groupby(["dataset", "model"]):
                summary = group.groupby("experiment_type")[metric].agg(["mean", "std"])
                axes.errorbar(
                    summary.index,
                    summary["mean"],
                    yerr=summary["std"].fillna(0),
                    marker="o",
                    capsize=3,
                    label=f"{model} ({dataset})",
                )
            axes.legend()
            axes.set(xlabel="Experiment type", ylabel=metric)
    axes.set_title(title)
    figure.tight_layout()
    save_figure(figure, path)
    figure.clear()


def _write_figures(
    data: pd.DataFrame, metrics: list[str], degradation: pd.DataFrame, directory: Path
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    primary = next(
        (m for m in ("accuracy", "balanced_accuracy", "macro_f1") if m in metrics),
        metrics[0],
    )
    calibration = next(
        (
            m
            for m in ("ece", "adaptive_ece", "mce", "brier_score", "log_loss")
            if m in metrics
        ),
        None,
    )
    figure_paths = [
        (
            "model_comparison.png",
            primary,
            "clean_baseline",
            "Model performance comparison",
            None,
        ),
        (
            "robustness_degradation.png",
            primary,
            None,
            "Performance degradation from baseline",
            degradation,
        ),
    ]
    saved = []
    for filename, metric, experiment_type, title, degradation_data in figure_paths:
        _plot_metric(
            data,
            metric,
            directory / filename,
            experiment_type=experiment_type,
            title=title,
            degradation=degradation_data,
        )
        saved.append(f"figures/{filename}")

    noise_data = data.loc[data["experiment_type"].eq("label_noise")]
    for metric, filename, title in (
        ("accuracy", "noise_robustness.png", "Accuracy under training-label noise"),
        ("ece", "ece_vs_noise.png", "ECE under training-label noise"),
        (
            "brier_score",
            "brier_score_vs_noise.png",
            "Brier score under training-label noise",
        ),
        ("macro_f1", "macro_f1_vs_noise.png", "Macro-F1 under training-label noise"),
    ):
        if metric in noise_data and noise_data[metric].notna().any():
            path = directory / filename
            _plot_noise_metric(noise_data, metric, path, title=title)
            saved.append(f"figures/{filename}")

    if not noise_data.empty:
        _plot_model_ranking_heatmap(noise_data, directory / "model_ranking_heatmap.png")
        saved.append("figures/model_ranking_heatmap.png")
        _plot_cross_dataset_comparison(
            noise_data, directory / "cross_dataset_comparison.png"
        )
        saved.append("figures/cross_dataset_comparison.png")

    if data["experiment_type"].eq("feature_shift").any():
        _plot_metric(
            data,
            primary,
            directory / "feature_shift_robustness.png",
            experiment_type="feature_shift",
            title="Performance under feature shifts",
        )
        saved.append("figures/feature_shift_robustness.png")

    if calibration is not None and _has_successful_calibration(data, calibration):
        _plot_calibration_comparison(
            data, calibration, directory / "calibration_comparison.png"
        )
        saved.append("figures/calibration_comparison.png")
    return saved


def _has_successful_calibration(data: pd.DataFrame, metric: str) -> bool:
    calibrated = f"calibrated_{metric}"
    if calibrated not in data:
        return False
    valid = data[metric].notna() & data[calibrated].notna()
    if "calibration_status" in data:
        valid &= data["calibration_status"].eq("succeeded")
    return bool(valid.any())


def _metric_curve_summary(
    group: pd.DataFrame, metric: str, seed_keys: tuple[object, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    levels, means, lows, highs = [], [], [], []
    for level, condition in group.groupby("corruption_level", sort=True):
        values = (
            pd.to_numeric(condition[metric], errors="coerce").dropna().to_numpy(float)
        )
        if not len(values):
            continue
        ci_seed = bootstrap_seed("figure_mean", *seed_keys, float(level).hex())
        mean, low, high = bootstrap_mean_ci(values, seed=ci_seed)
        levels.append(float(level))
        means.append(mean)
        lows.append(low)
        highs.append(high)
    return (
        np.asarray(levels),
        np.asarray(means),
        np.asarray(lows),
        np.asarray(highs),
    )


def _plot_noise_metric(
    data: pd.DataFrame, metric: str, path: Path, *, title: str
) -> None:
    datasets = sorted(data["dataset"].unique())
    figure = Figure(figsize=(6.5 * len(datasets), 5))
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(1, len(datasets), squeeze=False)[0])
    for axis, dataset in zip(axes, datasets, strict=True):
        selected = data.loc[data["dataset"].eq(dataset)]
        for model in sorted(selected["model"].unique()):
            group = selected.loc[selected["model"].eq(model)]
            levels, means, lows, highs = _metric_curve_summary(
                group, metric, ("noise_curve", dataset, model, metric)
            )
            if not len(levels):
                continue
            axis.plot(levels * 100, means, marker="o", label=model)
            if np.isfinite(lows).all() and np.isfinite(highs).all():
                axis.fill_between(levels * 100, lows, highs, alpha=0.16)
        axis.set(
            title=dataset.replace("_", " ").title(), xlabel="Training-label noise (%)"
        )
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_xticks(sorted(data["corruption_level"].unique() * 100))
        axis.legend(title="Model")
    figure.suptitle(title)
    figure.tight_layout()
    save_figure(figure, path)
    figure.clear()


def _plot_model_ranking_heatmap(data: pd.DataFrame, path: Path) -> None:
    datasets = sorted(data["dataset"].unique())
    figure = Figure(figsize=(12, max(4, 3.2 * len(datasets))))
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(len(datasets), 2, squeeze=False))
    for row, dataset in enumerate(datasets):
        subset = data.loc[data["dataset"].eq(dataset)]
        for column, metric in enumerate(("accuracy", "ece")):
            axis = axes[row, column]
            if metric not in subset:
                axis.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
                continue
            means = subset.groupby(["model", "corruption_level"])[metric].mean()
            ranks = means.groupby(level="corruption_level").rank(
                ascending=metric not in _HIGHER_IS_BETTER, method="average"
            )
            models = sorted(subset["model"].unique())
            levels = sorted(subset["corruption_level"].unique())
            matrix = np.full((len(models), len(levels)), np.nan)
            for i, model in enumerate(models):
                for j, level in enumerate(levels):
                    key = (model, level)
                    if key in ranks.index:
                        matrix[i, j] = ranks.loc[key]
            image = axis.imshow(
                matrix,
                aspect="auto",
                vmin=1,
                vmax=max(1, len(models)),
                cmap="viridis_r",
            )
            for i in range(len(models)):
                for j in range(len(levels)):
                    if np.isfinite(matrix[i, j]):
                        axis.text(
                            j,
                            i,
                            f"{matrix[i, j]:.1f}",
                            ha="center",
                            va="center",
                            color="white",
                        )
            axis.set(
                title=f"{dataset.replace('_', ' ').title()} — {metric.replace('_', ' ').title()} rank",
                xticks=np.arange(len(levels)),
                xticklabels=[f"{level:.0%}" for level in levels],
                yticks=np.arange(len(models)),
                yticklabels=models,
                xlabel="Training-label noise",
            )
            figure.colorbar(
                image, ax=axis, ticks=range(1, len(models) + 1), label="Rank (1 = best)"
            )
    figure.tight_layout()
    save_figure(figure, path)
    figure.clear()


def _plot_cross_dataset_comparison(data: pd.DataFrame, path: Path) -> None:
    datasets = sorted(data["dataset"].unique())
    metrics = ("accuracy", "ece")
    figure = Figure(figsize=(6.5 * len(datasets), 9))
    FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(len(metrics), len(datasets), squeeze=False))
    for row, metric in enumerate(metrics):
        for column, dataset in enumerate(datasets):
            axis = axes[row, column]
            subset = data.loc[data["dataset"].eq(dataset)]
            if metric not in subset:
                axis.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
                continue
            for model in sorted(subset["model"].unique()):
                group = subset.loc[subset["model"].eq(model)]
                levels, means, lows, highs = _metric_curve_summary(
                    group, metric, ("cross_dataset", dataset, model, metric)
                )
                axis.plot(levels * 100, means, marker="o", label=model)
                if np.isfinite(lows).all() and np.isfinite(highs).all():
                    axis.fill_between(levels * 100, lows, highs, alpha=0.16)
            axis.set(
                title=dataset.replace("_", " ").title(),
                xlabel="Training-label noise (%)",
                ylabel=metric.replace("_", " ").title(),
                xticks=sorted(subset["corruption_level"].unique() * 100),
            )
            if row == 0:
                axis.legend(title="Model")
    figure.suptitle("Cross-dataset performance and calibration comparison")
    figure.tight_layout()
    save_figure(figure, path)
    figure.clear()


def _plot_calibration_comparison(data: pd.DataFrame, metric: str, path: Path) -> None:
    datasets = sorted(data["dataset"].unique())
    calibrated_metric = f"calibrated_{metric}"
    figure = Figure(figsize=(6.5 * len(datasets), 5))
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(1, len(datasets), squeeze=False)[0])
    valid = data.loc[data["experiment_type"].eq("label_noise")].copy()
    if "calibration_status" in valid:
        valid = valid.loc[valid["calibration_status"].eq("succeeded")]
    valid = valid.loc[valid[metric].notna() & valid[calibrated_metric].notna()]
    for axis, dataset in zip(axes, datasets, strict=True):
        selected = valid.loc[valid["dataset"].eq(dataset)]
        for model in sorted(selected["model"].unique()):
            group = selected.loc[selected["model"].eq(model)]
            for column, label, style in (
                (metric, f"{model} raw", "-"),
                (calibrated_metric, f"{model} Platt", "--"),
            ):
                levels, means, lows, highs = _metric_curve_summary(
                    group.assign(**{metric: group[column]}),
                    metric,
                    ("raw_calibrated", dataset, model, column),
                )
                axis.plot(levels * 100, means, linestyle=style, marker="o", label=label)
                if np.isfinite(lows).all() and np.isfinite(highs).all():
                    axis.fill_between(levels * 100, lows, highs, alpha=0.12)
        axis.set(
            title=dataset.replace("_", " ").title(),
            xlabel="Training-label noise (%)",
            ylabel=metric.upper(),
        )
        axis.legend(title="Prediction set")
    figure.suptitle(
        "Raw versus Platt-calibrated ECE"
        if metric == "ece"
        else f"Raw versus calibrated {metric}"
    )
    figure.tight_layout()
    save_figure(figure, path)
    figure.clear()


def _write_critical_difference_diagrams(
    nemenyi: pd.DataFrame, directory: Path
) -> list[str]:
    """Write rank-axis CD plots only for groups passing gated Nemenyi."""
    valid = nemenyi.loc[nemenyi["status"].eq("ok")]
    if valid.empty:
        return []
    target = directory / "critical_difference"
    target.mkdir(parents=True, exist_ok=True)
    saved = []
    for (dataset, metric, level), group in valid.groupby(
        ["dataset", "metric", "noise_level"], sort=True
    ):
        ranks = {}
        nonsignificant = {}
        for row in group.itertuples(index=False):
            ranks[row.model_a] = row.average_rank_a
            ranks[row.model_b] = row.average_rank_b
            nonsignificant[frozenset((row.model_a, row.model_b))] = not bool(
                row.reject_null
            )
        ordered = sorted(ranks.items(), key=lambda item: item[1])
        cd = float(group["critical_difference"].iloc[0])
        model_names = list(ranks)
        cliques = []
        for size in range(2, len(model_names) + 1):
            for subset in combinations(model_names, size):
                if all(
                    nonsignificant.get(frozenset(pair), False)
                    for pair in combinations(subset, 2)
                ):
                    cliques.append(frozenset(subset))
        maximal_cliques = [
            clique for clique in cliques if not any(clique < other for other in cliques)
        ]
        figure = Figure(figsize=(8, 2.6))
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.set_xlim(0.5, len(ranks) + 0.5)
        axes.set_ylim(-0.8, max(1.0, 0.65 + 0.15 * len(maximal_cliques)))
        axes.set_yticks([])
        axes.set_xlabel("Average rank (1 is best)")
        axes.set_title(f"Critical difference: {dataset}, {metric}, noise={level:.0%}")
        axes.axhline(0, color="black", linewidth=1)
        for index, (model, rank) in enumerate(ordered):
            axes.plot(rank, 0, "o", color="black")
            axes.text(
                rank,
                0.12 if index % 2 == 0 else -0.22,
                model,
                ha="center",
                va="center",
                fontsize=8,
            )
        for index, clique in enumerate(maximal_cliques):
            clique_ranks = [ranks[model] for model in clique]
            y = 0.35 + index * 0.15
            axes.plot(
                [min(clique_ranks), max(clique_ranks)],
                [y, y],
                color="black",
                linewidth=2,
            )
            for rank in clique_ranks:
                axes.plot([rank, rank], [0, y], color="black", linewidth=0.7)
        cd_y = 0.48 + len(maximal_cliques) * 0.15
        axes.plot(
            [1, min(len(ranks), 1 + cd)], [cd_y, cd_y], color="tab:red", linewidth=2
        )
        axes.text(1, cd_y + 0.08, f"CD = {cd:.3f}", fontsize=8, color="tab:red")
        figure.tight_layout()

        def safe(value: object) -> str:
            return "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(value)
            )

        filename = (
            f"cd_{safe(dataset)}_{safe(metric)}_{int(round(level * 100)):02d}.png"
        )
        save_figure(figure, target / filename)
        figure.clear()
        saved.append(f"figures/critical_difference/{filename}")
    return saved


def generate_robustness_report(
    results: pd.DataFrame,
    output_dir: str | Path,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate Markdown, CSV, and PNG artifacts from unified benchmark rows.

    Aggregations keep dataset, experiment type, corruption type, and level
    separate. Degradation is positive when the stress result is worse than its
    clean baseline, with metric direction taken into account. Paired model
    comparisons use matched dataset/seed rows and Holm correction per scenario
    and metric. The returned mapping contains artifact paths and summary tables.
    """
    data, metrics = _validate_results(results)
    output = Path(output_dir)
    figures_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    report_title = title or "TrustTab Robustness Report"

    model_comparison = _metric_comparison(data, metrics)
    degradation = _degradation(data, metrics)
    inferential_metrics = [
        metric for metric in metrics if not metric.startswith("calibrated_")
    ]
    comparisons = _statistical_comparisons(data, inferential_metrics)
    summary = _summary_rows(data, metrics, comparisons, degradation, report_title)
    calibration_status = _calibration_status_summary(data)
    calibration_conditions = _calibration_condition_results(data)
    research_analysis = run_research_analysis(data)
    figure_files = _write_figures(data, metrics, degradation, figures_dir)
    cd_files = _write_critical_difference_diagrams(
        research_analysis["nemenyi_results"], figures_dir
    )

    summary.to_csv(output / "summary.csv", index=False)
    model_comparison.to_csv(output / "model_comparison.csv", index=False)
    degradation.to_csv(output / "robustness_degradation.csv", index=False)
    comparisons.to_csv(output / "statistical_comparisons.csv", index=False)
    calibration_status.to_csv(output / "calibration_status.csv", index=False)
    calibration_conditions.to_csv(output / "calibration_comparison.csv", index=False)
    for name, table in research_analysis.items():
        filename = (
            "degradation_slope_comparisons.csv"
            if name == "degradation_slope_comparisons"
            else f"{name}.csv"
        )
        table.to_csv(output / filename, index=False)

    markdown = [
        f"# {report_title}",
        "",
        "## Executive summary",
        "",
        _markdown_table(summary),
        "",
        "## Statistical model comparisons",
        "",
        "The compatibility table contains supplementary exploratory paired t-tests within each fixed dataset, using matched seeds and Holm correction within each metric/scenario. At five seeds, the normality assumption is difficult to assess; these are not the blueprint's primary omnibus model tests and should not be presented as confirmatory. Their 95% intervals are percentile bootstrap intervals over paired seed differences. The pre-specified model comparison is the seed-blocked Friedman/Nemenyi analysis below.",
        "",
        _markdown_table(comparisons),
        "",
        "## Calibration outcomes",
        "",
        "Condition counts retain disabled, succeeded, failed, and not_applicable calibration statuses. The paired calibration table compares raw and calibrated probability metrics on each original dataset/model/noise/seed row. It is descriptive; failed or unavailable calibrated values remain NaN and are not imputed. A raw-versus-Platt figure is emitted only when successful matched calibrated values exist. If calibration is disabled, the report explicitly has no RQ4 evidence.",
        "",
        _markdown_table(calibration_status),
        "",
        "## Seed-blocked research inference",
        "",
        "Friedman blocks are the matched seeds within one dataset, noise level, and metric. Holm correction is applied to the five noise-level omnibus tests within each dataset/metric. Nemenyi runs only after a significant Holm-adjusted omnibus test; its pairwise p-values are already adjusted and are not Holm-corrected again. Calibration Wilcoxon families are separate by dataset/metric across model/noise comparisons. Five seeds make these tests exploratory and low-powered; two fixed datasets do not support population-of-datasets inference. Failed calibration rows remain unavailable.",
        "",
        _markdown_table(research_analysis["friedman_results"]),
        "",
        "See `calibration_comparison.csv` for per-condition raw/calibrated Brier, ECE, adaptive ECE, MCE, and log-loss values and their calibrated-minus-raw differences. The four noise curves and cross-dataset panels use deterministic percentile bootstrap 95% intervals over seed-level observations (10,000 resamples); these intervals remain small-sample estimates.",
        "",
        "## Figures",
        "",
        *[f"- [{Path(path).name}]({path})" for path in figure_files],
        *[f"- [{Path(path).name}]({path})" for path in cd_files],
        "",
        "Metric curves include accuracy, ECE, Brier score, and Macro-F1 where those metrics are present. The report also includes model-rank heatmaps and a cross-dataset panel. Reliability diagrams and confidence histograms are not generated from aggregate result rows: this result schema does not retain per-observation labels and predicted probabilities. A raw-versus-Platt figure is omitted unless successful matched calibrated values exist.",
        "",
        "## Interpretation",
        "",
        "Positive absolute degradation indicates worse performance under stress, with direction adjusted for whether higher or lower metric values are better. Relative degradation divides by the absolute baseline value; it is undefined when that baseline is zero. Tables aggregate over seeds while keeping datasets and stress scenarios separate.",
        "",
    ]
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    return {
        "output_dir": output,
        "summary": summary,
        "model_comparison": model_comparison,
        "robustness_degradation": degradation,
        "statistical_comparisons": comparisons,
        "calibration_status": calibration_status,
        "calibration_comparison": calibration_conditions,
        **research_analysis,
        "artifacts": [
            "report.md",
            "summary.csv",
            "model_comparison.csv",
            "robustness_degradation.csv",
            "statistical_comparisons.csv",
            "calibration_status.csv",
            "calibration_comparison.csv",
            "friedman_results.csv",
            "nemenyi_results.csv",
            "wilcoxon_calibration_results.csv",
            "degradation_slope_results.csv",
            "degradation_slope_comparisons.csv",
            "effect_sizes.csv",
            "statistical_summary.csv",
            *figure_files,
            *cd_files,
        ],
    }
