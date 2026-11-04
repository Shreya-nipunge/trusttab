"""Automated analysis and artifact generation for robustness benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from trusttab.statistics import compare_models, compare_paired_scores
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
    scenario_keys = ["experiment_type", "corruption_type", "corruption_level"]
    rows = []
    for scenario, group in data.groupby(scenario_keys, sort=True, dropna=False):
        for metric in metrics:
            valid = group.loc[
                group[metric].notna(), ["dataset", "model", "seed", metric]
            ].copy()
            if valid["model"].nunique() < 2:
                continue
            # Adapt the unified benchmark scenario to the existing statistics API.
            valid["noise_rate"] = float(scenario[2])
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
                    on=["dataset", "seed"],
                    how="inner",
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )
                ci = (np.nan, np.nan)
                if len(paired) >= 2:
                    ci = compare_paired_scores(
                        paired[f"{metric}_a"].to_numpy(dtype=float),
                        paired[f"{metric}_b"].to_numpy(dtype=float),
                    ).confidence_interval
                rows.append(
                    {
                        "experiment_type": scenario[0],
                        "corruption_type": scenario[1],
                        "corruption_level": scenario[2],
                        **comparison,
                        "confidence_interval_low": ci[0],
                        "confidence_interval_high": ci[1],
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
    performance_metrics = [m for m in _HIGHER_IS_BETTER if m in metrics]
    calibration_metrics = [m for m in _RELIABILITY_METRICS if m in metrics]
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
    if calibration_metrics:
        # Rank within each dataset/metric to avoid mixing incompatible units.
        calibration = baseline.groupby(["dataset", "model"])[calibration_metrics].mean()
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
            "noise_robustness.png",
            primary,
            "label_noise",
            "Performance under label noise",
            None,
        ),
        (
            "feature_shift_robustness.png",
            primary,
            "feature_shift",
            "Performance under feature shifts",
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
    if calibration is not None:
        figure_paths.append(
            (
                "calibration_comparison.png",
                calibration,
                None,
                "Calibration comparison across scenarios",
                None,
            )
        )
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
    comparisons = _statistical_comparisons(data, metrics)
    summary = _summary_rows(data, metrics, comparisons, degradation, report_title)
    figure_files = _write_figures(data, metrics, degradation, figures_dir)

    summary.to_csv(output / "summary.csv", index=False)
    model_comparison.to_csv(output / "model_comparison.csv", index=False)
    degradation.to_csv(output / "robustness_degradation.csv", index=False)
    comparisons.to_csv(output / "statistical_comparisons.csv", index=False)

    markdown = [
        f"# {report_title}",
        "",
        "## Executive summary",
        "",
        _markdown_table(summary),
        "",
        "## Statistical model comparisons",
        "",
        "Paired tests use matched dataset and seed observations. Holm adjusted p-values are calculated within each metric and corruption scenario. Confidence intervals are 95% intervals for the mean paired difference (model A minus model B).",
        "",
        _markdown_table(comparisons),
        "",
        "## Figures",
        "",
        *[f"- [{Path(path).name}]({path})" for path in figure_files],
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
        "artifacts": [
            "report.md",
            "summary.csv",
            "model_comparison.csv",
            "robustness_degradation.csv",
            "statistical_comparisons.csv",
            *figure_files,
        ],
    }
