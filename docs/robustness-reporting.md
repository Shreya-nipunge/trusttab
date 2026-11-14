# Automated Robustness Reporting

`generate_robustness_report` converts the unified benchmark result DataFrame
into a reproducible set of summary tables, statistical comparisons, a Markdown
report, and diagnostic figures. It consumes results from
`run_robustness_benchmark` and does not retrain models.

## Usage

```python
from trusttab.experiments import run_robustness_benchmark
from trusttab.reporting import generate_robustness_report

results = run_robustness_benchmark(
    dataset_names=["german_credit"],
    label_noise_levels=[0.0, 0.1, 0.2],
    feature_shifts=[
        {"type": "mean_shift", "features": ["age"], "magnitude": 0.5}
    ],
    seeds=[1, 2, 3, 4, 5],
)
report = generate_robustness_report(
    results,
    output_dir="results/robustness_report",
    title="German Credit robustness analysis",
)
print(report["artifacts"])
```

`output_dir` is created if needed. The function returns a dictionary with the
output path, summary DataFrames, and relative artifact paths. Inputs are copied
before numeric normalization; the results DataFrame is not mutated.

## Required input columns

The input must contain one row per dataset, model, scenario, and seed, with:

- `dataset`, `model`, `seed`
- `experiment_type` (`clean_baseline`, `label_noise`, or `feature_shift`)
- `corruption_type`, `corruption_level`
- At least one of `accuracy`, `balanced_accuracy`, `macro_f1`, `roc_auc`,
  `brier_score`, `ece`, `adaptive_ece`, `mce`, or `log_loss`

Clean baseline rows are required. Every stress result must match a baseline
with the same dataset, model, and seed. NaN metric values are allowed and are
omitted from the corresponding means, standard deviations, and tests. Missing
metric columns are allowed.

## Generated artifacts

All reports include `calibration_status.csv` and `calibration_comparison.csv`.
The status table counts rows by scenario and calibration outcome. The
condition-level table retains calibration status/error and descriptive paired
raw-versus-calibrated probability metrics; unavailable values stay NaN.

```text
robustness_report/
├── report.md
├── summary.csv
├── model_comparison.csv
├── robustness_degradation.csv
├── statistical_comparisons.csv
└── figures/
    ├── model_comparison.png
    ├── noise_robustness.png
    ├── feature_shift_robustness.png
    ├── calibration_comparison.png   # when a reliability metric is present
    └── robustness_degradation.png
```

`model_comparison.csv` reports mean, sample standard deviation, and non-missing
sample count for each raw metric and each available `calibrated_` metric,
separately for dataset, model, experiment type, corruption type, and level.
The executive summary ranks performance on clean baselines and uses calibrated
reliability metrics for its calibration ranking when those columns are present;
otherwise it uses raw reliability metrics. Metrics with different scales are
ranked within dataset rather than averaged directly.

For label-noise rows, `n_corrupted` records the realized number of independent
Bernoulli flips in the training labels. It varies randomly around the expected
count `noise_rate × n_train`; it is not constrained to a fixed rounded count.
Evaluation labels remain clean.

Calibration-enabled benchmark rows keep raw metrics in standard columns and
add `calibrated_`-prefixed metrics on the same condition row. The generated
`calibration_comparison.csv` contains paired per-condition raw/calibrated
Brier, ECE, adaptive ECE, MCE, and log-loss values and calibrated-minus-raw
differences. This table is descriptive and does not add a significance test.
The report also writes a seed-level Wilcoxon analysis of raw versus calibrated
ECE and Brier values, using only successful calibration rows and matched
seeds. Failed values remain excluded and visible in the calibration status
table. The inferential comparison unit is a seed-level model run, never a test
observation. See [statistics.md](statistics.md) for the families and limits.
`calibration_status.csv` retains each condition's status and error text; the
report also includes counts by status. Failed or disabled calibrated values
remain missing and are not imputed. Raw results remain in the original columns.
Rows labeled `not_applicable` indicate calibration was not evaluated for that
experiment type, such as feature-shift rows in the current runner.

## Interpreting degradation and tests

Degradation is defined so positive values always indicate worse performance:
baseline minus stressed value for accuracy, balanced accuracy, macro F1, and
ROC-AUC; stressed minus baseline for Brier score, ECE, adaptive ECE, MCE, and
log loss. Relative degradation divides by the absolute baseline metric value
and is NaN when the baseline is zero. Per-seed values and the mean, sample
standard deviation, and count across seeds are included for absolute and
relative degradation. Scenario and dataset groups are never combined in these
tables.

Legacy pairwise model comparisons use matched seeds within each dataset,
scenario, and metric; no test pools different datasets. The six Phase 7
tables add per-dataset seed-blocked Friedman tests, gated Nemenyi comparisons,
calibration Wilcoxon tests, normalized degradation slopes, effect sizes, and
small-sample confidence intervals. Holm families and the Friedman/Nemenyi
gate are described in [statistics.md](statistics.md). Five seeds provide low
power; two fixed datasets do not support dataset-population inference.
