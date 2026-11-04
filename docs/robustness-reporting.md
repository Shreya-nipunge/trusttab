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
sample count for each metric, separately for dataset, model, experiment type,
corruption type, and level. The executive summary ranks performance on clean
baselines and calibration by within-dataset metric ranks so differently scaled
reliability measures are not averaged directly.

## Interpreting degradation and tests

Degradation is defined so positive values always indicate worse performance:
baseline minus stressed value for accuracy, balanced accuracy, macro F1, and
ROC-AUC; stressed minus baseline for Brier score, ECE, adaptive ECE, MCE, and
log loss. Relative degradation divides by the absolute baseline metric value
and is NaN when the baseline is zero. Per-seed values and the mean, sample
standard deviation, and count across seeds are included for absolute and
relative degradation. Scenario and dataset groups are never combined in these
tables.

Model comparisons use paired t-tests on matched dataset/seed observations,
through TrustTab's existing statistical utilities. Missing metric observations
are excluded and unmatched rows are not treated as pairs. Holm correction is
applied within each metric and scenario family. The report includes adjusted
p-values and 95% confidence intervals for the mean paired difference (model A
minus model B). Statistical significance does not establish practical
importance; consider the difference magnitude and interval alongside the
adjusted p-value. Paired tests assume independent pairs and approximately
normal paired differences; inspect distributions or use TrustTab's Wilcoxon
utility for non-normal differences when needed.
