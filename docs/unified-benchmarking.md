# Unified Robustness Benchmarking

TrustTab's unified runner evaluates clean performance alongside training-label
noise and test-time feature shifts in a single result table. For each dataset
and seed, it creates one stratified split. For each model, it fits one clean
baseline model. That fitted model supplies the clean baseline and all
feature-shift evaluations; positive label-noise levels fit additional models
using corrupted copies of the training labels. Test labels are never changed.

## Supported scenarios

- **Clean baseline:** original training labels and clean test features.
- **Label noise:** symmetric label corruption on training labels only.
- **Feature shift:** configured feature transformations on independent copies
  of the clean test features; training data remains unchanged.

Feature shifts reuse the existing `apply_feature_shifts` implementation and
accept Gaussian noise, mean shift, variance scaling, categorical probability
redistribution, and category replacement. A feature-shift configuration may
select types and magnitudes as a cross product. If no feature list is supplied,
numeric transforms use numeric non-boolean columns and categorical transforms
use non-numeric or boolean columns. Explicit feature selection is recommended
for controlled experiments.

## YAML configuration

```yaml
project:
  name: TrustTab
  version: "0.1.0"

dataset:
  names: [german_credit, adult]

experiments:
  test_size: 0.2
  seeds: [0, 1, 2, 3, 4]
  label_noise:
    enabled: true
    levels: [0.0, 0.1, 0.2, 0.3]
  feature_shift:
    enabled: true
    types: [mean_shift, variance_scaling]
    magnitudes: [0.5, 1.0, 1.5]
    features:
      mean_shift: [age]
      variance_scaling: [age]

models:
  - logistic_regression
  - random_forest

output:
  directory: results
  filename: unified_benchmark.csv
```

The `label_noise` and `feature_shift` blocks can be independently disabled.
Existing configurations using `experiments.noise_levels` or the previous
top-level `feature_shifts` list remain supported. Use either `dataset.name` for
one dataset or `dataset.names` for multiple supported datasets.

## Running a benchmark

```powershell
python -m trusttab.experiments.run --config configs/unified_benchmark.yaml
```

Or call the public API directly:

```python
from trusttab.experiments import run_robustness_benchmark

results = run_robustness_benchmark(
    dataset_names=["german_credit"],
    label_noise_levels=[0.0, 0.2],
    feature_shifts={
        "enabled": True,
        "types": ["mean_shift"],
        "magnitudes": [0.5, 1.0],
        "features": ["age"],
    },
    seeds=[1, 2, 3],
    model_names=["logistic_regression", "random_forest"],
    output_path="results/benchmark.csv",
)
```

## Result schema and interpretation

Every row uses the same columns:

| Column | Meaning |
|---|---|
| `dataset`, `model`, `seed` | Dataset, estimator, and repeat/split seed. |
| `experiment_type` | `clean_baseline`, `label_noise`, or `feature_shift`. |
| `corruption_type`, `corruption_level` | Perturbation name and its configured level or magnitude. |
| `accuracy`, `balanced_accuracy`, `macro_f1`, `roc_auc` | Classification metrics. |
| `brier_score`, `ece`, `adaptive_ece`, `mce`, `log_loss` | Probability quality and calibration metrics. |
| `n_train`, `n_test` | Split sizes. |
| `baseline_id` | Links scenarios sharing the same dataset/model/seed baseline. |
| `affected_features`, `n_corrupted`, `shift_seed` | Scenario-specific metadata; non-applicable values are empty/zero/NaN. |

Compare each robustness row to its `clean_baseline` row with the same
`baseline_id`. A label-noise level of `0.0` is also emitted when configured;
it reuses the baseline model and metrics rather than fitting a duplicate model.
Feature scenarios each start from the original clean test split, so one shift
cannot contaminate another. `shift_seed` is derived deterministically from the
experiment seed and scenario index. AUC or another metric unavailable for an
experiment may be represented as NaN by a supplied evaluator. If a test split
contains one target class, ROC-AUC is undefined and the unified runner records
it as NaN while retaining the other metrics.
