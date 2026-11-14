# Unified Robustness Benchmarking

TrustTab's unified runner evaluates clean performance alongside training-label
noise and test-time feature shifts in a single result table. For each dataset
and seed, it creates one stratified split. For each model, it fits one clean
baseline model. That fitted model supplies the clean baseline and all
feature-shift evaluations; positive label-noise levels fit additional models
using corrupted copies of the training labels. Test labels are never changed.

## Supported scenarios

- **Clean baseline:** original training labels and clean test features.
- **Label noise:** each training label is independently flipped with the
  configured Bernoulli probability. Only training labels are passed to the
  noise injector; test labels remain clean. The realized number of flips is
  random, with expectation `noise_rate × n_train`, and is not forced to equal a
  rounded percentage of the training set.
- **Feature shift:** configured feature transformations on independent copies
  of the clean test features; training data remains unchanged.

Feature shifts reuse the existing `apply_feature_shifts` implementation and
accept Gaussian noise, mean shift, variance scaling, categorical probability
redistribution, and category replacement. A feature-shift configuration may
select types and magnitudes as a cross product. If no feature list is supplied,
numeric transforms use numeric non-boolean columns and categorical transforms
use non-numeric or boolean columns. Explicit feature selection is recommended
for controlled experiments.

## Platt calibration rescue

Calibration is opt-in through `experiments.calibration.enabled`. It is disabled
by default so existing train/test experiments retain their current 80/20 split
and result rows. When enabled, the calibrated experiment uses the blueprint's
stratified 70/15/15 train/calibration/test split (configurable as three
fractions summing to one); the legacy `experiments.test_size` applies only when
calibration is disabled.

For each model and noise condition, the model is fit on the training partition
and its noisy training labels. A sigmoid `CalibratedClassifierCV` is then fit
with a `FrozenEstimator` around that already-fitted model, using only clean
calibration features and labels. Scikit-learn uses `decision_function` when
available and otherwise uses `predict_proba`. The frozen estimator does not
refit the preprocessing pipeline or model. The same clean test observations
and labels are evaluated for raw and calibrated metrics after calibration has
been fit.

The result remains one row per condition. Existing metric columns contain raw
metrics; paired calibrated metrics use a `calibrated_` prefix (for example
`ece` and `calibrated_ece`). `condition_id` identifies the condition. The
`calibration_status` field is `disabled`, `succeeded`, `failed`, or
`not_applicable` for experiment types that are not evaluated with calibrated
predictions (currently feature shifts). Failed calibrations retain raw
metrics, put NaN in calibrated metric columns, and include the error in
`calibration_error`. See
[`calibration-rescue.md`](calibration-rescue.md) for the full protocol and
output fields.

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
  calibration:
    enabled: false
    method: sigmoid
    train_size: 0.70
    calibration_size: 0.15
    test_size: 0.15
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

For the configured core matrix, use the resumable scheduler and a new,
dedicated checkpoint directory. The scheduler atomically replaces
`raw_results.csv` and `run_manifest.json` after every emitted condition, writes
the normalized configuration, dependency versions, dataset fingerprints, and
condition statuses, and uses a process lock to prevent concurrent writers:

```powershell
python -m trusttab.experiments.run `
  --config configs/experiments.yaml `
  --resume-dir experiments/runs/trusttab-core-2026-11-14
```

If execution stops, rerun the same command with the same configuration and
checkpoint directory. Completed condition IDs are skipped; failed or
unfinished conditions are retried. The scheduler refuses to resume if the
normalized configuration, dataset fingerprint, Python version, or installed
package versions changed. A run is marked `completed` only after all expected
condition IDs appear exactly once. On an incomplete run, inspect
`run_manifest.json` for failed conditions before retrying. The directory must
be empty for a new run, or contain the scheduler's own manifest and checkpoint
CSV for a resume. Do not point it at an existing unrelated results directory.

The chunk boundary is one dataset/model/seed group. If the benchmark raises,
the first unresolved condition in execution order is marked failed with the
exception; later conditions in that chunk are explicitly left pending with a
`last_error` note because they were not attempted. Other groups continue. Rows
already atomically checkpointed in that group are retained and not repeated.
Per-row `train_time_seconds`, scheduler `run_id`, and completion timestamp
support runtime and provenance review. The full matrix must still be
explicitly approved before starting it.

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
| `affected_features`, `n_corrupted`, `shift_seed` | Scenario-specific metadata; `n_corrupted` is the realized Bernoulli flip count and is random. Non-applicable values are empty/zero/NaN. |
| `condition_id`, `model_params` | Stable condition key and serialized model parameters. |
| `calibration_method`, `calibration_status`, `calibration_error` | Calibration method and outcome; errors are reported without fabricated calibrated metrics. |
| `calibration_split_seed`, `n_calibration` | Derived second-stage split seed and calibration partition size. |
| `calibrated_*` | Metric values for calibrated probabilities on the same clean test set; NaN when calibration is disabled or fails. |

Compare each robustness row to its `clean_baseline` row with the same
`baseline_id`. A label-noise level of `0.0` is also emitted when configured;
it reuses the baseline model and metrics rather than fitting a duplicate model.
Feature scenarios each start from the original clean test split, so one shift
cannot contaminate another. `shift_seed` is derived deterministically from the
experiment seed and scenario index. AUC or another metric unavailable for an
experiment may be represented as NaN by a supplied evaluator. If a test split
contains one target class, ROC-AUC is undefined and the unified runner records
it as NaN while retaining the other metrics.

For the five-seed research design, use the Phase 7 analysis in
`trusttab.statistics.run_research_analysis` or generate a report from saved
raw results. Statistical blocks are matched seeds within one fixed dataset
and noise level. Do not combine the two datasets as independent seed
repetitions. The analysis keeps incomplete blocks, failed calibration, and
undefined metrics explicit; it never fills them with zero. It uses `label_noise`
rows for the five planned levels because the runner separately stores a clean
baseline at zero noise. Holm families, slope normalization, effect sizes, and
the limits of five seeds are documented in [statistics.md](statistics.md).
