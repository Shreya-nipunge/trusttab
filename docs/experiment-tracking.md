# Experiment Tracking and Reproducibility

TrustTab can record unified benchmark executions locally without an external
tracking service. Tracking is optional and works offline. Untracked calls to
`run_robustness_benchmark` and existing CLI configurations keep their current
behavior.

## Enable tracking

Add a `tracking` section to an experiment YAML file:

```yaml
project:
  name: TrustTab
dataset:
  name: german_credit
experiments:
  test_size: 0.2
  seeds: [0, 1, 2]
  label_noise:
    enabled: true
    levels: [0.0, 0.1, 0.2]
models: [logistic_regression, random_forest]
output:
  directory: results
  filename: stress.csv
tracking:
  enabled: true
  directory: experiments/runs
  experiment_name: credit-noise-study
```

Run it using the existing entry point:

```powershell
python -m trusttab.experiments.run --config configs/tracked.yaml
```

Tracking-enabled CLI runs use the unified benchmark runner, including when the
configuration uses one of the supported legacy label-noise-only or
feature-shift-only formats. Direct API callers can opt in with
`tracking_dir`:

```python
from trusttab.experiments import run_robustness_benchmark

results = run_robustness_benchmark(
    dataset_names=["german_credit"],
    label_noise_levels=[0.0, 0.2],
    seeds=[1, 2, 3],
    tracking_dir="experiments/runs",
    experiment_name="credit-noise-study",
)
run_id = results.attrs["run_id"]
```

When direct callers do not supply `original_config` and `normalized_config`,
the runner records the arguments as its configuration representation. YAML
CLI runs preserve the parsed source configuration and the validated,
normalized configuration separately. YAML comments and whitespace are not
represented in JSON; configuration values are retained.

## Run directory and metadata

Each run receives a timestamp-plus-random run ID and a new directory. Existing
runs are never overwritten.

```text
experiments/runs/<run_id>/
├── manifest.json
├── config.json
├── environment.json
├── dataset.json
├── results.csv
├── report.md
├── summary.csv
├── model_comparison.csv
├── robustness_degradation.csv
├── statistical_comparisons.csv
└── figures/
```

The manifest starts with `status: running` and is atomically updated to
`completed` or `failed`; it includes UTC start/completion timestamps, run ID,
experiment name, TrustTab version, Git commit and working-tree cleanliness,
dataset names, selected models, seeds, configuration hash, result row count,
and failure type/message when applicable. A failure after result creation
retains files already written and records the failure state.

`config.json` includes the original and normalized configurations and a
SHA-256 digest over canonical JSON. NumPy values and paths are converted to
JSON-safe values, and non-finite floats are represented as `null`. The
environment record contains Python, operating system/platform, TrustTab,
NumPy, pandas, scikit-learn, SciPy, Matplotlib, optional XGBoost versions, and
Git provenance. Missing optional packages are recorded as `unavailable`.
TrustTab does not collect environment variables or credentials.

`dataset.json` contains sample/feature counts, target description, class
distribution, feature dtypes, and a SHA-256 fingerprint. Fingerprints include
ordered values, column names/order, and dtypes; the DataFrame index and source
file path are excluded. Raw observations are not copied into metadata. The
fingerprint is deterministic for equivalent ordered data under the recorded
pandas implementation; changes in data ordering, schema, or dependency hashing
behavior can change it.

## Run registry and CLI

List runs, with optional filters:

```powershell
python -m trusttab.experiments.run --list-runs --runs-dir experiments/runs
python -m trusttab.experiments.run --list-runs --dataset adult --status completed
```

Inspect one run, compare two records, or re-execute a saved configuration and
verify it against the original:

```powershell
python -m trusttab.experiments.run --inspect-run <run_id> --runs-dir experiments/runs
python -m trusttab.experiments.run --compare-runs <run_a> <run_b> --runs-dir experiments/runs
python -m trusttab.experiments.run --verify-run <run_id> --runs-dir experiments/runs
```

`--verify-run` uses the saved normalized configuration by default. Pass
`--config path/to/new.yaml` to verify against another explicitly supplied
configuration. The newly executed verification itself is recorded as a run,
and its directory receives `verification.json`.

Equivalent Python utilities are available from `trusttab.experiments`:

```python
from trusttab.experiments import (
    compare_runs,
    get_run,
    list_runs,
    verify_reproducibility,
)

runs = list_runs("experiments/runs", dataset="german_credit")
metadata = get_run("experiments/runs", runs[0]["run_id"])
comparison = compare_runs("experiments/runs", runs[0]["run_id"], runs[1]["run_id"])
verification = verify_reproducibility(
    "experiments/runs", runs[0]["run_id"], runs[1]["run_id"], rtol=1e-6, atol=1e-8
)
```

## Interpreting verification

Verification reports separate matches for configuration hash, dataset
fingerprints, Git commit/cleanliness, dependency versions, seeds, and saved
metric results. Result rows are aligned by dataset, model, experiment type,
corruption type/level, and seed. Numeric metrics use configurable absolute
and relative tolerances; NaNs match NaNs. A `numerically_equivalent` status
means the checked values match under those tolerances. It does not imply
bitwise reproducibility across hardware, operating systems, dependency builds,
or nondeterministic estimators.

## Limitations

- Tracking currently wraps the unified robustness benchmark; legacy runners
  continue to work unchanged when tracking is disabled.
- The local JSON registry is intended for single-user/local workflows and
  does not provide database transactions or shared multi-host coordination.
- Dataset fingerprints depend on pandas' deterministic object hashing and
  therefore should be interpreted alongside the recorded dependency versions.
- Git provenance is unavailable when Git metadata cannot be queried.
