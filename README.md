# TrustTab

TrustTab is a Python framework for measuring predictive performance and
probability reliability of tabular classifiers under controlled training-label
noise and test-time feature shifts. It provides reproducible experiments and
research-oriented summaries; it does not claim that any model is universally
robust.

## Capabilities

- Logistic Regression and Random Forest, with optional XGBoost and TabPFN
- German Credit and Adult Census Income dataset loaders
- Clean baselines, symmetric training-label noise, and numeric/categorical
  feature shifts
- Accuracy, balanced accuracy, macro F1, ROC-AUC, Brier score, ECE, adaptive
  ECE, maximum calibration error, and log loss
- Sigmoid and isotonic probability calibration
- Paired statistical comparisons with multiple-comparison corrections
- CSV benchmarks, Markdown reports, headless figures, and a local run registry

The package is organized under `src/trusttab/`: `data` and `models` prepare
inputs and estimators; `noise` creates perturbations; `experiments` orchestrates
runs and tracking; `evaluation`, `statistics`, `visualization`, and `reporting`
analyze results.

## Install

Python 3.11 or 3.12 is supported.

```bash
python -m pip install -e ".[dev]"
```

Optional models can be installed separately:

```bash
python -m pip install -e ".[xgboost]"
python -m pip install -e ".[tabpfn]"
```

## Quick start

Run the small two-model, two-seed end-to-end benchmark (including label noise,
feature shift, reporting, and local tracking):

```bash
trusttab --config configs/final_smoke_test.yaml
```

The Python API also supports a unified benchmark:

```python
from trusttab.experiments import run_robustness_benchmark

results = run_robustness_benchmark(
    dataset_name="german_credit",
    label_noise_levels=[0.0, 0.2],
    feature_shifts=[
        {"type": "mean_shift", "features": ["age"], "magnitude": 0.5}
    ],
    model_names=["logistic_regression", "random_forest"],
    seeds=[1, 2, 3],
    output_path="results/benchmark.csv",
)
```

Dataset loading may require network access on the first use if the source data
is not already available. Unit tests use local fixtures and do not download
datasets.

## Configuration and CLI

YAML configurations are validated before execution. See
[`configs/a6_smoke_test.yaml`](configs/a6_smoke_test.yaml) for a small runnable
configuration and [`configs/experiments.yaml`](configs/experiments.yaml) for a
larger setup. The smoke-test config uses the German Credit loader, which may
need access to the dataset source when the data is not cached.

```bash
trusttab --help
trusttab --config configs/experiments.yaml
trusttab --list-runs --runs-dir experiments/runs
trusttab --inspect-run RUN_ID --runs-dir experiments/runs
trusttab --compare-runs RUN_ID_A RUN_ID_B --runs-dir experiments/runs
trusttab --verify-run RUN_ID --runs-dir experiments/runs
```

Enable local tracking with `tracking: {enabled: true}` in YAML. Run commands
write benchmark output to the configured output location; tracked artifacts are
stored under `experiments/runs` unless configured otherwise.

## Analysis and reports

The reporting API accepts unified benchmark results:

```python
from trusttab.reporting import generate_robustness_report

artifacts = generate_robustness_report(results, "results/report")
```

It produces a Markdown report, CSV summaries, and headless PNG figures. See
the [benchmark](docs/unified-benchmarking.md),
[metrics](docs/metrics.md), [statistics](docs/statistics.md),
[visualization](docs/visualization.md), [reporting](docs/robustness-reporting.md),
and [tracking](docs/experiment-tracking.md) guides.

## Outputs and reproducibility

Benchmark CSV rows identify the dataset, model, scenario, perturbation,
corruption level, seed, metrics, sample counts, and baseline link. Tracked runs
also retain original and normalized configuration, configuration hash,
environment and Git metadata, dataset fingerprints, results, and report
artifacts. Re-execution checks numerical agreement using configurable absolute
and relative tolerances. Numerical reproducibility does not imply bitwise
identity across hardware, dependency versions, or nondeterministic estimators.

## Research assumptions and limitations

- Label noise is symmetric: each training label has the configured chance of
  being flipped. Real annotation errors may be class-dependent or structured.
- Feature shifts are controlled synthetic interventions and do not represent
  every real distribution shift.
- ECE and related binned calibration metrics depend on binning choices and can
  be noisy on small samples. ROC-AUC is undefined when an evaluation split has
  one class.
- Paired tests require aligned observations/seeds and rely on their own
  assumptions. Small seed counts have limited power; multiplicity correction
  does not remove this limitation.
- German Credit and Adult are limited benchmark datasets and should not be
  treated as representative of all tabular applications.
- Optional model implementations may have model-specific constraints and
  nondeterminism. Reproducibility metadata helps interpret reruns but cannot
  guarantee identical results everywhere.
- Robustness under a specified perturbation is distinct from clean predictive
  accuracy and does not establish universal reliability.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests
ruff format --check src tests
python -m build
```

CI runs these checks on Python 3.11 and 3.12 without downloading external
datasets or requiring optional model packages.
