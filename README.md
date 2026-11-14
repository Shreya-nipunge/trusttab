# TrustTab

TrustTab is a Python framework for measuring predictive performance and
probability reliability of tabular classifiers under controlled training-label
noise and test-time feature shifts. It provides reproducible experiments and
research-oriented summaries; it does not claim that any model is universally
robust.

## Capabilities

- Logistic Regression and Random Forest, with optional XGBoost and TabPFN
- German Credit and processed Cleveland Heart Disease loaders; Adult Census
  Income remains available as an additional dataset
- Clean baselines, symmetric training-label noise, and numeric/categorical
  feature shifts
- Opt-in Platt calibration experiments using separate clean calibration data
- Accuracy, balanced accuracy, macro F1, ROC-AUC, Brier score, ECE, adaptive
  ECE, maximum calibration error, and log loss
- Platt scaling in the integrated calibration experiment; isolated utilities
  also expose isotonic calibration
- Paired statistical comparisons with multiple-comparison corrections
- CSV benchmarks, Markdown reports, headless figures, and a local run registry

The package is organized under `src/trusttab/`: `data` and `models` prepare
inputs and estimators; `noise` creates perturbations; `experiments` orchestrates
runs and tracking; `evaluation`, `statistics`, `visualization`, and `reporting`
analyze results.

## Install

Python 3.11 or 3.12 is supported.

After the approved PyPI release, install the public distribution with:

```bash
python -m pip install trusttab
```

The package has not been published to PyPI yet. To install the current source
checkout, use:

```bash
python -m pip install .
```

For development, tests, linting, and builds:

```bash
python -m pip install -e ".[dev]"
```

Optional models can be installed separately:

```bash
python -m pip install -e ".[xgboost]"
python -m pip install -e ".[tabpfn]"
```

The `tabpfn` extra pins TabPFN 10.0.0; it is not installed with core TrustTab
dependencies. See the [TabPFN compatibility and hardware notes](docs/tabpfn.md)
before enabling it. TabPFN-3.5 weights are separately licensed and gated;
users must accept the model terms and authorize their own local download.
The default TabPFN-3.5 weights are under Prior Labs' non-commercial,
non-production license; commercial or production use requires rights from
Prior Labs. TrustTab does not bundle or redistribute model weights. See the
[TabPFN license and access notes](docs/tabpfn.md). The current license also
restricts use of model outputs to non-commercial purposes and includes user
eligibility conditions; review the [full Prior Labs license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)
before accessing the checkpoint.

## Platt calibration rescue

Calibration is an opt-in secondary experiment. See the [calibration-rescue
protocol](docs/calibration-rescue.md) and run its bounded one-model smoke
configuration with:

```bash
python -m trusttab.experiments.run --config configs/calibration_rescue_smoke.yaml
```

The calibration experiment uses separate stratified train, calibration, and
test subsets. Only training labels are noise-corrupted; calibration labels
remain clean, and test labels are used only for final evaluation. Raw metrics
remain in the standard columns and corresponding calibrated metrics use a
`calibrated_` prefix. Calibration is disabled in the core configuration by
default, consistent with the blueprint's secondary-experiment design.

## Quick start

From a repository checkout, run the small two-model, two-seed end-to-end
benchmark (including label noise, feature shift, reporting, and local tracking):

```bash
trusttab --config configs/final_smoke_test.yaml
```

The YAML templates are repository files and are not bundled in the wheel. A
pip-only installation can use the Python API below or provide its own YAML
configuration; the CLI does not silently select the 200-condition research
configuration.

The Python API also supports a unified benchmark:

```python
from trusttab.api import run_robustness_benchmark

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

See the [public API guide](docs/api.md) for supported imports and
[`examples/quickstart.py`](examples/quickstart.py) for a runnable synthetic
example that does not download a dataset.

## Datasets and attribution

The research benchmark pair is German Credit and the processed Cleveland
subset of UCI Heart Disease (UCI dataset ID 45). TrustTab reads only
`processed.cleveland.data`; it does not combine the Cleveland, Hungarian,
Switzerland, and VA databases. The original `num` target is preserved in
`y.attrs["original_num"]` and mapped to binary labels as `0 → 0` and
`1, 2, 3, 4 → 1`. The 303 rows and 13 source features are retained. Source
missing values in `ca` and `thal` remain missing in the returned features;
the experiment pipeline imputes them after splitting, using training data
only. An optional `data_home` argument caches the processed Cleveland rows,
including the original `num` target, as a local CSV.

Heart Disease citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano,
R. (1989). *Heart Disease*. UCI Machine Learning Repository,
[dataset ID 45](https://archive.ics.uci.edu/dataset/45/heart),
[DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X). The dataset is
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: Heart Disease dataset by Janosi, Steinbrunn, Pfisterer, and
Detrano, hosted by the UCI Machine Learning Repository, used under CC BY 4.0.
See [dataset loading details](docs/datasets.md) for source and missing-value
behavior.

## Configuration and CLI

YAML configurations are validated before execution. See
[`configs/a6_smoke_test.yaml`](configs/a6_smoke_test.yaml) for a small runnable
configuration and [`configs/experiments.yaml`](configs/experiments.yaml) for
the configured 200-condition research matrix. Use
[`configs/phase3_smoke_test.yaml`](configs/phase3_smoke_test.yaml) for a small
two-model, two-noise-level check. Dataset downloads may require network access
when the data is not cached.

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

The report also provides seed-blocked Friedman/Nemenyi model comparisons,
paired raw-versus-calibrated ECE/Brier tests, normalized degradation slopes,
effect sizes, and small-sample confidence intervals. These are exploratory
with five seeds, and the two datasets are analyzed separately rather than
treated as a sample of datasets. See [docs/statistics.md](docs/statistics.md)
for the analysis protocol and its limits. Running the report does not run
experiments or establish empirical conclusions by itself.

## Outputs and reproducibility

Benchmark CSV rows identify the dataset, model, scenario, perturbation,
corruption level, seed, metrics, sample counts, and baseline link. Tracked runs
also retain original and normalized configuration, configuration hash,
environment and Git metadata, dataset fingerprints, results, and report
artifacts. Re-execution checks numerical agreement using configurable absolute
and relative tolerances. Numerical reproducibility does not imply bitwise
identity across hardware, dependency versions, or nondeterministic estimators.

## Completed core research run

The completed core experiment evaluates whether predictive performance and
probability reliability change differently as training labels become noisier.
It covers Logistic Regression, Random Forest, XGBoost, and TabPFN on German
Credit (OpenML ID 31) and the processed Cleveland subset of UCI Heart Disease
(UCI ID 45), at noise rates 0%, 10%, 20%, 30%, and 40%, with seeds 0 through
4. This is 200 noise conditions plus 40 clean-baseline reference rows, for 240
result rows. Adult remains an optional loader and is not part of this core
matrix. Cleveland uses only the processed Cleveland file, not a merge of UCI's
four heart-disease databases.

Training labels use independent per-example Bernoulli flips; the realized
corruption count varies around its expectation. Calibration and test labels
remain clean. The implementation follows the blueprint's mathematical
definition of independent flips; an earlier fixed-count pseudocode fragment
was inconsistent and was corrected. The original research pair of German
Credit and Heart Disease is retained, with the approved processed Cleveland
subset as the Heart Disease source.

The run records Accuracy, Balanced Accuracy, Macro-F1, ROC-AUC, ECE, and Brier
Score. The primary model inference is dataset-specific, seed-blocked Friedman
testing with Nemenyi post-hoc comparisons gated on the omnibus result and
Holm correction within the planned families. Degradation-slope comparisons
use paired seed-level results. Descriptive intervals bootstrap the five seed
units. The generated compatibility table also includes supplementary paired
t-tests; these are exploratory, not the primary planned analysis. Calibration
was disabled in the core matrix, so no core raw-versus-calibrated inference is
available and **RQ4 (the Platt calibration question) remains unanswered**.

Interpretation is limited by five seeds and two fixed datasets; these do not
support broad claims about all datasets. Cleveland's 61-row test partition
makes its metric estimates especially coarse. The saved aggregate results do
not retain per-observation predictions, so final metrics cannot be
independently recomputed from the CSV alone. The preserved run, manifest,
report tables, figures, and RQ0-RQ4 findings are in the local working copy at
`experiments/runs/trusttab-core-20261114/`; they are research outputs, not
package data, and are not included in the wheel.

## Research assumptions and limitations

- Label noise uses independent Bernoulli flips on training labels: each label
  is flipped with the configured probability. The realized flip count is random
  (expected to be `noise_rate × n_train`), not fixed to that value. Test labels
  remain clean. Real annotation errors may be class-dependent or structured.
- Feature shifts are controlled synthetic interventions and do not represent
  every real distribution shift.
- ECE and related binned calibration metrics depend on binning choices and can
  be noisy on small samples. ROC-AUC is undefined when an evaluation split has
  one class.
- Paired tests require aligned observations/seeds and rely on their own
  assumptions. Small seed counts have limited power; multiplicity correction
  does not remove this limitation.
- German Credit, processed Cleveland Heart Disease, and Adult are limited
  benchmark datasets and should not be treated as representative of all
  tabular applications.
- Optional model implementations may have model-specific constraints and
  nondeterminism. Reproducibility metadata helps interpret reruns but cannot
  guarantee identical results everywhere.
- Robustness under a specified perturbation is distinct from clean predictive
  accuracy and does not establish universal reliability.

## Citation

For the software, cite the TrustTab repository and released version used in
your work: Shreya Nipunge et al., *TrustTab: Reliability-Aware Stress Testing
for Tabular Machine Learning*, GitHub repository
<https://github.com/Shreya-nipunge/trusttab>, version `0.1.0` (update the
version to the release actually used). For the Cleveland benchmark, cite the
UCI Heart Disease dataset separately using the DOI and CC BY 4.0 attribution
above. Do not cite smoke-test outputs as research findings.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests
ruff format --check src tests
python -m build
```

Run these checks locally on Python 3.11 or 3.12. The test suite uses local
fixtures and does not download datasets or require optional model packages.

Ruff format currently excludes 29 legacy source and test files from its format
check; the remaining files are checked. These exclusions are an explicit
cleanup backlog, not a claim that those files match formatter output.
