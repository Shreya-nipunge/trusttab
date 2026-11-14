# TrustTab Implementation Progress

## Phase 1 â€” Build metadata and verification

**Status:** Completed for current Phase 1 objectives. The existing project `.venv` ran the full test suite with an explicit pytest temporary directory; wheel and sdist builds succeeded from a clean source copy. A newly created environment could not import pandas because Windows Application Control blocked a native DLL, so clean-environment test execution remains an environment limitation.

### Blueprint and scope

`Project_Execution_Blueprint.md` was not found in the repository or workspace. As instructed, the implementation audit and the phased plan are the current requirements baseline; no additional blueprint requirements were inferred.

The tracked/source deliverables from this phase are `pyproject.toml` and this progress file. Build tooling also refreshed the pre-existing ignored generated metadata files under `src/trusttab.egg-info/` (`PKG-INFO`, `SOURCES.txt`, `dependency_links.txt`, `entry_points.txt`, `requires.txt`, and `top_level.txt`) and created ignored `build/lib` and `build/bdist.win-amd64` directories during an initial in-place build attempt. The two newly created build directories and the repository-local verification venv were removed afterward. The existing `build/clean-install` and `dist/` artifacts were preserved. No research code was changed, the 200-condition matrix was not run, and nothing was published.

### Confirmed starting state

- Branch: `chore/remove-github-actions`.
- At phase start, the only existing uncommitted item was the untracked `TRUSTTAB_IMPLEMENTATION_AUDIT.md`. It was preserved.
- Original packaging issue: `license = "MIT"` was rejected by the installed setuptools 65.5.0 build validator.
- Python 3.11.9 is available and within the projectâ€™s declared Python 3.11â€“3.12 range.
- Main `.venv` imports pandas 3.1.0 and scikit-learn 1.9.1 successfully. Full pytest initially hit a denied default temp-directory scan; specifying `--basetemp` resolved that issue.
- A fresh environment created in the operating system temp directory and another at the repositoryâ€™s ignored `venv/` path both failed to import pandas native extensions with `Application Control policy has blocked this file`. This is an environment restriction, not evidence of a TrustTab code error.

### Changed files

- `pyproject.toml`: changed license metadata to PEP 639 SPDX form, `license = "MIT"`, and declared `license-files = ["LICENSE"]`. The project already requires `setuptools>=77`, which supports this representation. Build-system requirements were otherwise unchanged.
- `IMPLEMENTATION_PROGRESS.md`: added this record.
- Ignored generated metadata: setuptools refreshed the six files listed in the scope note above. They are build outputs, not tracked source files.

### Verification commands and outcomes

Commands below were run from `D:\trusttab`. Temporary paths were unique per run; existing `dist/` artifacts were not overwritten.

| Check | Command | Outcome |
|---|---|---|
| Main environment | `.\.venv\Scripts\python.exe --version` | Python 3.11.9 |
| Main packaging versions | `.\.venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('setuptools'), m.version('build'), m.version('pandas'), m.version('scikit-learn'))"` | setuptools 65.5.0; build 1.6.1; pandas 3.1.0; scikit-learn 1.9.1 |
| Main imports | `.\.venv\Scripts\python.exe -c "import pandas, sklearn; print(pandas.__version__, sklearn.__version__)"` | Passed: 3.1.0 and 1.9.1 |
| Full tests | `$tmp = Join-Path $env:TEMP ('trusttab-pytest-phase1-' + [guid]::NewGuid().ToString('N')); .\.venv\Scripts\python.exe -m pytest -q --basetemp $tmp` | **268 passed, 1 warning** in 16.58s. Warning is sklearnâ€™s single-label confusion-matrix warning in `test_roc_auc_is_nan_when_test_split_contains_one_class`. |
| Lint | `.\.venv\Scripts\ruff.exe check src tests` | Passed: `All checks passed!` |
| Formatting | `.\.venv\Scripts\ruff.exe format --check src tests` | Passed for configured scope: `19 files already formatted`. `pyproject.toml` excludes 29 legacy source/test files, so this is not a full-tree format check. |
| Clean source-copy build | `venv\Scripts\python.exe -m build --no-isolation --wheel --sdist --outdir <unique-temp-dist> <unique-temp-source-copy>` | Passed with setuptools 84.2.1, wheel 0.48.0, build 1.6.1. Produced `trusttab-0.1.0-py3-none-any.whl` (54,486 bytes) and `trusttab-0.1.0.tar.gz` (65,912 bytes). |
| Wheel inspection | Python `zipfile` inspection of the newly built wheel | Passed: 29 Python modules; `METADATA` declares `License-Expression: MIT`; console `entry_points.txt` and bundled `licenses/LICENSE` present; experiment CLI module present. |
| Sdist inspection | Python `tarfile` inspection of the newly built sdist | Passed: 29 Python modules; `LICENSE`, `README.md`, `pyproject.toml`, and experiment CLI module present. |

An initial no-isolation build using the original `.venv` did not proceed because that environment has setuptools 65.5.0 and no `wheel`, below the declared build requirement. The isolated build in the default sandbox could not reach PyPI. After creating a separate environment with the declared project development dependencies and installing setuptools 84.2.1 plus wheel 0.48.0, the build succeeded from a temporary copy of the current source tree. This avoided overwriting the pre-existing `dist/` artifacts.

### Archive inspection details

- Wheel: `trusttab-0.1.0-py3-none-any.whl`, 29 importable `.py` package modules; correct CLI entry-point metadata and license file included.
- Source distribution: `trusttab-0.1.0.tar.gz`, 29 package modules; license, README, and build metadata included.
- The successful build establishes that the source can produce the two artifacts with the declared setuptools generation. It does **not** establish clean wheel installation, PyPI availability, optional-model operation, or package-index installation.

### Unresolved limitations

1. Clean-environment pandas/scikit-learn imports and tests remain blocked by Windows Application Control at fresh venv paths. The main `.venv` test run passed, but a clean isolated test run is still **NOT VERIFIED**.
2. The original `.venv` does not satisfy the projectâ€™s setuptools build requirement; the successful artifact build used a separately provisioned build environment.
3. Wheel installation from the built artifact, public API smoke tests, and installed console-script execution were not part of Phase 1 verification and remain for the package-release phase.
4. Phase 1 does not address the missing Heart Disease loader, 200-condition configuration, calibration-rescue experiment, or statistical-method gaps documented in `TRUSTTAB_IMPLEMENTATION_AUDIT.md`.

**Phase 1 conclusion:** The invalid license metadata is corrected, the full suite passes in the existing project environment when pytest uses an accessible temporary directory, and both distribution archives build and contain the expected package files. Clean-environment test execution remains an explicit limitation. No later implementation phase has started.

## Phase 2 â€” Processed Cleveland Heart Disease dataset

**Status:** Implemented and verified with offline fixtures. The live UCI archive request was not made during tests.

### Source and data contract

- Approved source: UCI Machine Learning Repository Heart Disease, dataset ID 45, processed Cleveland subset only: <https://archive.ics.uci.edu/dataset/45/heart>.
- DOI: <https://doi.org/10.24432/C52P4X>. License: CC BY 4.0. Attribution and citation are in the loader docstring, README, and `docs/datasets.md`.
- The loader downloads UCI's dataset archive using the Python standard library and extracts only `processed.cleveland.data`. It does not use or merge the other three database files and introduces no new runtime dependency.
- Expected shape is 303 rows Ã— 13 ordered source features. The original `num` distribution is validated as 0:164, 1:55, 2:36, 3:35, 4:13. Binary mapping is 0â†’0 and 1â€“4â†’1 (164 no-disease, 139 disease-present).
- The original numeric `num` is retained in `y.attrs["original_num"]`; when `data_home` is supplied it is also saved as the cache CSV's `num` column. `?` source markers become `NaN`. Rows are not dropped; missing values are allowed only in `ca` and `thal` and remain for the existing training-only imputation pipeline.
- Adult remains selectable as an additional dataset. Existing YAML experiment matrices were not changed; selecting Heart Disease is supported through the existing `dataset.name` or `dataset.names` config schema.

### Files changed in Phase 2

- `src/trusttab/data/heart_disease.py` â€” source download/archive selection, validation, target mapping, feature normalization, optional data-home cache.
- `src/trusttab/data/__init__.py` â€” public loader and validator exports.
- `src/trusttab/experiments/runner.py` â€” dataset registry entry shared by the label-noise, unified benchmark, and feature-shift runners.
- `src/trusttab/experiments/config.py` â€” accepts `heart_disease` while retaining `german_credit` and `adult`.
- `tests/test_heart_disease.py` â€” offline archive, schema, distribution, mapping, missingness, preprocessing, and cache tests.
- `tests/test_config.py` â€” Heart Disease config acceptance test.
- `tests/test_experiments.py` â€” registry dispatch test.
- `README.md` and `docs/datasets.md` â€” core dataset pair, Adult status, data handling, citation, DOI, and attribution.
- `IMPLEMENTATION_PROGRESS.md` â€” this Phase 2 record.

### Verification

| Check | Command | Outcome |
|---|---|---|
| Focused tests | `.\.venv\Scripts\python.exe -m pytest -q tests/test_heart_disease.py tests/test_config.py tests/test_experiments.py --basetemp <unique-temp-dir>` | **40 passed** after fixing an integration issue with pandas `pd.NA` and the existing scikit-learn imputer. |
| Full suite | `.\.venv\Scripts\python.exe -m pytest -q --basetemp <unique-temp-dir>` | **279 passed, 1 warning** in 25.10s. Warning remains the existing sklearn single-label confusion-matrix warning in `test_roc_auc_is_nan_when_test_split_contains_one_class`. |
| Lint | `.\.venv\Scripts\ruff.exe check src tests` | Passed: `All checks passed!` |
| Format | `.\.venv\Scripts\ruff.exe format --check src tests` | Passed: 21 files already formatted; existing configured legacy exclusions remain. |
| Diff whitespace | `git diff --check` | Passed. |

The offline fixture archive includes both processed Cleveland and a fake Hungarian member; assertions verify that only Cleveland rows are returned. A cached second load is tested with network access forced to fail. The preprocessing integration test verifies missing values are imputable after loading without the loader dropping or filling rows.

### Limitations and phase boundary

1. The actual UCI ZIP was not fetched in this environment. The exact URL and archive-member parsing are exercised with a synthetic ZIP fixture; live-source availability and payload consistency remain **NOT VERIFIED**.
2. The class distribution is enforced against the selected Cleveland profile, but the downloaded archive has not been checked against that profile here.
3. The binary target's original `num` is preserved in pandas `Series.attrs` and in the optional cache, rather than as a third return value; this keeps the existing `(X, y)` loader convention.
4. No 200-condition run was started. No later calibration, noise-methodology, statistical, or visualization phase was started.

**Phase 2 conclusion:** The approved processed Cleveland dataset is now available through the existing loader registry and config validation. Offline tests cover selection, target mapping, expected class counts, shape/schema, missing values, cache behavior, and preprocessing compatibility. Phase 3 has not started.

## Phase 3 â€” Core experiment matrix and seed/configuration routing

**Status:** Core configuration and bounded fake-runner verification implemented. The 200 real model-condition matrix was not run. TabPFN's real fit is not verified.

### Configuration and integration

- `configs/experiments.yaml` now selects exactly German Credit and processed Cleveland Heart Disease; Logistic Regression, Random Forest, XGBoost, and TabPFN; noise levels `[0.0, 0.1, 0.2, 0.3, 0.4]`; and base seeds `[0, 1, 2, 3, 4]`. The Cartesian product is exactly 200 unique label-noise conditions.
- Per-model YAML parameters are validated by `load_experiment_config`, passed by the CLI to the unified runner, and forwarded to `build_model`. Managed `random_state` overrides in YAML are rejected; the experiment's derived model seed takes precedence at the factory boundary.
- `configs/phase3_smoke_test.yaml` supplies a bounded four-condition German Credit smoke configuration using only Logistic Regression and a 10-tree Random Forest. It omits XGBoost and TabPFN to avoid optional dependencies.
- `expand_experiment_conditions` enumerates the normalized dataset Ã— model Ã— noise Ã— seed conditions without constructing models.
- `derive_seed` deterministically derives independent uint32 streams for split, label-noise, model initialization, and feature-shift operations using SHA-256 plus NumPy SeedSequence. The unified and legacy runners now record split/noise/model seeds in returned rows. Within a condition family, noise is shared across models for paired comparisons; a model seed is shared across noise conditions for a model/dataset/base-seed combination.
- The unified runner still splits before label-noise injection and passes a corrupted copy only as `y_train`; `y_test` remains the clean split target. The fake-runner coverage checks exact condition coverage, model parameter forwarding, input immutability, and that evaluated test labels continue to match the clean fixture labels.
- `pyproject.toml` keeps TabPFN optional and pins its extra to `tabpfn==10.0.0`. `docs/tabpfn.md` documents installation, Python compatibility, input-size/hardware guidance, the planned sample sizes, and the verification limitation. README points users to the smoke config and optional extra.

### Files changed in Phase 3

- `configs/experiments.yaml`
- `configs/phase3_smoke_test.yaml` (new)
- `docs/tabpfn.md` (new)
- `pyproject.toml` (TabPFN optional dependency pin)
- `README.md`
- `src/trusttab/experiments/config.py`
- `src/trusttab/experiments/seeding.py` (new)
- `src/trusttab/experiments/run.py`
- `src/trusttab/experiments/runner.py`
- `src/trusttab/experiments/benchmark.py`
- `src/trusttab/experiments/feature_shift_runner.py`
- `src/trusttab/models/model_factory.py`
- `tests/test_config.py`
- `tests/test_core_matrix.py` (new)
- `tests/test_benchmark.py`
- `tests/test_models.py`

Files listed in the Phase 1/2 sections above were already modified before this phase; no unrelated existing edits were reverted.

### Verification commands and outcomes

Commands were run from `D:\trusttab` using the existing project `.venv`:

| Check | Command | Outcome |
|---|---|---|
| Targeted config/runner/model tests | `\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_core_matrix.py tests/test_experiments.py tests/test_benchmark.py tests/test_models.py -q --basetemp .pytest-tmp-phase3` | **67 passed, 1 warning** in 11.68s. |
| Full suite | `\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase3-full` | **288 passed, 1 warning** in 18.58s. The warning is sklearn's single-label confusion-matrix warning in the existing one-class ROC-AUC test. |
| Lint | `\.venv\Scripts\ruff.exe check src tests` | Passed: `All checks passed!` |
| Formatting | `\.venv\Scripts\ruff.exe format --check src tests` | Passed: 23 files already formatted (the project's configured legacy exclusions still apply). |
| Diff whitespace | `git diff --check` | Passed; only Git line-ending conversion notices were emitted for files with LF endings. |
| CLI | `\.venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed; help output displayed and exited without loading or running the default matrix. |
| Core condition enumeration and runner coverage | Included in full suite as `test_core_config_expands_to_exactly_200_unique_conditions` and `test_runner_visits_every_core_condition_with_fake_models` | Asserted exact product contents, 200 unique tuples, 200 label-noise rows, and parameter propagation through 200 fake-model constructions. No real matrix was launched. |

### TabPFN and unresolved limitations

- The pinned release's published metadata and official TabPFN documentation were checked. TabPFN 10.0.0 requires Python 3.10+; the documented CPU sample limit for the current model family is 5,000 training samples, with GPU inference recommended and memory guidance varying by workload. At the configured 80/20 split, expected training sizes are 800 German rows and 242 Cleveland rows. Shared preprocessing may expand the feature count.
- A real bounded TabPFN fit was **NOT VERIFIED**: neither `tabpfn` nor `torch` is installed in this environment. Runtime on target hardware is therefore unknown. Hardware discovery through Windows CIM/WMI was denied in this environment. Do not treat the configuration or adapter tests as evidence that TabPFN fits successfully; run an optional-dependency bounded fit before the research matrix.
- XGBoost also remains an optional extra. The full core matrix requires installing `trusttab[xgboost]` as well as `trusttab[tabpfn]`; neither is added to core dependencies.
- At the time of Phase 3, `Project_Execution_Blueprint.md` had not been found in the workspace. Phase 5 later located it and resolved the label-noise protocol; see the Phase 5 record below.
- No end-to-end real-data smoke benchmark was run; the loader may require network access if its cache is empty. The fake-model test exercises the unified experiment runner without network, optional model installs, or expensive fits.

**Phase 3 conclusion (status at that phase boundary):** The configuration enumerated the intended 200 unique conditions and the unified runner routed all configured models, datasets, parameters, and independent seed streams. At that point research execution remained gated on the noise-method decision and a real TabPFN bounded fit. Phase 5 resolved the noise method; TabPFN verification remains outstanding. The full matrix was not started, and no Phase 4 or later work was performed.

## Phase 5 â€” Independent Bernoulli label noise

**Status:** Implemented and verified. This phase changed only the label-noise method, its regression tests, and directly relevant documentation. The four-model/two-dataset matrix was not changed, and the 200-condition experiment was not run.

### Method decision and implementation

- The blueprint's Part F mathematical definition specifies independent Bernoulli flips with probability `p`, but its pseudocode incorrectly selected exactly `round(p*n)` indices. The user explicitly approved the Bernoulli method. The implementation and Part F pseudocode now use an independent per-example mask `rng.random(n) < noise_rate`.
- The realized flip count is random, with expectation `noise_rate Ã— n_samples`; it is not forced to a rounded count. At rate 0 no labels change; at rate 1 every eligible label changes. Binary labels flip to the other class. Existing multiclass behavior is retained: each selected label is replaced uniformly by one of the other observed classes.
- Input containers are copied, pandas Series index/name are retained, and multidimensional inputs are rejected. The runner continues to split first, passes only `y_train` to the noise function, and forwards its independently derived `noise_seed`. Its evaluator receives clean `y_test`.
- The existing result schema's `n_corrupted` field records the realized number of changes by comparing noisy and original training labels; no return-value/API change was needed.

### Files changed in Phase 5

- `src/trusttab/noise/label_noise.py`
- `tests/test_noise.py`
- `tests/test_experiments.py`
- `README.md`
- `docs/unified-benchmarking.md`
- `docs/robustness-reporting.md`
- `Project_Execution_Blueprint.md` (corrected Part F pseudocode to match its mathematical definition and approved method)
- `TRUSTTAB_IMPLEMENTATION_AUDIT.md` (updated the noise finding and marked its resolution)
- `IMPLEMENTATION_PROGRESS.md`

### Verification

| Check | Command | Outcome |
|---|---|---|
| Focused noise and runner tests | `.venv\Scripts\python.exe -m pytest tests\test_noise.py tests\test_experiments.py tests\test_benchmark.py tests\test_core_matrix.py -q --basetemp .pytest-tmp-phase5-focused` | **50 passed, 1 warning** in 8.57s. |
| Full suite | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase5-full` | **293 passed, 1 warning** in 50.32s. The warning is the existing sklearn single-label confusion-matrix warning in the one-class ROC-AUC test. |
| Lint | `.venv\Scripts\ruff.exe check src tests` | Passed: `All checks passed!` |
| Formatting | `.venv\Scripts\ruff.exe format --check src tests` | Passed: 23 files already formatted (configured legacy exclusions apply). |
| Diff whitespace | `git diff --check` | Passed; Git emitted only existing LF-to-CRLF conversion notices. |

### Remaining limitations

- The blueprint's formal Part F text and pseudocode now agree. The statistical validation test checks a broad Â±2 percentage-point tolerance over 20,000 balanced binary labels; it does not claim a fixed flip count or infer model-level effects.
- No calibration, statistics, visualization, release work, or full matrix execution was performed in this phase.

## Phase 4 â€” Calibration rescue

**Status:** Implemented for the unified benchmark and verified with synthetic fixtures. Calibration remains opt-in, consistent with the blueprint's secondary-experiment designation. No full experiment matrix was run.

### Initial inspection and protocol

- The blueprint specifies a disjoint 70/15/15 train/calibration/test split. This split was retained. It leaves 45 calibration and 46 test observations for the 303-row Cleveland dataset (scikit-learn stratified rounding), so calibration estimates may be noisy; this is a limitation, not a change to the research protocol.
- Existing runners split stratified train/test data before fitting and inject Bernoulli noise into training labels only. An existing helper already used scikit-learn `CalibratedClassifierCV` with `FrozenEstimator` for a previously fitted model. The Phase 4 gap was integration into the unified runner.
- `evaluate_classifier` is the common metric implementation. The unified runner has a narrow fallback for a one-class test partition, where ROC-AUC is undefined and recorded as NaN.

### Implementation

- Added validated opt-in calibration settings (`enabled`, `method: sigmoid`, and configurable three-way proportions); disabled mode retains the previous test-size-controlled train/test path.
- Added a reproducible stratified two-stage split helper. `split_seed` and `calibration_split_seed` are independently derived using the existing named seed manager.
- The unified runner fits preprocessing and base estimators only on training data and noisy training labels. It passes only calibration features and clean calibration labels to the frozen-estimator Platt calibrator. It does not refit the estimator on calibration data. Calibration completes before any test metric evaluation.
- The same test rows and clean test labels feed raw and calibrated metric calculations. Existing raw metric columns remain unchanged; matching `calibrated_*` columns are added on the same row to preserve one-row-per-condition compatibility with the report generator. Failed calibration records its error and NaN calibrated metrics, while retaining raw metrics.
- Result rows include calibration method/status/error, both split seeds, all three split fractions, train/calibration/test counts, actual corrupted training-label count, existing derived noise/model seeds, and serialized per-model parameters.
- Only sigmoid/Platt is configurable in the experiment protocol. Existing isolated isotonic utility remains available but is not included in this phase's runner configuration.
- Added a bounded one-model/two-noise-level CLI smoke configuration. It was not run against downloaded data; the end-to-end runner is covered by an offline synthetic fixture test.
- Documented the protocol in README, unified benchmarking docs, and `docs/calibration-rescue.md`; updated calibration-specific audit findings with a Phase 4 addendum.

### Files changed in Phase 4

- `src/trusttab/experiments/calibration_protocol.py` (new): configuration defaults and validation.
- `src/trusttab/experiments/splitting.py` (new): stratified three-way splitting and input/split validation.
- `src/trusttab/experiments/config.py`: normalize calibration configuration and select unified path when enabled.
- `src/trusttab/experiments/run.py`: pass calibration settings from CLI-loaded config into runner.
- `src/trusttab/experiments/benchmark.py`: integrate splitting, fitting, metric calculation, error reporting, and result metadata.
- `src/trusttab/evaluation/calibration.py`: clarify frozen-estimator and decision-score/probability behavior in existing helper documentation.
- `configs/experiments.yaml`: explicit disabled-by-default calibration settings.
- `configs/calibration_rescue_smoke.yaml` (new): bounded smoke configuration.
- `tests/test_calibration_splitting.py` (new): split validity and reproducibility.
- `tests/test_calibration_experiment.py` (new): offline end-to-end and leakage/isolation checks.
- `tests/test_calibration.py`: fitted-estimator/no-refit and supported score-interface coverage.
- `tests/test_config.py`: enabled/disabled and invalid calibration config coverage.
- `README.md`, `docs/unified-benchmarking.md`, `docs/robustness-reporting.md`, `docs/calibration-rescue.md` (new): methodology and output documentation.
- `TRUSTTAB_IMPLEMENTATION_AUDIT.md`: calibration-specific findings and addendum; original audit remains a historical snapshot.
- `IMPLEMENTATION_PROGRESS.md`: this Phase 4 record.

### Verification commands and outcomes

Run from `D:\trusttab` using the existing `.venv`:

| Check | Command | Outcome |
|---|---|---|
| Focused calibration/config/runner tests, final formatted tree | `.venv\Scripts\python.exe -m pytest tests\test_calibration_splitting.py tests\test_config.py tests\test_calibration_experiment.py tests\test_calibration.py tests\test_benchmark.py -q --basetemp .pytest-tmp-phase4-postformat` | **65 passed, 1 warning** in 9.07s. |
| Full test suite, final formatted tree | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase4-final2-full` | **311 passed, 1 warning** in 46.18s. Warning: sklearn single-label confusion-matrix warning in the existing one-class ROC-AUC test. |
| Ruff lint | `.venv\Scripts\ruff.exe check src tests` | Passed: `All checks passed!` |
| Formatting, initial | `.venv\Scripts\ruff.exe format --check src tests` | Initially identified four files requiring format; those files were formatted with Ruff. |
| Formatting, final | `.venv\Scripts\ruff.exe format --check src tests` | Passed: 27 files already formatted. |
| CLI help | `.venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed; displayed help and did not start an experiment. |
| Whitespace | `git diff --check` | Passed; Git emitted line-ending conversion notices only. |

### Limitations and unresolved verification

- Live UCI dataset retrieval was not performed; offline fixtures were used for runner tests.
- A real TabPFN fit remains **NOT VERIFIED**; no TabPFN dependency or hardware-specific bounded fit was attempted in this phase.
- The complete 200-condition matrix was not run. No claim about calibration benefit is made.
- Calibration with only 45 Cleveland calibration observations may have substantial estimation uncertainty. This follows the original split proportions and should be described when interpreting future results.
- Existing `.pytest-tmp-phase4-*` directories were present at phase resumption; they were not removed because their provenance is unclear.

**Phase 4 conclusion:** Synthetic runner tests establish the implemented split/fitting/calibration/evaluation separation for the tested sklearn model path. Focused and full suites passed on the final formatted tree. Lint, final formatting, CLI help, and whitespace checks passed. Real UCI retrieval, real TabPFN fitting, the full matrix, and empirical calibration analysis remain unverified/not run.

## Pre-experiment readiness check (post-Phase 4)

**Scope:** Bounded environment/data/model checks recommended after Phase 4. No source, configuration, dependency, or experiment design changes were made in this check; this progress record is the only updated project file. The existing dirty worktree from prior phases was preserved.

### Environment and dataset checks

- Python is 3.11.9; `trusttab` imports from the checkout's `src` tree.
- `tabpfn` and `torch` are absent from the current `.venv` (`find_spec` returned false for each). TabPFN was not installed because doing so would modify the shared environment and may require a substantial optional dependency/hardware setup.
- The German Credit loader succeeded using the environment's existing OpenML cache. It validated OpenML data ID 31 and returned 1,000 rows, 20 features, no missing features, and mapped target counts `{0: 700, 1: 300}`.
- A fresh temporary `data_home` forced a retrieval attempt and failed because the sandbox denied socket access (`PermissionError: [WinError 10013]`). Thus the successful German load does not establish fresh live OpenML retrieval.
- UCI Heart Disease dataset ID 45 retrieval also failed with `PermissionError: [WinError 10013]` at the network connection. The live UCI source/schema/distribution check is therefore **NOT VERIFIED** in this environment; offline fixture tests remain the available evidence.

### TabPFN bounded attempt

- Built TrustTab's TabPFN pipeline with its configured CPU parameters and 32 real German Credit rows. Model construction succeeded (`TabPFNAdapter`). The bounded `.fit()` attempt stopped with the adapter's expected actionable ImportError: `TabPFN is optional. Install it with 'pip install trusttab[tabpfn]'...` because the optional `tabpfn` package is absent.
- Result: actual TabPFN fit, prediction interface, runtime, and hardware compatibility are **NOT VERIFIED**. No dependency was installed and no successful fit is claimed.

### Real-data smoke benchmark

Using cached German Credit, Logistic Regression, seed 0, noise levels `[0.0, 0.2]`, and `max_iter=500`, ran the unified runner once with calibration disabled and once enabled. Both CSVs were written to a temporary directory and removed after checking the outputs.

| Mode | Label-noise rows | Status | Train/calibration/test counts |
|---|---:|---|---|
| Calibration disabled | 2 | `disabled` | 800 / 0 / 200 |
| Platt calibration enabled | 2 | `succeeded` | 700 / 150 / 150 |

The smoke produced raw and calibrated accuracy/ECE/Brier columns for both conditions. These single-seed smoke values are only execution evidence; they do not establish calibration effectiveness and were not used to change the protocol. No Cleveland real-data smoke was possible because live UCI access is blocked and no verified local dataset cache was present.

### Commands and outcomes

| Check | Command or invocation | Outcome |
|---|---|---|
| Runtime and optional imports | `.venv\Scripts\python.exe -c "import sys, importlib.util; ..."` | Python 3.11.9 and TrustTab import passed; `tabpfn_available=False`, `torch_available=False`. |
| Cached German loader | `.venv\Scripts\python.exe -c "from trusttab.data.german_credit import load_german_credit; X,y=load_german_credit(); ..."` | Passed: (1000, 20), target counts 700/300, no missing features. |
| Fresh OpenML retrieval | `.venv\Scripts\python.exe -c "import tempfile; ... load_german_credit(data_home=d.name) ..."` | Failed at network access with WinError 10013; direct retrieval **NOT VERIFIED**. |
| UCI Cleveland retrieval | `.venv\Scripts\python.exe -c "from trusttab.data.heart_disease import load_heart_disease; ..."` | Failed at network access with WinError 10013; live retrieval **NOT VERIFIED**. |
| TabPFN bounded attempt | Python invocation constructed `build_model('tabpfn', X.iloc[:32], random_state=123, model_params={'device':'cpu','show_progress_bar':False})` and called `.fit(...)` | Pipeline construction passed; fit emitted expected missing-optional-dependency ImportError. Real fit **NOT VERIFIED**. |
| Real German smoke, calibration off/on | Python invocation called `run_robustness_benchmark` with dataset `german_credit`, LR, seed `[0]`, levels `[0.0,0.2]`; ran each `calibration_settings={'enabled': False}` and `{'enabled': True}`; wrote CSVs to `TemporaryDirectory` | Passed assertions: two label-noise rows per mode, expected statuses, split counts, and both CSV outputs. Enabled mode completed sigmoid calibration. |

### Readiness conclusion

The configured German Credit + Logistic Regression smoke path executes with calibration off and on, using the expected partition sizes and producing both metric sets. This is bounded integration evidence, not evidence for Cleveland live data or TabPFN. Fresh network retrieval for both sources and a real TabPFN fit remain **NOT VERIFIED** because the current environment blocks network sockets and lacks the optional TabPFN/PyTorch packages. Do not launch the 200-condition matrix until these two environment/data uncertainties are resolved or explicitly accepted with verified data and model environments. No matrix, statistical analysis, release task, commit, or push was performed.

## Phase 6 â€” Final research readiness

**Status:** Readiness checks, a bounded TabPFN verification script, a calibration-aware report path, and an analysis plan are prepared. The current environment cannot complete TabPFN fitting or fresh dataset retrieval. Blueprint-level statistical methods remain incomplete. The full experiment remains blocked and was not launched.

### Starting state and environment

- Branch: `chore/remove-github-actions`. The worktree was already dirty with changes and untracked artifacts from Phases 1â€“5. Those were preserved. Phase 6 edited `src/trusttab/reporting/report.py` and `src/trusttab/experiments/benchmark.py`, added regression tests in `tests/test_reporting.py` and `tests/test_benchmark.py`, updated `docs/robustness-reporting.md`, added `scripts/verify_tabpfn.py`, and updated this progress entry.
- Python 3.11.9, Windows 10 (build 26300), TrustTab 0.1.0 from the checkout, 18 logical CPUs, and 25,158,070,272 bytes total physical memory (about 23.4 GiB) were detected.
- Installed versions: NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.17.1, Matplotlib 3.11.2, PyYAML 6.0.3, and XGBoost 3.2.0. TabPFN and PyTorch are absent. TabPFN remains pinned as the optional `tabpfn==10.0.0` extra; dependencies were not installed or changed.
- XGBoost import was verified successfully (`xgboost.__version__ == 3.2.0`). This establishes package import only, not XGBoost model fit on both benchmark datasets.
- Windows CIM hardware queries returned Access denied; `nvidia-smi` is not available. CPU count and total RAM are known, but GPU presence/VRAM and detailed CPU model are **NOT VERIFIED**. Current configured TabPFN device remains CPU; no research configuration was changed.
- The inspected scikit-learn data cache contains German Credit (OpenML ID 31) and Adult (ID 1590) artifacts. No verified local UCI Cleveland cache was found in the inspected project/OpenML locations.

### TabPFN verification script and result

Added `scripts/verify_tabpfn.py`. It uses a fixed seed, loads German Credit through the actual TrustTab loader, selects a stratified 64-row sample (48 fit, 16 prediction), reads TabPFN parameters unchanged from `configs/experiments.yaml`, constructs TrustTab's preprocessing/model pipeline, times fit and prediction, checks prediction/probability shapes, finite/range/normalization and binary class labels, and emits JSON with Python/library/hardware metadata. It never installs dependencies. Exit code 2 denotes a blocked optional dependency or environment/data precondition; exit code 1 denotes an execution/validity failure; exit code 0 requires a fit and all output checks to succeed.

Executed `.venv\Scripts\python.exe scripts\verify_tabpfn.py --seed 123`. The script reported `status: BLOCKED`, with the exact adapter error that TabPFN is optional and should be installed using `pip install trusttab[tabpfn]`. It successfully loaded the cached 1,000 Ã— 20 German data, formed the 48/16 split, built `TabPFNAdapter` from the unchanged CPU config, and recorded a 0.016-second failed fit attempt. It did not predict. The actual script exit was 2 (captured as `SCRIPT_EXIT_CODE=2` in the verification wrapper). TabPFN fitting, probabilities, runtime, checkpoint access, and hardware compatibility therefore remain **NOT VERIFIED**.

### Dataset verification

- German Credit loader success from the existing cache: OpenML data ID 31, 1,000 rows, 20 features, no missing feature values, target counts 700/300. This validates the cached artifact through loader checks, not a fresh source download.
- A fresh temporary `data_home` retrieval attempt failed at OpenML network access with `PermissionError: [WinError 10013]`; fresh OpenML retrieval is **BLOCKED BY ENVIRONMENT**.
- `load_heart_disease()` attempted its configured UCI URL and failed at socket connection with the same `WinError 10013`; live UCI retrieval is **BLOCKED BY ENVIRONMENT**.
- Source inspection confirms the Cleveland loader reads only archive member `processed.cleveland.data`, enforces 303 Ã— 13 schema and original `num` counts `{0:164, 1:55, 2:36, 3:35, 4:13}`, maps 0â†’0 and 1â€“4â†’1, preserves `num` in target attrs, leaves rows intact, and preserves source missing markers as NaN in `ca`/`thal`. `tests/test_heart_disease.py` uses offline fixtures; it passed in focused and full test runs. Live payload equivalence is **NOT VERIFIED**.

### Reproducibility and leakage audit

- `derive_seed` uses SHA-256 keyed SeedSequence derivation for stable, distinct named streams. Runner code derives split, calibration-split, label-noise, model, and feature-shift seeds. The same model seed and same noisy-label realization are deliberately shared by model comparisons at a given dataset/noise/seed, enabling paired comparisons; noise differs by dataset/rate/seed.
- The runner splits before fitting preprocessing, applies the independent Bernoulli noise function only to `y_train`, retains clean calibration/test labels, calibrates the already-fitted estimator using calibration-only predictions and labels, and computes raw/calibrated metrics against the same test rows. Calibration failure retains raw metrics and sets calibrated metrics to NaN with `calibration_status=failed` and an error string. Existing Phase 4 runner tests check these invariants.
- Calibration mode is opt-in and uses 70/15/15; non-calibration mode preserves configured 80/20. No methodology change was made.

### Statistical analysis plan and limitations

No statistical procedure or claim was added in this phase. The existing `statistics/comparison.py` supports paired t-tests, Wilcoxon signed-rank tests, and multiple-testing adjustment; it does not implement Friedman/Nemenyi. The report's current pairwise model comparison uses matched dataset/seed keys, Holm correction, and a 95% t interval for the paired mean difference. It can pool paired rows across datasets, which should not be interpreted as inference over a population of datasets when only two fixed datasets are present.

For eventual analysis, preserve the seed-level raw table and analyze each dataset separately for primary inference:

1. For model comparisons at a fixed dataset and noise level, pair models by the same seed. The same split and, for a noise condition, the same generated noisy training labels are shared across models. Report all five paired differences, mean difference, and a 95% paired t interval as a small-sample descriptive interval; give the paired Wilcoxon result as exploratory and report the low power at n=5.
2. For noise degradation, compare each model's noise-level result to its zero-noise result at the same dataset and seed. Pair raw metrics within those matched conditions; report seed-level deltas and intervals. Do not count individual test observations as experimental replicates.
3. For calibration rescue, compare raw versus Platt ECE, adaptive ECE, MCE, Brier score, and log loss within the exact same model/dataset/noise/seed test set. Treat accuracy, balanced accuracy, macro-F1, and ROC-AUC as secondary outcomes; sigmoid calibration is not expected to improve class decisions by design. Report calibration failures and undefined metrics, never replace them with zero or another fabricated value.
4. Summarize accuracy, balanced accuracy, macro-F1, ROC-AUC, Brier, log loss, ECE, adaptive ECE, and MCE by dataset/model/noise, with seed means, SDs, and intervals. Keep metric conventions and ECE bins fixed in advance.
5. Correct p-values across a predeclared family of comparisons (for example, within each dataset and metric across planned model/noise contrasts) using Holm. If Friedman/Nemenyi is later added, use an explicitly aligned block design and do not treat the 2 datasets Ã— 5 levels Ã— 5 seeds as 50 independent blocks: noise levels reuse the same split/test set, and there are only two dataset identities. Five seeds and two datasets make broad generalization claims weak; present these analyses as exploratory and emphasize effect sizes/uncertainty.
6. Compare datasets descriptively and separately. Two datasets are insufficient to support inference about a wider population of tabular datasets. Cleveland's 45 calibration and 46 test examples under the required 70/15/15 split also make ECE/MCE and calibration estimates unstable; do not change split proportions after viewing results.

### Result/reporting readiness audit

- The unified result CSV includes dataset/model/scenario/seed, derived split/noise/model/shift seeds, noise level and actual `n_corrupted`, model parameters, calibration split fractions/counts/status/error, raw metrics, and `calibrated_*` metrics on the same row. The tracker stores run ID, UTC start/completion timestamps, configuration hash, dataset fingerprint, environment versions, and git metadata in run metadata/manifest. Per-condition training duration/timestamp and run ID are not present in each raw row.
- Calibration failures are distinguishable in raw results; whole-run failures are distinguished by tracker manifest status. Tracked `complete()` writes the raw results CSV before generating its report.
- **Reporting correction:** Phase 6 extends metric summaries to available `calibrated_*` metrics, writes `calibration_status.csv` with per-condition status/error information, and writes `calibration_comparison.csv` with same-condition descriptive raw/calibrated probability metrics. Failed and unavailable calibrated metrics remain NaN. The existing inferential table is intentionally limited to its prior raw-metric comparison family; calibration comparisons remain descriptive pending a predeclared analysis plan. Report tests verify failed rows are visible and no values are fabricated. Feature-shift rows are marked `not_applicable` when calibration is enabled but those rows are not evaluated with calibrated predictions.
- **Statistics gap:** Friedman/Nemenyi are absent; current report comparisons use paired t-tests with Holm and can pool across the two datasets. Do not interpret those pooled comparisons as dataset-population evidence. The plan above is not implemented in this phase.

### Phase 6 verification commands and outcomes

| Check | Command | Outcome |
|---|---|---|
| Python/dependency inventory | `.venv\Scripts\python.exe -c "import importlib.metadata as metadata; packages=('numpy','pandas','scikit-learn','scipy','matplotlib','PyYAML','xgboost','tabpfn','torch'); print({name:(metadata.version(name) if name in {dist.metadata['Name'] for dist in metadata.distributions()} else 'unavailable') for name in packages})"` | Versions recorded above; TabPFN and PyTorch unavailable. |
| CPU/RAM/GPU inventory | `.venv\Scripts\python.exe scripts\verify_tabpfn.py --seed 123`; `Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory`; `Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors`; `Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM`; `Get-Command nvidia-smi -ErrorAction SilentlyContinue` | Script found 18 logical CPUs and 25,158,070,272 bytes RAM; CIM access denied; `nvidia-smi` absent; GPU **NOT VERIFIED**. |
| XGBoost import | `.venv\Scripts\python.exe -c "import xgboost; print('xgboost_import=ok'); print('xgboost_version',xgboost.__version__)"` | Passed; 3.2.0. |
| Focused dataset/calibration/matrix/statistics/tracking/report tests | `.venv\Scripts\python.exe -m pytest tests\test_heart_disease.py tests\test_calibration_experiment.py tests\test_calibration_splitting.py tests\test_core_matrix.py tests\test_statistics.py tests\test_tracking.py tests\test_reporting.py -q --basetemp .pytest-tmp-phase6-focused` | **66 passed** in 71.63s. |
| Report-focused tests | `.venv\Scripts\python.exe -m pytest tests\test_reporting.py -q --basetemp .pytest-tmp-phase6-reporting` | **10 passed** in 19.33s. |
| Final report/runner regression tests | `.venv\Scripts\python.exe -m pytest tests\test_reporting.py tests\test_benchmark.py -q --basetemp .pytest-tmp-phase6-report-final` | **25 passed, 1 warning** in 25.53s. |
| Full test suite, final tree | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase6-final2-full` | **313 passed, 1 warning** in 43.46s. Warning: sklearn single-label confusion-matrix warning in existing one-class ROC-AUC test. |
| Ruff lint | `.venv\Scripts\ruff.exe check src tests scripts` | Passed: `All checks passed!` |
| Ruff format | `.venv\Scripts\ruff.exe format --check src tests scripts` | Passed: 28 files already formatted. |
| CLI help | `.venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed; help displayed; no experiment launched. |
| Whitespace | `git diff --check` | Passed; Git emitted only line-ending conversion notices. |
| TabPFN bounded script | `.venv\Scripts\python.exe scripts\verify_tabpfn.py --seed 123` | Script output status `BLOCKED`, process exit 2 due missing optional TabPFN; no fit/prediction was claimed. |

### Phase 6 conclusion

The cached German data path, both calibration modes on a bounded real German Logistic Regression run, offline Cleveland fixtures, seed/noise/split integration, calibration-aware report outputs, and the complete current test suite are **VERIFIED**. Fresh OpenML and UCI access are **BLOCKED BY ENVIRONMENT**. TabPFN construction is verified, but its real fit/prediction are **NOT VERIFIED** because TabPFN/PyTorch are not installed and no optional dependencies were installed. Blueprint-level statistical methods remain incomplete. Therefore TrustTab is **NOT READY for the full research run**. Do not launch the 200-condition matrix until a network-enabled verified dataset environment, real bounded TabPFN fit, and final statistical-method decision are addressed. No full matrix, commit, push, merge, or publication occurred.

### Phase 7 â€” Statistical analysis and inferential reporting

**Scope and method:** Implemented reusable seed-blocked research statistics in
`trusttab.statistics` and integrated them into `generate_robustness_report`.
No models were trained and no experiment results were generated. Analyses use
only `experiment_type == "label_noise"` rows so the separately emitted clean
baseline does not duplicate zero noise. Datasets remain separate.

- Friedman uses complete matched seed blocks for each dataset/noise/metric
  across four core models. Incomplete blocks and missing seeds are explicit;
  duplicate blocks fail validation. Average ranks and tie-corrected Friedman
  Q are reported, including an all-model-tied case (`Q=0`, `p=1`). Holm adjusts
  the planned noise-level omnibus tests within each dataset/metric.
- Nemenyi emits all six model-pair rows, average ranks and CD information, but
  computes adjusted pair p-values only after a significant Holm-adjusted
  omnibus result. CD plots are emitted only for eligible groups. Nemenyi's
  studentized-range p-values are not Holm-adjusted again.
- Raw/calibrated ECE and Brier comparisons pair by dataset/model/noise/seed.
  Only successful calibration rows form pairs. Incomplete comparisons are
  descriptive and have no inferential p-value or rank-biserial effect.
  Planned but unavailable members remain missing in output and count as p=1
  internally for Holm family-size correction, preventing failed rows from
  shrinking the predeclared family.
- Per-seed slopes use OLS across five noise levels. Normalized worsening is
  `direction * slope * 0.10 / (upper_bound - lower_bound)` with theoretical
  metric range `[0,1]`; positive means deterioration. H1 primary comparison is
  accuracy/ECE; accuracy/Brier and macro-F1/reliability comparisons are also
  included. Holm is applied within dataset across the model/performance/
  reliability slope family.
- Descriptive means include sample SD and 95% Student-t intervals over seeds
  when at least two finite values are available. Outputs include
  `friedman_results.csv`, `nemenyi_results.csv`,
  `wilcoxon_calibration_results.csv`, `degradation_slope_results.csv`,
  `degradation_slope_comparisons.csv`, `effect_sizes.csv`, and
  `statistical_summary.csv`. The legacy `statistical_comparisons.csv` is now
  computed separately per dataset rather than pooling datasets.
- Updated `docs/statistics.md`, `docs/robustness-reporting.md`,
  `docs/unified-benchmarking.md`, and `README.md` with pairing, multiplicity,
  slope normalization, uncertainty, and limitations.

**Limitations:** The five-seed/four-model Friedman chi-square approximation is
exploratory and low-reliability at this small block count. With five nonzero
paired observations, the smallest exact two-sided Wilcoxon p-value is 0.0625,
before Holm adjustment. The two datasets are fixed cases, not a sample of a
dataset population. Synthetic tests verify implementation only and are not
empirical research findings. The 200-condition experiment remains unlaunched.

**Phase 7 changed files:** `src/trusttab/statistics/research.py` (new analysis
functions), `src/trusttab/statistics/__init__.py` (public exports),
`src/trusttab/reporting/report.py` (CSV/report/CD integration and dataset-wise
legacy comparisons), `tests/test_research_statistics.py` (new independent
synthetic tests), `tests/test_reporting.py` (output, schema, and diagram
checks), `docs/statistics.md`, `docs/robustness-reporting.md`,
`docs/unified-benchmarking.md`, `README.md`, and this progress log. No research
configuration, dataset, model, noise, or calibration protocol was changed.

**Phase 7 verification commands and outcomes:**

| Check | Command | Outcome |
|---|---|---|
| Focused statistics/reporting | `.venv\Scripts\python.exe -m pytest tests\test_statistics.py tests\test_research_statistics.py tests\test_reporting.py -q --basetemp .pytest-tmp-phase7-holm` | **48 passed** in 38.11s. |
| Full test suite | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase7-release` | **321 passed, 1 warning** in 89.50s. Existing warning: scikit-learn single-label confusion-matrix warning in `test_roc_auc_is_nan_when_test_split_contains_one_class`. |
| Ruff lint | `.venv\Scripts\ruff.exe check src tests scripts` | Passed: `All checks passed!` |
| Ruff formatting | `.venv\Scripts\ruff.exe format --check src tests scripts` | Passed: 30 files already formatted. |
| CLI help | `.venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed; help displayed; no experiment launched. |
| Whitespace | `git diff --check` | Passed; Git printed only existing LF-to-CRLF conversion notices. |

No 200-condition run, commit, push, merge, release, or publication occurred.

## Phase 8 â€” Final pre-experiment readiness (preflight; blocked)

**Status:** Preflight checks and one bounded German Credit smoke completed.
Phase 8 acceptance criteria are **not met**; no full matrix was run.

### Current environment and TabPFN

- Python 3.11.9; NumPy 2.4.6; pandas 3.1.0; scikit-learn 1.9.1; SciPy
  1.17.1; Matplotlib 3.11.2; PyYAML 6.0.3; XGBoost 3.2.0.
- TabPFN and PyTorch are unavailable in the project `.venv`. The bounded
  verification script loaded the cached 1,000 Ã— 20 German Credit dataset,
  prepared 48 fit and 16 predict rows, constructed the configured TrustTab
  adapter, then stopped at the optional dependency import error. No fit or
  prediction was claimed. Command:
  `.venv\Scripts\python.exe scripts\verify_tabpfn.py --seed 123` (exit 1;
  JSON status `BLOCKED`).
- Hardware probe: Windows 10, 18 logical CPUs, 25,158,070,272 bytes reported
  physical RAM. `nvidia-smi` and PyTorch GPU discovery are unavailable, so GPU
  capability is **NOT VERIFIED**.
- The installed `.venv` was not changed. TabPFN requires PyTorch and its first
  model use downloads a model checkpoint after applicable account/license
  setup. A bounded fit therefore needs both a separate optional-dependency
  environment and checkpoint access. Current official installation guidance
  distinguishes CPU/GPU PyTorch setup; its CPU path avoids GPU wheel overhead,
  but CPU runtime still needs real measurement. The pinned TabPFN 10.0.0
  release is newly published (2026-11-13), so TrustTab API/runtime compatibility
  remains **NOT VERIFIED**. No credentials were requested or handled.

### Dataset retrieval

- German Credit is available through the existing OpenML cache and loaded as
  1,000 rows Ã— 20 features, target counts `{0: 700, 1: 300}`, with no missing
  feature values. This is cached-data validation, not fresh OpenML retrieval.
- `data/raw/` and `data/processed/` contain only `.gitkeep`; there is no local
  Cleveland payload cache there.
- A real call to `load_heart_disease(data_home="data/raw")` failed at the
  outbound socket with `PermissionError: [WinError 10013]`; fresh UCI data
  retrieval remains **BLOCKED BY ENVIRONMENT**. Fixture validation from prior
  Phase 2 is not a substitute for the current live-payload check.

### Bounded actual-data smoke

Ran a single-seed German Credit benchmark for Logistic Regression, Random
Forest, and XGBoost at 10% noise with Platt calibration enabled. The runner
returned six rows (baseline + noisy condition per model); raw and calibrated
ECE/Brier were finite and all six rows had `calibration_status=succeeded`.
The Phase 7 report generator consumed those actual rows and wrote every
required statistical CSV to a temporary directory, which was removed after
verification. Its Friedman output correctly marked all 30 dataset/noise/
metric conditions `insufficient_complete_blocks` because this deliberately
bounded smoke had only one seed and three models. This is an integration smoke,
not evidence for research claims and not coverage of both datasets or TabPFN.

Exact command:

```powershell
.venv\Scripts\python.exe -c "import tempfile; from trusttab.experiments.config import load_experiment_config; from trusttab.experiments.benchmark import run_robustness_benchmark; from trusttab.reporting import generate_robustness_report; cfg=load_experiment_config('configs/experiments.yaml'); models=['logistic_regression','random_forest','xgboost']; results=run_robustness_benchmark(dataset_names=['german_credit'],label_noise_levels=[0.1],seeds=[0],model_names=models,model_params=cfg['model_params'],calibration_settings={'enabled':True,'method':'sigmoid','train_size':0.7,'calibration_size':0.15,'test_size':0.15}); print('rows',len(results)); print('models',sorted(results.model.unique())); print('types',results.groupby(['experiment_type','calibration_status'],dropna=False).size().to_dict()); print('finite_raw',results[['accuracy','ece','brier_score']].notna().all(axis=1).sum()); print('finite_calibrated',results[['calibrated_ece','calibrated_brier_score']].notna().all(axis=1).sum()); temp=tempfile.TemporaryDirectory(prefix='trusttab-phase8-report-'); report=generate_robustness_report(results,temp.name); from pathlib import Path; required=['friedman_results.csv','nemenyi_results.csv','wilcoxon_calibration_results.csv','degradation_slope_results.csv','effect_sizes.csv','statistical_summary.csv']; print('report_files',{name:(Path(temp.name)/name).is_file() for name in required}); print('friedman_status',report['friedman_results'].status.value_counts(dropna=False).to_dict()); temp.cleanup()"
```

### Phase 8 blocker status and next actions

| Acceptance check | Status | Remaining evidence |
|---|---|---|
| Real TabPFN fit/predict | **BLOCKED BY ENVIRONMENT** | Create/use an isolated optional-dependency environment; install compatible CPU PyTorch + pinned TabPFN; access the checkpoint under its current license/auth requirements; then run the existing bounded verifier. |
| Fresh German Credit | **NOT VERIFIED** | A network-enabled OpenML retrieval and cache fingerprint comparison. |
| Fresh Cleveland | **BLOCKED BY ENVIRONMENT** | UCI request currently fails socket policy; retrieve the exact UCI 45 Cleveland payload in a network-enabled environment and validate loader schema/counts. |
| Classical and XGBoost models on German Credit | **PARTIALLY VERIFIED** | Bounded one-seed smoke succeeded with calibration; broader config remains unrun. |
| All four models on both datasets | **NOT VERIFIED** | Requires TabPFN fit and a usable Cleveland source. |
| Calibration on real-data smoke | **VERIFIED for tested German LR/RF/XGBoost rows** | Cleveland and TabPFN calibration rows are not verified. |
| Phase 7 reporting on actual smoke output | **VERIFIED for bounded German output** | Inferential outputs are correctly marked insufficient; full result completeness remains unverified. |
| Reproducibility metadata | **PARTIALLY VERIFIED** | Runner emits model/noise/split seeds and config parameters; live source fingerprints and real TabPFN package/checkpoint metadata remain unavailable. |

No dependencies were installed, no project experiment configuration or
methodology was changed, no persistent experiment output was written, and the
200-condition matrix was **NOT** launched. No commit, push, merge, release, or
publication occurred. The only Phase 8 file change is this progress entry.

### Phase 8 update â€” isolated TabPFN install and live-source verification (2026-11-14)

The user explicitly authorized isolated TabPFN dependency installation and Phase 8
blocker resolution. The project `.venv` was not modified. A separate environment
was created at `.venv\phase8-tabpfn` (inside the already ignored `.venv` tree).
It contains CPU-only PyTorch 2.15.0, TabPFN 10.0.0, and versions matching the
project environment for NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, SciPy
1.17.1, Matplotlib 3.11.2, PyYAML 6.0.3, and XGBoost 3.2.0. Extra TabPFN
transitive dependencies were installed only in this isolated environment.

Exact install commands:

```powershell
.venv\phase8-tabpfn\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5"
.venv\phase8-tabpfn\Scripts\python.exe -m pip install "tabpfn==10.0.0" "numpy==2.4.6" "pandas==3.1.0" "scikit-learn==1.9.1" "scipy==1.17.1" "matplotlib==3.11.2" "PyYAML==6.0.3" "xgboost==3.2.0"
```

`import tabpfn` and `from tabpfn import TabPFNClassifier` both succeed in this
environment. Running the existing bounded verifier with `TABPFN_NO_BROWSER=1`,
no `TABPFN_TOKEN` in the child process, a Phase 8-local `SKB_DATA_DIRECTORY`,
and a Phase 8-local `TABPFN_MODEL_CACHE_DIR` reached model fitting but failed
at checkpoint download with the concrete error
`TabPFNHuggingFaceGatedRepoError`: the default checkpoint `Prior-Labs/tabpfn_3_5`
is gated and requires the user to accept its terms and authenticate. No browser
login was launched, no credentials were read or printed, and no checkpoint was
downloaded. The bounded real TabPFN fit/prediction therefore remains
**BLOCKED BY REQUIRED ACCOUNT/LICENSE ACCEPTANCE**, not by Python compatibility
or adapter construction. The official TabPFN page documents checkpoint access
and says CPU execution is slow; the project uses only 48 fit rows in this
bounded check. See [TabPFN 10.0.0 on PyPI](https://pypi.org/project/tabpfn/10.0.0/)
and [Prior Labs gated-model access instructions](https://docs.priorlabs.ai/how-to-access-gated-models).

Fresh-source dataset verification succeeded:

- OpenML German Credit ID 31 was fetched into `.venv\phase8-tabpfn\openml-cache`.
  It is 1,000 Ã— 20, has target counts `{0: 700, 1: 300}`, and no missing
  features. Gzip source `credit-g.arff.gz` SHA-256:
  `a42c7d580d3415586b230a4315c118c93d787aaf2a145d0fd81e7e3efeb85be1`.
  Decompressed ARFF SHA-256:
  `768be84ef749110d5fcac109a6541f2df190e8c6987e1dc432c0a4f29d0642f5`.
  The normalized DataFrame fingerprint (pandas row hashing over 20 features and
  mapped target) is `929fac3aa8834d82be806c8e0dde2e9d82872674bcbe329380a08abf47cb61d1`.
- UCI Heart Disease ID 45 was fetched from the configured UCI ZIP, reading only
  `processed.cleveland.data`. It is 303 Ã— 13; original `num` counts are
  `{0:164, 1:55, 2:36, 3:35, 4:13}`; binary counts are `{0:164, 1:139}`; the
  six missing feature cells are preserved. UCI archive SHA-256:
  `b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc`;
  selected member SHA-256:
  `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`.
  The loader-created normalized source cache is
  `data/raw/heart_disease_uci45_cleveland.csv` (ignored by Git), SHA-256
  `953d45066c221ee00cf8de72cfa4ab2086eb4e231fef6dda8fc3051b17840e61`.

A bounded real-data smoke ran one seed and 10% noise for Logistic Regression,
Random Forest, and XGBoost on both datasets with the 70/15/15 Platt calibration
protocol. It returned 12 rows (baseline and noisy result for each of six
model/dataset combinations); all rows had finite raw accuracy, balanced
accuracy, macro-F1, ECE, and Brier; all 12 rows had successful calibration and
finite calibrated ECE/Brier. The report generator wrote all six required
statistical CSVs in a temporary directory, then the temporary output was
removed. Friedman correctly labeled all 60 requested groups
`insufficient_complete_blocks`; the 80 calibration comparisons were
`insufficient_pairs_descriptive`. This is smoke evidence only, not empirical
research evidence.

Smoke command (a process-local cache override is needed because the current
benchmark registry calls the Heart loader without its `data_home` argument):

```powershell
.venv\Scripts\python.exe -c "import tempfile; from pathlib import Path; import trusttab.experiments.benchmark as b; from trusttab.data import load_heart_disease; from trusttab.experiments.config import load_experiment_config; from trusttab.reporting import generate_robustness_report; b.DATASET_LOADERS['heart_disease']=lambda: load_heart_disease(data_home='data/raw'); cfg=load_experiment_config('configs/experiments.yaml'); models=['logistic_regression','random_forest','xgboost']; results=b.run_robustness_benchmark(dataset_names=['german_credit','heart_disease'],label_noise_levels=[0.1],seeds=[0],model_names=models,model_params=cfg['model_params'],calibration_settings={'enabled':True,'method':'sigmoid','train_size':0.7,'calibration_size':0.15,'test_size':0.15}); print('rows',len(results)); print('dataset_models',sorted(set(zip(results.dataset,results.model)))); print('status_counts',results.groupby(['experiment_type','calibration_status'],dropna=False).size().to_dict()); print('all_raw_core_metrics_finite',bool(results[['accuracy','balanced_accuracy','macro_f1','brier_score','ece']].notna().all(axis=1).all())); print('all_calibrated_probability_metrics_finite',bool(results[['calibrated_ece','calibrated_brier_score']].notna().all(axis=1).all())); temp=tempfile.TemporaryDirectory(prefix='trusttab-phase8-two-data-report-'); report=generate_robustness_report(results,temp.name); required=['friedman_results.csv','nemenyi_results.csv','wilcoxon_calibration_results.csv','degradation_slope_results.csv','effect_sizes.csv','statistical_summary.csv']; print('report_files',{name:(Path(temp.name)/name).is_file() for name in required}); print('friedman_status',report['friedman_results'].status.value_counts(dropna=False).to_dict()); print('calibration_pairs',report['wilcoxon_calibration_results'].status.value_counts(dropna=False).to_dict()); temp.cleanup()"
```

**Additional integration issue:** `benchmark.py` obtains a dataset using
`DATASET_LOADERS[name]()` with no cache/data-home argument. The UCI cache is
therefore not used by the unmodified runner; the bounded smoke used a
process-local override only. A future implementation change is needed if
benchmark runs must consume offline cached Cleveland without network access.
No source change was made for this issue during the Phase 8 preflight.

**Readiness:** Real German and Cleveland source retrieval/schema checks are now
**VERIFIED** in this environment. Logistic Regression, Random Forest, XGBoost,
Platt calibration, and Phase 7 report integration are **VERIFIED** on bounded
real data for both datasets. TabPFN fit/prediction is still blocked pending
user-side gated-checkpoint terms/authentication. The full four-model/two-dataset
smoke and all-four-model matrix are **NOT VERIFIED**. Do not launch the 200-run
matrix. No project configuration, model methodology, source implementation,
tracked data, or earlier user changes were overwritten; only this progress log
was edited for this update. No commit, push, merge, release, or publication
occurred.

### Phase 8 methodology compatibility note

Source inspection during readiness review found that `build_model()` currently
wraps TabPFN in the same `build_preprocessor()` as the classical estimators.
That preprocessor median-imputes and **StandardScales numeric columns**, then
one-hot encodes categorical columns. Blueprint Part I says TabPFN should not be
scaled and should share the categorical encoding convention used for LR. The
current shared scaling therefore deviates from the blueprint. Current TabPFN
10.0 documentation also recommends passing raw numeric/categorical columns
without scaling or one-hot encoding, which differs from the older blueprint's
encoding recommendation. This must be resolved explicitly before the final
matrix; no preprocessing change was made in this preflight. The priority is to
avoid silently comparing an unintended TabPFN input protocol. See the
[TabPFN 10.0 usage guidance](https://pypi.org/project/tabpfn/10.0.0/).

### Phase 8 update â€” TabPFN preprocessing alignment and regression verification

Following the source review, the blueprint requirement was applied narrowly:
TabPFN still receives training-fitted median imputation and one-hot categorical
encoding, but numeric inputs are no longer standardized. Logistic Regression
continues to use the existing scaler. Random Forest and XGBoost preprocessing
remain unchanged. This corrects the explicit Part I TabPFN preprocessing
instruction without changing the dataset/model/noise/seed matrix. Current
TabPFN 10 documentation recommends native categorical input; the project
retains the original blueprint's one-hot categorical convention and records
that version-specific limitation in `docs/tabpfn.md`.

Phase 8 source/doc/test additions in this update: `src/trusttab/data/preprocessing.py`
adds the optional numeric-scaling switch, `src/trusttab/models/model_factory.py`
uses it only for TabPFN, `tests/test_models.py` asserts the TabPFN pipeline has
imputation plus categorical one-hot encoding but no scaler, and `docs/tabpfn.md`
records the protocol and limitation. This update supersedes the earlier
preflight note saying no source changes were made; these four files were
changed afterward to correct the verified blueprint mismatch.

Verification after the change:

| Check | Command | Result |
|---|---|---|
| Focused model/preprocessing/adapter tests | `.venv\Scripts\python.exe -m pytest tests\test_models.py tests\test_preprocessing.py tests\test_tabpfn_adapter.py -q --basetemp .pytest-tmp-phase8-preprocess` | **24 passed** in 2.28s. |
| Full test suite | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase8-code-full` | **321 passed, 1 existing warning** in 26.78s. Warning: scikit-learn single-label confusion-matrix warning in the one-class ROC-AUC benchmark test. |
| Ruff lint | `.venv\Scripts\ruff.exe check src tests scripts` | Passed: all checks passed. |
| Ruff format | `.venv\Scripts\ruff.exe format --check src tests scripts` | Passed: 30 files already formatted. |

This code correction does not resolve the gated model checkpoint. Real TabPFN
fit/prediction remains blocked until the user accepts the checkpoint terms and
authenticates in their own local environment. No credential should be shared.
The two-dataset, three-classical-model smoke results above remain applicable;
no full four-model smoke or 200-condition experiment has been run.

### Phase 8 update â€” authentication source audit and cache-path integration

The isolated environment was inspected without reading or printing credential
values. It contains TabPFN 10.0.0 and CPU PyTorch 2.15.0. Hugging Face
authentication, Prior Labs authentication, and the TabPFN-3.5 checkpoint cache
were each unavailable at the time of this check.

The authentication paths are distinct in the installed TabPFN 10.0.0 source:
`tabpfn.browser_auth.get_cached_token()` reads `TABPFN_TOKEN` or its local
Prior Labs token cache for license acceptance, while the checkpoint loader
uses `huggingface_hub` to access `Prior-Labs/tabpfn_3_5`. The real prior fit
attempt raised `TabPFNHuggingFaceGatedRepoError` for this repository; the
installed exception explicitly directs the user to accept the Hugging Face
terms and run `hf auth login` (or provide `HF_TOKEN`). The official TabPFN
10.0.0 package documentation describes the separate Prior Labs license and
headless `TABPFN_TOKEN` requirement. Consequently this environment needs both
the Hugging Face gated-repository access and the Prior Labs license/token
authorization for a non-interactive fit. Neither credential was inspected,
requested, printed, or changed in this update. Official references:
[TabPFN 10.0.0 authentication and checkpoint documentation](https://pypi.org/project/tabpfn/10.0.0/)
and [Prior Labs model-weight access](https://docs.priorlabs.ai/models/accessing-model-weights).

The Cleveland runner integration issue was fixed. The benchmark now accepts
`dataset_data_home` as either a shared path or a dataset-to-path mapping and
forwards it to selected registered loaders. `configs/experiments.yaml` maps
only Cleveland to `data/raw`, while German Credit retains its existing
scikit-learn/OpenML cache behavior. This avoids sending the UCI cache path to
OpenML. Configuration validation checks the new path setting. An initial
bounded attempt using one shared path demonstrated why per-dataset mapping is
needed: it made OpenML query the network instead of using its separate cache;
the corrected mapping passed the following real-data smoke test.

The corrected integration ran one seed, 10% Bernoulli label noise, and
Logistic Regression on cached German Credit and Cleveland, with calibration
enabled (70/15/15). It produced four rows (baseline and noisy condition per
dataset), all four calibrations succeeded, all raw core metrics and calibrated
ECE/Brier values were finite, and tracked output contained `results.csv`, the
six Phase 7 statistical CSVs, `report.md`, `manifest.json`, `environment.json`,
`dataset.json`, and `config.json`. The output directory was temporary and
removed after verification. This does not satisfy the required four-model
smoke test and is not research evidence.

Files touched in this update: `src/trusttab/experiments/benchmark.py`,
`src/trusttab/experiments/config.py`, `src/trusttab/experiments/run.py`,
`configs/experiments.yaml`, `tests/test_benchmark.py`,
`tests/test_core_matrix.py`, `docs/tabpfn.md`, `docs/datasets.md`, and this
progress log. Existing working-tree changes were preserved.

Verification in this update:

| Check | Command | Result |
|---|---|---|
| Focused benchmark/config/core-matrix tests | `.venv\Scripts\python.exe -m pytest tests\test_benchmark.py tests\test_core_matrix.py tests\test_config.py -q --basetemp .pytest-tmp-phase8-focus` | **45 passed, 1 existing scikit-learn warning** in 3.79s. |
| Full suite | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase8-full` | **322 passed, 1 warning** in 29.41s. Warning is the existing single-label confusion-matrix warning from the one-class ROC-AUC test. |
| Lint | `.venv\Scripts\ruff.exe check .` | Passed: all checks passed. |
| Formatting | `.venv\Scripts\ruff.exe format --check .` | Passed: 30 files already formatted. |
| CLI | `$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed; CLI help rendered. |
| Whitespace | `git diff --check` | Passed; Git emitted only existing LF-to-CRLF working-copy notices. |
| Real-data loader/report smoke | `run_robustness_benchmark(dataset_names=['german_credit','heart_disease'], label_noise_levels=[0.1], seeds=[0], model_names=['logistic_regression'], dataset_data_home={'heart_disease':'data/raw'}, calibration_settings={'enabled':True,'method':'sigmoid','train_size':0.70,'calibration_size':0.15,'test_size':0.15}, tracking_dir=<temporary directory>)` | **4 rows; both datasets; calibration succeeded on all rows; raw metrics, calibrated ECE/Brier, and expected report files verified.** No TabPFN run. |

Remaining blocker: user-side Prior Labs license/token and Hugging Face gated
repository authorization must be completed locally. Current booleans were
checked without displaying credential values and were all false; the checkpoint
is not cached. After authorization is available, rerun the bounded verifier
before running the four-model/two-dataset calibrated smoke. The full 200-row
matrix is still not launched. Phase 8 is not complete, package-release
preparation has not started, and no commit, push, merge, or publication was
performed.

Additional CLI note: direct execution of `.venv\Scripts\trusttab.exe --help`
was attempted, but Windows Application Control blocked that executable. The
module-based CLI help passed; the installed console-script executable itself
is therefore **NOT VERIFIED** in this environment and must be checked during
the later clean-install/package phase.

### Phase 8 update â€” authorization state and matrix preparation (2026-11-14)

The user reported creating a Prior Labs API key named `TABPFN_TOKEN`. I checked
only whether authentication was available; no credential value was read or
printed. In the process running this task, `TABPFN_TOKEN` was unavailable,
there was no Prior Labs token in TabPFN's local auth cache, Hugging Face auth
was unavailable, and the expected TabPFN-3.5 checkpoint file was absent. The
Prior Labs license acceptance state therefore could **not be verified** from
this process. The earlier actual checkpoint error and the installed 10.0.0
source show the HF repo check is separate from TabPFN's Prior Labs license
check. The next local action is to run `hf auth login` from
`.venv\phase8-tabpfn`, accept the gated checkpoint terms on the HF model page,
then run the package's Prior Labs interactive license flow (or make the new
`TABPFN_TOKEN` available only in the verifier process). Do not put credentials
in project files, command history, or chat. The expired key was not used.

The current full-matrix configuration was revalidated without running model
conditions: 200 conditions and 200 unique tuples across German Credit and
Cleveland, four configured models, noise `[0.0, 0.1, 0.2, 0.3, 0.4]`, seeds
`[0, 1, 2, 3, 4]`. Output is configured as `results/core_matrix_results.csv`;
`dataset.data_home` sends only Cleveland to `data/raw`. Calibration is disabled
for this core matrix, preserving the approved primary protocol. Tracking is
currently disabled, so a normal CLI run writes the configured results CSV but
does not automatically create the tracked statistical-report bundle; use the
tracked-run path or explicitly regenerate reports from the saved CSV before
calling those artifacts complete.

Resource inventory in the isolated environment: Python 3.11.9, TabPFN 10.0.0,
CPU PyTorch 2.15.0, 18 logical CPUs, approximately 25.2 GB physical RAM, and
no CUDA GPU. The 800-row German and 242-row Cleveland uncalibrated training
partitions are within TabPFN 10.0.0's CPU row guardrail. The full-matrix runtime
cannot yet be credibly estimated because the required real TabPFN fit and
prediction timings are unavailable; do not infer that estimate from the
classical-only smoke.

Verification commands in this update:

| Check | Command | Result |
|---|---|---|
| Credential availability (boolean only) | Isolated Python queried `bool(TABPFN_TOKEN)`, TabPFN cached-token availability, `bool(huggingface_hub.get_token())`, and existence of the expected cached checkpoint | All unavailable; no credential contents were accessed or emitted. License acceptance is **NOT VERIFIED** without Prior Labs authorization. |
| Matrix expansion and hardware | `.venv\phase8-tabpfn\Scripts\python.exe -` loading `configs/experiments.yaml`, expanding conditions, and reading the bounded verifier's hardware metadata | Exactly 200 unique conditions; CPU-only, 18 logical CPUs, 25,158,070,272 bytes RAM. No model fit. |
| Full tests | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase8-authcheck` | **322 passed, 1 existing scikit-learn warning** in 29.48s. |
| Ruff lint | `.venv\Scripts\ruff.exe check .` | Passed. |
| Ruff formatting | `.venv\Scripts\ruff.exe format --check .` | Passed: 30 files already formatted. |

Phase 8 remains blocked by local authorization. No checkpoint download, TabPFN
fit/prediction, four-model smoke, or full matrix occurred in this update. After
authorization: verify both gates and license acceptance without printing
secrets; run `scripts/verify_tabpfn.py --seed 123`; run the bounded four-model,
two-dataset, 10%-noise, calibrated smoke with tracking into a fresh temporary
directory; verify metrics and outputs; then estimate runtime from measured
fits and inspect result/provenance paths. Only after those checks pass should
the user authorize the full matrix. The reusable-library preparation phase
remains subsequent work; no package changes were made here.

### Phase 8 authorization recheck (2026-11-14)

The user reported that the Prior Labs API key is now configured as
`TABPFN_TOKEN`. A sanitized recheck in the isolated environment found:

| Authorization check | Result |
|---|---|
| `TABPFN_TOKEN` in the Python process environment | **Unavailable** (`False`) |
| Prior Labs token in TabPFN's local cache | **Unavailable** (`False`) |
| `TABPFN_TOKEN` name in the Windows user environment registry | **Absent**; only the variable name was checked, not its value |
| Hugging Face token available to `huggingface_hub` | **Unavailable** (`False`) |
| TabPFN 3.5 checkpoint in the isolated cache | **Absent** (`False`) |
| Prior Labs token validity/license acceptance | **NOT VERIFIED** because no token was available to the process |

No interactive login was started. The key appears to be configured in a
different PowerShell/session than the process available to this agent, or has
not been injected into the agent's parent process. Required local actions:

1. Accept the gated terms for `Prior-Labs/tabpfn_3_5` while logged into the
   intended Hugging Face account, then run
   `.venv\phase8-tabpfn\Scripts\hf.exe auth login` locally.
2. Make the Prior Labs `TABPFN_TOKEN` available to the process that runs the
   verification, without placing it in project files or chat. If it was set
   after the agent started, restart the agent from the authorized local shell
   or use the local secret-injection mechanism so this process inherits it.
3. Ensure the Prior Labs license for TabPFN 3.5 has been accepted for that
   account. Once both authorization states are available, rerun the
   credential-safe checks, then verify license acceptance.

The bounded TabPFN fit, checkpoint download, four-model smoke, and full matrix
were not attempted after this recheck because authorization is absent. No
source/configuration files were changed in this recheck; only this progress
entry was appended. No credentials were read, printed, logged, or changed.

### Phase 8 update â€” user reports Prior Labs key created; process check (2026-11-14)

The user reported that the Prior Labs API key named `TABPFN_TOKEN` is available
in the PowerShell environment running the agent. Direct checks in the current
PowerShell process and the isolated Python child both returned false for
`TABPFN_TOKEN`; the isolated environment also has no Prior Labs cached token,
no Hugging Face token, and no TabPFN-3.5 checkpoint. These checks emitted only
booleans and did not read or print credential values.

Because no Prior Labs token reached this process, the license authorization
status remains **NOT VERIFIED**. Because Hugging Face auth is absent, the
checkpoint's separate gated-repository requirement remains unmet. No
interactive login was started, no authorization endpoint was called, no
checkpoint download was attempted, and no TabPFN fit or four-model smoke was
run. The exact next actions are:

1. Make the already-created `TABPFN_TOKEN` available to the agent's child
   processes (for example, start/restart the agent session from a PowerShell
   process that has the variable set). Do not paste it into chat or save it in
   repository files.
2. Separately accept the gated checkpoint terms on the Hugging Face model page
   and authenticate this isolated environment with
   `.venv\phase8-tabpfn\Scripts\hf.exe auth login`. This command has not been
   launched.
3. After both states are visible, verify Prior Labs license acceptance without
   displaying tokens, download/cache the checkpoint, and run the bounded fit.

No source or configuration files changed in this update; only this progress
log was appended. Tests and lint were not rerun because no executable project
code changed; the previous update's 322-test and Ruff results remain the latest
code verification. No full matrix, commit, push, merge, publication, or release
occurred.

### Phase 8 recheck â€” agent process still lacks authorization (2026-11-14)

The user again reported that `TABPFN_TOKEN` is set in their PowerShell session.
The current isolated Python child still reports the environment variable as
unavailable. A name-only check of the Windows user environment also found no
persisted variable; no value was read. The separate Hugging Face credential is
also unavailable, the checkpoint is absent, and the Prior Labs token validity
and license-acceptance state remain **NOT VERIFIED**. Because there is no token
in this process, no authorization endpoint was called and no interactive
authentication or checkpoint request was started.

The installed TabPFN 10.0.0 source confirms the two independent paths:
`browser_auth.py` checks the Prior Labs token/license, while
`model_loading.py` calls `huggingface_hub.hf_hub_download`; the installed
`errors.py` recommends `hf auth login` for the gated repository. Official
requirements are documented in the [TabPFN 10.0.0 package guide](https://pypi.org/project/tabpfn/10.0.0/),
[Prior Labs model-weight guide](https://docs.priorlabs.ai/models/accessing-model-weights),
and [Hugging Face gated-model guide](https://huggingface.co/docs/hub/models-gated).

Required local handoff: in the PowerShell session where the token is already
available, check only `$null -ne $env:TABPFN_TOKEN`. Fully exit and relaunch
the IDE/agent host from that same session so its process tree inherits the
variable; setting a variable in an already-running PowerShell window does not
change the environment of an already-running agent. Separately, accept access
to `Prior-Labs/tabpfn_3_5` and run
`.venv\phase8-tabpfn\Scripts\hf.exe auth login` locally. This interactive
command was not launched by the agent. Once the restarted process exposes the
token and HF access, resume with token-safe validity/license checks, then the
bounded fit and smoke sequence. No code tests were rerun because the gate
prevents execution and no project source changed; `git diff --check` passed
with the repository's existing line-ending notices. The 200-condition matrix
remains unrun.

## Phase 8 â€” TabPFN authorization, model smoke, and pre-run readiness

**Status:** TabPFN access and the bounded four-model/two-dataset smoke are
**VERIFIED**. The full 200-condition matrix was intentionally not launched,
per the current instruction.

### Authorization and checkpoint

- In the isolated Python process, `TABPFN_TOKEN` itself was not present in the
  process environment. TabPFN did have a cached Prior Labs token, and
  `huggingface_hub.get_token()` returned an available token (presence only;
  neither value was inspected or printed). The user reports
  `ensure_license_accepted('tabpfn_3_5')` returned `True`.
- A separate local network verification call for the Prior Labs token returned
  `service_unreachable`; however the actual TabPFN fit subsequently completed,
  downloaded the gated checkpoint, and proceeded through the package's license
  and Hugging Face model-loading path. The user's explicit accepted-license
  result is retained as user-reported evidence; the fit/checkpoint download is
  independent execution evidence.
- Official TabPFN 10.0.0 guidance requires Prior Labs license acceptance and
  documents `TABPFN_TOKEN` for headless use. Installed `model_loading.py`
  downloads through `huggingface_hub`; installed `errors.py` specifically
  recommends `hf auth login` for a gated repository. Both authorization paths
  were available through local caches at fit time. No interactive login was
  started by the agent and no credential values were exposed.
- Downloaded default TabPFN-3.5 checkpoint:
  `.venv/phase8-tabpfn/tabpfn-cache/tabpfn-v3.5-20260909.safetensors`,
  876,027,932 bytes, SHA-256
  `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.
  It is cached only in the isolated environment and was not added to Git.

### Bounded real TabPFN fit

Command used (network access was authorized for this download; the command
prints only the verifier's JSON report, never credentials):

```powershell
$env:SKB_DATA_DIRECTORY = "D:\trusttab\.venv\phase8-tabpfn\skrub-data"
$env:TABPFN_MODEL_CACHE_DIR = "D:\trusttab\.venv\phase8-tabpfn\tabpfn-cache"
$env:TABPFN_NO_BROWSER = "1"
$env:PYTHONPATH = "src"
.\.venv\phase8-tabpfn\Scripts\python.exe scripts\verify_tabpfn.py --seed 123
```

The verifier completed successfully on cached OpenML German Credit (ID 31),
using the unchanged configured TabPFN CPU parameters and TrustTab preprocessing:
48 training rows, 16 prediction rows, binary classes `[0,1]`, probability
shape `[16,2]`, prediction shape `[16]`, finite probabilities in `[0,1]`, and
rows summing to one. Fit took 154.17 seconds; prediction took 6.18 seconds.
Environment: Python 3.11.9, NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1,
TabPFN 10.0.0, CPU PyTorch 2.15.0, Windows 10, Intel 18 logical CPUs,
25,158,070,272 bytes RAM, no CUDA GPU. The checkpoint ID and digest above
identify the actual loaded weights. No TabPFN warning was emitted in the
verifier report.

### Four-model, two-dataset calibrated smoke

Run ID: `20261114T135306495232Z-86b0bf2b44f1`.
Artifacts: `experiments/phase8_smoke_runs/20261114T135306495232Z-86b0bf2b44f1/`.
This new path was confirmed absent before creating the run. It uses core model
parameters unchanged, both complete cached datasets, one seed (0), 10% training
label noise, and the existing 70/15/15 Platt protocol. It generated 16 result
rows: 8 clean baseline rows plus 8 noisy-training rows (one for each
dataset/model pair). German training size was 700, calibration 150, test 150;
Cleveland was 212/45/46. Realized Bernoulli flips were 74 and 24 respectively.

All four models (Logistic Regression, Random Forest, XGBoost, TabPFN) ran on
both datasets. All 16 calibration statuses were `succeeded`. Accuracy,
balanced accuracy, macro F1, ROC-AUC, ECE, Brier, adaptive ECE, MCE, log loss,
and their calibrated counterparts were present and finite. The raw and
calibrated outputs use the same clean test partitions through the runner's
shared evaluation path. The smoke is software integration evidence only; its
one seed and one noise rate are not inferential results.

All expected artifacts exist, including `results.csv`, `report.md`,
`manifest.json`, `config.json`, `environment.json`, `dataset.json`, the six
Phase 7 statistical CSVs, summary/comparison CSVs, and generated figure PNGs.
The manifest says `completed`, with 16 rows. The artifact records config hash
`2d65c87f899d88a3d792021b5b0b00af1f7a1557ea8c52725c8657bbeed2f6dd` and
fingerprints for both datasets. Derived split/noise/model/calibration seeds
are present in every row, and `n_corrupted` is recorded. Total tracked smoke
duration was 865.02 seconds (14m25s). Disk free space after the run was
682,638,561,280 bytes (about 635.8 GiB).

The first post-run console-summary wrapper raised `TypeError` because it tried
to JSON-encode tuple keys in an ad hoc grouped-count dictionary. The benchmark
had already completed and written artifacts; direct artifact inspection then
verified the completed manifest, all 16 rows, metrics, and files. No model
fits were repeated to recover the summary.

### Matrix readiness and limits

- `configs/experiments.yaml` still expands to 200 unique conditions: 2 datasets
  Ã— 4 models Ã— 5 label-noise levels Ã— 5 seeds. No research setting was changed.
- Based on 16 model fits completing in 865.02 seconds, a simple linear estimate
  for 200 fits is about 3.0 hours. The main uncalibrated matrix trains on
  80% rather than the smoke's 70%, so actual CPU runtimeâ€”especially TabPFNâ€”may
  be higher. This is a planning estimate, not a measured full-run duration.
- The available 25.2 GB RAM and over 635 GiB disk are adequate for the observed
  smoke and current checkpoint. Resource use peaked around 4.8 GB for the
  smoke's Python process. The complete 200-condition matrix was **NOT RUN**.
- The benchmark currently writes final result CSVs only when the call ends;
  tracking does not checkpoint every condition and offers no resume capability.
  The raw result should therefore not be entrusted to an interruption-prone
  full run without first adding/validating condition-level checkpointing or
  scheduling an uninterrupted run. Core YAML tracking remains disabled; a
  normal CLI invocation would not produce the tracked report bundle unless
  tracking is enabled or reports are generated explicitly afterward.
- The existing tracked artifact schema and report generation passed on the
  smoke, but reproducibility was verified through seeds, hashes, dataset
  fingerprints, versions, and complete conditionsâ€”not by rerunning the same
  TabPFN smoke a second time.

### Verification after smoke

| Check | Command | Result |
|---|---|---|
| Full test suite | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-phase8-final` | **322 passed, 1 existing scikit-learn warning** in 41.79s. |
| Ruff lint | `.venv\Scripts\ruff.exe check .` | Passed. |
| Ruff format | `.venv\Scripts\ruff.exe format --check .` | Passed: 30 files already formatted. |
| Whitespace | `git diff --check` | Passed; Git printed existing LF-to-CRLF notices. |

Phase 8's gated-model and bounded-smoke acceptance checks are **VERIFIED**.
Research execution is not complete: the 200-condition experiment remains
unrun by instruction, no condition-level resume is implemented, and the full
research outputs/statistical conclusions do not yet exist. No commits, pushes,
merges, publication, or release occurred.

## Public Python library preparation (2026-11-14)

**Status:** Package structure, public API documentation, metadata, optional
dependency groups, source/wheel builds, and CI configuration are prepared.
Publication is **NOT READY**: the clean-environment native pandas import test
is blocked by Windows Application Control, the CI workflow has not run on
GitHub, and no TestPyPI rehearsal or publication was performed.

### Package changes

- Added curated `trusttab.api` convenience exports and tests that exercise its
  public entry points and confirm importing the API does not import XGBoost or
  TabPFN. The optional backends remain optional and TabPFN remains pinned to
  `tabpfn==10.0.0` in its extra.
- Fixed the condition-expansion import in `trusttab.api` after its new tests
  exposed that the helper is defined in `trusttab.experiments.config`.
- Corrected project metadata: runtime dependencies are in `[project]`, the
  license uses the PEP 639 SPDX string representation, and the obsolete MIT
  license classifier was removed. Raised the minimum scikit-learn version to
  1.6 because `evaluation/calibration.py` imports `sklearn.frozen.FrozenEstimator`,
  which is not available under the previously declared 1.3 minimum.
- Added README installation/API/dataset citation/license notes, `docs/api.md`,
  `docs/releasing.md`, `docs/tabpfn.md` updates, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `MANIFEST.in`, and an offline `examples/quickstart.py`.
  The TabPFN guide now states that the TabPFN-3.5 weights have a separate
  non-commercial/non-production license, are not distributed with TrustTab,
  and require users to obtain their own authorization. No checkpoint weights
  were included in either archive.
- Added `.github/workflows/ci.yml` for Python 3.11/3.12 lint, formatting,
  tests, CLI help, wheel/sdist builds and archive inspection, plus clean
  outside-checkout installation checks. The workflow configuration was
  inspected and parsed locally but has not run on GitHub Actions.

Files changed for this library-preparation milestone: `.github/workflows/ci.yml`,
`CHANGELOG.md`, `CONTRIBUTING.md`, `MANIFEST.in`, `README.md`, `docs/api.md`,
`docs/releasing.md`, `docs/tabpfn.md`, `examples/quickstart.py`, `pyproject.toml`,
`src/trusttab/api.py`, `tests/test_package.py`, and this progress file. Other
dirty files in the worktree belong to earlier research phases and were
preserved.

### Verification evidence

| Check | Command or evidence | Result |
|---|---|---|
| Package tests | `.venv\Scripts\python.exe -m pytest tests/test_package.py -q --basetemp .pytest-tmp-package-focused` | 3 passed. |
| Full suite | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-package-final` | 324 passed, 1 existing scikit-learn one-label confusion-matrix warning, 34.01s. |
| Ruff lint | `.venv\Scripts\ruff.exe check .` | Passed. |
| Ruff format | `.venv\Scripts\ruff.exe format --check .` | Passed: 32 files already formatted; repository retains the existing formatter exclusions. |
| CLI help | `.venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed. |
| Wheel and sdist | `.venv\Scripts\python.exe -m build --sdist --wheel --outdir <unique %TEMP% directory>` | Final build succeeded with isolated setuptools 84.2.1: `trusttab-0.1.0-py3-none-any.whl` (75,128 bytes) and `trusttab-0.1.0.tar.gz` (119,068 bytes), under `%TEMP%\trusttab-final-build-a5a2ae5a6cb849ba85e49239a267dbdd`. |
| Archive inspection | Python `zipfile`/`tarfile` checks against both artifacts | Public API, package modules, runtime dependencies, optional extras, CLI entry point, LICENSE, API docs, example, and sdist contents present; no `.safetensors` weights or `.venv` content. TabPFN is only the pinned optional extra. |
| Installed wheel | Fresh Python 3.11 temporary environment and `pip install <wheel>` | Dependency resolution/install completed. Import in that environment failed because Windows Application Control blocked pandas' `timedeltas` native DLL, both from Temp and a newly created workspace environment. This is **NOT VERIFIED** as a clean functional install on this machine. |
| Installed wheel fallback | From outside the repository, installed wheel imported using core dependencies from the pre-existing trusted `.venv` via `PYTHONPATH`; `trusttab.__file__` resolved to the temporary installed wheel | Public imports, `python -m trusttab.experiments.run --help`, and `examples/quickstart.py` passed. This checks wheel contents and code path, but is not a clean dependency-environment verification. |
| Console executable | `<fresh-env>\Scripts\trusttab.exe --help` | Blocked when pandas native dependency loaded under the local Windows Application Control policy. The module CLI invocation above works. |
| Whitespace | `git diff --check` | Passed; Git emitted only existing LF-to-CRLF notices. |

The first in-sandbox isolated build could not fetch `setuptools>=77` because
network access was denied; the standard build was then rerun with authorized
network access and succeeded. The default machine Python was 3.14 and outside
TrustTab's declared support range, so clean-install checks explicitly used
Python 3.11. No project `.venv` was changed. The progress entry and generated
wheel/sdist are not a PyPI publication or TestPyPI rehearsal.

The build backend also created ignored local scratch paths under `build/` and
`src/trusttab.egg-info/`; the failed fresh-install check created
`.package-verify-4248394bedca44beabddcd84f157b135/`. Cleanup of these paths was
blocked by automatic review because the ignored build/egg-info paths could
contain prior user artifacts. They were left untouched. The pre-existing
`build/clean-install` directory and existing pytest scratch directories were
not changed.

### Remaining release and research limitations

1. Repeat clean wheel and sdist installation on a Python 3.11/3.12 host where
   the native dependency DLLs can load; run `trusttab --help` and the offline
   example from that installed environment.
2. Run the new CI workflow on GitHub and verify both supported Python jobs.
3. Run `twine check` and a TestPyPI rehearsal after owner approval. Do not
   publish to TestPyPI or PyPI as part of this work.
4. The complete 200-condition experiment and research conclusions remain
   outstanding; the two-dataset smoke is integration evidence only. Address
   condition-level resumability before scheduling an interruption-prone full
   run.
5. No commit, push, merge, package upload, release, or repository-setting
   change was made.

## Finalization phase update (2026-11-14)

### Task 1 â€” Clean installation and Windows DLL diagnosis

**Status: VERIFIED in a fresh Python 3.11 environment with the policy-approved
core dependency versions.** Windows Application Control event 3077 records the
blocked pandas 3.1.1 `timedeltas.cp311-win_amd64.pyd` and says it did not meet
the Enterprise signing-level requirement or violated code-integrity policy.
The same extension is unsigned in both distributions inspected. The installed
pandas 3.1.0 extension imports successfully in the existing project and
Phase 8 environments; a fresh environment with pandas 3.1.0 also imported it.
Therefore this failure is an environment signing-policy restriction on the
3.1.1 binary, not a TrustTab import or declared-dependency error. No security
policy was weakened and no project dependency bound was added to conceal the
restriction.

Fresh-install verification used Python 3.11.9 and the current built wheel in a
new environment under `%TEMP%`, with these compatible runtime versions:
NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2, PyYAML
6.0.3, and SciPy 1.17.1. The installed package loaded from that environment,
`trusttab --help` succeeded, and `examples/quickstart.py` completed when run
from outside the repository. The exact environment was
`%TEMP%\\trusttab-clean-pandas310-05239e558aa34766a71b914bccf34db9`.
This is a fresh install with explicit reproducible dependency versions; CI
will separately exercise the supported dependency ranges on Ubuntu.

The dependency installation command used was `python -m pip install
<wheel-path> numpy==2.4.6 pandas==3.1.0 scikit-learn==1.9.1
matplotlib==3.11.2 PyYAML==6.0.3 scipy==1.17.1`. From `%TEMP%`, the installed
environment then ran the public import check, `trusttab --help`, and the
absolute-path quick-start example.

### Task 2 â€” CI workflow

**Status: WORKFLOW VALIDATED LOCALLY; NOT RUN ON GITHUB.** The workflow covers
core dependencies on Python 3.11 and 3.12 and has a corresponding XGBoost
optional-extra job on both versions. The build job builds wheel and sdist,
runs `twine check`, checks archive contents, and installs each distribution
outside the checkout. TabPFN remains optional and is not installed by ordinary
CI because its real gated fit is separately verified in Phase 8. Official
GitHub action documentation confirms `actions/checkout@v7` and
`actions/setup-python@v7` are current supported tags.

Local checks: workflow YAML parsed; the exact core matrix expands to 200 unique
conditions; focused optional-model coverage passed (`30 passed, 1 existing
single-class scikit-learn warning`). The installed `gh` client reported no
available authentication. In addition, this workflow file is still local and
uncommitted; running it on GitHub requires an approved push/PR. No push was
made. Python 3.12 was not available locally, so the 3.12 job remains pending
remote CI evidence.

### Task 3 â€” PyPI artifact checks and rehearsal blocker

The final artifacts were rebuilt from the current package source into a new
temporary directory, preserving the pre-existing `dist/` wheel and sdist:

- `%TEMP%\\trusttab-final-build-a5a2ae5a6cb849ba85e49239a267dbdd\\trusttab-0.1.0-py3-none-any.whl` (75,128 bytes)
- `%TEMP%\\trusttab-final-build-a5a2ae5a6cb849ba85e49239a267dbdd\\trusttab-0.1.0.tar.gz` (119,068 bytes)

`python -m build --sdist --wheel --outdir <temporary-directory>` succeeded
with isolated setuptools 84.2.1. `python -m twine check` passed for both
artifacts (Twine 7.0.0). Archive inspection scanned 128 files and found no
credential-pattern matches, local datasets, `.safetensors`/other model
weights, temporary run outputs, `.venv` content, or local cache files. Both
archives include the package, public API, license metadata, docs, CLI entry
point, and source example. The wheel contains the six core dependencies and
optional `xgboost`, `tabpfn==10.0.0`, and `dev` extras.

Live index lookups returned HTTP 404 for both TestPyPI and production PyPI
project metadata for `trusttab` at check time. No local Twine credentials or
`.pypirc` file were available; values were never read or printed. A TestPyPI
upload/rehearsal is therefore **NOT VERIFIED** and was not attempted
interactively. To continue, configure a TestPyPI token locally for the
publishing process (`TWINE_USERNAME=__token__`, `TWINE_PASSWORD` set from the
local secret store), then run `python -m twine upload --repository testpypi`
against these exact artifacts. Do not send the token in chat. Production PyPI
was not contacted for upload and no release occurred.

### Task 4 â€” Core experiment readiness and resume plan

The current core configuration still has exactly 200 unique noise conditions:
2 datasets x 4 models x 5 noise levels x 5 seeds. It preserves German Credit
and UCI Cleveland, Logistic Regression/Random Forest/XGBoost/TabPFN, rates
0/10/20/30/40%, and seeds 0-4. No experimental factor was changed.

The unified runner also writes one clean-baseline row for each dataset/model/
seed. The 40 zero-noise rows reuse the corresponding clean fit, so execution
requires 200 actual model fits, while the combined CSV contains 240 rows
(200 label-noise conditions plus 40 baseline reference rows). The current
runner and report schemas distinguish these via `experiment_type` and
`condition_id`.

Tracking is now explicitly enabled in `configs/experiments.yaml`; this is an
output/provenance setting only. Each run records a unique run directory,
normalized/original configuration and hash, start/completion timestamps,
environment/dependency versions, Git commit and dirty-tree flag, dataset
fingerprints, result CSV, and generated report. Result rows record condition,
base/derived split/noise/model/calibration seeds, model parameters, sample
counts, realized corruptions, calibration status, and raw/calibrated metrics.
The output `results/core_matrix_results.csv` did not exist at verification
time; `experiments/runs/` already contains prior runs, but tracking allocates
new unique run-ID folders rather than overwriting them. Existing output files
and smoke artifacts were preserved.

Core calibration remains disabled as specified; the separate calibration
rescue config uses the existing stratified 70/15/15 protocol. Phase 8's real
four-model/two-dataset smoke recorded successful calibration for all 16 rows,
and the report generator emitted the expected statistical CSVs, report, and
figures from actual smoke outputs. This is integration evidence, not research
evidence.

**Important limitation:** The runner writes its result CSV only after the
whole invocation completes. Tracking marks an interrupted invocation failed
but does not checkpoint completed conditions, and result rows do not contain
per-condition fit duration. Do not treat tracking alone as resumability.

Proposed resume-preserving execution plan (not run): retain an immutable copy
and SHA-256 of the master YAML; construct 40 bounded sequential chunks, one
dataset x model x seed per chunk, each containing all five approved noise
levels; give each chunk a unique output filename and tracking run ID; preserve
all run folders. On restart, skip only chunks whose manifest is completed and
whose `condition_id` set matches the five expected rows; rerun failed or
missing chunks without altering rates, seeds, preprocessing, or model
parameters. Merge raw rows only after asserting the exact 200 unique
label-noise conditions and 40 baseline references, then regenerate the
statistical report and figures from the merged raw results. The chunk-level
failure exposure is at most five noise conditions, not a partial-row resume.
Record the master-config hash and chunk-to-run-ID map in a separate run index.

Using the 16-fit, 865-second smoke as a simple linear extrapolation gives
about 3.0 hours for 200 fits; larger 80% training splits and CPU TabPFN may
make the actual run longer. A practical planning allowance is 3-4 hours, not
a measured guarantee. The available machine has 18 logical CPUs, about 25.2
GB RAM, no CUDA GPU, and over 635 GiB free disk. The verified TabPFN-3.5
checkpoint is cached (876,027,932 bytes); German and Cleveland cached sources
were used by the successful smoke. The output volume is expected to be small
relative to available disk, but the full report has not been generated.

### Task 5 â€” Research output readiness

The existing generator and Phase 7 statistics successfully consumed the saved
smoke CSV and produced Friedman, Nemenyi, Wilcoxon calibration, degradation
slope, effect-size, summary, comparison, Markdown, and figure outputs. The
five-seed full-run outputs and inferential conclusions do not yet exist.
Publication-quality final interpretation is therefore **NOT VERIFIED** until
the complete experiment is run, completeness is checked, and the reports are
regenerated from all raw conditions. No statistical claim is made from the
one-seed smoke.

### Verification in this finalization update

| Check | Command | Result |
|---|---|---|
| Full suite | `.venv\\Scripts\\python.exe -m pytest -q --basetemp .pytest-tmp-finalization` | 324 passed, 1 existing single-class scikit-learn warning, 29.30s. |
| Optional XGBoost path | `.venv\\Scripts\\python.exe -m pytest -q tests/test_models.py tests/test_benchmark.py --basetemp .pytest-tmp-xgboost-final` | 30 passed, 1 same warning. |
| Ruff | `.venv\\Scripts\\ruff.exe check .` | Passed. |
| Format | `.venv\\Scripts\\ruff.exe format --check .` | Passed: 32 files already formatted. |
| CLI | `.venv\\Scripts\\python.exe -m trusttab.experiments.run --help` | Passed. |
| Matrix and workflow validation | Load YAML, expand conditions, check tracking and CI Python matrices | YAML valid; exactly 200 unique conditions; tracking enabled; core and XGBoost jobs both list Python 3.11/3.12. |
| Artifact metadata | `python -m build --sdist --wheel`, then `python -m twine check` | Wheel and sdist built; both Twine checks passed. |
| Artifact safety | Inspect wheel and sdist members and text | 128 files scanned; zero forbidden/credential-pattern findings. |
| Git whitespace | `git diff --check` | Passed with existing LF-to-CRLF notices. |

### Changed files in this finalization update

- `.github/workflows/ci.yml` â€” add the XGBoost extra matrix, Twine check, and
  archive exclusion checks.
- `configs/experiments.yaml` â€” enable tracked execution and report artifacts;
  matrix factors are unchanged.
- `tests/test_core_matrix.py` â€” assert core tracking is enabled while retaining
  exact 200-condition fake-runner verification.
- `IMPLEMENTATION_PROGRESS.md` â€” this record.

The current tree remains dirty from prior phases. No existing `dist/`, run,
dataset, checkpoint, or scratch artifact was overwritten or deleted. No full
200-condition experiment, GitHub workflow run, TestPyPI upload, production
upload, commit, push, merge, or release was performed. The remaining gates are
GitHub execution after an approved push/PR, TestPyPI credentials for an
authorized rehearsal, and explicit user approval before the research run.

## Final execution phase â€” protocol audit and resumable scheduler (2026-11-14)

**Status:** Research protocol audit completed; condition-level checkpointing
and resume support implemented and tested. The full 200-condition matrix was
**NOT LAUNCHED**, pending the user's explicit approval required by this phase.

### Protocol audit and deviations

- `configs/experiments.yaml` expands to exactly 200 distinct conditions:
  German Credit and Cleveland Ã— Logistic Regression, Random Forest, XGBoost,
  and TabPFN Ã— 0/10/20/30/40% noise Ã— seeds 0â€“4. Split, training-noise, and
  model random states are separately derived. Bernoulli noise is applied to
  training labels after splitting; test labels remain clean. Core calibration
  remains disabled; the separate rescue experiment uses the approved clean
  70/15/15 partitions.
- The benchmark records Accuracy, Balanced Accuracy, Macro-F1, ROC-AUC, ECE,
  and Brier Score, plus supporting metrics. Calibration result columns are
  retained when calibration is enabled. The reporting path integrates the
  Phase 7 Friedman/Nemenyi/Wilcoxon analyses, effect sizes, confidence
  intervals, and figures, but no full-run statistical outputs exist yet.
- The blueprint has an early executive-summary reference to 0â€“30% noise, but
  its formal Part F, Part G, and Part Z define the 0â€“40% five-level matrix.
  The accepted Part Z matrix is unchanged. Part E/Part Z describe Cleveland as
  approximately 297 rows after dropping missing observations; the user's
  explicit approved benchmark is the processed UCI Cleveland subset with 303
  rows and missing feature values preserved for train-fitted preprocessing.
  The loader follows the approved 303-row protocol. Part F's pseudocode was
  updated in Phase 5 to agree with its formal independent Bernoulli definition
  and the user's approval. These are documented source/protocol differences,
  not silent changes in this phase.
- The blueprint estimates well under two hours, while the measured 16-fit
  calibrated smoke took 865.02 seconds. A linear estimate is about 3.0 hours;
  the core's larger 80% training partitions make **3â€“4 hours a planning range,
  not a guarantee**. No research setting was changed to reduce runtime.
- The older progress entry documented a Cleveland `data_home` integration
  workaround. The current benchmark forwards `dataset_data_home` to registered
  loaders (and the core config maps Heart Disease to `data/raw`); this source
  inspection supersedes that old warning. The Phase 8 smoke used the then-
  current integration and successfully loaded both datasets.

### Scheduler implementation

- Added `run_resumable_experiment(config, run_dir)` and CLI option
  `--resume-dir`. It executes sequential dataset/model/seed chunks and passes
  the existing benchmark's model parameters, split behavior, and seed streams
  through unchanged.
- The benchmark invokes a result callback when each baseline or label-noise
  row is completed. The scheduler atomically replaces its raw CSV and JSON
  manifest after each callback. The manifest stores normalized configuration
  and hash, expected condition IDs, dataset fingerprints, Python/platform/
  package versions, attempt/status records, and failures. Rows include run ID,
  completion timestamp, and model fit seconds.
- An OS-managed process lock prevents concurrent writers and is released by
  the OS on process termination. Completed label-noise conditions are
  reconstructed from the checkpoint and skipped on resume. Configuration,
  dataset fingerprint, Python, and package-version changes block resume.
- Exceptions are explicit: the first unresolved condition in an aborted chunk
  is marked failed with its error; later unattempted conditions stay pending
  and carry a `last_error` explanation. Other chunks continue. Retrying uses
  the same config and seeds. Incomplete runs return an integrity report and
  the CLI exits nonzero; a run is complete only when all expected condition
  IDs exist exactly once and no unexpected or duplicate IDs are present.
- A new run directory must be empty. A resumed directory must contain this
  scheduler's manifest/checkpoint. The scheduler never targets or overwrites
  the ordinary configured `results/core_matrix_results.csv` path.
- The scheduler emits 200 label-noise result rows plus the benchmark's 40
  clean-baseline reference rows (240 total), with each 0% noise condition
  continuing to reuse the clean baseline model as before. Statistical tables
  and figures are regenerated from actual raw results only after completion;
  this phase did not generate or imply research findings from synthetic
  fixtures.

### Changed files in this phase

- `src/trusttab/experiments/resumable.py` â€” atomic run ledger, locking,
  checkpoint, resume, and integrity validation.
- `src/trusttab/experiments/benchmark.py` â€” condition-completion callback and
  training-duration recording; existing execution path and defaults retained.
- `src/trusttab/experiments/run.py` â€” `--resume-dir` CLI routing.
- `src/trusttab/experiments/__init__.py` â€” public scheduler exports.
- `tests/test_resumable.py` â€” interruption, resume, retry, failure, duplicate,
  changed-config, and actual benchmark callback tests.
- `tests/test_benchmark.py` â€” validates fit-time values separately from
  deterministic metrics and metadata.
- `docs/unified-benchmarking.md` â€” resume command and checkpoint contract.
- `IMPLEMENTATION_PROGRESS.md` â€” this audit and verification record.

### Verification in this phase

| Check | Command | Result |
|---|---|---|
| Focused scheduler/runner | `.venv\Scripts\python.exe -m pytest tests\test_resumable.py tests\test_benchmark.py tests\test_core_matrix.py -q --basetemp .pytest-tmp-resumable-focused3` | 22 passed, 1 existing single-class sklearn warning. |
| Scheduler interruption/retry | `.venv\Scripts\python.exe -m pytest tests\test_resumable.py -q --basetemp .pytest-tmp-resumable-lock` | 5 passed. |
| Full suite | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-final-execution` | 329 passed, 1 existing sklearn warning, 49.29s. |
| Ruff lint | `.venv\Scripts\ruff.exe check .` | Passed. |
| Ruff format | `.venv\Scripts\ruff.exe format --check .` | Passed after formatting the three changed Python files; 34 files already formatted. |
| CLI | `.venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed; displays `--resume-dir`. |
| Matrix validation | Load `configs/experiments.yaml`, call `expand_experiment_conditions`, compare length and unique tuple count | Exactly 200 unique conditions with approved datasets/models/noise/seeds. |
| Disk/output preflight | `shutil.disk_usage('D:/')`; `Test-Path experiments/runs/trusttab-core-20261114` | 635.3 GiB free; proposed dedicated output path absent. 18 logical CPUs. Existing environment inventory reports 25.2 GB RAM, CPU-only TabPFN. |
| Whitespace | `git diff --check` | Passed; only existing LF-to-CRLF notices. |

### Remaining gates and launch preparation

- The bounded real TabPFN fit, checkpoint cache, and four-model/two-dataset
  calibrated smoke are **VERIFIED by the Phase 8 record**. They were not
  repeated here; credentials were not read or emitted.
- The full matrix remains unrun. Before launch, report and confirm the 200
  conditions, estimated 3â€“4 hours, 635.3 GiB available disk, current checkpoint
  cache, and resume procedure. Recommended new output path:
  `experiments/runs/trusttab-core-20261114` (confirmed absent at this check).
  Start only after explicit user approval:

  ```powershell
  .\.venv\Scripts\python.exe -m trusttab.experiments.run `
    --config configs/experiments.yaml `
    --resume-dir experiments/runs/trusttab-core-20261114
  ```

  On interruption, rerun the identical command. Do not change the config or
  environment when resuming; the scheduler refuses those changes.
- After completion, verify the manifest reports `completed`, all 200 expected
  condition IDs exactly once, 40 unique baseline rows, and no failures; then
  regenerate aggregate/statistical/figure outputs from that raw CSV. No
  inference is claimed before this occurs.
- Prior package checks verified wheel/sdist metadata, archive contents, and an
  installed-wheel workflow; production PyPI was not published. GitHub Actions
  were not run because the updated workflow is local/unpushed and GitHub
  authorization is unavailable. TestPyPI rehearsal remains blocked pending
  local TestPyPI credentials/authorization. No credentials are stored by the
  scheduler.
- Current scheduler-inclusive distribution build: the isolated `python -m
  build` attempt was blocked while fetching `setuptools>=77` (`WinError 10013`
  network restriction). `--no-isolation` with the project venv's setuptools
  65.5.0 correctly rejected PEP 621 license metadata because it is below the
  declared backend requirement. Reusing already-installed setuptools 78.1.0
  from the isolated Phase 8 environment through a process-local `PYTHONPATH`,
  `python -m build --no-isolation --sdist --wheel` succeeded. Current artifacts
  are in `%TEMP%\trusttab-final-build-noiso-7735d648e3134f7ab0b7de6127584eb4`:
  wheel 79,674 bytes and sdist 124,793 bytes. Both archives contain
  `experiments/resumable.py`; 42 wheel and 105 sdist members were inspected,
  with no checkpoint, cache, raw-data, or token-named member. The wheel
  installed with `pip install --no-deps --target ...` and, from `%TEMP%`,
  imported `trusttab` and `run_resumable_experiment` and passed CLI `--help`.
  This import used the already-installed core dependencies; it is not a fresh
  dependency-resolution test.
- Current `twine check` was **NOT VERIFIED**: Twine is absent from both local
  Python environments and the isolated network cannot install it. An earlier
  Twine check covered the pre-scheduler artifacts only. Do not treat that as a
  check of the current distributions. The build also refreshed ignored local
  `build/lib`, `build/bdist.win-amd64`, and `src/trusttab.egg-info` generated
  metadata; they were not deleted. Earlier scratch files/directories remain.
- No matrix run, TestPyPI/PyPI upload, commit, push, merge, or release occurred.

## Final research execution â€” 2026-11-14

### Execution

- Explicitly authorized core matrix was executed with the resumable scheduler, unchanged `configs/experiments.yaml`, and isolated `.venv\phase8-tabpfn` environment. No optional matrix or calibration matrix was launched.
- Command (environment variables set in the invoking PowerShell process):

  ```powershell
  $env:PYTHONPATH = 'src'
  $env:SKB_DATA_DIRECTORY = 'D:\trusttab\.venv\phase8-tabpfn\skrub-data'
  $env:TABPFN_MODEL_CACHE_DIR = 'D:\trusttab\.venv\phase8-tabpfn\tabpfn-cache'
  $env:TABPFN_NO_BROWSER = '1'
  .\.venv\phase8-tabpfn\Scripts\python.exe -m trusttab.experiments.run `
    --config configs/experiments.yaml `
    --resume-dir experiments/runs/trusttab-core-20261114
  ```

- Run ID: `trusttab-e18ead6a25c341549ba41ce12d7f5ef8`; manifest status `completed`; manifest integrity: 200 expected, 200 completed, zero failed, missing, unexpected, or duplicate conditions. Manifest interval: 1h 07m 20.970s.
- Verified exact 200 unique label-noise conditions (2 datasets Ã— 4 models Ã— 5 noise levels Ã— 5 seeds), plus exactly 40 unique clean-baseline reference rows; total 240 rows, all condition IDs unique.
- Configuration hash: `f1f8d9f3bac2bdc614f6d956c96da839bc9453e04e3a61198862d4b4598afeb3`.
- Dataset fingerprints: German Credit `be2da28ee0a92114ae07549c876cb414e1dd2ead0a56fa9e19dc1535c4ac6fa6`; Cleveland `d9ed8e3110da6747811725a53fb068be2bd233684c011b44563bc93e6517dd27`.
- All 200 corruption counts were replayed from the stored split/noise seed protocol against cached dataset labels and matched. Required raw metrics (accuracy, balanced accuracy, macro-F1, ROC-AUC, ECE, Brier) are finite in all 240 rows. Required run, seed, condition, parameter, duration and timestamp metadata are populated. Test sizes are 200 German Credit and 61 Cleveland.
- Core configuration has calibration disabled. All 200 noise conditions report `disabled`; calibrated metrics remain missing as intended. No calibration conclusion is drawn from this core matrix. The runner's inspected path splits before noise, corrupts only a copy of `y_train`, and evaluates against the unchanged `y_test`.
- Package versions recorded in manifest: TrustTab 0.1.0, Python 3.11, NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.17.1, XGBoost 3.2.0, TabPFN 10.0.0, PyTorch 2.15.0+cpu. CPU-only TabPFN was used; checkpoint identity/cache evidence is recorded in the prior Phase 8 entry and local isolated cache (weights are not copied into run outputs).

### Generated results

From the saved raw CSV, `generate_robustness_report` produced 27 declared artifacts under `experiments/runs/trusttab-core-20261114/final_report/`. A separate `research_findings.md` was then written from those observed outputs, for 28 files total. The generated report artifacts include:

- `report.md`, summary/model-comparison/degradation CSVs;
- Friedman, Nemenyi, calibration-Wilcoxon, degradation-slope, effect-size and statistical-summary CSVs;
- model comparison, robustness, calibration and applicable critical-difference PNGs;
- `research_findings.md`, a cautious RQ0â€“RQ4 summary computed from the observed matrix.

RQ4 remains outstanding: this core matrix did not enable calibration. The report does not treat the earlier bounded calibration smoke as inferential evidence. The generated feature-shift figure is an empty/not-applicable figure because no feature-shift conditions were in this core protocol; it must not be presented as an experiment result.

Descriptively, mean performance and Brier Score deteriorated from 0% to high noise on both fixed datasets; ECE trajectories differed (Cleveland ECE peaked at 30%, then fell at 40%). The seed-blocked slope-contrast families did not reject after Holm correction. Some within-noise Friedman model comparisons remained significant after planned correction; these are model comparisons at fixed noise, not tests of noise effects. Five seeds and two fixed datasets limit inference; non-significance is not evidence of equivalence.

### Final verification

| Check | Command | Result |
|---|---|---|
| Full tests | `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-postrun-final` | 329 passed; one existing sklearn single-class warning; 62.06s |
| Ruff | `.venv\Scripts\ruff.exe check .` | Passed |
| Formatting | `.venv\Scripts\ruff.exe format --check .` | Passed; 34 files already formatted |
| CLI | `PYTHONPATH=src .venv\Scripts\python.exe -m trusttab.experiments.run --help` | Passed; resume and tracking options displayed |
| Diff whitespace | `git diff --check` | Passed; Git emitted LF-to-CRLF working-copy notices only |
| Raw result integrity | custom validation against manifest + config expansion | 200/200 conditions, 40/40 baseline references, no duplicate/missing/unexpected rows, finite required metrics |
| Noise replay | cached loader, train/test split and derived Bernoulli noise seeds | 200/200 per-condition counts matched |
| Report generation | isolated Python, report generator over saved raw CSV | All 27 declared report artifacts exist |

Previous scheduler-inclusive wheel/sdist build and outside-checkout wheel import evidence remain as documented above. Current `twine check` remains **NOT VERIFIED** because Twine is unavailable in the local environments and network installation is restricted. GitHub Actions was not run because workflow changes are local/unpushed and no GitHub execution authorization/access is available. TestPyPI rehearsal was not performed because no TestPyPI credentials were configured; no upload occurred.

### Files added/updated by final execution

- `experiments/runs/trusttab-core-20261114/run_manifest.json` â€” resumable manifest and integrity record.
- `experiments/runs/trusttab-core-20261114/raw_results.csv` â€” 240 observed result rows.
- `experiments/runs/trusttab-core-20261114/final_report/*` â€” generated tables, figures and report from raw results.
- `IMPLEMENTATION_PROGRESS.md` â€” this execution and verification record.

No source code, research configuration, pre-existing output, cache, or `.venv` was modified for the experiment. Existing dirty worktree content was preserved. No commit, push, merge, package upload, publication, or release occurred.

### Release and final checklist

- Research core matrix and raw evidence: **COMPLETE / VERIFIED**.
- Matrix completeness, duplicate prevention, saved noise counts and finite core metrics: **VERIFIED**.
- Statistical/report outputs regenerated from actual raw rows: **VERIFIED**; limited by five seeds and two datasets.
- Platt calibration RQ4 on the full matrix: **OUTSTANDING**; core calibration intentionally disabled, and no calibrated inference is available.
- Research findings report: **COMPLETE**, with RQ4 explicitly unanswered and limitations stated.
- Full test suite, Ruff, format, CLI help, and `git diff --check`: **PASS**.
- Build and outside-checkout installed-wheel verification: **previously verified for scheduler-inclusive package build**; no package source change followed that build.
- Current `twine check`: **NOT VERIFIED** (Twine unavailable offline).
- TestPyPI rehearsal: **NOT PERFORMED** (credentials absent); production PyPI publication: **NOT AUTHORIZED / NOT PERFORMED**.
- GitHub Actions execution: **NOT VERIFIED** (workflow local/unpushed; no remote CI run).
- Commits, pushes, merges, publishing and releases: **NOT PERFORMED**.


## Final release audit — 2026-11-14

- Independently expanded the configured matrix and compared it to `raw_results.csv`: 200/200 unique core conditions, 40/40 unique baseline references, 240 unique result IDs, no missing/unexpected/duplicate/failed rows. Recomputed both dataset fingerprints and matched the manifest. Required metrics are finite; all saved noise counts were replay-checked against the split/noise seeds.
- Re-generated the report in a separate temporary directory. All 13 statistical/descriptive CSVs, `report.md`, and 13 PNGs were byte-identical to the saved outputs. RQ4 remains unanswered because the core config disabled calibration.
- Research caveats identified: per-observation predictions are not stored, so final metrics cannot be independently recomputed from raw CSV alone; generated figures do not cover the full blueprint plan; generic supplementary `statistical_comparisons.csv` uses paired t-tests not listed as primary Part M tests; descriptive 95% intervals use Student-t rather than the blueprint's bootstrap or mean ± SD options. These are documented in `TRUSTTAB_FINAL_RELEASE_AUDIT.md` and must be considered before final academic claims.
- Package metadata inspected: 0.1.0, Python >=3.11,<3.13, MIT, setuptools>=77, core scientific dependencies, optional XGBoost>=2.0 and TabPFN==10.0.0 extras. The public API imports without loading optional models. No dataset, checkpoint, cache, credential, or result artifact was found in the distributions.
- Final artifacts were built with the installed setuptools 78.1 from the isolated environment available on `PYTHONPATH`; wheel 79,885 bytes and sdist 125,561 bytes are in `dist/release-candidate-final-2026-11-14/`. `twine check` passed on both. Existing `dist` files were not overwritten.
- A wheel-target import/API/CLI-module/quick-start check previously passed outside the checkout using the project's already-installed core dependencies. The final candidate differs only in documentation metadata. A genuinely clean dependency-resolving Python 3.11 install is **NOT VERIFIED**: package index network access failed with Windows WinError 10013. The current environment has Python 3.11.9 in `.venv`, global Python 3.14.6, and no Python 3.12 command. Windows Application Control blocks generated executable shims; no policy was changed.
- Full suite: 329 passed, one existing sklearn warning. Ruff lint/format and diff check pass. GitHub Actions not run; no push/commit was made. No PyPI/TestPyPI credentials were configured (presence checked without reading values), no upload occurred, and `trusttab` name availability could not be verified due network restrictions.
- Current Prior Labs license text says the checkpoint license has professional-purpose eligibility conditions and limits model outputs to non-commercial use; README and `docs/tabpfn.md` now point out these restrictions and the package contains no weights. Source: [official Prior Labs checkpoint license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE).

### Changed files/artifacts in this audit

- `README.md` — clarify model license/output restrictions and repository-only YAML templates.
- `docs/tabpfn.md` — detail current eligibility and output restrictions.
- `TRUSTTAB_FINAL_RELEASE_AUDIT.md` — full research/package audit and action list.
- `IMPLEMENTATION_PROGRESS.md` — this audit entry.
- `dist/release-candidate-final-2026-11-14/` — verified wheel and sdist, without overwriting earlier artifacts.

The dirty worktree and previous results/caches/scratch artifacts were preserved. No commit, push, merge, upload, publication, or release occurred.

## Final research-report and package revalidation - 2026-11-14

### Research reporting corrections

- Reviewed the latest full statistical outputs against Part M of the blueprint. The previously noted Student-t descriptive confidence intervals were replaced with deterministic percentile bootstrap intervals over the five seed-level units (10,000 resamples). The resampling seed is derived from the result keys and is recorded in output. Bootstrap is used for descriptive intervals only; hypothesis tests were not changed.
- The supplementary paired t-test compatibility table is explicitly labeled exploratory. The preplanned seed-blocked Friedman/Nemenyi analyses remain the primary model comparisons; existing Wilcoxon calibration analyses remain unavailable for this core run because calibration was disabled.
- Added dedicated noise curves for ECE, Brier score, and Macro-F1, plus a model-ranking heatmap and cross-dataset panel. The generated report includes metric curves and eligible critical-difference diagrams. Reliability diagrams and confidence histograms remain unavailable because saved run rows do not retain per-observation labels/probabilities. No raw-versus-calibrated figure is generated for an uncalibrated experiment.
- Fixed a report summary defect: it had called a model "best calibrated" by ranking raw ECE/Brier when no calibrated metrics existed. It now reports `unavailable` unless actual calibrated reliability metrics are present. This preserves RQ4 as unanswered.
- Generated a new, separate report directory, `experiments/runs/trusttab-core-20261114/final_report_finalized_2026-11-14/`, from the unchanged raw CSV. It contains 13 CSVs, 16 PNGs, `report.md`, and an updated `research_findings.md`; existing final report directories were not overwritten. The raw CSV remains 240 unique rows: 200 core conditions and 40 baseline references, all required metrics finite.

### Verification performed

- Focused report tests after the summary fix: `.venv\Scripts\python.exe -m pytest tests\test_reporting.py -q --basetemp .pytest-tmp-report-fix` - 11 passed.
- Full test suite after all source changes: `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-final-finalization-2026-11-14` - 331 passed, one existing scikit-learn single-class warning, 45.69 seconds.
- Lint: `.venv\Scripts\python.exe -m ruff check .` - passed.
- Formatting: `.venv\Scripts\python.exe -m ruff format --check src tests examples scripts` - 34 files already formatted. (A broader check over `.` also scanned a temporary installed-wheel copy and reported its vendored legacy files; the project source/test/example/script scope passes.)
- Whitespace: `git diff --check` - passed; Git emitted only existing LF-to-CRLF working-copy notices.
- Run integrity recheck: 240 rows, 240 unique IDs, 200 noise rows, 40 baseline rows, expected two datasets/four models/five noise levels/five seeds, finite required metrics, and calibration disabled on all noise rows.
- Distribution build: standard `python -m build` was attempted but its isolated build environment could not fetch `setuptools>=77` because package-index network access is denied (`WinError 10013`). Built wheel and sdist offline using the already-installed setuptools 78.1 backend in `.venv\phase8-tabpfn`, from a separate source copy at `.package-build-finalcheck-2026-11-14/`, into `dist\release-candidate-finalized-2026-11-14/`. The project's `.venv` and pre-existing distribution folders were not modified.
- Latest wheel/sdist archive inspection: 42 wheel members and 79 sdist members; no checkpoint, local dataset/result/cache, credential-named file, or credential-pattern match found. SHA256: wheel `1222c5f87355723723ba50ece7728de73c1761d24e4bbcdab101d936d045fe9c`; sdist `d05cf485950747bd3c8faf5c8e927d453059c83650006ae21d3e2c6802602c52`.
- Installed the latest wheel with `pip --no-deps --target` and ran outside the checkout: public API import passed (31 exports), module CLI `--help` passed, and `examples/quickstart.py` completed with finite metrics.
- `twine check` on the latest distributions is **NOT VERIFIED**: Twine is absent from the project and isolated environments, and network access prevents provisioning it. An earlier candidate had passed Twine, but that is not substituted as verification of these newly rebuilt archives.

### Remaining release and research limitations

- RQ4 remains unanswered by the completed core matrix. A separate calibration-rescue experiment needs separate scope/authorization; no such experiment was run here.
- The current raw result schema does not store per-row predictions/probabilities, so aggregate metrics cannot be independently recomputed from raw CSV alone. The report remains reproducible from the stored aggregate rows.
- A network-resolving clean dependency install, Python 3.12 verification, remote GitHub Actions run, current Twine check, TestPyPI rehearsal, and production PyPI project-name check remain outstanding/environment-blocked. The new local artifacts are a candidate only, not release-ready.
- No research matrix was rerun. No commit, push, merge, upload, publication, or release occurred.

## Public PyPI finalization sprint - 2026-11-14

- Inspected the full current status, metadata, README, public API guide, license, changelog, release guide, manifest include rules, CI workflow, available distributions, and dirty worktree. Current branch is `chore/remove-github-actions`; existing edits and research results were preserved.
- Confirmed package metadata is `trusttab` 0.1.0, MIT, Python `>=3.11,<3.13`; core scientific dependencies are separate from the optional XGBoost extra (`xgboost>=2.0`) and TabPFN extra (`tabpfn==10.0.0`). Public convenience imports are in `trusttab.api`; the README documents installation and the offline example.
- Archive scans for the latest wheel and sdist found no research datasets, experiment result files, model weights/checkpoints, cache directories, credential files, or credential-pattern matches. The latest local wheel was installed outside the checkout with `pip --no-deps --target`; public API import, CLI module help, and the offline quick-start passed using already-installed dependencies.
- Environment status: Python 3.11.9 is available in the project environment; Python 3.12 and the `py` launcher are unavailable. GitHub CLI is installed but has no authenticated session. No Twine/TestPyPI credential environment variables or `.pypirc` were detected (values were never read or printed). Twine is unavailable in both project and isolated environments.
- PyPI and TestPyPI project JSON endpoints could not be fetched through the available network tools, so project-name and existing-version availability remain **NOT VERIFIED**. Check that `trusttab`/version `0.1.0` is available before any upload.
- Standard isolated `python -m build` cannot download `setuptools>=77` because package-index network access is denied (`WinError 10013`). Latest wheel and sdist were built offline via installed setuptools 78.1 in the isolated environment from a separate source copy. This is evidence of backend buildability, but does not replace a clean isolated `python -m build` run.
- Latest archive paths and hashes are listed in `PUBLICATION_CHECKLIST.md`. `twine check` on these exact latest archives is **NOT VERIFIED**. No TestPyPI, GitHub Actions, or production PyPI operation occurred.
- Created `PUBLICATION_CHECKLIST.md` with verified, blocked, and outstanding tasks. Overall release state remains **NOT READY** pending clean dependency resolution, latest Twine validation, Python 3.12/remote CI, and TestPyPI rehearsal. No commit, push, upload, or publication occurred.

## Final repository cleanup and release check - 2026-11-14

### Cleanup

- Removed only the explicitly inventoried regenerable caches and pytest basetemp outputs listed in `PUBLICATION_CHECKLIST.md`: `.pytest_cache`, `.ruff_cache`, `.pytest-run2`, 19 `.pytest-tmp-*` directories, and 12 `__pycache__` directories under `src`, `tests`, and `scripts`. Git confirmed no tracked files were among these targets. No research, source, configuration, documentation, package, dataset, or environment file was deleted.
- Preserved all existing distributions, dataset/cache files, full experiment outputs, smoke runs, `build/` (it contains a clean-install venv), `.venv`, the large `.package-verify-*` environment, and an existing `.package-install-cleanup-final-2026-11-14/` package-verification copy. Kept other uncertain scratch directories and the empty tracked `requirements.txt` unchanged.
- The cleanup-sprint pytest basetemp and Ruff cache were regenerated by final verification; these are temporary outputs of the current check, not research outputs. The latest install-target copy created during this sprint was removed after its smoke checks.

### Fresh package candidate

- Fresh wheel and sdist were built to the previously unused `dist/release-candidate-cleanup-2026-11-14/` using setuptools 78.1 from a separate temporary copy. Standard isolated `python -m build --sdist --wheel --outdir dist\\release-candidate-cleanup-2026-11-14` failed before producing files because pip could not fetch `setuptools>=77` (`WinError 10013`).
- Archive inspection passed: wheel 42 members, sdist 79; version 0.1.0, Python `<3.13,>=3.11`, MIT expression, XGBoost extra, and pinned TabPFN 10.0.0 extra verified. No local datasets, results, model weights, cache content, credential file, or credential-pattern match was found.
- Fresh wheel installed outside checkout using `python -m pip install --no-deps --target <new-temp-target> <wheel>`. Public import/version and 31 API exports passed; `python -m trusttab.experiments.run --help` passed; `examples/quickstart.py` completed. This does not verify dependency resolution in a clean environment.
- `python -m twine check dist\\release-candidate-cleanup-2026-11-14\\*` failed because Twine is not installed. No network installation was attempted. Fresh artifact SHA256 values and research-file hashes are in `PUBLICATION_CHECKLIST.md`.

### Final checks and release status

- Full tests: 331 passed, one existing scikit-learn single-class warning (64.16 seconds).
- Ruff lint: passed. Configured source formatting check `ruff format --check src tests examples scripts`: 34 files passed. A broader `ruff format --check .` scanned the existing `.package-install-cleanup-final-2026-11-14/` installed package copy and reported 11 copied legacy modules as unformatted; that directory was preserved. `git diff --check` passed with only LF-to-CRLF notices.
- Research integrity: 240 unique rows (200 core plus 40 baselines), all required model/dataset/noise/seed combinations present, completed manifest, 13 report CSVs and 16 figures. Raw results, manifest, and finalized report were hashed; values are in `PUBLICATION_CHECKLIST.md`. RQ4 remains unanswered. No experiment was rerun.
- Python 3.12, clean dependency-resolving install, latest Twine check, GitHub Actions, TestPyPI install, and registry name/version lookup remain **NOT VERIFIED**. GitHub CLI is unauthenticated; standard build and registry access are blocked by network restrictions; no TestPyPI credential configuration was detected. No commit, push, merge, upload, or release occurred.
- TrustTab is **NOT READY for TestPyPI or production PyPI** until the latest candidate receives Twine validation and clean dependency-resolving install, Python 3.11/3.12 CI passes, registry name/version is confirmed, and TestPyPI rehearsal succeeds. Production publication still requires separate explicit approval.

### Status confirmation - 2026-11-15

- Rechecked the latest cleanup-sprint wheel install, metadata/archive scan, 240-row experiment integrity, and report hashes. The fresh wheel's `--no-deps` outside-checkout install, public API, module CLI help, and offline example passed. No clean resolver install was established.
- Final current-check pytest and formatting-generated temp/cache directories were removed after their checks. All older distribution candidates and research artifacts remain present. `PUBLICATION_CHECKLIST.md` contains exact deletions, SHA256 values, and the network-enabled commands for the remaining gates.
- No code or research artifacts changed after verification; no Git or registry write action was performed.

## Final repository cleanup and public library preparation — 2026-11-15

### Completed

- Reviewed the dirty branch and repository inventory before cleanup. Existing modified/untracked work was preserved; no reset, stash, checkout, commit, push, merge, upload, or publication occurred.
- Removed only inventoried disposable artifacts: obsolete package-install target copy `.package-install-cleanup-final-2026-11-14/` after validating it contained only an older installed-package copy and confirming a newer wheel had been built and smoke-tested; temporary source build copy `.package-build-final-2026-11-15/`; this pass's pytest temp directory and pytest/Ruff caches; and generated Python caches under `src/trusttab/` and `tests/`. All `dist/` candidates, `build/`, egg-info, virtual environments, data/caches, research outputs and source were retained.
- README and changelog describe the completed core research design and results without claiming calibration findings. RQ4 is explicitly unanswered because calibration was disabled. The independent Bernoulli training-label protocol and blueprint pseudocode discrepancy are disclosed, alongside the five-seed/two-fixed-dataset limitations and actual statistical methods.
- Built the latest wheel and sdist into a new output directory and inspected metadata/content. The outside-checkout wheel smoke passed for the 31-export public API, module CLI help, and offline quick-start using existing installed dependencies.
- Verified the completed research snapshot: 240 unique raw rows (200 core conditions and 40 baseline references), complete/integrity-valid manifest, expected dimensions, and no failed/missing/duplicate conditions. Fingerprints and artifact hashes are in `PUBLICATION_CHECKLIST.md`. Research results were neither modified nor rerun.

### Exact current verification

- `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-final-project-status-2026-11-15` — **331 passed, 1 warning** in 43.16s. Existing warning: scikit-learn confusion-matrix single-label warning in `tests/test_benchmark.py::test_roc_auc_is_nan_when_test_split_contains_one_class`.
- `.venv\Scripts\ruff.exe check .` — passed (`All checks passed!`).
- `.venv\Scripts\ruff.exe format --check .` — passed (`34 files already formatted`).
- `git diff --check` — passed; Git emitted only LF-to-CRLF working-copy notices.
- Standard isolated `python -m build` remains **BLOCKED** because fetching `setuptools>=77` fails with Windows `WinError 10013`. Latest artifacts were built using installed setuptools 78.1 from a temporary source copy.
- Archive inspection: wheel has 42 members, sdist has 79; metadata says `trusttab==0.1.0`, Python `>=3.11,<3.13`, MIT, optional XGBoost and TabPFN extras, with TabPFN pinned to 10.0.0. No local datasets/results, checkpoints, caches, or credential-pattern matches found.
- Latest wheel outside-checkout `--no-deps` install, `trusttab.api` import, 31 public exports, `python -m trusttab.experiments.run --help`, and offline quick-start passed. This does not prove clean dependency resolution. Windows Application Control blocks generated `trusttab.exe`.
- `twine check` — **NOT VERIFIED** (`twine` unavailable; network restrictions prevented installing it). Python 3.12, clean dependency-resolving installation, GitHub CI, registry name/version state and TestPyPI remain not verified/not performed.

### Latest distribution artifacts

- `dist/release-candidate-final-2026-11-15/trusttab-0.1.0-py3-none-any.whl` — SHA256 `d24ad8145c787826c81fe3e7ce1f83e834705c0dff23387ba0484a840ee8d523`.
- `dist/release-candidate-final-2026-11-15/trusttab-0.1.0.tar.gz` — SHA256 `1b828e03c59cc94d6e88267bce596a777ee1639744155c3a5d40a79094a9b037`.

### Remaining blockers

- Use a network-enabled Python 3.11 and 3.12 environment to perform isolated builds, latest-artifact Twine validation, dependency-resolving clean wheel/sdist installs outside the checkout, CLI and quick-start checks, and CI.
- Confirm PyPI and TestPyPI name/version state. No TestPyPI credentials/configuration or authenticated GitHub session were available; neither CI nor any upload was performed.
- Obtain separate explicit authorization before TestPyPI upload and production publication. Production publication was not performed.
- Keep RQ4 unanswered unless a distinct calibration experiment is authorized and completed.

See `PUBLICATION_CHECKLIST.md`, `TRUSTTAB_FINAL_RELEASE_AUDIT.md`, and `FINAL_PROJECT_STATUS.md` for the complete inventory, hashes, exact next commands, and release verdict. No core experiment was rerun.

## Main-branch delivery preparation — 2026-11-15

- Rechecked current branch: `chore/remove-github-actions` at `d175cab`, 17 commits ahead of local `main` (`431df4b` is the common ancestor; `main...HEAD` is `0 17`). Existing modifications and untracked files are TrustTab implementation/research docs plus local verification artifacts. The `.package-verify-*` directory is a Python virtual environment and is preserved, not staged.
- Added `.gitignore` protections for `.package-verify-*`, pytest/Ruff/bytecode caches, local phase8 smoke outputs, `.pypirc`, private key files, and `.cache/`. Existing `.env*`, `.venv/`, build/dist, local data, results, figures and experiment-run exclusions remain.
- Reviewed `.github/workflows/ci.yml`: Python 3.11/3.12 core test matrix, Python 3.11/3.12 XGBoost tests, lint/format/CLI, sdist/wheel build, Twine/archive inspection, and outside-checkout install/quick-start jobs. Official action refs `actions/checkout@v7` and `actions/setup-python@v7` exist. This workflow is restored in the pending deliverable after the current branch's earlier workflow-removal commit.
- Latest verification: `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-final-delivery` — **331 passed, 1 existing scikit-learn warning** (59.77s); `.venv\Scripts\ruff.exe check .` — passed; `.venv\Scripts\ruff.exe format --check .` — 34 files formatted; `.venv\Scripts\python.exe -m trusttab.experiments.run --help` — passed; `git diff --check` — passed with line-ending notices.
- A fresh `python -m build --sdist --wheel` attempt failed before building because network access to PyPI is denied (`WinError 10013`) while fetching `setuptools>=77`. `python -m twine check` is not runnable because Twine is absent. The previously built wheel/sdist remain preserved and their hashes are in `PUBLICATION_CHECKLIST.md`.
- GitHub CLI reports its configured account authentication is invalid (no credential value was accessed); `git ls-remote origin refs/heads/main` cannot connect to github.com:443. Therefore GitHub Actions status, remote main freshness, push permission, PyPI/TestPyPI name availability and uploads are **BLOCKED/NOT VERIFIED**. No interactive login or credential change was attempted.
- Delivery staging is limited to TrustTab source/tests/config/docs, CI and package metadata, audit/progress docs, and the finalized core run's raw CSV, manifest, final report tables/figures/findings. Local virtual environments, smoke outputs, downloaded datasets, old duplicate run reports, and `dist/` binaries are not staged. Research files remain intact locally.

## Cherry-pick delivery and GitHub CI — 2026-11-15

- Verified `origin/main` at `d57888a59c0bbdd2a60356e7f30837b17ace517a`, created `delivery/trusttab-final-2026-11-15` from it, and cherry-picked reviewed commit `1bafd0ef624de0fa1afbbf2124d7713f33bca709` without conflicts. Cherry-pick result: `4ef6beb6f6b2a51772c393606840e1030f718786`. No remote history was overwritten.
- Pushed the delivery branch and opened PR [#21](https://github.com/Shreya-nipunge/trusttab/pull/21) targeting `main`. The PR remains open and unmerged; GitHub reports merge state `CLEAN`. The main branch protection endpoint reports `Branch not protected` (404).
- GitHub Actions PR run [43632520439](https://github.com/Shreya-nipunge/trusttab/actions/runs/43632520439) passed all five jobs: core tests on Python 3.11 and 3.12, optional XGBoost tests on Python 3.11 and 3.12, and build/Twine/archive/outside-checkout install tests. The push run [43632487542](https://github.com/Shreya-nipunge/trusttab/actions/runs/43632487542) also passed all five jobs.
- Local checks on the cherry-picked branch: 331 tests passed with the existing single-label sklearn warning; Ruff lint passed; format check reported 34 files formatted; CLI help and offline quick-start passed. A fresh local wheel and sdist also built, and their archive scan passed. The new local artifacts are under `dist/release-delivery-verify-2026-11-15/`.
- Local clean dependency installation remains affected by Windows Application Control blocking pandas DLLs in the preserved isolated venv. GitHub's clean Linux wheel and sdist install jobs passed. No TabPFN weights or data are packaged.
- No merge, TestPyPI upload, production PyPI publication, release tag, or 200-condition rerun occurred. RQ4 remains unanswered. TestPyPI rehearsal and production release approval remain outstanding.
