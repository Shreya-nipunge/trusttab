# TrustTab Final Research and Release Audit

Date: 2026-11-14  
Scope: audit of the saved 200-condition run, report generation, package artifacts, local verification, and publication blockers. No experiment rerun, commit, push, or upload was performed.

## Executive assessment

The saved core experiment is structurally complete and its generated tabular report and figures reproduce deterministically from `raw_results.csv`. The scientific record still has two explicit limitations: calibration was disabled (RQ4 unanswered), and parts of the inferential/reporting implementation differ from the original blueprint or expected figure plan. The Python package builds and passes Twine validation, with optional XGBoost and TabPFN kept out of core dependencies. A fresh dependency-resolving install, Python 3.12 run, remote CI, and TestPyPI rehearsal remain unverified because this Windows environment lacks the required runtime/network/authorization. **Do not publish yet.**

## Research-run audit

Inputs inspected:

- `experiments/runs/trusttab-core-20261114/raw_results.csv`
- `experiments/runs/trusttab-core-20261114/run_manifest.json`
- `experiments/runs/trusttab-core-20261114/final_report/research_findings.md`
- `experiments/runs/trusttab-core-20261114/final_report/report.md`
- `configs/experiments.yaml`, `Project_Execution_Blueprint.md`, and current runner/statistics/reporting code.

| Check | Result | Evidence |
|---|---|---|
| Core matrix | PASS: 200/200 unique conditions; 2 datasets, 4 models, 5 noise levels, 5 seeds | Manifest integrity complete; IDs independently reconstructed from config |
| Clean baseline references | PASS: 40/40 unique rows | Expected dataset × model × seed baseline IDs |
| Total result rows | PASS: 240 unique `condition_id`s | Raw CSV |
| Failures/gaps/duplicates | PASS: none | Manifest and independent ID-set comparison |
| Dataset fingerprints | PASS: recomputed values match manifest | German `be2da28ee0a92114ae07549c876cb414e1dd2ead0a56fa9e19dc1535c4ac6fa6`; Cleveland `d9ed8e3110da6747811725a53fb068be2bd233684c011b44563bc93e6517dd27` |
| Required metrics | PASS for finiteness/range-bearing metric outputs: all six required raw metrics finite on all rows | `accuracy`, `balanced_accuracy`, `macro_f1`, `roc_auc`, `ece`, `brier_score` |
| Label-noise counts | PASS: all 200 recorded counts replayed from the saved split/noise seeds | Replay used the cached dataset labels and approved independent Bernoulli injector |
| Calibration status | As configured: all 200 core conditions `disabled`; calibrated ECE/Brier missing | No calibration values were fabricated or imputed |
| Report reproduction | PASS: 13 tabular CSVs, `report.md`, and 13 PNGs byte-identical on regeneration from raw CSV | Separate temporary output directory; no raw results overwritten |
| Direct metric recomputation | NOT POSSIBLE from stored run data | Per-observation predictions/probabilities are not in `raw_results.csv`; finite outputs and code/tests were checked, but scores cannot be independently recomputed from this artifact alone |

The experiment runner uses the original clean `y_test` for evaluation and passes only `y_train` to the noise function. All noise counts replayed as expected. Run ID is `trusttab-e18ead6a25c341549ba41ce12d7f5ef8`; normalized configuration hash is `f1f8d9f3bac2bdc614f6d956c96da839bc9453e04e3a61198862d4b4598afeb3`.

### Scientific interpretation and deviations

`research_findings.md` is cautious: it describes aggregate trajectories, treats rank comparisons descriptively, limits generalization to the two fixed datasets, and explicitly leaves RQ4 unanswered. The core matrix cannot answer whether Platt scaling rescues reliability. A separate, full calibration-rescue experiment would be needed for RQ4; the earlier bounded smoke is integration evidence only.

The planned Friedman/Nemenyi and Wilcoxon analyses are present, with seed blocks, dataset-specific tests, correction families and missing calibration pairs represented. The regeneration check establishes reproducibility of the generated tables and figures; it does not independently re-derive each statistical formula from first principles. Five seeds make the Friedman chi-square approximation weak, as the statistics documentation acknowledges.

Two analysis/reporting deviations need resolution or an explicit final-paper caveat:

1. `statistical_comparisons.csv` and the corresponding report section also contain paired *t*-tests with Holm correction. These are supplementary to Part M's planned Friedman/Nemenyi model tests, but the report does not prominently identify their parametric, five-pair limitations. The RQ summary does not rely on these tests. Do not present them as the pre-specified primary analysis.
2. `statistical_summary.csv` uses Student-*t* intervals. Part M specified bootstrap intervals over seeds or mean ± standard deviation; the current parametric interval is a documented implementation choice but a deviation from those stated options. Reconcile this before publication; five observations give unstable interval estimates.

The generated figure set is reproducible but does not fulfill the full planned figure list. It includes an accuracy performance/noise plot, model comparison, performance degradation, a generic ECE scenario plot, an empty/not-applicable feature-shift plot, and eligible critical-difference diagrams. Missing are dedicated ECE, Brier, and Macro-F1 versus noise plots; reliability diagrams; confidence histograms; model-ranking heatmaps; a dedicated cross-dataset comparison panel; and a raw-versus-calibrated ECE comparison. The latter cannot be generated from this core run because calibration is disabled. Do not describe the generic ECE scenario plot as raw-versus-calibrated evidence.

The report also contains supplementary paired model comparisons; do not use them to replace the blueprint's seed-blocked Friedman/Nemenyi analysis. No statistically supported calibration conclusion is available.

## Package and distribution audit

- Structure: `src/trusttab/` layout; package modules have `__init__.py`; curated public convenience surface is `trusttab.api`; top-level `trusttab` exposes version only.
- Metadata: package `trusttab`, version `0.1.0`, setuptools backend (`setuptools>=77`), MIT code license, Python `>=3.11,<3.13`, console entry point `trusttab=trusttab.experiments.run:main`.
- Core requirements: NumPy, pandas, scikit-learn, Matplotlib, PyYAML, SciPy. Metadata confirms XGBoost and TabPFN are optional extras; TabPFN is pinned to `tabpfn==10.0.0` and is not a core dependency.
- The project venv's `pip check` passes. `trusttab.api` imports and basic API/quick-start execution were verified from a wheel target outside the repository using the existing installed core dependencies; an import hook blocking both `xgboost` and `tabpfn` verified those optional packages are not imported for the core API.
- The wheel contains 42 members and the sdist 105. Both contain the package API and license; the sdist contains README/docs. Archive inspection found no dataset files, model checkpoints, cache directories, credential files, or forbidden result files.
- Final release candidate artifacts are preserved at `dist/release-candidate-final-2026-11-14/` (wheel 79,885 bytes; sdist 125,561 bytes). They were built after the documentation changes.
- `twine check` passed for both final artifacts. Twine was run as `python -m twine` from a temporary tool environment because Windows Application Control blocked its generated executable.
- The README now clarifies that YAML templates are repository files, not wheel package data; a pip-only user should use the public Python API or provide a YAML file. It also points to current TabPFN license terms.

### Environment-limited package checks

- The first clean install selected global Python 3.14.6, which is outside the supported range. The project venv is Python 3.11.9. No `py` launcher or `python3.12` command is available.
- A fresh Python 3.11 venv dependency-resolving wheel installation could not fetch core packages: pip's connection to its package index failed with Windows `WinError 10013` (socket access denied). Therefore, a clean dependency installation is **NOT VERIFIED**. The outside-checkout wheel-target import used already-installed core dependencies and is partial evidence, not a clean resolution.
- Module CLI help passed. Windows Application Control blocked the generated `trusttab.exe`; the security policy was not changed. The offline quick-start passed when executed with the installed wheel on `PYTHONPATH` and the existing environment's dependencies.
- Local test suite: 329 passed, one existing sklearn single-class warning. Ruff lint, formatting, and `git diff --check` passed.
- CI workflow was reviewed but not run on GitHub: it is local/uncommitted and the worktree contains broad pre-existing dirty changes. No commit or push was made. Python 3.12 CI coverage remains unverified.
- TestPyPI was not attempted. Safe presence checks found no configured Twine/TestPyPI credential variables and no `.pypirc`; upload additionally requires explicit authorization. Production PyPI was not attempted and requires separate explicit approval.
- PyPI project-name availability for `trusttab` is **NOT VERIFIED**; direct PyPI lookup could not be fetched under the environment's network restriction. Do not claim `pip install trusttab` works until a published release is installed from PyPI and tested.

## Files changed in this audit pass

- `README.md` — documents the current TabPFN output/eligibility caveat and distinguishes repository YAML examples from wheel-installed use.
- `docs/tabpfn.md` — records current checkpoint license eligibility and output restrictions; links to the upstream license.
- `IMPLEMENTATION_PROGRESS.md` — appends this audit's verification and blockers.
- `TRUSTTAB_FINAL_RELEASE_AUDIT.md` — this report.
- `dist/release-candidate-final-2026-11-14/*` — newly created, verified release candidate distributions; the prior `dist` artifacts were not overwritten.

Build also refreshed existing ignored `build/` and `src/trusttab.egg-info/` metadata. No source implementation, raw experimental result, research configuration, dataset cache, model checkpoint, or existing release artifact was changed/deleted. No commit, push, merge, TestPyPI upload, or production publication occurred.

## Readiness verdict

- **Core experiment integrity:** PASS.
- **Research interpretation/report completeness:** PARTIAL; RQ4 unanswered, some blueprint/statistical-method deviations require resolution/caveats, planned figures incomplete, and predictions are not retained for independent metric recomputation.
- **Package metadata/build/archive/Twine checks:** PASS for the local 0.1.0 candidate.
- **Fresh install / Python 3.12 / remote CI:** NOT VERIFIED or BLOCKED BY ENVIRONMENT.
- **TestPyPI / production PyPI:** NOT PERFORMED; authorization and account configuration are outstanding.
- **Overall public PyPI release readiness:** NOT READY. Keep the candidate local until the clean-install and remote CI evidence is available, the PyPI name is confirmed, and TestPyPI rehearsal is explicitly authorized and passes.

## Exact remaining actions

1. Resolve a network-enabled clean Python 3.11 install and provide a Python 3.12 environment or authorize an isolated GitHub CI run after the reviewed changes are committed/pushed in a way that preserves this dirty worktree.
2. Confirm PyPI name availability for `trusttab` from an authorized network.
3. Decide how to handle the additional paired *t*-tests and Student-*t* confidence intervals against Part M before the final academic report; do not silently treat these as pre-specified analyses.
4. Complete or explicitly defer the missing figure plan. Keep RQ4 unanswered unless the separate calibration-rescue experiment is run and analyzed.
5. Configure TestPyPI publishing credentials or Trusted Publishing locally, then give explicit authorization before upload. Do not share credentials in chat.
6. After all rehearsal checks pass, review the exact wheel/sdist and release notes and give separate explicit production PyPI approval if publication is desired.

## Final report and release-candidate recheck - 2026-11-14

This section supersedes the earlier notes above where they describe Student-t confidence intervals or missing dedicated metric curves. No experimental run was repeated and no raw result was edited.

- Reanalyzed the preserved 240-row result file. It still contains 200 unique core noise conditions and 40 unique clean-baseline references, with the full two-dataset/four-model/five-noise/five-seed grid and finite required raw metrics. Calibration remains disabled for all core noise conditions, so RQ4 remains unanswered.
- Updated reporting to use deterministic percentile bootstrap intervals over the five seed-level units (10,000 resamples) for descriptive intervals. This does not bootstrap hypothesis tests. Existing paired t-tests are marked as supplementary exploratory analyses; the planned Friedman/Nemenyi results remain the primary model comparisons.
- Added ECE, Brier, and Macro-F1 noise figures, model-ranking heatmap, and cross-dataset comparison. Reliability diagrams and confidence histograms remain unavailable because observation-level predictions are absent from saved results. Calibration comparison plots are omitted when the input has no successful calibrated results.
- Corrected the summary so `best_calibrated_model` is `unavailable` when no calibrated reliability metrics exist; the prior summary had mislabeled raw reliability ranking as calibrated performance.
- Reproducible report output is preserved separately at `experiments/runs/trusttab-core-20261114/final_report_finalized_2026-11-14/` (13 CSVs, 16 PNGs, report and updated research findings). The pre-existing report directory was not overwritten.
- Latest source verification: 331 tests passed, one existing scikit-learn single-class warning; Ruff lint passed; Ruff formatting passed on `src tests examples scripts` (34 files); `git diff --check` passed.
- Attempted `python -m build`; isolated build dependency retrieval was blocked by network policy (`WinError 10013`). Rebuilt both artifacts offline using setuptools 78.1 from a separate source copy. Latest wheel and sdist are under `dist/release-candidate-finalized-2026-11-14/`. Archive scan found 42 wheel and 79 sdist members with no dataset/checkpoint/cache/result artifacts or credential-pattern matches. Latest wheel was installed into a separate target outside the checkout; API import, CLI module help, and offline quick-start passed.
- `twine check` for these latest archives remains **NOT VERIFIED** because Twine is unavailable locally and network access is blocked. Earlier Twine evidence applies only to the earlier candidate. Clean dependency-resolving installation, Python 3.12, remote GitHub Actions, TestPyPI, and PyPI name availability are also outstanding. No upload or publication was attempted.

### Revised verdict

- Research-run integrity: **PASS**, limited to the fixed datasets and five seeds.
- Report consistency: **IMPROVED / PARTIAL**; bootstrap intervals and missing planned aggregate figures are addressed, but observation-level reliability figures and independent score recomputation remain unavailable, and RQ4 is unanswered.
- Local package build/import/API/CLI/quick-start: **PASS** for the latest local candidate.
- Public release readiness: **NOT READY** until latest `twine check`, clean install/Python-version CI, remote CI, and an authorized TestPyPI rehearsal pass. Production publishing still requires separate explicit approval.
- Actions in this recheck: no experiment run, commit, push, merge, upload, publication, or release.

## Public PyPI finalization sprint - 2026-11-14

The package candidate and worktree were re-inspected. Metadata remains version 0.1.0, MIT, Python `>=3.11,<3.13`. XGBoost (`>=2.0`) and TabPFN (`==10.0.0`) are optional extras; public API imports, README/API documentation, offline example, license, changelog, and release procedure are present. Archive inspection of the latest artifacts found no results, local datasets, model weights, caches, or credential patterns. The latest wheel was installed outside the source checkout with `--no-deps`; import, CLI help, and quick-start succeeded with preinstalled dependencies.

Release verification limitations remain material: clean dependency-resolving installation is blocked by PyPI network access (`WinError 10013`); Python 3.12 is unavailable locally; `twine check` could not run because Twine is absent; GitHub CLI has no authenticated session and workflow changes remain uncommitted; no TestPyPI credentials/configuration were found. No CI dispatch, upload, or publication occurred. See [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md) for exact artifact paths/hashes and remaining actions.

PyPI and TestPyPI project JSON lookups also could not be fetched with the available network tools; the `trusttab` project name and whether version `0.1.0` is already registered remain **NOT VERIFIED**. Confirm registry state before uploading so no existing published release is overwritten or duplicated.

The latest wheel and sdist were produced with installed setuptools 78.1 from a temporary source copy after the isolated standard build failed to fetch `setuptools>=77`. This does not establish clean build-isolation or clean dependency-resolution readiness. Therefore public release readiness remains **NOT READY**. Production PyPI publication was neither requested for this sprint nor performed.

## Final repository cleanup and candidate verification - 2026-11-14

- Removed only the identified pytest basetemp outputs, `.pytest_cache`, `.ruff_cache`, and Python `__pycache__` directories under source/tests/scripts. The deleted paths are enumerated in `PUBLICATION_CHECKLIST.md`. Git reported no tracked content in those targets.
- Preserved all pre-existing dist artifacts, local data, research and smoke outputs, `.venv`, `.package-verify-*`, and the mixed `build/` directory because it contains an environment. No raw experiment, manifest, report, configuration, source, or documentation file was deleted.
- Fresh wheel and sdist candidates were written to `dist/release-candidate-cleanup-2026-11-14/`; archive metadata and contents passed inspection. The new wheel was installed outside checkout with `--no-deps`; API imports, module CLI help, and offline example passed. This is not a clean dependency-resolving install.
- The normal isolated build and latest Twine check remain blocked: isolated setuptools fetch failed with `WinError 10013`; Twine is absent. Full tests passed (331, one existing sklearn warning), Ruff lint and source-scope formatting passed, and `git diff --check` passed. A recursive formatting scan over the dirty workspace encounters an existing package-install verification copy with legacy formatting; this copy was preserved.
- Current research checksums, new package checksums, deleted-cache inventory, and exact release gate commands are recorded in `PUBLICATION_CHECKLIST.md`. Experiment rows/hash, manifest, finalized report, and RQ4 status were verified; RQ4 remains unanswered. No experiment, upload, publication, commit, push, or merge occurred.
- GitHub Actions, Python 3.12, registry name/version availability, clean dependency resolution, current Twine check, and TestPyPI remain unverified. The release verdict remains **NOT READY for TestPyPI or production PyPI**.

### Status confirmation - 2026-11-15

The cleanup-sprint full test, fresh distribution, outside-checkout no-deps wheel smoke, and research hash checks are recorded in `IMPLEMENTATION_PROGRESS.md` and `PUBLICATION_CHECKLIST.md`. Existing release candidates and user work remain preserved. No further experiment, commit, push, upload, or release was performed. The latest candidate is suitable for external verification but is not yet a verified TestPyPI release candidate.

## Final repository cleanup and release candidate revalidation — 2026-11-15

### Research preservation and documentation

The completed `experiments/runs/trusttab-core-20261114/` run was left untouched. The preserved CSV has 240 unique rows: 200 model/dataset/noise/seed conditions plus 40 clean-baseline references. The manifest records complete status and no missing, duplicate, failed, or unexpected conditions. RQ4 remains unanswered because calibration was disabled in this core matrix. README and changelog now summarize the actual design, independent Bernoulli training-only noise, the pseudocode deviation, seed-blocked methods, and limitations of five seeds/two fixed datasets. No experimental claims were added beyond saved evidence.

Raw result SHA256: `f81a57fe1828804478dbb80ab2dd9c7488987555bc43df41980a2a934f75c29a`. Manifest SHA256: `5115289bd75a811e135ae450dae51f9c5c87ccfa5e755097fea02b2523ef4868`. Dataset fingerprints: German Credit `be2da28ee0a92114ae07549c876cb414e1dd2ead0a56fa9e19dc1535c4ac6fa6`; Cleveland `d9ed8e3110da6747811725a53fb068be2bd233684c011b44563bc93e6517dd27`. Final report hashes and output counts are recorded in `PUBLICATION_CHECKLIST.md`.

### Cleanup inventory and actions

- Deleted older `.package-install-cleanup-final-2026-11-14/` only after checking it was a disposable pip target containing an older TrustTab install (not a virtual environment or research artifact) and verifying a fresh replacement candidate outside the source tree. Deleted ten `__pycache__` subdirectories inside that copy.
- Deleted this pass's temporary source copy `.package-build-final-2026-11-15/`, pytest basetemp `.pytest-tmp-final-project-status-2026-11-15/`, `.pytest_cache/`, `.ruff_cache/`, and generated source/test `__pycache__` directories after verification.
- Preserved all `dist/` candidates, mixed-purpose `build/`, `src/trusttab.egg-info`, `.venv` and `.venv/phase8-tabpfn`, `.package-verify-*`, data and dataset caches, experiment results, figures, manifests, reports, scratch files of uncertain ownership, and all dirty user changes. No tracked source/research file was deleted.

### Latest candidate and checks

- Wheel: `dist/release-candidate-final-2026-11-15/trusttab-0.1.0-py3-none-any.whl`, SHA256 `d24ad8145c787826c81fe3e7ce1f83e834705c0dff23387ba0484a840ee8d523`.
- Sdist: `dist/release-candidate-final-2026-11-15/trusttab-0.1.0.tar.gz`, SHA256 `1b828e03c59cc94d6e88267bce596a777ee1639744155c3a5d40a79094a9b037`.
- Archive inspection: 42 wheel members/79 sdist members; package metadata and extras correct; no credentials, datasets, local caches, results, or model weights found. Package version 0.1.0 supports Python `>=3.11,<3.13`, is MIT-licensed, and keeps XGBoost and TabPFN optional (TabPFN pinned at 10.0.0).
- Full tests: 331 passed, one existing scikit-learn single-class confusion-matrix warning. Ruff lint and recursive format check pass (34 files formatted). `git diff --check` passes with only Git line-ending notices.
- Installed wheel outside checkout with `--no-deps`: public API (31 exports), module CLI help, and offline quick-start pass. Existing dependencies were used, so this is not a clean dependency-resolution install. Windows Application Control blocked the generated `trusttab.exe` shim; no policy was changed.
- Latest `twine check` is **NOT VERIFIED** because Twine is unavailable. Standard isolated build is **BLOCKED** by network `WinError 10013` fetching setuptools; this candidate was created with installed setuptools 78.1. Python 3.12, clean dependency-resolving install, GitHub Actions, registry state, and TestPyPI remain not verified/not performed.

### Verdict

The local distribution candidate is suitable for external network-enabled verification, but TrustTab is **NOT READY for TestPyPI upload** until clean isolated builds, Twine, clean installs, Python 3.11/3.12 CI and project/version state are verified. It is **NOT READY for production PyPI** until those checks pass and an explicitly authorized TestPyPI rehearsal succeeds. No commit, push, upload, release, or experiment run occurred.
