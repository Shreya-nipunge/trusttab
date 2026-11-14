# TrustTab — Final Project Status

**Updated:** 2026-11-15  
**Overall:** Local package candidate prepared; **not ready for TestPyPI or production PyPI** until external release checks pass.

## What TrustTab does

TrustTab is a Python toolkit for studying how training-label noise affects tabular-model performance and probability reliability. Its research implementation includes dataset loading, train-only preprocessing, independent Bernoulli training-label noise, Logistic Regression, Random Forest, optional XGBoost and TabPFN, Platt calibration, metrics, statistical comparisons, experiment execution, and reporting.

The completed core run used German Credit and processed Cleveland Heart Disease; four models; noise rates 0%, 10%, 20%, 30%, and 40%; and seeds 0–4. It contains 200 core rows and 40 clean-baseline references. RQ4 (whether post-hoc calibration rescues reliability) remains unanswered because calibration was disabled in the core matrix. The fixed two datasets and five seeds limit generalization.

## Public API and installation

The curated API is `trusttab.api` (31 public exports). Install core dependencies with:

```powershell
python -m pip install trusttab
```

Optional models are installed with `python -m pip install "trusttab[xgboost]"` or `python -m pip install "trusttab[tabpfn]"`. TabPFN is pinned to 10.0.0 and requires users to obtain their own eligible checkpoint access; weights are not distributed. The project CLI entry point is `trusttab`. Repository examples and API reference are under `examples/` and `docs/api.md`.

**Publication caveat:** the package is not published yet, so the command above is an intended future installation command, not a verified live PyPI install.

## Repository structure

- `src/trusttab/` — installable package, API, data, evaluation, experiments, models, noise, reporting, statistics, visualization.
- `tests/` — unit and integration coverage.
- `configs/` — research and bounded smoke configurations.
- `docs/`, `examples/` — user and research documentation/examples.
- `experiments/runs/trusttab-core-20261114/` — preserved raw run, manifest, fingerprints, finalized reports/tables/figures.
- `paper/`, `figures/`, `notebooks/`, `results/` — research materials and supporting artifacts.
- `dist/` — preserved historical candidates and the latest candidate below.

No structural moves were made in this cleanup.

## Completed research and artifact locations

- Raw results: `experiments/runs/trusttab-core-20261114/raw_results.csv` (240 rows).
- Run manifest: `experiments/runs/trusttab-core-20261114/run_manifest.json`.
- Final findings/report: `experiments/runs/trusttab-core-20261114/final_report_finalized_2026-11-14/research_findings.md` and `report.md`.
- The final report directory contains 13 CSV outputs and 16 figures. Raw results, manifest, dataset fingerprints, and reports were preserved; no experiment was rerun or altered during this cleanup.
- Noise is independent Bernoulli flipping on training labels only. The realized corruption count is random. Calibration and test labels remained clean in the protocol. The README records the difference from the blueprint’s fixed-count-looking Part F pseudocode.

## Verification

- Full tests: **331 passed, 1 warning**. The warning is scikit-learn’s single-label confusion-matrix warning in `tests/test_benchmark.py::test_roc_auc_is_nan_when_test_split_contains_one_class`.
- Ruff lint: passed. Ruff formatting: 34 files already formatted. `git diff --check`: passed (Git printed only line-ending notices).
- Outside-checkout wheel `--no-deps` install, public API import (31 exports), module CLI help, and offline quick-start: passed using dependencies already present in the environment.
- Distribution archives inspected: 42 wheel members, 79 sdist members. Metadata is trusttab 0.1.0, Python `>=3.11,<3.13`, MIT; XGBoost and TabPFN are optional and TabPFN is pinned. No data, results, checkpoints, caches, or credential patterns were found.
- Standard isolated build: blocked by network `WinError 10013` fetching `setuptools>=77`; latest artifacts were built using installed setuptools 78.1 from a temporary source copy. Latest `twine check`: not verified because Twine is unavailable. Clean dependency resolution: not verified. Python 3.12 and GitHub Actions: not verified. Windows Application Control blocks the generated `trusttab.exe` shim; module CLI works.

## Latest wheel and sdist

- Wheel: `dist/release-candidate-final-2026-11-15/trusttab-0.1.0-py3-none-any.whl`  
  SHA256: `d24ad8145c787826c81fe3e7ce1f83e834705c0dff23387ba0484a840ee8d523`
- Source distribution: `dist/release-candidate-final-2026-11-15/trusttab-0.1.0.tar.gz`  
  SHA256: `1b828e03c59cc94d6e88267bce596a777ee1639744155c3a5d40a79094a9b037`

## Cleanup performed and preservation

Deleted only verified disposable items: the obsolete `.package-install-cleanup-final-2026-11-14/` pip target copy after replacement-wheel verification; temporary `.package-build-final-2026-11-15/`; `.pytest-tmp-final-project-status-2026-11-15/`; `.pytest_cache/`, `.ruff_cache/`; and generated `__pycache__` under source/tests (including ten within the obsolete install copy). Earlier explicitly listed cache cleanup is documented in `PUBLICATION_CHECKLIST.md`.

Preserved all dirty user work, every `dist/` candidate, `.venv/` and `.venv/phase8-tabpfn`, `.package-verify-*`, `build/`, egg-info, data and dataset caches, all 240 research rows, manifests, fingerprints, reports, figures, and scratch artifacts whose ownership/purpose was uncertain. No commit, push, merge, upload, or publication occurred.

## Remaining blockers and exact next actions

1. In a network-enabled supported environment, build in a new output directory and run Twine:
   ```powershell
   python -m pip install build twine
   python -m build --sdist --wheel --outdir dist\release-network-verified
   python -m twine check dist\release-network-verified\*
   ```
2. Install both wheel and sdist with dependency resolution into fresh Python 3.11 and 3.12 environments outside the checkout; run public API import, CLI and quick-start. Run the configured GitHub Actions workflow for Python 3.11/3.12.
3. Check PyPI and TestPyPI name/version availability. Only after release-candidate review and explicit authorization, perform a TestPyPI rehearsal and validate installation from that index.
4. Production PyPI publication requires separate explicit approval. Do not claim `pip install trusttab` works until a published artifact has been installed and independently tested from PyPI.
5. Keep RQ4 unanswered unless a separate calibration experiment is authorized, run, and analyzed.

## Readiness verdict

- **Research artifact preservation:** READY; core experiment was not rerun or modified.
- **Local package candidate:** READY for external network-enabled validation.
- **TestPyPI:** **NOT READY** (Twine, clean dependency installs, Python 3.11/3.12 CI, and registry state unverified; upload not authorized/performed).
- **Production PyPI:** **NOT READY** (all preceding gates plus an authorized successful TestPyPI rehearsal and separate production approval remain outstanding).

## Main delivery update — 2026-11-15

Current branch: `chore/remove-github-actions` (`d175cab`), 17 commits ahead of the local `main` ancestor. The repository has substantial pre-existing TrustTab work; it is retained. Local verification environment `.package-verify-4248394bedca44beabddcd84f157b135/` is preserved and ignored, not staged. Local smoke outputs and all source/test/config/documentation changes remain preserved.

Latest local checks: 331 tests passed with one existing sklearn warning; Ruff lint/format, CLI module help, and `git diff --check` passed. The `.github/workflows/ci.yml` config covers Python 3.11/3.12, optional XGBoost, tests, lint, formatting, builds, archive inspection, and outside-checkout installation. **Remote CI has not run.** The standard isolated build is blocked by network `WinError 10013` fetching setuptools; Twine is absent. GitHub CLI authentication is invalid and direct remote access to github.com:443 is blocked. No push or remote CI action has occurred.

The finalized core research files remain locally intact; RQ4 is unanswered. The local 0.1.0 wheel and sdist hashes and prior outside-checkout no-deps smoke results remain in the publication checklist. TestPyPI/production publishing is not performed. See the latest delivery entry in `IMPLEMENTATION_PROGRESS.md` for the exact blockers.
