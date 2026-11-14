# Changelog

This project follows semantic versioning once public releases begin. Until the
first PyPI release, unreleased implementation work is recorded here and in
`IMPLEMENTATION_PROGRESS.md`.

## 0.1.0 (Unreleased)

- Add German Credit, processed Cleveland Heart Disease, and optional Adult
  dataset loaders.
- Add train-only preprocessing, reproducible independent Bernoulli label
  noise, four optional/classical model paths, Platt calibration, evaluation,
  statistical analysis, reporting, and local experiment tracking.
- Add the curated `trusttab.api` convenience surface and offline quick-start.
- Document the completed 200-condition core research run, its 40 clean-baseline
  references, the seed-level statistical protocol, and the unanswered RQ4
  calibration question. Experiment outputs remain separate from package data.
- Package is not published to PyPI; see `docs/releasing.md` before release.
