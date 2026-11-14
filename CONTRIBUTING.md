# Contributing to TrustTab

Thank you for helping improve TrustTab. Keep changes focused, preserve
existing user work, and do not include dataset records, model weights, API
keys, or other credentials in commits.

## Development setup

Use Python 3.11 or 3.12 and install the development extra:

```bash
python -m pip install -e ".[dev]"
```

Optional backends are separate:

```bash
python -m pip install -e ".[xgboost]"
python -m pip install -e ".[tabpfn]"
```

TabPFN weights are separately licensed and gated. Follow
[`docs/tabpfn.md`](docs/tabpfn.md) and never store tokens in source, config,
tests, or logs.

## Local checks

```bash
ruff check .
ruff format --check .
pytest -q
python -m build --sdist --wheel
```

Unit tests should use offline fixtures. Do not run the complete research
matrix as part of ordinary tests. Use a bounded configuration for integration
checks, and label synthetic tests as implementation tests rather than research
results.

## Research-method changes

Do not change dataset versions, split proportions, label-noise semantics,
random seed derivation, calibration separation, model parameters, or metric
conventions without documenting the scientific impact and obtaining review.
Never tune against the test set. Preserve failed and missing conditions as
such; do not fabricate results.
