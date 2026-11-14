# Platt calibration rescue

TrustTab's calibration rescue experiment asks whether Platt scaling fitted on
separate clean calibration labels improves probability reliability after a
model has been trained on noisy labels. This is an evaluation path, not a
claim that calibration improves performance; the experiment must be run and
analyzed before making any empirical claim.

## Data protocol

The original dataset is split into three stratified, disjoint partitions using
the blueprint's proportions:

- 70% training
- 15% calibration
- 15% final test

The split is performed before any model or preprocessing fit. The experiment
then follows this order:

```text
Original data → train/calibration/test split → training-only label noise
→ model and preprocessing fit → clean calibration → raw and calibrated test evaluation
```

Preprocessing and the base model are fit only on training features and noisy
training labels. The model is not refit on calibration data. Platt scaling is
fit using model outputs on the calibration features and their clean labels.
The test labels are never passed to the calibrator; raw and calibrated metrics
are computed against those same untouched test labels after calibration has
been fit. The `split_seed` and `calibration_split_seed` fields identify the two
deterministic split stages, while existing `noise_seed` and `model_seed` fields
continue to identify their respective operations.

The blueprint calls calibration a secondary experiment, so it is disabled by
default. Existing uncalibrated configurations retain their configured
train/test split (currently 80/20 in the core config). When calibration is
enabled, the nested calibration proportions are used instead of the legacy
`experiments.test_size` value. All three proportions are configurable and
must be positive and sum to 1. Both binary classes must appear in each split.
On processed Cleveland, the default produces 212 training, 45 calibration, and
46 test observations (rounding according to scikit-learn's stratified splitter);
the relatively small calibration partition means the fitted sigmoid can still
have substantial sampling uncertainty.

## Calibration method and model interface

The runner uses scikit-learn's `CalibratedClassifierCV(method="sigmoid")`
wrapped around `FrozenEstimator`. This calibrates an already-fitted TrustTab
pipeline without refitting its preprocessing or estimator. Scikit-learn uses
`decision_function` when the base model exposes it and otherwise calibrates
`predict_proba`. The registered TrustTab models expose probabilities for the
raw ECE and Brier calculations; a future decision-score-only raw model would
need a documented probability conversion before it could provide comparable
raw reliability metrics.

Calibrated probabilities are checked for binary class coverage, finite values,
the `[0, 1]` range, and rows summing to one. Positive-class probabilities are
selected by class label `1`, not by assuming a fixed column order. Existing
metric implementations and their conventions are used. Accuracy, balanced
accuracy, macro F1, ROC-AUC, ECE, Brier Score, and the existing additional
metrics are stored for both raw and calibrated outputs. Classification uses
each estimator's `.predict()` behavior; calibration is primarily interpreted
through probability metrics, not as an accuracy intervention.

## Configuration and use

```yaml
experiments:
  test_size: 0.2 # used only while calibration is disabled
  label_noise:
    enabled: true
    levels: [0.0, 0.2, 0.4]
  seeds: [0, 1, 2, 3, 4]
  calibration:
    enabled: true
    method: sigmoid
    train_size: 0.70
    calibration_size: 0.15
    test_size: 0.15
```

Run the bounded one-model smoke configuration with:

```bash
python -m trusttab.experiments.run --config configs/calibration_rescue_smoke.yaml
```

Dataset loading may require network access when no local cache is available.
Unit tests use deterministic local fixtures and do not download datasets.

## Result fields

The runner retains one row per experiment condition, preserving compatibility
with the existing reporting schema:

- Standard metric columns (`accuracy`, `ece`, `brier_score`, etc.) contain raw
  model results.
- The corresponding `calibrated_` columns contain Platt-calibrated results.
- `condition_id` links the raw and calibrated metrics on that row to the exact
  dataset/model/noise/seed condition.
- `calibration_method`, `calibration_status`, and `calibration_error` identify
  the method and outcome.
- `n_train`, `n_calibration`, `n_test`, `n_corrupted`, and the derived seed
  columns record the sample and stochastic-operation metadata.
- `model_params` records the serialized per-model parameters.

If calibration fails, the row is retained with `calibration_status: failed`,
an explanatory `calibration_error`, and NaN calibrated metrics. Raw metrics
remain available; no calibrated values are fabricated. If calibration is
disabled, the calibrated metric fields are NaN and status is `disabled`.
