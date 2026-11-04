# Feature Distribution Shift Testing

Feature distribution shift occurs when held-out examples differ in their input
features from the data used to train a model. TrustTab's feature-shift runner
fits each model on the original training split, then evaluates it on both the
unchanged test split and independently shifted copies of that test split. This
keeps the clean result available as a baseline and isolates the evaluation-time
input shift.

## Supported shifts

Use `apply_feature_shifts` from `trusttab.noise` or configure shifts in YAML:

- `gaussian_noise`: adds Gaussian noise with standard deviation equal to the
  feature's observed standard deviation times `magnitude`.
- `mean_shift`: adds `magnitude` in the feature's units. Positive and negative
  offsets are allowed.
- `variance_scaling`: multiplies variance by `magnitude` around the observed
  mean. A magnitude of 0 collapses observed values to their mean.
- `category_probability_redistribution`: mixes the empirical category
  frequencies toward supplied `probabilities`, or toward a uniform target if
  none is supplied. `magnitude` is the mixing fraction from 0 to 1.
- `category_replacement`: replaces a controlled fraction of observed values.
  By default each selected value changes to another observed category. A fixed
  `replacement` or per-feature `replacements` mapping can select the target.

All transformations preserve missing values and the original DataFrame. The
seed controls random noise and categorical sampling/replacement. Categorical
targets are checked against values available in the affected columns.

## YAML configuration

Add an optional top-level `feature_shifts` list to an existing experiment
configuration. In feature-shift mode, `noise_levels` may be omitted because
training labels are not corrupted. The CLI dispatches to the feature-shift
runner when this list is present; existing label-noise configurations remain
valid without it.

```yaml
feature_shifts:
  - type: gaussian_noise
    features: [age]
    magnitude: 0.2

  - type: mean_shift
    features: [age]
    magnitude: 5.0

  - type: category_probability_redistribution
    features: [job]
    magnitude: 0.5
    probabilities:
      worker: 0.2
      manager: 0.8

  - type: category_replacement
    features: [job]
    magnitude: 0.1
    replacement: manager
```

The configuration loader validates transformation names, feature lists,
magnitudes, and probability mapping structure. Feature existence, numerical
column types, and category availability are checked against the dataset when
the experiment runs.

## Running and interpreting experiments

```python
from trusttab.experiments import run_feature_shift_test

results = run_feature_shift_test(
    dataset_name="german_credit",
    model_names=["logistic_regression", "random_forest"],
    seeds=[1, 2, 3],
    feature_shifts=[
        {"type": "gaussian_noise", "features": ["age"], "magnitude": 0.2},
        {"type": "mean_shift", "features": ["age"], "magnitude": 5.0},
    ],
    output_path="results/feature_shift.csv",
)
```

The output has a `scenario` column (`clean` or `shifted`) plus `shift_type`,
`affected_feature`, `shift_magnitude`, experiment `seed`, and `shift_seed`.
Classification and reliability metric columns come from the shared
`evaluate_classifier` function. Compare each shifted row with the clean row
having the same model and seed. Larger classification scores are usually
better, while lower Brier score, ECE, MCE, adaptive ECE, and log loss are
better; deterioration direction therefore depends on the metric.

Each shift specification is applied independently to the clean test split.
Multiple features in one specification are shifted together and reported as a
comma-separated feature list. `shift_seed` is recorded separately so the
perturbation can be reproduced apart from the train/test split seed.
