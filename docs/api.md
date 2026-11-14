# Public Python API

TrustTab's curated convenience API is `trusttab.api`. Subpackages such as
`trusttab.data`, `trusttab.evaluation`, and `trusttab.visualization` remain
available for specialized workflows. The convenience module does not import
the optional XGBoost or TabPFN implementations until a user selects and builds
those models.

## Dataset loading

```python
from trusttab.api import load_german_credit, load_heart_disease

X_german, y_german = load_german_credit(data_home="data/raw")
X_heart, y_heart = load_heart_disease(data_home="data/raw")
```

German Credit is retrieved by OpenML ID 31. Heart Disease loads only UCI
dataset ID 45's processed Cleveland file. Both loaders return features and a
binary target. Cleveland preserves the original `num` target in
`y_heart.attrs["original_num"]`; missing feature values remain for
train-only preprocessing. `load_adult` is available as an additional dataset.
See [dataset details and attribution](datasets.md).

## Preprocessing and model construction

`build_preprocessor(X_train)` returns a scikit-learn transformer configured
from the training feature schema. Fit it only on training observations. The
normal supported route is `build_model`, which returns an unfitted sklearn
pipeline containing preprocessing and a classifier:

```python
from trusttab.api import build_model

model = build_model(
    "logistic_regression",
    features=X_train,
    random_state=7,
    model_params={"max_iter": 1000},
)
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)
class_probabilities = model.predict_proba(X_test)
```

Supported names are exposed by `SUPPORTED_MODELS`. XGBoost and TabPFN are
optional extras; selecting either without its extra raises an actionable
`ImportError`. TabPFN weight access has separate licensing and authorization
requirements; TrustTab never distributes those weights. See
[`tabpfn.md`](tabpfn.md).

## Label noise and evaluation

```python
from trusttab.api import evaluate_classifier, inject_symmetric_label_noise

noisy_y_train = inject_symmetric_label_noise(y_train, noise_rate=0.1, seed=19)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
positive_index = list(model.classes_).index(1)
metrics = evaluate_classifier(
    y_test,
    predictions,
    probabilities[:, positive_index],
)
```

Label noise uses independent per-row Bernoulli flips and returns a copy. The
realized count is random. For the binary metrics API, `y_prob` means the
probability of class label `1`; ECE is binary top-label ECE with equal-width
confidence bins. `evaluate_classifier` includes accuracy, balanced accuracy,
macro F1, ROC-AUC, Brier score, ECE, adaptive ECE, maximum calibration error,
and log loss. See [metric definitions](metrics.md).

## Calibration

`calibrate_fitted_classifier` calibrates an already-fitted estimator with
separate calibration features and labels; it does not retrain the base model.
Keep train, calibration, and test partitions disjoint, use clean calibration
labels, and evaluate once on untouched test labels:

```python
from trusttab.api import calibrate_fitted_classifier

calibrated = calibrate_fitted_classifier(
    model,
    X_calibration,
    y_calibration,
    method="sigmoid",
)
calibrated_probabilities = calibrated.predict_proba(X_test)
```

## Experiments and reports

`run_stress_test` is the simpler label-noise runner.
`run_robustness_benchmark` supports multiple datasets/models, label noise,
feature shifts, and optional Platt calibration. It returns a DataFrame and can
write CSV output. Provide `dataset_data_home` as a path or a mapping from
dataset name to path when loader caching is needed.

`load_experiment_config` validates YAML, and `expand_experiment_conditions`
enumerates conditions without fitting models. `generate_robustness_report`
creates CSV, Markdown, and PNG artifacts from saved unified benchmark rows;
it does not rerun training. Statistical and plotting helpers are also
available through `trusttab.api` and their specialist subpackages.

## Stability

Names listed in `trusttab.api.__all__` are the curated convenience surface.
The project is currently version `0.1.0`; public API additions and breaking
changes will be documented in the changelog. Advanced implementation details
not exported by the convenience API may change between releases.
