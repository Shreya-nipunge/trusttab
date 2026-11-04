# TrustTab

**Reliability-Aware Stress Testing for Tabular Machine Learning**

TrustTab is an open-source research framework for evaluating the predictive performance and probability reliability of tabular machine learning models under progressively corrupted training data.

## Research Study

**Accuracy Isn't Reliability: Evaluating Tabular Foundation Models Under Progressive Label Noise**

## Research Objective

This project investigates whether predictive performance and probability reliability deteriorate at the same rate when training-label noise progressively increases.

## Models

- Logistic Regression
- Random Forest
- XGBoost
- TabPFN

## Evaluation

### Predictive Performance
- Accuracy
- Balanced Accuracy
- Macro F1
- ROC-AUC

### Reliability
- Expected Calibration Error (ECE)
- Brier Score
- Calibration Curves

## Status

🚧 Research and development in progress.
## Unified robustness benchmark

TrustTab can compare clean baselines with label-noise and feature-shift
scenarios in one CSV:

```python
from trusttab.experiments import run_robustness_benchmark

results = run_robustness_benchmark(
    label_noise_levels=[0.0, 0.2],
    feature_shifts={
        "enabled": True,
        "types": ["mean_shift"],
        "magnitudes": [0.5, 1.0],
        "features": ["age"],
    },
    seeds=[1, 2, 3],
    output_path="results/unified_benchmark.csv",
)
```

See [the unified benchmarking guide](docs/unified-benchmarking.md) for YAML
configuration, result columns, and baseline interpretation.

## Automated robustness reports

Generate Markdown analysis, CSV summaries, statistical comparisons, and figures
from unified benchmark results:

```python
from trusttab.reporting import generate_robustness_report

report = generate_robustness_report(results, "results/robustness_report")
```

See [the reporting guide](docs/robustness-reporting.md) for expected columns,
artifact details, and metric interpretation.
