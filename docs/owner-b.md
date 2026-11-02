# Owner B — Evaluation, Reliability, and Analysis

## Overview

This document describes the evaluation, calibration, visualization, statistical comparison, and operating-region utilities implemented for TrustTab.

## B0 — Classification and Reliability Metrics

TrustTab provides utilities for evaluating binary classification models and their predicted probabilities.

Key capabilities:
- Classification performance metrics.
- Expected Calibration Error (ECE) for assessing probability calibration.
- Input validation for evaluation data and predicted probabilities.

These metrics help assess both predictive performance and the reliability of model confidence estimates.

See [metrics.md](metrics.md) for more details.

## B1 — TabPFN Integration

TrustTab includes optional TabPFN model integration through a model adapter.

The integration allows TabPFN to be used through the project's model interface when the optional dependency is available. Keeping this integration optional allows users to work with other supported models without requiring TabPFN in every environment.

## B2 — Probability Calibration

TrustTab provides probability calibration utilities to help improve the agreement between predicted probabilities and observed outcomes.

Calibration is useful when probability estimates are used for decision-making, risk assessment, or confidence-based analysis.

Calibration should be evaluated on appropriate held-out data to avoid data leakage.

## B3 — Reliability Visualizations

TrustTab includes visualization utilities for examining probability reliability.

Reliability visualizations help users compare predicted confidence with observed outcomes and identify areas where a model may be overconfident or underconfident.

See [visualization.md](visualization.md) for usage details.

## B4 — Statistical Comparisons

TrustTab provides paired statistical comparison utilities for comparing model scores on corresponding observations or evaluation runs.

The comparison utilities include:
- Paired statistical testing.
- Estimated differences between paired scores.
- Confidence intervals for the difference.

Paired comparisons should use scores obtained from appropriately matched evaluation cases. Statistical results should be interpreted alongside effect size, uncertainty, and the evaluation design.

## B5 — Operating-Region Analysis

TrustTab provides operating-region analysis for binary classification models.

The `analyze_operating_region` function evaluates model performance across decision thresholds.

For each threshold, it reports:
- **Accuracy:** Fraction of predictions that are correct.
- **Precision:** Fraction of predicted positives that are actual positives.
- **Recall:** Fraction of actual positives that are identified.
- **F1 score:** Harmonic mean of precision and recall.
- **Predicted-positive rate:** Fraction of observations predicted as positive.

By default, the analysis evaluates 21 thresholds evenly spaced between 0 and 1. Users can also supply a custom list of thresholds.

### Example

```python
import numpy as np

from trusttab.evaluation import analyze_operating_region

y_true = np.array([0, 0, 1, 1, 1])
y_prob = np.array([0.10, 0.35, 0.60, 0.80, 0.95])

results = analyze_operating_region(y_true, y_prob)

for point in results:
    print(
        point.threshold,
        point.accuracy,
        point.precision,
        point.recall,
        point.f1,
        point.predicted_positive_rate,
    )
```

A custom threshold list can be passed through the `thresholds` argument.

The operating threshold should be selected according to the application's objectives and the relative costs of false positives and false negatives. The threshold with the highest accuracy is not necessarily the most appropriate choice.

## B6 — Documentation

This document summarizes Owner B's evaluation and analysis utilities and provides an entry point for users exploring the relevant functionality.

The accompanying metrics and visualization guides provide additional information about those features.

## Testing

The evaluation and analysis utilities are covered by automated tests. Run the test suite from the repository root:

```bash
pytest -q
```

To check code style, run:

```bash
ruff check .
```

Style-check results may include pre-existing issues in unrelated files. Review those separately from changes made for a particular feature.