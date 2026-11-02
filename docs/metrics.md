# Classification and Probability Reliability Metrics

TrustTab provides classification metrics and probability reliability metrics for binary classification tasks.

## API

Use `evaluate_classifier(y_true, y_pred, y_prob)` to calculate six metrics.

## Inputs and validation

- `y_true`: Ground-truth binary labels (`0` or `1`).
- `y_pred`: Predicted binary labels (`0` or `1`).
- `y_prob`: Predicted probabilities for positive class `1`.

Inputs must be one-dimensional, non-empty, and have matching lengths. Labels must be binary. Probabilities must be finite and between `0` and `1`. ROC-AUC requires both classes in `y_true`.

## Metrics

| Metric | Key | Description |
|---|---|---|
| Accuracy | `accuracy` | Fraction of correct predictions. |
| Balanced accuracy | `balanced_accuracy` | Average recall across both classes. |
| Macro F1 | `macro_f1` | Mean F1 score across both classes. |
| ROC-AUC | `roc_auc` | Area under the ROC curve, using positive-class probabilities. |
| Brier Score | `brier_score` | Mean squared error between probabilities and true labels. Lower is better. |
| Expected Calibration Error | `ece` | Weighted difference between confidence and observed accuracy across bins. Lower is better. |

## Expected Calibration Error (ECE)

ECE uses binary top-label confidence:

1. Predict class `1` when `y_prob >= 0.5`; otherwise predict class `0`.
2. Calculate confidence as `max(y_prob, 1 - y_prob)`.
3. Divide confidence values between `0` and `1` into 10 equal-width bins.
4. For each non-empty bin, calculate the absolute difference between average confidence and the fraction of correct predictions.
5. Weight each difference by the fraction of all samples in that bin, then sum the results.

The default number of bins is 10. The function `expected_calibration_error(y_true, y_prob, n_bins=10)` supports a different positive integer number of bins. Confidence `1.0` belongs to the final bin. Empty bins contribute zero. Lower ECE indicates better calibration.

## Example

```python
from trusttab.evaluation import evaluate_classifier

y_true = [0, 0, 1, 1]
y_pred = [0, 1, 1, 1]
y_prob = [0.1, 0.4, 0.7, 0.9]

result = evaluate_classifier(y_true, y_pred, y_prob)
print(result)
```