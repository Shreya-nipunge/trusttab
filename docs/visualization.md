# Reliability Visualizations

TrustTab provides plotting utilities for inspecting classifier probabilities.

## Reliability diagram

Compare predicted probabilities with observed positive frequencies:

```python
from trusttab.visualization import plot_reliability_diagram

ax = plot_reliability_diagram(y_true, y_prob, n_bins=10)
ax.figure.savefig("reliability.png", bbox_inches="tight")
```

Use `strategy="uniform"` for equal-width bins or `strategy="quantile"` for approximately equal-sized bins.

## Probability distribution

Inspect the distribution of predicted positive-class probabilities:

```python
from trusttab.visualization import plot_probability_distribution

ax = plot_probability_distribution(y_prob, bins=10)
```

## ROC curve

Plot the true-positive rate against the false-positive rate:

```python
from trusttab.visualization import plot_roc_curve

ax = plot_roc_curve(y_true, y_prob)
```

The ROC plot requires both binary classes in `y_true`.

## Saving figures

Save a figure and create its parent directories automatically:

```python
from trusttab.visualization import save_figure

save_figure(ax.figure, "reports/reliability.png", dpi=150)
```