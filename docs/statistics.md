# Statistical Model Comparison

TrustTab's stress-test runner records a score for each model, noise level, and
random seed. Statistical comparisons should preserve those shared experimental
conditions: two scores from the same seed/noise configuration form a pair.
Pairing controls for variation shared by the runs and usually gives a more
informative comparison than treating the scores as independent samples.

## Paired tests

`compare_paired_scores(scores_a, scores_b)` runs a two-sided paired t-test and
returns the mean difference and its confidence interval. A positive difference
means `scores_a` is larger. It assumes independent pairs and approximately
normally distributed pairwise differences. At least two complete pairs are
required.

`compare_paired_scores_wilcoxon(scores_a, scores_b)` runs a two-sided Wilcoxon
signed-rank test. It is useful when paired differences are not well modeled by
a normal distribution. It ranks non-zero differences and assumes independent
pairs with a distribution symmetric about zero. Identical pairs return a
non-significant result (`p_value=1`).

```python
from trusttab.statistics import (
    compare_paired_scores,
    compare_paired_scores_wilcoxon,
)

model_a = [0.82, 0.79, 0.84, 0.80]
model_b = [0.78, 0.77, 0.81, 0.76]

ttest = compare_paired_scores(model_a, model_b)
print(ttest.mean_difference, ttest.confidence_interval, ttest.p_value)

rank_test = compare_paired_scores_wilcoxon(model_a, model_b)
print(rank_test.mean_difference, rank_test.p_value)
```

## Multiple-comparison correction

Testing many model pairs and noise levels increases the chance of false
positives. `adjust_p_values` returns adjusted p-values and decisions at the
chosen `alpha`:

- **Bonferroni** controls family-wise error and is the most conservative.
- **Holm-Bonferroni** also controls family-wise error and is generally more
  powerful than standard Bonferroni.
- **Benjamini-Hochberg** (`fdr_bh`) controls the expected false discovery rate,
  which can be useful for exploratory families of comparisons.

Reject a null hypothesis when its adjusted p-value is at most alpha. The
selected procedure and alpha define the family-level error guarantee; an
adjusted p-value is not the probability that the null hypothesis is true.

```python
from trusttab.statistics import adjust_p_values

adjustment = adjust_p_values([0.01, 0.04, 0.20], method="holm", alpha=0.05)
print(adjustment.adjusted_p_values)
print(adjustment.reject_null)
```

## Comparing experiment results

`compare_models` takes the DataFrame returned by `run_stress_test` and compares
all model pairs separately at each noise level. It matches rows by seed and,
when available, dataset. By default, any unmatched configuration raises an
error. Use `unmatched="drop"` to explicitly compare only shared configurations;
the output reports unmatched counts so those rows remain visible. The
`mean_difference` column is `model_a - model_b`, so its sign follows the pair
order in the output. `correction` adjusts the valid p-values together across
all pair/noise comparisons returned by the call.

```python
from trusttab.experiments import run_stress_test
from trusttab.statistics import compare_models

results = run_stress_test(seeds=[1, 2, 3, 4], noise_levels=[0.0, 0.2, 0.4])
comparisons = compare_models(
    results,
    metric="adaptive_ece",
    test="wilcoxon",
    unmatched="raise",
    correction="holm",
    alpha=0.05,
)
print(comparisons[[
    "model_a", "model_b", "noise_level", "n_paired",
    "mean_difference", "p_value", "adjusted_p_value", "reject_null",
]])
```

`analyze_noise_levels` returns model/noise means and standard deviations. Pass
`reference_model` to add paired test results for each other model against that
reference at the same noise levels. Without a reference, test columns are
empty and `n_paired` reports the number of available observations.

```python
from trusttab.statistics import analyze_noise_levels

summary = analyze_noise_levels(
    results, metric="brier_score", reference_model="logistic_regression"
)
print(summary)
```

For metric interpretation, larger values are better for scores such as
accuracy, while smaller values are better for loss/error measures such as
ECE, Brier score, and log loss. Statistical tests report the numerical
`model_a - model_b` direction; they do not infer which direction is desirable.
