# Research statistical analysis

`trusttab.statistics.run_research_analysis(results)` consumes existing unified
benchmark rows; it does not train models or create experiment outcomes. It
filters to `experiment_type == "label_noise"`, so the separately emitted clean
baseline is not double-counted as an extra zero-noise repetition.

## Pairing and omnibus model comparisons

For every fixed dataset, noise level, and metric, the four models are ranked
within each of the five matched seeds and tested with the tie-corrected
Friedman statistic. The block is the seed; datasets and noise levels are never
treated as repeated blocks of one pooled analysis. A comparison is run only
with all five complete model-by-seed values. Missing seeds and incomplete
blocks are recorded; duplicates are rejected. Constant all-model ties produce
statistic 0 and p=1. Other tied ranks use average ranks and the standard tie
correction. The output includes n, degrees of freedom, statistic, p-value,
average ranks, and Kendall's W.

Friedman omnibus p-values are Holm-adjusted within each dataset/metric across
the five planned noise levels. Nemenyi all-pairs testing runs only after a
significant Holm-adjusted omnibus result. It reports all six pairwise average
rank differences, studentized-range adjusted p-values, rejection decisions,
and critical difference. Nemenyi's own pairwise adjustment is not Holm
corrected again. Critical-difference diagrams are emitted only for eligible
post-hoc groups.

The chi-square approximation used by the Friedman test is especially weak at
five blocks. SciPy cautions that its p-value is unreliable for small block
counts (its guide recommends more than ten blocks and more than six groups).
These results are therefore exploratory, and non-significance is not evidence
of equivalence. The requested five-seed/four-model design does not meet that
large-sample recommendation. See the [SciPy Friedman documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).

## Calibration comparisons

ECE and Brier are compared between raw and Platt-calibrated results for the
same dataset/model/noise/seed. The paired difference is `raw - calibrated`,
so positive values mean calibration reduced the error. Only successful,
finite calibrated rows form pairs. Failed or missing rows are counted as
excluded; fewer than the five planned pairs yields descriptive means and
intervals, with no inferential p-value or effect size.

The two-sided Wilcoxon signed-rank statistic and p-value use the five seeds as
the independent paired units, not individual test observations. The effect
size is rank-biserial correlation on nonzero paired differences. Ties use
average absolute ranks and zero differences are omitted from the signed-rank
statistic/effect denominator; all-zero comparisons return statistic 0, p=1,
and effect 0. With five nonzero pairs, the smallest attainable exact two-sided
Wilcoxon p-value is 0.0625. Thus this design cannot reach an unadjusted 0.05
threshold for a conventional exact two-sided test, and Holm adjustment is
more conservative still.

Calibration Holm families are separate for each dataset and metric, across
the planned model-by-noise comparisons. These families are separate from the
Friedman families and from H1 slope families. When a planned comparison is
unavailable, it remains missing in the output but is conservatively treated as
`p=1` only when determining the Holm family size/adjustment of available
tests; this prevents missing runs from shrinking the predeclared family.

## H1 degradation slopes

For every dataset/model/seed and each required metric, ordinary least-squares
slope is calculated across the five configured noise levels. Accuracy,
balanced accuracy, macro-F1, and ROC-AUC are higher-is-better; ECE and Brier
are lower-is-better. All six metrics have theoretical range [0, 1]. The
normalized worsening per 10 percentage-point noise increase is:

```text
direction = -1 for higher-is-better metrics, +1 for lower-is-better metrics
normalized_degradation = direction * raw_slope * 0.10 / (upper_bound - lower_bound)
```

Positive means deterioration. Range normalization avoids dividing by a
possibly near-zero clean baseline; it does not mean that equal normalized
changes in accuracy and ECE have equal scientific or practical value. H1's
primary paired contrast is accuracy versus ECE. Accuracy versus Brier and
macro-F1 versus each reliability metric are also reported as secondary
contrasts, reflecting the blueprint's accuracy/F1 hypothesis. Other metric
slopes are available descriptively. Wilcoxon tests compare per-seed
reliability-minus-performance normalized slopes within a fixed dataset/model.
Holm is applied within each dataset across the model-by-performance-by-
reliability H1 contrast family.

## Effects, uncertainty, and limits

Wilcoxon comparisons report rank-biserial correlation. Friedman reports
Kendall's W, and Nemenyi reports average rank differences. Descriptive means
include seed-level sample SD and two-sided 95% percentile-bootstrap confidence
intervals when at least two finite seed values exist. Bootstrap resampling uses
the seed-level result as its unit (not individual test observations), 10,000
resamples, and a deterministic RNG seed derived from the comparison keys. The
seed, resample count, and interval method are recorded in the output. With only
five seeds, percentile intervals are discrete and unstable; they are
small-sample descriptive estimates, not guarantees of precise coverage.
Singletons have undefined intervals. Missing and undefined values remain
missing.

The report retains the compatibility `statistical_comparisons.csv`, which
contains supplementary paired t-tests with Holm correction. These tests are
not specified as the primary model-comparison method in Part M, and their
normality assumption is hard to assess at five pairs. Treat them as exploratory
only; primary multi-model inference is the seed-blocked Friedman/Nemenyi
analysis above. Their paired-difference intervals use the same deterministic
seed-level bootstrap convention.

The two benchmark datasets are fixed case studies, not a random sample of
datasets. Do not generalize an across-dataset p-value or confidence interval
to a dataset population. Statistical outputs summarize synthetic fixtures in
tests only; no empirical research conclusion is established until the actual
experiment results are analyzed.

## Output tables

The report writes `friedman_results.csv`, `nemenyi_results.csv`,
`wilcoxon_calibration_results.csv`, `degradation_slope_results.csv`,
`degradation_slope_comparisons.csv`, `effect_sizes.csv`, and
`statistical_summary.csv`. They preserve dataset and condition identifiers,
sample counts, status, and missing inferential values. The previous
`statistical_comparisons.csv` remains for compatibility, but is now computed
separately by dataset rather than pooling datasets.
