# TrustTab Core Experiment: Results Summary

Run ID: `trusttab-e18ead6a25c341549ba41ce12d7f5ef8`  
Configuration: `configs/experiments.yaml`  
Raw results: [`../raw_results.csv`](../raw_results.csv)  
Statistical tables and figures: this directory.

## Integrity and scope

The completed run contains 200 unique label-noise conditions (2 datasets × 4 models × 5 noise levels × 5 seeds) and 40 unique clean-baseline reference rows. The scheduler manifest reports all 200 conditions succeeded, no failed, missing, unexpected, or duplicate condition IDs, and complete integrity. All six required raw metrics are finite in every row. The 200 saved corruption counts were replayed from the recorded split/noise seeds and match the independent Bernoulli protocol. Test-set sizes are 200 for German Credit and 61 for Cleveland. Core calibration was disabled by configuration; calibrated metrics are unavailable by design.

These are results from five seeds on two fixed datasets. Confidence intervals and p-values should be read as small-sample exploratory estimates, not broad population claims. Results are not a substitute for independent external replication.

## Headline descriptive means

Means below pool the four models and five seeds within each fixed dataset/noise condition. This is a compact description, not an inferential unit.

| Dataset | Noise | Accuracy | Macro-F1 | ROC-AUC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| German Credit | 0% | 0.7688 | 0.6952 | 0.8108 | 0.0576 | 0.1574 |
| German Credit | 30% | 0.6930 | 0.5928 | 0.6977 | 0.0862 | 0.2075 |
| German Credit | 40% | 0.6340 | 0.5352 | 0.5720 | 0.0919 | 0.2379 |
| Cleveland | 0% | 0.8270 | 0.8240 | 0.9004 | 0.0901 | 0.1271 |
| Cleveland | 30% | 0.7123 | 0.7105 | 0.7925 | 0.1445 | 0.2020 |
| Cleveland | 40% | 0.6139 | 0.6030 | 0.6705 | 0.1091 | 0.2422 |

Across these means, performance metrics worsen with noise on both datasets. Brier score worsens monotonically. ECE rises through 40% on German Credit; on Cleveland it peaks at 30% and is lower at 40%, illustrating that ECE need not change monotonically and is sensitive to the small test sample and binning convention. Metric scales differ, so these raw changes do not establish equal or unequal degradation rates; the normalized per-seed slope analyses are in `degradation_slope_results.csv` and `degradation_slope_comparisons.csv`.

## Research questions

### RQ0 — Do reliability and predictive performance degrade together?

Descriptively, accuracy, Macro-F1 and ROC-AUC decline as noise rises on both datasets. Brier score worsens from 0% to 40% on both; ECE increases from 0% to 40% on German Credit but peaks at 30% on Cleveland. In the normalized per-seed slope contrasts, no performance-versus-reliability comparison rejects after the planned Holm correction (`degradation_slope_comparisons.csv`). With five seeds, this is insufficient evidence to claim equal rates or a reliable decoupling; it does not prove equality. The observable trajectories vary by dataset and metric.

### RQ1 — Does TabPFN resist calibration degradation as well as performance degradation?

On German Credit, TabPFN's mean accuracy falls by 0.077 from 0% to 40%, the smallest accuracy drop among the four models, while its ECE increases by 0.1161 and Brier worsens by 0.0800. On Cleveland, TabPFN accuracy falls by 0.2000 and ROC-AUC by 0.1390; ECE increases by 0.0470 and Brier worsens by 0.1188. These are descriptive endpoint changes across five seeds. They suggest that a smaller accuracy loss does not guarantee stable calibration, but the present data do not support a general TabPFN resistance claim.

### RQ2 — Do accuracy and calibration model rankings differ?

Yes, the mean rankings are not consistently aligned. Spearman correlations between four-model accuracy ranks and lower-is-better ECE ranks vary across dataset/noise conditions from −0.8 to 0.4 (with 0.0 at German Credit baseline and 0.8 at Cleveland baseline). Accuracy versus Brier rank correlations are also condition-dependent. With only four models, these rank correlations are descriptive. The per-condition Friedman/Nemenyi tables and critical-difference diagrams provide the planned matched-seed comparisons; they do not establish one universal ranking.

### RQ3 — Are patterns consistent across the two datasets?

Both datasets show substantial performance and Brier deterioration at higher noise, but the endpoint magnitudes and ECE trajectory differ. The Cleveland test partition has 61 observations versus 200 for German Credit, making its estimates coarser and more variable. These two fixed datasets provide a cross-dataset comparison only; they are not a sample from which to infer a population effect across tabular datasets.

### RQ4 — Does Platt calibration narrow the accuracy–calibration gap?

This core matrix cannot answer RQ4: `calibration.enabled` is false, and all 200 noise rows report calibration disabled with no calibrated metrics. The previously completed bounded calibration smoke is implementation evidence only, not the planned five-seed, all-noise comparative experiment. RQ4 remains outstanding and requires a separately authorized calibration-rescue matrix before any empirical conclusion.

## Inferential results and limits

The Phase 7 Friedman/Nemenyi analysis compares models within each dataset, noise level and metric using the five seeds as blocks. Holm-adjusted Friedman results flag some model differences for German Credit ROC-AUC at 30%, German Credit ECE at 20% and 40%, Cleveland ROC-AUC at 10%, 30% and 40%, and Cleveland Brier at 20% and 40%. Nemenyi post-hoc comparisons are limited to significant omnibus cases. These are model-comparison findings at fixed noise levels; they are not tests that noise caused within-model degradation. Five blocks yield low power and coarse exact-test p-values.

No slope-contrast family rejects after Holm adjustment. Calibration Wilcoxon comparisons are unavailable because the core run contains no calibrated pairs. The earlier report used Student-t intervals. The finalized statistical outputs now use deterministic percentile bootstrap intervals resampling the five seed-level units (10,000 resamples). These intervals are small-sample descriptive estimates, not hypothesis tests; they do not create additional independent observations. Do not interpret non-significance as equivalence.

## Reproducibility

The run manifest records normalized configuration and hash, dataset fingerprints, package versions, condition statuses, and run ID. Each raw row includes dataset/model/noise/seed, derived split/noise/model seeds where applicable, corruption count, model parameters, sample counts, training duration and completion timestamp. See `run_manifest.json` and `raw_results.csv` in the parent run directory.


## Final reporting addendum (2026-11-14)

The companion `report.md` and CSV/PNG artifacts in this directory were regenerated from the preserved `../raw_results.csv`; the experiment was not rerun and the raw file was not changed. Descriptive 95% intervals use deterministic percentile bootstrap resampling over the five seeds. The 648 paired t-test rows are compatibility/supplementary exploratory analyses, explicitly labeled as such; primary planned model inference remains seed-blocked Friedman with gated Nemenyi post-hoc comparisons. Holm correction is applied within its defined test families, not pooled indiscriminately across datasets, metrics, and noise levels.

Dedicated accuracy, ECE, Brier, and Macro-F1 noise curves, model-ranking heatmap, cross-dataset comparison, and eligible critical-difference diagrams are included. Reliability diagrams and confidence histograms cannot be reconstructed because per-observation probabilities and labels were not retained. No raw-versus-calibrated comparison figure is generated because calibration was disabled. RQ4 remains unanswered by the core experiment.

The summary field `best_calibrated_model` is reported as `unavailable` when no calibrated results exist; raw reliability metrics are not misrepresented as calibrated performance.
