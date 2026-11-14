# Execution Blueprint
## "Accuracy Isn't Reliability: Evaluating Tabular Foundation Models Under Progressive Label Noise"

---

## EXECUTIVE SUMMARY (read this first)

**The research question, tightened:** *As training-label noise increases from 0% to 30%, does model reliability (calibration) degrade at the same rate as predictive performance (accuracy/F1/AUC) — or do they decouple, and in which direction?*

**Verdict on the topic:** This is a sound, defensible, undergraduate-appropriate research question. It is not fully novel (calibration-under-noise has literature), but the *specific comparison* — a 2025 tabular foundation model (TabPFN) vs. classical ML, tracked across both performance **and** calibration **simultaneously** as noise increases — is a legitimate, verifiable **empirical extension**, not a reproduction. Full classification in Part C.

**One critical, load-bearing constraint discovered during this research that changes your plan:** TabPFN is **only practically CPU-fast up to ~1,000 rows**; beyond that it is officially described as "very slow" on CPU without a special flag, and recent TabPFN versions (2.5+) require creating a free account and license token. This single fact determines your dataset choice (Part E), your compute budget (Part G), and your top risk (Part Y). **German Credit (1,000 rows) becomes your primary dataset, not Adult Income.**

**Recommended final scope (full reasoning in Part Z):**
- **Models:** Logistic Regression, Random Forest, XGBoost, TabPFN (v2, pinned version — see Part I)
- **Noise levels:** 0%, 10%, 20%, 30%, 40% (revised from your 0/5/10/20/30 — reasoning in Part F)
- **Noise mechanism:** symmetric random label flipping on the **training set only**, test set always clean
- **Primary dataset:** German Credit (1,000 rows, UCI/OpenML #31) — fits TabPFN's CPU envelope exactly
- **Secondary dataset:** Heart Disease (Cleveland, ~297 rows) — small enough to run fast, used to show your finding isn't a one-dataset fluke
- **Core metrics:** Accuracy, F1 (macro), ROC-AUC for performance; ECE and Brier Score for reliability
- **Conformal prediction: OPTIONAL, not core** — reasoning in Part L. Include only if Weeks 1–4 go smoothly.
- **Statistics:** Friedman test + Nemenyi post-hoc across models per noise level; Wilcoxon signed-rank for paired before/after comparisons
- **Timeline:** 7 weeks (Part V)
- **Total experiment runs:** 4 models × 5 noise levels × 2 datasets × 5 seeds = **200 runs**, each taking seconds to at most a few minutes (TabPFN is the slow one) — realistic on your laptop in a few hours of wall-clock time, run overnight if needed. Full compute analysis in Part G.

If you read nothing else, read Part Z (Final Recommended Scope) and the "START HERE" checklist at the very end.

---

# PART A — Is This a Good Research Question? (Critical Evaluation)

**1. What exactly is the research problem?**
Model evaluation in most undergraduate (and a surprising amount of applied) ML work conflates "the model is right most of the time" (accuracy) with "the model knows when it might be wrong" (calibration). These are different properties. A model can hold steady accuracy while its predicted probabilities become meaningless — i.e., it says "95% confident" and is wrong 40% of the time. Your project tests whether label noise is a condition under which this gap opens up.

**2. Why is "accuracy ≠ reliability" important?** Because in any setting where a predicted probability is *used* for something — triage, lending decisions, flagging for human review, setting a threshold — a miscalibrated-but-accurate model silently gives wrong confidence even when its yes/no answer is still often right. This is a real, well-established concern in the calibration literature (Guo et al. 2017, *"On Calibration of Modern Neural Networks"*, is the canonical starting reference, though it is about deep nets, not tabular models — a gap you're partly addressing).

**3. Role of label noise:** Label noise is one of the most realistic data-quality problems in real tabular datasets (human annotation error, proxy labels, measurement error) and is a controllable variable — you can inject it precisely, which is what makes it good for controlled experimentation (unlike, say, "real-world messiness" which can't be dialed up and down).

**4. Why is *progressive* label noise meaningful?** A single noise level only tells you "is the model robust at X%." A progression lets you find the *rate* and *shape* of degradation (linear? threshold effect? one metric breaks before another?) — this is what turns the project from "point comparison" into "an actual degradation curve," which is the core of your research design.

**5. Why TabPFN vs classical models specifically?** Because TabPFN is trained once on millions of synthetic datasets and then performs inference via in-context learning — it does not gradient-descend on *your* corrupted labels the way Logistic Regression/RF/XGBoost do. This is a *mechanistically different* reason it might resist (or fail to resist) label noise, which gives your comparison an actual explanatory hook, not just "which number is bigger."

**6. What makes this different from a simple model comparison?** A simple comparison asks "which model wins." You are asking "do performance and reliability rank models the *same way*, and does that ranking *change* as noise increases." That's a comparison of **two parallel degradation curves**, with the interesting result being *divergence* between them — not a single winner.

**7. What is the actual research contribution?** A systematic, multi-seed, multi-dataset empirical answer to: does a 2025 tabular foundation model's known noise-resistance (in accuracy/AUC terms, per very recent literature — Part C) extend to calibration, or is it an accuracy-only illusion? That is a concrete, checkable, previously-not-directly-answered question.

**8. Reproduction, replication, extension, or novel?** **Empirical extension.** You are not inventing a new method (not novel). You are not simply re-running a published experiment on the same data with the same metrics (not reproduction/replication). You are taking an established finding (TabPFN resists noise, measured via accuracy/AUC — see Part C) and testing whether it holds under a dependent variable (calibration) that the original work didn't measure. This is a textbook definition of empirical extension, and it is a completely legitimate undergraduate research contribution — **say so explicitly in your report**, don't oversell it as "novel."

**9. What would make this scientifically weak?** Three failure modes to actively avoid: (a) reporting only accuracy curves and bolting on ECE as an afterthought — the whole point is *parallel, equally-rigorous* treatment of both; (b) testing only one dataset — a single-dataset finding could just be dataset idiosyncrasy, which is why you need the secondary dataset; (c) reporting a single run per condition — noise injection has randomness, and without multiple seeds you cannot tell a real effect from noise in your noise.

**10. What findings would actually be interesting?** Any of: (a) accuracy holds but ECE/Brier worsens — "the model lies about its confidence before it starts getting things wrong"; (b) TabPFN's accuracy advantage over GBDTs shrinks with noise, but its calibration advantage grows (or vice versa) — a genuine crossover; (c) all models degrade proportionally — a legitimate "null-ish" finding, still worth reporting honestly, and you should be prepared for this outcome (see Part P).

---

# PART B — Research Questions and Hypotheses

### Primary Research Question
**RQ0:** As training-label noise increases (0–40%), does model reliability (ECE, Brier Score) deteriorate at the same rate as predictive performance (Accuracy, F1, ROC-AUC), or do the two decouple?

### Secondary Research Questions
- **RQ1:** Does TabPFN's resistance to accuracy degradation under label noise (suggested by prior work) extend to resistance in calibration degradation?
- **RQ2:** Is there a noise level at which the *model ranking* by accuracy differs from the model ranking by calibration?
- **RQ3:** Does the direction/magnitude of the accuracy–calibration gap hold consistently across two different datasets, or is it dataset-specific?
- **RQ4:** Does applying a simple post-hoc calibration method (Platt scaling) at each noise level narrow the accuracy–calibration gap, and does it narrow it equally for all models?

### Hypotheses

| Hypothesis | Independent Variable | Dependent Variable | Expected Effect | Test |
|---|---|---|---|---|
| H1 | Label noise level (0–40%) | ECE, Brier Score | Reliability metrics worsen **faster** (steeper slope) than accuracy/F1 as noise increases | Compare normalized degradation slopes (performance metric vs. reliability metric) per model, per dataset; Wilcoxon signed-rank on slope differences across seeds |
| H2 | Model family (TabPFN vs. classical) × noise level | Accuracy, ECE | TabPFN shows smaller accuracy drop than GBDT/RF/LR at high noise, but **not necessarily** a smaller ECE increase | Friedman test across models at each noise level for accuracy; separately for ECE; compare whether the best-ranked model differs between the two |
| H3 | Noise level | Model ranking stability | The best model by accuracy at 0% noise is not necessarily the best model by ECE at 30–40% noise | Spearman correlation between accuracy-based and ECE-based model rankings, computed separately at each noise level |
| H4 | Calibration method (none vs. Platt scaling) × noise level | ECE | Platt scaling reduces ECE at all noise levels, but the *size* of the reduction shrinks as noise increases (there's less a post-hoc fix can do when the underlying probabilities are badly corrupted) | Paired Wilcoxon test, raw ECE vs. Platt-calibrated ECE, at each noise level |

Reject H1/H2/H3/H4 as stated if the data shows metrics moving together / no ranking change / no reduction — **these are legitimate possible outcomes and must be reported as such**, not reframed to appear confirmed.

---

# PART C — Literature Review

| Paper | Year | Problem | Method | Dataset | Key Finding | Relevance |
|---|---|---|---|---|---|---|
| [Hollmann et al., *Accurate predictions on small data with a tabular foundation model*](https://www.nature.com/articles/s41586-024-08328-6), *Nature* | 2025 | Introduces TabPFN v2 | Transformer trained on synthetic priors, in-context inference | Many small tabular benchmarks | TabPFN matches/beats tuned GBDTs on small tabular data without per-dataset training | **Your core model** — the foundational paper you build on |
| [Assessing the Robustness of Tabular Prior-Data Fitted Network Classifier](https://icml.cc/virtual/2025/47490), ICML workshop | 2025 | TabPFN robustness to label noise | TabPFN vs. 7 classical models, 0–30% noise, 15 OpenML datasets, accuracy + AUC only | TabPFN "consistently exhibits superior resilience to noise" in accuracy/AUC; boosting models most noise-sensitive | **Closest prior work.** Confirms the accuracy-side finding you're extending — but this paper does **not** measure calibration/ECE/Brier at all, which is exactly your gap |
| [Hu & Ghelichi, *Noise Immunity in In-Context Tabular Learning*](https://arxiv.org/pdf/2604.04868) | 2026 | Why TabPFN resists label noise | Attention-weight analysis under 0–35% label noise | ROC-AUC does not degrade; attention structure stays stable even at 35% noise | Reinforces the accuracy-robustness claim via a different mechanism; still no calibration analysis |
| [Eisenbürger, Otten, Hudde, Hopfgartner, *Training GBDTs on Tabular Data Containing Label Noise*](https://arxiv.org/html/2409.08647v2) | 2024–25 | GBDT behavior under label noise | Noise detection + GBDT training dynamics | GBDTs show "natural robustness" especially to symmetric noise, with early stopping helping | Establishes the classical-model (GBDT) reference behavior your project compares TabPFN against |
| Guo, Pleiss, Sun, Weinberger, *On Calibration of Modern Neural Networks*, ICML | 2017 | Why modern models are often overconfident | Measures ECE across architectures | Image classification benchmarks | Established ECE as the standard calibration metric; showed accuracy and calibration can move in opposite directions as models improve | Canonical calibration reference — cite as the origin of the "accuracy ≠ calibration" framing, but note it is about deep nets/images, not tabular/noise, which is precisely where your gap sits |
| [Benchmarking machine learning methods for the identification of mislabeled data](https://link.springer.com/article/10.1007/s10462-025-11293-9), *Artificial Intelligence Review* | 2025 | Survey + benchmark of 34/20 noise-detection methods | Various filters on tabular + genomic data | Noise-filtering methods peak in precision/recall around 20–30% noise | Useful for justifying your 0–40% noise range and as a related-work anchor; not directly about calibration |
| [A Perspective on Explainable AI Methods: SHAP and LIME](https://arxiv.org/abs/2305.02012) | 2023 | Model-dependency of XAI methods | Comparative case study | Biomedical binary classification | Not directly relevant to your project — included only if your Related Work section discusses why you did *not* pursue an XAI-stability framing instead | Background/contrast only |

### What has already been studied?
- TabPFN's **accuracy/AUC robustness** to label noise (ICML workshop paper, attention-analysis paper).
- GBDT behavior under label noise generally (well-established).
- Calibration as a concept, and that it can diverge from accuracy, in **deep learning / image classification** (Guo et al. 2017 and its large citing literature).
- Noise-detection/filtering methods as a separate sub-field.

### What has NOT been adequately studied?
- Calibration/ECE/Brier Score tracked **alongside** accuracy, specifically for **tabular foundation models under label noise**. No paper found in this search does this combination.
- Whether TabPFN's documented noise-resistance (accuracy-side) is matched by, exceeded by, or decoupled from reliability.

### Where does our project fit?
At the intersection of two established-but-separate literatures (TabPFN noise-robustness; calibration-under-degradation) that have not yet been combined for tabular foundation models specifically.

### Classification
**Empirical extension.** State this explicitly in your Introduction: *"We extend the accuracy/AUC-based noise-robustness findings of [ICML workshop paper] and [Hu & Ghelichi, 2026] by adding a calibration/reliability dimension that neither study measures."* Do not claim novelty beyond this — a grader who checks Google Scholar for 30 seconds will find the ICML workshop paper, and your report should pre-empt that by citing and differentiating from it directly, which strengthens rather than weakens your position.

---

# PART D — Research Gap

**Existing research says:** Tabular foundation models (TabPFN) resist label-noise-induced *accuracy/AUC* degradation better than classical GBDTs and tree ensembles (ICML workshop paper, 2025; Hu & Ghelichi, 2026). Calibration is known, from a separate and unrelated literature (Guo et al. 2017 and successors), to be a distinct property from accuracy that can degrade independently, particularly under distributional stress.

**Missing / insufficiently explored:** No identified study measures calibration (ECE/Brier) for TabPFN specifically under progressive label noise, nor compares the *rate* of calibration degradation to the *rate* of accuracy degradation for this model family.

**Our project investigates:** Whether TabPFN's accuracy-side noise robustness is matched by comparable calibration robustness, tested across a controlled 0–40% symmetric label-noise progression, against Logistic Regression/Random Forest/XGBoost, on two public tabular datasets with multiple seeds.

**Expected contribution:** An empirically grounded answer (not assumed in advance) to whether "TabPFN is noise-robust" should be qualified as "TabPFN is noise-robust *in accuracy terms only*" or whether the robustness is genuinely broader. This is a specific, bounded, defensible claim — not "no one has ever studied label noise," which would be false and indefensible in a viva.

---

# PART E — Dataset Selection

**Scoring (0–5 scale) across criteria. Datasets considered: Adult Income, German Credit, Bank Marketing, Heart Disease (Cleveland), Wine Quality.**

| Dataset | Rows | Binary clf. | Size (TabPFN-CPU fit) | Feature types | Class balance | Missing values | Preprocessing ease | Noise-injection suitability | Calibration suitability | Reproducibility | **Total /50** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **German Credit** | 1,000 | Yes | **5** (fits CPU envelope exactly) | Mixed cat+num | Moderate (70/30) | None | 5 | 5 | 5 | 5 | **45** |
| **Heart Disease (Cleveland)** | 297 | Yes (binarize 0 vs 1–4) | **5** (very fast even on CPU) | Mixed | Moderate | A few (handle via drop/impute) | 4 | 5 | 5 | 5 | **44** |
| Wine Quality (binarized) | ~1,599 (red) | Needs binarization | 4 (slightly over ideal CPU envelope, still workable) | Mostly numeric | Can be skewed depending on binarization threshold | None | 4 | 4 | 4 | 5 | 38 |
| Adult Income | 48,842 | Yes | **1** (needs subsampling to be CPU-feasible for TabPFN; defeats "use the real dataset size" purpose) | Mixed cat+num | Imbalanced (~24% positive) | A few ("?" values) | 3 | 3 | 3 | 5 | 28 |
| Bank Marketing | 45,211 | Yes | **1** (same subsampling problem) | Mixed | Highly imbalanced (~11% positive) | None | 3 | 3 | 3 | 5 | 28 |

**Why Adult/Bank score low here despite being "standard":** they are standard for *classical* ML benchmarking, but your specific project has TabPFN as a load-bearing model, and TabPFN is only CPU-fast to ~1,000 rows (officially documented — Part I). Running them would require subsampling to ~1,000 rows anyway, at which point you've lost the "large standard dataset" benefit while keeping all its imbalance/complexity — worse of both worlds for a 2-person team on a 7-week clock.

### Recommendation
- **Primary dataset: German Credit** (OpenML dataset ID 31 / UCI "Statlog (German Credit Data)"). 1,000 rows is TabPFN's CPU sweet spot exactly, binary classification, well-documented, zero missing values, widely used so your preprocessing choices are easy to justify and compare against prior literature.
- **Secondary (validation) dataset: Heart Disease (Cleveland)**, ~297 rows after removing the handful of missing-value rows. Confirms RQ3 (is the finding dataset-specific) without adding real compute burden — it's even smaller than German Credit.
- **Use 2 datasets, not 1 or 3.** One dataset risks "this is just a quirk of German Credit." Three datasets (adding Wine Quality) adds real engineering overhead (correct binarization is non-trivial and introduces a third preprocessing decision to justify) for limited marginal credibility gain in a 7-week, 2-person project. Two is the right trade-off: enough to claim "not dataset-specific," not so many that you spend week 6 still debugging a third pipeline.

---

# PART F — Label Noise Injection (exact mechanism)

**Mathematically:** For a binary classification label $y_i \in \{0,1\}$, symmetric label noise at rate $p$ independently flips each training label with probability $p$: $\tilde{y}_i = 1-y_i$ with probability $p$, else $\tilde{y}_i = y_i$. This is **Noisy Completely at Random (NCAR)** — the simplest, most standard, and most defensible-to-a-grader mechanism, because it has no hidden assumptions about *which* labels get corrupted.

**Why not asymmetric/class-conditional noise as your primary mechanism?** Asymmetric noise (e.g., "positive class is more likely to be mislabeled as negative than vice versa") is more realistic in some domains but adds a second free parameter (the noise transition matrix) that you'd need to justify and that complicates every downstream statistical claim. **Recommendation: use symmetric noise as your core, required mechanism; if time permits in Week 5, add one asymmetric-noise condition as a single bonus experiment (Part N, optional tier), not as a parallel full study.**

**Revised noise levels: 0%, 10%, 20%, 30%, 40%** (changed from your proposed 0/5/10/20/30). Reasoning: 5% is too close to 0% to show a visually or statistically distinguishable effect on a 1,000-row dataset (5% of 700 training rows ≈ 35 flipped labels — within normal cross-validation noise). Going to 40% gives you a cleaner, more visible degradation curve and a genuine "breakdown" regime to discuss, which matters for the "interesting finding" criterion. If your early results (Week 3) show 40% is *too* destructive (models collapse to random guessing for everyone, erasing between-model differences), scale back to 35%.

**Which split gets corrupted — this is critical and must be stated explicitly in your methodology:**

| Split | Corrupted? | Why |
|---|---|---|
| Training set | **Yes** — this is the experimental manipulation | You are testing the effect of noisy *training* labels |
| Calibration set (if used for Platt scaling / conformal) | **No** — must remain clean | Calibrating against noisy labels would measure "how well the model fits noise," not reliability |
| Test set | **No — always clean** | Evaluation must measure performance against ground truth; a noisy test set would make accuracy itself undefined/unreliable as a metric, destroying your ability to interpret any result |

This clean-test-set rule is the single most important leakage-prevention decision in the project (see also Part H) — state it explicitly and prominently in your Methodology section, because it's exactly the kind of thing a professor will ask about in viva (Part X).

**Pseudocode:**
```python
def inject_label_noise(y_train: np.ndarray, noise_rate: float, seed: int) -> np.ndarray:
    """
    Symmetric (NCAR) label noise injection for binary labels.
    Applied ONLY to the training split. Never call this on val/test labels.
    """
    rng = np.random.default_rng(seed)
    y_noisy = y_train.copy()
    n = len(y_train)
    flip_mask = rng.random(n) < noise_rate             # independent Bernoulli(p)
    flip_idx = np.flatnonzero(flip_mask)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]         # binary flip
    return y_noisy, flip_idx   # return flip_idx too, for leakage audits/debugging
```

---

# PART G — Experimental Matrix and Compute Feasibility

**Matrix:**

| Factor | Levels |
|---|---|
| Models | Logistic Regression, Random Forest, XGBoost, TabPFN |
| Noise levels | 0%, 10%, 20%, 30%, 40% |
| Datasets | German Credit, Heart Disease |
| Seeds | 5 (recommended — see below) |

**Total runs = 4 models × 5 noise levels × 2 datasets × 5 seeds = 200 runs.**

**Why 5 seeds, not fewer or more?** 3 seeds is the bare statistical minimum and makes your error bars look thin/unconvincing; 10 doubles your compute for diminishing returns on a project this size. 5 is the standard compromise in small-scale ML research papers and gives you enough spread to run Wilcoxon/Friedman tests meaningfully (these tests want ≥5 paired observations to be minimally sensible).

**Compute reality check on your hardware (Intel Core Ultra 5 125H, 24GB RAM, no dedicated GPU):**
- Logistic Regression, Random Forest, XGBoost: each run is **seconds**, even with moderate hyperparameter settings — 150 non-TabPFN runs (3 models × 5 × 2 × 5) will complete in well under an hour total.
- TabPFN: per official documentation, CPU inference is "only feasible for small datasets (≲1,000 samples)" — **German Credit (1,000 rows) and Heart Disease (297 rows) are exactly in this feasible zone**, which is precisely why they were chosen in Part E. Expect each TabPFN run to take **on the order of several seconds to low tens of seconds** on your CPU for datasets this size (official docs note CPU is "slow," not "infeasible," at ≲1,000 rows — budget conservatively). 50 TabPFN runs (1 model × 5 × 2 × 5) → realistically **15–40 minutes total**, possibly longer on first run due to model-weight download. Run it once interactively to confirm timing before scripting the full sweep.

**Verdict: 200 runs is realistic and comfortably completes in under 2 hours of wall-clock compute, even conservatively estimated — not an overnight job unless you also add conformal prediction and the asymmetric-noise bonus experiment, in which case budget an overnight run as a safety margin, not a requirement.**

---

# PART H — Preventing Experimental Leakage

| Leakage source | Risk in this project | Prevention |
|---|---|---|
| Label noise leaking into test data | High if not careful — easy mistake to inject noise before splitting | **Split first, inject noise only into the training split, always.** Enforce this in code: `inject_label_noise()` must only ever be called on `y_train`, never on `y_val`/`y_test`. Add an assertion/unit test that compares `y_test` before/after the pipeline runs and fails loudly if they differ. |
| Scaling/encoding before train/test split | Medium — `StandardScaler().fit()` on the whole dataset leaks test-set statistics into training | Fit all preprocessors (`StandardScaler`, `OneHotEncoder`, imputers) **only** on the training split; `.transform()` (never `.fit()`) on val/test |
| Calibration leakage | Medium — calibrating on the same data used for final evaluation inflates apparent calibration quality | Carve a dedicated calibration subset from the training data (e.g., 70% train / 15% calibration / 15% test), never reuse the test set for calibration fitting |
| Hyperparameter tuning on test data | Medium — tempting to "peek" at test accuracy while tuning | Use the training split (with internal CV or a held-out validation slice) for any tuning; test set is touched exactly once, at final evaluation, per condition |
| Using test data repeatedly across many design iterations | Medium — re-running on test many times while debugging pipeline = implicit overfitting to test | Debug/iterate using a small dev subset or the validation split; only run the full test evaluation once the pipeline is frozen |
| Feature selection leakage | Low (project doesn't currently plan feature selection) | If added later, fit feature-selection decisions on training data only |
| Preprocessing using the full dataset before splitting | High if not careful (common beginner mistake) | **Always split before any `.fit()` call of any kind** — this is the single rule that prevents most of the above |
| Random seed problems | Medium — reusing the same seed for data splitting AND noise injection AND model init can accidentally correlate results in misleading ways | Use **separate, explicitly named seeds** for (a) train/test split, (b) noise injection, (c) model initialization where applicable — document the seed scheme in your config file (Part S) |
| Duplicate records | Low-medium, check German Credit/Heart Disease for exact duplicate rows | Run a `df.duplicated().sum()` check during preprocessing and report the count, even if zero |

**Protocol summary (put this near-verbatim in your Methodology section):**
1. Load raw data.
2. Split into train (70%) / calibration (15%, only if calibration/conformal experiments are run) / test (15%) — **stratified by label**, fixed split seed per repetition.
3. Fit all preprocessors on train only; transform val/calibration/test.
4. Inject label noise into **train labels only**, at the condition's noise level, using the noise-injection seed for that repetition.
5. Train model on noisy train split.
6. Evaluate on **clean, untouched** test split.
7. Repeat steps 2–6 for each of the 5 seeds, independently re-splitting and re-injecting noise each time (not just re-training on the same noisy data).

---

# PART I — Model Configuration

### Logistic Regression
- **Preprocessing:** One-hot encode categorical features; standardize numeric features (`StandardScaler`) — required because LR is scale-sensitive.
- **Regularization:** L2 (default, `penalty='l2'`), as a robust, well-understood default; don't spend tuning budget hunting for marginal L1/elastic-net gains — that's not the research question.
- **Solver:** `lbfgs` (scikit-learn default, handles small-to-medium data fine).
- **Hyperparameters:** `C=1.0` as default; optionally a small grid (`C ∈ {0.1, 1, 10}`) via internal CV on the training split only — keep this light, LR isn't the star of the show.

### Random Forest
- **Key hyperparameters:** `n_estimators=300–500` (diminishing returns beyond this on datasets this size), `max_depth` left unrestricted or lightly capped (e.g., 10) to avoid memorizing noisy labels too perfectly — actually **leaving depth uncapped is scientifically more interesting here**, since you want to see natural noise-sensitivity, not hyperparameter-suppressed noise-sensitivity.
- **Class weighting:** `class_weight='balanced'` if German Credit's ~70/30 split causes any concerning minority-class behavior; report whichever choice you make and why.
- **No scaling needed** (tree-based, scale-invariant).

### XGBoost
- **Key parameters:** `n_estimators=200–300`, `max_depth=4–6`, `learning_rate=0.05–0.1`.
- **Regularization:** default `reg_lambda` is fine; don't over-tune.
- **Early stopping:** use a validation slice from the training split (not test) with `early_stopping_rounds=20` — early stopping is specifically relevant here because the GBDT label-noise literature (Eisenbürger et al., Part C) found early stopping helps noise robustness, so including/excluding it is itself a small, citable, justified methodological choice.

### TabPFN
- **Library:** official `tabpfn` package from Prior Labs (`pip install tabpfn`), GitHub: `priorlabs/tabpfn`.
- **Version — important, verify before Week 1 is over:** TabPFN has moved fast (v2 → 2.5 → 2.6 → "TabPFN-3" as the evolving default at time of writing). **Pin an exact version in `requirements.txt`** (e.g., `tabpfn==2.0.x`, check current PyPI for the exact v2 release tag matching the *Nature* paper) so your results are reproducible and don't silently change if you `pip install` again in week 6. Record the exact version and its model-weights hash/date if available in your reproducibility log (Part S).
- **License/token:** per current official documentation, **TabPFN versions 2.5 and later require creating a free account and accepting a license to obtain a `TABPFN_TOKEN`.** Do this in Week 1, not Week 4 — if there's any friction (institutional email issues, etc.), you want to discover it early. If you deliberately pin to the original v2 release (the one matching the *Nature* paper), check whether that specific version still requires a token; document whichever is true for the version you pin.
- **CPU feasibility:** officially documented as "only feasible for small datasets (≲1,000 samples)" on CPU; an environment variable (`TABPFN_ALLOW_CPU_LARGE_DATASET=true`) exists to force larger datasets through but is explicitly described as "very slow" — **you should not need this flag at all**, since both your chosen datasets are at or under 1,000 rows.
- **Preprocessing:** minimal — TabPFN handles raw numeric/categorical features reasonably well out of the box per its documentation; still apply the same categorical encoding you use for LR for consistency across models, and do **not** scale features for TabPFN (not required, and changes nothing meaningful for a transformer-based in-context model).
- **Fine-tuning:** **not needed and not recommended** — the entire point of TabPFN is zero-shot in-context inference; fine-tuning it would change what you're testing (you'd be testing a fine-tuned derivative, not the foundation model's native noise-robustness) and adds real engineering risk for no research benefit here.
- **Class limit:** TabPFN has a documented hard class-count limit (historically around 10 classes) — irrelevant for your binary classification setup, but worth knowing if you ever extend to Wine Quality's multi-class form.

---

# PART J — Evaluation Metrics

### 1. Predictive performance — what to include

| Metric | Include? | Reasoning |
|---|---|---|
| Accuracy | **Yes (core)** | Simple, universally understood, needed as the "performance" half of your core comparison |
| Balanced Accuracy | **Yes (core)** | German Credit has a moderate imbalance (~70/30); balanced accuracy prevents noise-robustness results from being an artifact of majority-class prediction |
| Precision, Recall | Optional / report in appendix | Useful diagnostic detail but not needed as headline figures — including both separately clutters your main results tables without adding a distinct research-question answer |
| F1 (macro) | **Yes (core)** | Single-number summary of precision/recall trade-off, standard, comparable across models |
| ROC-AUC | **Yes (core)** | Threshold-independent, directly comparable to the prior literature you're extending (both cited prior-work papers report AUC) |
| PR-AUC | Optional | More informative than ROC-AUC specifically under imbalance; include only if your "optional" time budget allows, as a secondary robustness check on your imbalance-related claims |

**Final core set: Accuracy, Balanced Accuracy, F1 (macro), ROC-AUC.** This is four metrics, not seven — enough to be thorough without metric-sprawl diluting your report.

### 2. Calibration / reliability

| Metric | Formula (informal) | Intuition | Why needed | Interpretation | Direction |
|---|---|---|---|---|---|
| **Expected Calibration Error (ECE)** | Bin predictions into $M$ confidence bins; $\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \lvert \text{acc}(B_m) - \text{conf}(B_m) \rvert$ | "On average, how far off is the model's stated confidence from its actual correctness rate, within each confidence bucket?" | The standard, most-cited calibration metric — makes your work directly comparable to the broader calibration literature (Guo et al. 2017 and successors) | 0 = perfectly calibrated; higher = worse | **Lower is better** |
| **Brier Score** | $\text{Brier} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i)^2$ | Mean squared error between predicted probability and the true (0/1) outcome — a single proper scoring rule combining calibration *and* sharpness | Complements ECE: ECE can be "gamed" somewhat by binning choices; Brier Score is bin-free and a strictly proper scoring rule, giving you a robustness check on your ECE findings | 0 = perfect; 0.25 = the score of a constant 50% predictor on a balanced set (useful mental anchor) | **Lower is better** |
| Calibration curve (reliability diagram) | Plot of predicted-confidence bin vs. empirical accuracy in that bin | Visual version of ECE | Needed as a figure (Part O), not as a reported scalar per-run | Diagonal = perfect calibration; curve below diagonal = overconfidence, above = underconfidence | — |
| Confidence distribution | Histogram of $\max(\hat{p}, 1-\hat{p})$ across test predictions | Shows whether the model is "bunching up" near 100% confidence even when wrong | Useful diagnostic, especially for showing *overconfidence under noise* directly | — | — |

**Final core set: ECE and Brier Score as your two headline reliability numbers; calibration curves and confidence distributions as supporting figures, not separately tabulated metrics.** This avoids the "20 unnecessary metrics" trap you were explicitly warned against.

---

# PART K — Calibration (Post-Hoc)

1. **When to apply:** As a *secondary* experiment (Part N, "should-do" tier) layered on top of your core noise-degradation results — not baked into every run by default, since your primary question is about *raw* model reliability, and pre-calibrating everything would hide the effect you're trying to measure in the first place.
2. **Which data to use:** A dedicated calibration subset carved from the **training** split (e.g., 70% train-proper / 15% calibration / 15% test) — fit the calibrator (Platt scaling = logistic regression on model outputs, via `sklearn.calibration.CalibratedClassifierCV` with `method='sigmoid'`) on this calibration subset.
3. **Why not the test set:** Using test data to fit a calibrator and then evaluating calibration quality on that same test data is leakage — your reported ECE/Brier would be artificially good, not a measure of generalizable calibration.
4. **Separate per noise level?** **Yes** — fit a separate calibrator for each (model, noise level, dataset, seed) combination, because the whole point of H4 is to see whether calibration can "rescue" reliability *at each specific noise level*, and a single calibrator fit at 0% noise and reused everywhere wouldn't test that.
5. **Platt vs. Isotonic vs. Temperature scaling — recommendation:** Use **Platt scaling (sigmoid)** as your primary method. It's simpler, needs less calibration data (relevant since German Credit is only 1,000 rows and you're already splitting it three ways), and is the standard choice for small tabular calibration sets; Isotonic regression is more flexible but overfits more easily on small calibration subsets (a real risk here), and Temperature scaling is designed for multi-class neural network logits, not a natural fit for your binary tabular setting. Mention Isotonic as a "we considered and rejected due to calibration-set size" footnote in your Methodology — this kind of explicit trade-off reasoning is exactly what strengthens a viva answer.
6. **Experiment design:** Compare raw ECE/Brier vs. Platt-calibrated ECE/Brier, per model, per noise level, per dataset (this *is* Experiment 7 in Part N) — yes, worthwhile, and directly tests H4.

---

# PART L — Conformal Prediction: Include or Drop?

**Evaluation:** Conformal prediction is conceptually well-matched to your research theme (it's another lens on "reliability" beyond accuracy), and you already correctly identified it as optional. Given your constraints:

- **Argument for including:** It directly extends RQ0 with a third reliability lens (distribution-free coverage guarantees, not just calibration), and recent literature (2025–2026, found in prior research) shows this combination (tabular foundation models + conformal prediction + corruption) is an active area — including it would make your project feel more current.
- **Argument against:** You are a 2-person team on a 7-week clock, already running a full noise × model × dataset × seed matrix (200 runs) plus a calibration ablation. Conformal prediction adds a genuinely new concept (nonconformity scores, calibration-set-based quantile thresholds) that needs to be *understood*, not just library-called, if you're going to defend it in viva — and doing it badly (e.g., miscomputing coverage, leaking the calibration set) is worse than not doing it.

**Recommendation: Treat as OPTIONAL / "if time permits" (Part N, lowest tier), not core.** If you do include it:
- **Library:** `MAPIE` (`pip install mapie`, scikit-learn-contrib, officially maintained) — compatible with scikit-learn-style `.predict_proba()` models including wrapped TabPFN.
- **Calibration set:** reuse the same calibration subset from Part K (15% of training data) — do not create yet another split.
- **Nonconformity score:** use the simple score method (`s(x,y) = 1 - \hat{p}(y|x)$, MAPIE's `method='score'`) — simplest to explain and defend.
- **Coverage:** report empirical coverage at a target of 90% (`alpha=0.1`) — i.e., what fraction of test points actually had their true label inside the predicted set.
- **Prediction-set size:** report average set size — a model that achieves 90% coverage via huge, uninformative prediction sets ("could be either class") is technically "valid" but useless; this is the informativeness/coverage trade-off worth one paragraph of discussion.
- **Experiment:** does empirical coverage stay near the 90% target as noise increases, or does it drift (over- or under-cover)? One figure (coverage vs. noise level, one line per model) is sufficient — don't expand this into a second full experimental matrix.

**If Weeks 1–4 are behind schedule at all, drop this entirely without guilt — your core project (performance + calibration under noise) is already complete and defensible without it.**

---

# PART M — Statistical Analysis

**What's being compared, and with what test:**

| Comparison | Test | When to use | Significance threshold | How to report |
|---|---|---|---|---|
| Across all 4 models, at a fixed noise level and dataset (e.g., "are the 4 models' accuracies significantly different at 20% noise?") | **Friedman test** (non-parametric, repeated measures across seeds) | Default choice for comparing >2 models across multiple paired runs (your 5 seeds = repeated measures) | α = 0.05 | Report Friedman χ² statistic and p-value; if significant, follow with post-hoc |
| Pairwise model comparison after a significant Friedman result | **Nemenyi post-hoc test** | Standard companion to Friedman for all-pairs comparison | Report critical difference (CD) diagram or pairwise p-values with Bonferroni-style correction already built into Nemenyi | Use a CD diagram (one of your 8–12 figures, Part O) — visually clean and standard in ML benchmarking papers |
| Raw vs. calibrated ECE for the same model/noise/dataset (paired, before/after) | **Wilcoxon signed-rank test** | Two related (paired) samples, non-parametric, appropriate for 5 seeds (small n) | α = 0.05 | Report W statistic, p-value, and the direction of the effect |
| Accuracy-slope vs. ECE-slope degradation rate (H1) | Compute per-seed slopes (simple linear regression of metric vs. noise level), then **Wilcoxon signed-rank** on the paired slope values (accuracy-slope vs. ECE-slope, normalized to comparable scales) | Testing whether one degrades faster than the other, per model | α = 0.05 | Report alongside the degradation-curve figure |
| Effect size | **Rank-biserial correlation** (natural companion to Wilcoxon) or simple **Cohen's d** on raw metric differences if roughly normal | Always report alongside p-values — p-values alone don't convey magnitude | — | State explicitly: "statistically significant AND practically meaningful" or "statistically significant but small effect," as appropriate |
| Confidence intervals | Bootstrap CI (resample your 5 seeds with replacement, or report mean ± std across seeds directly given small n) | For all headline summary numbers in your results tables | 95% CI | Report as mean (95% CI) or mean ± std in every summary table |
| ANOVA | **Not recommended** | Your sample sizes per condition (5 seeds) are small and metrics like ECE are often not normally distributed — non-parametric tests (Friedman/Wilcoxon) are the safer, more defensible choice here | — | Mention in your Limitations section that non-parametric tests were chosen deliberately for this reason — this is a sophistication signal, not a weakness |

**Library:** `scikit-posthocs` provides Nemenyi post-hoc and CD-diagram utilities compatible with `scipy.stats.friedmanchisquare` output; `scipy.stats.wilcoxon` for the paired tests. Both are standard, well-maintained, free.

**Keep it appropriate for undergraduate research:** Friedman + Nemenyi + Wilcoxon + effect size + CI is already a complete, credible statistical toolkit — do not add ANOVA, bootstrapped hypothesis testing, or Bayesian methods on top; it would be disproportionate to the project's scope and risks you not being able to explain every test in viva (Part X), which is worse than doing fewer tests well.

---

# PART N — Final Experiment List

### MUST DO

| # | Objective | Setup | Variables | Models | Dataset | Metrics | Expected graph | Answers |
|---|---|---|---|---|---|---|---|---|
| 1 | Clean-data baseline | 0% noise, all models, both datasets, 5 seeds | — | All 4 | Both | All core metrics | Baseline table | Establishes reference point |
| 2 | Progressive noise: performance degradation | 0/10/20/30/40% noise | Noise level | All 4 | Both | Accuracy, Bal. Acc., F1, ROC-AUC | Performance vs. noise (line, 4 lines/panel) | RQ1, H2 (performance side) |
| 3 | Progressive noise: reliability degradation | Same runs as #2, same models | Noise level | All 4 | Both | ECE, Brier | Reliability vs. noise (line, 4 lines/panel) | RQ0, H1, H2 (reliability side) |
| 4 | Model ranking stability | Derived from #2 and #3 | — | All 4 | Both | Spearman correlation of rankings | Ranking-change heatmap or table | RQ2, H3 |
| 5 | Cross-dataset validation | Compare #2–#4 results between German Credit and Heart Disease | Dataset | All 4 | Both | All core + reliability metrics | Side-by-side panel figure | RQ3 |
| 6 | Statistical significance | Friedman/Nemenyi/Wilcoxon on #2–#4 results | — | All 4 | Both | p-values, CD diagram | CD diagram | Confirms whether observed differences are real |

### SHOULD DO

| # | Objective | Setup | Variables | Models | Dataset | Metrics | Expected graph | Answers |
|---|---|---|---|---|---|---|---|---|
| 7 | Calibration rescue experiment | Platt scaling on calibration subset, compare raw vs. calibrated | Calibration on/off × noise level | All 4 | Both | ECE, Brier (before/after) | Before/after bar or line chart | RQ4, H4 |
| 8 | Confidence distribution under noise | Histogram of predicted confidence at 0% vs. 40% noise | Noise level | TabPFN + best classical model | German Credit (primary) | Confidence histogram | Overlaid histogram | Visual evidence of overconfidence (supports RQ0 narrative) |

### OPTIONAL / IF TIME PERMITS

| # | Objective | Setup | Variables | Models | Dataset | Metrics | Expected graph | Answers |
|---|---|---|---|---|---|---|---|---|
| 9 | Conformal coverage experiment | MAPIE split-conformal, 90% target coverage | Noise level | All 4 | German Credit | Empirical coverage, set size | Coverage vs. noise (with 90% reference line) | Extended reliability lens (Part L) |
| 10 | Asymmetric noise bonus condition | One class-conditional noise condition at 20% | Noise mechanism (symmetric vs. asymmetric) | All 4 | German Credit | Accuracy, ECE | Grouped bar chart | Tests whether findings are noise-mechanism-specific |

---

# PART O — Visualization Plan (10 figures)

| # | Figure | X-axis | Y-axis | Lines/Groups | Error bars | Demonstrates |
|---|---|---|---|---|---|---|
| 1 | Accuracy vs. noise level | Noise % | Accuracy | One line per model | 95% CI band (across seeds) | Core performance degradation curve |
| 2 | ECE vs. noise level | Noise % | ECE | One line per model | 95% CI band | Core reliability degradation curve — **your headline figure**, paired visually with Fig. 1 |
| 3 | Brier Score vs. noise level | Noise % | Brier | One line per model | 95% CI band | Robustness check on Fig. 2 |
| 4 | F1 vs. noise level | Noise % | F1 (macro) | One line per model | 95% CI band | Supplementary performance view, accounts for imbalance |
| 5 | Reliability diagrams at 0% and 40% noise | Predicted confidence bin | Empirical accuracy | One panel per model, two noise levels overlaid | — | Direct visual of calibration quality, before/after heavy noise |
| 6 | Confidence distribution histograms | Predicted confidence | Count/density | 0% vs. 40% noise, overlaid | — | Shows overconfidence emerging |
| 7 | Model ranking heatmap | Noise level | Model | Color = rank (1–4) by accuracy vs. by ECE, two sub-heatmaps | — | Directly visualizes RQ2 (ranking instability) |
| 8 | Critical-difference (CD) diagram | — | — | All 4 models | — | Statistical significance of model differences, standard ML-benchmarking figure |
| 9 | Cross-dataset comparison panel | Noise % | ECE (or accuracy) | Side-by-side: German Credit vs. Heart Disease | 95% CI band | RQ3 — is the effect dataset-specific |
| 10 | Calibration-rescue bar chart | Model | ECE | Raw vs. Platt-calibrated, grouped bars, faceted by noise level | Error bars across seeds | H4 — does calibration help, and does the benefit shrink with noise |

**(Optional, if Experiment 9/10 are run) Figure 11:** Conformal coverage vs. noise level, with a horizontal reference line at 90% target. **Figure 12:** Symmetric vs. asymmetric noise grouped bar comparison. Only produce these if the underlying optional experiments were actually run.

---

# PART P — Expected Results (Hypotheses, Not Guarantees)

Clearly hypothetical, framed as such in your report's Discussion section (never pre-written as findings):

- **Plausible outcome 1:** Accuracy for TabPFN degrades slowly (consistent with ICML workshop paper, Part C), but ECE degrades at a comparable or even *faster* rate than classical models — i.e., TabPFN "looks robust" on accuracy while quietly becoming overconfident. This would be your most interesting and most citable finding if it occurs.
- **Plausible outcome 2:** Both accuracy and calibration degrade roughly proportionally for all models — a legitimate "no decoupling found" result. Report honestly; discuss what conditions (noise range, dataset size) might be needed to reveal decoupling if it exists but wasn't visible here.
- **Plausible outcome 3:** TabPFN outperforms classical models on *both* axes at all noise levels — in which case your contribution becomes "the first calibration-inclusive confirmation that TabPFN's robustness is genuinely broad, not accuracy-specific," which is still a real, reportable finding (it closes the gap identified in Part D either way).
- **Plausible outcome 4:** A genuine crossover — TabPFN has worse calibration than GBDTs at low noise but better at high noise (or vice versa) — this would be the most "paper-worthy" result and should be highlighted prominently if it emerges, with appropriate statistical backing (don't overclaim from visual inspection alone — confirm with Part M's tests).

**What would be surprising:** Any result where the *best* model by accuracy and the *best* model by ECE are **different models**, and this flips consistently as noise increases (RQ2/H3 directly) — this is the single most "textbook-research-paper" finding this project could produce, and is worth structuring your Discussion section around if you observe it.

---

# PART Q — Python Project Structure

```
project/
│
├── data/
│   ├── raw/                     # original downloaded CSVs, untouched
│   └── processed/                # cleaned, encoded versions (generated, not hand-edited)
│
├── configs/
│   ├── experiment_config.yaml    # noise levels, seeds, model list, dataset list — single source of truth
│   └── model_params.yaml         # hyperparameters per model
│
├── src/
│   ├── data/
│   │   ├── loaders.py             # load_dataset()
│   │   └── preprocessing.py       # preprocess_data(), split_dataset()
│   ├── noise/
│   │   └── injection.py           # inject_label_noise()
│   ├── models/
│   │   ├── classical.py           # wrappers for LR/RF/XGBoost
│   │   └── tabpfn_wrapper.py      # TabPFN-specific setup, token handling, CPU guards
│   ├── calibration/
│   │   └── calibrate.py           # calibrate_model() (Platt scaling)
│   ├── conformal/                 # only populated if Experiment 9 is attempted
│   │   └── conformal.py
│   ├── evaluation/
│   │   ├── metrics.py             # calculate_metrics(), calculate_ece(), calculate_brier_score()
│   │   └── statistics.py          # friedman/nemenyi/wilcoxon wrappers
│   └── visualization/
│       └── plots.py               # plot_results() and all Part O figure functions
│
├── experiments/
│   └── run_experiment.py          # orchestrates the full matrix via run_experiment()
│
├── results/
│   └── raw_results.csv            # schema defined in Part T
│
├── figures/                       # generated output figures, named per Part O numbering
│
├── notebooks/
│   └── exploratory/                # scratch work only — nothing load-bearing lives here permanently
│
├── requirements.txt                # pinned versions, especially tabpfn==X.Y.Z
├── README.md                       # how to reproduce, per Part S
└── reproducibility_log.md          # seeds, hardware, package versions, run dates
```

**What each key file does:**
- `experiment_config.yaml`: the **single place** that defines noise levels, seeds, dataset names, model list — every script reads from here, nothing is hardcoded twice. Prevents the classic bug where Week 6's figures used slightly different noise levels than Week 4's raw results.
- `tabpfn_wrapper.py`: isolates all TabPFN-specific quirks (token/env var setup, version pinning, CPU-size guard that raises a clear error if accidentally run on a >1,000-row dataset) so the rest of the codebase treats TabPFN like any other scikit-learn-style model.
- `run_experiment.py`: the single entry point that loops over the full matrix (Part G) and writes to `results/raw_results.csv` — this is what you literally run overnight if needed.

---

# PART R — Core Function Signatures

```python
def load_dataset(name: str) -> pd.DataFrame:
    """Load 'german_credit' or 'heart_disease' from data/raw/, return raw dataframe."""

def preprocess_data(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, encode categoricals, handle the handful of missing values (Heart Disease only).
    Returns (X, y). Does NOT scale — scaling happens per-model after splitting."""

def split_dataset(X, y, seed: int, train_frac=0.70, calib_frac=0.15, test_frac=0.15):
    """Stratified split. Returns X_train, X_calib, X_test, y_train, y_calib, y_test.
    Calibration split only used when Part K/L experiments run; otherwise merge calib into train."""

def inject_label_noise(y_train: np.ndarray, noise_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """See Part F pseudocode. Returns (y_noisy, flipped_indices)."""

def train_model(model_name: str, X_train, y_train, params: dict):
    """Dispatch to the correct fit routine (LR/RF/XGBoost/TabPFN) based on model_name.
    Raises a clear error if model_name == 'tabpfn' and len(X_train) > 1000 without the CPU override flag."""

def get_probabilities(model, X) -> np.ndarray:
    """Unified .predict_proba()-style interface across all 4 model types, returns P(y=1)."""

def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    """Returns {'accuracy':..., 'balanced_accuracy':..., 'f1':..., 'roc_auc':...}"""

def calculate_ece(y_true, y_proba, n_bins: int = 10) -> float:
    """Expected Calibration Error, per Part J formula."""

def calculate_brier_score(y_true, y_proba) -> float:
    """Standard Brier score."""

def calibrate_model(model, X_calib, y_calib, method: str = 'sigmoid'):
    """Wraps a fitted model with sklearn.calibration.CalibratedClassifierCV(cv='prefit', method=method)."""

def run_experiment(config: dict) -> pd.DataFrame:
    """Top-level orchestrator: loops model × noise_level × dataset × seed per experiment_config.yaml,
    calls the above functions in the Part H protocol order, appends one row per run to results.
    This is the function invoked by experiments/run_experiment.py."""

def aggregate_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Groups by (model, noise_level, dataset), computes mean/std/95% CI across seeds for every metric."""

def plot_results(aggregated: pd.DataFrame, figure_type: str, **kwargs):
    """Dispatches to the correct Part O figure function based on figure_type string."""
```

---

# PART S — Reproducibility Protocol

- **Random seeds:** three *separately named* seed roles per Part H — `split_seed`, `noise_seed`, `model_init_seed` (where applicable, e.g., Random Forest's `random_state`). Use seeds `[0, 1, 2, 3, 4]` for your 5 repetitions, applied consistently across all three roles per repetition index (i.e., repetition 0 uses seed 0 for all three roles — simple and auditable).
- **Package versions:** pin everything in `requirements.txt`, including the exact `tabpfn` version (Part I) — run `pip freeze > requirements_frozen.txt` once your environment is final and commit both files.
- **Dataset version:** record the exact OpenML dataset ID / UCI download date and a checksum (`hashlib.md5` of the raw CSV) in `reproducibility_log.md`, since UCI/OpenML pages occasionally get minor corrections.
- **Hardware info:** record CPU model, RAM, OS, and Python version once at the top of `reproducibility_log.md` — relevant because TabPFN timing claims (Part G) are hardware-dependent.
- **Configuration files:** `experiment_config.yaml` and `model_params.yaml` (Part Q) are the single source of truth — never hardcode a noise level or seed list anywhere else in the codebase.
- **Experiment IDs:** every row in `raw_results.csv` gets a unique `run_id` (e.g., `{dataset}_{model}_{noise_level}_{seed}`) for traceability.
- **Logging:** a simple `logging` call at the start/end of each run in `run_experiment.py`, writing to `results/run_log.txt`, timestamped — useful for debugging overnight runs and for showing your professor you tracked execution, not just final numbers.
- **Saving trained models:** not required for most models (cheap to retrain), but **do save TabPFN's fitted state is unnecessary (it doesn't gradient-train)** — instead, save the exact `(X_train_noisy, y_train_noisy)` arrays used per run if disk space allows, so any individual run can be exactly re-inspected without re-injecting noise (noise injection is stochastic per seed, so saving the realized noisy labels, not just the seed, is the safest reproducibility choice).
- **Saving raw results:** every run appends to `results/raw_results.csv` immediately after completion (not buffered in memory until the end) — protects against losing hours of compute to a crash on run 180 of 200.

**For another student to reproduce your results:** clone the repo, `pip install -r requirements_frozen.txt`, set `TABPFN_TOKEN` per Part I if required by your pinned version, run `python experiments/run_experiment.py --config configs/experiment_config.yaml`, then `python src/visualization/plots.py` to regenerate all figures from `results/raw_results.csv`. This should be literally stated in your README.

---

# PART T — Result Storage Schema

`results/raw_results.csv` — one row per (model, noise_level, dataset, seed) run:

| Column | Type | Notes |
|---|---|---|
| `run_id` | str | `{dataset}_{model}_{noise_level}_{seed}` |
| `dataset` | str | `german_credit` / `heart_disease` |
| `model` | str | `logistic_regression` / `random_forest` / `xgboost` / `tabpfn` |
| `noise_level` | float | 0.0–0.4 |
| `seed` | int | 0–4 |
| `n_train` | int | after split, for audit |
| `n_flipped` | int | actual number of labels flipped (from `inject_label_noise` return value) |
| `accuracy` | float | |
| `balanced_accuracy` | float | |
| `f1_macro` | float | |
| `roc_auc` | float | |
| `ece` | float | |
| `brier_score` | float | |
| `ece_calibrated` | float | nullable — only filled if Experiment 7 run for this row |
| `brier_calibrated` | float | nullable |
| `conformal_coverage` | float | nullable — only if Experiment 9 run |
| `conformal_set_size` | float | nullable |
| `train_time_seconds` | float | useful for Part G compute reporting |
| `timestamp` | datetime | run completion time |

**Aggregation across seeds:** group by `(dataset, model, noise_level)`, compute `mean`, `std`, and 95% CI (either normal-approximation `mean ± 1.96*std/sqrt(5)` or a simple percentile bootstrap) for every numeric metric column — store as `results/aggregated_results.csv`, which is what all Part O figures should actually read from (never recompute aggregation inside plotting code — keep it in one place).

---

# PART U — Two-Person Division of Work

Both students read the full blueprint and understand every part before dividing work — division is about **primary ownership**, not exclusive knowledge.

| | Student 1 | Student 2 |
|---|---|---|
| **Primary technical ownership** | Data pipeline (`src/data/`), noise injection (`src/noise/`), classical models (`src/models/classical.py`), leakage-prevention testing | TabPFN integration (`src/models/tabpfn_wrapper.py`), calibration (`src/calibration/`), conformal (if attempted), metrics implementation (`src/evaluation/metrics.py`) |
| **Experiments led** | 1, 2, 4, 5 | 3, 6, 7, 8 (and 9/10 if attempted) |
| **Statistics** | Shared — both must understand Friedman/Wilcoxon well enough to defend in viva; Student 1 implements, Student 2 reviews/cross-checks | — |
| **Visualization** | Figures 1, 4, 7, 9 | Figures 2, 3, 5, 6, 8, 10 |
| **Report sections (primary drafting)** | Introduction, Literature Review, Methodology, Dataset Description, Preprocessing | Results, Statistical Analysis, Discussion, Limitations |
| **Report sections (shared)** | Abstract, Research Gap, Conclusion/Future Scope — written together after everything else is done | |
| **Reproducibility/engineering hygiene** | `requirements.txt`, config files, README | `reproducibility_log.md`, result-schema integrity, run logging |

**Both students must be able to explain every figure and every statistical test in viva** — the division above is about who *builds* what, not who is allowed to *understand* what.

---

# PART V — 7-Week Timeline

| Week | Deliverables | Student 1 | Student 2 | Definition of "done" |
|---|---|---|---|---|
| **1** | Literature review finalized; final RQs/hypotheses locked; TabPFN environment working (token obtained if needed, version pinned); datasets downloaded and checksummed | Literature table (Part C); dataset download + raw EDA | TabPFN install, token setup, version pin, one smoke-test run on 50 rows of German Credit | TabPFN runs successfully end-to-end on a tiny sample; datasets committed to `data/raw/` with checksums |
| **2** | Preprocessing pipeline complete; baseline (0% noise) models trained on both datasets | `preprocess_data()`, `split_dataset()` implemented and unit-tested for leakage (Part H) | Classical model training wrappers; first full baseline run (Experiment 1) | `raw_results.csv` has clean 0%-noise rows for all 4 models × 2 datasets × 5 seeds |
| **3** | Noise injection pipeline complete; full noise × model sweep running (Experiments 2–3) | `inject_label_noise()` + leakage assertion tests | Metrics implementation (`calculate_ece`, `calculate_brier_score`) validated against a hand-computed toy example | Full 200-run matrix (minus optional experiments) completes without errors; spot-check 2–3 runs by hand |
| **4** | TabPFN fully integrated into the sweep; calibration (Experiment 7) implemented | Cross-dataset comparison code (Experiment 5) | `calibrate_model()`, calibration-subset wiring, Experiment 7 runs | All MUST-DO and Experiment 7 rows present in `raw_results.csv`; decide go/no-go on conformal prediction (Experiment 9) based on remaining time |
| **5** | All core experiments complete; statistics implemented | Friedman/Nemenyi implementation + CD diagram | Wilcoxon tests for H1/H4; model-ranking analysis (Experiment 4) | `aggregated_results.csv` finalized; all significance tests produce p-values with interpretations written down |
| **6** | All figures generated; results sections drafted | Figures 1, 4, 7, 9; Results section draft | Figures 2, 3, 5, 6, 8, 10; Statistical Analysis section draft | All 10 (or 12) figures exist in `figures/`, captioned, referenced in draft text |
| **7** | Full report assembled; presentation built; viva prep (Part X) done | Report assembly, Methodology/Literature sections polished | Results/Discussion polished; presentation slides | Report draft complete end-to-end; both students can answer every Part X question out loud without notes |

**Buffer note:** if Week 3 or 4 slips, the first things to cut are, in order: (1) asymmetric-noise bonus (Experiment 10), (2) conformal prediction (Experiment 9), (3) the 297-row Heart Disease secondary dataset shrinks to a smaller noise-level subset (e.g., only 0/20/40% instead of all 5) rather than being dropped entirely — losing the second dataset entirely would weaken RQ3's claim meaningfully, so it's the last thing to cut, not the first.

---

# PART W — Final Report Structure

| # | Chapter | What to write |
|---|---|---|
| 1 | Abstract | 150–250 words: the question, the method (2 datasets, 4 models, 5 noise levels, 5 seeds), the headline finding (written last, once you know it), one sentence on implication |
| 2 | Introduction | Motivate accuracy-vs-reliability generally (cite Guo et al. 2017), then narrow to tabular foundation models and label noise specifically |
| 3 | Problem Definition | State RQ0 precisely; define "reliability" operationally (ECE + Brier) up front so the rest of the report can use the term precisely |
| 4 | Research Questions | RQ0–RQ4 from Part B, verbatim |
| 5 | Literature Review | Table from Part C, expanded into prose paragraphs per theme (TabPFN robustness; calibration theory; label-noise handling) |
| 6 | Research Gap | Part D, near-verbatim — this is a strong, pre-written section |
| 7 | Methodology | The full protocol from Part H, the noise mechanism from Part F, split/calibration-set definitions — this is your most-scrutinized section, be precise |
| 8 | Dataset Description | German Credit and Heart Disease: size, features, source, license, summary statistics table |
| 9 | Preprocessing | Encoding, scaling (per-model, since LR needs it and trees/TabPFN don't), missing-value handling for Heart Disease |
| 10 | Label Noise Generation | Part F formalized, with the pseudocode as a figure/listing |
| 11 | Models | Part I, condensed — one paragraph per model plus the hyperparameter table |
| 12 | Experimental Design | Part G's matrix, Part N's experiment list |
| 13 | Evaluation Metrics | Part J's tables, with formulas |
| 14 | Results | Organized by research question (RQ0 results, then RQ1, RQ2, RQ3, RQ4) rather than by experiment number — easier for a reader to follow the argument |
| 15 | Statistical Analysis | Part M's tests applied to your actual numbers — p-values, CD diagram, effect sizes |
| 16 | Discussion | Interpret findings against Part C's prior literature and Part P's hypothesized outcomes — explicitly say which outcome materialized |
| 17 | Limitations | Two datasets (not ten); CPU-only TabPFN constrained dataset size choice; symmetric noise only (asymmetric as future work if not run); 5 seeds (not 20+) |
| 18 | Conclusion | Direct answer to RQ0 in 2–3 sentences, no hedging beyond what the data supports |
| 19 | Future Scope | Larger datasets via TabPFN GPU/hosted client; asymmetric noise; more TFMs (TabM, FT-Transformer) as foundation-model diversity grows |
| 20 | References | Full citation list from Part C plus any added during writing |

---

# PART X — Viva Preparation

**Basic understanding**
- *What is label noise?* Incorrect labels in training data — here, randomly flipped binary labels (symmetric/NCAR noise) at a controlled rate.
- *Why this topic?* Because "is the model still accurate" and "is the model still trustworthy about its confidence" are different questions that almost no undergraduate project — and surprisingly few papers — ask together, specifically for a 2025 foundation model.
- *What is TabPFN?* A transformer-based tabular foundation model (Hollmann et al., *Nature*, 2025) that performs in-context inference on new tabular datasets without per-dataset gradient training.
- *Why not use a neural network (generic deep learning)?* Because the research question is specifically about *tabular foundation models vs. classical tabular ML*, not deep learning generally; a generic MLP wouldn't test the specific "in-context, not gradient-trained-on-your-noisy-labels" mechanism that motivates comparing TabPFN.
- *Why compare against classical ML?* They are the real-world incumbent for tabular data and the direct comparison point in the prior literature (Part C) you're extending.

**Research methodology**
- *What is your research gap?* Part D, memorize the three-sentence version.
- *Independent/dependent variables?* IV: label noise level (and model family). DV: performance metrics (Accuracy/F1/AUC) and reliability metrics (ECE/Brier), tracked in parallel.
- *Why these noise levels (0/10/20/30/40)?* 5% was too subtle to be distinguishable from split-to-split variance on a 1,000-row dataset; 40% gives a visible degradation/breakdown regime (Part F).
- *Why keep the test set clean?* Because a noisy test set would make "accuracy" itself an unreliable measurement — you need ground truth to evaluate against (Part F/H).
- *Why multiple random seeds?* To distinguish a real effect of noise from the randomness inherent in which specific labels got flipped and how the data happened to split (Part G/M).

**Reliability**
- *What is calibration?* Whether a model's stated confidence matches its actual correctness rate — a 70%-confidence prediction should be right about 70% of the time, averaged over many such predictions.
- *What is ECE?* Average gap between confidence and accuracy, computed within binned confidence ranges (Part J formula).
- *What is Brier Score?* Mean squared error between predicted probability and the true binary outcome — a single number combining calibration and sharpness.
- *Can a model have high accuracy but poor calibration?* Yes — this is the entire premise of the project; a model can be right 90% of the time while claiming 99% confidence every time, which is accurate but badly miscalibrated.

**Experimental validity**
- *How did you prevent data leakage?* Part H's protocol, recite the "split before any `.fit()` call" rule and the clean-test-set rule specifically.
- *Why is your experiment reproducible?* Pinned package versions (especially TabPFN), fixed and separately-named seeds, config-file-driven parameters, full raw results stored per run (Part S/T).
- *Why these datasets?* TabPFN's documented CPU feasibility ceiling (≲1,000 rows) directly drove the choice of German Credit and Heart Disease over larger standard datasets like Adult Income (Part E) — be ready to explain this trade-off confidently, it shows you made an evidence-based engineering decision, not a lazy one.
- *Why these statistical tests?* Non-parametric (Friedman/Wilcoxon) because of small sample size (5 seeds) and no guarantee of normally-distributed metrics like ECE (Part M).

**Results**
- *What if TabPFN performs worse (on either axis)?* Still a valid, reportable finding — it would narrow or contradict the accuracy-side claims of prior literature (Part C) on your specific datasets/noise range, which is itself worth discussing, not a failure of the project.
- *What if all models behave similarly?* A legitimate null-ish result — report it honestly and discuss what might reveal a difference (more extreme noise, different datasets) as future work (Part P, outcome 2).
- *What if accuracy decreases but calibration improves?* Counterintuitive but possible (e.g., a model could become "appropriately less confident" as it gets things wrong more — that's actually *good* calibration behavior under degradation); discuss this as evidence the model "knows it's struggling," a positive reliability signal even amid falling accuracy.
- *What is your most important finding?* Answer this only after you have the actual results — do not pre-script this answer before Week 6.

---

# PART Y — Risks and Backup Plans

| Risk | Probability | Impact | Prevention | Backup plan |
|---|---|---|---|---|
| TabPFN installation/token friction | Medium | Medium | Set up and smoke-test TabPFN in **Week 1**, not later; pin version immediately | If token/signup is blocked, fall back to the TabPFN Client (hosted inference, per official docs) for the handful of runs needed, or drop TabPFN entirely and reframe as "classical models only" (weakens novelty but preserves the calibration-vs-accuracy core question) |
| TabPFN CPU performance worse than expected | Medium | Low-Medium | Datasets chosen specifically within documented CPU-feasible range (Part E/G); time one run in Week 1 before committing to the full sweep | If still too slow, reduce seeds from 5 to 3 for TabPFN specifically (keep 5 for classical models), clearly noting this asymmetry and its statistical implications in Limitations |
| Dataset incompatibility (e.g., unexpected encoding issues) | Low | Low | Both chosen datasets are extremely standard, decades of prior use | Heart Disease has a well-documented "?"-as-missing-value quirk — handle explicitly in preprocessing, don't let it silently become a string column |
| Memory issues | Low (24GB RAM is ample for these dataset sizes) | Low | — | — |
| Calibration implementation issues (ECE computed incorrectly) | Medium (ECE has several slightly different conventions in different papers/libraries) | Medium | Validate `calculate_ece()` against a hand-constructed toy example with known ECE before trusting it on real data (Week 3 task) | Cross-check against an established library implementation (e.g., `netcal` or a well-reviewed reference implementation) as a sanity check, even if you keep your own for transparency |
| Too many experiments attempted | Medium | High | Strict MUST/SHOULD/OPTIONAL tiering (Part N) with explicit cut order (Part V buffer note) | Cut optional tier first, always — never let optional experiments threaten MUST-DO completion |
| Statistical analysis complexity | Low-Medium | Medium | Scoped deliberately to Friedman/Wilcoxon/Nemenyi only (Part M), avoiding ANOVA/Bayesian methods | If still struggling by Week 5, fall back to simpler paired t-tests with an explicit, honest caveat in Limitations about the normality assumption, rather than silently skipping significance testing altogether |
| Insufficiently interesting results (all models behave the same) | Medium | Low (this is a valid finding, not a failure — see Part P) | — | Frame the report around "we tested whether X holds, and found Y" regardless of which Y occurred; a null-ish result reported with rigor is still a strong undergraduate research report |
| Project becoming too broad | Medium | High | This blueprint's entire MUST/SHOULD/OPTIONAL structure exists specifically to prevent this | Revisit Part Z's "Final Recommended Scope" at the Week 4 checkpoint and consciously re-cut if behind |

### Minimum Viable Project
If everything that can go wrong does: **Experiments 1–4 (MUST-DO tier) on German Credit alone**, with 4 models, 3 noise levels (0/20/40% instead of all 5) and 3 seeds (instead of 5), reporting Accuracy/F1/ROC-AUC alongside ECE/Brier, with Friedman + Wilcoxon statistics. This is still a complete, defensible answer to RQ0 — smaller in scope, not weaker in rigor.

---

# PART Z — Final Recommended Scope

**Final title:** *"Accuracy Isn't Reliability: Evaluating Tabular Foundation Models Under Progressive Label Noise"* (your title is good as-is — keep it)

**Final research question:** As training-label noise increases (0–40%), does model reliability (ECE, Brier Score) deteriorate at the same rate as predictive performance (Accuracy, F1, ROC-AUC)?

**Final datasets:** German Credit (primary, 1,000 rows), Heart Disease Cleveland (secondary, 297 rows)

**Final models:** Logistic Regression, Random Forest, XGBoost, TabPFN (version pinned — see Part I)

**Final noise levels:** 0%, 10%, 20%, 30%, 40% (symmetric/NCAR label flipping, training set only)

**Final metrics:** Accuracy, Balanced Accuracy, F1 (macro), ROC-AUC (performance) + ECE, Brier Score (reliability)

**Final MUST-DO experiments:** 1–6 (Part N) — baseline, performance degradation, reliability degradation, ranking stability, cross-dataset validation, statistical significance

**Final SHOULD-DO experiments:** 7–8 — calibration rescue, confidence distribution

**Optional experiments:** 9 (conformal coverage via MAPIE), 10 (asymmetric noise bonus)

**Final statistical tests:** Friedman + Nemenyi post-hoc (across-model comparisons), Wilcoxon signed-rank (paired before/after comparisons), effect sizes and 95% CIs throughout

**Final number of experimental runs:** 200 (MUST-DO + SHOULD-DO core matrix: 4 models × 5 noise × 2 datasets × 5 seeds), +40 more if conformal prediction is added, +20 more if the asymmetric-noise bonus is added

**Estimated computational time:** Well under 2 hours wall-clock for the core 200 runs on your hardware (seconds per classical-model run; tens of seconds per TabPFN run, within its documented CPU-feasible range) — not an overnight job unless optional experiments are added, in which case budget an overnight run as a safety margin

**Difficulty:** 6/10 — conceptually requires real understanding of calibration and non-parametric statistics, but no component requires cutting-edge math or unavailable compute

**Research quality:** 8/10 — a well-scoped, honestly-framed empirical extension with a verified literature gap, not an inflated novelty claim

**Feasibility for 2 students:** 8/10 — comfortably achievable in 7 weeks given the MUST/SHOULD/OPTIONAL tiering and the dataset choice matched to TabPFN's actual CPU constraints

**Biggest risk:** TabPFN setup friction (version/token/CPU-timing surprises) — because it's the one dependency outside your full control and sits on the critical path for nearly every experiment

**Best backup plan:** Resolve TabPFN end-to-end (install, token, one real timed run) in **Week 1**, before writing a single line of the main experiment pipeline — if it's going to be a problem, you want to know in week 1, not week 4, when the Part Y backup plans (hosted client, or classical-models-only reframing) are still easy to execute without losing your timeline.

---

# START HERE — First 10 Tasks

1. **Both students:** read this entire blueprint once, end to end, before writing any code.
2. **Student 2:** `pip install tabpfn`, check the current PyPI version, pin it in `requirements.txt`; attempt a real import and a 50-row toy classification run today — surface any token/license friction immediately (Part I/Y).
3. **Student 1:** download German Credit (OpenML ID 31) and Heart Disease (Cleveland, UCI) into `data/raw/`; record checksums in `reproducibility_log.md`.
4. **Both:** set up the shared GitHub repo using the Part Q folder structure; commit the empty skeleton and `requirements.txt` first.
5. **Student 1:** write and unit-test `preprocess_data()` and `split_dataset()`, including an explicit leakage assertion test (train-fit-only preprocessing, Part H).
6. **Student 2:** write and validate `calculate_ece()` and `calculate_brier_score()` against a hand-constructed toy example with a known, pre-computed correct ECE value.
7. **Student 1:** write `inject_label_noise()` exactly per the Part F pseudocode, plus a test confirming the test set is never touched.
8. **Student 2:** write `train_model()`/`get_probabilities()` wrappers for all 4 models, confirming TabPFN runs successfully on the full 1,000-row German Credit training split within a reasonable time (time it — this validates your Part G compute estimate).
9. **Both, together:** finalize `experiment_config.yaml` with the exact noise levels, seeds, models, and datasets from Part Z — this file becomes the single source of truth for everything after this point.
10. **Both, together:** run Experiment 1 (clean-data baseline, 0% noise, all models/datasets/seeds) end-to-end and manually sanity-check a handful of rows in `raw_results.csv` before moving on to the full noise sweep — catching a bug here is far cheaper than catching it after 200 runs.
