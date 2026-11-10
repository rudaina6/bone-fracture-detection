# Phase 5: comparative analysis (2026-11-10)

Task: explain the Phase 3 / Phase 4 result instead of only reporting it — *why* the unified and region-specific
models come out where they do, and what that implies for Phase 6. This is novelty claim (2) of the proposal
("identify the factors behind the region differences"); claim (1) was Phases 3-4.

**This file records the protocol and the decision rules before any Phase 5 result was seen.** Anything added after
results are in is marked `[added after results]` with the reason. Status: `docs/PROJECT_STATUS.md`.

**No new model selection, and no new test-set evaluation.** Phase 5 touches the test set only by re-reading the
per-image scores that Phases 3 and 4 already wrote (`results/phase3/test_scores.csv`,
`results/phase4/test_scores.csv`); the locks `results/phase3|phase4/TEST_EVALUATED.json` stay as they are. Every
analysis that needs *fitting* a model (learning curves, transfer, feature importance) runs on the 3,266 **training**
images with the frozen folds 0-4, never on the 817 test images. The only quantities computed from test scores are
descriptive summaries and paired tests of models that were already fixed.

## Code
| File | Role |
|---|---|
| `src/phase5_main.py` | 5.1. Consolidated unified-vs-region comparison (CV and test) from the saved scores: tables, per-region forest plot of differences, PR curves, score distributions, McNemar. No fitting. |
| `src/phase5_curves.py` | 5.2. Learning curves (PR-AUC vs number of region training images, region arm vs unified arm) and the cross-region transfer matrix. Fits models on training folds only. |
| `src/phase5_features.py` | 5.2. Feature-space view (PCA / t-SNE by region and fracture status), region separability, cosine similarity of the per-region fracture directions, per-region feature-family importance. |
| `src/phase5_errors.py` | 5.3. Error analysis: agreement of the two approaches, and what distinguishes false negatives from true positives per region, using the metadata proxies. |
| `src/phase5_box_contrast.py` | 5.3 (optional). Computes box-local contrast and box area fraction from the images, so "low contrast" and "subtle" are measured rather than assumed. Writes `results/tables/phase5_box_stats.csv`; the other scripts use it if it exists and skip those columns if it does not. |
| reused unchanged | `evaluate.py` (metrics, `bootstrap_ci`, `paired_bootstrap`), `phase4_regional.fold_paired_bootstrap` and its fast exact `_fast_ap` / `_fast_auc`, `model_zoo.build/SPECS`, `cv_protocol.load_train`, `feature_utils.load_features` |

Outputs: `results/phase5/` (raw per-run CSV/JSON), `results/tables/phase5_*.csv`, `results/figures/phase5_*.png`.

## Index: why each analysis was run, and what decides its answer

Every analysis in this phase, the question that justifies it, and the criterion that settles it. The last column
distinguishes rules **fixed before any Phase 5 result was seen** (the numbered rules below) from conventions
**added afterwards** - a criterion chosen after seeing the numbers is not a pre-registration and is not presented
as one.

| Analysis | Why it was run | What decides the answer | Fixed when |
|---|---|---|---|
| 5.1 consolidated comparison | Phases 3-4 produced the numbers but no single place to read them, and the figures must not drift from the Phase 4 text | Phase 4's rule: a difference counts only if its 95% paired CI excludes 0. Phase 5 adds no verdict; the script asserts it reproduces Phase 4 | before (carried from Phase 4) |
| 5.1 McNemar on test decisions | PR-AUC compares *rankings*; nothing in Phases 3-4 compared the *decisions* the two approaches actually make at their thresholds | Exact binomial on the discordant pairs. **No alpha was fixed in advance**, so p is reported with the raw counts and read as strength of evidence, not as a pass/fail | **after** |
| 5.2 learning curves | The central question of the phase: is Phase 4's picture "the regions differ" or "each region alone has too little data"? Those have opposite implications for Phase 6 | Rule 2 (below): the gap at full size against the fold SD, and whether the region arm is still rising | before |
| 5.2 zero-shot point (`p = 0`) | The sharpest single test of shared signal: how well does a model that has never seen a region do on it? | Read against that region's chance level and against its own full-data value; no threshold | before |
| 5.2 transfer matrix | The learning curve says whether a region needs its own model, not *which* regions resemble each other | Rule 3 (below): size-matched transfer ratio above 0.50 in both directions = the pair shares its signal | before |
| 5.2 size-matching | Regions differ 7x in size, so a weak raw transfer row could be scarcity rather than mismatch | Both matrices reported; the size-matched one is primary by construction | before |
| 5.2 region separability | Bounds what Phase 6's explicit region term can add: if the region is already readable from the features, a unified model could in principle condition on it unaided | **Descriptive, no threshold fixed.** Macro-F1 and the confusion matrix are reported; the conclusion drawn is only the qualitative one that the information is present | **after** |
| 5.2 cosine of fracture directions | Transfer says *that* regions do not share; this says *why* - whether they want different boundaries | Ordering only. In 8,641 dimensions unrelated vectors sit near 0, so absolute values carry no meaning and no cut is applied | before |
| 5.2 family ablation | Which feature families carry the signal in each region, and whether any region prefers a different representation | Ordering within a row only; no family is declared important or not by a threshold. Plain CV, so absolute values are slightly optimistic | before |
| 5.3 error proxies | FracAtlas has no fracture-type or view label, so "subtle" has to be measured through proxies or not claimed at all | Rule 4 (below): an interval excluding the null marks an *association* with being missed, never a cause | before |
| 5.3 box contrast | "Low contrast" was in the plan but is not in the metadata; measuring it beats asserting it | Rule 4, same as the other proxies | before |
| 5.3 primary on out-of-fold, not test | 141 test positives split four ways cannot support an error analysis; the out-of-fold scores give 569 with the same honesty | Not a verdict - a choice of sample, stated so the reader knows which numbers carry the claims | before |
| 5.4 theory | The plan asks for it, and it is only worth writing if it is grounded in this project's own measurements | No criterion; every claim cites a number from 5.1-5.3 or an earlier phase | before |
| Hip question | Left open by Phase 4 | Rule 5 (below), including the instruction to record "not resolvable" rather than guess | before |

## 5.1 The main result (no fitting)

Sources: `results/phase3/oof_rbfsvm__none.csv` and `results/phase4/oof_rbfsvm__none.csv` (same images, same frozen
folds, both honest out-of-fold) for CV; `results/phase4/test_scores.csv` (both models' per-image test scores and
thresholds, 812 images of the four modelled regions) for the test set.

- **Metrics**: as every earlier phase — PR-AUC primary, then ROC-AUC, recall, precision, F2, specificity at the
  tuned thresholds. PR-AUC chance level is the region's fracture rate (hand 0.30, leg 0.10, hip 0.18, shoulder 0.18),
  so PR-AUC is compared *between models within a region*, never across regions.
- **Differences** are region-specific minus unified, with the Phase 4 machinery unchanged: in CV, fold-mean
  differences for PR-AUC / ROC-AUC with images resampled within folds (`fold_paired_bootstrap`), and pooled
  out-of-fold threshold metrics (`evaluate.paired_bootstrap`); on test, paired bootstrap over the region's test images.
  Phase 5 recomputes these from the saved scores rather than copying the Phase 4 tables, and **asserts that it
  reproduces them** (`--check`), so the figures and the Phase 4 text cannot drift apart.
- **McNemar's test** (the plan's alternative to the paired bootstrap) is added on the test set, per region, at the two
  models' tuned thresholds: the discordant pairs (unified right / region wrong and the reverse) with an exact binomial
  two-sided p. It answers a different question from the PR-AUC bootstrap — it compares the *decisions at the chosen
  thresholds*, not the ranking — so the two are reported side by side, not as substitutes.
  **[added after results] No significance threshold was fixed in advance for this test.** It was added to the
  protocol as "report McNemar" without an alpha, so the p-values are reported next to the raw discordant counts
  and read as strength of evidence rather than as a pass/fail at 0.05. Nothing in this phase's conclusions turns
  on a borderline p: the two results that matter are 7e-08 and 0.004, and the two null results (0.61, 0.58) rest
  on 15 and 13 discordant pairs and are reported as "no power" rather than as evidence of equality.
- **Figures**: (a) per-region PR-AUC, region vs unified, CV and test panels with error bars; (b) forest plot of the
  PR-AUC and F2 differences with 95% CIs, CV and test side by side; (c) test PR curves per region, both models;
  (d) out-of-fold score distributions by fracture status per region, with the tuned thresholds marked.

## 5.2 Learning curves

The question the plan asks of them: **is the Phase 4 picture "the regions genuinely need different models" or "each
region alone has too little data"?** Those two have opposite implications for Phase 6, so this is the central analysis
of the phase.

**Design.** For a region `r`, outer fold `k` and a size fraction `p`:
1. take the region's rows in folds `!= k`, draw a **stratified subsample** of `m = round(p * n)` of them (fracture rate
   preserved) — call it `S`;
2. **region arm**: fit on `S` only;
3. **unified arm**: fit on `S` **plus every training row of the other regions** in folds `!= k`;
4. score both on region `r`'s rows in fold `k`; metric = PR-AUC on that fold.

Both arms see the *same* subsample, which makes every point paired, and the unified arm is the honest unified setting
at that `p`: the other regions' data is always available, only the region's own data is scarce.

- `p = 0` is run for the unified arm only, and is the sharpest single number in the phase: PR-AUC on region `r` from a
  model that has seen **no** image of region `r`. If that is close to the region arm at `p = 1`, the regions share their
  fracture signal; if it collapses, they do not.
- **Sizes** `p` in {0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}. **Repeats**: 5 subsample draws per cell for `p < 1`
  (`p = 1` and `p = 0` are deterministic, 1 draw). Reported as mean over folds and draws with a SD over folds.
- **Hyperparameters are fixed at the Phase 3 values (C=10, gamma=0.05), with no tuning, in both arms.** A learning
  curve must vary training size and nothing else; re-tuning at each point would mix in the tuning noise that Phase 4
  already measured separately, and tuning noise at `p = 0.05` on 10 hip images would dominate the curve. The same
  caveat as the Phase 4 sensitivity run applies — those values were chosen on all 3,266 training images, so both arms
  are slightly optimistic — but it applies **equally to both arms**, which is what the comparison needs. The threshold
  plays no part: PR-AUC is threshold-free.
- **Guard**: a cell is skipped (and recorded as skipped) if its subsample has fewer than 2 fractured or fewer than 2
  non-fractured images. With hip at `p = 0.05` that is about 10 images and 2 positives, so the small-`p` end of hip and
  shoulder is expected to be thin and must not be over-read.
- "Other" (7 fractured training images) is not a region here, exactly as in Phase 4, but its rows **are** part of the
  unified arm's other-region data, as they were in Phase 3.

**Cross-region transfer matrix.** Same fitting code, fixed hyperparameters, training folds only: fit on region `A`'s
rows in folds `!= k`, score region `B`'s rows in fold `k`, for all 16 ordered pairs plus an "all regions" row,
averaged over the 5 folds.
- Raw training-set sizes differ 7x across regions (leg 1,699 vs hip 256), so a weak row could be scarcity rather than
  mismatch. The **primary** matrix is therefore **size-matched**: every training region is subsampled to the same
  30 fractured and 140 non-fractured images (feasible for all four in 4/5 folds), 5 draws per cell. The raw matrix is
  reported next to it as the "as much data as that region actually has" version.
- Size-matching fixes prevalence at about 0.18 for every training region, which is hip's and shoulder's natural rate
  but far from hand's 0.30. That is a deliberate trade (it holds the training distribution constant so the only
  difference is *which* region the images come from) and is stated with the result.
- Read as a **transfer ratio** `(PR-AUC(A->B) - chance_B) / (PR-AUC(B->B) - chance_B)`, which removes region `B`'s
  different chance level and scale, next to the raw PR-AUC.

## 5.2 Feature-space view and per-region feature importance

All on the 3,266 training rows (`cv_protocol.load_train`); nothing is fitted on test rows.

- **PCA** fitted on the training rows after the standard in-fold preprocessing (`make_preprocessor`, so
  `StandardScaler` + equal family weights): scatter of PC1-PC2 coloured by region, and a small-multiple of the same
  projection coloured by fracture status, one panel per region. Explained-variance curve alongside.
- **t-SNE** (perplexity 30, `init="pca"`, seed 42) on the first 50 PCs of the same projection, coloured the same two
  ways. t-SNE is used for looking only: no claim rests on it, because its distances are not metric and its clusters
  are not stable under perplexity. Every quantitative statement about separability comes from the next two items.
- **Region separability (quantitative)**: multinomial logistic regression predicting the *region* from the features,
  5-fold CV over the frozen folds, macro-F1 and a confusion matrix. This matters for Phase 6: if the regions are
  nearly perfectly separable from the features, the unified RBF SVM can in principle learn region-conditional
  behaviour by itself, and the gain from the planned multi-task kernel should be small; if they are not, an explicit
  region term has something to add. **[added after results] No cut-off was fixed for "separable".** The macro-F1 and
  the confusion matrix are reported as they come out, and the only conclusion drawn from them is the qualitative
  one - that the region information is present in the features - which is what the Phase 6 argument needs. No
  verdict in this phase depends on where a threshold would have been put.
- **Is the *fracture* direction region-dependent?** A linear SVM (C=0.1, `class_weight` as in Phase 3's chosen
  strategy: none; block-weighted features) fitted per region on that region's training rows, plus one on all of them,
  giving five weight vectors; report the 5x5 **cosine-similarity matrix**. Two regions whose fracture directions are
  nearly parallel can share a model by construction; near-orthogonal ones cannot. This is the linear shadow of the
  question the RBF SVM answers, and it is nearly free, so it is reported as a *descriptive* companion to the transfer
  matrix, not as a result about the RBF model.
- **Feature-family importance per region**: for each region, and for the unified set, the 5-fold CV PR-AUC of each
  **single family** and of each **leave-one-family-out** subset, with the RBF SVM at the fixed Phase 3
  hyperparameters. When a subset is used, its block weights are **rescaled by `sqrt(5 / k)` for `k` retained
  families**, so the total feature weight — and therefore the typical squared distance that `gamma` multiplies —
  stays what it was for all five families. Without that rescaling a single-family run would silently be a run at a
  5x smaller effective gamma and the comparison would measure the wrong thing. This is a plain 5-fold CV, not nested:
  it is a descriptive ranking of families within a region, the same slight optimism applies to every cell, and no
  model choice is made from it. The linear SVM's per-family share of `|w|` is reported next to it as a cheap
  cross-check, and the unified column is comparable to the Phase 2 family table (`results/tables/features_family_comparison_256_clahe.csv`).

## 5.3 Error analysis

- **Primary on the out-of-fold CV scores, secondary on test.** The test set has 143 fractured images; the training
  out-of-fold scores cover 576 fractured images with the same honest "never seen by the model that scored it"
  property. Error analysis on 143 positives split over four regions cannot say anything, so the CV scores carry it and
  the test numbers are shown next to them only as a consistency check.
- **Agreement**: per region, the 2x2 of unified-correct x region-correct at each model's tuned threshold, split into
  both right / only unified right / only region right / both wrong. This is what McNemar reads in 5.1, shown as counts.
- **What the errors look like**: among fractured images, false negatives vs true positives; among non-fractured
  images, false positives vs true negatives. Compared on the available proxies, since FracAtlas has no fracture-type
  or view labels:
  `fracture_count`, `n_boxes`, **box area fraction** (largest box area / image area, and total box area fraction),
  `hardware`, `multiscan`, image size (`width*height`), and — if `src/phase5_box_contrast.py` has been run —
  **box-local contrast** (the contrast inside the box against a surrounding ring, on the same 256 px CLAHE
  preprocessing the features use).
- **How a proxy is judged**: for a continuous proxy, the rank-based AUC of that proxy for separating detected from
  missed positives, with a bootstrap CI; for a binary proxy, the detection rate in each group with a CI on the
  difference. An AUC near 0.5 means the proxy says nothing. Medians and IQRs are tabulated alongside so the direction
  is readable.
- **"Joint-adjacent" is not available.** FracAtlas gives no joint annotation and no view label, and a bounding box
  cannot be turned into "near a joint" without one, so that part of the plan's 5.3 is **reported as not testable**
  rather than approximated by something that does not mean it. "Subtle" and "low contrast" are tested, by box area
  fraction and by box-local contrast.
- Per-region proxy distributions are also reported for *all* images, not only errors, because the regions differ in
  these proxies to begin with (hand boxes are a larger fraction of the image than leg boxes), and that difference is
  itself a candidate explanation for the region differences in 5.1.

## Decision rules (fixed before seeing Phase 5 results)

1. **Nothing in Phase 5 changes a model, a threshold or a Phase 3 / Phase 4 verdict.** Phase 5 explains and
   quantifies; any change it motivates belongs to Phase 6 and must be re-validated there. The Phase 4 verdicts stand
   as written in `docs/phase4_regional.md`.
2. **Learning-curve reading.** For a region, comparing the two arms at the largest `p`:
   - arms **converged** (curves within each other's fold-SD at `p = 1`) and the region arm **still rising** (its last
     two points differ by more than its fold-SD) -> the evidence says *"per-region data is small"*, and the Phase 6
     lever is sharing data across regions (the proposed multi-task kernel) rather than separate models;
   - arms converged and the region arm **flat** -> the regions are *not* detectably different for this feature set and
     model, and a region-specific model has nothing left to gain;
   - region arm **above** the unified arm at `p = 1` by more than the fold-SD -> a genuine region effect, and the
     amount of data at which it appears is the number to quote.
   Any such statement is reported per region with the fold-SD, and is called a *tendency*, not a significant
   difference, unless a paired interval over folds excludes 0.
3. **The transfer matrix decides "do the regions share a signal", not the learning curve alone.** A region pair counts
   as sharing if the size-matched transfer ratio is above 0.5 in both directions; as asymmetric if it is above 0.5 in
   one direction only (and that asymmetry is then reported as a finding, since it bears directly on the Phase 6
   design); as not sharing if below 0.5 both ways. The 0.5 cut is arbitrary and is fixed here so it cannot be chosen
   after the fact; the raw numbers are always shown.
4. **No proxy is promoted to a cause.** The proxies (box size, hardware, multiscan, image size, contrast) are
   correlated with each other and with region. A proxy is reported as "associated with being missed" when its CI
   excludes the null, and explicitly *not* as the reason the model fails. Where two proxies are confounded (hardware
   is on 97/99 fractured images; multiscan is 0% in hip and 13.5% in leg) that is stated with the number.
5. **The hip question gets an answer in words, with its power stated.** The open question (hip clearly better
   region-specific in CV, not on test) is answered with: the fold-level spread of the hip CV difference, the hip
   learning curve and transfer row, the width of the test CI (that is, the smallest effect the 11 fractured hip test
   images could have detected), and the already-known fact that hip has no `multiscan` images so near-duplicate
   leakage cannot explain it. If those do not separate "sampling noise" from "a real hip effect", the conclusion is
   **"not resolvable with this test set"**, stated as such, and the Phase 6 plan does not assume the hip gain is real.
6. **Reproducibility.** Every fitting run in Phase 5 goes into `results/experiment_log.csv` via `src/expt_log.log_run`
   with `test_metrics="not used (CV only)"`, every table into `results/tables/phase5_*.csv`, every figure into
   `results/figures/phase5_*.png`, and the commands into `README.md`. Seed 42 for models, seed 0 for bootstraps and
   subsample draws, as in Phases 3-4.

## Known threats (carry into the report)

- **Fixed hyperparameters in 5.2.** The learning curves, the transfer matrix and the family ablation all fix
  C=10, gamma=0.05, chosen in Phase 3 on all training images. They are diagnostics, not model selection, and both
  arms carry the same advantage; but a curve for a region whose own best hyperparameters are far from those values
  (shoulder, per Phase 4) will understate that region's achievable performance.
- **Small regions.** Hip and shoulder have 46-48 fractured training images. Their learning curves start at 2-3
  positives per draw and their transfer rows rest on 30 positives per cell. Fold-SDs there will be large and are
  reported rather than smoothed.
- **No patient ID.** Views of one patient can sit in different folds, so every CV number here, like every earlier one,
  may be optimistic. Hip has no `multiscan` images, hand 8.6% and leg 13.5%, so this threat is unevenly distributed
  across exactly the comparison being made.
- **Region label assumed known.** As in Phase 4, the region-specific arm and the transfer matrix assume the region of
  an incoming image is known.
- **Descriptive CV, not nested, in the family ablation and the region-separability model.** Those numbers are slightly
  optimistic in absolute terms; only their ordering within a region is used.
- **t-SNE is not evidence.** Stated again because it is the figure most likely to be over-read in a report.
- **The proxies are not the thing.** "Subtle" is being measured by box area fraction and local contrast because
  FracAtlas has no subtlety label. A null result on those proxies is not evidence that subtlety does not matter.

**[added after results] Which of these actually bit.** Three of them did, and the report should say so rather than
list them as hypotheticals:

1. **Small regions / large fold SDs** bit hardest, on shoulder. Its fold SD (0.123) is larger than any effect its
   learning curve could show, so decision rule 2 returns "flat" for a curve that climbs from 0.355 to 0.604 and has
   plainly not plateaued. The pre-registered verdict is recorded as it came out, with the low power stated beside it.
   Hip's SD (0.148) is the largest of the four, but there the effect (+0.209) is larger still, so the rule fired.
2. **No patient ID** stopped being hypothetical: 5.3 measured multi-scan images as +0.14 easier to detect. The threat
   is real, it is quantified, and its uneven distribution (hand 8.6%, leg 13.5%, hip and shoulder 0%) turned out to
   matter for the central question - in hip's favour, as it happens.
3. **Fixed hyperparameters** cost nothing detectable in the end: the region arm at full size reproduced the Phase 4
   fixed-hyperparameter run exactly, and Phase 4 had already shown those values cost shoulder little (0.560 -> 0.604).
   The caveat stands for shoulder, whose own optimum is furthest from C=10, gamma=0.05.

The other threats were carried as written and none was contradicted. t-SNE, in the event, carries no claim in this
document: every statement about region structure rests on the region classifier, the transfer matrix or the cosine
matrix instead.

---

## Results

*(the protocol above was fixed before any result. 5.1, 5.2, 5.3 and 5.4 are complete and all scripts have
been run on the real data.)*

### 5.1 The main result

Tables: `results/tables/phase5_main_cv.csv`, `phase5_main_test.csv`, `phase5_mcnemar.csv`,
`phase5_agreement.csv`. Figures: `phase5_main_pr_auc.png`, `phase5_main_forest.png`,
`phase5_test_pr_curves.png`, `phase5_score_dists.png`. `phase5_main.py` recomputes the Phase 4 paired
differences from the saved scores and asserts it reproduces them before plotting; the check passes for every
region and metric, so the figures and the Phase 4 text cannot have drifted apart.

**Ranking (PR-AUC): the Phase 4 verdicts are unchanged.** Region-specific minus unified, 95% paired bootstrap:

| Region | Train pos. | Test pos. | CV PR-AUC diff | Test PR-AUC diff |
|---|---|---|---|---|
| Hand | 305 | 76 | -0.018 [-0.047, +0.018] | +0.045 [-0.028, +0.114] |
| Leg | 170 | 42 | +0.023 [-0.013, +0.063] | +0.013 [-0.066, +0.088] |
| Hip | 46 | 11 | **+0.223 [+0.107, +0.325]** | -0.096 [-0.251, +0.041] |
| Shoulder | 48 | 12 | -0.077 [-0.149, +0.006] | -0.044 [-0.176, +0.082] |

**The new result of 5.1: the two approaches rank almost identically but DECIDE very differently.** McNemar on
the test decisions at each model's own tuned threshold (exact binomial; "only X right" are the discordant
pairs):

| Region | Both right | Only region-specific | Only unified | Discordant | p |
|---|---|---|---|---|---|
| Hand | 101 | 20 | **71** | 91 | 7.2e-08 |
| Leg | 342 | **30** | 11 | 41 | 4.3e-03 |
| Hip | 42 | 6 | 9 | 15 | 0.61 |
| Shoulder | 45 | 8 | 5 | 13 | 0.58 |

Hand and leg are both highly significant and they point in **opposite directions**. That is not a difference
in what the models learned — their PR-AUCs are within noise in both regions — it is a difference in where
each one's F2 threshold landed:

| | Hand test | | Leg test | |
|---|---|---|---|---|
| | region-specific | unified | region-specific | unified |
| Recall | 0.974 | 0.776 | 0.667 | 0.714 |
| Specificity | **0.261** | 0.628 | **0.898** | 0.843 |
| Precision | 0.280 | 0.411 | 0.418 | 0.330 |

The hand region model calls almost every hand image fractured (specificity 0.26); the leg region model is
markedly more conservative than the unified one. The same pattern is in the out-of-fold scores (hand
specificity 0.369 region vs 0.514 unified; leg 0.899 vs 0.776), so it is not a test-set accident. Both
thresholds were chosen by the same F2 rule on the same kind of inner out-of-fold scores — the region-specific
one just had one region's images to choose from instead of all 3,266.

**So the honest one-line summary of Phases 3-5 so far is: with equal tuning, splitting the model per region
changes the decision threshold much more than it changes the learned ranking.** That is a finding about
where the cost of specialising actually falls, and it is a direct input to Phase 6: per-region *thresholds*
on the single unified model are a much cheaper lever than per-region *models*, and most of the routed
system's recall/specificity trade (Phase 4: recall 0.844 vs 0.759, specificity 0.708 vs 0.773) is reachable
without training four models. Phase 6 must validate that properly; nothing here changes any Phase 3/4 model.

### 5.2 Separating the causes

Run on the real features, 1,260 learning-curve fits + 150 transfer fits in 11.1 min and 275 family-ablation
fits in 0.7 min. **The correctness gate passed exactly**: the region arm at full training size reproduces all
20 region x fold PR-AUCs of `results/phase4/oof_rbfsvm__none__fixed.csv` with a largest difference of
0.00e+00, so this script's fitting is the same fitting Phase 4 did.

Tables: `phase5_learning_curves.csv`, `phase5_transfer.csv`, `phase5_region_separability.csv`,
`phase5_cosine.csv`, `phase5_family_importance.csv`. Figures: `phase5_learning_curves.png`,
`phase5_transfer.png`, `phase5_region_separability.png`, `phase5_fracture_direction.png`,
`phase5_family_importance.png`, `phase5_feature_space_pca.png`, `phase5_feature_space_tsne.png`.

#### Learning curves: the answer is different for hip than for everything else

PR-AUC on the held-out fold, mean over the 5 outer folds (+/- fold SD). "Zero-shot" is the unified arm at
p = 0: a model that has seen **no image of that region at all**.

| Region | Chance | Zero-shot (unified, p=0) | Unified at p=1 | Region-specific at p=1 | Gap | Rule 2 verdict |
|---|---|---|---|---|---|---|
| Hand | 0.30 | 0.528 | 0.637 +/- 0.070 | 0.621 +/- 0.054 | -0.016 | converged, flat -> **not detectably different** |
| Leg | 0.10 | 0.430 | 0.599 +/- 0.064 | 0.614 +/- 0.077 | +0.015 | converged, flat -> **not detectably different** |
| Hip | 0.18 | **0.267** | 0.523 +/- 0.118 | **0.733 +/- 0.148** | **+0.209** | **genuine region effect** |
| Shoulder | 0.18 | 0.523 | 0.647 +/- 0.127 | 0.604 +/- 0.123 | -0.043 | converged, flat -> not detectably different |

**The zero-shot column is the single most informative number in Phase 5.** A unified model trained without a
single hand image already reaches 0.528 on hand, against 0.621 for a model trained on all 244 hand
positives - about 85% of the way there from nothing. For leg it is 0.430 against a 0.10 chance level, for
shoulder 0.523 against 0.604. **For hip it is 0.267 against a chance level of 0.18**: a model built from the
other three regions is barely better than guessing on hip, while hip's own 204 images reach 0.733.

**Hip's gap widens with data rather than closing** (+0.083 at p=0.05, +0.148 at p=0.35, +0.209 at p=1.00).
That is the opposite of the data-scarcity signature, which would show the two arms converging as the region
arm catches up. Hip is the one region where more of its own data makes specialising *more* worthwhile, and
the curve is still climbing at the full 204 images, so 0.733 is a lower bound on what hip-specific modelling
could reach.

For hand, leg and shoulder the two arms converge and the pre-registered test returns "flat". One honest
qualification: **for shoulder the rule returns "flat" because the fold SD is large (0.123), not because the
curve looks flat** - it climbs from 0.355 at p=0.05 to 0.604 at p=1 and visibly has not plateaued. The rule
was fixed in advance and its verdict stands as recorded, but it has little power at shoulder's sample size,
and "shoulder needs more data" remains plausible and untested rather than refuted.

#### Cross-region transfer: hip is an isolate

Size-matched (every training region cut to 30 fractured / 140 not, 5 draws), as transfer ratio - 1.00 means
"as good as that region's own model", 0.00 means "no better than chance on that region":

| trained on \ evaluated on | Hand | Leg | Hip | Shoulder |
|---|---|---|---|---|
| **Hand** | +1.00 | +0.46 | +0.16 | +0.52 |
| **Leg** | +0.99 | +1.00 | +0.15 | +0.27 |
| **Hip** | +0.16 | +0.00 | +1.00 | -0.03 |
| **Shoulder** | +0.51 | +0.18 | +0.05 | +1.00 |
| **All regions** | +0.87 | +0.67 | +0.26 | +0.66 |

Applying rule 3's 0.50 cut, fixed before these numbers existed:

- **share** (both directions above 0.50): hand <-> shoulder (+0.52, +0.51).
- **asymmetric** (one direction only): hand <-> leg. Leg's images transfer to hand almost perfectly (+0.99)
  while hand's transfer to leg only moderately (+0.46). Leg is the larger and more varied set (1,699 images,
  box sizes IQR 14-26 px against hand's 11-18), so a leg-trained model appears to cover hand's narrower
  regime while the reverse does not hold - which is a finding about the *data*, not about anatomy.
- **do not share**: every pair involving hip (all six ratios between -0.03 and +0.16), and leg <-> shoulder
  (+0.27, +0.18).

The hip row and the hip column are both near zero in every off-diagonal cell. Hip neither borrows from nor
lends to any other region. Even "all regions" - 2,600 images - reaches only +0.26 on hip size-matched and
+0.62 raw, against hip's own +1.00 from 204 images.

#### Why: the regions are visibly separable, and hip's fracture direction is orthogonal to the pooled one

- **Region separability** (multinomial logistic, 5-fold CV): **macro-F1 0.765, accuracy 0.940**. The regions
  are almost perfectly readable from the same features. Hip is recovered 234/256 times, confused only with
  leg (18); the largest confusion is hand <-> leg (73 + 50). **This matters for Phase 6**: the unified RBF
  SVM *could* in principle condition its behaviour on region without being told the region, because the
  information is plainly there - and on hip it evidently does not (unified 0.523 vs region-specific 0.733).
  The headroom the proposed multi-task kernel is aiming at is therefore real and is not just a re-encoding of
  something the unified model already has.
- **Cosine similarity of the per-region linear fracture directions** (descriptive, linear shadow of the RBF
  model): hand-leg 0.155, hand-shoulder 0.027, leg-shoulder 0.032, and hip against the other three **-0.063,
  -0.017, -0.104**. Against the pooled direction: hand 0.625, leg 0.555, shoulder 0.289, **hip 0.193**. In
  8,641 dimensions two unrelated vectors sit near 0, so the ordering is what matters, and the ordering is
  unambiguous: the pooled boundary is a compromise dominated by the two large regions, and hip - 256 of 3,266
  training images, 7.8% - barely influences it while needing a direction nearly orthogonal to it.
- Together with 5.3's finding that **hip fractures are about twice the linear size of every other region's**
  (median box side 31.6 px against 14-19 px), there is a coherent mechanism rather than a bare correlation:
  hip's lesions are large enough for the pooled descriptors to represent directly, they occupy a different
  part of feature space, they require a different decision direction, and hip contributes too little data to
  pull the unified boundary towards it.

#### Feature families per region

PR-AUC above that scope's chance level, one family alone (left), and the change from dropping one family
from all five (right, negative = the family was helping). Plain 5-fold CV, a ranking within a row only.

| Scope | Gabor | GLCM | HOG | Intensity | LBP |
|---|---|---|---|---|---|
| All regions | 0.285 | 0.263 | 0.327 | 0.176 | **0.343** |
| Hand | 0.235 | 0.203 | **0.270** | 0.207 | 0.262 |
| Leg | 0.280 | 0.268 | 0.430 | 0.262 | **0.468** |
| Hip | 0.261 | 0.196 | 0.411 | 0.306 | **0.471** |
| Shoulder | 0.369 | **0.429** | 0.335 | 0.272 | 0.371 |

- **Shoulder is the exception**: GLCM alone (0.429) beats every other family there and beats GLCM anywhere
  else, and dropping LBP (+0.043) or intensity (+0.038) *improves* shoulder. Shoulder is the one region whose
  preferred representation differs - which did not translate into a region-specific advantage, because its
  learning curve has not converged.
- **Hand is uniformly weak**: every family lands between 0.203 and 0.270, the flattest profile of any region,
  and no leave-one-out change exceeds 0.017. There is no family that works on hand, consistent with 5.3's
  finding that hand fractures sit at the resolution limit (median 14 px against a 16 px HOG cell).
- **Leg and hip rely on LBP and HOG** (0.47 and 0.41-0.43). Hip is also the only scope where dropping
  families clearly hurts (HOG -0.068, LBP -0.068, Gabor -0.049, intensity -0.036): hip uses several families
  jointly, while everywhere else the families are largely redundant (all-regions leave-one-out changes are
  -0.021 to 0.000).
- The all-regions column is consistent with the Phase 2 family comparison, as expected.

### 5.3 Error analysis

Tables: `results/tables/phase5_error_proxies.csv`, `phase5_proxy_by_region.csv`; per-image decisions and
proxies in `results/phase5/error_cases.csv`. Figures: `phase5_error_agreement.png`,
`phase5_error_proxies.png`, `phase5_recall_vs_size.png`. Primary numbers are out-of-fold (569 fractured
images in the four modelled regions); the test set (141) is the consistency check.

**How big a fracture actually is, in the image the model sees.** `box_px_256` is the geometric mean side of
the largest annotated box after the Phase 2 preprocessing (pad to square, resize to 256, CLAHE); it is
verified exact against `preprocess`'s own geometry (largest absolute difference 0.000 px).

| Region | Median box side (px of the 256 input) | IQR | Box area / image area | Multi-fracture share | Multiscan share |
|---|---|---|---|---|---|
| Hand | **14.0** | 11.3-17.5 | 0.4% | 0.24 | 0.086 |
| Leg | 19.0 | 14.1-25.7 | 0.7% | 0.33 | 0.135 |
| Hip | **31.6** | 22.8-40.1 | 1.9% | 0.05 | 0.000 |
| Shoulder | 15.3 | 11.6-19.0 | 0.4% | 0.07 | 0.000 |

**The HOG cell is 16 px.** The median hand and shoulder fracture is therefore smaller than the cell its
gradient histogram is pooled over: it cannot occupy a HOG bin of its own and can only appear as a small
perturbation of one cell's histogram, diluted by whatever else is in that cell. This is the most concrete
statement this project can make about why handcrafted features struggle here, and it is measured, not
assumed. Hip is the one region where the lesion is comfortably larger than the cell.

**Box-local contrast** (`src/phase5_box_contrast.py`, all 719 fractured images, measured on the same 256 px
CLAHE input the features use; medians):

| Region | Box SD | Ring SD | Contrast ratio | Weber | Michelson | Box side (px, from the true geometry) |
|---|---|---|---|---|---|---|
| Hand | 0.132 | 0.165 | 0.809 | 0.247 | 0.474 | 14.046 |
| Leg | 0.133 | 0.170 | 0.813 | 0.186 | 0.560 | 18.968 |
| Hip | 0.130 | 0.145 | **0.889** | 0.146 | **0.639** | 31.582 |
| Shoulder | 0.103 | 0.122 | 0.854 | 0.170 | 0.400 | 15.328 |

The contrast ratio is **below 1 in every region**: the annotated fracture region is consistently *less*
textured than the ring of anatomy around it, by 11-19%. A fracture is not a locally busy patch against a
quiet background — it is a subtle disruption inside a region that is already structured, which is the
unfavourable case for descriptors that summarise texture energy over a cell. Hip again sits apart, with the
contrast ratio closest to 1 (0.889) and the highest Michelson contrast (0.639).

This run also **independently confirms the size measurement**. `box_px_256_geom` is computed from
`preprocess`'s actual output geometry, by a completely different route from the metadata formula used
elsewhere in 5.3, and the two agree across all 719 images to three significant figures (hand 14.046 vs
14.0, leg 18.969 vs 19.0, hip 31.582 vs 31.6, shoulder 15.328 vs 15.3).

**What separates a missed fracture from a detected one** (rank AUC, 95% bootstrap CI, out-of-fold; 0.5 = the
proxy says nothing). Unified model, with the region-specific model in agreement throughout:

| Proxy | Hand | Leg | Hip | Shoulder | Overall |
|---|---|---|---|---|---|
| Fracture size | 0.48 [0.40, 0.56] | **0.69 [0.60, 0.78]** | 0.63 [0.40, 0.82] | 0.25 [0.06, 0.48] | 0.52 [0.47, 0.58] |
| Fracture count | **0.57 [0.52, 0.62]** | **0.66 [0.59, 0.72]** | 0.48 [0.36, 0.56] | 0.53 [0.50, 0.56] | **0.58 [0.55, 0.62]** |
| Image size | 0.52 [0.47, 0.55] | **0.54 [0.50, 0.58]** | 0.58 [0.53, 0.65] | **0.71 [0.63, 0.78]** | **0.55 [0.52, 0.57]** |
| Aspect ratio | **0.38 [0.30, 0.46]** | **0.41 [0.34, 0.46]** | 0.40 [0.29, 0.50] | **0.32 [0.22, 0.43]** | **0.40 [0.35, 0.44]** |

Binary proxies, as a difference in detection rate (null 0): hardware **+0.12 [+0.03, +0.19]** overall
(leg +0.23, shoulder +0.20), multiscan **+0.14 [+0.06, +0.21]** overall (hand +0.18).

**The contrast proxies, and what they rule out.** Same format, out-of-fold, after joining
`phase5_box_stats.csv`:

| Proxy | Region-specific (overall) | Unified (overall) | Strongest cell |
|---|---|---|---|
| Box texture (`box_std`) | **0.625 [0.564, 0.684]** | **0.593 [0.530, 0.650]** | shoulder 0.806 [0.681, 0.912] |
| Ring texture (`ring_std`) | **0.596 [0.532, 0.659]** | **0.588 [0.526, 0.649]** | shoulder 0.741, leg 0.695 |
| Weber contrast | **0.632 [0.573, 0.687]** | **0.556 [0.495, 0.619]** | leg 0.661 / 0.672 |
| Michelson contrast | **0.561 [0.495, 0.625]** | **0.576 [0.517, 0.635]** | hip (unified) 0.769 [0.566, 0.938] |
| **Contrast ratio** (box/ring) | 0.523 [0.463, 0.582] | 0.500 [0.439, 0.559] | — nothing anywhere |

This pattern is more informative than any single row. Box texture and ring texture are each clearly
associated with detection, by almost the same amount and in the same direction — **and their ratio is flat
at exactly the null**. So what predicts detection is not the fracture standing out from its surroundings; it
is **how much texture energy is in the neighbourhood at all**. A quiet patch of bone is missed whether or not
the fracture inside it is distinctive. That is a deflating reading of the model's competence and it is a
direct corroboration of the 5.4 argument: a descriptor that pools gradient or co-occurrence energy over a
cell responds to the presence of energy, not to the structure of the disruption.

**It also corroborates 5.2's hip result from an independent direction.** On hip, the *unified* model's errors
are strongly predicted by texture (`box_std` 0.783 [0.612, 0.924], Michelson 0.769 [0.566, 0.938],
`ring_std` 0.733) while the *region-specific* model's are not (0.425, 0.367, 0.425 — all intervals spanning
0.5). The unified model, which 5.2 showed has not learned hip's actual signal (0.523 against the region
model's 0.733), is falling back on generic texture energy there. The hip-specific model is doing something
else. Two analyses built on different quantities — held-out PR-AUC under subsampling, and the correlates of
individual errors — point at the same conclusion.

Five things follow, all of them narrower than the literature-expected story:

1. **"Small fractures are missed" is a leg effect, not a general one.** In leg, recall rises from 0.36 in the
   smallest size bin to 0.80 in the largest. In hand it is flat (AUC 0.48), and in shoulder it points the
   other way with a wide interval. Four regions, one clear effect: the report should say that rather than the
   general version. A plausible reading, which this project cannot test, is that hand fractures are *all*
   near the resolution limit (median 14 px, IQR 11-18), so there is no large-fracture end of the range left
   for size to discriminate on; leg spans a much wider range (IQR 14-26).
2. **Multi-scan images are detected more often** (+0.14 overall). This is the no-patient-ID threat showing up
   as a measurable effect rather than a hypothetical one, and it is unevenly distributed across exactly the
   comparison being made: hand 8.6%, leg 13.5%, **hip and shoulder 0%**. It therefore cuts *against* the
   sceptical reading of the hip CV result — hip is the one region whose CV number cannot be inflated by
   near-duplicate views, so leakage is not what made the hip region model look good.
3. **Hardware images are detected more often** (+0.12 overall) — but 97 of the 99 hardware images are
   fractured, so a model that keyed on implants would both score well here and be useless. Phase 3 already
   checked the other side of this (test recall 0.756 without hardware, 0.750 with), so there is still no sign
   of an implant shortcut; the association is reported, and nothing is concluded from it (decision rule 4).

4. **Local texture energy, not fracture distinctiveness, is the strongest single correlate of detection**
   (box 0.625, ring 0.596, ratio flat at 0.523) — see the contrast table above.
5. **Shoulder's errors are the most predictable of any region, and not from the fracture at all**: box
   texture 0.806 [0.681, 0.912] and image size 0.742 [0.657, 0.824], while fracture size points the wrong
   way (0.34). Whatever determines whether a shoulder fracture is found, it is a property of the image
   rather than of the lesion. That fits 5.2, where shoulder was the region with an idiosyncratic feature
   profile (GLCM alone 0.429, the best single-family result anywhere) and a learning curve that had not
   converged.

The aspect-ratio and image-size associations are consistent across regions and almost certainly stand in for
acquisition device or protocol rather than for anything anatomical. They are reported because they are real
and because they are a caution: the model has access to study-level nuisance information, and some of its
apparent skill may rest on it. The same caution applies with more force to finding 4 — "detects texture" is
a mechanism that would transfer badly to a new scanner.

**"Joint-adjacent" is reported as not testable.** FracAtlas has no joint annotation and no view label, and a
bounding box cannot be turned into "near a joint" without one. It is left out rather than approximated by
something that does not mean it.

### 5.4 Theory write-up

**Bias-variance.** The two arms are the textbook trade in a concrete form. The unified model pools 3,266
images and accepts that one decision boundary must serve four anatomies — higher bias if the regions truly
differ, lower variance. A region model removes that bias but estimates a boundary in 8,641 dimensions from as
few as 46 fractured images — lower bias, much higher variance. The learning curves in 5.2 separate the two
terms directly, and they come out differently per region, which is the main theoretical result of the phase.

For hand, leg and shoulder the pooling bias is **small**: a model that has never seen the region already
reaches 0.43-0.53 PR-AUC on it, the two arms converge, and specialising buys nothing measurable. What those
regions pay is variance, and Phase 4 and 5.1 locate it in two places. Phase 4 found that fixing
hyperparameters at the Phase 3 values recovered most of shoulder's apparent loss (0.560 -> 0.604), so a large
part of what looked like "shoulder is different" was the noise of choosing C and gamma on 9-10 positives per
inner fold. 5.1 found a second and larger variance cost in the **decision threshold**: an F2 threshold is a
one-parameter estimate, but fitted on one region's inner out-of-fold scores it is a noisy one, and it is what
produced a hand model with specificity 0.26. The ranking survived specialisation; the calibration did not.

For hip the balance is reversed and the pooling bias **dominates**. The unified boundary is a compromise
weighted by region size, hip supplies 7.8% of the images, and hip's own fracture direction is near-orthogonal
to the pooled one (cosine 0.193, against hand's 0.625). The result is a bias so large that 204 hip images
beat 2,600 mixed ones (0.733 vs 0.523) and a model trained without hip barely clears chance on it (0.267 vs
0.18). Hip is also the region where the variance penalty is most visibly being paid at the same time — its
fold SD is the largest of the four (0.148) and its curve is still climbing at full size — which is exactly
what the bias-variance picture predicts for a region that genuinely needs its own boundary but has the least
data to estimate one with. **The same architecture therefore sits on opposite sides of the trade-off
depending on the region, and a single global choice between "one model" and "four models" is the wrong
question**; that is the finding Phase 6's shared-plus-region-specific kernel is meant to exploit.

**Curse of dimensionality and regularisation.** With 8,641 features and 3,266 training images the problem is
p >> n even pooled, and p/n is about 34 for hip. Three choices in this project are responses to it, and each
has a measurable trace: the equal-family block weighting (without it HOG's 8,100 columns would set the
geometry by sheer count, and Phase 2 found L1 selection kept nothing once it was applied); the margin-based
RBF SVM, whose complexity is controlled by C and the margin rather than by the number of columns; and the
choice not to reduce dimension, since Phase 2 found PCA (0.541) and tree selection (0.544) no better than the
full set (0.552). The block weighting is also why the family-subset analysis in 5.2 has to rescale the
weights: a subset would otherwise be a run at a different effective gamma, and the ablation would measure
scaling rather than information.

**Class imbalance and the choice of metric.** The fracture rate is 0.176 overall and 0.10 in leg, so accuracy
is worthless as a target: "never fracture" scores 0.824 accuracy, 0.5 ROC-AUC and 0 recall, and PR-AUC equal
to the prevalence. PR-AUC is the primary metric because its chance level *is* the prevalence and it ignores
the true negatives that dominate the set; it is compared between models within a region and never across
regions, because hand's chance level (0.30) and leg's (0.10) are different. F2 is the operating-point metric
because a missed fracture costs more than a false alarm in a screening role. Phase 3 found that the three
imbalance strategies (none, class weights, SMOTE) are equivalent once the threshold is tuned, which is the
expected result: for a margin-based classifier, reweighting the classes and moving the decision threshold are
two ways of asking the same question, and only one of them costs a refit. The place imbalance still bites is
the *variance* of the threshold estimate, which is what 5.1 measured.

**Why handcrafted features struggle with subtle fractures.** 5.3 makes this quantitative rather than
rhetorical. HOG, LBP, GLCM and Gabor are all *pooled* descriptors: each summarises a cell, a grid square or
the whole image, and the pooling is what gives them their invariance and their manageable size. A fracture
whose median extent is 14 px in hand and 15 px in shoulder, against a 16 px HOG cell, is below the scale at
which that pooling preserves it — the lesion contributes a small perturbation to one cell's histogram and is
averaged against everything else in that cell. Raising the resolution does not fix it: Phase 2 tried 448 px
with 28 px cells and scored 0.537 against 0.552, because the cell grows with the image. 5.3's contrast analysis closes the argument from the other end: detection is predicted by how much texture
energy sits in the neighbourhood (box 0.625, ring 0.596) and **not at all** by how much the fracture stands
out from it (ratio 0.523, interval spanning the null). A pooled descriptor responds to the presence of
energy rather than to the structure of the disruption, which is exactly what those three numbers say when
read together. What would be needed is a representation that is local and adaptive rather than pooled on a
fixed grid — which is precisely what a learned convolutional feature extractor provides, and which this
project excludes by design (`docs/design_note.md`). That is the honest form of this section's conclusion: the ceiling observed here is a
property of fixed pooled descriptors at this lesion scale, and the classical model choice is not what is
binding.

### The hip question (decision rule 5)

Open question from Phase 4: hip was clearly better region-specific in CV (+0.223 [+0.107, +0.325], better in
5/5 folds) and not on test (-0.096 [-0.251, +0.047]). What is settled so far:

- **Near-duplicate leakage is ruled out**, and 5.3 strengthens this rather than merely restating it.
  Multiscan images are measurably easier (+0.14 overall), and hip has none, so hip is the region whose CV
  number is *least* inflated by that mechanism.
- **The test had almost no power to confirm it.** The hip test CI spans 0.30 PR-AUC on 11 fractured images.
  An effect of +0.22 would have had to be enormous and lucky to clear that interval; the observed -0.096 is
  comfortably inside the range a true +0.22 could produce on 11 positives. CV and test are *not* in conflict
  — the test simply did not measure anything at that sample size.
- **There is a data-based candidate for a real hip effect**: hip fractures are about twice the linear size of
  the other regions' (31.6 px vs 14-19 px) and hip has the lowest multi-fracture share (0.05). If hip's
  advantage is real, "the lesion is large enough for the pooled descriptors to represent it" is a mechanism
  that predicts it.

**Verdict (rule 5), now that 5.2 has run.** The question has to be split in two, because the evidence
answers one half decisively and leaves the other half open, and collapsing them would overclaim.

**(a) Was the CV result sampling noise? No - this is settled.** Five independent lines of evidence, none of
which depends on the 46 hip training positives being a lucky draw:

1. the hip region arm beats the unified arm at **every** training size, from 10 images upward, and the gap
   *widens* monotonically (+0.083 -> +0.209) instead of closing;
2. the gap at full size (+0.209) exceeds the fold SD (0.148), which is rule 2's pre-registered threshold for
   a genuine region effect;
3. a model trained on the other three regions scores 0.267 on hip against a 0.18 chance level - the shared
   signal the unified model relies on essentially does not exist for hip;
4. all six single-region transfer ratios into and out of hip lie between -0.03 and +0.16 (the pooled
   "all regions" row does better but still reaches only +0.26 size-matched, against hip's own +1.00);
5. there is a mechanism, not just a correlation: hip's fracture direction is near-orthogonal to the pooled
   one (0.193 against hand's 0.625), its lesions are twice the linear size of any other region's, and it
   contributes 7.8% of the training images.

Leakage was already ruled out and 5.3 strengthened that: multi-scan images are measurably easier (+0.14) and
hip has none, so hip's CV number is the *least* inflated of the four.

**(b) Does it generalise to unseen hip data? Not established, and this test set cannot establish it.** The
hip test difference was -0.096 with a CI spanning 0.30 PR-AUC on 11 fractured images. An interval that wide
is consistent with a true +0.21 and with a true -0.21 alike, so the test neither confirms nor contradicts
(a) - it did not measure anything at that sample size. The CV and test results are **not in conflict**; one
is an estimate and the other is noise around it.

**What Phase 6 may and may not do with this.** It may treat hip as the region where sharing fails and design
for it - that is what the 5.2 evidence supports, and it is a training-set finding that needed no test data.
It may **not** cite the Phase 4 hip test number as support, and it may not claim the hip advantage has been
shown to generalise. Any Phase 6 mechanism that exploits it has to be validated on its own terms, and the
report should state plainly that confirming a hip-specific effect on held-out data would need more than 11
positives.
