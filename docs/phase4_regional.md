# Phase 4: region-specific classifiers (COMPLETE, 2026-11-10)

Task: one image-level fracture classifier per region (hand, leg, hip, shoulder), each trained only on that region's
training images, with **the same features, model family, grid, CV protocol and imbalance strategy as the Phase 3 unified
model**, and compared with the unified model **on the same images**. **This file records the protocol and the decision
rules before any Phase 4 result was seen**; they were followed as written. The one protocol change (fold-mean instead of
pooled PR-AUC for the CV comparison) was made after a synthetic dry run, before any real result. Status: `docs/PROJECT_STATUS.md`.

**Result in one line:** with equal tuning, region-specific RBF SVMs are not detectably better or worse than the unified
model on the test set in any region. In CV hip looked clearly better (+0.22 PR-AUC), but the test (11 fractured hip images)
did not confirm it.

## Code
| File | Role |
|---|---|
| `src/phase4_regional.py` | CV only (training images). For each region: `cv_protocol.nested_cv` on that region's training rows, with the region's own slice of the frozen folds. Saves `results/phase4/oof_<run>.csv`, `run_<run>.json`; `--summarize` builds the unified-vs-region tables. |
| `src/phase4_final_test.py` | The single test evaluation of the four region models (lock file `results/phase4/TEST_EVALUATED.json`), compared with the unified model's saved test scores (`results/phase3/test_scores.csv`) on the same images. |
| reused unchanged | `model_zoo.py` (`build`, `SPECS`, grids), `cv_protocol.py` (`nested_cv`, `tune_cv`, `edge_warnings`), `evaluate.py` (metrics, `bootstrap_ci`, `paired_bootstrap`) |

## Data per region (training images, frozen folds 0-4)
| Region | Train images | Fractured | Fractured per fold | Test images | Test fractured |
|---|---|---|---|---|---|
| Hand | 1,025 | 305 | 61 | 256 | 76 |
| Leg | 1,699 | 170 | 34 | 425 | 42 |
| Hip | 256 | 46 | 9-10 | 64 | 11 |
| Shoulder | 267 | 48 | 9-10 | 67 | 12 |
| Other | 19 | 7 | 1-2 | 5 | 2 |

"Other" is too small to model; it is not part of any Phase 4 comparison (the unified model still covers it in Phase 3).
The region label comes from the metadata (`region_group`), so the region-specific system assumes the region is known
when an image arrives. That is realistic for a radiograph (the study is ordered for a body part) but it is an assumption.

## Protocol
- **Same everything as Phase 3, only the training rows change.** Model RBF SVM (`rbfsvm`), imbalance `none`, grid
  C {0.1, 1, 10, 100} x gamma {0.8, 0.2, 0.05, 0.0125}, F2-tuned threshold, all 8,641 block-weighted features,
  `StandardScaler` fitted inside each fold on that region's rows only, seed 42.
- **Nested CV per region:** the outer folds are the region's rows of frozen folds 0-4; for outer fold k the 16 configs are
  scored by an inner CV over the region's rows of the other four folds (mean PR-AUC), the threshold is the F2 optimum of the
  winner's inner out-of-fold scores, and the winner is refitted on the four folds and scored on fold k.
- **Comparison on the same images.** The unified counterpart is the Phase 3 nested-CV run with the same model and strategy
  (`results/phase3/oof_rbfsvm__none.csv`): its outer-fold scores for the same region rows, from the same folds, with its own
  fold-wise tuned thresholds. Both are therefore honest out-of-fold numbers. The difference is region-specific minus unified,
  with a **paired bootstrap** over the region's images (2,000 resamples, thresholds held fixed):
  - **PR-AUC and ROC-AUC: mean of the 5 per-fold differences**, with images resampled *within* each fold
    (`phase4_regional.fold_paired_bootstrap`). Not the pooled out-of-fold PR-AUC: each fold has its own fitted model, and in
    small regions the five models often pick different C/gamma, so their margins are on different scales. Pooling them
    penalises whichever approach varies more across folds. (Found on a synthetic dry run before any real Phase 4 result:
    pooled 0.41 vs fold-mean 0.73 for the same scores.) Pooled values are still printed for reference.
  - **Recall, precision, specificity, F2: pooled out-of-fold**, each image at its own fold's tuned threshold
    (`evaluate.paired_bootstrap`). Thresholds already adapt to each fold's scale, so pooling is fine here.
  - On the test set there is one model per region, so all test metrics are computed directly on the region's test images.
- **Metrics** (as Phase 3): PR-AUC primary (chance = the region's fracture rate: hand 0.30, leg 0.10, hip 0.18,
  shoulder 0.18, so PR-AUC is not comparable *across* regions, only between models within a region); ROC-AUC, recall,
  precision, F2, specificity at the tuned threshold. Reported as mean +/- SD over the 5 outer folds and as pooled out-of-fold.
- **Routed system** (all four regions together, each image scored by its own region's model at its region's threshold) vs
  the unified model on the same images: threshold metrics only (recall, precision, F2, specificity). Pooled PR-AUC/ROC-AUC
  are not reported for the routed system because scores from four different models are not on one scale.

### Why the full grid, although the small regions make inner tuning noisy
Hip and shoulder have about 9-10 fractured images per fold, so each inner validation fold holds about 9 positives and the
inner PR-AUC that picks C and gamma is noisy. A smaller grid was considered and rejected: the plan's rule is that differences
must come from the region split, not from unequal tuning effort, and the noise of choosing hyperparameters on little data is a
real cost of training a small specialised model, which the comparison should include. To show how large that cost is, one
**sensitivity run** fixes the hyperparameters at the Phase 3 values (C=10, gamma=0.05) and tunes only the threshold. Caveat:
those values were picked on all 3,266 training images, which include each region's rows, so the fixed run is slightly
optimistic and is a diagnostic only, never the Phase 4 model.

### Family check (plan: is the same family best per region?)
The linear SVM (`linsvm`, imbalance `none`, grid C {0.01..1}) is also run per region; its unified counterpart
`results/phase3/oof_linsvm__none.csv` exists, so it gets the same paired comparison. It is the runner-up family of Phase 3 and
the simpler model, which is the natural alternative for small data. The tree models are not rerun per region (Phase 3: 5-10x
slower and not better; with 46 fractured hip images and 1,000 or 300 ANOVA-selected columns they are not expected to help);
they can be added with `--models rf hgb --imbalance balanced` if there is time, and would be compared with their Phase 3
`balanced` runs.

## Decision rules (fixed before seeing results)
1. **The Phase 4 model is the RBF SVM, imbalance none, full grid, F2 threshold, in all four regions.** The family is not
   re-chosen per region: choosing per region on 9-10 positives per fold would be extra tuning that the unified model did not
   get. The family check and the fixed-hyperparameter run are reported as findings (input for Phase 5), and cannot change the
   model sent to the test set.
2. **Verdict per region** (CV and test separately): if the 95% paired-bootstrap CI of the PR-AUC difference (region minus
   unified; in CV the fold-mean difference) excludes 0, the region-specific model is "better" or "worse" in that region; otherwise "no detectable
   difference". The same is reported for F2 at the tuned thresholds as a secondary result. For hip and shoulder a "no
   detectable difference" is expected from sample size alone and is not evidence that the two models are equal.
3. **Grid edges:** if a chosen value sits on a grid edge in at least 3 of 5 outer folds of a region, it is reported. The grid
   is not extended for one region alone (that would be unequal tuning). If it happens, a labelled sensitivity run with the
   extended grid may be added; it is not sent to the test set.
4. **Test set, once:** `phase4_final_test.py --model rbfsvm --imbalance none` is run only after the main CV run is finished
   and its results are written into this file. Per region, settings and threshold come from `tune_cv` on that region's training
   rows; the model is refitted on all of them and scores that region's test images once. The unified test scores and threshold
   (-0.844) are read from Phase 3, not recomputed. No model or threshold is changed after seeing test numbers.

## Known threats (carry into the report)
- Inner CV tuning on hip and shoulder rests on about 9 positives per validation fold: hyperparameters can vary from fold to
  fold. Reported (chosen params per fold) and measured (fixed-hyperparameter run).
- Test CIs for hip (11 fractured) and shoulder (12) are wide; the paired bootstrap narrows them somewhat because both models
  are scored on the same images, but a real difference of a few points is undetectable.
- No patient ID: views of one patient can sit in train and test. A region model sees only that region, so near-duplicate views
  matter more for it in relative terms.
- The bootstrap holds thresholds fixed, so it ignores the noise of having chosen them (larger for the small regions).
- The routed system assumes the region label is known at prediction time.

---

## Results
The protocol above was not changed after the results were seen. Tables: `results/tables/phase4_cv_summary.csv`,
`phase4_cv_paired.csv`; per-image out-of-fold scores: `results/phase4/oof_*.csv`. Each run took about 2 minutes on 4 cores.

### 1. Nested CV per region (training images): the Phase 4 model (RBF SVM, none, full grid)
PR-AUC is the mean +/- SD over the 5 outer folds; the difference is region-specific minus unified on the same images
(fold-mean, 95% fold-stratified paired bootstrap, 2,000 resamples). Chance PR-AUC = the region's fracture rate.

| Region | Fractured (train) | Chance | Region-specific PR-AUC | Unified PR-AUC (same images) | Difference [95% CI] | Folds region better | Verdict (rule 2) |
|---|---|---|---|---|---|---|---|
| Hand | 305 | 0.298 | 0.619 +/- 0.062 | 0.637 +/- 0.070 | -0.018 [-0.047, +0.017] | 2/5 | no detectable difference |
| Leg | 170 | 0.100 | 0.608 +/- 0.064 | 0.584 +/- 0.039 | +0.023 [-0.012, +0.060] | 4/5 | no detectable difference |
| **Hip** | 46 | 0.180 | **0.754 +/- 0.119** | 0.531 +/- 0.122 | **+0.223 [+0.105, +0.324]** | 5/5 | **region-specific better** |
| Shoulder | 48 | 0.180 | 0.560 +/- 0.165 | 0.636 +/- 0.134 | -0.077 [-0.156, +0.003] | 1/5 | no detectable difference (borderline worse) |

At the tuned thresholds (pooled out-of-fold, differences with 95% paired-bootstrap CIs):

| Region | Recall reg / uni | Specificity reg / uni | F2 reg / uni | F2 difference |
|---|---|---|---|---|
| Hand | 0.885 / 0.816 | 0.369 / 0.514 | 0.694 / 0.684 | +0.010 [-0.018, +0.040] |
| Leg | 0.659 / 0.753 | 0.899 / 0.776 | 0.592 / 0.556 | +0.036 [-0.012, +0.081] |
| Hip | 0.870 / 0.783 | 0.743 / 0.595 | 0.719 / 0.590 | **+0.129 [+0.023, +0.231]** |
| Shoulder | 0.688 / 0.812 | 0.644 / 0.594 | 0.545 / 0.609 | -0.065 [-0.157, +0.026] |
| Routed (4 regions) | recall +0.005 [-0.026, +0.037] | specificity +0.047 [+0.031, +0.065] | precision +0.038 [+0.022, +0.055] | **+0.027 [+0.003, +0.050]** |

- **Hip is the only region with a clear difference**, and it favours the specialised model in all 5 folds. The unified model
  ranks hip worst of the four regions (0.531, compared with 0.58-0.64 elsewhere), so a pooled model dominated by hand and
  leg images seems to fit hip radiographs poorly. Phase 5 has to separate "hip is different" from other explanations
  (learning curves, feature-space view).
- **Hand and leg: no difference in ranking.** The threshold metrics differ because the thresholds differ, not because one
  model ranks better: the hand model's threshold buys recall (+0.07) with specificity (-0.14); the leg model's does the
  opposite (recall -0.09, specificity +0.12). That is the per-region threshold question of Phase 6.
- **Shoulder is borderline worse,** and its chosen hyperparameters were unstable (C from 0.1 to 10, gamma from 0.0125 to 0.8
  across folds). Section 2 shows most of that loss is tuning noise.
- **Routed system** (each image scored by its own region's model): same recall as the unified model, slightly higher
  specificity and precision, F2 +0.027 [+0.003, +0.050]. Small, and driven mostly by hip.
- **Grid edge (rule 3):** hip chose C=100, the top of the grid, in 4/5 folds (the linear SVM also chose its top C=1 in
  5/5 hip folds). Reported, not extended: both families want weaker regularisation on hip. No extended-grid run was made.

### 2. Fixed-hyperparameter sensitivity and family check
| Region | RBF, tuned (Phase 4 model) | RBF, fixed C=10 gamma=0.05 | Difference vs unified, fixed | Linear SVM (region) | Linear SVM, difference vs unified linear SVM |
|---|---|---|---|---|---|
| Hand | 0.619 +/- 0.062 | 0.621 +/- 0.054 | -0.016 [-0.042, +0.013] | 0.608 +/- 0.069 | -0.000 [-0.030, +0.029] |
| Leg | 0.608 +/- 0.064 | 0.614 +/- 0.077 | +0.029 [-0.004, +0.067] | 0.615 +/- 0.053 | **+0.048 [+0.015, +0.081]** |
| Hip | 0.754 +/- 0.119 | 0.733 +/- 0.148 | **+0.201 [+0.084, +0.314]** | 0.695 +/- 0.164 | **+0.287 [+0.152, +0.404]** |
| Shoulder | 0.560 +/- 0.165 | 0.604 +/- 0.123 | -0.033 [-0.122, +0.061] | 0.590 +/- 0.128 | +0.026 [-0.099, +0.124] |
(fold-mean PR-AUC +/- SD; differences with 95% fold-stratified paired-bootstrap CIs)

- **Tuning noise:** fixing the hyperparameters changes hand, leg and hip by at most 0.02, but lifts shoulder from 0.560 to
  0.604, so about 0.04 of shoulder's 0.077 deficit is the cost of choosing C and gamma on 9-10 positives per fold. The hip
  gain survives without any per-region tuning (+0.201), so it is not a tuning artefact. (The fixed values were chosen on all
  training images, so this run is slightly optimistic; diagnostic only, per the protocol.)
- **Family check:** the RBF and linear region models are within one fold-SD of each other in every region (hand 0.619 vs
  0.608, leg 0.608 vs 0.615, hip 0.754 vs 0.695, shoulder 0.560 vs 0.590). No region clearly prefers another family, so
  keeping the RBF SVM everywhere (rule 1) loses nothing measurable.
- The same pattern appears with the linear SVM: hip is much better specialised (+0.287), and here leg is too (+0.048, CI
  excludes 0). The unified linear SVM is weak on hip (0.408), so the hip gap is not specific to the kernel model.

### 3. Test set (used once, 2026-11-10)
Run after the CV results above were written and committed (commit `f300daa`). Per region, settings and threshold come from a
grid search over that region's 5 training folds (`tune_cv`; CV PR-AUC optimistic), then a refit on all its training images:

| Region | Params | F2 threshold (margin) | Tuning-fold PR-AUC |
|---|---|---|---|
| Hand | C=10, gamma=0.2 | -0.971 | 0.634 |
| Leg | C=10, gamma=0.05 | -0.758 | 0.614 |
| Hip | C=100, gamma=0.0125 | -0.624 | 0.751 |
| Shoulder | C=10, gamma=0.2 | -0.609 | 0.618 |

The unified model is the Phase 3 final model (RBF C=10, gamma=0.05, threshold -0.844), evaluated from its saved test scores on
the same 812 images (the 4 regions; "other" excluded). 95% CIs: 2,000 bootstrap resamples, thresholds held fixed.

**Ranking (PR-AUC), region-specific vs unified on the same test images:**

| Region | Test fractured | Region-specific | Unified | Difference [95% paired CI] | Verdict (rule 2) |
|---|---|---|---|---|---|
| Hand | 76 | 0.686 [0.586, 0.782] | 0.641 [0.535, 0.752] | +0.045 [-0.028, +0.112] | no detectable difference |
| Leg | 42 | 0.636 [0.488, 0.768] | 0.623 [0.471, 0.759] | +0.013 [-0.070, +0.085] | no detectable difference |
| Hip | 11 | 0.628 [0.333, 0.865] | 0.724 [0.466, 0.928] | -0.096 [-0.256, +0.047] | no detectable difference |
| Shoulder | 12 | 0.607 [0.334, 0.836] | 0.651 [0.389, 0.866] | -0.044 [-0.171, +0.082] | no detectable difference |

**At the tuned thresholds** (recall / specificity / F2; TN/FP/FN/TP):

| Region | Region-specific | Unified | F2 difference [95% paired CI] |
|---|---|---|---|
| Hand | 0.974 / 0.261 / 0.724 (47/133/2/74) | 0.776 / 0.628 / 0.686 (113/67/17/59) | +0.038 [-0.035, +0.111] |
| Leg | 0.667 / 0.898 / 0.596 (344/39/14/28) | 0.714 / 0.843 / 0.581 (323/60/12/30) | +0.014 [-0.066, +0.090] |
| Hip | 0.727 / 0.755 / 0.615 (40/13/3/8) | 0.727 / 0.811 / 0.645 (43/10/3/8) | -0.030 [-0.239, +0.176] |
| Shoulder | 0.750 / 0.800 / 0.662 (44/11/3/9) | 0.833 / 0.727 / 0.685 (40/15/2/10) | -0.023 [-0.183, +0.101] |
| **Routed (4 regions)** | 0.844 [0.781, 0.902] / 0.708 / 0.677 [0.617, 0.726] (475/196/22/119) | 0.759 [0.688, 0.827] / 0.773 / 0.650 [0.589, 0.708] (519/152/34/107) | +0.027 [-0.021, +0.076] |

Routed-system differences: recall +0.085 [+0.021, +0.147], specificity -0.066 [-0.099, -0.031], precision -0.035
[-0.072, +0.002]. Full tables: `results/tables/phase4_test_{metrics,ci,paired,confusion}.csv`; per-image scores of both
models: `results/phase4/test_scores.csv`.

**What this says**
- **On the test set, region-specific models are not detectably better or worse than the unified model in any region.**
  Point estimates favour specialisation for hand and leg (+0.045, +0.013) and the unified model for hip and shoulder
  (-0.096, -0.044), and every CI includes 0.
- **The CV hip gain does not replicate.** CV gave +0.223 [+0.105, +0.324]; the test gives -0.096 [-0.256, +0.047]. Both are
  reported, not reconciled away. The test hip subset has only 11 fractured images, and it happens to be easy for the unified
  model (test PR-AUC 0.724 vs 0.531 in CV), so the test cannot confirm or rule out a gain of the CV size. Taken together the
  evidence does not support a *reliable* advantage for a hip-specific model. A hypothesis for Phase 5: hip training images
  include multiple views of the same patient (no patient ID), and a weakly regularised small model (the hip model chose C=100)
  can profit from near-duplicates across CV folds more than the unified model. The share of `multiscan` images in hip
  can show whether that is plausible.
  - **[Phase 5 update - this hypothesis was wrong, and the question is now answered: see `docs/phase5_analysis.md`,
    "The hip question".]** Hip has **0% `multiscan` images** (hand 8.6%, leg 13.5%), so near-duplicate leakage cannot be
    the explanation - and Phase 5.3 found that multi-scan images *are* measurably easier (+0.14 detection rate), which
    makes hip's CV number the *least* inflated of the four rather than the most. The CV effect is real: the hip region arm
    beats the unified arm at every training size from 10 images upward and the gap **widens** with data (+0.083 to
    +0.209), a unified model trained on zero hip images scores 0.267 against a 0.18 chance level, and all six single-region
    transfer ratios into and out of hip lie between -0.03 and +0.16 (the pooled "all regions" row reaches only
    +0.26 size-matched, against hip's own +1.00). The mechanism is that hip's fracture direction is
    near-orthogonal to the pooled one (cosine 0.193 against hand's 0.625), its lesions are about twice the linear size of
    any other region's, and it supplies only 7.8% of the training images. What stays unproven is whether the effect
    **generalises**: 11 fractured test images could not measure it either way. This bullet's "does not support a reliable
    advantage" therefore stands as written *for the test set*, and is not evidence against the CV result.
- **Region models generalise like the unified one:** CV-to-test PR-AUC for the region models was hand 0.619 to 0.686, leg
  0.608 to 0.636, hip 0.754 to 0.628, shoulder 0.560 to 0.607, all inside the test CIs.
- **Operating point, not ranking:** the hand model's F2 threshold is very liberal (recall 0.97 but specificity 0.26, 133
  false alarms out of 180 normal hand images), as it already was in CV (specificity 0.37). With 30% fractured hand images, the
  F2 optimum sits low. This is what makes the routed system trade specificity (-0.066) for recall (+0.085), with F2
  unchanged within noise. Per-region thresholds are the Phase 6 question.
- **Answer for the plan (Phase 4 rule):** the four region models used the same features, family, grid, CV protocol and
  imbalance strategy as the unified model, and were scored on identical images. On this data the region split does not
  produce a measurable gain on the test set. The CV result for hip and the learning curves are what Phase 5 has to explain.
- **[Phase 5 update]** Phase 5 explained both. The learning curves and the cross-region transfer matrix show that **hip is
  a genuine region effect and hand, leg and shoulder are not** (`docs/phase5_analysis.md`, 5.2), and 5.1 found that what
  specialisation mainly changes is not the ranking but the **decision threshold** - McNemar on the test decisions is
  significant for hand (p=7e-08, favouring the unified model) and for leg (p=0.004, favouring the region model) while
  every PR-AUC difference stays inside its CI. That makes the "Operating point, not ranking" bullet above the main Phase 4
  finding in hindsight, and per-region thresholds the cheaper of the two Phase 6 levers.

### Reproduce
```
python src/phase4_regional.py --models rbfsvm --imbalance none                             # main run, 4 regions
python src/phase4_regional.py --models rbfsvm --imbalance none --fixed-params C=10 gamma=0.05   # sensitivity
python src/phase4_regional.py --models linsvm --imbalance none                             # family check
python src/phase4_regional.py --summarize                  # -> results/tables/phase4_cv_*.csv
python src/phase4_final_test.py --model rbfsvm --imbalance none   # once; lock: results/phase4/TEST_EVALUATED.json
```
