# Phase 4: region-specific classifiers (IN PROGRESS: protocol 2026-11-09, CV done 2026-11-10, test pending)

Task: one image-level fracture classifier per region (hand, leg, hip, shoulder), each trained only on that region's
training images, with **the same features, model family, grid, CV protocol and imbalance strategy as the Phase 3 unified
model**, and compared with the unified model **on the same images**. **This file records the protocol and the decision
rules before any Phase 4 result was seen.** Results are added below the line once the runs are done; the rules are not
changed after that. Status: `docs/PROJECT_STATUS.md`.

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

### 3. Test set (used once)
*(to be filled after `phase4_final_test.py`)*

### Reproduce
```
python src/phase4_regional.py --models rbfsvm --imbalance none                             # main run, 4 regions
python src/phase4_regional.py --models rbfsvm --imbalance none --fixed-params C=10 gamma=0.05   # sensitivity
python src/phase4_regional.py --models linsvm --imbalance none                             # family check
python src/phase4_regional.py --summarize                  # -> results/tables/phase4_cv_*.csv
python src/phase4_final_test.py --model rbfsvm --imbalance none   # once; lock: results/phase4/TEST_EVALUATED.json
```
