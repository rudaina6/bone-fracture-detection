# Phase 7b: localization improvement (protocol written 2026-11-22, before any 7b model was trained)

Task: find the fractures, as in Phase 7, with a two-stage cascade instead of a single linear sliding-window detector.
Phase 7 (`docs/phase7_localization.md`) is finished and stays as it is: its detector reached AP@50 0.0001 in CV and
0.0003 on the test. Phase 6 is final and is not touched. Status: `docs/PROJECT_STATUS.md`.

**This file records the design, every candidate variant, the CV protocol and the decision rules before any Phase 7b
model is trained.** Anything added after results are in is marked `[added after results]` with the reason.

**Data rules.** Training images (3,266) and their frozen folds (`results/splits.csv`) only. Every window inherits its
image's fold, and no step that builds, tunes, calibrates or thresholds a model for outer fold k sees fold k's images.
Phase 7's map cache (`features/phase7/maps/`), windows and outputs are read, never rewritten.

**The test set was already used once, by Phase 7.** Phase 7b uses it **one more time**, at the very end, with its own
lock (`results/phase7b/TEST_EVALUATED.json`), after the CV results are written here and approved. That result is
reported as a **post-hoc follow-up**: 7b was designed after Phase 7's test result was known. No 7b choice uses test data
(the design comes from the training-data diagnosis below), but the test is no longer untouched, and the report says so.
The Phase 3, 4, 6 and 7 locks stay as they are.

## Starting point: the Phase 7 diagnosis (training images only, read-only, run 2026-11-22 after the Phase 7 test)

The diagnosis was run before this protocol and did not change any Phase 7 output. Scripts, logs, tables and figures:
`results/phase7b/diagnosis/` (README there). Before analysing, it reproduced the saved Phase 7 CV detections of fold 0
exactly (difference 0 for rounds 0, 1 and 2), so it describes the same models; the 7b gates check round 0 of every fold.

| Question | Finding | Consequence for 7b |
|---|---|---|
| Is the patch classifier weak? | No. Round-0 linear SVM, held-out patches per outer fold: PR-AUC 0.856-0.875 at 1:3, ROC-AUC 0.955-0.963 against uniform negatives (training 0.965, so no overfitting). Its tail is weak: at FPR 1e-3 / 1e-4 it keeps 9.6% / 2.1% of fracture windows, and 13.7% of uniform windows (about 3,300 per image) score above 0. | The linear model is a reasonable **filter**, not a ranker: keep it as stage 1. |
| Where do fracture windows rank? | The best positive window of a box ranks 125th of about 24,000 in its image (median; quartiles 28th and 413th). Hand 80, leg 128, hip 302, shoulder 436. Boxes under 20 px: 248; 40 px or more: 57. Within its own window type the percentile is 0.7-1.8% for seven of the nine types (3.0% and 4.4% for the two largest, which only 28 and 7 boxes need): 16 px windows lose by number (about 10,000 per image), not by a worse per-window ranking. | Share of boxes whose best positive window is in the image's top K: K = 100: 44.0%, 500: 73.1%, **1,000: 81.4%**, 2,000: 88.4% (reachability 93.9%). Per-type calibration is worth testing. |
| What are round-0 negatives? | Mostly air and soft tissue: 61% contain no edge pixel; median CLAHE intensity 0.19 against 0.52 for fracture windows; edge density alone separates fracture windows from them with AUC 0.91. Visual count of 40 random used negatives (one rater): 27 air, 10 soft tissue, 2 bone [95% CI 0.6-17%], 1 border. | The patch task the model learned is "bone and edges against background". |
| What are the false boxes? | Bone. Each image's top round-0 box, when false, is as edge-rich as a fracture window (gradient AUC 0.49, edge density 0.58) and brighter (intensity AUC 0.31); 0.2% of these touch the padding, 1.4% contain a saturated pixel (fracture windows 1.1%), 2.4% are on hardware images (base rate). Visual count of 40 random false boxes: 36 bone [76-97%], 17 of them joints [27-59%]; 3 soft tissue, 1 border, 0 air. The 24 highest-scoring ones on normal images: 10 joints or growth plates, 6 bone shafts, 3 vertebrae, 3 overexposed strips, 2 soft tissue. | Stage 2 must separate fractures from bone and joints; negatives should come from where the confusers are (stage-1 proposals). Context around a window may tell a line crossing the whole bone (growth plate, joint) from a fracture. |
| Why did mining make AP worse? | Fold 0: round 1 added 52,240 hard negatives (19 per positive; the round-0 hot spots, with the same intensity and edge statistics as the false boxes). The linear model flipped (weight cosine -0.64 with round 0), held-out patch ROC-AUC fell from 0.963 to 0.679, 42% of its own round-0 negatives scored above 0, and the median fracture-window rank went from 130 to 4,346 (0 hits). Round 2 swung back (0.933, rank 199) but kept a worse tail (TPR at FPR 1e-3: 3.3% against 13.2%). 2,719 (round 1) and 9,182 (round 2) eligible windows per image scored above 0; the cap took 20. | A linear model cannot separate fractures from the bone it is confused with. Stage 2 is **non-linear**, trained once on a **fixed ratio** of proposal negatives, with no mining loop. |
| Would image-level gating fix it? | No. Even an oracle that removes every box on normal images lifts round-0 CV AP@50 only from 0.0011 to 0.0042. | Not a 7b variant. The problem is ranking inside fractured images. |

Two automatic bone masks (global 3-class Otsu; two-stage Otsu with local contrast) were tried and rejected after visual
checks (they mislabel thin limbs and bright strips); the bone / soft-tissue counts above are visual, from one rater.

## Design

```
image (448 px, Phase 7 preprocessing)
  -> 24,213 windows (Phase 7 window set {16, 40, 64} px x {1:1, 1:2, 2:1}; Phase 7 map cache)
  -> stage 1: Phase 7 round-0 linear detector (no mining)       -> top 1,000 windows per image = proposals
  -> features: window (366) + context (188) + window type (9) + stage-1 score (1)
  -> stage 2: HistGradientBoosting (random forest dropped at STOP 2) -> score per proposal
  -> optional per-window-type calibration
  -> NMS 0.3 across window types, at most 10 boxes per image   -> box-score threshold from CV
```

### Stage 1: proposals
- **Model:** the Phase 7 round-0 detector: linear SVM, `class_weight="balanced"`, round-0 patches (every positive
  window; 3 negatives per positive, uniform over eligible windows, seed 0), C by `cv_protocol.tune_cv` over its
  training folds (patch PR-AUC, grid {0.001, 0.01, 0.1, 1}), model seed 42. **No mining.** The outer-fold models are
  therefore Phase 7's round-0 models, and a gate checks that they reproduce Phase 7's saved round-0 detections exactly.
- **Proposals:** the 1,000 highest-scoring windows of each image, all nine types pooled, no NMS. **K = 1,000 is fixed**,
  not tuned (diagnosis: 81.4% of boxes have their best positive window there).
- **Proposal recall** (the stage-2 ceiling): the share of boxes with a proposal at IoU >= 0.5, and at IoU >= 0.3,
  overall and per region; also reported descriptively at K = 250, 500 and 2,000.
- **Cross-fitting:** every proposal used to *train or tune* stage 2 comes from a stage-1 model that did not see its
  image:

| Proposals on images of fold j, used for | Stage-1 model | Trained on |
|---|---|---|
| Scoring held-out outer fold k = j (CV evaluation) | M(not k) | the four folds other than k (= Phase 7 round 0) |
| Stage-2 training / inner tuning inside outer fold k (j != k) | M(not j, not k) | the three folds other than j and k (10 such models) |
| Final detector's stage-2 training (all training images) | M(not j) | the four folds other than j (the outer models) |
| Test images | M(all) | all five folds (= Phase 7's T1 round-0 model, C 0.1) |

### Stage 2: features (per proposal)
- **Window (366):** the Phase 7 descriptor: HOG 3 x 3 blocks of the window's 4 x 4 cells (324), uniform LBP histograms
  of its 2 x 2 quarters (40), intensity mean and standard deviation (2), from the cached maps.
- **Context (188):** the window plus a margin of half its size on every side (a 2w x 2h region, 8 x 8 cells of the
  window type's own level), described from the same cached maps, so no new extraction is needed. For a window starting
  at cell column i, row j of its level (it covers cells i..i+3, j..j+3; HOG block (r, c) covers cells r..r+1, c..c+1):
  - HOG: the mean block vector of each side of the ring around the window, top (block row j-2, block columns i-2..i+4),
    bottom (row j+4, columns i-2..i+4), left (rows j..j+2, column i-2), right (rows j..j+2, column i+4): 4 x 36 = 144;
  - LBP: the uniform-code histogram of each ring side's cells, normalised to sum 1: 4 x 10 = 40;
  - intensity: ring mean, ring standard deviation, window mean minus ring mean: 3;
  - ring coverage: the share of ring cells inside the level grid (windows at the image edge): 1.
  Ring cells outside the level grid are left out of the means (zeros if a side has none). A horizontal line that
  crosses the window and both side strips (growth plate, joint space) then looks different from one that does not.
- **Window type** one-hot (9) and the **stage-1 score** (1).
- Window-only variant: 376 dims; window + context: 564 dims.

### Stage 2: training set (inside each training set, never the held-out fold)
- **Positives:** every window with IoU >= 0.5 with a ground-truth box on the training images (all of them, whether or
  not they are proposals).
- **Negatives:** proposals with IoU < 0.1 with every box, from normal and fractured training images, **10 per positive**
  (fixed), drawn uniformly over the pooled (image, proposal) pairs with a generator seeded 0. Proposals with
  0.1 <= IoU < 0.5 are not used for training.
- **Class weights:** balanced (sample weights), as in every phase.

### Stage 2: classifiers and their inner tuning

*[Revised at the STOP 2 review, 2026-11-23, for runtime only, before any CV run and before any result: **random forest
is dropped** and the **HGB grid is `max_leaf_nodes` {15, 63} with `min_samples_leaf` 20** (the two fallback HGB
configurations). Why these two: tree size is the capacity setting that matters most for this task (stage 2 has to
combine window and context features to tell a fracture from the bone and joints it is confused with, which needs
interactions), and 15 and 63 leaves span a shallow and a rich model; `min_samples_leaf` 20 is scikit-learn's default
and keeps leaves small enough to isolate small groups of fracture windows (about 2,100 positives per inner training
set, against 21,000 negatives), where 100 would tend to merge them with the confusers. Learning rate 0.1 and 300
iterations are unchanged. Nothing was chosen from a CV result: none exists. The CV time is dominated by stage-2 fits,
and RF's configurations were about 40% of the measured fit time (`results/phase7b/timing_v1.json`). Everything else
in this protocol is unchanged. The original text follows.]*

- **HistGradientBoostingClassifier:** learning rate 0.1, 300 iterations, no early stopping (its internal split would mix
  windows of one image), l2 0; grid `max_leaf_nodes` {15, 63} x `min_samples_leaf` {20, 100} *(revised: {15, 63} with
  20)*.
- ~~**RandomForestClassifier:** 300 trees; grid `max_features` {"sqrt", 0.2} x `min_samples_leaf` {1, 10}.~~ *(dropped)*
- Seed 42; scikit-learn classes as installed in the repo's `.venv`.
- **Inner tuning** (per outer fold, per classifier, per feature set): 4-fold inner CV over the outer training folds.
  Inner validation fold j: train on the other three folds' training set, score fold j's proposals. Metric:
  **weighted window-level average precision among proposals**: all positive proposals (IoU >= 0.5) of fold j's images
  and a uniform sample of 100 non-positive proposals per image (IoU < 0.5), weighted by (number of non-positive
  proposals of that image) / 100. The highest fold-mean inner score picks the configuration, which is refitted on the
  whole outer training set. *Why this proxy and not inner localization AP: AP@50 at these levels is driven by a handful
  of hits and would tune on noise; the proxy measures what stage 2 is for (ranking positive windows above the other
  proposals) and uses every positive.*
- **Fallback grids** (only if the Stage 2 timing projects more than 6 h for the stage-2 fits of the CV, decided before
  the CV): HGB `max_leaf_nodes` {15, 63} with `min_samples_leaf` 20; RF `max_features` {"sqrt", 0.2} with
  `min_samples_leaf` 1. *(Superseded by the STOP 2 revision above.)*

### Per-window-type calibration
For each window type t, a logistic regression of "proposal is a positive (IoU >= 0.5)" on the logit of the stage-2
score (Platt scaling), fitted on the inner out-of-fold stage-2 scores of the training proposals (the inner validation
samples above, with their weights, so each type's base rate among proposals is kept). The calibrated probability
replaces the score before NMS. Purpose: a 16 px window and a 64 px window with the same raw score need not have the same
chance of being a fracture window, and 16 px windows outnumber the others.

### Inference
Stage-2 score (raw or calibrated) of the image's 1,000 proposals; greedy NMS at IoU 0.3 across all window types; at
most 10 boxes per image. The image-level score is the top box's score.

## Candidate variants (complete list)

| Id | Stage 2 | Features | Calibration | Role |
|---|---|---|---|---|
| P7 | none (Phase 7 detector: linear, 2 mining rounds) | window | no | reference (saved Phase 7 CV detections) |
| S1 | none (stage 1 alone = Phase 7 round 0, NMS, top 10) | window | no | baseline (saved Phase 7 round-0 detections) |
| H-wc | HGB | window + context | no | B2 |
| H-w | HGB | window only | no | B2 |
| H-wc-cal, H-w-cal | HGB | as named | per type | B3 |

*[STOP 2 revision: the RF variants R-wc, R-w, R-wc-cal and R-w-cal are dropped; the 4 HGB variants (context x
calibration) stay.]*

Every variant is evaluated and reported; rules B1-B3 pick one. Fixed for all variants, not chosen from results: the
window set, K = 1,000, the 10:1 negative ratio, the training IoU thresholds (0.5 / 0.1), the grids above, NMS 0.3,
10 boxes per image.

## CV protocol (training images, nested, Phase 7's frozen folds)

For each outer fold k = 0..4:
1. **Stage 1:** M(not k) (tune_cv over the four training folds; reproduces Phase 7 round 0) scores every window of
   fold k's images -> held-out proposals. For each training fold j != k, M(not j, not k) (tune_cv over its three
   folds) scores fold j's images -> cross-fitted training proposals.
2. **Stage-2 training sets** from the four training folds (positives, 10:1 proposal negatives).
3. **Inner tuning** of each feature set (HGB; 4-fold, metric above); refit at the chosen configuration.
4. **Calibration** fitted on the inner out-of-fold scores of the chosen configuration (for the -cal variants).
5. **Held-out scoring:** fold k's proposals scored by each of the 2 refitted models (window + context, window only),
   raw and calibrated; NMS 0.3; top 10 -> out-of-fold detections of 4 variants. *(STOP 2 revision; previously 4 models
   and 8 variants.)*

Fold k's images are never used in steps 1-4 of fold k, so every out-of-fold number is honest. *Accepted
approximation:* inside outer fold k, the training proposals of fold j' come from M(not j', not k), which saw the inner
validation fold j's patches; this touches only which negatives are drawn for inner tuning (choosing among 4
configurations), not the outer evaluation. Removing it would need stage-1 models trained on two folds.

**Final detector's stage-2 hyperparameters** (needed by B4): the same inner tuning on all 3,266 training images (5-fold,
proposals from the outer models M(not j)), for the configuration B1-B3 select, computed at the end of the CV.

## Metrics

As Phase 7, on the same folds and images:
- **AP@50** (primary), **AP@50:95**, **recall, precision and F1 at IoU 0.5** at the box-score threshold (B4), **recall at
  IoU 0.3**, overall and per region (hand, leg, hip, shoulder; "other" in the overall numbers only).
- **Fold means**, each fold on its own images (normal images included, so they contribute false positives); pooled
  numbers beside them for reference. **95% CIs:** fold-stratified bootstrap over images (2,000 resamples, seed 0), as
  `phase7_detect.boot_folds`.
- **Ceilings next to recall:** reachability (Phase 7 design table, 93.9%) and proposal recall (IoU 0.5 and 0.3).
- **Comparisons on the same folds:** paired fold-mean differences (same image multiplicities in every resample) of the
  selected localizer against **P7** (the pre-registered Phase 7 detector) and against **S1** (stage 1 alone, Phase 7's
  best round). S1 is the meaningful baseline: beating P7 alone would only undo the mining collapse.
- **Descriptive:** the full variant table; image-level PR-AUC / ROC-AUC of the top box score against Phase 3
  (`fold_paired_bootstrap` on `results/phase3/oof_rbfsvm__none.csv`); AP@50 on images without hardware (fracture windows
  sit on hardware images 12.4% of the time against 2.4% for false boxes, so an implant shortcut is possible).
- Recall, precision and F1 at a CV-chosen threshold are optimistic in CV (stated with them); the test gives the honest
  number.

## Decision rules (fixed before any 7b model)

Each comparison is the paired fold-mean AP@50 difference with its 95% fold-stratified bootstrap CI.

B1. **Classifier: HGB.** *[Revised at the STOP 2 review, 2026-11-23, before any CV run: random forest is dropped for
    runtime, so there is no comparison. The original rule was: H-wc against R-wc, the higher fold-mean AP@50 wins; if
    the CI of the difference includes 0, HGB.]*
B2. **Context.** With B1's classifier, uncalibrated: window + context is kept only if its AP@50 is higher than window
    only with a CI excluding 0; otherwise window only (fewer features).
B3. **Calibration.** On the B1-B2 configuration: per-type calibration is kept only if it raises AP@50 with a CI
    excluding 0; otherwise raw scores.
B4. **Box-score threshold** (CV only): the F1-optimal score at IoU 0.5 on the pooled out-of-fold detections of the
    selected variant, if every outer fold chose the final detector's stage-2 hyperparameters (and stage-1 C); otherwise
    a **threshold CV** re-runs steps 3-5 of every outer fold with the hyperparameters fixed at the final detector's, and
    the threshold comes from its pooled out-of-fold detections.
B5. **Nothing else is chosen from CV results**, and nothing at all from test numbers.
B6. **The test runs only if the CV-selected localizer beats BOTH S1 (stage 1 alone) AND P7 (the Phase 7 detector)** on
    fold-mean AP@50, each paired difference with a 95% CI excluding 0. Then it runs once, after the CV results and the
    B1-B4 outcomes are written here and approved. Otherwise Phase 7b is reported on CV only and **the test set is not
    used again**. *[Changed at the STOP 1 review, 2026-11-22, before any 7b model was trained; the first draft ran the
    test whatever CV showed.]* The final test script checks this verdict in `results/phase7b/cv_summary.json` and
    refuses to run without it.

## Gates and self-tests (training data or synthetic only, before the CV)

- **Context features = naive:** for 500 random windows on training images, the context descriptor from the cached maps
  equals an independent computation from the level image (crop with margin, skimage HOG / LBP, direct sums), max
  difference <= 1e-5.
- **Stage 1 = Phase 7 round 0:** the five outer stage-1 models reproduce `results/phase7/cv_fold{k}_detections.csv`
  round 0 exactly (scores and boxes).
- **Proposal recall two ways:** from window ranks and from a direct IoU check on the proposal list, equal.
- **No leakage:** asserted for every outer fold: no fold-k image in any stage-1 training set, stage-2 training or
  tuning set, or calibration set; every training proposal comes from a model that did not see its image.
- **Unit tests:** context ring indices on hand-made grids (edge windows, coverage); 10:1 sampler; weighted AP against
  a duplicated-row computation; calibration monotone within a type.
- **End-to-end debug CV** on about 90 training images (own output folders) and a **mock final test** (lock written
  before test rows are read, re-run refused, T1 mismatch stops before the lock), as Phase 7's self-tests.
- **Timing:** proposals + features + scoring for 10 training images, and one fit of each classifier on a full-size
  training set; projected CV runtime, and the fallback-grid decision.

## Runtime estimate (to be replaced by the Stage 2 timing)

| Step | Work | Estimate |
|---|---|---|
| Stage-1 models | 5 outer + 10 pair models (tune_cv: 16 LinearSVC fits on 6-8k patches each) | 15 min |
| Proposals | 16,330 scorings from the cache (5 per training image), top 1,000 kept | 10 min |
| Features | 1,000 proposals x 3,266 held-out images, plus training and inner validation samples | 20 min |
| Stage-2 fits | 5 outer folds x 4 models x (4 configs x 4 inner folds + 1 refit) = 340 fits on 23-31k rows; HGB about 20 s, RF about 60 s | 3-4 h |
| Held-out scoring | 4 models x 3,266 images x 1,000 proposals | 10 min |
| Summary | 10 variants, 2,000 bootstrap resamples | 20 min |
| Final hyperparameters, threshold CV if needed | 21 + 5 fits | 30 min |
| Final test | stage-2 fit on all training images, lock, maps of 817 test images, scoring, T6 | 30 min |
| **Total** | | **about 5-6 h** (fallback grids: about 3 h) |

Storage: proposal indices (16,330 x 1,000, about 65 MB) in `features/phase7b/` (not committed).

*[Stage 2 timing, 2026-11-22, before the CV (`results/phase7b/timing.json`, `stage2_timing.log`): one fit of every
configuration on a full-size training set (30,352 rows) took 151-1,245 s for HGB and 82-579 s for RF, much more and
much more variable than the 20 s / 60 s assumed above (the same HGB configuration took 56 s in a fresh process, so
part of it is contention on this machine). Projected stage-2 fits of the CV: 24.2 h with the full grids, 12.0 h with
the fallback grids; other CV steps 0.6 h. The rule (more than 6 h) therefore selects the **fallback grids**, which
`--grid auto` reads from `timing.json`. The projection with the fallback grids is still above 6 h; how to proceed
was put to the STOP 2 review before any CV run.]*

*[Engineering check, 2026-11-22/23, before any CV run; no protocol change (`results/phase7b/engineering/`): (1) thread
settings: fitted models are identical in every setting (max prediction difference 0); one HGB fit at a time is faster
with 4 OpenMP threads than with 8 (4 physical cores), HGB inner fits run fastest 4 (or `--n-jobs`) at a time with
2 threads each, RF fastest one at a time on all cores. (2) The very slow and erratic timings (an identical block of 4
HGB fits took 177 s in one run and 2,541 s in another) were mostly the laptop in **Modern Standby**, where Windows
throttles desktop programs: 34%, 9%, 86% and 37% of the four timing runs were spent in standby (Kernel-Power events
506 / 507), with power-source changes during them; the 86% run ended in an unclean shutdown. Awake, one HGB fit on the
full-size set takes 110 s (CPU at 93-158% of base clock, no throttling). Long runs now ask Windows to keep the system
and display awake. (3) The random-forest drop and the two-configuration HGB grid above (STOP 2 revision) then cut the
remaining stage-2 cost. Re-timing on the revised grid (`results/phase7b/timing.json`, `stage2_timing.log`;
`--n-jobs 3`; no standby or power event during the run): 4 inner fits 102-260 s and one refit 41-99 s per
configuration; CV-path predictions equal a single-threaded fit (max difference 0). **Projected CV: 1.8 h** (stage-2
fits 1.1 h, other steps 0.6 h), plus 0.3 h for the final hyperparameters.]*

## Final test (only if B6 passes; run once, after approval; post-hoc follow-up)

T0. `results/phase7b/cv_summary.json` must record that B6 passed (the selected localizer beats S1 and P7 on fold-mean
    AP@50, each with a 95% CI excluding 0); otherwise the script stops before anything is fitted or read.

T1. Stage 1: `tune_cv` on all training round-0 patches; must equal `results/phase7/final_c.json` (C 0.1), else stop
    before the lock.
T2. Stage 2: positives and 10:1 negatives from the training images' cross-fitted proposals (outer models M(not j)), the
    selected classifier and features, the final hyperparameters (inner 5-fold tuning, computed in Stage 3), refitted on
    all training images; calibration (if B3 kept it) on the inner out-of-fold scores.
T3. The box-score threshold from B4.
T4. `results/phase7b/TEST_EVALUATED.json` (variant, hyperparameters, threshold, K, NMS, seeds) is written **before any
    test image is read**; a second run is refused.
T5. The 817 test images are scored once: maps computed after the lock, stage 1 M(all), 1,000 proposals, features,
    stage 2, calibration if kept, NMS 0.3, at most 10 boxes.
T6. AP@50, AP@50:95, recall / precision / F1 at the threshold, recall at IoU 0.3, reachability and proposal recall,
    overall and per region, with 95% image-bootstrap CIs (2,000 resamples, seed 0); paired AP@50 difference against
    the Phase 7 test detections (`results/phase7/test_detections.csv`); image-level PR-AUC / ROC-AUC against the Phase 3
    test scores (paired bootstrap); hip and shoulder CI half-widths ("not resolvable" where a CI is degenerate at 0
    hits).
T7. Reported as a post-hoc follow-up on a test set used for the second time. No setting is changed after the test
    numbers are seen.

## Code and outputs

| File | Role |
|---|---|
| `src/phase7b_cascade.py` | Stage-1 models and cross-fitted proposals, context features from the cached maps, stage-2 training sets, inner tuning, calibration, the CV, threshold, tables and figures. Reuses `phase7_windows`, `phase7_detect` (EvalSet, AP, NMS, bootstrap). |
| `src/phase7b_final_test.py` | T1-T7 with its own lock. |
| `tests/test_phase7b.py` | Unit tests, end-to-end debug CV, mock final test. Real-data gates: `phase7b_cascade.py --gates`. |
| `results/phase7b/diagnosis/` | The Phase 7 diagnosis this design is based on (scripts as run, logs, tables, figures). |

Outputs: `results/phase7b/` (JSON, CSV, and a log of stdout and stderr for every long step), `results/tables/phase7b_*.csv`,
`results/figures/phase7b_*.png`; every fitting run gets a row in `results/experiment_log.csv`.

## Steps and stops

1. **Stage 1 (this file).** STOP 1: protocol review.
2. **Stage 2:** code, self-tests, real-data gates, timing; one commit each. Report; continue unless a gate fails or the
   fallback grids are triggered.
3. **Stage 3:** stage-1 models and proposals (commit); outer folds 0-4 (one commit each); final hyperparameters
   (commit); threshold CV if B4 needs it; summary, B1-B4 outcomes (commit); results written here (commit). **STOP 3.**
4. **Stage 4 (only if B6 passes):** the final test after approval (commit); test results written here (commit).
   If B6 fails, Phase 7b ends at STOP 3 with CV results only.

## Known threats (carry into the report)

- **Stage-1 ceiling:** about 81% of boxes have a proposal at IoU >= 0.5 (diagnosis); stage 2 cannot recover the rest.
- **Small effects, noisy decisions:** AP@50 is at the 0.001 level; B2 and B3 default to the simpler option.
- **Inner tuning uses a proxy** (weighted window-level AP among proposals), not localization AP.
- **Stage-1 score feature shift:** training proposals come from three-fold models, held-out ones from four-fold models.
- **Implant shortcut:** fracture windows sit on hardware images far more often than false boxes; reported descriptively.
- **Growth plates and joints** are fracture mimics (diagnosis); context features target them, with no guarantee.
- **Unannotated fractures** become negatives, as in Phase 7.
- **Test reuse:** second look at the test set; post-hoc follow-up, not a fresh estimate.
- **Small regions:** hip 49 / 11 and shoulder 50 / 14 training / test boxes.
- **No patient ID**, as in every phase.

---

## Results

CV on the 3,266 training images (576 fractured, 748 boxes), Phase 7's frozen folds, run 2026-11-23 (revised grid,
`--n-jobs 3`, keep-awake on; 2 h 29 min end to end). No test image or test box was read; `TEST_EVALUATED.json` does
not exist. Outputs: `results/phase7b/` (`stage1.json`, `cv_fold{0-4}.json` + detections, `selection.json`,
`final_params.json`, `cv_summary.json`, a `stage3_*.log` per step), `results/tables/phase7b_cv_*.csv`,
`results/figures/phase7b_cv_*.png`.

**Headline: the cascade raises CV AP@50 about 30-fold over stage 1 alone (0.032 against 0.001) and passes B6, but it
is still a weak localizer.** At the CV threshold it finds 11% of the boxes at IoU 0.5 (22% at IoU 0.3) with precision
9%, and its top box score ranks images worse than Phase 3 does.

### Self-test and gates

- End-to-end self-test rerun on the revised code: all 7 checks pass (`stage2_selftests_revised.log`).
- Real-data gates before the CV (`gates.json`): all pass. Leakage gates (1) no test data before the 7b lock
  (L1: all 3,266 maps and every image read are training; no test identifier in the cascade code); (2) one fold per
  image (L2; also asserted for every stage-2 set at runtime by `check_sets`); (3) cross-fitted proposals (L3; re-checked
  after the CV from the fold provenance: every training proposal of every outer fold comes from a pair model
  M(not j, not k) that saw neither the image's fold j nor the held-out fold k); (4) fits on training folds only (L4:
  stage-1 C, stage-2 hyperparameters, calibration and the threshold all come from training-fold or out-of-fold data).

### Stage 1

C = 0.1 in all five outer models (as Phase 7's `final_c.json`). Held-out proposal recall (top 1,000): **81.4%** of the
boxes at IoU 0.5, **94.8%** at IoU 0.3 (reachability 93.9%), the ceiling for stage 2.

### Variants and selection (B1-B3)

Fold-mean AP@50 (95% fold-stratified bootstrap CI, 2,000 resamples, seed 0), out-of-fold detections:

| Variant | AP@50 | 95% CI | AP@50:95 |
|---|---|---|---|
| **H-wc** (window + context, raw) | **0.0321** | 0.0250-0.0478 | 0.0065 |
| H-w (window only, raw) | 0.0101 | 0.0075-0.0206 | 0.0019 |
| H-wc-cal (window + context, per-type calibration) | 0.0343 | 0.0267-0.0525 | 0.0067 |
| H-w-cal | 0.0113 | 0.0080-0.0226 | 0.0025 |
| S1 (stage 1 alone, Phase 7 round 0) | 0.0011 | 0.0008-0.0023 | 0.0002 |
| P7 (Phase 7 detector, round 2) | 0.0001 | 0.0001-0.0005 | 0.0000 |

- **B1:** HGB (fixed at the STOP 2 revision; no comparison).
- **B2:** H-wc - H-w = **+0.0221 [+0.0136, +0.0337]**, CI excludes 0, so **context is kept**.
- **B3:** H-wc-cal - H-wc = +0.0022 [-0.0052, +0.0115], CI includes 0, so **no calibration** (raw scores).
- **Selected: H-wc.** Every outer fold's inner tuning chose max_leaf_nodes 63, min_samples_leaf 20, and the final
  detector's inner tuning on all training images chose the same (`final_params.json`; inner weighted AP 0.022 against
  0.041 for 15 / 63 leaves).
- **B4:** every outer fold chose the final stage-2 configuration and stage-1 C (0.1), so no threshold CV; the threshold
  is the F1-optimal score at IoU 0.5 on the pooled out-of-fold H-wc detections: **0.9694** (pooled F1 0.100).

![variants](../results/figures/phase7b_cv_variants.png)

### Selected localizer (H-wc), per region

Fold means with 95% fold-stratified bootstrap CIs (2,000, seed 0). Recall, precision and F1 are at the B4 threshold,
chosen on the same out-of-fold detections, so they are optimistic.

| Group | Images | AP@50 | AP@50:95 | Recall IoU 0.5 | Recall IoU 0.3 | Precision | F1 |
|---|---|---|---|---|---|---|---|
| Overall | 3,266 | 0.032 [0.025, 0.048] | 0.0065 [0.0050, 0.0107] | 0.114 [0.091, 0.137] | 0.217 [0.189, 0.246] | 0.090 [0.071, 0.109] | 0.100 [0.081, 0.120] |
| Hand | 1,025 | 0.034 [0.025, 0.054] | 0.0070 [0.0047, 0.0129] | 0.131 [0.099, 0.166] | 0.298 [0.254, 0.345] | 0.078 [0.058, 0.100] | 0.098 [0.074, 0.123] |
| Leg | 1,699 | 0.057 [0.037, 0.099] | 0.0146 [0.0093, 0.0265] | 0.120 [0.081, 0.163] | 0.164 [0.118, 0.215] | 0.121 [0.082, 0.165] | 0.121 [0.083, 0.160] |
| Hip | 256 | 0.090 [0.028, 0.203] | 0.0181 [0.0049, 0.0408] | 0.061 [0.000, 0.139] | 0.082 [0.017, 0.170] | 0.143 [0.000, 0.800] | 0.086 [0.000, 0.200] |
| Shoulder | 267 | 0.002 [0.000, 0.009] | 0.0004 [0.000, 0.0025] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

Ceilings: proposal recall at IoU 0.5 is 81.4% overall (hand 86.5%, leg 79.2%, hip 73.2%, shoulder 66.2%); reachability
93.9%. Hip (49 boxes) and shoulder (50 boxes) are not resolvable: the hip CIs span an order of magnitude, and shoulder
has no hit at the threshold.

### Comparison with stage 1 alone and the Phase 7 detector

Paired fold-mean AP@50 differences (same resamples):

| Group | H-wc - S1 | H-wc - P7 |
|---|---|---|
| Overall | **+0.0311 [+0.0236, +0.0465]** | **+0.0320 [+0.0248, +0.0476]** |
| Hand | +0.0302 [+0.0196, +0.0494] | +0.0330 [+0.0235, +0.0531] |
| Leg | +0.0553 [+0.0352, +0.0962] | +0.0567 [+0.0369, +0.0982] |
| Hip | +0.0897 [+0.0276, +0.2022] | +0.0896 [+0.0271, +0.2023] |
| Shoulder | +0.0020 [+0.0002, +0.0091] | +0.0021 [+0.0002, +0.0092] |

Image level (descriptive; top box score as image score, against Phase 3's RBF-SVM out-of-fold scores): PR-AUC 0.370
against 0.602 (difference -0.232 [-0.271, -0.181]), ROC-AUC 0.714 against 0.822 (-0.108 [-0.137, -0.080]). The
localizer is not a better image classifier; Phase 3 stays the image-level model.

### Implant check

AP@50 on the training images without hardware: H-wc **0.0325**, S1 0.0011, P7 0.0002, the same as on all images
(0.0321). Dropping the hardware images does not lower the cascade's AP, so the gain is not explained by an implant
shortcut.

### B6 outcome

H-wc beats S1 (+0.0311 [+0.0236, +0.0465]) **and** P7 (+0.0320 [+0.0248, +0.0476]) on fold-mean AP@50, both CIs
excluding 0. **B6 passes**: `cv_summary.json` records "final test allowed (after approval)". The final test (T1-T7)
has **not** been run; it waits for approval at STOP 3.

### Post-CV observation: near-duplicate training images (not part of the protocol)

Found while checking the example figure: IMG0002935 (fold 0) and IMG0004217 (fold 3) are the same radiograph (pixel
correlation 0.99998, different files). A read-only scan of the training images (`results/phase7b/post_cv/`, 32x32
thumbnail correlation > 0.99, then full-image correlation > 0.999) finds **67 near-duplicate pairs** over 120 images,
**58 pairs across folds**, involving **34 of the 576 fractured images** (17 pairs both fractured). FracAtlas has no
patient ID, so the frozen folds could not separate them (a known threat in every phase).

Sensitivity (descriptive, same out-of-fold detections, nothing refitted or re-selected):

| Images | H-wc AP@50 | S1 | P7 | H-wc - S1 | H-wc - P7 |
|---|---|---|---|---|---|
| All (3,266) | 0.0321 | 0.0011 | 0.0001 | +0.0311 [+0.0236, +0.0465] | +0.0320 [+0.0248, +0.0476] |
| Without any near-duplicate (3,146) | 0.0236 | 0.0011 | 0.0002 | +0.0226 [+0.0165, +0.0361] | +0.0235 [+0.0174, +0.0373] |
| Without cross-fold near-duplicates (3,164) | 0.0243 | 0.0010 | 0.0002 | +0.0233 [+0.0171, +0.0367] | +0.0241 [+0.0182, +0.0380] |

The cascade's CV AP@50 drops by about a quarter without the duplicates while S1 does not move, so stage 2 partly
memorizes repeated images across folds and the CV number above is **optimistic**. The B6 comparison still holds without
them (both CIs well above 0). B6 itself was applied exactly as written, on all training images. Whether test images
duplicate training images cannot be checked without reading the test set, which is allowed only after the 7b lock;
the same inflation could affect a test number.

### Reading

- Most of the gain comes from context: window-only HGB (0.010) is 10 times S1, and the ring context triples it again
  (B2). Calibration adds nothing measurable (B3).
- The cascade is still far from useful localization: 11% recall at IoU 0.5 against an 81% proposal ceiling, and AP@50:95
  stays below 0.02 because windows are fixed-size with no box regression.
- Leg and hand carry the result; shoulder is a failure, hip unresolvable.
- Threats carried into any test report: near-duplicate images inflate the cascade's CV AP@50 by about a third
  (0.024 without them); the threshold-dependent numbers are optimistic in CV; the stage-1 score shift
  (three-fold training proposals, four-fold held-out ones) is untested; the test would be a second look at the test
  set (post-hoc follow-up).

### Reproduce

Run with the repository's `.venv` Python (`results/phase7b/run_cv.sh` ran the Stage 3 steps below in order, with
`--n-jobs 3`, one log and one commit per step).
```
python tests/test_phase7b.py                      # unit tests, debug CV, mock final test
python src/phase7b_cascade.py --gates             # real-data gates (training images)
python src/phase7b_cascade.py --timing            # timing and runtime projection
python src/phase7b_cascade.py --stage1            # stage-1 models and cross-fitted proposals
python src/phase7b_cascade.py --cv-fold 0         # one outer fold per call (0 .. 4)
python src/phase7b_cascade.py --final-params      # final detector's stage-2 hyperparameters
python src/phase7b_cascade.py --threshold-cv 0    # only if B4 needs it
python src/phase7b_cascade.py --summarize         # B1-B4, tables, figures
python results/phase7b/post_cv/near_duplicates.py   # post-CV observation (training images)
python results/phase7b/post_cv/dup_sensitivity.py   # AP@50 without near-duplicate images (descriptive)
python src/phase7b_final_test.py                  # test set, ONCE, after approval (lock: results/phase7b/TEST_EVALUATED.json)
```
