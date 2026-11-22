# Phase 7: localization (protocol written 2026-11-22, before any model was trained)

Task: find the fractures, not only the fractured images. A patch classifier is slid over each image, overlapping
detections are merged with non-maximum suppression, and the result is scored with IoU and average precision (plan,
Phase 7). Status: `docs/PROJECT_STATUS.md`.

**This file records the protocol and the decision rules before any Phase 7 model was trained.** The only thing
computed before it is the window-set design analysis below, which uses the geometry of the training boxes only (no
image, no feature, no model). Anything added after results are in is marked `[added after results]` with the reason.

**The test set is used once**, by `src/phase7_final_test.py` with its own lock (`results/phase7/TEST_EVALUATED.json`),
after the CV results are written into this file and approved. Until then no test image is loaded, and test boxes
appear only in counts. Every patch inherits its image's frozen fold, so all patches of one image stay in one fold;
mining, tuning and thresholds use training images only. The Phase 3, 4 and 6 locks stay as they are.

## Starting point

| Fact | Number | Source |
|---|---|---|
| Fracture boxes | 924 on the 719 fractured images (1 box: 548 images, 2: 146, 3: 17, 4: 7, 5: 1); none degenerate or out of bounds | `data/master.csv` (COCO), read-only check 2026-11-22 |
| Training / test boxes | 748 / 176 (hand 389 / 95, leg 250 / 53, hip 49 / 11, shoulder 50 / 14, other 10 / 3) | `results/splits.csv` |
| Box side sqrt(w*h) at 448 px, training, p25 / p50 / p75 | hand 18.6 / 23.2 / 29.6, leg 22.6 / 29.3 / 41.6, hip 34.8 / 53.9 / 65.5, shoulder 19.5 / 26.8 / 33.0 | design analysis below |
| Box shape | long / short side: median 1.39, p75 1.81, p90 2.57 | read-only check |
| The plan's square windows | 32 / 48 / 64 px at 448 reach IoU >= 0.5 for only 51.6% of training boxes (52.5% even at stride 1): 37% of boxes are too small, 18% too elongated | read-only check |
| Image-level reference | Phase 3 unified RBF SVM: nested-CV PR-AUC 0.602 +/- 0.036; test PR-AUC 0.626 | `docs/phase3_unified.md` |

**Working size 448 px**, same preprocessing as every phase (`preprocess.py`: EXIF rotation, long side to 448, pad to
a square, CLAHE clip 2.0 tile 8). 512 would mostly upsample: 91% of the images are 454 px on the long side
(`docs/data_decisions.md:58`). Boxes are mapped with `preprocess()`'s geometry (`x_out = x_in * scale + off_x`).

## Design analysis: the window set (training boxes only, no model)

`src/phase7_window_design.py` (run 2026-11-22; `results/tables/phase7_window_design.csv`, `phase7_reachability.csv`,
`results/phase7/window_design.json`, `results/figures/phase7_window_design.png`).

- **Windows** are rectangles at a base size s with aspect ratios 1:1 (s x s), 1:2 (s x 2s, tall) and 2:1 (2s x s,
  wide), on the grid `x = i * w/4`, `y = j * h/4` (stride = window/4, as the plan asks), fully inside the 448 x 448
  image, kept only if at least half of the window lies in the content rectangle (not the black padding). That filter
  never removes a useful window: IoU >= 0.5 with a box inside the content means at least half the window covers the box.
- **Reachable** box: some window of the set has IoU >= 0.5 with it. Reachability is the upper bound on recall at
  IoU 0.5 and is reported next to recall in every results table.
- **Rule** (fixed before the analysis ran): from the candidate base sizes {12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80,
  96}, the smallest number of sizes with >= 90% of training boxes reachable; among those, the fewest windows per image
  (mean over the 3,266 training images). 12 px was added to the brief's 16-96 range so that the smallest boxes had a
  chance; the rule did not pick it.

**The rule's choice was base sizes 16 and 40 px** (six window types, 23,110 windows per image, 92.9% of training boxes
reachable: hand 96.1%, leg 88.0%, hip 83.7%, shoulder 100%). No single size reaches 90% (best 71.3%), and that pair
is also the cheapest set reaching 90% with any number of sizes.

**Adopted: base sizes 16, 40 and 64 px, nine window types, 24,213 windows per image.** *[Decided at the STOP 1 review,
2026-11-22, before any model was trained: adding 64 px lifts hip reachability from 83.7% to 91.8% (leg from 88.0% to
89.2%) for 4.8% more windows, and hip is the region Phases 5-6 focused on. The rule's choice is kept on record in
`results/phase7/window_design.json`; the numbers below are for the adopted set
(`python src/phase7_window_design.py --sizes 16 40 64`).]*

| Group | Training boxes | Reachable, IoU >= 0.5 | Reachable, IoU >= 0.3 | Positive windows (IoU >= 0.5) | Per box, median |
|---|---|---|---|---|---|
| All | 748 | **93.9%** | 99.7% | 3,511 | 4 |
| Hand | 389 | 96.1% | 100% | 1,790 | 4 |
| Leg | 250 | 89.2% | 99.2% | 1,117 | 4 |
| Hip | 49 | 91.8% | 100% | 305 | 6 |
| Shoulder | 50 | 100% | 100% | 244 | 4 |
| Other | 10 | 100% | 100% | 55 | 6 |

| Window (w x h, px at 448) | 16 x 16 | 16 x 32 | 32 x 16 | 40 x 40 | 40 x 80 | 80 x 40 | 64 x 64 | 64 x 128 | 128 x 64 | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Windows per image (mean) | 10,102 | 4,962 | 5,055 | 1,514 | 721 | 756 | 575 | 259 | 269 | 24,213 |

Leg (89.2%) stays just below 90% on its own; the rule is about all boxes.

## Pipeline

### Canonical window features (Dalal-Triggs style, computed densely)
Every window is resampled to a canonical 32 x 32 patch. Implementation: for each of the nine window types, the 448 px
image is resized by (32/w, 32/h) into one pyramid level of size `(round(448*32/w), round(448*32/h))` (16 x 16: 896 x
896; 40 x 80: 358 x 179; 128 x 64: 112 x 224), features are computed once per level, and windows are slices of the level's feature grid. A
window at cell (i, j) of its level covers image pixels `x = i * w/4`, `y = j * h/4`, w x h; the level's cell grid
reproduces the design grid exactly (a gate checks it).
- **HOG:** 9 orientations, 8 px cells, 2 x 2 cells per block, L2-Hys (as Phase 2), on the level image; a window is the
  3 x 3 blocks inside its 4 x 4 cells -> **324 dims**.
- **LBP:** uniform, P=8, R=1 (codes 0-9) on the level image; 10-bin histograms on a 2 x 2 grid of the window, each
  normalised to sum 1, computed with one integral image per code -> **40 dims**.
- **Intensity:** window mean and standard deviation of the level image, from integral images of x and x^2 -> **2 dims**.
- 366 dims per window. Family block weights 1/sqrt(dims) and `StandardScaler` fitted inside the folds, as
  `model_zoo.build` does.
- **Deviations from the Phase 2 features, and why:** 8 px instead of 16 px cells, one LBP scale instead of two, no
  GLCM or Gabor, a 2 x 2 instead of 4 x 4 LBP grid. Windows are 32 px in canonical form and are scored 24,213 times
  per image, so the descriptor has to be small, local and computable from shared per-level maps.

### Patch dataset (training images only)
- **Positives:** every window with IoU >= 0.5 with a ground-truth box (median 4 per box; 3,511 on all training images).
- **Negatives:** windows with IoU < 0.1 with every box of their image, from normal and fractured images, 3 per positive
  at the start, sampled uniformly from all such windows of the training images in use (all nine types), with a
  generator seeded 0 for every fold.
- Windows with 0.1 <= IoU < 0.5 are discarded.

### Classifier
Linear SVM, `class_weight="balanced"` (`model_zoo.build("linsvm", "balanced", {"C": C}, col_weights)`), C from the grid
{0.001, 0.01, 0.1, 1} (passed explicitly; the Phase 3 `linsvm` spec is not changed). C is chosen by
`cv_protocol.tune_cv` on the round-0 patches with the patches' own folds (= their images' folds), by patch PR-AUC, and
kept for the mining rounds. Why linear and not Phase 3's RBF: every window of every image has to be scored, and a
linear model folds into one dot product per window (the scaler and the block weights are folded into the weight vector
and the intercept). Per-region patch classifiers are not trained (hip has 49 and shoulder 50 training boxes);
per-region results come from the unified detector.

### Hard-negative mining
Two rounds (plan). In each round every training image in use is scored with the current classifier. Candidates are
windows with IoU < 0.1 to every box of that image and a score above 0 (the linear SVM's default threshold). NMS at IoU
0.3 is applied to the candidates, so that near-duplicate windows of one hot spot are not all added, and up to the 20
highest-scoring survivors per image are added as negatives. The classifier is retrained with the same C on all
positives and all negatives gathered so far.

### Inference
Score every window; greedy NMS at IoU 0.3 across all nine window types; keep at most 10 boxes per image. The
image-level score is the best window score (= the top box's score).

## Evaluation

### Detection CV (training images only)
For each outer fold k = 0..4:
1. build the round-0 patches from the training images in the other four folds;
2. choose C with `tune_cv` on those patches over their four folds (patch PR-AUC);
3. train round 0, mine round 1 on the other four folds' images, retrain, mine round 2, retrain;
4. score fold k's images with the round-0, round-1 and round-2 detectors (out-of-fold detections per round).

Fold k's images are never used to build, tune or mine its detector, so every out-of-fold number is honest.

### Metrics
- **Matching:** per image, detections are taken in descending score; each is a true positive if its IoU with an
  unmatched ground-truth box of the same image is >= t (the box with the highest IoU is taken), otherwise a false
  positive. Every detection on a normal image is a false positive. The ground-truth count is every box of the
  evaluated images.
- **AP@50:** all-point interpolated average precision at t = 0.5, single class "fracture". **AP@50:95:** mean of AP at
  t = 0.50, 0.55, ..., 0.95. Expected to be near zero by construction: fixed windows, no box regression.
- **Recall, precision, F1 at IoU 0.5** at the operating box-score threshold (below), overall and per region, over all
  images, so normal images contribute false positives.
- **Per region:** the same quantities on that region's images and boxes only.
- **CV aggregation: the fold mean.** AP and the image-level metrics are computed per outer fold, on that fold's
  images, and averaged over the five folds. Each fold has its own detector, possibly with its own C, so its scores are
  on its own scale, and pooling them would mix scales (the reason Phases 4 and 6 used fold means). Pooled AP over all
  out-of-fold detections is reported beside it. 95% CIs: bootstrap over images resampled within each fold (2,000
  resamples, seed 0), as `phase4_regional.fold_paired_bootstrap` does.
- **Reachability** (from the design analysis) next to every recall.
- **Secondary, descriptive:** recall at IoU >= 0.3 at the operating threshold (boxes are tiny, so a 2 px shift costs
  a lot of IoU); fold-mean AP@50 after mining rounds 0, 1 and 2.
- **Image level:** PR-AUC and ROC-AUC of the best-window score, against the Phase 3 unified classifier on the same
  images: in CV, fold-mean differences with `fold_paired_bootstrap` against `results/phase3/oof_rbfsvm__none.csv`; on
  test, a paired bootstrap against `results/phase3/test_scores.csv`. Overall and per region.
- **Hip (11 test boxes) and shoulder (14):** the CI half-width is reported as the smallest detectable effect (an
  effect that size would be detected about half the time); a null there is "not resolvable with this test set".

### Operating box-score threshold
One threshold for the final detector: the F1-optimal score at IoU 0.5 on pooled out-of-fold detections, from
detectors that share the final detector's configuration. If all five outer folds chose the C that `tune_cv` picks on
all training patches (the final detector's C), the pooled out-of-fold detections of the D1 round are used directly.
Otherwise a second pass (**threshold CV**) re-runs steps 1, 3 and 4 with C fixed at that value and the D1 number of
rounds, and the threshold comes from its pooled out-of-fold detections. Recall, precision and F1 at this threshold are
optimistic in CV because the threshold is chosen on the same detections (stated with them); the test gives the
honest number.

## Decision rules (fixed before any model)

D1. **Mining rounds.** The final detector uses 2 mining rounds (plan), unless the CV AP@50 after round 2 is lower than
    after round 1 with a 95% CI excluding 0 (fold-mean difference, fold-stratified paired bootstrap over images), in
    which case it uses 1. No other choice is made from CV results.
D2. **C and the box-score threshold come from training data only**, as above. The window set is fixed by the design
    analysis (training box geometry), not by any CV result.
D3. **Nothing is chosen from the test numbers.** The test reports the pre-registered detector once.

## Gates and self-tests (synthetic or training data only, before the detection CV)

- **Box mapping:** boxes mapped through `preprocess()` overlay the anatomy (figure of 12 training images, 3 per region,
  `results/figures/phase7_box_mapping.png`) and lie inside the content rectangle; the loaded image shape of every
  training fractured image equals the (height, width) the geometry was computed from.
- **Window grid:** the implementation's windows equal the design grid (same count per type for every image shape), and
  reachability recomputed from the implementation equals the design table.
- **Dense = naive:** for 500 random windows on training images, the dense slice features equal an independent
  per-window computation built to match (cropped from the level image with the margin the gradients and LBP need),
  max difference <= 1e-5. *[Clarified in Stage 2, before any model: the first run failed on LBP (difference 0.10)
  because skimage's LBP interpolates the diagonal neighbours with a rounding error that depends on the absolute pixel
  position, so near-tie pixels can get a different code in a crop than in the full image. The naive LBP is therefore
  computed at the window's own coordinates (the crop pasted into an otherwise empty level-sized image), which still
  shows that a window's codes depend only on its neighbourhood and that the slicing is right. HOG and intensity stay
  crop-based. Consequence: LBP codes are not exactly translation-invariant at near-ties, a negligible effect on
  features but worth knowing.]*
- **Folded linear scoring** equals `pipeline.decision_function` on the same windows, max difference <= 1e-6.
- **IoU, NMS and AP** against hand-computed cases (no detections, duplicates, several boxes in one image, images
  without boxes); AP@50 = 1 for perfect detections; an independent naive AP implementation agrees.
- **Folds:** no image's patches appear in two folds.
- **Timing:** full scoring of 10 training images, with an estimate of the detection CV and final test runtimes.

## Final test (run once, after approval at STOP 3)

T1. C: `tune_cv` on the round-0 patches of all 3,266 training images (computed in Stage 3, because the threshold
    rule needs it).
T2. Train on all training patches; mine the D1 number of rounds on all training images.
T3. The box-score threshold from Stage 3 (D2).
T4. `results/phase7/TEST_EVALUATED.json` (window set, C, rounds, threshold) is written **before any test image is
    loaded**.
T5. Score the 817 test images once; NMS at 0.3; at most 10 boxes per image.
T6. Report AP@50, AP@50:95, recall / precision / F1 at the threshold, recall at IoU >= 0.3, and test-box reachability
    (computed after the lock), overall and per region, with bootstrap CIs over images (2,000 resamples, seed 0); the
    image-level PR-AUC / ROC-AUC against the Phase 3 test scores (paired bootstrap); the hip and shoulder CI
    half-widths. No setting is changed after the test numbers are seen.

## Code

| File | Role |
|---|---|
| `src/phase7_window_design.py` | The design analysis above (done; training boxes only). |
| `src/phase7_windows.py` | Pyramid levels, per-image feature maps (HOG blocks, LBP per-cell code counts, intensity per-cell sums), window enumeration with the content filter, window features as slices of the maps, folded linear scoring as a correlation over the maps, IoU, NMS. |
| `src/phase7_detect.py` | Gates, timing, the map cache, round-0 patches, C tuning, training, mining, matching and AP, the detection CV (and the threshold CV if needed), tables and figures. |
| `src/phase7_final_test.py` | T1-T6 with its own lock. |
| `tests/test_phase7.py` | Unit and synthetic self-tests, an end-to-end debug CV on a few training images, and a mock final test (lock order, re-run refusal, T1 failure). The real-data gates are `phase7_detect.py --gates`. |

*[Implementation note, Stage 2, before any model: windows sit on the 8 px cell grid, so a window's LBP quarters and
intensity sums are sums of per-cell code counts and pixel sums (exact). The maps of each training image are computed
once (about 1-1.7 s per image) and cached in `features/phase7/maps/` (about 5 MB per image, 17 GB in all; not
committed). Every later scoring reads them and costs about 0.05-0.08 s per image instead of a fresh extraction. The
cache does not change any feature; the gates compare it with independent computations.]*

*[Timing, Stage 2 (`phase7_detect.py --timing`, `results/phase7/timing.json`; 10 training images): maps 1.7 s per
image and scoring + NMS + selection 0.08 s per image in one process. Four workers do not scale 4x on this machine
(measured 0.7-0.96 images/s for the maps, under 25-35% background load), so the realistic projections are: map cache
about 1 h; detection CV about 30-40 min (five folds of two mining passes over about 2,600 images and one scoring of
about 650, plus the LinearSVC fits on up to about 115,000 patches); the threshold CV, if needed, about the same; the
summary 10-20 min; the final test about 30 min (maps of the 817 test images computed after the lock).]*
| reused | `preprocess.py`, `model_zoo.build` (`linsvm`), `cv_protocol.tune_cv`, `evaluate.py`, `phase4_regional.fold_paired_bootstrap`, `expt_log.log_run`, `phase5_viz` |

Outputs: `results/phase7/`, `results/tables/phase7_*.csv`, `results/figures/phase7_*.png`; long runs log their console
output to `results/phase7/<step>.log`. Every fitting run gets a row in `results/experiment_log.csv`.

## Index: why each analysis is run, and what decides its answer

| Analysis | Why it is run | What decides the answer | Fixed when |
|---|---|---|---|
| Window design | Square 32-64 px windows reach only 52% of boxes | Rule above (>= 90% reachable, fewest sizes, fewest windows), then 64 px added at the STOP 1 review for hip | before (training box geometry only; no model) |
| Gates and self-tests | Dense features, folded scoring and AP must be right before anything is trusted | Exact or 1e-5 / 1e-6 agreement; hand-computed cases | before |
| Detection CV | Honest out-of-fold detection performance | Reported with fold-stratified CIs; no verdict except D1 | before |
| Mining rounds | The plan asks for two rounds; a third choice would be tuning | D1 | before |
| Box-score threshold | One operating point for the test | D2, F1-optimal on out-of-fold detections of the final configuration | before |
| Image-level comparison | Does localization also make a better image classifier than Phase 3? | Fold-mean paired difference with CI; descriptive | before |
| Final test | One honest number | T1-T6 | before |

## Known threats (carry into the report)

- **Reachability caps recall:** 93.9% of training boxes overall, 89.2% of leg and 91.8% of hip boxes.
- **IoU 0.5 is strict for tiny boxes.** The median hand box is 23 px at 448; a 2-3 px disagreement between annotator
  and window costs a lot of IoU. Recall at IoU 0.3 is reported for that reason.
- **AP@50:95 near zero by construction** (fixed windows, no box regression).
- **A linear patch model** is weaker than Phase 3's RBF SVM; it is the price of scoring 24,213 windows per image.
- **Resampling:** 16 px windows are upsampled 2x to the canonical patch, 128 px ones downsampled 4x, and 1:2 windows
  are stretched anisotropically, which changes gradient orientations.
- **Unannotated fractures** in fractured images would be sampled or mined as negatives; the boxes are assumed complete.
- **Scale mixing across folds** is avoided by fold means; the pooled numbers are shown for reference only.
- **Threshold optimism** in CV (above).
- **Small regions:** hip 49 / 11 and shoulder 50 / 14 training / test boxes; their test CIs will be wide.
- **No patient ID**, as in every phase.
- **The mining cap may bind everywhere.** *[Added in Stage 2, from the debug run on 90 training images, not a result:
  the round-0 detector scored more than 20 eligible negative windows above 0 in every image, so each round added
  exactly 20 per image. If that holds on the full data, mining is limited by the cap (as pre-registered), not by the
  supply of false positives.]*

---

## Results

Detection CV on the 3,266 training images, out of fold (Stage 3, 2026-11-22). Everything above was fixed before
these numbers existed; nothing in the protocol was changed after them. No test image has been loaded. Fold means with
95% CIs from 2,000 fold-stratified image bootstrap resamples (seed 0). Files: `results/tables/phase7_cv_detection.csv`,
`phase7_cv_rounds.csv`, `phase7_cv_image_level.csv`; `results/phase7/cv_summary.json`;
`results/figures/phase7_cv_rounds.png`, `phase7_cv_examples.png`; log `results/phase7/stage3_6_summarize.log`.

**Headline: the sliding-window detector does not localize fractures.** AP@50 is 0.0001 (95% CI 0.0001-0.0005)
although 93.9% of the boxes are reachable, and its best-window score does not separate fractured from normal images
(ROC-AUC 0.511; PR-AUC 0.196 at 17.6% prevalence), far below Phase 3's image classifier on the same images (ROC-AUC
0.822, PR-AUC 0.602).

### Run facts

| Step | Result |
|---|---|
| Map cache | 3,266 training images, 50.8 min, 16.5 GB; 0 fractured images with a shape mismatch, 0 boxes outside the content rectangle (16 normal hip/shoulder images are EXIF-rotated relative to master's height/width; they have no boxes, and `train_geometry.csv` holds the loaded shape) |
| Round-0 patches | 3,511 positive windows (= the design table, per region); negative pools of 40 per image |
| C per outer fold | 0.1 in all five (patch PR-AUC at C 0.1: 0.863-0.865) |
| Final detector's C (T1) | 0.1 (`tune_cv` on all training round-0 patches; patch PR-AUC 0.868, against 0.867 at C 1) |
| Threshold CV | not needed: every outer fold chose the final C (rule D2) |
| Hard negatives per round | 52,240-52,260 in every fold and both rounds, i.e. 20 per image: the mining cap binds everywhere, as the Stage 2 debug run suggested |
| Fold runtimes | 4.9, 23.5, 4.6, 7.7 and 7.8 min (folds 0-4) |

### Detection CV (D1 round: 2 mining rounds)

AP is the fold mean. Recall, precision and F1 are at the box-score threshold 0.7414, at IoU 0.5, with counts pooled
over the folds, and are **optimistic**: the threshold was chosen on these detections. Reachability (IoU >= 0.5) is the
ceiling on recall.

| Group | Images | Boxes | Reachable | AP@50 | AP@50:95 | Recall | Precision | F1 | Recall, IoU >= 0.3 |
|---|---|---|---|---|---|---|---|---|---|
| **All** | 3,266 | 748 | 93.9% | **0.0001** [0.0001, 0.0005] | 0.0000 [0.0000, 0.0001] | 2.4% [1.4, 3.5] | 0.21% [0.12, 0.30] | 0.0038 [0.0021, 0.0056] | 9.6% [7.6, 11.9] |
| Hand | 1,025 | 389 | 96.1% | 0.0006 [0.0002, 0.0035] | 0.0001 [0.0000, 0.0010] | 1.3% [0.3, 2.5] | 0.28% [0.06, 0.55] | 0.0047 [0.0010, 0.0090] | 6.7% [4.3, 9.2] |
| Leg | 1,699 | 250 | 89.2% | 0.0002 [0.0001, 0.0007] | 0.0000 [0.0000, 0.0001] | 4.0% [1.8, 6.6] | 0.21% [0.09, 0.36] | 0.0040 [0.0017, 0.0067] | 14.0% [9.8, 18.5] |
| Hip | 256 | 49 | 91.8% | 0.0004 [0.0000, 0.0020] | 0.0000 [0.0000, 0.0003] | 6.1% [0.0, 14.0] | 0.21% [0.00, 0.49] | 0.0041 [0.0000, 0.0095] | 18.4% [7.1, 31.4] |
| Shoulder | 267 | 50 | 100% | 0.0000 [0.0000, 0.0004] | 0.0000 [0.0000, 0.0000] | 0.0% | 0.00% | 0.0000 | 4.0% [0.0, 10.0] |

The 19 "other" images (10 boxes) count in "All" only, as in earlier phases. At the threshold the detector keeps 8,767
boxes on the 3,266 images (2.7 per image; at least one box on 79.5% of the images, normal ones included), and 18 of the
748 boxes are hit at IoU 0.5 (hand 5, leg 10, hip 3, shoulder 0); 72 at IoU 0.3. Pooled AP@50 over all out-of-fold
detections (reference only): all 0.0001, hand 0.0003, leg 0.0001, hip 0.0001, shoulder 0.0000.

### Mining rounds and D1

Fold-mean AP@50 after each round (`results/figures/phase7_cv_rounds.png`):

| Round | All | Hand | Leg | Hip | Shoulder |
|---|---|---|---|---|---|
| 0 | 0.0011 [0.0007, 0.0023] | 0.0034 [0.0017, 0.0103] | 0.0016 [0.0008, 0.0051] | 0.0003 [0.0000, 0.0026] | 0.0001 [0.0000, 0.0005] |
| 1 | 0 (no hit) | 0 | 0 | 0 | 0 |
| 2 | 0.0001 [0.0001, 0.0005] | 0.0006 [0.0002, 0.0035] | 0.0002 [0.0001, 0.0007] | 0.0004 [0.0000, 0.0020] | 0.0000 [0.0000, 0.0004] |

**D1: round 2 minus round 1 = +0.00015 [+0.00008, +0.00052].** Round 2 is not lower than round 1, so **the final
detector uses 2 mining rounds (plan)**. Round 1 is exactly 0: none of the 10 boxes of any training image hits a box at
IoU 0.5. Round 0 scored higher than both mined rounds (0.0011), but D1 compares rounds 2 and 1 only and "no other
choice is made from CV results", so nothing is changed for that; it goes into the report as a finding.

### Image level: best-window score against Phase 3

Fold means on the same 3,266 images and folds; difference = detector minus Phase 3 unified RBF SVM
(`results/phase3/oof_rbfsvm__none.csv`), `fold_paired_bootstrap`. Chance PR-AUC is the prevalence: all 0.176, hand
0.298, leg 0.100, hip 0.180, shoulder 0.180.

| Group | PR-AUC detector | PR-AUC Phase 3 | Difference | ROC-AUC detector | ROC-AUC Phase 3 | Difference |
|---|---|---|---|---|---|---|
| **All** | 0.196 | 0.602 | **-0.405** [-0.438, -0.364] | 0.511 | 0.822 | **-0.311** [-0.341, -0.279] |
| Hand | 0.356 | 0.637 | -0.280 [-0.327, -0.220] | 0.545 | 0.759 | -0.213 [-0.261, -0.165] |
| Leg | 0.151 | 0.584 | -0.434 [-0.495, -0.346] | 0.542 | 0.845 | -0.303 [-0.359, -0.246] |
| Hip | 0.218 | 0.531 | -0.313 [-0.416, -0.181] | 0.423 | 0.766 | -0.343 [-0.447, -0.236] |
| Shoulder | 0.311 | 0.636 | -0.325 [-0.422, -0.189] | 0.611 | 0.801 | -0.190 [-0.289, -0.097] |

The detector's best window is close to chance as an image classifier in every region (hip's ROC-AUC is below 0.5),
and Phase 3 is better everywhere, with every CI excluding 0.

### Operating box-score threshold (D2)

**0.7414**, the F1-optimal score at IoU 0.5 on the pooled round-2 out-of-fold detections (pooled F1 0.0038). Every
outer fold chose C 0.1 = the final detector's C, so these detections share the final configuration and no threshold
CV was run. Final test configuration (T1-T3): window set {16, 40, 64} px, C 0.1, 2 mining rounds, threshold 0.7414.

### Example detections

`results/figures/phase7_cv_examples.png`: out-of-fold detections at the threshold on 8 training images. The four
"good" images (the highest-scoring images with a hit) each carry several false boxes along the bone shafts and joints
next to the hit. The two "missed" images (lowest top score among images without a hit) have no box above the
threshold. The two "false alarms" (normal images with the highest top score) are covered in boxes along the long
bones and the pelvis.

### Checks run after the results (diagnostic, not pre-registered; nothing was chosen from them)

- **Evaluation:** an independent re-implementation (plain-loop IoU, greedy matching and all-point AP; sklearn for the
  image-level AUCs) reproduces the fold-mean AP@50 of rounds 0 / 1 / 2 (0.00106 / 0 / 0.00015) and the image-level
  PR-AUC / ROC-AUC (0.196 / 0.511).
- **Saved CV outputs:** refitting fold 0's round-0 detector from the same patches (C 0.1, seed 42) reproduces its
  saved detections on 80 fold-0 training images exactly (largest difference 0, scores and boxes).
- **Geometry:** no round's top box lies in the black padding; ground-truth boxes overlay the fractures in the
  example figure.
- **Why AP is near zero:** on 59 fractured fold-0 images, the round-0 detector ranks the image's best positive window
  (IoU >= 0.5 with a box) 76th of about 24,000 windows (median; quartiles 10th and 275th); it is in the top 0.1% for
  32% of the images. After NMS keeps 10 boxes, the other high-scoring bone windows win. The share of training boxes
  hit by any of an image's 10 boxes, at any score, is 11.6% after round 0, 0.0% after round 1 and 5.5% after round 2.
  After round 1 the top boxes shift towards the 16 x 16 and 16 x 32 windows (46% of top boxes, against 23% after
  round 0).

**Reading.** The patch classifier separates fracture patches from random patches (patch PR-AUC 0.87 at one positive per
three negatives), but at detection time an image has about 24,000 windows against about 4 positive windows per box,
and with HOG/LBP/intensity features a linear model scores bone edges, cortex and joints as high as fracture lines.
Hard-negative mining with 20 windows per image did not fix that: the first round lost every hit and the second
recovered a few. This is the "linear patch model" threat listed above at its worst; the test (T1-T6) will measure it
once.

*Run note: the first `--summarize` run (started 05:25) stopped with exit code 4 after printing its first line, with no
traceback (stderr was not logged), no exit line and no outputs. No code path in the summary exits with 4 (an exception
or a failed gate exits with 1 and a message), and the same code on the same CV files completes, so the process was
terminated from outside. It was rerun unchanged with stderr logged (exit 0, about 10 min); no code, CV output or
setting was changed.*

### Reproduce

Run with the repository's `.venv` Python.
```
python src/phase7_window_design.py --sizes 16 40 64   # window set, training boxes only
python tests/test_phase7.py                           # unit and synthetic self-tests
python src/phase7_detect.py --gates                   # real-data gates (training images)
python src/phase7_detect.py --cache                   # map cache: about 50 min, 16.5 GB (not committed)
python src/phase7_detect.py --patches                 # round-0 patches
python src/phase7_detect.py --cv-fold 0 1 2 3 4       # detection CV, 5-25 min per fold
python src/phase7_detect.py --final-c                 # final detector's C (T1)
python src/phase7_detect.py --summarize               # metrics, D1, threshold, tables, figures: about 10 min
python src/phase7_final_test.py                       # test set, ONCE, after approval (lock: results/phase7/TEST_EVALUATED.json)
```
