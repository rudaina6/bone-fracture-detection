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
  max difference <= 1e-5.
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

## Code (planned)

| File | Role |
|---|---|
| `src/phase7_window_design.py` | The design analysis above (done; training boxes only). |
| `src/phase7_windows.py` | Pyramid levels, dense HOG / LBP / intensity window features, window enumeration with the content filter, folded linear scoring. |
| `src/phase7_detect.py` | Patch sampling, C tuning, training, mining, NMS, IoU, AP, the detection CV (and the threshold CV if needed), tables and figures. |
| `src/phase7_final_test.py` | T1-T6 with its own lock. |
| `tests/test_phase7.py` | The gates and self-tests above. |
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

---

## Results

*(Empty until the runs. The protocol above was fixed before any model was trained.)*

### Reproduce (planned)

Run with the repository's `.venv` Python.
```
python src/phase7_window_design.py      # window set (done), training boxes only
python tests/test_phase7.py             # gates and self-tests
python src/phase7_detect.py ...         # detection CV (training images only), step by step
python src/phase7_final_test.py         # test set, ONCE, after approval (lock: results/phase7/TEST_EVALUATED.json)
```
