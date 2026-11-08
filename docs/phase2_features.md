# Phase 2: preprocessing and feature engineering (complete)

Completed 2026-11-09. This note records what was built, what was chosen and why, so Phases 3-7 can start from it.
Status of the whole project: `docs/PROJECT_STATUS.md`.

## What Phase 3 should use
```python
from feature_utils import load_features, make_preprocessor
d = load_features()                         # features/s256_m448_clahe_c16/, all five families
train = d["split"] == "train"
X, y, fold = d["X"][train], d["y"][train], d["fold"][train]       # 3,266 images, 576 fractured, folds 0-4
pipe = make_pipeline(make_preprocessor(d["col_weights"]), LinearSVC(C=0.1, class_weight="balanced"))
```
- `X` is the **raw** matrix; the scaler lives inside the pipeline so it is fitted on the training part of each fold only. Do not standardise the whole matrix first.
- Rows with `d["split"] == "test"` are for the final evaluation only.
- `d["region"]` gives hand / leg / hip / shoulder / other for the Phase 4+ region slices; `d["slices"]` maps each family to its columns.
- Regenerate the features (about 6-8 minutes) with `python src/extract_features.py`; the files in `features/` are not committed.

## The Phase 2 choice
| Item | Choice |
|---|---|
| Preprocessing | EXIF rotation, grayscale, resize to long side 448 (area interpolation), centre-pad with black to a square, resize to **256 x 256**, **CLAHE** (clip 2.0, 8x8 tiles), scale to [0, 1] (`src/preprocess.py`) |
| Features | **All five families** (8,641 dims): HOG 8,100 (9 orientations, 16 px cells, 2x2 blocks), LBP 448 (P=8,R=1 and P=16,R=2 over a 4x4 grid), GLCM 48, Gabor 24 (3 frequencies x 4 orientations), intensity/edge 21 (`src/feature_extractors.py`) |
| Scaling | `StandardScaler` fitted inside each CV fold, then **each family multiplied by 1/sqrt(its number of dims)** so every family has equal total weight (`make_preprocessor`) |
| Dimensionality reduction | **None** (PCA 95% and tree-based selection gave no gain) |
| Reference model | Linear SVM, `class_weight="balanced"`, **C = 0.1** with the block weighting |

## Evidence (5-fold CV on the 3,266 training images; PR-AUC mean +/- SD over folds; chance = 0.176)

All numbers are for all five families, linear SVM, unless stated. Fold-to-fold SD is about 0.03-0.05, so differences below that are ties.

**Equal family weight and C.** Without weighting the 8,100 HOG columns swamp the other families and the best C is very small:

| C | 0.01 | 0.001 | 0.0003 | 0.0001 | 0.00003 |
|---|---|---|---|---|---|
| no weighting | 0.423 | 0.468 | 0.498 | 0.518 | 0.521 |

| C | 0.01 | 0.03 | **0.1** | 0.3 | 1 |
|---|---|---|---|---|---|
| block-weighted | 0.521 | 0.542 | **0.552** | 0.550 | 0.529 |

**Settings compared at block-weighted C=0.1:**

| Setting | PR-AUC | Decision |
|---|---|---|
| 256 px, CLAHE (baseline) | **0.552 +/- 0.041** (ROC-AUC 0.813) | **chosen** |
| 256 px, z-score instead of CLAHE | 0.550 +/- 0.027 | tie; CLAHE kept (plan default) |
| 448 px, CLAHE, HOG 28 px cells | 0.537 +/- 0.027 | no gain, and extraction took about 3x longer; 256 kept |
| PCA keeping 95% variance | 0.541 +/- 0.043 | no gain |
| Tree-based selection (ExtraTrees, median threshold) | 0.544 +/- 0.032 | no gain |
| L1-based selection | failed | selected 0 features after block weighting; not pursued because tree-based selection already covers plan item 2.3(c) |
| Without the 351 high-resolution images (2,994 train) | 0.540 +/- 0.035 | small drop, within noise |
| Without the 99 hardware images (3,188 train, 499 fractured) | 0.525 +/- 0.061 | small drop, very noisy (77 fractured images removed) |

**Single families** (C=0.1, block-weighted, 256 px CLAHE): HOG 0.467, LBP 0.437, GLCM 0.412, Gabor 0.385, intensity 0.343. The top feature subsets (for example HOG+GLCM+intensity+Gabor, 0.556) are tied with all five within noise, so no subset was chosen: fewer decisions made on the same CV folds.

**Per region** (out-of-fold, all five families; PR-AUC vs the region's fracture rate): hand 0.587 (0.298), leg 0.559 (0.100), hip 0.354 (0.180), shoulder 0.489 (0.180). Every region beats its own base rate; leg shows the largest gain over chance. Hip and shoulder have only 46 and 48 fractured training images.

**Intensity is not a region shortcut:** it is the weakest family alone (0.343) and removing it changes the full-set score by less than 0.005. (The Phase 1 EDA had flagged it as a possible shortcut.)

## Cross-check with Aiman's pipeline (`aiman_branch`)
Aiman built the same phase independently (preprocess, five families, family and reduction comparison, resolution and contrast
ablations); her scripts, notebook, ablations and tables are merged in (`src/aiman/`, `src/Ablations.py`,
`notebooks/02_features.ipynb`, `results/tables/features_*` and `phase2_ablations.csv`). Same data, same frozen folds, linear SVM, PR-AUC.
Her runs use no family weighting, so they are comparable with the "no weighting" rows above, not with the 0.552.

| Question | Aiman's branch (unweighted) | This pipeline |
|---|---|---|
| All five families, no reduction | 0.419 | 0.423 (C=0.01, unweighted) |
| PCA | 0.456 (400 components) | 0.541 (95% variance, block-weighted) |
| L1 selection | **0.458** (1,093 of 8,641 columns kept) | failed once block weighting was applied (selected 0 features) |
| Tree selection | 0.415 | 0.544 (block-weighted) |
| CLAHE vs z-score (all families + PCA) | 0.456 vs 0.435 | 0.552 vs 0.550 |
| 256 vs 448 px (all families + PCA / block-weighted) | 0.456 vs 0.440 | 0.552 vs 0.537 |
| Intensity alone, CLAHE vs z-score | 0.334 vs 0.296 | 0.343 (CLAHE) |

Both pipelines agree on the conclusions: 448 px does not help, CLAHE is at least as good as z-score, intensity statistics are not a shortcut
and PCA is a reasonable choice. The one difference is the role of reduction. With unscaled family blocks, PCA (+0.04) and
L1 selection (+0.04) clearly help because HOG's 8,100 columns dominate; after equal family weighting the full feature set already scores
0.552 and reduction no longer adds anything. Weighting and reduction are two ways of fixing the same imbalance.
Aiman's code also drops constant columns (`VarianceThreshold`, the black padded corners of HOG); the shared pipeline leaves them in, which is
harmless for scaling (they become zeros) but counts them in the HOG block size.
Her L1 setting (C=0.05, applied to the standardised features without family weights) might still be worth trying inside Phase 3 for the non-linear models.

## Caveats for the report
- These scores were used to choose C, resolution, weighting and the feature set on the same folds, so they are slightly optimistic. Phase 3 should tune inside nested CV or say so. The test set has not been used.
- The decision threshold is the default (score > 0); recall is about 0.68 and F2 about 0.60 there. Threshold tuning belongs to Phase 3.2.
- Only a linear SVM was used for the comparison. Reduction or selection may matter for the RBF SVM, Random Forest and boosting in Phase 3 (speed with 8,641 dims), even though it did not help here.
- 21 normal images (20 hip, 1 shoulder) carry an EXIF rotation tag, so their stored width/height are the unrotated sizes. They have no boxes, so nothing is affected; `check_images.py` reports them (80 problem files = 59 truncated + 21 rotated).
- The image-level split cannot separate views of the same patient (no patient id; 396 images have the `multiscan` flag).

## Reproduce
```
python src/preview_preprocessing.py
python src/extract_features.py                                   # baseline features
python src/compare_families.py --block-weight --C 0.1            # the chosen setting
python src/compare_families.py --block-weight --C 0.01 0.03 0.1 0.3 1 --no-log     # C sweep
python src/extract_features.py --norm zscore;  python src/compare_families.py --tag s256_m448_zscore_c16 --block-weight --C 0.1
python src/extract_features.py --size 448 --hog-cell 28;  python src/compare_families.py --tag s448_m448_clahe_c28 --block-weight --C 0.1
python src/compare_families.py --block-weight --C 0.1 --exclude-hires      # and --exclude-hardware
python src/compare_families.py --block-weight --C 0.1 --reduction pca      # and tree
```
Every run is in `results/experiment_log.csv`; result tables are in `results/tables/feature_family_*.csv`.
