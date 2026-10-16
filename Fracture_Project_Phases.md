# Multi-Region Bone Fracture Detection and Localization Under High Spatial Variance
### Detailed Project Plan (Classical ML) — CS-245, Fall 2026, NUST SEECS

**Dataset:** FracAtlas (hand, leg, hip, shoulder; 4,083 X-ray images; 717 fracture images with bounding boxes)
**Approach:** Classical machine learning only: handcrafted features + SVM / Random Forest / gradient boosting / logistic regression. No deep learning.
**Core question:** Does one unified model hold up across anatomical regions, or do region-specific models perform meaningfully better, and why?

> All numeric settings below (image size, grids, thresholds) are sensible starting values, not fixed requirements. Tune them where the data says otherwise and record the final choices in the report. Week numbers are relative to the start of the redesign; map them onto the real deadlines.

---

## Overview of the pipeline

1. **Prepare data** → parse labels, explore, freeze splits
2. **Extract features** → preprocess images, compute handcrafted features, scale and reduce
3. **Classification (unified)** → image-level fracture vs. no fracture, one model on all regions
4. **Classification (region-specific)** → same pipeline trained separately per region
5. **Compare** → unified vs. region-specific, per region, with a controlled analysis of why
6. **Close the gap** → improve the unified model without training four separate models
7. **Localization** → patch classifier + sliding window + non-maximum suppression, evaluated with IoU/mAP
8. **Report and presentation**

Classification (phases 3–6) is a complete project on its own. Localization (phase 7) builds on it and is reported as partial results if time runs short.

---

## Phase 0 — Setup
**Target: Week 1**

1. Install the extra libraries in the `fracture-detect` env: `scikit-learn`, `scikit-image`, `opencv-python`, `pandas`, `matplotlib`, `seaborn`, `joblib`, `imbalanced-learn`, and optionally `xgboost` or `lightgbm`.
2. Create the repo structure: `data/` (not committed), `notebooks/`, `src/` (feature extraction, evaluation helpers), `features/` (saved feature matrices, not committed), `results/`, `docs/`.
3. Create a shared experiment log with these columns: date, run id, region(s), feature set, model, imbalance strategy, key hyperparameters, CV score, test metrics, notes.
4. Write a short design note in `docs/` explaining why classical methods are used (course requirement) and listing the models and features being compared.

**Output:** working environment, repo, experiment log, design note.

---

## Phase 1 — Data Preparation and EDA
**Target: Week 1–2**

### 1.1 Load and understand the labels
- Load the metadata CSV that ships with FracAtlas (per-image region indicators, fractured flag, fracture count; verify the exact column names) and the bounding-box annotations (COCO JSON or YOLO txt).
- Build one master table with one row per image: image id, file path, region, fractured (0/1), number of fracture boxes, box coordinates, image width and height.
- Decide how to handle edge cases and write the decision down:
  - Images that contain more than one region (a "mixed" flag, if present)
  - Images with surgical hardware or implants
  - Images with multiple fractures

### 1.2 Exploratory analysis (each as a figure or table for the report)
- Image counts: region × fractured (confirm ~4,083 total, 717 fracture, ~17.6%)
- Fracture rate per region (imbalance can differ by region)
- Image size and aspect ratio distribution
- Bounding-box statistics: width, height, area as a fraction of the image, per region (this decides the patch sizes in phase 7)
- Sample grid: 8–10 fracture and 8–10 normal images per region with boxes overlaid
- Intensity histogram per region (checks for contrast/exposure differences across hospitals and machines)

### 1.3 Freeze the splits
- Stratified **80/20 train/test split**, stratified jointly on region × fractured, fixed random seed, saved to `splits.csv`.
- Within the training set, use **stratified 5-fold cross-validation** for all tuning (per-region subsets are small, so CV is more reliable than a fixed validation set).
- The test set is not touched until final evaluation. Every model in the project uses these exact splits.

**Output:** master label table, EDA notebook, `splits.csv`.

---

## Phase 2 — Preprocessing and Feature Engineering
**Target: Week 2–3**

### 2.1 Preprocessing (identical for every image)
1. Convert to grayscale.
2. Pad to a square (preserve aspect ratio), then resize to **256×256**.
3. Contrast enhancement with CLAHE (clip limit ≈ 2.0, tile grid 8×8).
4. Scale intensities to [0, 1].

### 2.2 Handcrafted features
| Family | Settings (starting point) | Approx. dimension |
|---|---|---|
| HOG (edges and shape) | 9 orientations, 16×16 pixels per cell, 2×2 cells per block | ~8,100 |
| LBP (local texture) | uniform LBP with (P=8, R=1) and (P=16, R=2), histograms over a 4×4 grid of image cells, concatenated | ~450 |
| GLCM (texture statistics) | 32 gray levels, distances {1, 3}, angles {0°, 45°, 90°, 135°}; contrast, dissimilarity, homogeneity, energy, correlation, ASM | ~48 |
| Gabor (frequency/orientation) | 3 frequencies × 4 orientations; mean and variance of each response | ~24 |
| Intensity and edge statistics | mean, std, skewness, kurtosis, 16-bin histogram, Canny edge density | ~21 |

### 2.3 Feature processing
- Standardize features with `StandardScaler` **fit on training data only** (inside CV folds).
- Dimensionality reduction: compare (a) none, (b) PCA keeping 95% of variance, (c) L1-based or tree-based feature selection. Justify the choice using the high-dimensional, small-sample setting (curse of dimensionality).
- Run a quick feature-family comparison with one fixed model (e.g. linear SVM) using CV only: each family alone, then combinations. Keep the best combination for all later experiments.

**Output:** feature extraction code in `src/`, saved feature matrices, a table comparing feature families.

---

## Phase 3 — Unified Classification Baseline
**Target: Week 3–4**

**Task:** image-level fracture vs. no fracture, one model trained on all four regions pooled.

### 3.1 Models and tuning grids (stratified 5-fold CV on the training set)
| Model | Hyperparameters to search |
|---|---|
| Logistic regression (L1 and L2) | C ∈ {0.01, 0.1, 1, 10, 100} |
| Linear SVM | C ∈ {0.01, 0.1, 1, 10} |
| RBF SVM | C ∈ {0.1, 1, 10, 100}, gamma ∈ {scale, 1e-3, 1e-4} |
| Random Forest | n_estimators ∈ {200, 500}, max_depth ∈ {None, 10, 20}, min_samples_leaf ∈ {1, 3, 5} |
| Gradient boosting (HistGradientBoosting or XGBoost) | learning_rate ∈ {0.03, 0.1}, max_depth ∈ {3, 5}, n_estimators ∈ {200, 400} |

Include a trivial "always predict no fracture" baseline for reference, and optionally k-NN as a simple comparison.

### 3.2 Class imbalance (compare all, keep the best)
- No handling (shows why accuracy is misleading)
- Class weights (`class_weight="balanced"`)
- Oversampling with SMOTE, applied **inside training folds only**
- Decision-threshold tuning: choose the threshold from cross-validated predictions using the precision–recall curve, with emphasis on recall (e.g. maximizing F2)

### 3.3 Evaluation (final test set, once)
- Recall (sensitivity), precision, F1, F2, specificity, ROC-AUC, PR-AUC, confusion matrix
- Reported overall **and per region**
- Bootstrap confidence intervals on the test metrics, because per-region test sets are small

**Output:** the best unified classifier, a full model comparison table, an imbalance-strategy comparison.

---

## Phase 4 — Region-Specific Classifiers
**Target: Week 4–5**

1. For each region (hand, leg, hip, shoulder), train a separate classifier using the **same features, model families, tuning grids, CV protocol, and imbalance strategy** as phase 3.
2. Use the region's slice of the same frozen train/test split.
3. Evaluate each region-specific model on its region's test images with the same metrics.
4. For a fair comparison, also evaluate the **unified model on each region's test subset**, so both approaches are scored on identical images.
5. Log everything in the shared experiment log.

**Rule:** any difference between unified and region-specific results must come from the region split itself, not from unequal tuning effort.

**Output:** four region-specific classifiers and results in the same format as phase 3.

---

## Phase 5 — Comparative Analysis
**Target: Week 5–6**

### 5.1 The main result
- Table and plots: unified vs. region-specific, per region, per metric (recall, precision, F1, PR-AUC).
- Significance: paired bootstrap or McNemar's test on the same test images.

### 5.2 Separating the causes (what makes this a real analysis, not just a comparison)
- **Learning curves:** performance vs. training-set size, for the unified and region-specific models. This separates "regions are different" from "per-region data is small."
- **Feature-space view:** PCA or t-SNE of the features colored by region and by fracture status, to see whether regions form separate clusters.
- **Feature importance per region:** do different regions rely on different features (e.g. texture vs. edges)?

### 5.3 Error analysis
- Collect false negatives and false positives from both approaches; view them per region.
- Check whether errors concentrate on subtle, joint-adjacent, or low-contrast cases, as the literature suggests.

### 5.4 Theory write-up (course-concept justification)
- Bias–variance trade-off: pooled data (more samples, more variance across regions) vs. specialized models (less data, less heterogeneity).
- Regularization and the curse of dimensionality for the high-dimensional features.
- Class imbalance and metric choice.
- Why handcrafted features struggle with subtle fractures.

**Output:** comparison results, analysis figures, draft of the results and theory sections.

---

## Phase 6 — Gap-Closing Strategy
**Target: Week 6–7**

Goal: improve the unified model so it approaches region-specific performance without training four separate models. Implement the first two, and the third if time allows:

1. **Region as an input feature:** add the region (one-hot) to the feature vector of the unified model.
2. **Per-region decision thresholds:** one shared model, with a separate threshold tuned per region.
3. **Per-region class weighting or resampling** inside the unified model.

Evaluate each against both the plain unified model and the region-specific models, using the same splits and metrics. Report what helped, what did not, and the likely reason.

**Output:** improved unified model and an assessment of how much of the gap it closes.

---

## Phase 7 — Localization (sliding window)
**Target: Week 7–9**

Uses the best feature set and classifier from the earlier phases.

### 7.1 Build the patch dataset (training images only)
- Choose 2–3 patch sizes from the box-size statistics in phase 1 (e.g. around the 25th, 50th, and 75th percentile of box size).
- **Positives:** patches with IoU ≥ 0.5 with a ground-truth fracture box.
- **Negatives:** patches with IoU < 0.1, sampled from fracture-free areas and from normal images, starting at about 3 negatives per positive. Discard ambiguous patches in between.
- Keep the image-level train/test split, so patches from a test image never appear in training.

### 7.2 Train the patch classifier
- Use a faster subset of features (e.g. HOG + LBP) to keep window scoring affordable.
- Tune with CV grouped by image, so patches from the same image stay in the same fold.

### 7.3 Hard-negative mining
- Run the classifier over training images, collect confident false-positive windows, add them as negatives, and retrain. Repeat for about two rounds.

### 7.4 Sliding-window inference
- Resize each test image to a fixed size, slide windows at each patch size with a stride of about patch/4 (coarser if runtime is a problem), and score every window.
- Apply non-maximum suppression (IoU threshold ≈ 0.3) to merge overlapping detections into final boxes with confidence scores.
- Image-level prediction: an image is "fracture" if its best window score passes the threshold.

### 7.5 Evaluation
- A predicted box is a true positive if IoU ≥ 0.5 with an unmatched ground-truth box.
- Report mAP@50, mAP@50:95, recall, precision, F1, per region and overall.
- If time allows: compare unified vs. region-specific patch classifiers.
- Expect modest absolute numbers; discuss the reasons (handcrafted features, small data, subtle fractures) as part of the results.

**Scope guard:** if time runs short, report localization on a subset (one region or one patch size) as partial results.

**Output:** working sliding-window pipeline, localization results, discussion of limits.

---

## Phase 8 — Final Report and Presentation
**Target: Week 9–10 (plus buffer)**

**Report structure:**
1. Introduction and problem statement
2. Related work (including the gap: no controlled unified vs. region-specific comparison on FracAtlas)
3. **Objectives as a standalone numbered list**
4. Dataset and EDA
5. Methods: preprocessing, features, models, imbalance handling, evaluation protocol
6. Results: unified vs. region-specific, learning curves, gap-closing, localization
7. Discussion: why the differences appear, limitations, what did not work
8. Conclusion and future work (e.g. where deep learning would be the natural next step)
9. A short note that classical methods were used per the course requirement

**Presentation:** centered on the unified vs. region-specific comparison and the causal analysis (learning curves, feature-space view).

**Output:** final report, slides, rehearsed delivery.

---

## Standing Practices
- One protocol for every model: same splits, same preprocessing, same metric script.
- Fit scalers, PCA, feature selection, and resampling on training data only (inside CV folds) to avoid leakage.
- Test set is used once, at the end of each stage's evaluation.
- Log every run in the experiment log.
- Keep buffer time before the final deadline; phases 3, 6, and 7 are the likeliest to overrun.

---

## Open Items
- [ ] Actual deadlines for remaining deliverables and the final presentation
- [ ] Confirm with the instructor whether pretrained CNN features used only as feature extractors are allowed (assume not allowed until confirmed)
- [ ] Confirm FracAtlas metadata column names and the treatment of multi-region and hardware images
- [ ] Choose patch sizes and window stride after the box-size statistics from phase 1
- [ ] Choose which gap-closing strategies to prioritize after phase 5 results
