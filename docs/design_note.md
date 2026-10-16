# Design Note: Classical ML for Multi-Region Fracture Detection

**Course:** CS-245, Fall 2026, NUST SEECS
**Dataset:** FracAtlas (hand, leg, hip, shoulder; 4,083 X-ray images; 717 fracture images with bounding boxes)

## Why classical methods
The course requires classical machine learning, so this project uses handcrafted features with
SVM, Random Forest, gradient boosting and logistic regression. No deep learning is used.
(Open item: confirm with the instructor whether pretrained CNN features used only as feature
extractors are allowed. Assumed NOT allowed until confirmed.)

## Core question
Does one unified model hold up across anatomical regions, or do region-specific models perform
meaningfully better, and why?

## Protocol (same for every model)
- Frozen stratified 80/20 train/test split (region x fractured), fixed seed, saved in `splits.csv`
- Stratified 5-fold CV on the training set for all tuning
- Scalers, PCA, feature selection and resampling fit on training folds only
- Test set used once per stage
- Every run logged in `results/experiment_log.csv`

## Features compared
| Family | Captures |
|---|---|
| HOG | edges and shape |
| LBP (uniform, two radii, 4x4 grid) | local texture |
| GLCM | texture statistics |
| Gabor | frequency / orientation |
| Intensity and edge statistics | global intensity, edge density |

Preprocessing: grayscale, pad to square, resize 256x256, CLAHE, scale to [0, 1].

## Models compared
- Logistic regression (L1, L2)
- Linear SVM
- RBF SVM
- Random Forest
- Gradient boosting (HistGradientBoosting or XGBoost)
- Baselines: always predict "no fracture"; optionally k-NN

## Class imbalance strategies compared
No handling, class weights, SMOTE (inside folds only), decision-threshold tuning (F2).

## Metrics
Recall, precision, F1, F2, specificity, ROC-AUC, PR-AUC, confusion matrix; overall and per region,
with bootstrap confidence intervals. Localization: mAP@50, mAP@50:95.

## Starting values
All numeric settings (image size, grids, thresholds) are starting values and will be tuned;
final choices are recorded in the report.
