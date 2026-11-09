# Phase 3: unified classifier (COMPLETE, 2026-11-09)

Task: image-level fracture vs no fracture, one model trained on all regions pooled, on the Phase 2
features (`load_features()`, 8,641 dims, block-weighted). **This file records the protocol and the decision rules
before any result was seen** and were followed as written. Status: `docs/PROJECT_STATUS.md`.

## Code
| File | Role |
|---|---|
| `src/evaluate.py` | The ONE metric script for every phase: recall, precision, F1, F2, specificity, accuracy, ROC-AUC, PR-AUC, confusion counts; `best_threshold` (F-beta), per-region tables, bootstrap CIs, paired bootstrap (for Phase 5), never-fracture baseline. Run `python src/evaluate.py` for a self-test. |
| `src/model_zoo.py` | Models, tuning grids and imbalance strategies (`build`, `scores`, `SPECS`). Phase 4 imports the same objects, so unified and region-specific models cannot drift apart. |
| `src/cv_protocol.py` | Nested CV (`nested_cv`) and plain grid search (`tune_cv`) on the frozen folds; `load_train` drops the test rows immediately. |
| `src/phase3_unified.py` | CV-only experiments, results saved in `results/phase3/`, tables in `results/tables/phase3_cv_*.csv`. Never loads test labels into any computation. |
| `src/phase3_final_test.py` | The single test-set evaluation. Writes a lock file; a second run needs `--allow-rerun`. |

## Protocol
- **Data:** the 3,266 training images (576 fractured, folds 0-4 from `results/splits.csv`, never regenerated). Features are
  standardised and block-weighted inside every fold (`make_preprocessor`); feature selection and SMOTE are pipeline steps, so
  they are fitted on the training part of a fold only.
- **Nested CV (outer = the 5 frozen folds).** For each outer fold k: every grid configuration is scored by an inner CV over the
  other four folds (mean PR-AUC); the winner's inner out-of-fold scores give the F2-optimal threshold; the winner is refitted on
  the four folds and scored on fold k. All reported CV numbers are the outer scores, so they are **not** inflated by tuning
  C, the model or the threshold. (The Phase 2 numbers were tuned on the same folds and are slightly optimistic; the Phase 3
  numbers for the same linear SVM are the honest counterpart.)
- **Two operating points for every run:** `default` (score 0, or probability 0.5 for forest and k-NN) and `tuned`
  (F2-optimal threshold chosen inside each outer training set). F2 weights recall four times as much as precision (a missed
  fracture costs more than a false alarm).
- **Metrics:** PR-AUC (`average_precision_score`, chance = fracture rate 0.176) is the primary, threshold-free selection
  metric; ROC-AUC, recall, precision, F2, specificity as secondary. Accuracy is reported only to show why it is misleading
  (never-fracture baseline: accuracy 0.824, recall 0).
- **Per-region numbers** are pooled out-of-fold scores (hand, leg, hip, shoulder; "other" has 7 fractured training images and is
  only part of the overall numbers). Hip and shoulder have 46-48 fractured training images.

## Models and grids (differences from the plan, and why)
| Model | Grid | Note |
|---|---|---|
| Logistic regression L2 | C in {0.03, 0.1, 0.3, 1, 3, 10} | plan's {0.01..100} shifted: block weighting shrinks inputs (squared norm about 5) |
| Logistic regression L1 | C in {0.1, 0.3, 1, 3, 10, 30} | liblinear; L1 needs larger C (Phase 2: L1 at C=0.01 selected 0 features) |
| Linear SVM | C in {0.01, 0.03, 0.1, 0.3, 1} | centred on the Phase 2 optimum (0.1) as planned |
| RBF SVM | C in {0.1, 1, 10, 100} x gamma in {0.8, 0.2, 0.05, 0.0125} (0.8 added after run 1: gamma 0.2 was at the upper edge in 4/5 folds) | with block weighting sklearn's gamma="scale" equals 0.2 (checked); the plan's 1e-3 / 1e-4 assume unweighted features and would be almost linear. Kernel matrix computed with BLAS (`PrecomputedRBFSVC`, same model as `SVC(kernel="rbf")`, max difference 6e-7 in decision values) |
| Random forest | max_depth in {None, 10, 20} x min_samples_leaf in {1, 3, 5}; 500 trees fixed | n_estimators is not tuned (more trees only lower variance). Top-1,000 columns by ANOVA F inside the fold (`SelectKBest`) for speed |
| Gradient boosting (HistGB) | learning_rate {0.03, 0.1} x max_depth {3, 5} x max_iter {200, 400} | top-300 columns by ANOVA F inside the fold: boosting cost grows with the number of columns |
| k-NN (optional) | k in {5, 15, 45, 135}, distance-weighted | no class weights |
| never-fracture | | trivial baseline, reported next to every table |
`--fast` shrinks the rf and hgb grids if time is short (state it in the report if used). If a chosen value sits on the edge of
its grid in at least 3 of the 5 folds, the script prints a warning; extend that grid and re-run that model with `--force`.

## Imbalance strategies (plan 3.2)
`none`, `balanced` (class_weight="balanced"; "balanced_subsample" for the forest), `smote` (inside the pipeline, after
scaling and selection, default 1:1, k=5) and threshold tuning (the `tuned` operating point, reported for all of them).
Class weights are the Phase 2 default; the comparison below is done for the best model(s).

## Decision rules (fixed before seeing results)
1. **Model comparison** at `balanced`: the model family with the highest mean outer-fold PR-AUC. If the best two are within one
   fold-SD, take the simpler/faster one (order: linear SVM, L2 logistic, L1 logistic, RBF SVM, forest, boosting).
2. **Imbalance comparison** for the winner (and the linear SVM as the Phase 2 reference): the strategy with the highest
   PR-AUC; within one fold-SD, the cheaper one (none < class weights < SMOTE). Threshold tuning is then applied to the winner
   because the operating point is a separate decision from the ranking quality.
3. The winner goes to `phase3_final_test.py` exactly once. No model is chosen or changed after seeing test numbers.

## Known threats (carry into the report)
- No patient ID: views of one patient can sit in train and test (396 `multiscan` images), so CV and test scores can be too high.
- **Surgical hardware is nearly a label:** 97 of the 99 images with hardware are fractured (train 77/78, test 20/21). A classifier
  can partly "detect implants" instead of fractures. Phase 2 saw a small, noisy drop without them (0.552 to 0.525). The final test
  script reports recall with and without hardware (`phase3_test_hardware.csv`).
- Hip and shoulder test sets have 11 and 12 fractured images: bootstrap CIs are wide and per-region differences are mostly
  within noise. "other" (2 fractured test images) is not reported per region.
- The bootstrap CIs keep the threshold fixed, so they ignore the uncertainty of having chosen it.
- Threshold tuning picks the F2 maximum of a noisy curve; scores from different fold models are pooled for it.

## Results

### 1. Model comparison (nested CV on the 3,266 training images, class weights, mean of the 5 outer folds)
Chance PR-AUC 0.176. Never-fracture baseline: accuracy 0.824, recall 0, F2 0, ROC-AUC 0.5.

| Model | PR-AUC | ROC-AUC | Recall / F2 at default thr | Recall / precision / F2 at tuned thr | Minutes (4 cores) |
|---|---|---|---|---|---|
| **RBF SVM** | **0.583 +/- 0.030** | 0.823 | 0.460 / 0.478 | 0.799 / 0.353 / 0.634 | 3 |
| Linear SVM | 0.548 +/- 0.045 | 0.810 | 0.674 / 0.600 | 0.811 / 0.329 / 0.627 | 4 |
| Logistic regression L2 | 0.548 +/- 0.041 | 0.809 | 0.651 / 0.590 | 0.804 / 0.327 / 0.621 | 1 |
| Random forest | 0.529 +/- 0.032 | 0.809 | 0.158 / 0.187 | 0.812 / 0.322 / 0.622 | 15 |
| Gradient boosting | 0.528 +/- 0.014 | 0.797 | 0.379 / 0.407 | 0.837 / 0.290 / 0.607 | 32 |
| Logistic regression L1 | 0.507 +/- 0.042 | 0.792 | 0.646 / 0.567 | 0.819 / 0.309 / 0.614 | 2 |

- The nested linear SVM (0.548) is almost the Phase 2 number (0.552): tuning optimism on C was small.
- RBF beats the linear SVM in all 5 folds (paired differences +0.025, +0.012, +0.005, +0.057, +0.074; mean +0.035, larger than its fold-SD 0.030), so rule 1 selects it. Only 5 folds that share data: modest evidence.
- After threshold tuning all models are within 0.03 F2 of each other (0.607-0.634). Trees were 5-10x slower and not better.
- RBF grid: gamma 0.2 sat on the upper edge in 4/5 folds, so 0.8 was added and the RBF runs repeated. 0.8 was never chosen and the results were identical. Tree models use top-1,000 (RF) / top-300 (boosting) ANOVA-F columns, so they were not run on all 8,641 features.

### 2. Imbalance comparison (nested CV, mean of 5 outer folds)
| Model | Strategy | PR-AUC | Recall / F2 at default thr | Recall / precision / F2 at tuned thr |
|---|---|---|---|---|
| RBF SVM | **none** | **0.602 +/- 0.036** | 0.347 / 0.387 | 0.797 / 0.345 / 0.631 |
| RBF SVM | class weights | 0.583 +/- 0.030 | 0.460 / 0.478 | 0.799 / 0.353 / 0.634 |
| RBF SVM | SMOTE | 0.573 +/- 0.034 | 0.469 / 0.488 | 0.774 / 0.353 / 0.622 |
| Linear SVM | none | 0.566 +/- 0.029 | 0.312 / 0.352 | 0.779 / 0.353 / 0.624 |
| Linear SVM | class weights | 0.548 +/- 0.045 | 0.674 / 0.600 | 0.811 / 0.329 / 0.627 |
| Linear SVM | SMOTE | 0.543 +/- 0.038 | 0.649 / 0.586 | 0.787 / 0.334 / 0.616 |

Strategy differences (0.01-0.03 PR-AUC) are within one fold-SD and vanish at the tuned threshold (F2 0.62-0.63). SMOTE was never
best and is the slowest. Without class weights the default threshold gives poor recall, which is why the threshold is tuned.
**Rule 2 -> no reweighting + F2 threshold tuning** (highest PR-AUC, cheapest). The SMOTE RBF run used the grid without gamma 0.8 (not repeated; it is not the winner).
Phase 4 should use the same strategy (none + tuned threshold) for a fair comparison.

### 3. Final test evaluation (817 test images, 143 fractured; used once)
Model: RBF SVM, C=10, gamma=0.05, no reweighting, 8,641 features. Settings and threshold came from the 5 training folds
(CV PR-AUC 0.608, optimistic); F2 threshold -0.844 on the SVM margin. 95% bootstrap CIs, 2,000 resamples, threshold held fixed.

| Group | n | fractured | Recall | Precision | F1 | F2 | Specificity | ROC-AUC | PR-AUC | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 817 | 143 | 0.755 [0.684, 0.830] | 0.411 [0.353, 0.471] | 0.532 | 0.647 [0.585, 0.707] | 0.770 [0.739, 0.802] | 0.834 [0.793, 0.874] | 0.626 [0.544, 0.711] | 519/155/35/108 |
| Hand | 256 | 76 | 0.776 [0.679, 0.871] | 0.468 | 0.584 | 0.686 [0.604, 0.763] | 0.628 | 0.777 [0.711, 0.840] | 0.641 [0.533, 0.750] | 113/67/17/59 |
| Leg | 425 | 42 | 0.714 [0.564, 0.850] | 0.333 | 0.455 | 0.581 [0.456, 0.684] | 0.843 | 0.844 [0.762, 0.916] | 0.623 [0.465, 0.767] | 323/60/12/30 |
| Hip | 64 | 11 | 0.727 [0.428, 1.000] | 0.444 | 0.552 | 0.645 [0.364, 0.833] | 0.811 | 0.834 [0.641, 0.973] | 0.724 [0.445, 0.922] | 43/10/3/8 |
| Shoulder | 67 | 12 | 0.833 [0.583, 1.000] | 0.400 | 0.541 | 0.685 [0.455, 0.833] | 0.727 | 0.820 [0.649, 0.945] | 0.651 [0.390, 0.860] | 40/15/2/10 |

Full CIs for every cell: `results/tables/phase3_test_ci.csv`; counts: `phase3_test_confusion.csv`; per-image scores for Phase 4-5: `results/phase3/test_scores.csv`.
- At the **default** threshold (0) the same model has recall 0.392, precision 0.824, F2 0.438, specificity 0.982: it is conservative, so the tuned threshold is the operating point to report. The never-fracture baseline has accuracy 0.825 and recall 0, so accuracy alone would call this model worse than doing nothing (accuracy at the tuned threshold: 0.767).
- **CV vs test:** nested CV gave PR-AUC 0.602, tuned recall 0.797, precision 0.345, F2 0.631; the test set gives 0.626, 0.755, 0.411, 0.647. All inside the test CIs: no sign of overfitting to the folds, no surprise.
- **Regions:** all four per-region CIs overlap heavily, so no region is reliably better or worse; hip (11 fractured) and shoulder (12) are too small to say anything. One threshold gives different false-alarm rates (specificity 0.63 hand, 0.84 leg), which is the motivation for region-aware thresholds in Phase 6.
- **Hardware:** recall 0.756 without hardware (93/123) and 0.750 with it (15/20); the hardware images are not being recognised more easily, so there is no sign that implants drive the score. (21 images, 1 normal: the specificity there is meaningless.)
- Caveat: no patient ID, so near-duplicate views of a patient could inflate CV and test scores alike.

### Reproduce
```
python src/phase3_unified.py --models linsvm logreg_l2 logreg_l1 rbfsvm --imbalance balanced
python src/phase3_unified.py --models rf hgb --imbalance balanced            # 47 min
python src/phase3_unified.py --models rbfsvm linsvm --imbalance none smote
python src/phase3_unified.py --models rbfsvm --imbalance none balanced --force   # after adding gamma 0.8
python src/phase3_unified.py --summarize                                      # results/tables/phase3_cv_*.csv
python src/phase3_final_test.py --model rbfsvm --imbalance none               # once; lock file blocks a repeat
```
Log note: the RBF rows logged before gamma 0.8 was added (balanced, none, smote) used the grid {0.2, 0.05, 0.0125}; the later RBF none/balanced rows use the extended grid and give identical numbers. The final-test row is the one marked "Phase 3 FINAL test evaluation".
