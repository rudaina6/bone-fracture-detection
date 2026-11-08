# Project status (living document)

**Last updated:** 2026-11-09 (Phase 2 closed) · **Read this first** if you are picking up the project. The plan is in
`Fracture_Project_Phases.md`; this file says where we actually are, what was decided, and how to reproduce it.
Percentages are rough, by planned effort (weeks in the plan), not by calendar.

## Dashboard

| Phase | What | Status | Done | Where the work is |
|---|---|---|---|---|
| 0 | Setup | Complete | 100% | `README.md`, `requirements.txt`, `docs/design_note.md`, `results/experiment_log.csv` |
| 1 | Data preparation and EDA | Complete | 100% | `data/master.csv`, `results/splits.csv`, `notebooks/01_eda.ipynb`, `docs/data_decisions.md` |
| 2 | Preprocessing and features | **Complete** | 100% | `docs/phase2_features.md`, `src/preprocess.py`, `src/feature_extractors.py`, `src/extract_features.py`, `src/compare_families.py`, `src/feature_utils.py` |
| 3 | Unified classifier | **Next: can start now** (uses `load_features()`) | 0% | |
| 4 | Region-specific classifiers | Not started | 0% | |
| 5 | Comparative analysis | Not started | 0% | |
| 6 | Gap-closing | Not started | 0% | |
| 7 | Localization | Not started | 0% | |
| 8 | Report and presentation | Not started | 0% | |
| | **Whole project** | | **~30%** | |

## Phase 2 summary (full detail: `docs/phase2_features.md`)
- **Choice:** 256 px, CLAHE, all five handcrafted families (8,641 dims), `StandardScaler` inside the CV folds, each family weighted 1/sqrt(dims), no dimensionality reduction. Reference linear SVM: C=0.1, `class_weight="balanced"`.
- **Score of the reference model** (5-fold CV on the 3,266 training images): PR-AUC 0.552 +/- 0.041 (chance 0.176), ROC-AUC 0.813, recall about 0.68 and F2 about 0.60 at the default threshold.
- **Compared and not better:** z-score instead of CLAHE (0.550), 448 px (0.537), PCA (0.541), tree selection (0.544); L1 selection failed (selected nothing after block weighting). Removing high-resolution images (0.540) or hardware images (0.525) changes the score only within noise.
- **Hand-off:** `from feature_utils import load_features, make_preprocessor` (see the top of `docs/phase2_features.md`). Features are regenerated with `python src/extract_features.py` (not committed).
- These CV scores also guided the choices, so they are slightly optimistic; the test set is untouched.

## Decisions to keep (details in `docs/data_decisions.md`)
- Images: 4,083 total, 719 fractured (the plan says 717; 719 is correct, see `docs/data_decisions.md`).
- Region groups: hand, leg (+leg-only), hip (+leg+hip), shoulder (+hand+shoulder), other (9 fractured images).
- Frozen split: `results/splits.csv` (80/20, stratified on region x fractured, 5 folds inside train). **Never regenerate it.**
- Image paths in `master.csv` are relative to the FracAtlas folder (`data/FracAtlas/` or `$FRACATLAS_DIR`).
- EXIF rotation is applied to every image. 21 normal images (20 hip, 1 shoulder) store unrotated width/height in the metadata; harmless (no boxes), and `check_images.py` reports them (80 problem files = 59 truncated + 21 rotated).
- Class imbalance in the comparison runs: `class_weight="balanced"`.

## Known caveats (carry into the report)
- Image-level split: the dataset has no patient ID, so views of one patient may fall in both train and test (396 images carry the `multiscan` flag).
- Hip and shoulder have only about 46-48 fractured training images and 11-12 fractured test images; their metrics will have wide confidence intervals.
- Tuning C and the feature set on the same CV folds makes the CV scores optimistic; use nested CV or report this.

## Open items (from the plan)
- [ ] Actual deadlines for remaining deliverables and the final presentation
- [ ] Ask the instructor whether pretrained CNN features used only as feature extractors are allowed (assume not)
- [x] FracAtlas metadata column names and treatment of multi-region / hardware images (Phase 1)
- [ ] Patch sizes and window stride for Phase 7 (box-size percentiles are in `results/tables/eda_box_side_percentiles_256.csv`)
- [ ] Which gap-closing strategies to prioritise (after Phase 5)

## When a phase is finished (checklist)
1. Update the dashboard and the phase detail in this file (status, percentage, date, key numbers).
2. Write or update `docs/phaseN_<topic>.md` with the decisions, the final numbers and the reasons.
3. Add the commands to reproduce it to `README.md`.
4. Make sure every run is in `results/experiment_log.csv` and every table/figure is in `results/`.
5. Re-pin `requirements.txt` if libraries changed (`python src/freeze_requirements.py`).
6. Commit and push, then tell the team what changed and what the next phase can now use.
