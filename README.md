# fracture-detect

Multi-region bone fracture detection and localization on FracAtlas using classical ML
(CS-245, Fall 2026, NUST SEECS).

**Where are we?** See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) (status, decisions, open items). The plan is `Fracture_Project_Phases.md`.

## Setup
```
conda create -n fracture-detect python=3.11 -y
conda activate fracture-detect
pip install -r requirements.txt
```

## Data
Download FracAtlas and extract it to `data/FracAtlas/` (it must contain `dataset.csv`, `images/`, `Annotations/`).
`data/master.csv` stores image paths relative to that folder; resolve them with `src/paths.py::image_path()`.
To keep the dataset elsewhere, set the environment variable `FRACATLAS_DIR` to its location.

## Phase 2 (preprocessing and features) - complete, see `docs/phase2_features.md`
```
python src/preview_preprocessing.py     # sanity-check figure: boxes must follow the anatomy
python src/extract_features.py          # -> features/s256_m448_clahe_c16/*.npy   (not committed, ~6-8 min)
python src/compare_families.py --block-weight --C 0.1      # feature-family comparison (CV, training images only)
python src/freeze_requirements.py       # pin requirements.txt to the installed versions
```
Later phases load the features with `from feature_utils import load_features, make_preprocessor`.
Preprocessing is in `src/preprocess.py`, the feature families in `src/feature_extractors.py`, and every
comparison run is appended to `results/experiment_log.csv` by `src/expt_log.py`.

## Phase 3 (unified classifier) - complete, see `docs/phase3_unified.md`
```
python src/evaluate.py                                    # self-test of the shared metrics (used by every phase)
python src/phase3_unified.py --models linsvm logreg_l2 logreg_l1 rbfsvm --imbalance balanced   # nested CV, ~10 min
python src/phase3_unified.py --models rf hgb --imbalance balanced        # ~45 min; add --fast for coarser grids
python src/phase3_unified.py --models rbfsvm linsvm --imbalance none smote
python src/phase3_unified.py --summarize                  # -> results/tables/phase3_cv_summary.csv, phase3_cv_per_region.csv
python src/phase3_final_test.py --model rbfsvm --imbalance none      # test set, ONCE (lock: results/phase3/TEST_EVALUATED.json)
```
Models, grids and imbalance strategies: `src/model_zoo.py`; CV protocol: `src/cv_protocol.py`; metrics and bootstrap CIs: `src/evaluate.py`.

## Phase 4 (region-specific classifiers) - complete, see `docs/phase4_regional.md`
```
python src/phase4_regional.py --models rbfsvm --imbalance none                                  # nested CV per region, ~2 min
python src/phase4_regional.py --models rbfsvm --imbalance none --fixed-params C=10 gamma=0.05   # sensitivity: no per-region tuning
python src/phase4_regional.py --models linsvm --imbalance none                                  # family check
python src/phase4_regional.py --summarize               # -> results/tables/phase4_cv_summary.csv, phase4_cv_paired.csv
python src/phase4_final_test.py --model rbfsvm --imbalance none   # test set, ONCE (lock: results/phase4/TEST_EVALUATED.json)
```
Each region model uses the same `model_zoo` / `cv_protocol` / `evaluate` code as Phase 3; the comparison partner is the
Phase 3 unified model on the same images (`results/phase3/oof_*.csv` in CV, `results/phase3/test_scores.csv` on the test set).

## Layout
| Folder | Purpose |
|---|---|
| `data/` | FracAtlas images and labels (not committed) |
| `notebooks/` | EDA and experiments |
| `src/` | feature extraction, evaluation helpers |
| `features/` | saved feature matrices (not committed) |
| `results/` | experiment log, tables, figures |
| `docs/` | design note, report drafts |

## Experiment log
Log every run in `results/experiment_log.csv`
(date, run id, regions, feature set, model, imbalance strategy, key hyperparameters, CV score, test metrics, notes).
