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
