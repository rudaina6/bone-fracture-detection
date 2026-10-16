# fracture-detect

Multi-region bone fracture detection and localization on FracAtlas using classical ML
(CS-245, Fall 2026, NUST SEECS).

## Setup
```
conda create -n fracture-detect python=3.11 -y
conda activate fracture-detect
pip install -r requirements.txt
```

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
