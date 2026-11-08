# Aiman's Phase 2 scripts (kept alongside the shared pipeline)

Aiman built a parallel Phase 2 pipeline on `aiman_branch` (Oct 17-21) while the shared pipeline in `src/` was built
separately. Both were kept. They use the same `data/master.csv` and `results/splits.csv`, but different file layouts, so they
do not overwrite each other:

| | Shared pipeline (`src/*.py`) | Aiman's pipeline (`src/aiman/`, `src/Ablations.py`, `notebooks/02_features.ipynb`) |
|---|---|---|
| Preprocessing | `preprocess.py` (on the fly) | `src/aiman/preprocess.py` caches `features/images_<size>.npy` |
| Features | `src/extract_features.py` -> `features/<tag>/<family>.npy` | `src/aiman/extract_features.py` -> `features/<family>_<size>_<norm>.npy` |
| Comparison | `src/compare_families.py` (block weighting, C sweep, all reductions, logs to the experiment log) | `notebooks/02_features.ipynb` (figures, PCA variance) and `src/Ablations.py` |
| Hand-off to Phase 3 | `feature_utils.load_features()` | not provided; use the shared loader |

Use the shared pipeline for everything from Phase 3 on. To re-run Aiman's notebook or ablations from the repo root:
```
python src/aiman/preprocess.py --size 256
python src/aiman/extract_features.py --size 256 --norm clahe       # and --norm zscore
python src/aiman/preprocess.py --size 448; python src/aiman/extract_features.py --size 448 --norm clahe
python src/Ablations.py
```
The only change to her two scripts is that image paths now go through `src/paths.py`, because `master.csv` stores paths
relative to the FracAtlas folder. Everything else is as she wrote it.
