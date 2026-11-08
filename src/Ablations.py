"""Phase 2 ablations: the two open questions in docs/data_decisions.md.

  Q1 (section 9)  Does a per-image z-score kill the brightness/region shortcut
                  that CLAHE leaves in place?
  Q2 (sections 7, 8)  Does a 448px working size help the edge features (HOG),
                  now that the fracture is bigger than one 16x16 HOG cell?

Run from the repo root, AFTER these have been produced:
    python src/extract_features.py --size 256 --norm zscore
    python src/preprocess.py       --size 448
    python src/extract_features.py --size 448 --norm clahe

Then:
    python src/ablations.py

Every high-dimensional run uses PCA to N_PCA components. That is not a shortcut:
section 2 of the notebook showed PCA scores BETTER than using all features
(0.456 vs 0.419) and 54x faster, and it lets the solver converge, so the
comparison is fair in both directions instead of being limited by the solver.

Output: results/tables/phase2_ablations.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (average_precision_score, fbeta_score,
                             precision_score, recall_score)
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED, N_PCA = 42, 400
REGIONS = ["hand", "leg", "hip", "shoulder"]
ALL = ["hog", "lbp", "glcm", "gabor", "intensity"]
ROOT = Path.cwd()
FEAT, TAB = ROOT / "features", ROOT / "results" / "tables"


def context(size):
    ids = pd.read_csv(FEAT / f"images_{size}_ids.csv")
    master = pd.read_csv(ROOT / "data" / "master.csv")
    splits = pd.read_csv(ROOT / "results" / "splits.csv")
    df = (ids
          .merge(master[["image_id", "region_group", "fractured"]], on="image_id",
                 how="left", validate="one_to_one")
          .merge(splits[["image_id", "split", "fold"]], on="image_id",
                 how="left", validate="one_to_one"))
    assert df.notna().all().all(), f"alignment failed for size {size}"
    tr = (df["split"] == "train").to_numpy()
    return (tr, df["fractured"].to_numpy(), df["region_group"].to_numpy(),
            PredefinedSplit(df.loc[tr, "fold"].to_numpy()))


def load(size, norm, families):
    mats = []
    for f in families:
        p = FEAT / f"{f}_{size}_{norm}.npy"
        if not p.exists():
            raise SystemExit(
                f"MISSING {p}\nRun: python src/extract_features.py "
                f"--size {size} --norm {norm}")
        mats.append(np.load(p))
    return np.hstack(mats) if len(mats) > 1 else mats[0]


def evaluate(X, label, question, use_pca):
    tr, y, region, cv = CTX[SIZE_OF[label]]
    Xtr, ytr, rtr = X[tr], y[tr], region[tr]

    steps = [("var", VarianceThreshold(0.0)), ("sc", StandardScaler())]
    if use_pca:
        # PCA is fitted inside each fold, which holds only 4/5 of the training
        # rows, so the component cap must come from the smallest fold, not from
        # the full training set.
        min_fold = min(len(i) for i, _ in cv.split(Xtr))
        k = min(N_PCA, Xtr.shape[1], min_fold - 1)
        steps.append(("pca", PCA(n_components=k, svd_solver="randomized",
                                 random_state=SEED)))
    steps.append(("clf", LinearSVC(C=0.01, tol=1e-3, class_weight="balanced",
                                   dual="auto", max_iter=2000, random_state=SEED)))

    t0 = time.time()
    s = cross_val_predict(Pipeline(steps), Xtr, ytr, cv=cv,
                          method="decision_function", n_jobs=1)
    pred = (s > 0).astype(int)
    row = {"question": question, "setup": label, "dim": X.shape[1],
           "PR_AUC": average_precision_score(ytr, s),
           "recall": recall_score(ytr, pred),
           "precision": precision_score(ytr, pred, zero_division=0),
           "F2": fbeta_score(ytr, pred, beta=2, zero_division=0)}
    for g in REGIONS:
        m = rtr == g
        row[f"AP_{g}"] = (average_precision_score(ytr[m], s[m])
                          if ytr[m].sum() > 1 else np.nan)
    row["sec"] = round(time.time() - t0)
    print(f"  {label:34s} dim {X.shape[1]:6d}  PR-AUC {row['PR_AUC']:.3f}  "
          f"F2 {row['F2']:.3f}  ({row['sec']}s)", flush=True)
    return row


# label -> (size, norm, families, use_pca). Order matters: references first.
RUNS = [
    ("Q1 brightness shortcut",
     [("intensity  256 clahe",      256, "clahe",  ["intensity"], False),
      ("intensity  256 zscore",     256, "zscore", ["intensity"], False),
      ("all+PCA    256 clahe",      256, "clahe",  ALL,           True),
      ("all+PCA    256 zscore",     256, "zscore", ALL,           True)]),
    ("Q2 image size",
     [("hog+PCA    256 clahe",      256, "clahe",  ["hog"],       True),
      ("hog+PCA    448 clahe",      448, "clahe",  ["hog"],       True),
      ("all+PCA    448 clahe",      448, "clahe",  ALL,           True)]),
]

SIZE_OF = {lab: sz for _, specs in RUNS for lab, sz, *_ in specs}
CTX = {}

if __name__ == "__main__":
    for size in sorted({sz for _, specs in RUNS for _, sz, *_ in specs}):
        if not (FEAT / f"images_{size}_ids.csv").exists():
            raise SystemExit(f"features/images_{size}_ids.csv not found. "
                             f"Run: python src/preprocess.py --size {size}")
        CTX[size] = context(size)

    rows = []
    for question, specs in RUNS:
        print(f"\n=== {question} ===")
        for label, size, norm, families, use_pca in specs:
            rows.append(evaluate(load(size, norm, families), label, question, use_pca))

    out = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB / "phase2_ablations.csv", index=False)
    print(f"\nWrote {TAB / 'phase2_ablations.csv'}\n")
    print(out[["question", "setup", "dim", "PR_AUC", "recall", "F2",
               "AP_hand", "AP_leg", "AP_hip", "AP_shoulder"]].round(3).to_string(index=False))