"""Phase 1.3: freeze the train/test split and the CV folds.

Run from the repo root (after build_master.py):
    python src/make_splits.py

- Stratified 80/20 train/test split, stratified jointly on region x fractured, fixed seed.
- Stratified 5-fold CV assignment for the training images (fold 0-4); test images get fold = -1.
- Strata with fewer than MIN_STRATUM images are merged (per fractured label) so stratification works.

Output: results/splits.csv  (image_id, region, fractured, split, fold). Commit this file.
Every model in the project must use exactly these splits and folds.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from paths import REPO_ROOT

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
MIN_STRATUM = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=REPO_ROOT / "data" / "master.csv", type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "results" / "splits.csv", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    df["stratum"] = df["region_group"] + "_" + df["fractured"].astype(str)
    counts = df["stratum"].value_counts()
    rare = counts[counts < MIN_STRATUM].index
    if len(rare):
        print(f"Merging rare strata (<{MIN_STRATUM} images): {list(rare)}")
        df.loc[df["stratum"].isin(rare), "stratum"] = "rare_" + df["fractured"].astype(str)

    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=df["stratum"], random_state=SEED
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    df["fold"] = -1
    train = df.loc[train_idx]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, val_pos) in enumerate(skf.split(train, train["stratum"])):
        df.loc[train.index[val_pos], "fold"] = k

    out = df[["image_id", "region_group", "fractured", "split", "fold"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nSeed {SEED}, test size {TEST_SIZE}, {N_FOLDS} folds")
    print("\nImages per split x region_group x fractured:")
    print(out.groupby(["split", "region_group", "fractured"]).size().unstack("fractured", fill_value=0))
    print("\nTrain images per CV fold:")
    print(out[out["split"] == "train"]["fold"].value_counts().sort_index())
    if "hardware" in df.columns:
        print("\nHardware images per split (columns: hardware, fractured):")
        print(pd.crosstab(df["split"], [df["hardware"], df["fractured"]]))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()