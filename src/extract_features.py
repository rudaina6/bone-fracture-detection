"""Phase 2.2: preprocess every image in data/master.csv and save the handcrafted features.

    python src/extract_features.py                      # 256 px, mid 448, CLAHE, all families
    python src/extract_features.py --size 448 --hog-cell 28 --tag s448   # compare resolutions
    python src/extract_features.py --norm zscore
    python src/extract_features.py --limit 50           # quick smoke test

Output (not committed): features/<tag>/<family>.npy  (rows in master.csv order),
                        features/<tag>/ids.csv, features/<tag>/config.json
Computes ALL images (train and test) once; no fitting happens here, so nothing leaks. Scalers, PCA
and selection are fitted later inside the CV folds.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from feature_extractors import FAMILIES, extract
from paths import DATA_DIR, REPO_ROOT
from preprocess import NORMS, load_gray, preprocess


def one(rel_path, cfg, families, data_dir):
    img, _ = preprocess(load_gray(rel_path, data_dir), cfg["size"], cfg["mid"], cfg["norm"])
    return extract(img, families, cfg["hog_cell"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=REPO_ROOT / "data" / "master.csv", type=Path)
    ap.add_argument("--data-dir", default=DATA_DIR, type=Path)
    ap.add_argument("--out-dir", default=REPO_ROOT / "features", type=Path)
    ap.add_argument("--size", type=int, default=256, help="working size (square)")
    ap.add_argument("--mid", type=int, default=448, help="intermediate long side; 0 = skip")
    ap.add_argument("--norm", choices=NORMS, default="clahe")
    ap.add_argument("--hog-cell", type=int, default=16, help="HOG pixels per cell")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--tag", default=None, help="output sub-folder (default: built from the settings)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    args = ap.parse_args()

    cfg = dict(size=args.size, mid=args.mid, norm=args.norm, hog_cell=args.hog_cell)
    tag = args.tag or f"s{args.size}_m{args.mid}_{args.norm}_c{args.hog_cell}"
    out = args.out_dir / tag
    out.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master)
    if args.limit:
        master = master.head(args.limit)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5, batch_size=8)(
        delayed(one)(p, cfg, args.families, args.data_dir) for p in master["file_path"])
    for fam in args.families:
        arr = np.stack([r[fam] for r in res]).astype(np.float32)
        np.save(out / f"{fam}.npy", arr)
        print(f"  {fam:<10} {arr.shape}")
    master[["image_id"]].to_csv(out / "ids.csv", index=False)
    json.dump(dict(cfg, families=args.families, n_images=len(master)), open(out / "config.json", "w"), indent=1)
    print(f"{len(master)} images in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
