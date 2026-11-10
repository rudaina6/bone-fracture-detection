"""Phase 5.3 (optional): measure how visible each annotated fracture actually is, from the images.

    python src/phase5_box_contrast.py                 # all fractured images -> results/tables/phase5_box_stats.csv
    options: --n-jobs 4  --limit N  --size 256 --mid 448 --norm clahe

Why this exists: the plan's 5.3 asks whether errors concentrate on "subtle" and "low-contrast" fractures, but
FracAtlas has no subtlety label. Size is already available from the metadata (`phase5_errors.box_px_256`);
contrast is not, so it is measured here - on EXACTLY the preprocessed image the features are computed from
(`src/preprocess.py` with the Phase 2 settings), not on the raw file, because that is what the model sees.

For the largest annotated box of each fractured image, mapped into the 256 x 256 input with the geometry
`preprocess` returns, against a surrounding ring of the same shape dilated by 50%:

    box_std         texture inside the fracture box (SD of pixel values, image scaled to [0, 1])
    ring_std        the same in the surrounding ring - the local "business" of the anatomy around it
    contrast_ratio  box_std / ring_std: below 1 means the fracture region is FLATTER than its surroundings
    weber           |mean(box) - mean(ring)| / mean(ring): how much the box stands out in brightness
    michelson       (max - min) / (max + min) inside the box
    box_px_256      the box side in input pixels (recomputed here from the true geometry, as a cross-check
                    on the metadata-only version in phase5_errors.py)

Non-fractured images have no box and are not listed; `phase5_errors.py` joins this file if it exists and
skips the contrast columns if it does not, so this script is never required.

Needs the FracAtlas images (`data/FracAtlas/` or $FRACATLAS_DIR), which are not in the repository.
"""
import argparse
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from expt_log import log_run
from paths import REPO_ROOT
from preprocess import NORMS, load_gray, preprocess


def _one(rel_path, boxes_xywh, size, mid, norm):
    boxes = ast.literal_eval(boxes_xywh) if isinstance(boxes_xywh, str) else []
    if not boxes:
        return None
    img, geom = preprocess(load_gray(rel_path), size=size, mid=mid, norm=norm)
    s, ox, oy = geom["scale"], geom["off_x"], geom["off_y"]
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])          # the largest annotated fracture
    x0, y0 = x * s + ox, y * s + oy
    bw, bh = max(w * s, 1.0), max(h * s, 1.0)

    def clip_slice(a, b, n):
        return slice(int(max(0, np.floor(a))), int(min(n, np.ceil(b))))

    ys, xs = clip_slice(y0, y0 + bh, size), clip_slice(x0, x0 + bw, size)
    box = img[ys, xs]
    if box.size < 4:
        return None
    # ring: the same box dilated by 50% in each direction, minus the box itself
    rys = clip_slice(y0 - bh / 4, y0 + bh * 1.25, size)
    rxs = clip_slice(x0 - bw / 4, x0 + bw * 1.25, size)
    patch = img[rys, rxs]
    mask = np.ones(patch.shape, bool)
    mask[ys.start - rys.start:ys.stop - rys.start, xs.start - rxs.start:xs.stop - rxs.start] = False
    ring = patch[mask]
    if ring.size < 4:
        return None
    eps = 1e-6
    lo, hi = float(box.min()), float(box.max())
    return dict(box_std=float(box.std()), ring_std=float(ring.std()),
                contrast_ratio=float(box.std() / (ring.std() + eps)),
                weber=float(abs(box.mean() - ring.mean()) / (ring.mean() + eps)),
                michelson=float((hi - lo) / (hi + lo + eps)),
                box_px_256_geom=float(np.sqrt(bw * bh)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N fractured images (debug)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--mid", type=int, default=448)
    ap.add_argument("--norm", default="clahe", choices=NORMS)
    ap.add_argument("--master", type=Path, default=REPO_ROOT / "data" / "master.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "tables" / "phase5_box_stats.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.master)
    f = m[(m.fractured == 1) & (m.n_boxes > 0)].reset_index(drop=True)
    if args.limit:
        f = f.head(args.limit)
    print(f"{len(f)} fractured images with boxes; preprocessing at size={args.size} mid={args.mid} "
          f"norm={args.norm} (the Phase 2 settings)", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_one)(r.file_path, r.boxes_xywh, args.size, args.mid, args.norm) for r in f.itertuples())
    ok = [(i, d) for i, d in enumerate(res) if d is not None]
    if not ok:
        raise SystemExit("no image could be measured - is the FracAtlas folder present "
                         "(data/FracAtlas/ or $FRACATLAS_DIR)?")
    out = pd.DataFrame([d for _, d in ok])
    out.insert(0, "image_id", f.image_id.iloc[[i for i, _ in ok]].to_numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(args.out, index=False)
    skipped = len(f) - len(ok)
    print(f"\nmeasured {len(ok)} images in {(time.time() - t0) / 60:.1f} min"
          + (f"; {skipped} skipped (box too small after resizing)" if skipped else ""))
    pd.set_option("display.width", 200, "display.max_columns", 20)
    j = out.merge(m[["image_id", "region_group"]], on="image_id")
    print("\nMedian per region:")
    print(j.groupby("region_group")[["box_std", "ring_std", "contrast_ratio", "weber", "michelson",
                                     "box_px_256_geom"]].median().round(3).to_string())
    if not args.no_log:
        med = j.groupby("region_group").contrast_ratio.median()
        log_run(regions="all fractured images, per region",
                feature_set="box-local contrast measured on the 256 px CLAHE input (not a model feature set)",
                model="n/a (measurement only, nothing fitted)", imbalance_strategy="n/a",
                key_hyperparameters=f"size={args.size} mid={args.mid} norm={args.norm}; largest box; "
                                    f"ring = box dilated 50%",
                cv_score="; ".join(f"{r} contrast_ratio median {v:.3f}" for r, v in med.items()),
                test_metrics="not used (measurement of the annotations, train and test alike)",
                notes="Phase 5.3 optional: box-local contrast proxies -> "
                      "results/tables/phase5_box_stats.csv")
    print(f"\nWrote {args.out}. src/phase5_errors.py picks it up automatically on its next run.")


if __name__ == "__main__":
    main()
