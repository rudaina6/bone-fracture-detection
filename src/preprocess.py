"""Phase 2.1: preprocess every image to a fixed square and cache it as one array.

Run from the repo root (after build_master.py):
    python src/preprocess.py --size 256
    python src/preprocess.py --size 448

Steps per image (see docs/data_decisions.md sections 6 and 8):
  1. Read as grayscale. cv2.imread applies the EXIF orientation tag, so IMG0002628
     comes out the same way round as its COCO box.
  2. Scale-match: downscale so the long side is at most MATCH_LONG_SIDE, using
     INTER_AREA. This is the mitigation for the resolution shortcut - the ~350
     high-resolution images are resampled the same way as everything else instead
     of being crushed 11x in a single step.
  3. Pad to a square with zeros, image centred (preserves aspect ratio).
  4. Resize to --size x --size with INTER_AREA.

Contrast normalisation (CLAHE vs per-image z-score) is deliberately NOT done here.
It is a variable being compared in Phase 2.2, so it is applied at feature-extraction
time instead - that way this cache is built once and serves both options.

Output:
    features/images_<size>.npy        uint8, shape (4083, size, size)
    features/images_<size>_ids.csv    image_id in row order

IMPORTANT: every later step must align to the ids file, not to master.csv's order.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MATCH_LONG_SIDE = 448


def preprocess_one(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"could not read {path}")

    # 2. scale-match - only ever downscale, never invent pixels
    h, w = img.shape
    if max(h, w) > MATCH_LONG_SIDE:
        s = MATCH_LONG_SIDE / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)

    # 3. pad to square, centred
    h, w = img.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = img

    # 4. resize to the working size
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(square, (size, size), interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--master", default=Path("data/master.csv"), type=Path)
    ap.add_argument("--out-dir", default=Path("features"), type=Path)
    args = ap.parse_args()

    root = Path.cwd()
    df = pd.read_csv(args.master)
    paths = [root / p for p in df["file_path"]]

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} images not found, e.g. {missing[0]}\n"
            "Run this from the repo root: python src/preprocess.py --size 256"
        )

    n = len(df)
    print(f"{n} images -> {args.size}x{args.size} "
          f"(scale-matched to long side {MATCH_LONG_SIDE} first)")
    out = np.zeros((n, args.size, args.size), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = preprocess_one(p, args.size)
        if (i + 1) % 250 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy = args.out_dir / f"images_{args.size}.npy"
    ids = args.out_dir / f"images_{args.size}_ids.csv"
    np.save(npy, out)
    df[["image_id"]].to_csv(ids, index=False)

    # ---- sanity checks ----
    print(f"\nWrote {npy}   shape {out.shape}   {out.nbytes / 1e6:.0f} MB")
    print(f"Wrote {ids}")
    print(f"\nIntensity: mean {out.mean():.1f}, std {out.std():.1f}, "
          f"min {out.min()}, max {out.max()}")
    blank = int((out.reshape(n, -1).std(axis=1) < 1).sum())
    print(f"Near-blank images (std < 1): {blank}   (expect 0)")

    per_region = pd.DataFrame({
        "region_group": df["region_group"],
        "mean": out.reshape(n, -1).mean(axis=1),
    }).groupby("region_group")["mean"].median().round(1)
    print("\nMedian intensity per region (compare with EDA section 9: "
          "hand 43, leg 44, hip 59, shoulder 90):")
    print(per_region.to_string())


if __name__ == "__main__":
    main()