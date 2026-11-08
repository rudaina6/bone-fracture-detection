"""Phase 2.1 sanity check: show original vs preprocessed images with the ground-truth boxes mapped
through the preprocessing geometry. If a box sits on the same anatomy in every column, the geometry
(rotation, padding, resizing) is correct. Also shows the EXIF-rotated image IMG0002628.

    python src/preview_preprocessing.py            # -> results/figures/phase2_preprocessing_preview.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import DATA_DIR, REPO_ROOT
from preprocess import load_gray, preprocess

VARIANTS = [("original", None), ("256 / CLAHE", dict(size=256, mid=448, norm="clahe")),
            ("448 / CLAHE", dict(size=448, mid=448, norm="clahe")),
            ("256 / z-score", dict(size=256, mid=448, norm="zscore"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=REPO_ROOT / "data" / "master.csv", type=Path)
    ap.add_argument("--data-dir", default=DATA_DIR, type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "results" / "figures" / "phase2_preprocessing_preview.png", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = pd.read_csv(args.master)
    m = m[(m.fractured == 1) & m.file_path.map(lambda p: (args.data_dir / p).exists())]
    picks = [m[m.region_group == g].sample(1, random_state=args.seed) for g in ["hand", "leg", "hip", "shoulder"]
             if (m.region_group == g).any()]
    rot = m[m.image_id == "IMG0002628.jpg"]                     # EXIF-rotated image
    big = m[(m.width * m.height > 1e6)].head(1)                 # one high-resolution image
    rows = pd.concat(picks + [rot, big]).drop_duplicates("image_id")

    fig, axes = plt.subplots(len(rows), len(VARIANTS), figsize=(3.2 * len(VARIANTS), 3.2 * len(rows)))
    for i, (_, r) in enumerate(rows.iterrows()):
        gray = load_gray(r.file_path, args.data_dir)
        boxes = json.loads(r.boxes_xywh)
        for j, (name, kw) in enumerate(VARIANTS):
            ax = axes[i, j]
            if kw is None:
                img, s, ox, oy = gray, 1.0, 0.0, 0.0
            else:
                img, g = preprocess(gray, **kw)
                s, ox, oy = g["scale"], g["off_x"], g["off_y"]
            ax.imshow(img, cmap="gray", vmin=0 if kw else None, vmax=1 if kw else None)
            for x, y, w, h in boxes:
                ax.add_patch(plt.Rectangle((x * s + ox, y * s + oy), w * s, h * s, fill=False, ec="red", lw=1.2))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r.image_id[3:-4]} {r.region_group}  {name}" if j == 0 else name, fontsize=8)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
