import sys
from pathlib import Path
sys.path.insert(0, r"D:\bone-fracture-detection\src")
import numpy as np, pandas as pd
import phase7_detect as P, phase7_windows as W
from phase5_viz import setup
import matplotlib.pyplot as plt
D = Path(sys.argv[1])
TYPES = ["16x16", "16x32", "32x16", "40x40", "40x80", "80x40", "64x64", "64x128", "128x64"]
T = P.train_table()
fp = dict(zip(T.image_id, T.file_path))
d = pd.read_csv(D / "c_windows.csv.gz")


def montage(x, title, path, ncol=8, ctx=1.0):
    setup()
    nrow = int(np.ceil(len(x) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.45 * nrow))
    for n, (ax, r) in enumerate(zip(np.ravel(axes), x.itertuples()), 1):
        img, _ = W.load_image(fp[r.image_id])
        cx, cy = r.x + r.w / 2, r.y + r.h / 2
        half = max(r.w, r.h) * (0.5 + ctx)
        x0, y0, x1, y1 = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(448, cx + half)), int(min(448, cy + half)))
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, extent=(x0, x1, y1, y0))
        ax.add_patch(plt.Rectangle((r.x, r.y), r.w, r.h, fill=False, ec="#e8743b", lw=1.5))
        ax.set_title(f"#{n} {r.region} {TYPES[r.wtype]}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in np.ravel(axes)[len(x):]:
        ax.axis("off")
    fig.suptitle(title, x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


montage(d[d.kind == "pool_neg_used"].sample(40, random_state=1), "Random round-0 negatives used in training (n=40)", D / "m_neg.png")
montage(d[d.kind == "det_r0_fp"].sample(40, random_state=1), "Random round-0 false boxes (top 10 per image, n=40)", D / "m_fp0.png")
montage(d[d.kind == "gt_pos"].sample(24, random_state=1), "Random fracture windows (IoU >= 0.5, n=24; calibration)", D / "m_pos.png")
