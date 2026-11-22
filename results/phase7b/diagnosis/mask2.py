"""Two-stage bone mask (training images only): air vs body by Otsu on the smoothed raw image; then, inside the body
and excluding saturated pixels, bone vs soft tissue by Otsu on the local-contrast image (raw minus a wide blur), so
that a global brightness gradient or a bright strip does not decide what is bone."""
import sys
from pathlib import Path
sys.path.insert(0, r"D:\bone-fracture-detection\src")
import numpy as np
from skimage.filters import gaussian, threshold_otsu
import phase7_detect as P, phase7_windows as W
from preprocess import load_gray, preprocess
from phase5_viz import setup
import matplotlib.pyplot as plt


def masks(gray):
    cl, geom = preprocess(gray, size=W.SIZE, mid=W.MID, norm="clahe")
    raw, _ = preprocess(gray, size=W.SIZE, mid=W.MID, norm="none")
    x0, y0, x1, y1 = geom["content"]
    content = np.zeros(cl.shape, bool)
    content[y0:y1, x0:x1] = True
    sm = gaussian(raw, 2)
    body = content & (sm >= threshold_otsu(sm[content]))
    sat = content & (raw >= 0.95)
    lc = sm - gaussian(raw, 12)                      # local contrast: bone is brighter than the tissue around it
    ok = body & ~sat
    bone = ok & (lc >= threshold_otsu(lc[ok])) if ok.sum() > 100 else np.zeros_like(ok)
    return cl, raw, geom, content, body, sat, bone, ok & ~bone


if __name__ == "__main__":
    T = P.train_table()
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    picks = [rng.choice(T.index[T.region == g]) for g in ("hand", "leg", "hip", "shoulder") for _ in range(2)]
    setup()
    fig, axes = plt.subplots(2, 8, figsize=(17.6, 4.6))
    for k, i in enumerate(picks):
        cl, raw, geom, content, body, sat, bone, tissue = masks(load_gray(T.file_path[i]))
        lab = np.full(cl.shape, np.nan)
        lab[content & ~body], lab[tissue], lab[bone], lab[sat] = 0, 1, 2, 3
        axes[0, k].imshow(cl, cmap="gray", vmin=0, vmax=1)
        axes[1, k].imshow(lab, cmap="viridis", vmin=0, vmax=3)
        axes[0, k].set_title(T.region[i], fontsize=8)
        for a in axes[:, k]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("Two-stage mask: purple air, blue soft tissue, green bone (local contrast), yellow saturated",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(sys.argv[1], dpi=100)
