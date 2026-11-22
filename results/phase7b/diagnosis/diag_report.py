"""Report for the READ-ONLY Phase 7b diagnosis (training images only). Reads the scratch outputs of diag_phase7b.py and
writes tables / figures into this scratch folder only."""
import sys
from pathlib import Path

REPO = Path(r"D:\bone-fracture-detection")
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd

import phase7_detect as P
import phase7_windows as W
from preprocess import load_gray, preprocess
from phase5_viz import setup
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TYPES = ["16x16", "16x32", "32x16", "40x40", "40x80", "80x40", "64x64", "64x128", "128x64"]
CATS = ["padding border", "very bright", "bone interior", "bone edge", "soft tissue", "background (air)"]


def category(d):
    c = np.select([d.pad_frac >= 0.10, d.bright >= 0.25, d.bone >= 0.9, d.bone >= 0.1, d.tissue >= d.air],
                  CATS[:5], CATS[5])
    return pd.Categorical(c, CATS)


def groups(d):
    g = {
        "fracture windows (IoU>=0.5 with a box)": d.kind == "gt_pos",
        "round-0 negatives: pool (uniform, 40/image)": d.kind.str.startswith("pool_neg"),
        "round-0 negatives used (final detector)": d.kind == "pool_neg_used",
        "round 0: top box when false (rank 0)": (d.kind == "det_r0_fp") & (d["rank"] == 0),
        "round 0: all false boxes (top 10)": d.kind == "det_r0_fp",
        "round 0: hits (TP boxes)": d.kind == "det_r0_tp",
        "round 1: all false boxes (top 10)": d.kind == "det_r1_fp",
        "round 2: all false boxes (top 10)": d.kind == "det_r2_fp",
        "round 2: hits (TP boxes)": d.kind == "det_r2_tp",
        "mined negatives, round 1 (fold 0)": d.kind == "mined_r1",
        "mined negatives, round 2 (fold 0)": d.kind == "mined_r2",
    }
    return g


def summary(d):
    d = d.assign(cat=category(d))
    rows = []
    for name, m in groups(d).items():
        x = d[m]
        if not len(x):
            continue
        row = dict(group=name, n=len(x), bone_frac=x.bone.mean(), edge_density=x.edges.mean(), rel_gradient=x.grad.mean(),
                   clahe_mean=x.clahe.mean(), raw_mean=x.raw.mean(), bright=x.bright.mean(), pad_frac=x.pad_frac.mean(),
                   on_hardware_image=x.hardware.mean(), small_16px=x.wtype.isin([0, 1, 2]).mean())
        row.update({f"% {c}": 100 * (x.cat == c).mean() for c in CATS})
        rows.append(row)
    return pd.DataFrame(rows)


def crops(d, title, path, n=24, ncol=8, ctx=1.0):
    setup()
    fp = dict(zip(P.train_table().image_id, P.train_table().file_path))
    nrow = int(np.ceil(min(n, len(d)) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2.15 * nrow))
    for ax, r in zip(np.ravel(axes), d.head(n).itertuples()):
        img, _ = W.load_image(fp[r.image_id])
        cx, cy = r.x + r.w / 2, r.y + r.h / 2
        half = max(r.w, r.h) * (0.5 + ctx)
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W.SIZE, cx + half)), int(min(W.SIZE, cy + half))
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, extent=(x0, x1, y1, y0))
        ax.add_patch(plt.Rectangle((r.x, r.y), r.w, r.h, fill=False, ec="#e8743b", lw=1.5))
        ax.set_title(f"{r.region} {TYPES[r.wtype]} b{r.bone:.1f} e{r.edges:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in np.ravel(axes)[min(n, len(d)):]:
        ax.axis("off")
    fig.suptitle(title, x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mask_check(T, path, n=8):
    from skimage.filters import gaussian, threshold_multiotsu
    setup()
    rng = np.random.default_rng(3)
    picks = [rng.choice(T.index[T.region == g]) for g in ("hand", "leg", "hip", "shoulder") for _ in range(2)]
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for k, i in enumerate(picks):
        r = T.loc[i]
        gray = load_gray(r.file_path)
        cl, geom = preprocess(gray, size=W.SIZE, mid=W.MID, norm="clahe")
        raw, _ = preprocess(gray, size=W.SIZE, mid=W.MID, norm="none")
        x0, y0, x1, y1 = geom["content"]
        content = np.zeros(cl.shape, bool)
        content[y0:y1, x0:x1] = True
        sm = gaussian(raw, 2)
        t1, t2 = threshold_multiotsu(sm[content], classes=3)
        lab = np.full(cl.shape, np.nan)
        lab[content & (sm < t1)], lab[content & (sm >= t1) & (sm < t2)], lab[content & (sm >= t2)] = 0, 1, 2
        axes[0, k].imshow(cl, cmap="gray", vmin=0, vmax=1)
        axes[1, k].imshow(lab, cmap="viridis", vmin=0, vmax=2)
        axes[0, k].set_title(f"{r.region}", fontsize=8)
        for a in axes[:, k]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("Bone mask check: 3-class multi-Otsu on the raw image (purple air, teal soft tissue, yellow bone)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40)
    T = P.train_table()
    d = pd.read_csv(HERE / "c_windows.csv.gz")
    s = summary(d)
    s.to_csv(HERE / "r_appearance_summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    # Q3 by region and by image class
    q = d[d.kind.str.startswith("pool_neg")].assign(cat=lambda x: category(x))
    print("\nround-0 negatives (pool) by region: % per category")
    print((pd.crosstab(q.region, q.cat, normalize="index") * 100).round(1).to_string())
    fpos = d[d.kind == "gt_pos"].assign(cat=lambda x: category(x))
    print("\nfracture windows by region: % per category")
    print((pd.crosstab(fpos.region, fpos.cat, normalize="index") * 100).round(1).to_string())
    # top false windows: the highest-scoring round-0 rank-0 false boxes on normal images, and round-1 ones
    top0 = d[(d.kind == "det_r0_fp") & (d["rank"] == 0) & (d.fractured == 0)].sort_values("score", ascending=False)
    top1 = d[(d.kind == "det_r1_fp") & (d["rank"] == 0) & (d.fractured == 0)].sort_values("score", ascending=False)
    pos = d[d.kind == "gt_pos"].sample(24, random_state=0)
    crops(top0, "Round 0: highest-scoring false windows (top box of normal training images; b = bone fraction, e = "
          "edge density)", HERE / "r_top_fp_round0.png")
    crops(top1, "Round 1: highest-scoring false windows (top box of normal training images)", HERE / "r_top_fp_round1.png")
    crops(pos, "Fracture windows (IoU >= 0.5 with a box), random sample", HERE / "r_fracture_windows.png")
    mask_check(T, HERE / "r_mask_check.png")
    print("\nfigures written")


if __name__ == "__main__":
    main()
