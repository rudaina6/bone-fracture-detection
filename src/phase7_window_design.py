"""Phase 7, design analysis: which sliding-window set to use. TRAINING BOXES ONLY; no image is loaded, nothing is fitted.

    python src/phase7_window_design.py          # -> results/tables/phase7_window_design.csv, phase7_reachability.csv,
                                                #    results/phase7/window_design.json, results/figures/phase7_window_design.png

Every training box is mapped to the 448 px working image with preprocess()'s own geometry (computed on an array of the
image's original shape; the geometry depends on the shape only). Windows are rectangles w x h at a base size s with
aspect ratios 1:1 (s x s), 1:2 (s x 2s, tall) and 2:1 (2s x s, wide), placed on the grid x = i * w/4, y = j * h/4
(stride = window/4, i.e. one 8 px HOG cell once the window is resampled to the canonical 32 x 32 patch), fully inside
the 448 x 448 image, and kept only if at least half of the window lies in the content rectangle (not the black
padding). That filter cannot remove a useful window: IoU >= 0.5 with a box inside the content means at least half of
the window overlaps the box.

A box is REACHABLE if some window of the set has IoU >= 0.5 with it; the share of reachable boxes is the upper bound on
recall at IoU 0.5. Selection rule (Phase 7 protocol, docs/phase7_localization.md): among subsets of the candidate base
sizes, the smallest number of sizes with >= 90% of training boxes reachable; among those, the fewest windows per
image (mean over the training images).
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import REGIONS
from expt_log import log_run
from paths import REPO_ROOT
from preprocess import preprocess
from phase5_viz import C, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

SIZE, MID = 448, 448
CANDIDATES = (12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96)     # base size s in px at 448
ASPECTS = ((1, 1), (1, 2), (2, 1))                                 # (w, h) multiples of s
REACH_IOU, TARGET, CONTENT_MIN = 0.5, 0.90, 0.5


def window_types(sizes):
    return [(s * a, s * b) for s in sizes for a, b in ASPECTS]


def grid(w, h):
    """Top-left corners of the w x h windows on the stride-(w/4, h/4) grid inside the SIZE x SIZE image."""
    sx, sy = w / 4, h / 4
    return np.arange(int((SIZE - w) // sx) + 1) * sx, np.arange(int((SIZE - h) // sy) + 1) * sy


def geometry(h0, w0, cache={}):
    """preprocess()'s geometry for an image of original shape (h0, w0), at the Phase 7 working size."""
    if (h0, w0) not in cache:
        cache[(h0, w0)] = preprocess(np.zeros((h0, w0), np.uint8), size=SIZE, mid=MID, norm="none")[1]
    return cache[(h0, w0)]


def content_ok(xs, ys, w, h, content):
    """Mask (len(ys), len(xs)) of windows with at least CONTENT_MIN of their area inside the content rectangle."""
    x0, y0, x1, y1 = content
    fx = np.clip(np.minimum(xs + w, x1) - np.maximum(xs, x0), 0, None) / w
    fy = np.clip(np.minimum(ys + h, y1) - np.maximum(ys, y0), 0, None) / h
    return fy[:, None] * fx[None, :] >= CONTENT_MIN


def box_stats(b, w, h, content):
    """Best IoU of box b = (x, y, bw, bh) with any kept w x h window, and the numbers of windows at IoU >= 0.5 / 0.3."""
    xs, ys = grid(w, h)
    bx, by, bw, bh = b
    ix = np.clip(np.minimum(bx + bw, xs + w) - np.maximum(bx, xs), 0, None)
    iy = np.clip(np.minimum(by + bh, ys + h) - np.maximum(by, ys), 0, None)
    inter = iy[:, None] * ix[None, :]
    iou = inter / (bw * bh + w * h - inter)
    iou[~content_ok(xs, ys, w, h, content)] = 0.0
    return float(iou.max()), int((iou >= 0.5).sum()), int((iou >= 0.3).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase7")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args(argv)

    m = pd.read_csv(REPO_ROOT / "data" / "master.csv").merge(
        pd.read_csv(REPO_ROOT / "results" / "splits.csv")[["image_id", "split", "fold"]], on="image_id",
        validate="one_to_one")
    tr = m[m.split == "train"].reset_index(drop=True)            # test rows are never used here
    boxes = []
    for r in tr.itertuples():
        g = geometry(int(r.height), int(r.width))
        for x, y, w, h in json.loads(r.boxes_xywh):
            boxes.append(dict(image_id=r.image_id, region=r.region_group, fold=r.fold, content=g["content"],
                              b=(x * g["scale"] + g["off_x"], y * g["scale"] + g["off_y"], w * g["scale"],
                                 h * g["scale"])))
    print(f"{len(tr)} training images, {len(boxes)} training boxes (448 px working image)", flush=True)
    types = window_types(CANDIDATES)
    best = np.zeros((len(boxes), len(types)))
    n50 = np.zeros_like(best, dtype=int)
    n30 = np.zeros_like(best, dtype=int)
    for i, bx in enumerate(boxes):
        for t, (w, h) in enumerate(types):
            best[i, t], n50[i, t], n30[i, t] = box_stats(bx["b"], w, h, bx["content"])
    # mean number of kept windows per training image, per window type
    shapes = tr.groupby(["height", "width"]).size()
    per_type = np.zeros(len(types))
    for (h0, w0), n in shapes.items():
        g = geometry(int(h0), int(w0))
        for t, (w, h) in enumerate(types):
            xs, ys = grid(w, h)
            per_type[t] += n * content_ok(xs, ys, w, h, g["content"]).sum()
    per_type /= shapes.sum()

    region = np.array([b["region"] for b in boxes])
    tix = {wh: t for t, wh in enumerate(types)}
    rows, chosen = [], None
    for k in range(1, len(CANDIDATES) + 1):
        found = []
        for sub in combinations(CANDIDATES, k):
            cols = [tix[wh] for wh in window_types(sub)]
            reach = best[:, cols].max(1) >= REACH_IOU
            found.append(dict(n_sizes=k, sizes=" ".join(map(str, sub)), reach=reach.mean(),
                              windows_per_image=per_type[cols].sum(),
                              **{f"reach_{r}": reach[region == r].mean() for r in REGIONS}))
        rows += found
        ok = [f for f in found if f["reach"] >= TARGET]
        if ok and chosen is None:
            chosen = min(ok, key=lambda f: f["windows_per_image"])
        if chosen is not None and k >= chosen["n_sizes"] + 1:
            break
    design = pd.DataFrame(rows).sort_values(["n_sizes", "windows_per_image"])
    sizes = [int(s) for s in chosen["sizes"].split()]
    cols = [tix[wh] for wh in window_types(sizes)]
    reach50, reach30 = best[:, cols].max(1) >= 0.5, best[:, cols].max(1) >= 0.3
    pos = n50[:, cols].sum(1)
    rt = []
    for g in ["all"] + REGIONS + ["other"]:
        s = np.ones(len(boxes), bool) if g == "all" else region == g
        rt.append(dict(group=g, n_boxes=int(s.sum()), reach_iou50=reach50[s].mean(), reach_iou30=reach30[s].mean(),
                       pos_windows_total=int(pos[s].sum()), pos_windows_per_box_median=float(np.median(pos[s]))))
    reach_tab = pd.DataFrame(rt)
    # cheapest set reaching the target regardless of the number of sizes (for the record)
    ok_all = design[design.reach >= TARGET]
    cheapest = ok_all.loc[ok_all.windows_per_image.idxmin()] if len(ok_all) else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    design.round(4).to_csv(args.tables_dir / "phase7_window_design.csv", index=False)
    reach_tab.round(4).to_csv(args.tables_dir / "phase7_reachability.csv", index=False)
    out = dict(rule="fewest base sizes with >= 90% of training boxes reachable at IoU 0.5 (stride window/4, "
                    "content filter), then fewest windows per image",
               candidates=list(CANDIDATES), aspects=[list(a) for a in ASPECTS], size=SIZE, mid=MID,
               chosen_sizes=sizes, window_types=[list(wh) for wh in window_types(sizes)],
               windows_per_image=float(chosen["windows_per_image"]),
               windows_per_image_by_type={f"{w}x{h}": float(per_type[tix[(w, h)]]) for w, h in window_types(sizes)},
               reach_iou50=float(reach50.mean()), reach_iou30=float(reach30.mean()),
               reach_by_region={r["group"]: float(r["reach_iou50"]) for r in rt},
               positives_total=int(pos.sum()), n_boxes=len(boxes), n_train_images=int(len(tr)),
               cheapest_any_size_count=None if cheapest is None else dict(
                   sizes=cheapest["sizes"], reach=float(cheapest["reach"]),
                   windows_per_image=float(cheapest["windows_per_image"])),
               date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "window_design.json").write_text(json.dumps(out, indent=1))

    # figure: reachability against windows per image for every evaluated subset
    setup()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, col in zip(sorted(design.n_sizes.unique()), (C["muted"], C["unified"], C["region"], C["third"])):
        d = design[design.n_sizes == k]
        ax.scatter(d.windows_per_image, d.reach, s=10, color=col, alpha=0.6, label=f"{k} base size(s)")
    ax.axhline(TARGET, color=C["ink2"], ls=(0, (3, 2)), lw=1)
    ax.scatter([chosen["windows_per_image"]], [chosen["reach"]], s=70, facecolor="none", edgecolor=C["ink"], lw=1.5,
               label=f"chosen: {chosen['sizes']} px")
    ax.set_xscale("log")
    ax.set_xlabel("windows per image (mean over training images, log scale)")
    ax.set_ylabel("training boxes reachable (IoU >= 0.5)")
    header(fig, "Phase 7 window design: reachability against cost", ax, ncols=2, top=0.78, legend_y=0.92)
    save(fig, args.figures_dir / "phase7_window_design.png",
         "Each point is a set of base sizes s, each with windows s x s, s x 2s and 2s x s at stride window/4 on the "
         "448 px image. Training boxes only. Reachable = some window has IoU >= 0.5 (the upper bound on recall).")

    pd.set_option("display.width", 200)
    print(f"\nChosen base sizes {sizes} px -> window types {window_types(sizes)}")
    print(f"windows per image {chosen['windows_per_image']:.0f}; reachability IoU 0.5 {reach50.mean():.1%}, "
          f"IoU 0.3 {reach30.mean():.1%}; positive windows {pos.sum()}")
    print(reach_tab.round(3).to_string(index=False))
    print("\nBest subsets per number of sizes (top 3 by windows among those reaching the target):")
    for k, d in design.groupby("n_sizes"):
        d = d[d.reach >= TARGET].nsmallest(3, "windows_per_image")
        if len(d):
            print(d[["sizes", "reach", "windows_per_image"] + [f"reach_{r}" for r in REGIONS]].round(3).to_string(index=False))
        else:
            print(f"  {k} size(s): none reaches {TARGET:.0%} (best {design[design.n_sizes == k].reach.max():.1%})")
    if cheapest is not None:
        print(f"\nCheapest set reaching {TARGET:.0%} with any number of sizes: {cheapest['sizes']} "
              f"({cheapest['windows_per_image']:.0f} windows, reach {cheapest['reach']:.1%})")
    if not args.no_log:
        log_run(regions="all (pooled), training boxes", feature_set="n/a (box geometry only)",
                model="n/a (window design analysis, nothing fitted)", imbalance_strategy="n/a",
                key_hyperparameters=f"448 px; base sizes {sizes}; aspects 1:1 1:2 2:1; stride window/4; content>=50%",
                cv_score=f"reachability IoU0.5={reach50.mean():.3f}; windows/image={chosen['windows_per_image']:.0f}",
                test_metrics="not used (training boxes only)",
                notes="Phase 7 design analysis (before any model): window set chosen by the protocol rule")


if __name__ == "__main__":
    main()
