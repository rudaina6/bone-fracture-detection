"""Phase 5.3: error analysis - what do the two approaches get wrong, and what do those images look like?

    python src/phase5_errors.py                  # tables + figures
    options: --n-boot 2000  --seed 0  --no-contrast

NOTHING IS FITTED. Reads the saved out-of-fold and test scores and the metadata proxies in data/master.csv.

PRIMARY ON THE OUT-OF-FOLD CV SCORES, secondary on test. The test set has 141 fractured images in the four
modelled regions; the training out-of-fold scores cover 569 with the same "never seen by the model that
scored it" property. An error analysis split four ways over 141 positives cannot say anything, so the CV
scores carry it and the test numbers are shown beside them as a consistency check.

FracAtlas has no fracture-type and no view label, so "subtle" is measured by how large the fracture is in the
image the model actually sees:

    box_px_256 = sqrt(box width * box height) / max(image width, image height) * 256

- the geometric mean side of the fracture box, in pixels of the 256 x 256 CLAHE input the features are
  computed on. The HOG cell is 16 px, so this number says directly how many HOG cells a fracture spans.
Low contrast is measured by src/phase5_box_contrast.py (optional, reads the images); if its output is not
there, the contrast columns are skipped and everything else still runs.

"Joint-adjacent" from the plan is NOT TESTABLE and is not approximated: FracAtlas gives no joint annotation
and no view label, and a bounding box cannot be turned into "near a joint" without one.

Decision rule 4 (docs/phase5_analysis.md): no proxy is promoted to a cause. The proxies are correlated with
each other and with region, and a proxy whose interval excludes the null is reported as "associated with
being missed", never as the reason the model fails.

Outputs: results/tables/phase5_error_proxies.csv    per region x proxy x model: the association, with a CI
         results/tables/phase5_proxy_by_region.csv  what the proxies look like per region, all images
         results/phase5/error_cases.csv             per-image: both models' decisions and every proxy
         results/figures/phase5_error_agreement.png, phase5_error_proxies.png, phase5_recall_vs_size.png
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import REGIONS
from expt_log import log_run
from paths import REPO_ROOT
from phase4_regional import _fast_auc
from phase5_main import load_cv, load_test
from phase5_viz import C, REGION_LABEL, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

# proxy -> (label, is_binary). Continuous proxies are judged by a rank AUC, binary ones by a rate difference.
PROXIES = {
    "box_px_256": ("Fracture size (px of the 256 input)", False),
    "box_area_frac": ("Fracture area / image area", False),
    "fracture_count": ("Number of fractures annotated", False),
    "image_px": ("Image size (width x height)", False),
    "aspect": ("Image aspect ratio", False),
    "hardware": ("Surgical hardware present", True),
    "multiscan": ("Multi-scan (near-duplicate views)", True),
}
NEG_PROXIES = ["image_px", "aspect", "hardware", "multiscan"]      # defined for non-fractured images too


def box_stats(row):
    """Largest box of the image, as a fraction of area and as a side length in the 256 px model input."""
    boxes = ast.literal_eval(row.boxes_xywh) if isinstance(row.boxes_xywh, str) else []
    if not boxes:
        return pd.Series(dict(box_area_frac=np.nan, box_px_256=np.nan))
    areas = [w * h for _, _, w, h in boxes]
    i = int(np.argmax(areas))
    w, h = boxes[i][2], boxes[i][3]
    return pd.Series(dict(box_area_frac=areas[i] / (row.width * row.height),
                          box_px_256=np.sqrt(w * h) / max(row.width, row.height) * 256))


def build_proxies(master_path, contrast_path=None):
    m = pd.read_csv(master_path)
    m = pd.concat([m, m.apply(box_stats, axis=1)], axis=1)
    m["image_px"] = m.width * m.height
    m["aspect"] = m.width / m.height
    cols = ["image_id", "region_group", "fractured", "hardware", "multiscan", "fracture_count", "n_boxes",
            "width", "height", "box_area_frac", "box_px_256", "image_px", "aspect"]
    m = m[cols]
    if contrast_path and Path(contrast_path).exists():
        cs = pd.read_csv(contrast_path)
        keep = [c for c in cs.columns if c != "image_id"]
        m = m.merge(cs, on="image_id", how="left")
        # box_px_256_geom is the same quantity as box_px_256 computed the long way round; it is kept as a
        # cross-check column but must not become a second, duplicate proxy.
        for c in [k for k in keep if k != "box_px_256_geom"]:
            PROXIES.setdefault(c, (c.replace("_", " ").capitalize(), False))
        print(f"    box contrast joined from {Path(contrast_path).name}: {', '.join(keep)}", flush=True)
    else:
        print("    [note] no results/tables/phase5_box_stats.csv - contrast proxies skipped "
              "(run src/phase5_box_contrast.py to add them)", flush=True)
    return m


def cases(cv, te, prox):
    """One row per image per source: both models' decisions, the error label, and every proxy."""
    out = []
    for src, df, sa, ta, sb, tb in (("cv_oof", cv, "score", "thr_tuned", "score_u", "thr_tuned_u"),
                                    ("test", te, "score", "thr_tuned", "score_unified", "thr_unified")):
        g = df[["image_id", "y", "region"]].copy()
        g["source"] = src
        g["pred_region"] = (df[sa] >= df[ta]).to_numpy().astype(int)
        g["pred_unified"] = (df[sb] >= df[tb]).to_numpy().astype(int)
        for arm in ("region", "unified"):
            p = g[f"pred_{arm}"]
            g[f"err_{arm}"] = np.where(g.y == 1, np.where(p == 1, "TP", "FN"), np.where(p == 1, "FP", "TN"))
            g[f"correct_{arm}"] = (p == g.y).astype(int)
        g["agreement"] = np.select(
            [(g.correct_region == 1) & (g.correct_unified == 1),
             (g.correct_region == 1) & (g.correct_unified == 0),
             (g.correct_region == 0) & (g.correct_unified == 1)],
            ["both right", "only region-specific", "only unified"], default="both wrong")
        out.append(g)
    return pd.concat(out).merge(prox.drop(columns=["region_group", "fractured"]), on="image_id", how="left")


def assoc(detected, value, binary, n_boot, rng):
    """How strongly a proxy separates detected from missed images, with a percentile bootstrap CI.
    Continuous: rank AUC (0.5 = the proxy says nothing; >0.5 = larger value means more often detected).
    Binary: difference in detection rate (value==1 minus value==0); null is 0."""
    d, v = np.asarray(detected, int), np.asarray(value, float)
    ok = ~np.isnan(v)
    d, v = d[ok], v[ok]
    empty = dict(estimate=np.nan, lo=np.nan, hi=np.nan, n=len(d), n_missed=int((d == 0).sum()))
    if len(d) < 8 or d.min() == d.max():
        return empty
    if len(np.unique(v)) < 2:
        return empty          # a proxy with one level says nothing here (e.g. multiscan is 0% in hip)

    def stat(dd, vv):
        if binary:
            a, b = dd[vv == 1], dd[vv == 0]
            return (a.mean() - b.mean()) if len(a) and len(b) else np.nan
        return _fast_auc(dd, vv) if 0 < dd.sum() < len(dd) else np.nan

    est = stat(d, v)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        j = rng.integers(0, len(d), len(d))
        draws[b] = stat(d[j], v[j])
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return dict(estimate=est, lo=lo, hi=hi, n=len(d), n_missed=int((d == 0).sum()))


def proxy_table(cs, n_boot, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for src in cs.source.unique():
        for r in REGIONS + ["overall"]:
            g = cs[(cs.source == src) & ((cs.region == r) if r != "overall" else True)]
            for subset, is_pos in (("missed vs detected (fractured)", True),
                                   ("false alarm vs correct (not fractured)", False)):
                gg = g[g.y == (1 if is_pos else 0)]
                if len(gg) < 8:
                    continue
                for arm in ("region", "unified"):
                    # "detected" = the model got this image right (a hit among positives, a pass among negatives)
                    det = gg[f"correct_{arm}"].to_numpy()
                    for px, (lab, binary) in PROXIES.items():
                        if px not in gg.columns:
                            continue
                        if not is_pos and px not in NEG_PROXIES:
                            continue          # box proxies are undefined for non-fractured images
                        a = assoc(det, gg[px], binary, n_boot, rng)
                        rows.append(dict(source=src, region=r, subset=subset, model=arm, proxy=px,
                                         label=lab, binary=binary, **a))
    return pd.DataFrame(rows)


def proxy_by_region(prox):
    """What the proxies look like per region, over ALL images - the regions differ before any model sees them."""
    rows = []
    for r in REGIONS + ["other"]:
        g = prox[prox.region_group == r]
        if not len(g):
            continue
        f = g[g.fractured == 1]
        rows.append(dict(region=r, n=len(g), n_fractured=len(f), fracture_rate=g.fractured.mean(),
                         box_px_256_median=f.box_px_256.median(), box_px_256_q1=f.box_px_256.quantile(.25),
                         box_px_256_q3=f.box_px_256.quantile(.75),
                         box_area_frac_median=f.box_area_frac.median(),
                         multi_fracture_share=(f.fracture_count > 1).mean(),
                         hardware_share=g.hardware.mean(), multiscan_share=g.multiscan.mean(),
                         image_px_median=g.image_px.median()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def fig_agreement(cs, figdir):
    setup()
    order = ["both right", "only region-specific", "only unified", "both wrong"]
    cols = [C["third"], C["region"], C["unified"], "#b9b9b2"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (src, title) in zip(axes, [("cv_oof", "Cross-validation (out-of-fold, 569 fractured)"),
                                       ("test", "Test set (141 fractured)")]):
        g = cs[cs.source == src]
        data = {}
        for r in REGIONS:
            for status, mask in (("fractured", 1), ("not fractured", 0)):
                gg = g[(g.region == r) & (g.y == mask)]
                if len(gg):
                    v = gg.agreement.value_counts(normalize=True)
                    data[(r, status)] = [v.get(o, 0.0) for o in order]
        keys = [(r, s) for r in REGIONS for s in ("fractured", "not fractured")]
        y = np.arange(len(keys))
        left = np.zeros(len(keys))
        for oi, (o, col) in enumerate(zip(order, cols)):
            vals = np.array([data.get(k, [0] * 4)[oi] for k in keys])
            ax.barh(y, vals, left=left, color=col, label=o, height=0.72, edgecolor=C["surface"],
                    linewidth=1.2)
            left += vals
        ax.set_yticks(y, [f"{REGION_LABEL[r]} - {s}" for r, s in keys])
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("share of images")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    header(fig, "Where the two approaches agree and where they differ", axes[0], ncols=4, top=0.84,
           legend_y=0.945)
    save(fig, figdir / "phase5_error_agreement.png",
         "Each image is classified by both models at their own tuned thresholds. 'Only region-specific' and "
         "'only unified' are the discordant pairs McNemar counts. Fractured and not-fractured rows are shown "
         "separately because the two approaches trade recall against specificity rather than being uniformly "
         "better or worse.")


def fig_proxies(pt, figdir):
    setup()
    g = pt[(pt.source == "cv_oof") & (pt.subset.str.startswith("missed")) & (~pt.binary)]
    pxs = [p for p in PROXIES if p in set(g.proxy) and not PROXIES[p][1]]
    fig, axes = plt.subplots(1, len(pxs), figsize=(2.3 * len(pxs) + 0.6, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    ypos = np.arange(len(REGIONS) + 1)[::-1]
    rows = REGIONS + ["overall"]
    for ax, px in zip(axes, pxs):
        for arm, off, col in (("region", +0.16, C["region"]), ("unified", -0.16, C["unified"])):
            v = [g[(g.region == r) & (g.model == arm) & (g.proxy == px)] for r in rows]
            est = np.array([float(x.estimate.iloc[0]) if len(x) else np.nan for x in v])
            lo = np.array([float(x.lo.iloc[0]) if len(x) else np.nan for x in v])
            hi = np.array([float(x.hi.iloc[0]) if len(x) else np.nan for x in v])
            ax.errorbar(est, ypos + off, xerr=np.vstack([est - lo, hi - est]), fmt="o", color=col,
                        ecolor=col, elinewidth=1.5, capsize=2.5, markersize=5,
                        markeredgecolor=C["surface"], markeredgewidth=0.9,
                        label={"region": "Region-specific", "unified": "Unified"}[arm])
        ax.axvline(0.5, color=C["muted"], ls=(0, (3, 2)), lw=1.2)
        ax.set_title(PROXIES[px][0], fontsize=8.5)
        ax.set_xlabel("rank AUC")
        ax.set_xlim(0.1, 0.9)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [REGION_LABEL.get(r, "Overall") for r in rows])
    header(fig, "What separates a detected fracture from a missed one (out-of-fold)", axes[0], ncols=2,
           top=0.82, legend_y=0.94)
    save(fig, figdir / "phase5_error_proxies.png",
         "Rank AUC of each proxy for telling detected fractures from missed ones, with a 95% bootstrap CI. "
         "0.5 (dashed) means the proxy says nothing; above 0.5 means a larger value is detected more often. "
         "These proxies are correlated with each other and with region: an interval away from 0.5 marks an "
         "association with being missed, not a cause of the failure (decision rule 4).")


def fig_recall_vs_size(cs, figdir, n_bins=5):
    setup()
    g = cs[(cs.source == "cv_oof") & (cs.y == 1) & cs.box_px_256.notna()].copy()
    fig, axes = plt.subplots(1, len(REGIONS) + 1, figsize=(12.0, 3.2), sharey=True)
    for ax, r in zip(axes, REGIONS + ["overall"]):
        gg = g if r == "overall" else g[g.region == r]
        if len(gg) < n_bins * 3:
            ax.set_title(f"{REGION_LABEL.get(r, 'Overall')} - too few", fontsize=9.5)
            continue
        gg = gg.assign(bin=pd.qcut(gg.box_px_256, n_bins, duplicates="drop"))
        bins = [b for b in gg.bin.cat.categories if (gg.bin == b).any()]
        mids = [gg.loc[gg.bin == b, "box_px_256"].median() for b in bins]
        for arm, col, nm in (("region", C["region"], "Region-specific"),
                             ("unified", C["unified"], "Unified")):
            rec = [gg.loc[gg.bin == b, f"correct_{arm}"].mean() for b in bins]
            ax.plot(mids, rec, "-o", color=col, label=nm, markeredgecolor=C["surface"],
                    markeredgewidth=0.8)
        ax.set_title(f"{REGION_LABEL.get(r, 'Overall')} - {len(gg)} fractured", fontsize=9.5)
        ax.set_xlabel("fracture size (px of 256 input)")
        ax.set_ylim(0, 1.02)
        ax.axvline(16, color=C["muted"], ls=(0, (2, 2)), lw=1.1)
    axes[0].set_ylabel("recall")
    header(fig, "Recall against how large the fracture is in the image the model sees", axes[0], ncols=2,
           top=0.84, legend_y=0.945)
    save(fig, figdir / "phase5_recall_vs_size.png",
         "Out-of-fold recall within equal-count bins of fracture size; x is the geometric mean box side in "
         "pixels of the 256 x 256 CLAHE input, plotted at each bin's median. The dashed line is 16 px, one "
         "HOG cell: a fracture to the left of it is smaller than the cell the gradient histogram is pooled "
         "over, so it cannot have its own HOG bin and can only show up as a small perturbation of one.")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-contrast", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--master", type=Path, default=REPO_ROOT / "data" / "master.csv")
    ap.add_argument("--phase3-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    ap.add_argument("--phase4-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase5")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 5.3 error analysis (nothing fitted; saved scores + metadata proxies)", flush=True)
    prox = build_proxies(args.master,
                         None if args.no_contrast else args.tables_dir / "phase5_box_stats.csv")
    cv, te = load_cv(args.phase3_dir, args.phase4_dir), load_test(args.phase4_dir)
    cs = cases(cv, te, prox)
    cs.to_csv(args.out_dir / "error_cases.csv", index=False)

    pbr = proxy_by_region(prox)
    pbr.round(4).to_csv(args.tables_dir / "phase5_proxy_by_region.csv", index=False)
    print("\n[1/2] proxies per region (all images) -> phase5_proxy_by_region.csv", flush=True)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(pbr.round(3).to_string(index=False))

    print(f"\n[2/2] proxy associations ({args.n_boot} bootstrap resamples)...", flush=True)
    pt = proxy_table(cs, args.n_boot, args.seed)
    pt.round(4).to_csv(args.tables_dir / "phase5_error_proxies.csv", index=False)

    fig_agreement(cs, args.figures_dir)
    fig_proxies(pt, args.figures_dir)
    fig_recall_vs_size(cs, args.figures_dir)

    g = pt[(pt.source == "cv_oof") & (pt.subset.str.startswith("missed"))]
    print("\n=== Missed vs detected fractures, out-of-fold: association of each proxy (95% CI) ===")
    print("continuous proxies: rank AUC, null 0.5 | binary proxies: detection-rate difference, null 0")
    show = g.assign(text=[("n/a" if np.isnan(r.estimate) else f"{r.estimate:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
                           if r.binary else f"{r.estimate:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
                          for r in g.itertuples()])
    print(show.pivot_table(index=["proxy", "region"], columns="model", values="text",
                           aggfunc="first").to_string())
    print("\nDecision rule 4: a proxy whose interval excludes the null is ASSOCIATED with being missed; "
          "it is not\nthe cause. The proxies are correlated with each other and with region.")
    print('"Joint-adjacent" is not testable here: FracAtlas has no joint annotation and no view label.')
    if not args.no_log:
        log_run(regions="per region (hand, leg, hip, shoulder) + overall",
                feature_set="n/a (metadata proxies: box size, fracture count, hardware, multiscan, image size)",
                model="RBF SVM C=10 gamma=0.05 (both approaches, saved scores)",
                imbalance_strategy="none + F2 threshold (as chosen in Phases 3-4)",
                key_hyperparameters=f"Phase 5.3 error analysis; bootstrap n_boot={args.n_boot}; "
                                    f"seed={args.seed}; primary on out-of-fold scores",
                cv_score="see results/tables/phase5_error_proxies.csv",
                test_metrics="re-read from results/phase4/test_scores.csv (no new test evaluation)",
                notes="Phase 5.3: nothing fitted; proxies are associations, never causes (decision rule 4)")
    print(f"\nWrote 2 tables to {args.tables_dir}, per-image cases to {args.out_dir / 'error_cases.csv'}, "
          f"3 figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
