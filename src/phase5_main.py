"""Phase 5.1: the main comparison, unified vs region-specific, consolidated from the SAVED scores.

    python src/phase5_main.py                 # tables + figures (+ reproduction check against Phase 4)
    python src/phase5_main.py --no-check      # skip the Phase 4 reproduction check
    options: --n-boot 2000  --seed 0

NOTHING IS FITTED HERE and no model, threshold or verdict is chosen. The script reads the per-image scores
Phases 3 and 4 already wrote and turns them into the Phase 5 tables and figures:

  CV   results/phase3/oof_rbfsvm__none.csv   (unified, outer out-of-fold)
       results/phase4/oof_rbfsvm__none.csv   (region-specific, SAME images and folds)
  test results/phase4/test_scores.csv        (both models' per-image test scores and thresholds)

The test-set locks (results/phase3|phase4/TEST_EVALUATED.json) are not touched; this is a re-reading of a
single evaluation that already happened, not a second one.

Protocol and decision rules: docs/phase5_analysis.md (written before any Phase 5 result).

Outputs
  results/tables/phase5_main_cv.csv        per region: both models' CV metrics and the paired difference
  results/tables/phase5_main_test.csv      per region: both models' test metrics and the paired difference
  results/tables/phase5_mcnemar.csv        McNemar on the test decisions (and on the out-of-fold decisions)
  results/tables/phase5_agreement.csv      per region: both right / only one right / both wrong
  results/figures/phase5_main_pr_auc.png   per-region PR-AUC, both models, CV and test
  results/figures/phase5_main_forest.png   paired differences with 95% CIs (the key figure)
  results/figures/phase5_test_pr_curves.png
  results/figures/phase5_score_dists.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_curve

from evaluate import REGIONS, all_metrics, bootstrap_ci, paired_bootstrap
from expt_log import log_run
from phase4_regional import fold_paired_bootstrap, _fast_ap
from paths import REPO_ROOT
from phase5_viz import C, REGION_LABEL, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

MODEL, IMB = "rbfsvm", "none"
PAIRED_METRICS = ("pr_auc", "roc_auc", "recall", "precision", "specificity", "f2")
THR_METRICS = ("recall", "precision", "specificity", "f2")


# ----------------------------------------------------------------------------- loading

def load_cv(phase3_dir, phase4_dir):
    """Region-specific and unified out-of-fold scores on the same images, as one frame."""
    reg = pd.read_csv(phase4_dir / f"oof_{MODEL}__{IMB}.csv")
    uni = pd.read_csv(phase3_dir / f"oof_{MODEL}__{IMB}.csv")
    j = reg.merge(uni, on="image_id", suffixes=("", "_u"), validate="one_to_one")
    assert len(j) == len(reg), "unified out-of-fold file is missing images that the region file has"
    assert (j.y == j.y_u).all() and (j.fold == j.fold_u).all() and (j.region == j.region_u).all(), \
        "unified and region out-of-fold files disagree on labels, folds or regions"
    return j[j.region.isin(REGIONS)].reset_index(drop=True)


def load_test(phase4_dir):
    t = pd.read_csv(phase4_dir / "test_scores.csv")
    need = {"image_id", "y", "region", "score", "thr_tuned", "score_unified", "thr_unified"}
    missing = need - set(t.columns)
    assert not missing, f"results/phase4/test_scores.csv is missing {sorted(missing)}"
    return t[t.region.isin(REGIONS)].reset_index(drop=True)


# ----------------------------------------------------------------------------- tables

def cv_table(cv, n_boot, seed):
    """Per region: fold-mean PR-AUC of both models, pooled threshold metrics, and the paired difference."""
    rows = []
    for r in REGIONS:
        g = cv[cv.region == r]
        folds = sorted(g.fold.unique())
        pr_r = np.array([_fast_ap(g[g.fold == k].y.to_numpy(), g[g.fold == k].score.to_numpy()) for k in folds])
        pr_u = np.array([_fast_ap(g[g.fold == k].y.to_numpy(), g[g.fold == k].score_u.to_numpy()) for k in folds])
        m_r = all_metrics(g.y, g.score, g.thr_tuned.to_numpy())
        m_u = all_metrics(g.y, g.score_u, g.thr_tuned_u.to_numpy())
        row = dict(region=r, n=len(g), n_pos=int(g.y.sum()), chance=g.y.mean(),
                   reg_pr_auc_fold=pr_r.mean(), reg_pr_auc_fold_sd=pr_r.std(ddof=1),
                   uni_pr_auc_fold=pr_u.mean(), uni_pr_auc_fold_sd=pr_u.std(ddof=1),
                   folds_region_better=int((pr_r > pr_u).sum()),
                   **{f"reg_{k}": m_r[k] for k in THR_METRICS},
                   **{f"uni_{k}": m_u[k] for k in THR_METRICS})
        # paired differences: ranking metrics fold-mean (images resampled within folds), threshold metrics pooled
        fb = fold_paired_bootstrap(g.y, g.score, g.score_u, g.fold, n_boot=n_boot, seed=seed) \
            if g.y.sum() >= 10 else {}
        for m, (d, lo, hi, p) in fb.items():
            row.update({f"d_{m}": d, f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"d_{m}_p": p})
        rows.append(row)
    pb = paired_bootstrap(cv.y, cv.score, cv.score_u, cv.thr_tuned.to_numpy(), cv.thr_tuned_u.to_numpy(),
                          groups=cv.region, n_boot=n_boot, seed=seed, metrics=THR_METRICS)
    out = pd.DataFrame(rows)
    for r in REGIONS:
        for m in THR_METRICS:
            s = pb[(pb.group == r) & (pb.metric == m)]
            if len(s):
                s = s.iloc[0]
                out.loc[out.region == r, [f"d_{m}", f"d_{m}_lo", f"d_{m}_hi", f"d_{m}_p"]] = \
                    [s["diff"], s.lo, s.hi, s.p_two_sided]
    return out


def test_table(te, n_boot, seed):
    """Per region: both models' test metrics with bootstrap CIs, and the paired difference."""
    rows = []
    for r in REGIONS:
        g = te[te.region == r]
        m_r = all_metrics(g.y, g.score, g.thr_tuned.to_numpy())
        m_u = all_metrics(g.y, g.score_unified, g.thr_unified.to_numpy())
        rows.append(dict(region=r, n=len(g), n_pos=int(g.y.sum()), chance=g.y.mean(),
                         **{f"reg_{k}": m_r[k] for k in ("pr_auc", "roc_auc") + THR_METRICS},
                         **{f"uni_{k}": m_u[k] for k in ("pr_auc", "roc_auc") + THR_METRICS}))
    out = pd.DataFrame(rows)
    # per-model CIs (for the error bars) and the paired difference
    for arm, sc, th in (("reg", "score", "thr_tuned"), ("uni", "score_unified", "thr_unified")):
        ci = bootstrap_ci(te.y, te[sc], te[th].to_numpy(), groups=te.region, n_boot=n_boot, seed=seed,
                          metrics=("pr_auc", "f2"))
        for r in REGIONS:
            for m in ("pr_auc", "f2"):
                s = ci[(ci.group == r) & (ci.metric == m)]
                if len(s):
                    out.loc[out.region == r, [f"{arm}_{m}_lo", f"{arm}_{m}_hi"]] = \
                        [s.iloc[0].lo, s.iloc[0].hi]
    pb = paired_bootstrap(te.y, te.score, te.score_unified, te.thr_tuned.to_numpy(),
                          te.thr_unified.to_numpy(), groups=te.region, n_boot=n_boot, seed=seed,
                          metrics=PAIRED_METRICS)
    for r in REGIONS:
        for m in PAIRED_METRICS:
            s = pb[(pb.group == r) & (pb.metric == m)]
            if len(s):
                s = s.iloc[0]
                out.loc[out.region == r, [f"d_{m}", f"d_{m}_lo", f"d_{m}_hi", f"d_{m}_p"]] = \
                    [s["diff"], s.lo, s.hi, s.p_two_sided]
    return out


def mcnemar(y, pred_a, pred_b):
    """Exact (binomial) two-sided McNemar on the discordant pairs. A = region-specific, B = unified.
    b = A right & B wrong, c = A wrong & B right. Returns the counts and p."""
    y, pred_a, pred_b = np.asarray(y).astype(int), np.asarray(pred_a).astype(int), np.asarray(pred_b).astype(int)
    ra, rb = pred_a == y, pred_b == y
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return dict(both_right=int((ra & rb).sum()), only_region_right=b, only_unified_right=c,
                both_wrong=int((~ra & ~rb).sum()), n_discordant=b + c, p_exact=p)


def agreement_tables(cv, te):
    """McNemar and the agreement counts, per region, on test (primary for 5.1) and out-of-fold."""
    mc, ag = [], []
    for src, df, sa, ta, sb, tb in (("test", te, "score", "thr_tuned", "score_unified", "thr_unified"),
                                    ("cv_oof", cv, "score", "thr_tuned", "score_u", "thr_tuned_u")):
        for r in REGIONS:
            g = df[df.region == r]
            pa, pb_ = g[sa] >= g[ta], g[sb] >= g[tb]
            d = mcnemar(g.y, pa, pb_)
            mc.append(dict(source=src, region=r, n=len(g), n_pos=int(g.y.sum()), **d))
            for lab, mask in (("fractured", g.y == 1), ("not fractured", g.y == 0)):
                gg, aa, bb = g[mask], pa[mask], pb_[mask]
                dd = mcnemar(gg.y, aa, bb)
                ag.append(dict(source=src, region=r, subset=lab, n=len(gg), **dd))
    return pd.DataFrame(mc), pd.DataFrame(ag)


# ----------------------------------------------------------------------------- figures

def fig_pr_auc(cvt, tet, figdir, n_boot=2000):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    x = np.arange(len(REGIONS))
    w = 0.36

    def errs(src, key, sd_col=None):
        """Symmetric fold SD (cross-validation) or asymmetric bootstrap CI (test)."""
        if sd_col:
            return src[sd_col].to_numpy()
        v = src[f"{key}_pr_auc"]
        return np.vstack([(v - src[f"{key}_pr_auc_lo"]).to_numpy(), (src[f"{key}_pr_auc_hi"] - v).to_numpy()])

    panels = [(axes[0], "Cross-validation (training images, out-of-fold)", cvt,
               ("reg_pr_auc_fold", "reg_pr_auc_fold_sd"), ("uni_pr_auc_fold", "uni_pr_auc_fold_sd")),
              (axes[1], "Test set (used once, Phase 4)", tet,
               ("reg_pr_auc", None), ("uni_pr_auc", None))]
    for ax, title, src, (rc, rsd), (uc, usd) in panels:
        for off, vcol, sd, key, lab, col in ((-w / 2, rc, rsd, "reg", "Region-specific", C["region"]),
                                             (+w / 2, uc, usd, "uni", "Unified", C["unified"])):
            vals = src[vcol].to_numpy()
            ax.bar(x + off, vals, w, label=lab, color=col, edgecolor=C["surface"], linewidth=1.2)
            ax.errorbar(x + off, vals, yerr=errs(src, key, sd), fmt="none", ecolor=C["ink2"],
                        elinewidth=1.0, capsize=2.5)
            for xi, v in zip(x + off, vals):
                ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=C["surface"])
        for xi, ch in zip(x, src.chance.to_numpy()):          # chance = that region's fracture rate
            ax.plot([xi - 0.46, xi + 0.46], [ch, ch], ls=(0, (3, 2)), color=C["muted"], lw=1.2,
                    zorder=3, solid_capstyle="butt")
        ax.set_xticks(x, [f"{REGION_LABEL[r]}\n{int(n)} pos." for r, n in zip(REGIONS, src.n_pos)])
        ax.set_title(title)
        ax.set_ylim(0, 0.95)
    axes[0].set_ylabel("PR-AUC")
    axes[0].plot([], [], ls=(0, (3, 2)), color=C["muted"], lw=1.2, label="chance (region fracture rate)")
    header(fig, "Region-specific vs unified RBF SVM, per region", axes[0], ncols=3)
    save(fig, figdir / "phase5_main_pr_auc.png",
         f"Error bars: cross-validation = SD over the 5 outer folds; test = 95% percentile bootstrap CI "
         f"({n_boot:,} resamples, threshold fixed). PR-AUC chance level differs by region, so bars are "
         f"comparable between the two models within a region, not across regions.")


def fig_forest(cvt, tet, figdir):
    setup()
    metrics = [("pr_auc", "PR-AUC"), ("f2", "F2 at the tuned threshold")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    ypos = np.arange(len(REGIONS))[::-1]
    for ax, (m, lab) in zip(axes, metrics):
        for src, off, col, nm in ((cvt, +0.17, C["region"], "Cross-validation"),
                                  (tet, -0.17, C["unified"], "Test set")):
            d = src[f"d_{m}"].to_numpy()
            lo, hi = src[f"d_{m}_lo"].to_numpy(), src[f"d_{m}_hi"].to_numpy()
            ax.errorbar(d, ypos + off, xerr=np.vstack([d - lo, hi - d]), fmt="o", color=col,
                        ecolor=col, elinewidth=1.6, capsize=3, markersize=5.5,
                        markeredgecolor=C["surface"], markeredgewidth=1.0, label=nm)
        ax.axvline(0, color=C["muted"], lw=1.2, ls=(0, (3, 2)), zorder=1)
        ax.set_title(lab)
        ax.set_xlabel("region-specific minus unified")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [REGION_LABEL[r] for r in REGIONS])
    header(fig, "Paired difference, region-specific minus unified (95% CI)", axes[0], ncols=2, top=0.84)
    save(fig, figdir / "phase5_main_forest.png",
         "Positive = the region-specific model is better. Same images in both arms. Cross-validation PR-AUC is "
         "the mean of the 5 per-fold differences (images resampled within folds); F2 is pooled out-of-fold at "
         "each fold's tuned threshold. A CI that includes 0 means no detectable difference (Phase 4 rule 2) - "
         "for hip and shoulder that is expected from 11-12 test positives alone and is not evidence of equality.")


def fig_pr_curves(te, figdir):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=True)
    for ax, r in zip(axes, REGIONS):
        g = te[te.region == r]
        for sc, lab, col in (("score", "Region-specific", C["region"]),
                             ("score_unified", "Unified", C["unified"])):
            p, rc, _ = precision_recall_curve(g.y, g[sc])
            ax.step(rc, p, where="post", color=col, lw=1.8,
                    label=f"{lab} (AP {_fast_ap(g.y.to_numpy(), g[sc].to_numpy()):.2f})")
        ax.axhline(g.y.mean(), color=C["muted"], ls=(0, (3, 2)), lw=1.2)
        ax.set_title(f"{REGION_LABEL[r]} - {int(g.y.sum())} fractured of {len(g)}")
        ax.set_xlabel("Recall")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1.02)
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylabel("Precision")
    header(fig, "Test-set precision-recall curves per region", top=0.86)
    save(fig, figdir / "phase5_test_pr_curves.png",
         "Dashed line: chance (the region's test fracture rate). Threshold-free, so these curves do not depend "
         "on the F2 thresholds. Hip and shoulder curves are step functions over 11-12 positives and are shown "
         "for completeness, not as reliable shapes.")


def fig_score_dists(cv, figdir):
    setup()
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.0), sharex="col")
    for col, r in enumerate(REGIONS):
        g = cv[cv.region == r]
        both = np.concatenate([g.score.to_numpy(), g.score_u.to_numpy()])
        bins = np.linspace(both.min(), both.max(), 34)        # shared per column (the x axis is shared)
        for row, (sc, th, nm) in enumerate((("score", "thr_tuned", "Region-specific"),
                                            ("score_u", "thr_tuned_u", "Unified"))):
            ax = axes[row, col]
            ax.hist(g.loc[g.y == 0, sc], bins=bins, color=C["neg"], alpha=0.75, label="not fractured")
            ax.hist(g.loc[g.y == 1, sc], bins=bins, color=C["pos"], alpha=0.80, label="fractured")
            for t in sorted(g[th].unique()):
                ax.axvline(t, color=C["ink"], lw=1.0, ls=(0, (2, 2)))
            ax.set_title(f"{REGION_LABEL[r]} - {nm}" if row == 0 else f"{nm}")
            if col == 0:
                ax.set_ylabel("images")
            if row == 1:
                ax.set_xlabel("out-of-fold score (SVM margin)")
    axes[0, 0].plot([], [], color=C["ink"], lw=1.0, ls=(0, (2, 2)), label="tuned threshold (one per fold)")
    header(fig, "Out-of-fold score distributions by fracture status, with the tuned thresholds",
           axes[0, 0], ncols=3, top=0.90, legend_y=0.955)
    save(fig, figdir / "phase5_score_dists.png",
         "Dashed vertical lines: the F2 thresholds chosen inside the training folds (one per outer fold, so up "
         "to five per panel). Training images only, each scored by a model that did not see it.")


# ----------------------------------------------------------------------------- reproduction check

def check_against_phase4(cvt, tet, tables_dir, tol=5e-3):
    """Phase 5 recomputes the Phase 4 differences from the saved scores; they must match the Phase 4 tables.
    Bootstrap CIs are resampled, so the tolerance is loose for interval ends and tight for point estimates."""
    probs = []
    for name, mine, cols in (("phase4_cv_paired.csv", cvt, None), ("phase4_test_paired.csv", tet, None)):
        p = tables_dir / name
        if not p.exists():
            print(f"    [check] {name} not found - skipped")
            continue
        old = pd.read_csv(p)
        if "run" in old.columns:                      # the CV table holds several runs
            old = old[(old.run == f"{MODEL}__{IMB}")]
        for r in REGIONS:
            for m in PAIRED_METRICS:
                s = old[(old.group == r) & (old.metric == m)]
                if not len(s) or f"d_{m}" not in mine.columns:
                    continue
                a, b = float(s.iloc[0]["diff"]), float(mine.loc[mine.region == r, f"d_{m}"].iloc[0])
                if np.isnan(a) and np.isnan(b):
                    continue
                if np.isnan(a) != np.isnan(b) or abs(a - b) > tol:
                    probs.append(f"{name} {r}/{m}: Phase 4 {a:+.4f} vs Phase 5 {b:+.4f}")
    if probs:
        raise SystemExit("Phase 5 does not reproduce the Phase 4 differences:\n  " + "\n  ".join(probs))
    print("    [check] reproduces the Phase 4 paired differences (point estimates within "
          f"{tol}) for every region and metric", flush=True)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--phase3-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    ap.add_argument("--phase4-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase5")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args()

    cv, te = load_cv(args.phase3_dir, args.phase4_dir), load_test(args.phase4_dir)
    print(f"Phase 5.1: {len(cv)} training images ({int(cv.y.sum())} fractured) with out-of-fold scores from both "
          f"models;\n           {len(te)} test images ({int(te.y.sum())} fractured) with saved test scores. "
          "Nothing is fitted.", flush=True)

    print("\n[1/3] tables (paired bootstraps)...", flush=True)
    cvt = cv_table(cv, args.n_boot, args.seed)
    tet = test_table(te, args.n_boot, args.seed)
    mc, ag = agreement_tables(cv, te)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cvt.round(4).to_csv(args.tables_dir / "phase5_main_cv.csv", index=False)
    tet.round(4).to_csv(args.tables_dir / "phase5_main_test.csv", index=False)
    mc.to_csv(args.tables_dir / "phase5_mcnemar.csv", index=False)
    ag.to_csv(args.tables_dir / "phase5_agreement.csv", index=False)

    if not args.no_check:
        print("\n[2/3] reproduction check against the Phase 4 tables...", flush=True)
        check_against_phase4(cvt, tet, args.tables_dir)

    print("\n[3/3] figures...", flush=True)
    fig_pr_auc(cvt, tet, args.figures_dir, args.n_boot)
    fig_forest(cvt, tet, args.figures_dir)
    fig_pr_curves(te, args.figures_dir)
    fig_score_dists(cv, args.figures_dir)

    pd.set_option("display.width", 220, "display.max_columns", 50)

    def fmt(src, m):
        return [("n/a" if np.isnan(r[f"d_{m}"]) else
                 f"{r[f'd_{m}']:+.3f} [{r[f'd_{m}_lo']:+.3f}, {r[f'd_{m}_hi']:+.3f}]")
                for _, r in src.iterrows()]

    print("\n=== 5.1 Region-specific minus unified (positive = region-specific better) ===")
    summary = pd.DataFrame(dict(
        region=[REGION_LABEL[r] for r in cvt.region], train_pos=cvt.n_pos, test_pos=tet.n_pos,
        cv_pr_auc=fmt(cvt, "pr_auc"), test_pr_auc=fmt(tet, "pr_auc"),
        cv_f2=fmt(cvt, "f2"), test_f2=fmt(tet, "f2")))
    print(summary.to_string(index=False))
    print("\n=== McNemar on the test decisions at each model's tuned threshold (exact binomial) ===")
    show = mc[mc.source == "test"].assign(region=lambda d: [REGION_LABEL[r] for r in d.region])
    print(show[["region", "n", "n_pos", "both_right", "only_region_right", "only_unified_right",
                "both_wrong", "n_discordant", "p_exact"]].to_string(index=False))
    print("\nRule 1: Phase 5 changes no model, threshold or verdict. The Phase 4 verdicts stand.")
    if not args.no_log:
        log_run(regions="per region (hand, leg, hip, shoulder)", feature_set="n/a (saved scores)",
                model="RBF SVM C=10 gamma=0.05 (Phase 3 unified vs Phase 4 region-specific)",
                imbalance_strategy="none + F2 threshold (as chosen in Phases 3-4)",
                key_hyperparameters=f"Phase 5.1 consolidation; paired bootstrap n_boot={args.n_boot}; "
                                    f"seed={args.seed}; McNemar exact binomial",
                cv_score="see results/tables/phase5_main_cv.csv",
                test_metrics="re-read from results/phase4/test_scores.csv (no new test evaluation)",
                notes="Phase 5.1: nothing fitted; reproduces the Phase 4 paired differences before plotting")
    print(f"\nWrote 4 tables to {args.tables_dir} and 4 figures to {args.figures_dir}")
    (args.out_dir / "phase5_main.json").write_text(json.dumps(
        dict(n_boot=args.n_boot, seed=args.seed, model=MODEL, imbalance=IMB,
             n_cv=len(cv), n_cv_pos=int(cv.y.sum()), n_test=len(te), n_test_pos=int(te.y.sum()),
             date=pd.Timestamp.now().isoformat(timespec="seconds"), fitted="nothing (saved scores only)"),
        indent=1))


if __name__ == "__main__":
    main()
