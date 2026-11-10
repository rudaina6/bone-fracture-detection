"""Phase 5.2: learning curves and the cross-region transfer matrix. TRAINING IMAGES ONLY.

    python src/phase5_curves.py                      # curves + both transfer matrices (+ the Phase 4 check)
    python src/phase5_curves.py --only curves
    python src/phase5_curves.py --only transfer
    python src/phase5_curves.py --summarize          # rebuild tables and figures from the saved runs
    options: --n-jobs 4  --reps 5  --match-pos 30 --match-neg 140  --seed 42  --force  --no-log  --quick

The question: is the Phase 4 picture "the regions genuinely need different models" or "each region alone has
too little data"? Those have opposite implications for Phase 6, so the two are separated here.

LEARNING CURVES. For region r, outer fold k and size fraction p: draw a stratified subsample S of m = p * n
of the region's rows in folds != k, then
    region arm   fit on S only
    unified arm  fit on S + every training row of the OTHER regions in folds != k
and score both on region r's rows in fold k (PR-AUC). Both arms see the SAME subsample, so every point is
paired. p = 0 runs the unified arm alone: PR-AUC on region r from a model that has seen no image of region r.

TRANSFER MATRIX. Fit on region A's rows in folds != k, score region B's rows in fold k, for all ordered
pairs. The primary matrix is SIZE-MATCHED (every training region subsampled to the same counts) so that a
weak row means "that region's images do not transfer", not "that region has fewer images"; the raw matrix is
reported beside it.

Hyperparameters are FIXED at the Phase 3 values (C=10, gamma=0.05) in every arm and every cell: a learning
curve must vary training size and nothing else. Same caveat as the Phase 4 sensitivity run (those values were
chosen on all 3,266 training images, so both arms are slightly optimistic), but it applies equally to both
arms. PR-AUC is threshold-free, so no threshold is chosen anywhere in this script.

Protocol and decision rules: docs/phase5_analysis.md (written before any Phase 5 result).

Outputs: results/phase5/curves.csv, transfer_matched.csv, transfer_raw.csv, run_curves.json
         results/tables/phase5_learning_curves.csv, phase5_transfer.csv
         results/figures/phase5_learning_curves.png, phase5_transfer.png
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cv_protocol import load_train
from evaluate import REGIONS
from expt_log import log_run
from model_zoo import build, scores
from paths import REPO_ROOT
from phase4_regional import _fast_ap
from phase5_viz import C, REGION_LABEL, SEQ, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

FIXED = {"C": 10, "gamma": 0.05}          # the Phase 3 choice; never re-tuned in this script
MODEL, IMB = "rbfsvm", "none"
SIZES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
QUICK_SIZES = (0.0, 0.1, 0.35, 1.0)


# ----------------------------------------------------------------------------- fitting

def _fit_eval(X, y, tr_idx, eval_sets, col_weights, seed):
    """Fit on tr_idx, return PR-AUC on each (name, idx) in eval_sets. One fit, several evaluation sets."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build(MODEL, IMB, FIXED, col_weights, seed).fit(X[tr_idx], y[tr_idx])
        out = {}
        for name, idx in eval_sets:
            s = scores(pipe, X[idx])
            out[name] = _fast_ap(y[idx].astype(int), s) if 0 < y[idx].sum() < len(idx) else np.nan
        return out


def stratified_draw(y, idx, n_pos, n_neg, rng):
    """n_pos fractured and n_neg non-fractured image indices drawn from idx without replacement."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    if len(pos) < n_pos or len(neg) < n_neg:
        return None
    return np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])


# ----------------------------------------------------------------------------- learning curves

def curve_jobs(tr, sizes, reps, seed):
    """One job per (region, fold, p, rep, arm). Both arms of a cell share the same subsample."""
    y, fold, region = tr["y"], tr["fold"], tr["region"]
    jobs = []
    for r in REGIONS:
        in_r = region == r
        for k in sorted(set(fold)):
            avail = np.flatnonzero(in_r & (fold != k))
            others = np.flatnonzero(~in_r & (fold != k))
            ev = np.flatnonzero(in_r & (fold == k))
            n_pos, n_neg = int(y[avail].sum()), int(len(avail) - y[avail].sum())
            for p in sizes:
                n_rep = 1 if p in (0.0, 1.0) else reps
                for rep in range(n_rep):
                    rng = np.random.default_rng([seed, REGIONS.index(r), k, int(p * 1000), rep])
                    if p == 0.0:
                        S = np.empty(0, int)
                    elif p == 1.0:
                        S = avail
                    else:
                        S = stratified_draw(y, avail, max(1, round(p * n_pos)), max(1, round(p * n_neg)), rng)
                    skip = S is None or (len(S) and (y[S].sum() < 2 or (len(S) - y[S].sum()) < 2))
                    base = dict(region=r, fold=k, p=p, rep=rep, m=0 if S is None else len(S),
                                m_pos=0 if S is None else int(y[S].sum()))
                    if skip and p > 0:
                        jobs.append((dict(base, arm="region", skipped=True), None, ev))
                        jobs.append((dict(base, arm="unified", skipped=True), None, ev))
                        continue
                    if p > 0:
                        jobs.append((dict(base, arm="region", skipped=False), S, ev))
                    jobs.append((dict(base, arm="unified", skipped=False),
                                 np.concatenate([S, others]), ev))
    return jobs


def run_curves(tr, args):
    sizes = QUICK_SIZES if args.quick else SIZES
    jobs = curve_jobs(tr, sizes, args.reps, args.seed)
    todo = [(m, s, e) for m, s, e in jobs if not m["skipped"]]
    print(f"  learning curves: {len(todo)} fits ({len(jobs) - len(todo)} cells skipped for too few images)",
          flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_fit_eval)(tr["X"], tr["y"], s, [("pr_auc", e)], tr["col_weights"], args.seed)
        for m, s, e in todo)
    out = []
    it = iter(res)
    for m, s, e in jobs:
        out.append(dict(m, n_eval=len(e), n_eval_pos=int(tr["y"][e].sum()),
                        pr_auc=np.nan if m["skipped"] else next(it)["pr_auc"]))
    print(f"  learning curves done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- transfer matrix

def run_transfer(tr, args, matched):
    """Fit on region A's training rows (optionally size-matched), score every region B's fold-k rows."""
    y, fold, region = tr["y"], tr["fold"], tr["region"]
    sources = list(REGIONS) + ["all regions"]
    jobs = []
    for a in sources:
        in_a = np.ones(len(y), bool) if a == "all regions" else region == a
        for k in sorted(set(fold)):
            avail = np.flatnonzero(in_a & (fold != k))
            ev = [(b, np.flatnonzero((region == b) & (fold == k))) for b in REGIONS]
            n_rep = args.reps if matched else 1
            for rep in range(n_rep):
                if matched:
                    rng = np.random.default_rng([args.seed, sources.index(a), k, rep, 99])
                    S = stratified_draw(y, avail, args.match_pos, args.match_neg, rng)
                    if S is None:
                        raise SystemExit(
                            f"size-matched transfer: {a} fold!={k} has only {int(y[avail].sum())} fractured / "
                            f"{int(len(avail) - y[avail].sum())} non-fractured rows, cannot draw "
                            f"{args.match_pos}/{args.match_neg}. Lower --match-pos / --match-neg.")
                else:
                    S = avail
                jobs.append((dict(train=a, fold=k, rep=rep, m=len(S), m_pos=int(y[S].sum()),
                                  matched=matched), S, ev))
    kind = "size-matched" if matched else "raw"
    print(f"  transfer matrix ({kind}): {len(jobs)} fits", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_fit_eval)(tr["X"], tr["y"], s, ev, tr["col_weights"], args.seed) for _, s, ev in jobs)
    rows = []
    for (meta, _, _), pr in zip(jobs, res):
        for b, v in pr.items():
            rows.append(dict(meta, eval=b, pr_auc=v))
    print(f"  transfer ({kind}) done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- check

def check_vs_phase4_fixed(curves, phase4_dir, tol=1e-4):
    """The region arm at p = 1 is exactly the Phase 4 fixed-hyperparameter run (same rows, same pipeline,
    same seed), so their per-fold PR-AUCs must agree. A real end-to-end check of this script's fitting."""
    p = phase4_dir / f"oof_{MODEL}__{IMB}__fixed.csv"
    if not p.exists():
        print(f"    [check] {p.name} not found - skipped (run phase4_regional.py --fixed-params C=10 gamma=0.05)")
        return
    saved = pd.read_csv(p)
    mine = curves[(curves.arm == "region") & (curves.p == 1.0)]
    bad, worst = [], 0.0
    for r in REGIONS:
        for k in sorted(saved.fold.unique()):
            g = saved[(saved.region == r) & (saved.fold == k)]
            a = _fast_ap(g.y.to_numpy().astype(int), g.score.to_numpy())
            b = mine[(mine.region == r) & (mine.fold == k)].pr_auc
            if not len(b):
                continue
            worst = max(worst, abs(a - float(b.iloc[0])))
            if abs(a - float(b.iloc[0])) > tol:
                bad.append(f"{r} fold {k}: Phase 4 fixed {a:.6f} vs Phase 5 region arm {float(b.iloc[0]):.6f}")
    if bad:
        raise SystemExit("the region arm at p=1 does not reproduce the Phase 4 fixed-hyperparameter run:\n  "
                         + "\n  ".join(bad))
    print(f"    [check] region arm at p=1 reproduces results/phase4/oof_rbfsvm__none__fixed.csv "
          f"(all 20 region x fold PR-AUCs, largest difference {worst:.2e})", flush=True)


# ----------------------------------------------------------------------------- tables and figures

def curve_table(curves):
    """Mean and fold-SD per (region, arm, p). Averaged over repeats first, so a cell is one number per fold."""
    per_fold = (curves[~curves.skipped].groupby(["region", "arm", "p", "fold"], as_index=False)
                .agg(pr_auc=("pr_auc", "mean"), m=("m", "mean"), m_pos=("m_pos", "mean")))
    t = (per_fold.groupby(["region", "arm", "p"], as_index=False)
         .agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), m=("m", "mean"),
              m_pos=("m_pos", "mean"), n_folds=("pr_auc", "size")))
    sk = (curves.groupby(["region", "arm", "p"], as_index=False).skipped.sum()
          .rename(columns={"skipped": "cells_skipped"}))
    return t.merge(sk, on=["region", "arm", "p"], how="left")


def transfer_table(df, chance):
    """Mean PR-AUC per (train region -> eval region) and the transfer ratio against that pair's diagonal."""
    t = (df.groupby(["train", "eval", "matched"], as_index=False)
         .agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), m=("m", "mean")))
    t["chance"] = t["eval"].map(chance)
    diag = {(r["eval"], r["matched"]): r.pr_auc for _, r in t.iterrows() if r.train == r["eval"]}
    t["within_region_pr_auc"] = [diag.get((r["eval"], r["matched"]), np.nan) for _, r in t.iterrows()]
    denom = t.within_region_pr_auc - t.chance
    t["transfer_ratio"] = np.where(denom > 0, (t.pr_auc - t.chance) / denom, np.nan)
    return t


def fig_curves(ct, chance, figdir):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.4))
    for ax, r in zip(axes, REGIONS):
        g = ct[ct.region == r]
        for arm, lab, col in (("region", "Region-specific (this region's images only)", C["region"]),
                              ("unified", "Unified (this region's images + all other regions)", C["unified"])):
            a = g[g.arm == arm].sort_values("m")
            if not len(a):
                continue
            ax.plot(a.m, a.pr_auc, "-o", color=col, label=lab, markeredgecolor=C["surface"],
                    markeredgewidth=0.8)
            ax.fill_between(a.m, a.pr_auc - a.pr_auc_sd, a.pr_auc + a.pr_auc_sd, color=col, alpha=0.15,
                            linewidth=0)
        ax.axhline(chance[r], color=C["muted"], ls=(0, (3, 2)), lw=1.2)
        full = g[(g.arm == "region") & (g.p == 1.0)]
        ax.set_title(f"{REGION_LABEL[r]} - {int(full.m_pos.iloc[0]) if len(full) else 0} fractured of "
                     f"{int(full.m.iloc[0]) if len(full) else 0}")
        ax.set_xlabel(f"{REGION_LABEL[r].lower()} training images used")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("PR-AUC on the held-out fold")
    axes[0].plot([], [], ls=(0, (3, 2)), color=C["muted"], lw=1.2, label="chance (region fracture rate)")
    header(fig, "Learning curves: how much does a region's own data buy?", axes[0], ncols=3, top=0.84,
           legend_y=0.945)
    save(fig, figdir / "phase5_learning_curves.png",
         "Both arms are fitted on the SAME subsample of the region's images (the unified arm additionally keeps "
         "every training image of the other regions), and both are scored on the same held-out fold, so each "
         "pair of points is paired. The unified arm's leftmost point uses NO image of that region. Hyper"
         "parameters fixed at C=10, gamma=0.05 in both arms; band = +/-1 SD over the 5 outer folds.")


def fig_transfer(tt, figdir):
    setup()
    kinds = [(True, "Size-matched (every training region cut to the same counts)"),
             (False, "Raw (each training region uses all the images it has)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    srcs = list(REGIONS) + ["all regions"]
    for ax, (matched, title) in zip(axes, kinds):
        g = tt[tt.matched == matched]
        M = np.full((len(srcs), len(REGIONS)), np.nan)
        R = np.full_like(M, np.nan)
        for i, a in enumerate(srcs):
            for j, b in enumerate(REGIONS):
                s = g[(g.train == a) & (g["eval"] == b)]
                if len(s):
                    M[i, j], R[i, j] = s.pr_auc.iloc[0], s.transfer_ratio.iloc[0]
        im = ax.imshow(R, cmap=SEQ, vmin=0.0, vmax=1.1, aspect="auto")
        for i in range(len(srcs)):
            for j in range(len(REGIONS)):
                if np.isnan(M[i, j]):
                    continue
                dark = R[i, j] > 0.60          # pale ink on the dark end of the ramp
                ax.text(j, i, f"{M[i, j]:.2f}\n{R[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                        color=C["surface"] if dark else C["ink"])
        ax.set_xticks(range(len(REGIONS)), [REGION_LABEL[b] for b in REGIONS])
        ax.set_yticks(range(len(srcs)), [REGION_LABEL.get(a, "All regions") for a in srcs])
        ax.set_xlabel("evaluated on")
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("trained on")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("transfer ratio", fontsize=8.5)
    cb.outline.set_visible(False)
    fig.suptitle("Cross-region transfer: PR-AUC (top) and transfer ratio (bottom) in each cell",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=11, fontweight="bold")
    save(fig, figdir / "phase5_transfer.png",
         "Transfer ratio = (PR-AUC - chance) / (within-region PR-AUC - chance), which removes the evaluated "
         "region's different chance level: 1.00 = as good as that region's own model, 0.00 = no better than "
         "chance. Size-matching fixes every training region at the same counts, so a weak row means the images "
         "do not transfer rather than that the region is small; it also fixes prevalence near 0.18 for every "
         "row, which is hip's and shoulder's natural rate but not hand's 0.30.")


# ----------------------------------------------------------------------------- main

def summarize(out_dir, tables_dir, figures_dir, chance):
    cp, mp, rp = (out_dir / "curves.csv", out_dir / "transfer_matched.csv", out_dir / "transfer_raw.csv")
    pd.set_option("display.width", 220, "display.max_columns", 40)
    if cp.exists():
        ct = curve_table(pd.read_csv(cp))
        tables_dir.mkdir(parents=True, exist_ok=True)
        ct.round(4).to_csv(tables_dir / "phase5_learning_curves.csv", index=False)
        fig_curves(ct, chance, figures_dir)
        print("\n=== Learning curves: PR-AUC on the held-out fold (mean over folds, +/- fold SD) ===")
        w = ct.pivot_table(index=["region", "p"], columns="arm", values=["pr_auc", "pr_auc_sd", "m_pos"])
        print(w.round(3).to_string())
    tf = [p for p in (mp, rp) if p.exists()]
    if tf:
        df = pd.concat([pd.read_csv(p) for p in tf])
        tt = transfer_table(df, chance)
        tt.round(4).to_csv(tables_dir / "phase5_transfer.csv", index=False)
        fig_transfer(tt, figures_dir)
        for matched in (True, False):
            g = tt[tt.matched == matched]
            if not len(g):
                continue
            print(f"\n=== Cross-region transfer, {'size-matched' if matched else 'raw'}: "
                  "PR-AUC (transfer ratio) ===")
            piv = g.assign(text=[f"{r.pr_auc:.3f} ({r.transfer_ratio:+.2f})" for r in g.itertuples()]) \
                .pivot_table(index="train", columns="eval", values="text", aggfunc="first")
            print(piv.reindex(index=[r for r in list(REGIONS) + ["all regions"] if r in piv.index],
                              columns=list(REGIONS)).to_string())
        print("\nRule 3: a pair 'shares' its fracture signal if the size-matched transfer ratio is above 0.50 "
              "in both directions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["curves", "transfer"], default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--match-pos", type=int, default=30)
    ap.add_argument("--match-neg", type=int, default=140)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quick", action="store_true", help="4 sizes instead of 8 (a fast sanity run)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase5")
    ap.add_argument("--phase4-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tr = load_train(args.tag)
    assert len(tr["y"]) == 3266 and tr["y"].sum() == 576, "unexpected training set; is this the frozen split?"
    chance = {r: float(tr["y"][tr["region"] == r].mean()) for r in REGIONS}

    if not args.summarize:
        print(f"{len(tr['y'])} training images, {tr['y'].sum()} fractured, {tr['X'].shape[1]} features; "
              f"fixed hyperparameters {FIXED}; reps={args.reps}; n_jobs={args.n_jobs}", flush=True)
        t0 = time.time()
        if args.only in (None, "curves"):
            p = args.out_dir / "curves.csv"
            if p.exists() and not args.force:
                print(f"[skip] {p.name} exists; --force to redo")
            else:
                curves = run_curves(tr, args)
                curves.to_csv(p, index=False)
                if not args.no_check:
                    check_vs_phase4_fixed(curves, args.phase4_dir)
                if not args.no_log:
                    log_run(regions="per region (hand, leg, hip, shoulder)",
                            feature_set=f"all five families [{args.tag or 'default'}]",
                            model="RBF SVM (fixed C=10, gamma=0.05)", imbalance_strategy="none (PR-AUC only)",
                            key_hyperparameters=f"Phase 5.2 learning curves; sizes="
                                                f"{QUICK_SIZES if args.quick else SIZES}; reps={args.reps}; "
                                                f"seed={args.seed}",
                            cv_score="see results/tables/phase5_learning_curves.csv",
                            test_metrics="not used (CV only)",
                            notes="Phase 5 diagnostic: region arm vs unified arm vs training-set size, "
                                  "fixed hyperparameters (no tuning), training folds only")
        if args.only in (None, "transfer"):
            for matched, p in ((True, args.out_dir / "transfer_matched.csv"),
                               (False, args.out_dir / "transfer_raw.csv")):
                if p.exists() and not args.force:
                    print(f"[skip] {p.name} exists; --force to redo")
                    continue
                run_transfer(tr, args, matched).to_csv(p, index=False)
                if not args.no_log:
                    log_run(regions="all ordered region pairs + all regions",
                            feature_set=f"all five families [{args.tag or 'default'}]",
                            model="RBF SVM (fixed C=10, gamma=0.05)", imbalance_strategy="none (PR-AUC only)",
                            key_hyperparameters=("size-matched "
                                                 f"{args.match_pos} fractured / {args.match_neg} not, "
                                                 f"reps={args.reps}" if matched else "raw (all rows)")
                                                + f"; seed={args.seed}",
                            cv_score="see results/tables/phase5_transfer.csv",
                            test_metrics="not used (CV only)",
                            notes="Phase 5 diagnostic: cross-region transfer matrix, training folds only")
        print(f"\nFitting finished in {(time.time() - t0) / 60:.1f} min", flush=True)
        (args.out_dir / "run_curves.json").write_text(json.dumps(
            dict(fixed=FIXED, model=MODEL, imbalance=IMB, sizes=list(QUICK_SIZES if args.quick else SIZES),
                 reps=args.reps, match_pos=args.match_pos, match_neg=args.match_neg, seed=args.seed,
                 chance=chance, tag=tr["config"], date=pd.Timestamp.now().isoformat(timespec="seconds")),
            indent=1, default=str))

    summarize(args.out_dir, args.tables_dir, args.figures_dir, chance)


if __name__ == "__main__":
    main()
