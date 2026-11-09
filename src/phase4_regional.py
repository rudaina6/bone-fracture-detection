"""Phase 4: region-specific classifiers (one per region). CROSS-VALIDATION ONLY, no test images.

    python src/phase4_regional.py --models rbfsvm --imbalance none                     # the Phase 4 model, 4 regions
    python src/phase4_regional.py --models rbfsvm --imbalance none --fixed-params C=10 gamma=0.05   # sensitivity
    python src/phase4_regional.py --models linsvm --imbalance none                     # family check
    python src/phase4_regional.py --summarize                                          # rebuild the tables
    options: --regions hand leg hip shoulder  --n-jobs 4  --force  --seed 42  --no-log  --n-boot 2000

For each region, the SAME nested CV as Phase 3 (src/cv_protocol.py) is run on that region's training rows only, with the
region's slice of the frozen folds; same model, grid and imbalance strategy (src/model_zoo.py). Protocol and decision
rules: docs/phase4_regional.md (written before any result).

The comparison partner is the Phase 3 unified nested-CV run with the same model and strategy
(results/phase3/oof_<model>__<imb>.csv): its outer-fold scores on the SAME images, from the SAME folds. The difference
(region-specific minus unified) gets a paired bootstrap over images: PR-AUC / ROC-AUC as the mean of per-fold differences
(fold_paired_bootstrap, images resampled within folds), threshold metrics pooled (evaluate.paired_bootstrap).

Outputs: results/phase4/oof_<run>.csv (each row scored by its own region's model), run_<run>.json,
         results/tables/phase4_cv_summary.csv (per run x region), phase4_cv_paired.csv (differences with CIs)
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cv_protocol import edge_warnings, load_train, nested_cv
from evaluate import REGIONS, all_metrics, paired_bootstrap
from expt_log import log_run
from model_zoo import IMBALANCE, SPECS, grid_for
from paths import REPO_ROOT

PHASE3_DIR = REPO_ROOT / "results" / "phase3"
PAIRED_METRICS = ("pr_auc", "roc_auc", "recall", "precision", "specificity", "f2")
THR_METRICS = ("recall", "precision", "specificity", "f2")


def run_name(model, imbalance, fixed=None):
    return f"{model}__{imbalance}" + ("__fixed" if fixed else "")


def parse_fixed(items, model):
    """["C=10", "gamma=0.05"] -> {"C": 10.0, "gamma": 0.05}; keys must be the model's grid keys."""
    if not items:
        return None
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        if k not in SPECS[model].grid:
            raise SystemExit(f"--fixed-params: {k} is not a hyperparameter of {model} ({list(SPECS[model].grid)})")
        out[k] = None if v == "None" else (int(v) if v.isdigit() else float(v))
    missing = set(SPECS[model].grid) - set(out)
    if missing:
        raise SystemExit(f"--fixed-params must set every hyperparameter of {model}; missing {sorted(missing)}")
    return out


def check_region_folds(y, fold, region):
    ks = sorted(set(fold))
    assert ks == [0, 1, 2, 3, 4], f"{region}: folds {ks}"
    for k in ks:
        n_pos = int(y[fold == k].sum())
        assert 0 < n_pos < (fold == k).sum(), f"{region} fold {k} has a single class"


def run_one(tr, model, imbalance, fixed, args, out_dir):
    name = run_name(model, imbalance, fixed)
    done = out_dir / f"run_{name}.json"
    if done.exists() and not args.force:
        print(f"[skip] {name}: already done ({done.name}); use --force to redo", flush=True)
        return
    grid = {k: [v] for k, v in fixed.items()} if fixed else grid_for(model)
    print(f"\n######## {SPECS[model].label} | imbalance={imbalance} | grid={grid} | regions={args.regions} ########",
          flush=True)
    oof_parts, per_region = [], {}
    for reg in args.regions:
        m = tr["region"] == reg
        X, y, fold = tr["X"][m], tr["y"][m], tr["fold"][m]
        check_region_folds(y, fold, reg)
        print(f"\n=== {reg}: {m.sum()} training images, {y.sum()} fractured "
              f"({', '.join(str(int(y[fold == k].sum())) for k in range(5))} per fold) ===", flush=True)
        r = nested_cv(X, y, fold, model, imbalance, tr["col_weights"], grid=grid, seed=args.seed, n_jobs=args.n_jobs)
        warns = [] if fixed else edge_warnings(r["folds"], grid)
        f = pd.DataFrame(r["folds"])
        chosen = Counter(json.dumps(x["params"], sort_keys=True) for x in r["folds"]).most_common()
        print(f"    outer-fold PR-AUC {f.pr_auc.mean():.3f} +- {f.pr_auc.std():.3f} (chance {y.mean():.3f}) | "
              f"ROC-AUC {f.roc_auc.mean():.3f} | tuned thr: recall {f.tuned_recall.mean():.3f} "
              f"precision {f.tuned_precision.mean():.3f} F2 {f.tuned_f2.mean():.3f} "
              f"spec {f.tuned_specificity.mean():.3f} | {r['seconds'] / 60:.1f} min", flush=True)
        print(f"    chosen per fold: {[x['params'] for x in r['folds']]}", flush=True)
        for w in warns:
            print(f"    WARNING grid edge: {w} (rule 3: report it, do not extend the grid for one region)", flush=True)
        oof_parts.append(pd.DataFrame(dict(image_id=tr["image_id"][m], fold=fold, y=y, region=reg, score=r["oof"],
                                           thr_default=r["thr_default"], thr_tuned=r["thr_tuned"])))
        per_region[reg] = dict(n_train=int(m.sum()), n_pos=int(y.sum()), seconds=r["seconds"], edge_warnings=warns,
                               folds=r["folds"], modal_params=json.loads(chosen[0][0]),
                               modal_count=int(chosen[0][1]))
        if not args.no_log:
            log_run(regions=f"{reg} (region-specific)",
                    feature_set=f"all five families [{args.tag or 'default'}]",
                    model=SPECS[model].label, imbalance_strategy=f"{imbalance} + F2 threshold tuning",
                    key_hyperparameters=(f"fixed params={json.dumps(fixed)} (threshold tuned only)" if fixed else
                                         f"nested 5x4 CV on {reg} rows; modal params={chosen[0][0]} "
                                         f"({chosen[0][1]}/5 folds)") + f"; kbest={SPECS[model].kbest}; seed={args.seed}",
                    cv_score=f"PR-AUC={f.pr_auc.mean():.3f}+-{f.pr_auc.std():.3f}; ROC-AUC={f.roc_auc.mean():.3f}; "
                             f"tuned-thr recall={f.tuned_recall.mean():.3f} precision={f.tuned_precision.mean():.3f} "
                             f"F2={f.tuned_f2.mean():.3f} spec={f.tuned_specificity.mean():.3f}",
                    test_metrics="not used (CV only)",
                    notes=("Phase 4 SENSITIVITY (fixed Phase 3 hyperparameters; slightly optimistic, diagnostic only)"
                           if fixed else "Phase 4 region-specific, outer CV scores (tuning inside training folds)")
                          + ("; " + "; ".join(warns) if warns else ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(oof_parts).to_csv(out_dir / f"oof_{name}.csv", index=False)
    meta = dict(model=model, imbalance=imbalance, fixed_params=fixed,
                grid={k: [str(v) for v in vs] for k, vs in grid.items()}, kbest=SPECS[model].kbest, seed=args.seed,
                tag=tr["config"], regions=per_region, date=pd.Timestamp.now().isoformat(timespec="seconds"))
    done.write_text(json.dumps(meta, indent=1, default=str))


def _fold_pr_auc(df):
    """PR-AUC per outer fold for one region's rows (unified scores are evaluated the same way as region scores)."""
    from sklearn.metrics import average_precision_score
    return np.array([average_precision_score(g.y, g.score) for _, g in df.groupby("fold")])


def _fast_ap(y, s):
    """average_precision_score(y, s), same definition (ties handled the same way), ~17x faster for the bootstrap.
    Checked against sklearn to 1e-12 on 300 random cases with ties."""
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]          # last index of each distinct score
    tps = np.cumsum(y)[last]
    prec, rec = tps / (last + 1), tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _fast_auc(y, s):
    """roc_auc_score(y, s) via the rank-sum (Mann-Whitney) formula, ties get average ranks."""
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def fold_paired_bootstrap(y, s_a, s_b, fold, n_boot=2000, seed=0, level=0.95):
    """Fold-mean PR-AUC and ROC-AUC difference (A - B) with a paired bootstrap that resamples images WITHIN each fold.
    Why not pooled: each outer fold has its own fitted model (and, for small regions, often different C/gamma), so
    margins from different folds are on different scales; pooling them penalises whichever approach varies more across
    folds. Averaging per-fold metrics compares like with like. Returns {metric: (diff, lo, hi, p_two_sided)}."""
    y, s_a, s_b, fold = (np.asarray(v) for v in (y, s_a, s_b, fold))
    y, s_a, s_b = y.astype(int), s_a.astype(float), s_b.astype(float)
    idx = [np.flatnonzero(fold == k) for k in sorted(set(fold))]
    fns = dict(pr_auc=_fast_ap, roc_auc=_fast_auc)

    def stat(sel):
        out = {}
        for m, fn in fns.items():
            out[m] = np.mean([fn(y[i], s_a[i]) - fn(y[i], s_b[i]) for i in sel])
        return out

    est = stat(idx)
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in fns}
    for _ in range(n_boot):
        sel = [i[rng.integers(0, len(i), len(i))] for i in idx]
        if any(y[i].min() == y[i].max() for i in sel):
            continue
        for m, v in stat(sel).items():
            draws[m].append(v)
    a = (1 - level) / 2
    res = {}
    for m in fns:
        dr = np.array(draws[m])
        lo, hi = np.percentile(dr, [100 * a, 100 * (1 - a)])
        p = min(1.0, 2 * min((dr <= 0).mean(), (dr >= 0).mean()))
        res[m] = (est[m], lo, hi, p)
    return res


def summarize(out_dir, tables_dir, n_boot, seed):
    summ, paired = [], []
    runs = sorted(out_dir.glob("run_*.json"))
    if not runs:
        print("no finished Phase 4 runs in", out_dir)
        return
    for p in runs:
        meta = json.loads(p.read_text())
        name = p.stem[4:]
        print(f"[summary] {name}: bootstrapping ({n_boot} resamples)...", flush=True)
        reg_oof = pd.read_csv(out_dir / f"oof_{name}.csv")
        uni_path = PHASE3_DIR / f"oof_{meta['model']}__{meta['imbalance']}.csv"
        uni = pd.read_csv(uni_path) if uni_path.exists() else None
        if uni is not None:
            j = reg_oof.merge(uni, on="image_id", suffixes=("", "_u"), validate="one_to_one")
            assert len(j) == len(reg_oof), "unified OOF file is missing some images"
            assert (j.y == j.y_u).all() and (j.fold == j.fold_u).all() and (j.region == j.region_u).all(), \
                "unified and region OOF disagree on labels/folds/regions"
        else:
            print(f"[note] {name}: no unified counterpart {uni_path.name}; region-only numbers")
            j = reg_oof
        for reg in [r for r in REGIONS if r in set(reg_oof.region)]:
            g = j[j.region == reg]
            info = meta["regions"][reg]
            f = pd.DataFrame(info["folds"])
            row = dict(run=name, model=meta["model"], imbalance=meta["imbalance"], fixed=bool(meta["fixed_params"]),
                       region=reg, n=len(g), n_pos=int(g.y.sum()), chance=g.y.mean(),
                       reg_pr_auc_fold=f.pr_auc.mean(), reg_pr_auc_fold_sd=f.pr_auc.std(),
                       modal_params=json.dumps(info["modal_params"]), modal_count=info["modal_count"],
                       params_per_fold="; ".join(json.dumps(x["params"]) for x in info["folds"]),
                       edge_warnings="; ".join(info["edge_warnings"]), minutes=info["seconds"] / 60)
            m_reg = all_metrics(g.y, g.score, g.thr_tuned.to_numpy())
            row.update({f"reg_{k}": m_reg[k] for k in ("pr_auc", "roc_auc") + THR_METRICS})
            if uni is not None:
                gu = g.rename(columns={"score": "s_r", "score_u": "score"})
                fu = _fold_pr_auc(gu)
                fr = _fold_pr_auc(g)
                m_uni = all_metrics(g.y, g.score_u, g.thr_tuned_u.to_numpy())
                row.update(uni_pr_auc_fold=fu.mean(), uni_pr_auc_fold_sd=fu.std(ddof=1),
                           folds_region_better=int((fr > fu).sum()),
                           **{f"uni_{k}": m_uni[k] for k in ("pr_auc", "roc_auc") + THR_METRICS})
            summ.append(row)
        if uni is None:
            continue
        # paired bootstrap, region-specific minus unified, per region (ranking + threshold metrics)
        j4 = j[j.region.isin(REGIONS)].reset_index(drop=True)
        pb = paired_bootstrap(j4.y, j4.score, j4.score_u, j4.thr_tuned.to_numpy(), j4.thr_tuned_u.to_numpy(),
                              groups=j4.region, n_boot=n_boot, seed=seed, metrics=PAIRED_METRICS)
        # ranking metrics: replace the pooled differences by fold-mean differences (fold-stratified bootstrap); the
        # "overall" (routed) row mixes four region models, so it gets no ranking metrics at all
        rk = pb.metric.isin(["pr_auc", "roc_auc"])
        pb.loc[rk, ["diff", "lo", "hi", "p_two_sided"]] = np.nan
        for g in [r for r in REGIONS if r in set(j4.region)]:
            gg = j4[j4.region == g]
            if gg.y.sum() < 10:
                continue
            fb = fold_paired_bootstrap(gg.y, gg.score, gg.score_u, gg.fold, n_boot=n_boot, seed=seed)
            for m, vals in fb.items():
                pb.loc[(pb.group == g) & (pb.metric == m), ["diff", "lo", "hi", "p_two_sided"]] = vals
        pb["group"] = pb.group.replace({"overall": "routed (4 regions)"})
        paired.append(pb.assign(run=name, model=meta["model"], imbalance=meta["imbalance"],
                                fixed=bool(meta["fixed_params"])))

    summ = pd.DataFrame(summ)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summ.round(4).to_csv(tables_dir / "phase4_cv_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    print("\nPhase 4 nested CV per region (training images). reg_ = region-specific, uni_ = Phase 3 unified on the SAME")
    print("images and folds. *_fold = mean of the 5 outer folds; others pooled out-of-fold, tuned thresholds.")
    show = ["run", "region", "n_pos", "chance", "reg_pr_auc_fold", "reg_pr_auc_fold_sd", "uni_pr_auc_fold",
            "uni_pr_auc_fold_sd", "folds_region_better", "reg_pr_auc", "uni_pr_auc", "reg_f2", "uni_f2",
            "reg_recall", "uni_recall", "reg_specificity", "uni_specificity"]
    print(summ[[c for c in show if c in summ.columns]].round(3).to_string(index=False))
    print("\nChosen hyperparameters (modal, count of 5 folds):")
    print(summ[["run", "region", "modal_params", "modal_count", "edge_warnings"]].to_string(index=False))
    if paired:
        paired = pd.concat(paired)
        paired.round(4).to_csv(tables_dir / "phase4_cv_paired.csv", index=False)
        pv = paired.assign(text=[("n/a" if np.isnan(r.diff) else
                                  f"{r.diff:+.3f}" + ("" if np.isnan(r.lo) else f" [{r.lo:+.3f}, {r.hi:+.3f}]"))
                                 for r in paired.itertuples()])
        wide = pv.pivot_table(index=["run", "group"], columns="metric", values="text", aggfunc="first")
        print(f"\nRegion-specific MINUS unified, paired bootstrap 95% CI ({n_boot} resamples, thresholds fixed).\n"
              "pr_auc / roc_auc: mean of the 5 per-fold differences (images resampled within folds); recall, precision,\n"
              "specificity, f2: pooled out-of-fold at each fold's tuned threshold.")
        print("Rule 2: a region is 'better'/'worse' only if the PR-AUC CI excludes 0.")
        print(wide[[m for m in PAIRED_METRICS if m in wide.columns]].to_string())
    print(f"\nWrote {tables_dir / 'phase4_cv_summary.csv'}" + (" and phase4_cv_paired.csv" if len(paired) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rbfsvm"], choices=list(SPECS))
    ap.add_argument("--imbalance", nargs="+", default=["none"], choices=list(IMBALANCE))
    ap.add_argument("--regions", nargs="+", default=REGIONS, choices=REGIONS)
    ap.add_argument("--fixed-params", nargs="+", default=None, metavar="KEY=VALUE",
                    help="sensitivity run: no hyperparameter tuning, e.g. C=10 gamma=0.05 (threshold still tuned)")
    ap.add_argument("--tag", default=None, help="feature set (default: the Phase 2 choice)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="only rebuild the summary tables from saved runs")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    args = ap.parse_args()

    if not args.summarize:
        if args.fixed_params and len(args.models) > 1:
            ap.error("--fixed-params needs exactly one model")
        if args.regions != REGIONS:
            print(f"[note] only regions {args.regions}: the saved run will contain these regions only (debug use)")
        tr = load_train(args.tag)
        assert len(tr["y"]) == 3266 and tr["y"].sum() == 576, "unexpected training set; is this the frozen split?"
        print(f"{len(tr['y'])} training images, {tr['y'].sum()} fractured, {tr['X'].shape[1]} features; "
              f"models={args.models}; imbalance={args.imbalance}; regions={args.regions}", flush=True)
        t0 = time.time()
        for model in args.models:
            fixed = parse_fixed(args.fixed_params, model)
            for imb in args.imbalance:
                if imb == "balanced" and not SPECS[model].supports_cw:
                    print(f"[skip] {model} has no class weights")
                    continue
                run_one(tr, model, imb, fixed, args, args.out_dir)
        print(f"\nAll requested runs finished in {(time.time() - t0) / 60:.1f} min")
    summarize(args.out_dir, args.tables_dir, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
