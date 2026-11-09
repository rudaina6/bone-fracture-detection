"""Phase 3: unified (all regions pooled) image-level fracture classifier. CROSS-VALIDATION ONLY, no test images.

    python src/phase3_unified.py --models linsvm logreg_l2 --imbalance balanced      # some models, one strategy
    python src/phase3_unified.py --models all --imbalance balanced                   # model comparison (6 models)
    python src/phase3_unified.py --models linsvm --imbalance none smote             # imbalance comparison
    python src/phase3_unified.py --summarize                                         # rebuild the comparison tables
    options: --n-jobs 4  --fast (coarse grids for rf/hgb)  --force (re-run)  --seed 42  --no-log  --out-dir DIR

Each (model, imbalance) run is a NESTED CV (src/cv_protocol.py): hyperparameters and the F2 threshold are chosen
inside each outer training set, so the scores below are not inflated by tuning. A finished run is saved in
results/phase3/ and skipped next time unless --force. Every run is appended to results/experiment_log.csv.

Imbalance strategies (src/model_zoo.py): none | balanced (class_weight) | smote (inside the training folds only).
Threshold tuning is reported for EVERY run as a second operating point ("tuned": F2-optimal threshold chosen inside
the training part of each fold) next to the model's default threshold ("default": score 0 / probability 0.5).

Outputs: results/phase3/oof_<model>__<imb>.csv, run_<model>__<imb>.json,
         results/tables/phase3_cv_summary.csv (one row per run), phase3_cv_per_region.csv
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cv_protocol import edge_warnings, load_train, nested_cv
from evaluate import REGIONS, metrics_by_group, never_fracture_metrics
from expt_log import log_run
from model_zoo import DEFAULT_MODELS, IMBALANCE, SPECS, default_threshold, grid_for
from paths import REPO_ROOT


def run_name(model, imbalance):
    return f"{model}__{imbalance}"


def run_one(tr, model, imbalance, args, out_dir):
    name = run_name(model, imbalance)
    done = out_dir / f"run_{name}.json"
    if done.exists() and not args.force:
        print(f"[skip] {name}: already done ({done.name}); use --force to redo", flush=True)
        return
    grid = grid_for(model, args.fast)
    print(f"\n=== {SPECS[model].label} | imbalance={imbalance} | grid={grid} ===", flush=True)
    r = nested_cv(tr["X"], tr["y"], tr["fold"], model, imbalance, tr["col_weights"], grid=grid, seed=args.seed,
                  n_jobs=args.n_jobs)
    warns = edge_warnings(r["folds"], grid)
    f = pd.DataFrame(r["folds"])
    for w in warns:
        print(f"    WARNING grid edge: {w} -> consider extending the grid", flush=True)
    print(f"    outer-fold PR-AUC {f.pr_auc.mean():.3f} +- {f.pr_auc.std():.3f} | ROC-AUC {f.roc_auc.mean():.3f} | "
          f"default thr: recall {f.def_recall.mean():.3f} F2 {f.def_f2.mean():.3f} | "
          f"tuned thr: recall {f.tuned_recall.mean():.3f} precision {f.tuned_precision.mean():.3f} "
          f"F2 {f.tuned_f2.mean():.3f} | {r['seconds'] / 60:.1f} min", flush=True)
    chosen = Counter(json.dumps(x["params"], sort_keys=True) for x in r["folds"]).most_common()
    print(f"    chosen per fold: {[x['params'] for x in r['folds']]}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(image_id=tr["image_id"], fold=tr["fold"], y=tr["y"], region=tr["region"], score=r["oof"],
                      thr_default=r["thr_default"], thr_tuned=r["thr_tuned"])).to_csv(out_dir / f"oof_{name}.csv",
                                                                                      index=False)
    meta = dict(model=model, imbalance=imbalance, grid={k: [str(v) for v in vs] for k, vs in grid.items()},
                kbest=SPECS[model].kbest, seed=args.seed, tag=tr["config"], n_train=int(len(tr["y"])),
                n_pos=int(tr["y"].sum()), seconds=r["seconds"], edge_warnings=warns, folds=r["folds"],
                modal_params=json.loads(chosen[0][0]), date=pd.Timestamp.now().isoformat(timespec="seconds"))
    done.write_text(json.dumps(meta, indent=1, default=str))

    if not args.no_log:
        log_run(regions="all (pooled)", feature_set=f"all five families [{args.tag or 'default'}]",
                model=SPECS[model].label, imbalance_strategy=imbalance,
                key_hyperparameters=f"nested 5x4 CV; modal params={chosen[0][0]} ({chosen[0][1]}/5 folds); "
                                    f"kbest={SPECS[model].kbest}; fast={args.fast}; seed={args.seed}",
                cv_score=f"PR-AUC={f.pr_auc.mean():.3f}+-{f.pr_auc.std():.3f}; ROC-AUC={f.roc_auc.mean():.3f}; "
                         f"default-thr recall={f.def_recall.mean():.3f} F2={f.def_f2.mean():.3f}; "
                         f"tuned-thr recall={f.tuned_recall.mean():.3f} precision={f.tuned_precision.mean():.3f} "
                         f"F2={f.tuned_f2.mean():.3f}",
                test_metrics="not used (CV only)",
                notes="Phase 3 unified, outer CV scores (hyperparameters and threshold chosen inside training folds)"
                      + ("; " + "; ".join(warns) if warns else ""))


def summarize(out_dir, tables_dir):
    rows, region_rows = [], []
    for p in sorted(out_dir.glob("run_*.json")):
        meta = json.loads(p.read_text())
        oof = pd.read_csv(out_dir / f"oof_{run_name(meta['model'], meta['imbalance'])}.csv")
        f = pd.DataFrame(meta["folds"])
        pooled_def = metrics_by_group(oof.y, oof.score, oof.region, oof.thr_default.to_numpy()).set_index("group")
        pooled_tun = metrics_by_group(oof.y, oof.score, oof.region, oof.thr_tuned.to_numpy()).set_index("group")
        rows.append(dict(
            model=meta["model"], imbalance=meta["imbalance"],
            pr_auc=f.pr_auc.mean(), pr_auc_sd=f.pr_auc.std(), roc_auc=f.roc_auc.mean(),
            def_recall=f.def_recall.mean(), def_precision=f.def_precision.mean(), def_f2=f.def_f2.mean(),
            def_specificity=f.def_specificity.mean(),
            tuned_recall=f.tuned_recall.mean(), tuned_precision=f.tuned_precision.mean(),
            tuned_f2=f.tuned_f2.mean(), tuned_f2_sd=f.tuned_f2.std(), tuned_specificity=f.tuned_specificity.mean(),
            pooled_pr_auc=pooled_tun.loc["overall", "pr_auc"],
            **{f"pr_auc_{g}": pooled_tun.loc[g, "pr_auc"] for g in REGIONS},
            modal_params=json.dumps(meta["modal_params"]), edge_warnings="; ".join(meta["edge_warnings"]),
            minutes=meta["seconds"] / 60))
        for op, tab in (("default", pooled_def), ("tuned", pooled_tun)):
            for g, r in tab.iterrows():
                region_rows.append(dict(model=meta["model"], imbalance=meta["imbalance"], threshold=op, group=g,
                                        n=r.n, n_pos=r.n_pos, recall=r.recall, precision=r.precision, f2=r.f2,
                                        specificity=r.specificity, roc_auc=r.roc_auc, pr_auc=r.pr_auc))
    if not rows:
        print("no finished runs in", out_dir)
        return
    summ = pd.DataFrame(rows).sort_values(["pr_auc"], ascending=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summ.round(4).to_csv(tables_dir / "phase3_cv_summary.csv", index=False)
    pd.DataFrame(region_rows).round(4).to_csv(tables_dir / "phase3_cv_per_region.csv", index=False)

    # chance / trivial baseline from any oof file (same training labels)
    y = oof.y.to_numpy()
    base = never_fracture_metrics(y)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = ["model", "imbalance", "pr_auc", "pr_auc_sd", "roc_auc", "def_recall", "def_f2", "tuned_recall",
            "tuned_precision", "tuned_f2", "tuned_specificity", "minutes"]
    print(f"\nNested-CV summary on the training images (mean over the 5 outer folds; sorted by PR-AUC)")
    print(f"Chance PR-AUC = fracture rate = {y.mean():.3f}. Never-fracture baseline: accuracy {base['accuracy']:.3f}, "
          f"recall 0, F2 0, ROC-AUC 0.5, PR-AUC {base['pr_auc']:.3f}")
    print(summ[show].round(3).to_string(index=False))
    print("\nPooled out-of-fold PR-AUC by region (chance = region fracture rate):")
    prev = pd.Series(oof.groupby("region").y.mean())
    print("  chance: " + ", ".join(f"{g} {prev[g]:.3f}" for g in REGIONS))
    print(summ[["model", "imbalance"] + [f"pr_auc_{g}" for g in REGIONS]].round(3).to_string(index=False))
    if summ.edge_warnings.str.len().any():
        print("\nGrid-edge warnings:")
        print(summ.loc[summ.edge_warnings.str.len() > 0, ["model", "imbalance", "edge_warnings"]].to_string(index=False))
    print(f"\nWrote {tables_dir / 'phase3_cv_summary.csv'} and phase3_cv_per_region.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["linsvm"], help=f"{list(SPECS)} or 'all' ({DEFAULT_MODELS})")
    ap.add_argument("--imbalance", nargs="+", default=["balanced"], choices=list(IMBALANCE))
    ap.add_argument("--tag", default=None, help="feature set (default: the Phase 2 choice)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--fast", action="store_true", help="coarser grids for rf and hgb")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="only rebuild the summary tables from saved runs")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--subsample", type=int, default=0, help="DEBUG: use only N random training images")
    args = ap.parse_args()
    if args.subsample:
        args.no_log = True                      # debug runs never go into the experiment log

    if not args.summarize:
        models = DEFAULT_MODELS if args.models == ["all"] else args.models
        bad = [m for m in models if m not in SPECS]
        if bad:
            ap.error(f"unknown models {bad}; choose from {list(SPECS)}")
        tr = load_train(args.tag)
        if args.subsample:
            keep = np.random.default_rng(0).choice(len(tr["y"]), args.subsample, replace=False)
            for k in ("X", "y", "fold", "region", "image_id"):
                tr[k] = tr[k][keep]
        else:
            assert len(tr["y"]) == 3266 and tr["y"].sum() == 576, "unexpected training set; is this the frozen split?"
        print(f"{len(tr['y'])} training images, {tr['y'].sum()} fractured, {tr['X'].shape[1]} features; "
              f"models={models}; imbalance={args.imbalance}", flush=True)
        t0 = time.time()
        for model in models:
            for imb in args.imbalance:
                if imb == "balanced" and not SPECS[model].supports_cw:
                    print(f"[skip] {model} has no class weights")
                    continue
                run_one(tr, model, imb, args, args.out_dir)
        print(f"\nAll requested runs finished in {(time.time() - t0) / 60:.1f} min")
    summarize(args.out_dir, args.tables_dir)


if __name__ == "__main__":
    main()
