"""Phase 4 FINAL test evaluation of the four region-specific classifiers. THE TEST SET IS USED ONCE.

    python src/phase4_final_test.py --model rbfsvm --imbalance none
    options: --n-boot 2000, --allow-rerun, --n-jobs 4, --no-log

Run only after the main CV run (phase4_regional.py --models rbfsvm --imbalance none) is finished and written into
docs/phase4_regional.md (decision rule 4). Per region (hand, leg, hip, shoulder), test rows untouched until step 3:
  1. grid search over the region's rows of the 5 frozen folds -> best hyperparameters   (cv_protocol.tune_cv)
  2. F2-optimal threshold from that config's out-of-fold training scores
  3. refit on ALL the region's training images, score the region's test images once
Then, on the same test images:
  4. region-specific metrics with 95% bootstrap CIs (tuned and default thresholds)
  5. the unified Phase 3 model on the same images (its saved scores, results/phase3/test_scores.csv, and its threshold
     from results/phase3/test_run.json; nothing is refitted)
  6. paired bootstrap of region-specific minus unified, per region and for the routed system (threshold metrics only)
A lock file (results/phase4/TEST_EVALUATED.json) records the evaluation; running again needs --allow-rerun, and then
BOTH results must be reported.

Outputs: results/phase4/test_scores.csv, test_run.json,
         results/tables/phase4_test_{metrics,ci,paired,confusion}.csv
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cv_protocol import tune_cv
from evaluate import REGIONS, bootstrap_ci, format_ci_table, metrics_by_group, paired_bootstrap
from expt_log import log_run
from feature_utils import load_features
from model_zoo import SPECS, build, default_threshold, grid_for, scores
from paths import REPO_ROOT

PAIRED_METRICS = ("pr_auc", "roc_auc", "recall", "precision", "specificity", "f2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rbfsvm", choices=list(SPECS))
    ap.add_argument("--imbalance", default="none", choices=["none", "balanced", "smote"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--phase3-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    args = ap.parse_args()

    lock = args.out_dir / "TEST_EVALUATED.json"
    if lock.exists() and not args.allow_rerun:
        raise SystemExit(f"The Phase 4 test set was already evaluated ({lock.read_text()[:300]}...).\n"
                         "Do not look at it again to choose between models. If this is a deliberate re-run "
                         "(e.g. a crash), pass --allow-rerun and report both results.")
    cv_run = args.out_dir / f"run_{args.model}__{args.imbalance}.json"
    if not cv_run.exists():
        raise SystemExit(f"{cv_run} not found: run the CV first (decision rule 4):\n"
                         f"  python src/phase4_regional.py --models {args.model} --imbalance {args.imbalance}")
    uni_scores = args.phase3_dir / "test_scores.csv"
    uni_run = json.loads((args.phase3_dir / "test_run.json").read_text())
    uni_thr = float(uni_run["thr_tuned"])

    t0 = time.time()
    d = load_features(args.tag) if args.tag else load_features()
    trm, tem = d["split"] == "train", d["split"] == "test"
    assert trm.sum() == 3266 and d["y"][trm].sum() == 576 and tem.sum() == 817 and d["y"][tem].sum() == 143, \
        "not the frozen split"
    spec, grid, w = SPECS[args.model], grid_for(args.model), d["col_weights"]
    thr_def = default_threshold(args.model)
    print(f"Region-specific models: {spec.label}, imbalance={args.imbalance}, grid={grid}", flush=True)

    # 1-3 per region: tune on that region's training rows, then score its test rows once
    parts, settings = [], {}
    for reg in REGIONS:
        rtr, rte = trm & (d["region"] == reg), tem & (d["region"] == reg)
        X, y, fold = d["X"][rtr], d["y"][rtr].astype(int), d["fold"][rtr]
        print(f"\n=== {reg}: tuning on {rtr.sum()} training images ({y.sum()} fractured) ===", flush=True)
        tuned = tune_cv(X, y, fold, args.model, args.imbalance, w, grid=grid, seed=args.seed, n_jobs=args.n_jobs)
        print(f"  best params {tuned['params']}  CV PR-AUC {tuned['cv_pr_auc']:.3f} (optimistic: tuned on these folds); "
              f"F2 threshold {tuned['thr_tuned']:.4f}, OOF F2 there {tuned['f2_at_thr']:.3f}", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = build(args.model, args.imbalance, tuned["params"], w, args.seed).fit(X, y)
            s = scores(pipe, d["X"][rte])
        settings[reg] = dict(params=tuned["params"], thr_tuned=tuned["thr_tuned"], cv_pr_auc=tuned["cv_pr_auc"],
                             oof_f2=tuned["f2_at_thr"], n_train=int(rtr.sum()), n_train_pos=int(y.sum()))
        parts.append(pd.DataFrame(dict(image_id=d["image_id"][rte], y=d["y"][rte].astype(int), region=reg, score=s,
                                       thr_tuned=tuned["thr_tuned"],
                                       hardware=d["meta"]["hardware"].to_numpy()[rte])))
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(dict(model=args.model, imbalance=args.imbalance, settings=settings,
                                    date=pd.Timestamp.now().isoformat(timespec="seconds")), default=str))

    # 5: attach the unified model's saved scores for the same images
    out = pd.concat(parts, ignore_index=True)
    uni = pd.read_csv(uni_scores)[["image_id", "y", "region", "score"]].rename(
        columns={"y": "y_u", "region": "region_u", "score": "score_unified"})
    out = out.merge(uni, on="image_id", how="left", validate="one_to_one")
    assert out.score_unified.notna().all(), "some region test images have no unified score"
    assert (out.y == out.y_u).all() and (out.region == out.region_u).all(), "unified scores disagree on labels"
    out = out.drop(columns=["y_u", "region_u"]).assign(thr_unified=uni_thr)
    out.to_csv(args.out_dir / "test_scores.csv", index=False)
    y, reg, s_r, s_u = out.y.to_numpy(), out.region.to_numpy(), out.score.to_numpy(), out.score_unified.to_numpy()
    thr_r = out.thr_tuned.to_numpy()

    # 4-5: metrics for both models on the same images
    tables = args.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    res = pd.concat([
        metrics_by_group(y, s_r, reg, thr_r).assign(model="region-specific", threshold="tuned"),
        metrics_by_group(y, s_r, reg, thr_def).assign(model="region-specific", threshold="default"),
        metrics_by_group(y, s_u, reg, uni_thr).assign(model="unified", threshold="tuned"),
    ])
    # the "overall" row of the region-specific model = the routed system; its ranking metrics mix 4 models' scales
    routed = (res.model == "region-specific") & (res.group == "overall")
    res.loc[routed, ["roc_auc", "pr_auc"]] = np.nan
    res["group"] = res.group.replace({"overall": "routed (4 regions)"})
    res.round(4).to_csv(tables / "phase4_test_metrics.csv", index=False)
    res[["model", "threshold", "group", "n", "n_pos", "tn", "fp", "fn", "tp"]].to_csv(
        tables / "phase4_test_confusion.csv", index=False)
    ci = pd.concat([
        bootstrap_ci(y, s_r, thr_r, reg, n_boot=args.n_boot, seed=args.seed).assign(model="region-specific"),
        bootstrap_ci(y, s_u, uni_thr, reg, n_boot=args.n_boot, seed=args.seed).assign(model="unified"),
    ])
    ci.loc[(ci.model == "region-specific") & (ci.group == "overall") & ci.metric.isin(["roc_auc", "pr_auc"]),
           ["estimate", "lo", "hi"]] = np.nan
    ci.round(4).to_csv(tables / "phase4_test_ci.csv", index=False)

    # 6: paired bootstrap, region-specific minus unified, same images
    pb = paired_bootstrap(y, s_r, s_u, thr_r, uni_thr, groups=reg, n_boot=args.n_boot, seed=args.seed,
                          metrics=PAIRED_METRICS)
    pb.loc[(pb.group == "overall") & pb.metric.isin(["pr_auc", "roc_auc"]), ["diff", "lo", "hi", "p_two_sided"]] = np.nan
    pb["group"] = pb.group.replace({"overall": "routed (4 regions)"})
    pb.round(4).to_csv(tables / "phase4_test_paired.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    for name in ("region-specific", "unified"):
        print(f"\n=== TEST, {name} model, tuned threshold(s): estimate [95% bootstrap CI], {args.n_boot} resamples ===")
        t = format_ci_table(ci[ci.model == name])
        if name == "region-specific":
            t = t.rename(index={"overall": "routed (4 regions)"})
        print(t.to_string())
    print("\nConfusion (tuned thresholds):")
    print(res[res.threshold == "tuned"][["model", "group", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print("\nRegion-specific MINUS unified on the same test images (paired bootstrap 95% CI, thresholds fixed):")
    pv = pb.assign(text=[("n/a" if np.isnan(r.diff) else f"{r.diff:+.3f}"
                          + ("" if np.isnan(r.lo) else f" [{r.lo:+.3f}, {r.hi:+.3f}] p={r.p_two_sided:.2f}"))
                         for r in pb.itertuples()])
    print(pv.pivot(index="group", columns="metric", values="text")[list(PAIRED_METRICS)].to_string())
    print("Rule 2: 'better'/'worse' only if the PR-AUC CI excludes 0.")

    run = dict(model=args.model, imbalance=args.imbalance, settings=settings, thr_default=thr_def,
               unified_thr=uni_thr, unified_params=uni_run.get("params"), seed=args.seed,
               grid={k: [str(v) for v in vs] for k, vs in grid.items()}, seconds=time.time() - t0,
               date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "test_run.json").write_text(json.dumps(run, indent=1, default=str))

    if not args.no_log:
        tun = res[res.threshold == "tuned"].set_index(["model", "group"])
        for g in REGIONS:
            r, u = tun.loc[("region-specific", g)], tun.loc[("unified", g)]
            dpr = pb[(pb.group == g) & (pb.metric == "pr_auc")].iloc[0]
            cr = ci[(ci.model == "region-specific") & (ci.group == g)].set_index("metric")
            st = settings[g]
            log_run(regions=f"{g} (region-specific), test on {g}",
                    feature_set=f"all five families [{args.tag or 'default'}]", model=spec.label,
                    imbalance_strategy=f"{args.imbalance} + F2 threshold tuning",
                    key_hyperparameters=f"{json.dumps(st['params'])}; thr={st['thr_tuned']:.4f}; seed={args.seed}",
                    cv_score=f"tuning-fold PR-AUC={st['cv_pr_auc']:.3f} (optimistic); OOF F2 at thr={st['oof_f2']:.3f}",
                    test_metrics=(f"TEST n={int(r.n)} pos={int(r.n_pos)}: recall={r.recall:.3f} precision={r.precision:.3f} "
                                  f"F2={r.f2:.3f} spec={r.specificity:.3f} PR-AUC={r.pr_auc:.3f} "
                                  f"[{cr.loc['pr_auc', 'lo']:.3f}, {cr.loc['pr_auc', 'hi']:.3f}] "
                                  f"TN/FP/FN/TP={int(r.tn)}/{int(r.fp)}/{int(r.fn)}/{int(r.tp)}; unified same images: "
                                  f"recall={u.recall:.3f} F2={u.f2:.3f} PR-AUC={u.pr_auc:.3f}; "
                                  f"dPR-AUC={dpr['diff']:+.3f} [{dpr.lo:+.3f}, {dpr.hi:+.3f}]"),
                    notes="Phase 4 FINAL test evaluation (test set used once)")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Per-image scores for Phase 5: {args.out_dir / 'test_scores.csv'}")


if __name__ == "__main__":
    main()
