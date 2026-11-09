"""Phase 3 FINAL test evaluation of the chosen unified classifier. THE TEST SET IS USED ONCE.

    python src/phase3_final_test.py --model linsvm --imbalance balanced
    options: --n-boot 2000, --save-model, --fast, --allow-rerun, --n-jobs 4

Run this only after the CV comparison (phase3_unified.py) has settled the model and the imbalance strategy.
Steps (the test rows are not touched until step 3):
  1. grid search over the 5 frozen folds of the 3,266 training images -> best hyperparameters
  2. the F2-optimal threshold is chosen from that best config's out-of-fold training scores
  3. refit on ALL training images with those settings, score the 817 test images once
  4. recall, precision, F1, F2, specificity, ROC-AUC, PR-AUC, confusion matrix: overall and per region, with
     95% bootstrap CIs (images resampled), at the tuned threshold and at the model's default threshold;
     plus the never-fracture baseline and a with/without-hardware split.
A lock file (results/phase3/TEST_EVALUATED.json) records the evaluation; running again needs --allow-rerun, and then
BOTH results must be reported (a second look at the test set is a form of tuning on it).

Outputs: results/phase3/test_scores.csv (per-image test scores: Phase 4-5 compare against these),
         results/tables/phase3_test_{overall_region,confusion,ci,hardware,baseline}.csv, results/phase3/test_run.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cv_protocol import tune_cv
from evaluate import (REGIONS, bootstrap_ci, format_ci_table, metrics_by_group, never_fracture_metrics,
                      threshold_metrics)
from expt_log import log_run
from feature_utils import load_features
from model_zoo import SPECS, build, default_threshold, grid_for, scores
from paths import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--imbalance", required=True, choices=["none", "balanced", "smote"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", action="store_true", help="also save the fitted model (joblib; can be large)")
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    args = ap.parse_args()

    lock = args.out_dir / "TEST_EVALUATED.json"
    if lock.exists() and not args.allow_rerun:
        raise SystemExit(f"The test set was already evaluated ({lock.read_text()[:300]}...).\n"
                         "Do not look at it again to choose between models. If this is a deliberate re-run "
                         "(e.g. a crash), pass --allow-rerun and report both results.")
    t0 = time.time()
    d = load_features(args.tag) if args.tag else load_features()
    trm, tem = d["split"] == "train", d["split"] == "test"
    X, y, fold, w = d["X"][trm], d["y"][trm].astype(int), d["fold"][trm], d["col_weights"]
    assert len(y) == 3266 and y.sum() == 576 and tem.sum() == 817 and d["y"][tem].sum() == 143, "not the frozen split"
    spec, grid = SPECS[args.model], grid_for(args.model, args.fast)
    print(f"Final model: {spec.label}, imbalance={args.imbalance}", flush=True)

    # 1-2: settings and threshold from TRAINING data only
    print("Step 1-2: tuning on the 5 frozen folds (training images only)...", flush=True)
    tuned = tune_cv(X, y, fold, args.model, args.imbalance, w, grid=grid, seed=args.seed, n_jobs=args.n_jobs)
    params, thr_tuned, thr_def = tuned["params"], tuned["thr_tuned"], default_threshold(args.model)
    print(f"  best params {params}  CV PR-AUC {tuned['cv_pr_auc']:.3f} (optimistic: tuned on these folds)\n"
          f"  F2 threshold {thr_tuned:.4f} (default {thr_def}); out-of-fold F2 there {tuned['f2_at_thr']:.3f}", flush=True)

    # 3: refit on all training images and score the test images, once
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build(args.model, args.imbalance, params, w, args.seed).fit(X, y)
        s_test = scores(pipe, d["X"][tem])
    yt, reg = d["y"][tem].astype(int), d["region"][tem]
    meta_t = d["meta"][tem].reset_index(drop=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(dict(model=args.model, imbalance=args.imbalance, params=params, thr_tuned=thr_tuned,
                                    date=pd.Timestamp.now().isoformat(timespec="seconds"))))
    out = pd.DataFrame(dict(image_id=d["image_id"][tem], y=yt, region=reg, score=s_test,
                            hardware=meta_t["hardware"].to_numpy()))
    out.to_csv(args.out_dir / "test_scores.csv", index=False)

    # 4: metrics
    tables = args.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    res = []
    for op, thr in (("tuned", thr_tuned), ("default", thr_def)):
        res.append(metrics_by_group(yt, s_test, reg, thr).assign(threshold=op, thr_value=thr))
    res = pd.concat(res)
    res.round(4).to_csv(tables / "phase3_test_overall_region.csv", index=False)
    cm = res[["threshold", "group", "n", "n_pos", "tn", "fp", "fn", "tp"]]
    cm.to_csv(tables / "phase3_test_confusion.csv", index=False)
    ci_all = []
    for op, thr in (("tuned", thr_tuned), ("default", thr_def)):
        ci_all.append(bootstrap_ci(yt, s_test, thr, reg, n_boot=args.n_boot, seed=args.seed).assign(threshold=op))
    ci_all = pd.concat(ci_all)
    ci_all.round(4).to_csv(tables / "phase3_test_ci.csv", index=False)
    base = pd.DataFrame([dict(model="never fracture", **never_fracture_metrics(yt))])
    base.round(4).to_csv(tables / "phase3_test_baseline.csv", index=False)

    # hardware split (97 of 99 hardware images are fractured: implants could act as a shortcut)
    hw_rows = []
    for h, name in ((0, "no hardware"), (1, "hardware")):
        m = out.hardware.to_numpy() == h
        r = threshold_metrics(yt[m], s_test[m], thr_tuned)
        hw_rows.append(dict(subset=name, n=int(m.sum()), n_pos=int(yt[m].sum()), tp=r["tp"], fn=r["fn"], fp=r["fp"],
                            tn=r["tn"], recall=r["recall"], specificity=r["specificity"]))
    hw = pd.DataFrame(hw_rows)
    hw.round(4).to_csv(tables / "phase3_test_hardware.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    for op in ("tuned", "default"):
        thr = thr_tuned if op == "tuned" else thr_def
        print(f"\n=== TEST SET, {op} threshold ({thr:.4f}) : estimate [95% bootstrap CI], {args.n_boot} resamples ===")
        print(format_ci_table(ci_all[ci_all.threshold == op]).to_string())
        print(res[res.threshold == op][["group", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print("\nNever-fracture baseline on the test set:")
    print(base[["accuracy", "recall", "precision", "f2", "roc_auc", "pr_auc"]].round(3).to_string(index=False))
    print("\nWith / without surgical hardware (tuned threshold):")
    print(hw.round(3).to_string(index=False))

    run = dict(model=args.model, imbalance=args.imbalance, params=params, thr_tuned=thr_tuned, thr_default=thr_def,
               cv_pr_auc=tuned["cv_pr_auc"], seed=args.seed, grid={k: [str(v) for v in vs] for k, vs in grid.items()},
               seconds=time.time() - t0, date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "test_run.json").write_text(json.dumps(run, indent=1, default=str))
    if args.save_model:
        import joblib
        joblib.dump(pipe, args.out_dir / "final_model.joblib")

    if not args.no_log:
        ov = res[(res.threshold == "tuned") & (res.group == "overall")].iloc[0]
        ovd = res[(res.threshold == "default") & (res.group == "overall")].iloc[0]
        co = ci_all[(ci_all.threshold == "tuned") & (ci_all.group == "overall")].set_index("metric")
        txt = (f"TEST tuned thr={thr_tuned:.4f}: recall={ov.recall:.3f} precision={ov.precision:.3f} F1={ov.f1:.3f} "
               f"F2={ov.f2:.3f} spec={ov.specificity:.3f} ROC-AUC={ov.roc_auc:.3f} "
               f"PR-AUC={ov.pr_auc:.3f} [{co.loc['pr_auc', 'lo']:.3f}, {co.loc['pr_auc', 'hi']:.3f}]; "
               f"TN/FP/FN/TP={int(ov.tn)}/{int(ov.fp)}/{int(ov.fn)}/{int(ov.tp)}; "
               f"default thr: recall={ovd.recall:.3f} F2={ovd.f2:.3f}; per region in phase3_test_overall_region.csv")
        log_run(regions="all (pooled), test on all regions", feature_set=f"all five families [{args.tag or 'default'}]",
                model=spec.label, imbalance_strategy=f"{args.imbalance} + F2 threshold tuning",
                key_hyperparameters=f"{json.dumps(params)}; kbest={spec.kbest}; seed={args.seed}",
                cv_score=f"tuning-fold PR-AUC={tuned['cv_pr_auc']:.3f} (optimistic); F2 at tuned thr (OOF)="
                         f"{tuned['f2_at_thr']:.3f}",
                test_metrics=txt, notes="Phase 3 FINAL test evaluation (test set used once)")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Test scores saved for later phases: {args.out_dir / 'test_scores.csv'}")


if __name__ == "__main__":
    main()
