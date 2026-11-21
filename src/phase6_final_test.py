"""Phase 6 FINAL test evaluation (T1-T6 of docs/phase6_gap_closing.md). THE TEST SET IS USED ONCE.

    python src/phase6_final_test.py --phase6-model R
    python src/phase6_final_test.py --phase6-model kernel --kernel mtrbf_all      # only if a kernel was adopted
    options: --n-jobs 4  --n-boot 2000  --boot-seed 0  --allow-rerun  --no-log

Run only after the Phase 6 CV results and the model chosen by rule S are written into docs/phase6_gap_closing.md and
approved. --phase6-model is G | R | R-shrunk | kernel | kernel+R | kernel+R-shrunk. Test rows are untouched until T5:
  T1  tune_cv of the Phase 3 model (rbfsvm, none) on the 3,266 training images: must give Phase 3's parameters and a
      threshold within 1e-3 of Phase 3's (results/phase3/test_run.json), or the script stops
  T2  R and R-shrunk thresholds from that tune_cv's out-of-fold scores (npos = the region's fractured training images;
      R-shrunk shrinks towards Phase 3's saved threshold, which is the G threshold on test)
  T3  kernel models only: tune_cv of the kernel, refit on all training images, global F2 threshold (and per-region
      thresholds for kernel+R / kernel+R-shrunk) from its tune_cv out-of-fold scores. If tune_cv picks lambda = 0
      the kernel arm is the unified model refitted on this machine, and is reported as that.
  T4  the lock results/phase6/TEST_EVALUATED.json (every setting, every threshold, the chosen model) is written
  T5  G, R and R-shrunk on the saved Phase 3 test scores (results/phase3/test_scores.csv, no refit; G uses Phase 3's
      saved threshold, so G is exactly the Phase 3 test result); the kernel scores the 817 test images once; P4 is
      read from results/phase4/test_scores.csv
  T6  bootstrap CIs (2,000 resamples, seed 0) for every arm; paired differences R-G, R-shrunk-G, K-G (with PR-AUC),
      R-P4, R-shrunk-P4, K-P4; the CI half-width is reported as the smallest detectable effect (hip, shoulder)
Running again needs --allow-rerun, and then BOTH results must be reported. No model or threshold is changed after the
test numbers are seen.

Outputs: results/phase6/TEST_EVALUATED.json, test_scores.csv, test_run.json,
         results/tables/phase6_test_metrics.csv, phase6_test_ci.csv, phase6_test_paired.csv
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cv_protocol import tune_cv
from evaluate import REGIONS, best_threshold, bootstrap_ci, format_ci_table, metrics_by_group, paired_bootstrap
from expt_log import log_run
from feature_utils import load_features
from model_zoo import SPECS, build, grid_for, scores, with_region_tag
from paths import REPO_ROOT
from phase6_thresholds import IMB, MODEL, SHRINK_K, THR_METRICS, fmt_ci, read_csv

MODELS = ("G", "R", "R-shrunk", "kernel", "kernel+R", "kernel+R-shrunk")
ARM_OF = {"G": "G", "R": "R", "R-shrunk": "R-shrunk", "kernel": "K", "kernel+R": "K+R", "kernel+R-shrunk": "K+R-shrunk"}
EXPECTED_SPLIT = (3266, 576, 817, 143)    # train images, train fractured, test images, test fractured; None skips
T1_THR_TOL = 1e-3
PAIRED_METRICS = ("pr_auc", "roc_auc", *THR_METRICS)


def region_thresholds(y, s, region, thr_global, k_shrink=SHRINK_K):
    """R and R-shrunk thresholds from out-of-fold training scores ("other" has none and falls back to the global)."""
    out = {"R": {}, "R-shrunk": {}, "npos": {}, "w": {}}
    for r in REGIONS:
        m = region == r
        npos = int(y[m].sum())
        t = best_threshold(y[m], s[m])[0]
        w = npos / (npos + k_shrink)
        out["R"][r], out["R-shrunk"][r], out["npos"][r], out["w"][r] = t, w * t + (1 - w) * thr_global, npos, w
    return out


def per_image(region, rule, thr_global):
    return np.array([rule.get(r, thr_global) for r in region], dtype=float)


def read_saved_test_scores(path):
    """Saved per-image test scores of Phases 3 / 4 (read only after the lock is written)."""
    return read_csv(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase6-model", required=True, choices=MODELS)
    ap.add_argument("--kernel", choices=["mtrbf_all"], default=None, help="the adopted kernel (rule S1: mtrbf_all)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42, help="model seed")
    ap.add_argument("--boot-seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase6")
    ap.add_argument("--phase3-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    ap.add_argument("--phase4-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    args = ap.parse_args(argv)
    is_kernel = args.phase6_model.startswith("kernel")
    if is_kernel != (args.kernel is not None):
        ap.error("--kernel mtrbf_all is required for the kernel models, and only for them")
    if args.out_dir.resolve() in (args.phase3_dir.resolve(), args.phase4_dir.resolve()):
        raise SystemExit("--out-dir must not be the Phase 3 or Phase 4 results folder")
    lock = args.out_dir / "TEST_EVALUATED.json"
    if lock.exists() and not args.allow_rerun:
        raise SystemExit(f"The Phase 6 test set was already evaluated ({lock.read_text()[:300]}...).\n"
                         "Do not look at it again to choose between models. If this is a deliberate re-run "
                         "(e.g. a crash), pass --allow-rerun and report both results.")
    p3_run = json.loads((args.phase3_dir / "test_run.json").read_text())
    thr_g = float(p3_run["thr_tuned"])

    t0 = time.time()
    d = load_features()
    trm, tem = d["split"] == "train", d["split"] == "test"
    if EXPECTED_SPLIT:
        assert (int(trm.sum()), int(d["y"][trm].sum()), int(tem.sum()), int(d["y"][tem].sum())) == EXPECTED_SPLIT, \
            "not the frozen split"
    Xtr, ytr, ftr, rtr = d["X"][trm], d["y"][trm].astype(int), d["fold"][trm], d["region"][trm]

    # T1: the Phase 3 model's settings, re-derived on this machine from the training images only
    print("T1: tune_cv of the Phase 3 model on the training images ...", flush=True)
    t1 = tune_cv(Xtr, ytr, ftr, MODEL, IMB, d["col_weights"], grid=grid_for(MODEL), seed=args.seed, n_jobs=args.n_jobs)
    t1_ok = t1["params"] == p3_run["params"] and abs(t1["thr_tuned"] - thr_g) <= T1_THR_TOL
    print(f"    params {t1['params']} (Phase 3 {p3_run['params']}); threshold {t1['thr_tuned']:.6f} "
          f"(Phase 3 {thr_g:.6f}, |diff| {abs(t1['thr_tuned'] - thr_g):.1e}, tol {T1_THR_TOL:g}) -> "
          f"{'OK' if t1_ok else 'FAILED'}", flush=True)
    if not t1_ok:
        raise SystemExit("T1 FAILED: this machine does not re-derive the Phase 3 settings. No lock was written and no "
                         "test score was read.")
    # T2: per-region thresholds from the same out-of-fold training scores
    th = region_thresholds(ytr, t1["oof"], rtr, thr_g)
    # T3: the adopted kernel, tuned and refitted on the training images only
    kern, pipe, kth, Xk = None, None, None, None
    if is_kernel:
        mode = SPECS[args.kernel].extra["region_mode"]
        Xk, wk = with_region_tag(d["X"], d["col_weights"], d["region"], mode)
        print(f"T3: tune_cv of {args.kernel} on the training images ...", flush=True)
        kt = tune_cv(Xk[trm], ytr, ftr, args.kernel, IMB, wk, grid=grid_for(args.kernel), seed=args.seed,
                     n_jobs=args.n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = build(args.kernel, IMB, kt["params"], wk, args.seed).fit(Xk[trm], ytr)
        kth = region_thresholds(ytr, kt["oof"], rtr, kt["thr_tuned"])
        kern = dict(model=args.kernel, params=kt["params"], cv_pr_auc=kt["cv_pr_auc"], thr_global=kt["thr_tuned"],
                    R=kth["R"], R_shrunk=kth["R-shrunk"], lam_zero=kt["params"]["lam"] == 0)
        print(f"    params {kt['params']}, global F2 threshold {kt['thr_tuned']:.4f}"
              + ("  NOTE: lambda = 0, the kernel arm is the unified model refitted on this machine" if kern["lam_zero"]
                 else ""), flush=True)

    # T4: the lock, before any saved test score is read or any test image is scored
    settings = dict(phase6_model=args.phase6_model, kernel=args.kernel,
                    T1=dict(params=t1["params"], thr_tuned=t1["thr_tuned"], cv_pr_auc=t1["cv_pr_auc"],
                            phase3_params=p3_run["params"], phase3_thr=thr_g),
                    thresholds=dict(G=thr_g, R=th["R"], R_shrunk=th["R-shrunk"], npos=th["npos"], w=th["w"],
                                    shrink_k=SHRINK_K),
                    kernel_settings=kern, date=pd.Timestamp.now().isoformat(timespec="seconds"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(settings, indent=1, default=str))
    print(f"T4: lock written -> {lock}", flush=True)

    # T5: scores of every arm on the test images
    p3 = read_saved_test_scores(args.phase3_dir / "test_scores.csv")
    te = pd.DataFrame(dict(image_id=d["image_id"][tem], y=d["y"][tem].astype(int), region=d["region"][tem]))
    te = te.merge(p3[["image_id", "y", "region", "score"]].rename(
        columns={"y": "y_p3", "region": "region_p3", "score": "score_unified"}), on="image_id", how="left",
        validate="one_to_one")
    assert te.score_unified.notna().all() and (te.y == te.y_p3).all() and (te.region == te.region_p3).all(), \
        "the saved Phase 3 test scores do not match the test images"
    te = te.drop(columns=["y_p3", "region_p3"]).assign(thr_G=thr_g)
    te["thr_R"] = per_image(te.region, th["R"], thr_g)
    te["thr_R-shrunk"] = per_image(te.region, th["R-shrunk"], thr_g)
    arms = {"G": ("score_unified", "thr_G"), "R": ("score_unified", "thr_R"),
            "R-shrunk": ("score_unified", "thr_R-shrunk")}
    if is_kernel:
        te["score_kernel"] = scores(pipe, Xk[tem])
        te["thr_K"] = kern["thr_global"]
        arms["K"] = ("score_kernel", "thr_K")
        if args.phase6_model != "kernel":
            rule = args.phase6_model.split("+", 1)[1]
            te[f"thr_K+{rule}"] = per_image(te.region, kth[rule], kern["thr_global"])
            arms[f"K+{rule}"] = ("score_kernel", f"thr_K+{rule}")
    p4 = read_saved_test_scores(args.phase4_dir / "test_scores.csv")
    te = te.merge(p4[["image_id", "score", "thr_tuned"]].rename(columns={"score": "score_p4", "thr_tuned": "thr_p4"}),
                  on="image_id", how="left", validate="one_to_one")
    m4 = te.region.isin(REGIONS).to_numpy()
    assert te.score_p4[m4].notna().all(), "some test images of the four regions have no Phase 4 score"
    te.to_csv(args.out_dir / "test_scores.csv", index=False)

    # T6: CIs for every arm and the paired differences
    y, reg = te.y.to_numpy(), te.region.to_numpy()
    res, ci = [], []
    for a, (sc, tc) in arms.items():
        res.append(metrics_by_group(y, te[sc], reg, te[tc].to_numpy()).assign(arm=a))
        ci.append(bootstrap_ci(y, te[sc], te[tc].to_numpy(), reg, n_boot=args.n_boot, seed=args.boot_seed).assign(arm=a))
    t4 = te[m4]
    r4 = metrics_by_group(t4.y, t4.score_p4, t4.region, t4.thr_p4.to_numpy()).assign(arm="P4")
    r4.loc[r4.group == "overall", ["roc_auc", "pr_auc"]] = np.nan          # routed: four models' scales
    res.append(r4.assign(group=r4.group.replace({"overall": "routed (4 regions)"})))
    c4 = bootstrap_ci(t4.y, t4.score_p4, t4.thr_p4.to_numpy(), t4.region, n_boot=args.n_boot,
                      seed=args.boot_seed).assign(arm="P4")
    c4.loc[(c4.group == "overall") & c4.metric.isin(["roc_auc", "pr_auc"]), ["estimate", "lo", "hi"]] = np.nan
    ci.append(c4.assign(group=c4.group.replace({"overall": "routed (4 regions)"})))
    res, ci = pd.concat(res, ignore_index=True), pd.concat(ci, ignore_index=True)

    k_arms = [a for a in arms if a.startswith("K")]
    pairs = [("R", "G"), ("R-shrunk", "G")] + [(k, "G") for k in k_arms] \
        + [("R", "P4"), ("R-shrunk", "P4")] + [(k, "P4") for k in k_arms]
    pt = []
    for a, b in pairs:
        if b == "P4":
            pb = paired_bootstrap(t4.y, t4[arms[a][0]], t4.score_p4, t4[arms[a][1]].to_numpy(), t4.thr_p4.to_numpy(),
                                  groups=t4.region, n_boot=args.n_boot, seed=args.boot_seed, metrics=PAIRED_METRICS)
            pb.loc[(pb.group == "overall") & pb.metric.isin(["pr_auc", "roc_auc"]),
                   ["diff", "lo", "hi", "p_two_sided"]] = np.nan
            pb["group"] = pb.group.replace({"overall": "4 regions"})
        else:
            same_scores = arms[a][0] == arms[b][0]            # a threshold cannot change a ranking
            pb = paired_bootstrap(y, te[arms[a][0]], te[arms[b][0]], te[arms[a][1]].to_numpy(),
                                  te[arms[b][1]].to_numpy(), groups=reg, n_boot=args.n_boot, seed=args.boot_seed,
                                  metrics=THR_METRICS if same_scores else PAIRED_METRICS)
        pt.append(pb.assign(pair=f"{a} - {b}", half_width=(pb.hi - pb.lo) / 2))
    pt = pd.concat(pt, ignore_index=True)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(args.tables_dir / "phase6_test_metrics.csv", index=False)
    ci.round(4).to_csv(args.tables_dir / "phase6_test_ci.csv", index=False)
    pt.round(4).to_csv(args.tables_dir / "phase6_test_paired.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    for a in list(arms) + ["P4"]:
        print(f"\n=== TEST, arm {a}: estimate [95% bootstrap CI], {args.n_boot} resamples ===")
        print(format_ci_table(ci[ci.arm == a].assign(group=ci[ci.arm == a].group.replace(
            {"routed (4 regions)": "overall"}))).to_string())
    print("\n=== Paired differences on the test images (95% CI, thresholds fixed; half-width = smallest detectable "
          "effect) ===")
    pv = pt.assign(text=[fmt_ci(r["diff"], r["lo"], r["hi"]) for r in pt.to_dict("records")])
    print(pv.pivot_table(index=["pair", "group"], columns="metric", values="text", aggfunc="first",
                         sort=False).to_string())
    hw = pt[pt.group.isin(["hip", "shoulder"]) & (pt.metric == "f2")][["pair", "group", "half_width"]]
    print("\nHip / shoulder F2 CI half-widths (a null below these is 'not resolvable with this test set'):")
    print(hw.round(3).to_string(index=False))

    chosen = ARM_OF[args.phase6_model]
    ov = res[(res.arm == chosen) & (res.group == "overall")].iloc[0]
    run = dict(settings, seconds=time.time() - t0, chosen_arm=chosen,
               chosen_overall={k: float(ov[k]) for k in ("recall", "precision", "specificity", "f2", "pr_auc")})
    (args.out_dir / "test_run.json").write_text(json.dumps(run, indent=1, default=str))
    if not args.no_log:
        co = ci[(ci.arm == chosen) & (ci.group == "overall")].set_index("metric")
        log_run(regions="all (pooled), test on all regions", feature_set="all five families [default]",
                model=f"Phase 6 model: {args.phase6_model}" + (f" ({args.kernel} {kern['params']})" if is_kernel else
                                                                " (unified RBF SVM C=10 gamma=0.05)"),
                imbalance_strategy=f"{IMB} + F2 thresholds ({args.phase6_model})",
                key_hyperparameters=f"thresholds: {json.dumps(settings['thresholds'], default=str)[:300]}",
                cv_score="see docs/phase6_gap_closing.md (CV results, rule S)",
                test_metrics=(f"TEST {chosen}: recall={ov.recall:.3f} precision={ov.precision:.3f} F2={ov.f2:.3f} "
                              f"[{co.loc['f2', 'lo']:.3f}, {co.loc['f2', 'hi']:.3f}] spec={ov.specificity:.3f} "
                              f"TN/FP/FN/TP={int(ov.tn)}/{int(ov.fp)}/{int(ov.fn)}/{int(ov.tp)}; all arms in "
                              f"results/tables/phase6_test_*.csv"),
                notes="Phase 6 FINAL test evaluation (test set used once)")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Phase 6 model {args.phase6_model}: overall F2 {ov.f2:.3f}, "
          f"recall {ov.recall:.3f}, specificity {ov.specificity:.3f}")


if __name__ == "__main__":
    main()
