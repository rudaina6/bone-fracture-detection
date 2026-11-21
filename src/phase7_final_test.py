"""Phase 7 FINAL test evaluation (T1-T6 of docs/phase7_localization.md). THE TEST SET IS USED ONCE.

    python src/phase7_final_test.py
    options: --n-jobs 4  --n-boot 2000  --boot-seed 0  --allow-rerun  --no-log

Run only after the detection CV results are written into docs/phase7_localization.md and approved (STOP 3). No test
image or test box is read before step T4:
  T1  C: tune_cv on the round-0 patches of all 3,266 training images; must equal results/phase7/final_c.json
  T2  train on all training patches; mine the D1 number of rounds (results/phase7/cv_summary.json) on all training images
  T3  the box-score threshold from results/phase7/cv_summary.json
  T4  the lock results/phase7/TEST_EVALUATED.json (window set, C, rounds, threshold) is written
  T5  the 817 test images are scored once (feature maps computed on the fly; NMS 0.3, at most 10 boxes per image)
  T6  AP@50, AP@50:95, recall / precision / F1 at the threshold and recall at IoU >= 0.3, test-box reachability, overall
      and per region, with 95% bootstrap CIs over images (2,000 resamples, seed 0); the image-level PR-AUC / ROC-AUC of
      the best-window score against the Phase 3 test scores (paired bootstrap); hip and shoulder CI half-widths
Running again needs --allow-rerun, and then BOTH results must be reported.

Outputs: results/phase7/TEST_EVALUATED.json, test_detections.csv, test_run.json,
         results/tables/phase7_test_detection.csv, phase7_test_image_level.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import phase7_detect as D
import phase7_windows as W
from cv_protocol import tune_cv
from evaluate import paired_bootstrap
from expt_log import log_run
from paths import REPO_ROOT


def test_table():
    """The test images (read only after the lock): image_id, file_path, region, boxes (original px)."""
    m = pd.read_csv(REPO_ROOT / "data" / "master.csv").merge(
        pd.read_csv(REPO_ROOT / "results" / "splits.csv")[["image_id", "split"]], on="image_id", validate="one_to_one")
    t = m[m.split == "test"].reset_index(drop=True)
    return pd.DataFrame(dict(image_id=t.image_id, file_path=t.file_path, region=t.region_group,
                             fractured=t.fractured.astype(int), boxes=t.boxes_xywh.map(json.loads)))


def _test_chunk(rows, w, b):
    out = []
    for image_id, file_path, boxes in rows:
        img, geom = W.load_image(file_path)
        maps, idx = W.image_maps(img), W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        s = W.maps_scores(maps, idx, w, b)
        keep = W.nms(B, s, D.NMS_IOU, D.KEEP_PER_IMAGE)
        gt = W.map_boxes(boxes, geom)
        reach = W.iou_matrix(gt, B).max(1) >= D.POS_IOU if len(gt) else np.zeros(0, bool)
        out.append(dict(image_id=image_id, gt=gt, reach=reach,
                        det=[dict(image_id=image_id, rank=r, score=s[k], x=B[k, 0], y=B[k, 1], w=B[k, 2], h=B[k, 3],
                                  wtype=int(Tt[k])) for r, k in enumerate(keep)]))
    return out


def score_test_images(Te, n_jobs, w, b):
    return D.run_parallel(_test_chunk, [(r.image_id, r.file_path, r.boxes) for r in Te.itertuples()], n_jobs, w, b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42, help="model seed")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--debug-subset", type=int, default=0, help="DEBUG (mock runs only): about N training images")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase7")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "features" / "phase7")
    ap.add_argument("--phase3-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    args = ap.parse_args(argv)
    if args.debug_subset:
        args.no_log = True
        if {args.out_dir, args.tables_dir, args.cache_dir} & {REPO_ROOT / "results" / "phase7",
                                                             REPO_ROOT / "results" / "tables",
                                                             REPO_ROOT / "features" / "phase7"}:
            ap.error("--debug-subset (mock runs) needs its own --out-dir, --tables-dir and --cache-dir")
    lock = args.out_dir / "TEST_EVALUATED.json"
    if lock.exists() and not args.allow_rerun:
        raise SystemExit(f"The Phase 7 test set was already evaluated ({lock.read_text()[:300]}...).\n"
                         "Do not look at it again to choose anything. If this is a deliberate re-run (e.g. a crash), "
                         "pass --allow-rerun and report both results.")
    cvs = json.loads((args.out_dir / "cv_summary.json").read_text())
    fc = json.loads((args.out_dir / "final_c.json").read_text())
    thr, rounds = cvs["threshold"], int(cvs["D1"]["final_rounds"])
    if thr is None or not np.isfinite(thr):
        raise SystemExit(f"no box-score threshold in cv_summary.json ({cvs.get('threshold_source')}); finish Stage 3")
    t0 = time.time()
    T = D.train_table(args.debug_subset or None)
    R0 = D.load_round0(args.cache_dir, T)

    # T1-T2: the final detector, from training images only
    X, y, f, _ = D.round0_set(R0, T, np.ones(len(T), bool))
    tc = tune_cv(X, y, f, "linsvm", "balanced", W.COL_WEIGHTS, grid={"C": D.C_GRID}, seed=args.seed, n_jobs=args.n_jobs)
    C = tc["params"]["C"]
    print(f"T1: tune_cv on all training round-0 patches -> C={C} (final_c.json: {fc['C']})", flush=True)
    if C != fc["C"]:
        raise SystemExit("T1 FAILED: C differs from results/phase7/final_c.json. No lock written, no test image read.")
    wb = D.fit_linear(X, y, C, args.seed)
    hard = []
    for r in range(1, rounds + 1):
        Xh, _ = D.mine(T, np.ones(len(T), bool), args, *wb)
        hard.append(int(len(Xh)))
        X, y = np.concatenate([X, Xh]), np.r_[y, np.zeros(len(Xh), int)]
        wb = D.fit_linear(X, y, C, args.seed)
        print(f"T2: mining round {r}: {len(Xh)} hard negatives -> {len(y)} patches", flush=True)

    # T4: the lock, before any test image is read
    settings = dict(window_types=[list(t) for t in W.WINDOW_TYPES], C=C, rounds=rounds, hard_negatives=hard,
                    threshold=thr, threshold_source=cvs.get("threshold_source"), nms_iou=D.NMS_IOU,
                    keep_per_image=D.KEEP_PER_IMAGE, n_patches=int(len(y)),
                    date=pd.Timestamp.now().isoformat(timespec="seconds"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(settings, indent=1, default=str))
    print(f"T4: lock written -> {lock}", flush=True)

    # T5: the test images, once
    Te = test_table()
    res = score_test_images(Te, args.n_jobs, *wb)
    det = pd.DataFrame([d for r in res for d in r["det"]])
    det.to_csv(args.out_dir / "test_detections.csv", index=False)
    gt = {r["image_id"]: r["gt"] for r in res}
    reach = {r["image_id"]: r["reach"] for r in res}
    region = dict(zip(Te.image_id, Te.region))

    # T6: metrics with bootstrap CIs over images (one detector, so a plain image bootstrap)
    groups = ["overall"] + D.REGIONS
    imgs = {g: [i for i in Te.image_id if g == "overall" or region[i] == g] for g in groups}
    es = {0: {g: D.EvalSet(imgs[g], det.assign(round=0), gt) for g in groups}}

    def ap_fn(k, esg, m):
        out = {}
        for g, e in esg.items():
            a = e.ap(m[D._subset_index(esg, g)])
            out[f"{g}|AP0.5"] = a[0.5]
            out[f"{g}|AP50:95"] = float(np.mean([a[t] for t in D.AP_IOUS]))
        return out
    apb = D.boot_folds(es, args.n_boot, args.boot_seed, ap_fn)
    tm = D._threshold_boot(es, thr, args.n_boot, args.boot_seed)
    rows = []
    for g in groups:
        nb = int(sum(len(gt[i]) for i in imgs[g]))
        rv = np.concatenate([reach[i] for i in imgs[g]]) if nb else np.zeros(0)
        row = dict(group=g, n_images=len(imgs[g]), n_boxes=nb, reach_iou50=float(rv.mean()) if nb else np.nan)
        for key, lab in (("AP0.5", "ap50"), ("AP50:95", "ap50_95")):
            e, d = apb[f"{g}|{key}"]
            lo, hi = D.ci(d)
            row.update({lab: e, f"{lab}_lo": lo, f"{lab}_hi": hi, f"{lab}_half_width": (hi - lo) / 2})
        for key in ("recall", "precision", "f1", "recall_iou30"):
            e, lo, hi = tm[(g, key)]
            row.update({key: e, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_half_width": (hi - lo) / 2})
        rows.append(row)
    table = pd.DataFrame(rows)

    best = det[det["rank"] == 0].set_index("image_id").score
    p3 = pd.read_csv(args.phase3_dir / "test_scores.csv", float_precision="round_trip")
    j = Te[["image_id", "region", "fractured"]].merge(p3[["image_id", "score"]], on="image_id", validate="one_to_one")
    j["score_det"] = j.image_id.map(best)
    pb = paired_bootstrap(j.fractured, j.score_det, j.score, 0.0, 0.0, groups=j.region, n_boot=args.n_boot,
                          seed=args.boot_seed, metrics=("pr_auc", "roc_auc"))
    il = []
    for g in groups:
        d = j if g == "overall" else j[j.region == g]
        row = dict(group=g, det_pr_auc=D._fast_ap(d.fractured.to_numpy(), d.score_det.to_numpy()),
                   p3_pr_auc=D._fast_ap(d.fractured.to_numpy(), d.score.to_numpy()),
                   det_roc_auc=D._fast_auc(d.fractured.to_numpy(), d.score_det.to_numpy()),
                   p3_roc_auc=D._fast_auc(d.fractured.to_numpy(), d.score.to_numpy()))
        for met in ("pr_auc", "roc_auc"):
            r = pb[(pb.group == g) & (pb.metric == met)].iloc[0]
            row.update({f"d_{met}": r["diff"], f"d_{met}_lo": r["lo"], f"d_{met}_hi": r["hi"]})
        il.append(row)
    iltab = pd.DataFrame(il)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.tables_dir / "phase7_test_detection.csv", index=False)
    iltab.round(4).to_csv(args.tables_dir / "phase7_test_image_level.csv", index=False)
    run = dict(settings, seconds=time.time() - t0, n_test_images=len(Te), n_test_boxes=int(table.n_boxes.iloc[0]))
    (args.out_dir / "test_run.json").write_text(json.dumps(run, indent=1, default=str))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== TEST, Phase 7 detector (used once) ===")
    print(table.round(3).to_string(index=False))
    print("\n=== TEST, image level: best-window score vs Phase 3 test scores (paired) ===")
    print(iltab.round(3).to_string(index=False))
    if not args.no_log:
        o = table.set_index("group").loc["overall"]
        log_run(regions="all (pooled), test on all regions", feature_set="Phase 7 window features",
                model="linear SVM sliding-window detector", imbalance_strategy="balanced + hard-negative mining",
                key_hyperparameters=f"C={C}; rounds {rounds}; threshold {thr}; windows {list(W.WINDOW_TYPES)}",
                cv_score="see docs/phase7_localization.md (detection CV)",
                test_metrics=(f"TEST AP@50={o.ap50:.3f} [{o.ap50_lo:.3f}, {o.ap50_hi:.3f}]; AP@50:95={o.ap50_95:.3f}; "
                              f"recall={o.recall:.3f} precision={o.precision:.3f} F1={o.f1:.3f} at IoU 0.5; "
                              f"reachability {o.reach_iou50:.3f}"),
                notes="Phase 7 FINAL test evaluation (test set used once)")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
