"""Phase 7b FINAL test (T0-T7 of docs/phase7b_improvement.md). THE TEST SET IS USED ONE MORE TIME, ONLY IF B6 PASSED.

    python src/phase7b_final_test.py
    options: --n-jobs 4  --n-boot 2000  --boot-seed 0  --allow-rerun  --no-log

This is the second use of the test set (Phase 7 used it first): the result is a post-hoc follow-up. No test image or
test box is read before step T4:
  T0  results/phase7b/cv_summary.json must record that B6 passed (else stop before anything is fitted or read)
  T1  stage 1: tune_cv on all training round-0 patches; C must equal results/phase7/final_c.json
  T2  stage 2: positives and 10:1 negatives from the training images' cross-fitted proposals (outer models), the
      selected classifier / features, the final hyperparameters and calibration (results/phase7b/final_params.json)
  T3  the box-score threshold from results/phase7b/cv_summary.json
  T4  the lock results/phase7b/TEST_EVALUATED.json is written
  T5  the 817 test images are scored once (maps computed after the lock; 1,000 proposals; NMS 0.3; at most 10 boxes)
  T6  AP@50, AP@50:95, recall / precision / F1 at the threshold, recall at IoU 0.3, reachability and proposal recall,
      overall and per region, 95% image-bootstrap CIs; paired AP@50 difference against the Phase 7 test detections;
      image-level PR-AUC / ROC-AUC against the Phase 3 test scores; hip and shoulder CI half-widths
Running again needs --allow-rerun, and then BOTH results must be reported.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import phase7_detect as D
import phase7_windows as W
import phase7b_cascade as C
from evaluate import paired_bootstrap
from expt_log import log_run
from paths import REPO_ROOT


def test_table():
    """The test images (read only after the lock)."""
    import phase7_final_test as F7
    return F7.test_table()


def _test_chunk(rows, w1, b1):
    out = []
    for image_id, file_path, boxes in rows:
        img, geom = W.load_image(file_path)
        maps, idx = W.image_maps(img), W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        s1 = W.maps_scores(maps, idx, w1, b1)
        prop = np.argsort(-s1, kind="stable")[:C.K_PROP]
        gt = W.map_boxes(boxes, geom)
        reach = W.iou_matrix(gt, B).max(1) >= D.POS_IOU if len(gt) else np.zeros(0, bool)
        piou = W.iou_matrix(gt, B[prop]).max(1) if len(gt) else np.zeros(0)
        out.append(dict(image_id=image_id, gt=gt, reach=reach, prop_iou=piou, B=B[prop], Tt=Tt[prop],
                        X=C.window_rows(maps, idx, Tt, prop, s1[prop])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--debug-subset", type=int, default=0, help="DEBUG (mock runs only)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase7b")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "features" / "phase7")
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "features" / "phase7b")
    ap.add_argument("--p7-dir", type=Path, default=REPO_ROOT / "results" / "phase7")
    ap.add_argument("--phase3-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    args = ap.parse_args(argv)
    if args.debug_subset:
        args.no_log = True
        if {args.out_dir, args.tables_dir, args.cache_dir, args.work_dir, args.p7_dir} & {
                REPO_ROOT / "results" / "phase7b", REPO_ROOT / "results" / "tables", REPO_ROOT / "features" / "phase7",
                REPO_ROOT / "features" / "phase7b", REPO_ROOT / "results" / "phase7"}:
            ap.error("--debug-subset (mock runs) needs its own folders")
    lock = args.out_dir / "TEST_EVALUATED.json"
    if lock.exists() and not args.allow_rerun:
        raise SystemExit(f"The Phase 7b test set was already evaluated ({lock.read_text()[:300]}...).\n"
                         "Do not look at it again to choose anything. If this is a deliberate re-run (e.g. a crash), "
                         "pass --allow-rerun and report both results.")
    # T0: B6
    cvs = json.loads((args.out_dir / "cv_summary.json").read_text())
    if not cvs.get("B6", {}).get("passed", False):
        raise SystemExit("T0: B6 did not pass (the CV-selected localizer does not beat both stage 1 alone and the "
                         "Phase 7 detector with CIs excluding 0). Phase 7b is reported on CV only; the test set is "
                         "not used. No lock written, nothing fitted, no test image read.")
    fp = json.loads((args.out_dir / "final_params.json").read_text())
    sel, thr = fp["selected"], cvs["threshold"]
    assert sel == cvs["selection"]["selected"], "final_params.json and cv_summary.json disagree on the selection"
    t0 = time.time()
    T = D.train_table(args.debug_subset or None)
    R0 = D.load_round0(args.cache_dir, T)
    # T1: stage 1 on all training images
    m1 = C.fit_stage1(T, R0, sorted(T.fold.unique()), args.seed, args.n_jobs)
    final_c = json.loads((args.p7_dir / "final_c.json").read_text())["C"]
    print(f"T1: stage-1 C={m1['C']} (Phase 7 final_c.json: {final_c})", flush=True)
    if m1["C"] != final_c:
        raise SystemExit("T1 FAILED: stage-1 C differs from results/phase7/final_c.json. No lock written, no test image read.")
    # T2: stage 2 on all training images (cross-fitted proposals of the outer models)
    sets = C.stage2_sets(T, R0, args, None, want_val=False)
    C.check_sets(sets, T, None)
    cols = C.FEAT_COLS[sel["feat"]]
    model = C.fit_clf(sel["clf"], fp["cfg"], sets["train"]["X"][:, cols], sets["train"]["y"], args.seed)
    cal = fp["calibration"] if sel["cal"] else None
    print(f"T2: stage 2 {sel['variant']} {fp['cfg']} on {len(sets['train']['y'])} rows "
          f"({int(sets['train']['y'].sum())} positives)", flush=True)
    # T4: the lock, before any test image is read
    settings = dict(variant=sel["variant"], cfg=fp["cfg"], calibration=cal, stage1_C=m1["C"], threshold=thr,
                    threshold_source=cvs.get("threshold_source"), k_prop=C.K_PROP, neg_ratio=C.NEG_RATIO,
                    nms_iou=D.NMS_IOU, keep_per_image=D.KEEP_PER_IMAGE, n_train_rows=int(len(sets["train"]["y"])),
                    b6=cvs["B6"], date=pd.Timestamp.now().isoformat(timespec="seconds"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(settings, indent=1, default=str))
    print(f"T4: lock written -> {lock}", flush=True)

    # T5: the test images, once
    Te = test_table()
    rows = [(r.image_id, r.file_path, r.boxes) for r in Te.itertuples()]
    dets, gt, reach, piou = [], {}, {}, {}
    for s0 in range(0, len(rows), 64):
        for r in D.run_parallel(_test_chunk, rows[s0:s0 + 64], args.n_jobs, m1["w"], m1["b"]):
            s = C.predict(model, r["X"][:, cols])
            if cal is not None:
                s = C.apply_calibration(cal, s, r["Tt"])
            for rank, q in enumerate(W.nms(r["B"], s, D.NMS_IOU, D.KEEP_PER_IMAGE)):
                dets.append(dict(image_id=r["image_id"], rank=rank, score=float(s[q]), x=r["B"][q, 0], y=r["B"][q, 1],
                                 w=r["B"][q, 2], h=r["B"][q, 3], wtype=int(r["Tt"][q])))
            gt[r["image_id"]], reach[r["image_id"]], piou[r["image_id"]] = r["gt"], r["reach"], r["prop_iou"]
    det = pd.DataFrame(dets)
    det.to_csv(args.out_dir / "test_detections.csv", index=False)

    # T6: metrics (one detector: plain image bootstrap = one "fold")
    Tq = Te.assign(fold=0)
    groups = C.GROUPS
    p7 = pd.read_csv(args.p7_dir / "test_detections.csv", float_precision="round_trip")
    res = C.boot_ap(Tq, gt, {"7b": det, "P7": p7}, groups, args.n_boot, args.boot_seed)
    region = dict(zip(Te.image_id, Te.region))
    evt = {0: {g: D.EvalSet([i for i in Te.image_id if g == "overall" or region[i] == g], det, gt) for g in groups}}
    tm = D._threshold_boot(evt, thr, args.n_boot, args.boot_seed)
    rng = np.random.default_rng(args.boot_seed)
    ids = list(Te.image_id)
    nb = np.array([[len(piou[i]), (piou[i] >= 0.5).sum(), (piou[i] >= 0.3).sum(), reach[i].sum()] for i in ids], float)
    rg = np.array([region[i] for i in ids])
    pdraws = {g: [] for g in groups}
    for _ in range(args.n_boot):
        m = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids))).astype(float)
        for g in groups:
            sel_g = np.ones(len(ids), bool) if g == "overall" else rg == g
            t = (m[sel_g, None] * nb[sel_g]).sum(0)
            pdraws[g].append(t[1:3] / t[0] if t[0] else [np.nan, np.nan])
    rows = []
    for g in groups:
        sg = np.ones(len(ids), bool) if g == "overall" else rg == g
        t = nb[sg].sum(0)
        row = dict(group=g, n_images=int(sg.sum()), n_boxes=int(t[0]), reach_iou50=t[3] / t[0] if t[0] else np.nan,
                   proposal_recall50=t[1] / t[0] if t[0] else np.nan, proposal_recall30=t[2] / t[0] if t[0] else np.nan)
        pdr = np.array(pdraws[g], float)
        row.update(proposal_recall50_lo=D.ci(pdr[:, 0])[0], proposal_recall50_hi=D.ci(pdr[:, 0])[1])
        for key, lab in (("ap50", "ap50"), ("ap50_95", "ap50_95")):
            e, d = res[("7b", g, key)]
            lo, hi = D.ci(d)
            row.update({lab: e, f"{lab}_lo": lo, f"{lab}_hi": hi, f"{lab}_half_width": (hi - lo) / 2})
        for key in ("recall", "precision", "f1", "recall_iou30"):
            e, lo, hi = tm[(g, key)]
            row.update({key: e, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_half_width": (hi - lo) / 2})
        dp = C.paired(res, "7b", "P7", g)
        row.update(p7_ap50=res[("P7", g, "ap50")][0], d_p7=dp["diff"], d_p7_lo=dp["lo"], d_p7_hi=dp["hi"])
        rows.append(row)
    table = pd.DataFrame(rows)
    best = det[det["rank"] == 0].set_index("image_id").score
    p3 = pd.read_csv(args.phase3_dir / "test_scores.csv", float_precision="round_trip")
    j = Te[["image_id", "region", "fractured"]].merge(p3[["image_id", "score"]], on="image_id", validate="one_to_one")
    j["score_det"] = j.image_id.map(best).fillna(-np.inf)
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
            r = pb[(pb.group == g) & (pb.metric == met)]
            if len(r):
                r = r.iloc[0]
                row.update({f"d_{met}": r["diff"], f"d_{met}_lo": r["lo"], f"d_{met}_hi": r["hi"]})
        il.append(row)
    iltab = pd.DataFrame(il)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(args.tables_dir / "phase7b_test_localizer.csv", index=False)
    iltab.round(4).to_csv(args.tables_dir / "phase7b_test_image_level.csv", index=False)
    run = dict(settings, seconds=time.time() - t0, n_test_images=len(Te), n_test_boxes=int(table.n_boxes.iloc[0]),
               post_hoc="second use of the test set (Phase 7 used it first)")
    (args.out_dir / "test_run.json").write_text(json.dumps(run, indent=1, default=str))
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print("\n=== TEST (second use, post-hoc follow-up), Phase 7b localizer ===")
    print(table.round(4).to_string(index=False))
    print(iltab.round(3).to_string(index=False))
    if not args.no_log:
        o = table.set_index("group").loc["overall"]
        log_run(regions="all (pooled), test on all regions", feature_set=f"Phase 7b {sel['feat']} features",
                model=f"cascade {sel['variant']}", imbalance_strategy="balanced; 10:1 proposal negatives",
                key_hyperparameters=f"cfg {fp['cfg']}; threshold {thr}; K {C.K_PROP}",
                cv_score="see docs/phase7b_improvement.md",
                test_metrics=(f"TEST (2nd use) AP@50={o.ap50:.4f} [{o.ap50_lo:.4f}, {o.ap50_hi:.4f}]; "
                              f"vs Phase 7 {o.d_p7:+.4f} [{o.d_p7_lo:+.4f}, {o.d_p7_hi:+.4f}]"),
                notes="Phase 7b FINAL test (post-hoc follow-up, second use of the test set)")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
