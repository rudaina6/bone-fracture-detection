"""Phase 7b self-tests (docs/phase7b_improvement.md, "Gates and self-tests"). Run before the CV:

    python tests/test_phase7b.py         # about 10-15 minutes; one line per check, stops at the first failure

Unit checks run on hand-made or synthetic data. The end-to-end check runs the real pipeline on a small subset of
TRAINING images in a temporary folder (--debug-subset, never logged, never written to results/ or features/): the
Phase 7 debug CV (references S1 / P7), then the whole 7b CV with selection, final hyperparameters, threshold CV and
summary, with an access log and a provenance audit (leakage gates 2-4); then a mock final test where other training
images play the test set (B6 refusal, lock before any "test" row is read, re-run refusal, T1 mismatch). No test image
is ever loaded. Real-data gates: `python src/phase7b_cascade.py --gates`.
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import phase7_detect as D                                                   # noqa: E402
import phase7_windows as W                                                  # noqa: E402
import phase7b_cascade as C                                                 # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def expect_exit(fn, what):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(f"{what} did not stop")


@check
def context_ring_hand_cases():
    cy, cx = 12, 14
    H = np.zeros((cy - 1, cx - 1, 36), np.float32)
    H[..., 0] = np.arange(cy - 1)[:, None] * 100 + np.arange(cx - 1)[None, :]     # block (r, c) -> 100 r + c
    L = np.zeros((cy, cx, 10), np.uint8)
    L[..., 3] = 64                                                                 # every cell: code 3 only
    S = np.zeros((cy, cx, 2))
    S[..., 0], S[..., 1] = 64 * 0.5, 64 * 0.25                                     # constant intensity 0.5
    S[4:8, 4:8, 0], S[4:8, 4:8, 1] = 64 * 1.0, 64 * 1.0                            # the window at (4, 4) is brighter
    M = dict(hog=H, lbp=L, inten=S)
    f = C._type_context(M, np.array([4, 0]), np.array([4, 0]))
    top, bottom, left, right = (f[:, 36 * q] for q in range(4))
    assert np.allclose(top[0], np.mean([200 + c for c in range(2, 9)]))           # row j-2 = 2, cols 2..8
    assert np.allclose(bottom[0], np.mean([800 + c for c in range(2, 9)]))        # row j+4 = 8
    assert np.allclose(left[0], np.mean([r * 100 + 2 for r in range(4, 7)]))      # col i-2 = 2, rows 4..6
    assert np.allclose(right[0], np.mean([r * 100 + 8 for r in range(4, 7)]))     # col i+4 = 8
    assert top[1] == 0 and left[1] == 0                                           # window at the corner: no top / left
    assert np.allclose(bottom[1], np.mean([400 + c for c in range(0, 5)]))        # row 4, cols -2..4 -> 0..4
    lbp = f[:, 144:184].reshape(2, 4, 10)
    assert np.allclose(lbp[0, :, 3], 1)                                            # sides: top, bottom, left, right
    assert np.allclose(lbp[1, [0, 2]], 0) and np.allclose(lbp[1, [1, 3], 3], 1)
    mean, std, diff, cov = f[0, 184:188]
    assert np.isclose(mean, 0.5) and np.isclose(std, 0.0, atol=1e-6) and np.isclose(diff, 0.5) and cov == 1.0
    assert np.isclose(f[1, 187], (36 - 16) / 48)                                  # corner: 20 ring cells inside


@check
def context_equals_naive_on_synthetic_image():
    rng = np.random.default_rng(0)
    import cv2
    img = cv2.GaussianBlur(rng.random((448, 448)).astype(np.float32), (0, 0), 3)
    img = (img - img.min()) / (img.max() - img.min())
    content = (0, 40, 448, 400)
    maps, idx = W.image_maps(img), W.window_index(content)
    B, Tt = W.index_boxes(idx)
    sizes = np.cumsum([0] + [len(j) for j, _, _ in idx])
    sel = np.r_[[sizes[t] for t in range(9)], [sizes[t + 1] - 1 for t in range(9)], rng.choice(len(B), 30, replace=False)]
    X = C.context_features(maps, idx, sel)
    worst = 0.0
    for row, s in zip(X, sel):
        t = int(Tt[s])
        jj, ii, _ = idx[t]
        ref = C.naive_context(img, *W.WINDOW_TYPES[t], int(ii[s - sizes[t]]), int(jj[s - sizes[t]]))
        worst = max(worst, float(np.abs(row - ref).max()))
    assert worst <= 1e-5, worst


@check
def weighted_ap_equals_duplicated_rows():
    rng = np.random.default_rng(1)
    s, y = rng.random(60), (rng.random(60) < 0.3).astype(int)
    w = np.where(y == 1, 1, rng.integers(1, 5, 60))
    a = C.weighted_ap(s, y, w.astype(float))
    rep = np.repeat(np.arange(60), w)
    b = D.average_precision(s[rep], y[rep], float(y[rep].sum()))
    assert np.isclose(a, b), (a, b)


@check
def calibration_monotone_per_type():
    rng = np.random.default_rng(2)
    n = 4000
    tt = rng.integers(0, 9, n)
    y = (rng.random(n) < 0.05 + 0.02 * tt).astype(int)
    s = np.clip(0.3 * y + 0.5 * rng.random(n) + 0.05 * tt / 9, 1e-3, 1 - 1e-3)
    cal = C.fit_calibration(s, y, np.ones(n), tt)
    grid = np.linspace(0.01, 0.99, 50)
    for t in range(9):
        p = C.apply_calibration(cal, grid, np.full(50, t))
        assert (np.diff(p) >= -1e-12).all(), t
    assert set(cal) == {"all"} | {str(t) for t in range(9)}


@check
def inner_parallel_equals_sequential():
    rng = np.random.default_rng(3)
    X = rng.random((600, 40)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.random(600) > 0.9).astype(int)
    sets = dict(train=dict(X=X, y=y, fold=np.arange(600) % 4), val={j: dict(X=X[:50]) for j in range(4)})
    cols = np.arange(40)
    for name, cfgs in (("hgb", C.GRIDS["fallback"]["hgb"]), ("rf", C.GRIDS["fallback"]["rf"][:1])):
        got = C.inner_predictions(name, cfgs, sets, cols, 42)
        for (ci, j), p in got.items():
            m = sets["train"]["fold"] != j
            ref = C.predict(C.fit_clf(name, cfgs[ci], X[m], y[m], 42, n_jobs=1, threads=1), X[:50])
            assert np.abs(p - ref).max() == 0, (name, ci, j)


@check
def plan_and_sources():
    plan = C.stage1_plan(range(5))
    assert len(plan) == 15
    for k in [None] + list(range(5)):
        for j in range(5):
            if j == k:
                continue
            tr, scored = plan[C.proposal_source(j, k)]
            assert j not in tr and j in scored and (k is None or k not in tr)
    assert C.proposal_source(3, 3) == "not3" and C.proposal_source(1, 4) == C.proposal_source(4, 1) == "not14"


@check
def end_to_end_debug_cv_and_mock_final_test():
    import phase7b_final_test as F
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        d = {x: tmp / x for x in ("p7", "p7tables", "p7figs", "cache", "work", "out", "tables", "figs", "p3", "p7mock")}
        for x in d.values():
            x.mkdir()
        shutil.copy(REPO / "results" / "tables" / "phase7_reachability.csv", d["p7tables"])
        shutil.copy(REPO / "results" / "tables" / "phase7_reachability.csv", d["tables"])
        c7 = ["--debug-subset", "90", "--out-dir", str(d["p7"]), "--tables-dir", str(d["p7tables"]), "--figures-dir",
              str(d["p7figs"]), "--cache-dir", str(d["cache"]), "--n-jobs", "4", "--n-boot", "30"]
        D.main(c7 + ["--cache", "--patches"])
        D.main(c7 + ["--cv-fold", "0", "1", "2", "3", "4", "--final-c"])
        c7b = ["--debug-subset", "90", "--out-dir", str(d["out"]), "--tables-dir", str(d["tables"]), "--figures-dir",
               str(d["figs"]), "--cache-dir", str(d["cache"]), "--work-dir", str(d["work"]), "--p7-dir", str(d["p7"]),
               "--n-jobs", "1", "--n-boot", "30", "--grid", "revised"]
        T = D.train_table(90)
        R0 = D.load_round0(d["cache"], T)
        ids = set(T.image_id)
        log = C.AccessLog()
        with log.active():
            C.main(c7b + ["--stage1"])
            C.main(c7b + ["--cv-fold", "0", "1", "2", "3", "4", "--select", "--final-params"])
            args = C.argparse.Namespace(out_dir=d["out"], p7_dir=d["p7"], work_dir=d["work"], cache_dir=d["cache"],
                                        n_jobs=1, seed=42)
            need, _ = C.threshold_needed(args)
            C.main(c7b + ["--threshold-cv"] + (["0", "1", "2", "3", "4"] if need else ["0"]))
            C.main(c7b + ["--summarize"])
        # L1 (debug): every map and image read was a training image of the subset
        assert log.maps and set(log.maps) <= {Path(i).stem for i in ids}, "a map outside the training subset was read"
        assert set(log.images) <= set(T.file_path), "an image outside the training subset was read"
        # stage 1 outer models = Phase 7 round 0 (debug), proposal recall two ways
        models = C.load_stage1(args)
        p7det, _ = D.load_fold_detections(d["p7"], "cv")
        for k in range(5):
            m = models[C.tag_outer(k)]
            pr = C.load_props(args, C.tag_outer(k))
            for n in pr["n"][:3]:
                maps, geom = W.load_maps(D.map_path(d["cache"], T.image_id[n]))
                idx = W.window_index(geom["content"])
                B, _ = W.index_boxes(idx)
                s = W.maps_scores(maps, idx, m["w"], m["b"])
                keep = W.nms(B, s, D.NMS_IOU, D.KEEP_PER_IMAGE)
                sv = p7det[(p7det.image_id == T.image_id[n]) & (p7det["round"] == 0)].sort_values("rank")
                assert np.abs(sv.score.to_numpy() - s[keep]).max() == 0, "stage 1 differs from Phase 7 round 0"
            assert ((pr["box_iou"] >= 0.5) == ((pr["box_r50"] > 0) & (pr["box_r50"] <= C.K_PROP))).all()
        # L2-L4 (debug): stage-2 sets and the saved provenance of every fold
        fold = T.fold.to_numpy()
        for k in range(5):
            sets = C.stage2_sets(T, R0, args, k, models)
            assert C.check_sets(sets, T, k)
            tr = sets["train"]
            npos = int(tr["y"].sum())
            assert int((tr["y"] == 0).sum()) == C.NEG_RATIO * npos, "not 10 negatives per positive"
            for n in np.unique(tr["img"])[:5]:
                maps, geom = W.load_maps(D.map_path(d["cache"], T.image_id[n]))
                B, _ = W.index_boxes(W.window_index(geom["content"]))
                gt = W.map_boxes(T.boxes[n], geom)
                sel = tr["img"] == n
                if len(gt):
                    best = W.iou_matrix(B[tr["win"][sel]], gt).max(1)
                    assert (best[tr["y"][sel] == 0] < D.NEG_IOU).all() and (best[tr["y"][sel] == 1] >= D.POS_IOU).all()
            info = json.loads((d["out"] / f"cv_fold{k}.json").read_text())
            assert k not in info["stage1_outer"]["train_folds"]
            for j, p in info["provenance"].items():
                if j.isdigit():
                    assert int(j) != k and int(j) not in p["source_train_folds"] and k not in p["source_train_folds"]
            assert k not in info["provenance"]["train_image_folds"]
            det = pd.read_csv(d["out"] / f"cv_fold{k}_detections.csv")
            assert set(det.image_id) <= set(T.image_id[fold == k]) and set(det.variant) == set(C.VARIANTS)
        fp = json.loads((d["out"] / "final_params.json").read_text())
        assert sorted(int(f) for f in fp["provenance"]["train_image_folds"]) == list(range(5))
        cvs = json.loads((d["out"] / "cv_summary.json").read_text())
        assert {"B6", "threshold", "d_vs_S1", "d_vs_P7", "selection"} <= set(cvs) and np.isfinite(cvs["threshold"])
        for f in ("phase7b_cv_variants.csv", "phase7b_cv_localizer.csv", "phase7b_cv_image_level.csv"):
            assert (d["tables"] / f).exists(), f
        assert (d["figs"] / "phase7b_cv_variants.png").exists() and (d["figs"] / "phase7b_cv_examples.png").exists()
        expect_exit(lambda: C.main(["--debug-subset", "90", "--summarize"]), "a debug run writing into results/")

        # mock final test: other TRAINING images play the test set
        Tall = D.train_table()
        fake = Tall[~Tall.image_id.isin(T.image_id)].sample(12, random_state=0)
        fake_test = pd.DataFrame(dict(image_id=fake.image_id, file_path=fake.file_path, region=fake.region,
                                      fractured=fake.fractured, boxes=fake.boxes)).reset_index(drop=True)
        oof = pd.read_csv(REPO / "results" / "phase3" / "oof_rbfsvm__none.csv")
        oof[oof.image_id.isin(fake.image_id)][["image_id", "y", "region", "score"]].to_csv(d["p3"] / "test_scores.csv",
                                                                                            index=False)
        shutil.copy(d["p7"] / "final_c.json", d["p7mock"])
        rng = np.random.default_rng(0)
        pd.DataFrame([dict(image_id=i, rank=r, score=float(rng.random()), x=100.0, y=100.0, w=40.0, h=40.0, wtype=3)
                      for i in fake.image_id for r in range(3)]).to_csv(d["p7mock"] / "test_detections.csv", index=False)
        lock = d["out"] / "TEST_EVALUATED.json"
        saved = F.test_table

        def guarded():
            assert lock.exists(), "test rows were read before the lock was written"
            return fake_test
        F.test_table = guarded
        try:
            ft = ["--debug-subset", "90", "--out-dir", str(d["out"]), "--tables-dir", str(d["tables"]), "--cache-dir",
                  str(d["cache"]), "--work-dir", str(d["work"]), "--p7-dir", str(d["p7mock"]), "--phase3-dir",
                  str(d["p3"]), "--n-jobs", "1", "--n-boot", "30"]
            summ = d["out"] / "cv_summary.json"
            real = json.loads(summ.read_text())
            summ.write_text(json.dumps(dict(real, B6=dict(real["B6"], passed=False))))
            expect_exit(lambda: F.main(ft), "the final test with B6 failed")
            assert not lock.exists(), "B6 failed but the lock was written"
            summ.write_text(json.dumps(dict(real, B6=dict(real["B6"], passed=True))))
            F.main(ft)
            assert lock.exists() and (d["out"] / "test_detections.csv").exists()
            assert (d["tables"] / "phase7b_test_localizer.csv").exists()
            expect_exit(lambda: F.main(ft), "a second final-test run")
            bad = tmp / "out_bad"
            shutil.copytree(d["out"], bad)
            (bad / "TEST_EVALUATED.json").unlink()
            badp7 = tmp / "p7bad"
            shutil.copytree(d["p7mock"], badp7)
            (badp7 / "final_c.json").write_text(json.dumps(dict(C=-1)))
            expect_exit(lambda: F.main([a if a not in (str(d["out"]), str(d["p7mock"])) else
                                        (str(bad) if a == str(d["out"]) else str(badp7)) for a in ft]),
                        "a T1 mismatch")
            assert not (bad / "TEST_EVALUATED.json").exists(), "T1 failed but the lock was written"
        finally:
            F.test_table = saved


def main():
    t_all = time.time()
    for fn in CHECKS:
        t0 = time.time()
        print(f"... {fn.__name__}", flush=True)
        fn()
        print(f"PASS {fn.__name__} ({time.time() - t0:.1f} s)", flush=True)
    print(f"\nAll {len(CHECKS)} Phase 7b self-tests passed in {time.time() - t_all:.0f} s.")


if __name__ == "__main__":
    main()
