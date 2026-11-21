"""Phase 7 self-tests (docs/phase7_localization.md, "Gates and self-tests"). Run before the detection CV:

    python tests/test_phase7.py          # a few minutes; one line per check, stops at the first failure

Unit checks run on hand-made or synthetic data. The two end-to-end checks run the real pipeline on a small subset of
TRAINING images in a temporary folder (--debug-subset, never logged, never written to results/): a full detection CV
with summary, and a mock final test where some other training images play the test set, checking that nothing
"test" is read before the lock. No test image is ever loaded. The real-data gates (window grid, reachability, box
mapping, dense = naive on 500 training windows, folded scoring) are in `python src/phase7_detect.py --gates`.
"""
import json
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import cv2                                                                  # noqa: E402

import phase7_detect as D                                                   # noqa: E402
import phase7_windows as W                                                  # noqa: E402
from model_zoo import build                                                 # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ----------------------------------------------------------------------------- IoU, NMS, matching, AP

@check
def iou_hand_cases():
    a = np.array([[0, 0, 10, 10]])
    cases = [([0, 0, 10, 10], 1.0), ([20, 20, 5, 5], 0.0), ([5, 0, 10, 10], 50 / 150), ([0, 0, 5, 10], 0.5),
             ([2, 2, 6, 6], 36 / 100), ([10, 0, 10, 10], 0.0)]
    for b, want in cases:
        got = W.iou_matrix(a, np.array([b]))[0, 0]
        assert abs(got - want) < 1e-12, (b, got, want)
    assert W.iou_matrix(np.zeros((0, 4)), a).shape == (0, 1)
    assert W.iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0] == 0.0          # zero area: no division by zero


@check
def nms_hand_cases_and_cv2():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [30, 30, 10, 10], [0, 0, 10, 10.0]])
    scores = np.array([0.9, 0.8, 0.7, 0.95])
    assert list(W.nms(boxes, scores, 0.3)) == [3, 2]                           # 0 and 1 suppressed by 3
    assert list(W.nms(boxes, scores, 0.3, max_keep=1)) == [3]
    assert len(W.nms(np.zeros((0, 4)), np.zeros(0), 0.3)) == 0
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = 60
        b = np.c_[rng.uniform(0, 100, (n, 2)), rng.uniform(5, 30, (n, 2))]
        s = (rng.permutation(n) + 1) / n                  # distinct positive scores (OpenCV needs a threshold >= 0)
        ours = set(W.nms(b, s, 0.3).tolist())
        ref = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), 0.0, 0.3)            # suppresses IoU > 0.3, as ours
        assert ours == set(np.asarray(ref).ravel().tolist()), "NMS disagrees with cv2.dnn.NMSBoxes"


def _naive_ap(scores, tp, n_gt):
    """VOC devkit style all-point AP (independent of D.average_precision)."""
    if n_gt == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tpc = fpc = 0
    rec, prec = [], []
    for i in order:
        tpc += bool(tp[i])
        fpc += not bool(tp[i])
        rec.append(tpc / n_gt)
        prec.append(tpc / (tpc + fpc))
    mrec, mpre = [0.0] + rec + [1.0], [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1) if mrec[i + 1] != mrec[i])


@check
def matching_and_ap_hand_cases():
    gt = np.array([[0, 0, 10, 10], [50, 50, 10, 10.0]])
    # perfect detections -> AP 1
    tp = D.match_flags(gt, [0.9, 0.8], gt, 0.5)
    assert tp.tolist() == [True, True] and D.average_precision([0.9, 0.8], tp, 2) == 1.0
    # duplicate on one box: the lower-scored copy is a false positive
    dets = np.array([[0, 0, 10, 10], [1, 0, 10, 10.0]])
    tp = D.match_flags(dets, [0.9, 0.8], gt[:1], 0.5)
    assert tp.tolist() == [True, False] and D.average_precision([0.9, 0.8], tp, 1) == 1.0
    assert D.average_precision([0.9, 0.8], [False, True], 1) == 0.5             # false positive ranked first
    # no detections, no ground truth
    assert D.average_precision([], [], 3) == 0.0 and np.isnan(D.average_precision([0.5], [False], 0))
    # image without boxes: all its detections are false positives
    assert D.match_flags(dets, [0.9, 0.8], np.zeros((0, 4)), 0.5).tolist() == [False, False]
    # a detection matches the unmatched box with the highest IoU
    two = np.array([[0, 0, 10, 10], [4, 0, 10, 10.0]])
    tp = D.match_flags(np.array([[3, 0, 10, 10], [0, 0, 10, 10.0]]), [0.9, 0.8], two, 0.5)
    assert tp.tolist() == [True, True]
    # random cases against the independent VOC-style implementation
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        s, t, g = np.round(rng.random(n), 2), rng.random(n) < 0.4, int(rng.integers(1, 15))
        t[np.cumsum(t) > g] = False
        assert abs(D.average_precision(s, t, g) - _naive_ap(list(s), list(t), g)) < 1e-12


@check
def weighted_ap_equals_duplicated_images():
    rng = np.random.default_rng(2)
    imgs = [f"i{k}" for k in range(12)]
    gt = {i: rng.uniform(0, 80, (int(rng.integers(0, 3)), 2)) for i in imgs}
    gt = {i: np.c_[v, np.full((len(v), 2), 15.0)] for i, v in gt.items()}
    rows = []
    for i in imgs:
        for r in range(int(rng.integers(1, 5))):
            x, y = rng.uniform(0, 80, 2)
            rows.append(dict(image_id=i, score=float(rng.random()), x=x, y=y, w=15.0, h=15.0))
    det = pd.DataFrame(rows)
    es = D.EvalSet(imgs, det, gt)
    m = rng.multinomial(len(imgs), np.full(len(imgs), 1 / len(imgs))).astype(float)
    dup_imgs, dup_rows, dup_gt = [], [], {}
    for k, i in enumerate(imgs):
        for c in range(int(m[k])):
            nid = f"{i}_{c}"
            dup_imgs.append(nid)
            dup_gt[nid] = gt[i]
            dup_rows += [dict(r, image_id=nid) for r in rows if r["image_id"] == i]
    es_dup = D.EvalSet(dup_imgs, pd.DataFrame(dup_rows, columns=det.columns), dup_gt)
    a, b = es.ap(m), es_dup.ap()
    assert all(abs(a[t] - b[t]) < 1e-12 or (np.isnan(a[t]) and np.isnan(b[t])) for t in a), (a, b)


@check
def f1_threshold_hand_case():
    gt = {"a": np.array([[0, 0, 10, 10.0]]), "b": np.array([[0, 0, 10, 10.0]]), "c": np.zeros((0, 4))}
    det = pd.DataFrame([dict(image_id="a", score=0.9, x=0, y=0, w=10, h=10),
                        dict(image_id="c", score=0.8, x=0, y=0, w=10, h=10),
                        dict(image_id="b", score=0.7, x=0, y=0, w=10, h=10),
                        dict(image_id="c", score=0.1, x=40, y=0, w=10, h=10)])
    thr, f1 = D.f1_threshold(D.EvalSet(["a", "b", "c"], det, gt))
    # F1 at 0.9: 2/3 ; at 0.8: 2*1/(2+2)=0.5 ; at 0.7: 2*2/(3+2)=0.8 ; at 0.1: 2*2/(4+2)=0.667
    assert thr == 0.7 and abs(f1 - 0.8) < 1e-12, (thr, f1)


# ----------------------------------------------------------------------------- features and scoring (synthetic image)

@check
def maps_dense_equals_naive_on_synthetic_image():
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.random((448, 448)).astype(np.float32), (0, 0), 2)
    img = (img - img.min()) / (img.max() - img.min())
    geom = dict(content=(40, 0, 408, 448))
    maps, idx = W.image_maps(img), W.window_index(geom["content"])
    B, Tt = W.index_boxes(idx)
    sel = rng.choice(len(B), 120, replace=False)
    X = W.maps_features(maps, idx, sel)
    sizes = np.cumsum([0] + [len(jj) for jj, _, _ in idx])
    worst = 0.0
    for row, s in zip(X, sel):
        t = int(Tt[s])
        jj, ii, _ = idx[t]
        ref = D.naive_window_features(img, *W.WINDOW_TYPES[t], int(ii[s - sizes[t]]), int(jj[s - sizes[t]]))
        worst = max(worst, float(np.abs(row - ref).max()))
    assert worst <= 1e-5, worst
    # every window type and the content rule: counts per type equal the design grid
    import phase7_window_design as Dd
    design = [int(Dd.content_ok(*Dd.grid(w, h), w, h, geom["content"]).sum()) for w, h in W.WINDOW_TYPES]
    assert design == W.window_counts(geom["content"])


@check
def folded_scoring_on_synthetic_patches():
    rng = np.random.default_rng(4)
    X = rng.random((600, W.N_FEAT)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.random(600) > 0.8).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build("linsvm", "balanced", {"C": 0.1}, W.COL_WEIGHTS).fit(X, y)
    w, b = W.fold_linear(pipe)
    X64 = X.astype(np.float64)
    assert np.abs(X64 @ w + b - pipe.decision_function(X64)).max() <= 1e-6


@check
def round0_sampler():
    rng = np.random.default_rng(5)
    n_img = 30
    T = pd.DataFrame(dict(fold=np.arange(n_img) % 5))
    npos = rng.integers(0, 4, n_img)
    R0 = dict(X_pos=rng.random((npos.sum(), 3)), img_pos=np.repeat(np.arange(n_img), npos),
              X_pool=rng.random((n_img * D.POOL, 3)), img_pool=np.repeat(np.arange(n_img), D.POOL),
              rank_pool=np.tile(np.arange(D.POOL), n_img), n_elig=rng.integers(100, 300, n_img))
    use = T.fold.to_numpy() != 2
    X, y, f, img = D.round0_set(R0, T, use)
    assert y.sum() == npos[use].sum() and (y == 0).sum() == D.NEG_PER_POS * npos[use].sum()
    assert use[img].all() and (f == T.fold.to_numpy()[img]).all() and (f != 2).all()
    X2, y2, f2, img2 = D.round0_set(R0, T, use)
    assert np.array_equal(X, X2) and np.array_equal(img, img2), "sampling is not deterministic (seed 0)"


# ----------------------------------------------------------------------------- end to end on a few TRAINING images

@check
def end_to_end_debug_cv_and_mock_final_test():
    import phase7_final_test as F
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out, tables, figs, cache, p3 = (tmp / x for x in ("phase7", "tables", "figures", "cache", "phase3"))
        for d in (out, tables, figs, cache, p3):
            d.mkdir()
        shutil.copy(REPO / "results" / "tables" / "phase7_reachability.csv", tables)
        common = ["--debug-subset", "90", "--out-dir", str(out), "--tables-dir", str(tables), "--figures-dir",
                  str(figs), "--cache-dir", str(cache), "--n-jobs", "4", "--n-boot", "30"]
        D.main(common + ["--cache", "--patches"])
        D.main(common + ["--cv-fold", "0", "1", "2", "3", "4", "--final-c", "--threshold-cv", "0", "1", "2", "3", "4"])
        D.main(common + ["--summarize"])
        cvs = json.loads((out / "cv_summary.json").read_text())
        assert {"D1", "threshold", "C_per_fold", "final_C", "threshold_source"} <= set(cvs)
        assert np.isfinite(cvs["threshold"]), cvs["threshold_source"]
        for f in ("phase7_cv_detection.csv", "phase7_cv_rounds.csv", "phase7_cv_image_level.csv"):
            assert (tables / f).exists(), f
        assert (figs / "phase7_cv_rounds.png").exists() and (figs / "phase7_cv_examples.png").exists()
        try:
            D.main(["--debug-subset", "90", "--summarize"])          # default output folders must be refused
        except SystemExit:
            pass
        else:
            raise AssertionError("a debug run was allowed to write into results/")

        # mock final test: other TRAINING images play the test set; nothing "test" is read before the lock
        Tsub = D.train_table(90)
        Tall = D.train_table()
        fake = Tall[~Tall.image_id.isin(Tsub.image_id)].sample(15, random_state=0)
        fake_test = pd.DataFrame(dict(image_id=fake.image_id, file_path=fake.file_path, region=fake.region,
                                      fractured=fake.fractured, boxes=fake.boxes)).reset_index(drop=True)
        oof = pd.read_csv(REPO / "results" / "phase3" / "oof_rbfsvm__none.csv")
        oof[oof.image_id.isin(fake.image_id)][["image_id", "y", "region", "score"]].to_csv(p3 / "test_scores.csv",
                                                                                            index=False)
        lock = out / "TEST_EVALUATED.json"
        saved = F.test_table

        def guarded():
            assert lock.exists(), "test rows were read before the lock was written"
            return fake_test
        F.test_table = guarded
        try:
            ft = ["--debug-subset", "90", "--out-dir", str(out), "--tables-dir", str(tables), "--cache-dir",
                  str(cache), "--phase3-dir", str(p3), "--n-jobs", "4", "--n-boot", "30"]
            F.main(ft)
            assert lock.exists() and (out / "test_detections.csv").exists()
            assert (tables / "phase7_test_detection.csv").exists()
            try:
                F.main(ft)
            except SystemExit:
                pass
            else:
                raise AssertionError("a second final-test run was not refused")
            bad = tmp / "phase7_bad"
            shutil.copytree(out, bad)
            (bad / "TEST_EVALUATED.json").unlink()
            fc = json.loads((bad / "final_c.json").read_text())
            (bad / "final_c.json").write_text(json.dumps(dict(fc, C=-1)))
            try:
                F.main([a if a != str(out) else str(bad) for a in ft])
            except SystemExit:
                pass
            else:
                raise AssertionError("a T1 mismatch did not stop the final test")
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
    print(f"\nAll {len(CHECKS)} Phase 7 self-tests passed in {time.time() - t_all:.0f} s.")


if __name__ == "__main__":
    main()
