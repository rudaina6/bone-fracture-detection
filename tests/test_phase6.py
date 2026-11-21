"""Phase 6 self-tests on SYNTHETIC data (docs/phase6_gap_closing.md, "Correctness checks"). Run before any real run:

    python tests/test_phase6.py          # a few minutes; one line per check, stops at the first failure

No real feature, split or result file is read or written. The end-to-end mock runs of phase6_thresholds.py,
phase6_kernel.py and phase6_final_test.py use a synthetic data set in a temporary folder, always with --no-log, and
the real data loaders are replaced inside those modules for the duration of the test only.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sklearn.base import clone                                                    # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score                # noqa: E402

import cv_protocol                                                                # noqa: E402
from cv_protocol import tune_cv                                                   # noqa: E402
from evaluate import REGIONS, best_threshold                                      # noqa: E402
from feature_utils import make_preprocessor                                       # noqa: E402
from model_zoo import (PrecomputedRBFSVC, RegionKernelSVC, build, grid_for, region_tag, scores,  # noqa: E402
                       with_region_tag)
from phase4_regional import _fast_ap, _fast_auc                                   # noqa: E402

PRE_PHASE6_COMMIT = "0922a08"        # src/cv_protocol.py before return_inner was added
SMALL_GRID = {"C": [1, 10], "gamma": [0.2, 0.05]}
REGION_P = {"hand": 0.30, "leg": 0.42, "hip": 0.12, "shoulder": 0.12, "other": 0.04}
BLOCKS = (16, 6, 2)                  # three "families"; block weight 1/sqrt(size) as in feature_utils
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ----------------------------------------------------------------------------- synthetic data

def synthetic(n=900, seed=0, hip_shift=0.0):
    """Shaped like load_features(): separable regions, a hip-specific fracture direction, an 80/20 split and 5 folds
    inside train, both stratified on region x fractured. hip_shift moves the hip cluster away from the rest."""
    rng = np.random.default_rng(seed)
    d = sum(BLOCKS)
    names = np.array(list(REGION_P))
    region = rng.choice(names, size=n, p=list(REGION_P.values()))
    centre = {r: rng.normal(0, 1.5, d) for r in names}
    centre["hip"] = centre["hip"] + hip_shift
    noise = rng.normal(size=(n, d))
    X = np.stack([centre[r] for r in region]) + noise
    w_shared, w_hip = rng.normal(size=d), rng.normal(size=d)
    z = np.where(region == "hip", noise @ w_hip, noise @ w_shared) / np.sqrt(d)
    base = {"hand": -0.6, "leg": -1.5, "hip": -1.0, "shoulder": -1.0, "other": -1.0}
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.5 * z + np.array([base[r] for r in region]))))).astype(int)
    split, fold = np.full(n, "train", dtype=object), np.full(n, -1)
    strata = np.char.add(region.astype(str), y.astype(str))
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_test = int(round(0.2 * len(idx)))
        split[idx[:n_test]] = "test"
        fold[idx[n_test:]] = (np.arange(len(idx) - n_test) + rng.integers(5)) % 5
    w = np.concatenate([np.full(b, b ** -0.5, np.float32) for b in BLOCKS])
    ids = np.array([f"SYN{i:05d}.jpg" for i in range(n)])
    meta = pd.DataFrame(dict(image_id=ids, region_group=region, fractured=y, hardware=0))
    return dict(X=X.astype(np.float32), y=y, fold=fold, region=region, split=split.astype(str), image_id=ids,
                meta=meta, col_weights=w, config=dict(synthetic=True, n=n, seed=seed))


def train_part(d):
    """Like cv_protocol.load_train: training rows only."""
    tr = d["split"] == "train"
    return dict(X=d["X"][tr], y=d["y"][tr].astype(int), fold=d["fold"][tr], region=d["region"][tr],
                image_id=d["image_id"][tr], meta=d["meta"][tr].reset_index(drop=True), col_weights=d["col_weights"],
                config=d["config"])


# ----------------------------------------------------------------------------- model checks

@check
def lam0_bit_identical_through_build():
    d = synthetic()
    tr, te = d["split"] == "train", d["split"] == "test"
    ref = build("rbfsvm", "none", {"C": 10, "gamma": 0.05}, d["col_weights"]).fit(d["X"][tr], d["y"][tr])
    for model, mode in (("mtrbf_all", "all"), ("mtrbf_hip", "hip")):
        Xt, wt = with_region_tag(d["X"], d["col_weights"], d["region"], mode)
        k = build(model, "none", {"C": 10, "gamma": 0.05, "lam": 0}, wt).fit(Xt[tr], d["y"][tr])
        assert np.array_equal(ref[:-1].transform(d["X"][te]), k[:-1].transform(Xt[te])[:, :-1]), \
            f"{model}: the scaler / block weight do not transform the feature columns bit-identically"
        assert np.array_equal(ref[-1].sv_, k[-1].sv_) and np.array_equal(ref[-1].dual_coef_, k[-1].dual_coef_)
        assert np.array_equal(scores(ref, d["X"][te]), scores(k, Xt[te])), f"{model}: lambda = 0 is not bit-identical"


@check
def kernel_psd_and_formula():
    d = synthetic(n=260, seed=1)
    X, region, y = d["X"], d["region"], d["y"]
    lam, gamma = 0.7, 0.05
    rbf = np.exp(-gamma * ((X[:, None, :].astype(np.float64) - X[None, :, :]) ** 2).sum(-1))
    for mode in ("all", "hip"):
        Xt, _ = with_region_tag(X, d["col_weights"], region, mode)
        clf = RegionKernelSVC(C=1, gamma=gamma, lam=lam, mode=mode).fit(Xt, y)
        Z, t = clf._split(Xt)
        K = clf._kernel(Z, t, Z, t).astype(np.float64)
        M = (region[:, None] == region[None, :]) if mode == "all" else \
            ((region == "hip")[:, None] & (region == "hip")[None, :])
        assert np.allclose(K, rbf * (1 + lam * M), rtol=1e-4, atol=1e-6), f"{mode}: K != rbf * (1 + lam * M)"
        ev = np.linalg.eigvalsh((K + K.T) / 2)
        assert ev.min() > -1e-4 * ev.max(), f"{mode}: the kernel matrix is not PSD (min eigenvalue {ev.min():.2e})"


@check
def tags_survive_preprocessing_on_held_out_rows():
    d = synthetic(seed=2)
    tr, te = d["split"] == "train", d["split"] == "test"
    for mode in ("all", "hip"):
        Xt, wt = with_region_tag(d["X"], d["col_weights"], d["region"], mode)
        prep = make_preprocessor(wt).fit(Xt[tr])
        t_tr, t_te = prep.transform(Xt[tr])[:, -1], prep.transform(Xt[te])[:, -1]
        raw_tr, raw_te = region_tag(d["region"][tr], mode), region_tag(d["region"][te], mode)
        vals = {}
        for g in np.unique(raw_tr):
            v = np.unique(t_tr[raw_tr == g])
            assert len(v) == 1, "equal tags did not stay equal"
            vals[g] = v[0]
            assert (t_te[raw_te == g] == v[0]).all(), "a held-out row's tag differs from the training rows' tag"
        assert len(set(vals.values())) == len(vals), "two regions collapsed onto one tag"
        if mode == "hip":
            assert vals[1.0] > vals[0.0], "the hip tag is no longer the larger value"
        # end to end: held-out scores equal a manual computation from the RAW region labels
        lam, gamma = 1.5, 0.05
        pipe = build(f"mtrbf_{mode}", "none", {"C": 10, "gamma": gamma, "lam": lam}, wt).fit(Xt[tr], d["y"][tr])
        clf = pipe[-1]
        Z_te = np.ascontiguousarray(pipe[:-1].transform(Xt[te])[:, :-1], dtype=np.float32)
        inv = {v: g for g, v in vals.items()}
        sv_raw = np.array([inv[t] for t in clf.sv_tag_])
        same = (raw_te[:, None] == sv_raw[None, :]) if mode == "all" else \
            ((raw_te == 1)[:, None] & (sv_raw == 1)[None, :])
        rbf = PrecomputedRBFSVC._rbf(Z_te, clf.sv_, gamma).astype(np.float64)
        manual = (rbf * (1 + lam * same)) @ clf.dual_coef_.astype(np.float64) + clf.intercept_
        s = scores(pipe, Xt[te])
        # float32 accumulation over many support vectors: differences of ~1e-4 on scores of size ~3 are rounding
        assert np.abs(s - manual).max() < 2e-3, f"{mode}: region mapping is wrong ({np.abs(s - manual).max():.1e})"
        wrong = (rbf * (1 + lam * np.roll(same, 1, axis=0))) @ clf.dual_coef_.astype(np.float64) + clf.intercept_
        assert np.abs(s - wrong).max() > 0.05, "the check cannot tell a wrong region mapping from the right one"


@check
def large_lambda_approaches_hip_only_model():
    d = synthetic(n=2000, seed=3, hip_shift=4.0)
    tr, te = d["split"] == "train", d["split"] == "test"
    hip_tr, hip_te = tr & (d["region"] == "hip"), te & (d["region"] == "hip")
    Xt, wt = with_region_tag(d["X"], d["col_weights"], d["region"], "hip")
    Z = make_preprocessor(wt).fit(Xt[tr]).transform(Xt)           # same preprocessing for every model
    C, gamma, big = 1.0, 0.05, 1e4
    hip_only = PrecomputedRBFSVC(C=C * (1 + big), gamma=gamma).fit(Z[hip_tr][:, :-1], d["y"][hip_tr])
    s_hip = hip_only.decision_function(Z[hip_te][:, :-1])
    corr = {lam: np.corrcoef(RegionKernelSVC(C=C, gamma=gamma, lam=lam, mode="hip").fit(Z[tr], d["y"][tr])
                             .decision_function(Z[hip_te]), s_hip)[0, 1] for lam in (0, big)}
    print(f"    corr with the hip-only model on {hip_te.sum()} held-out hip images: lambda 0 -> {corr[0]:.3f}, "
          f"lambda {big:g} -> {corr[big]:.3f}")
    assert corr[big] > 0.95 and corr[big] > corr[0], "a very large lambda does not approach the hip-only model"


@check
def sklearn_clone_works():
    est = RegionKernelSVC(C=3, gamma=0.1, lam=1.0, mode="hip")
    c = clone(est)
    assert c is not est and c.get_params() == est.get_params()
    d = synthetic(n=300, seed=4)
    Xt, wt = with_region_tag(d["X"], d["col_weights"], d["region"], "all")
    pipe = build("mtrbf_all", "none", {"C": 1, "gamma": 0.2, "lam": 0.25}, wt)
    assert np.array_equal(scores(clone(pipe).fit(Xt, d["y"]), Xt), scores(clone(pipe).fit(Xt, d["y"]), Xt))


@check
def region_kernel_rejects_bad_input():
    d = synthetic(n=200, seed=7)
    X, y = d["X"], d["y"]
    Xa, _ = with_region_tag(X, d["col_weights"], d["region"], "all")
    cases = [(RegionKernelSVC(lam=-0.1, mode="all"), Xa, "lam must be"),
             (RegionKernelSVC(lam=1, mode="hip"), Xa, "binary"),
             (RegionKernelSVC(lam=1, mode="hip"), np.c_[X, np.zeros(len(X), np.float32)], "binary"),
             (RegionKernelSVC(lam=1, mode="all"), X, "at most"),
             (RegionKernelSVC(lam=1, mode="both"), Xa, "mode")]
    for est, XX, msg in cases:
        try:
            est.fit(XX, y)
        except ValueError as e:
            assert msg in str(e), f"wrong error for {est}: {e}"
        else:
            raise AssertionError(f"{est} accepted bad input")
    try:
        region_tag(np.array(["hand", "knee"]), "all")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown region was accepted")


@check
def fast_ap_auc_guard():
    rng = np.random.default_rng(6)
    for _ in range(100):
        n = int(rng.integers(20, 200))
        y = (rng.random(n) < 0.3).astype(int)
        if 0 < y.sum() < n:
            s = np.round(rng.normal(size=n) + y, 1)                       # many ties
            assert abs(_fast_ap(y, s) - average_precision_score(y, s)) < 1e-12
            assert abs(_fast_auc(y, s) - roc_auc_score(y, s)) < 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                     # no division-by-zero warning either
        for y in (np.zeros(10, int), np.ones(10, int)):
            assert np.isnan(_fast_ap(y, np.arange(10.0))) and np.isnan(_fast_auc(y, np.arange(10.0)))


# ----------------------------------------------------------------------------- nested_cv(return_inner)

def _same_run(a, b):
    assert np.array_equal(a["oof"], b["oof"]) and np.array_equal(a["thr_tuned"], b["thr_tuned"])
    assert a["thr_default"] == b["thr_default"] and a["configs"] == b["configs"]
    assert json.dumps(a["folds"], default=str) == json.dumps(b["folds"], default=str)


def _pre_phase6_cv_protocol():
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_PHASE6_COMMIT}:src/cv_protocol.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    mod = types.ModuleType("cv_protocol_pre_phase6")
    exec(compile(src, "cv_protocol_pre_phase6.py", "exec"), mod.__dict__)
    return mod


@check
def nested_cv_return_inner():
    t = train_part(synthetic(seed=5))
    args = (t["X"], t["y"], t["fold"], "rbfsvm", "none", t["col_weights"])
    r0 = cv_protocol.nested_cv(*args, grid=SMALL_GRID, n_jobs=1)
    r1 = cv_protocol.nested_cv(*args, grid=SMALL_GRID, n_jobs=1, return_inner=True)
    assert set(r0) == {"oof", "thr_default", "thr_tuned", "folds", "configs", "seconds"}
    assert set(r1) == set(r0) | {"inner"}
    _same_run(r0, r1)
    old = _pre_phase6_cv_protocol()
    if old is None:
        print("    (comparison with the pre-Phase-6 nested_cv skipped: git or the commit is not available)")
    else:
        _same_run(old.nested_cv(*args, grid=SMALL_GRID, n_jobs=1), r0)
    for f in r1["folds"]:
        idx, s = r1["inner"][f["fold"]]
        assert np.array_equal(np.sort(idx), np.flatnonzero(t["fold"] != f["fold"])) and len(set(idx)) == len(idx)
        assert best_threshold(t["y"][idx], s)[0] == f["thr_tuned"], "inner scores do not reproduce thr_tuned"


@check
def threshold_arms():
    import phase6_thresholds as p6t
    rng = np.random.default_rng(8)
    parts = []
    for k in range(5):
        for r in REGION_P:
            y = (rng.random(60) < 0.3).astype(int)
            y[:2] = [0, 1]
            parts.append(pd.DataFrame(dict(outer_fold=k, region=r, y=y,
                                           inner_score=rng.normal(size=60) + y + (0.5 if r == "hand" else 0))))
    inner = pd.concat(parts, ignore_index=True)
    folds = [dict(fold=k, thr_tuned=best_threshold(g.y.to_numpy(), g.inner_score.to_numpy())[0])
             for k, g in inner.groupby("outer_fold")]
    tab = p6t.fold_thresholds(inner, folds).set_index(["fold", "region", "arm"]).thr
    for k, g in inner.groupby("outer_fold"):
        for r in REGIONS:
            gr = g[g.region == r]
            t_r = best_threshold(gr.y.to_numpy(), gr.inner_score.to_numpy())[0]
            w = gr.y.sum() / (gr.y.sum() + 40)
            assert tab[(k, r, "R")] == t_r
            assert np.isclose(tab[(k, r, "R-shrunk")], w * t_r + (1 - w) * folds[k]["thr_tuned"])
    try:
        p6t.fold_thresholds(inner, [dict(f, thr_tuned=f["thr_tuned"] + 1e-9) for f in folds])
    except AssertionError:
        pass
    else:
        raise AssertionError("a thr_tuned that the inner scores do not reproduce was not caught")
    df = pd.DataFrame(dict(fold=[0, 0, 1], region=["hip", "other", "leg"]))
    t = p6t.image_thresholds(df, p6t.fold_thresholds(inner, folds), "R")
    assert t[0] == tab[(0, "hip", "R")] and t[1] == tab[(0, "all", "G")] and t[2] == tab[(1, "leg", "R")]


# ----------------------------------------------------------------------------- end-to-end mock runs

def _fake_phase3_cv(tr, p3):
    r = cv_protocol.nested_cv(tr["X"], tr["y"], tr["fold"], "rbfsvm", "none", tr["col_weights"],
                              grid=grid_for("rbfsvm"), n_jobs=1)
    pd.DataFrame(dict(image_id=tr["image_id"], fold=tr["fold"], y=tr["y"], region=tr["region"], score=r["oof"],
                      thr_default=r["thr_default"], thr_tuned=r["thr_tuned"])).to_csv(p3 / "oof_rbfsvm__none.csv",
                                                                                        index=False)
    (p3 / "run_rbfsvm__none.json").write_text(json.dumps(dict(folds=r["folds"]), default=str))
    return r


def _fake_phase4_cv(tr, r3, p4, rng):
    m = np.isin(tr["region"], REGIONS)
    df = pd.DataFrame(dict(image_id=tr["image_id"], fold=tr["fold"], y=tr["y"], region=tr["region"],
                           score=r3["oof"] + rng.normal(0, 0.3, len(tr["y"]))))[m]
    df["thr_default"] = 0.0
    df["thr_tuned"] = df.groupby(["region", "fold"]).score.transform("median")
    df.to_csv(p4 / "oof_rbfsvm__none.csv", index=False)


def _fake_test_files(d, p3, p4, rng):
    t = train_part(d)
    tc = tune_cv(t["X"], t["y"], t["fold"], "rbfsvm", "none", t["col_weights"], grid=grid_for("rbfsvm"), n_jobs=1)
    te = d["split"] == "test"
    s = scores(build("rbfsvm", "none", tc["params"], t["col_weights"]).fit(t["X"], t["y"]), d["X"][te])
    df = pd.DataFrame(dict(image_id=d["image_id"][te], y=d["y"][te], region=d["region"][te], score=s, hardware=0))
    df.to_csv(p3 / "test_scores.csv", index=False)
    (p3 / "test_run.json").write_text(json.dumps(dict(params=tc["params"], thr_tuned=tc["thr_tuned"])))
    q = df[df.region.isin(REGIONS)].copy()
    q["score_unified"], q["thr_unified"] = q.score, tc["thr_tuned"]
    q["score"] = q.score + rng.normal(0, 0.3, len(q))
    q["thr_tuned"] = q.groupby("region").score.transform("median")
    q.to_csv(p4 / "test_scores.csv", index=False)


def _expect_exit(fn, what):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(f"{what} did not stop the script")


@check
def mock_end_to_end_runs():
    import phase6_final_test as p6f
    import phase6_kernel as p6k
    import phase6_thresholds as p6t
    d = synthetic(n=900, seed=9)
    tr = train_part(d)
    rng = np.random.default_rng(10)
    saved = dict(t_load=p6t.load_train, t_exp=p6t.EXPECTED_TRAIN, k_load=p6k.load_train, k_exp=p6k.EXPECTED_TRAIN,
                 f_load=p6f.load_features, f_exp=p6f.EXPECTED_SPLIT, f_read=p6f.read_saved_test_scores,
                 f_scores=p6f.scores)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p3, p4, out, tables, figs = (tmp / x for x in ("phase3", "phase4", "phase6", "tables", "figures"))
        p3.mkdir()
        p4.mkdir()
        try:
            p6t.load_train = p6k.load_train = lambda tag=None: tr
            p6t.EXPECTED_TRAIN = p6k.EXPECTED_TRAIN = None
            r3 = _fake_phase3_cv(tr, p3)
            _fake_phase4_cv(tr, r3, p4, rng)
            common = ["--out-dir", str(out), "--phase4-dir", str(p4), "--tables-dir", str(tables),
                      "--figures-dir", str(figs), "--no-log", "--n-jobs", "1", "--n-boot", "60"]
            p6t.main(common + ["--phase3-dir", str(p3)])
            g1 = json.loads((out / "gate_G1.json").read_text())
            assert g1["passed"] and g1["max_score_diff"] == 0.0, f"G1 on synthetic data: {g1}"
            v1 = json.loads((out / "lever1_verdicts.json").read_text())
            assert {"L1", "L2", "L4", "L5", "rule_S2_threshold_rule"} <= set(v1)
            p6k.main(common + ["--gate"])
            assert json.loads((out / "gate_G2.json").read_text())["passed"], "G2 failed on synthetic data"
            p6k.main(common + ["--kernels", "mtrbf_all", "mtrbf_hip"])
            p6k.main(common + ["--diagnostic", "--kernels", "mtrbf_all"])
            p6k.main(common + ["--summarize"])
            v2 = json.loads((out / "lever2_verdicts.json").read_text())
            assert set(v2["kernels"]) == {"mtrbf_all", "mtrbf_hip"} and "phase6_model" in v2["rule_S"]
            assert all(k in v2["kernels"]["mtrbf_all"] for k in ("K1", "K2", "K3", "K4", "verdict", "adopted"))
            assert not v2["kernels"]["mtrbf_hip"]["adopted"], "the secondary kernel must never be adopted"
            # rule S3 (fifth arm with its own L1 check) on the kernel's saved inner scores
            runs = p6k.load_runs(out)
            base = p6t.read_csv(out / "oof_baseline_rbfsvm.csv")
            met, pb, chk = p6k.s3_fifth_arm(out, base, runs, "R", 60, 0)
            assert set(met.arm) == {"kernel+R", "kernel"} and "verdict" in chk
            for f in ("phase6_thresholds_metrics.csv", "phase6_thresholds_paired.csv", "phase6_threshold_stability.csv",
                      "phase6_kernel_summary.csv", "phase6_kernel_paired.csv", "phase6_kernel_lambda.csv"):
                assert (tables / f).exists(), f
            assert (figs / "phase6_thresholds.png").exists() and (figs / "phase6_kernel_lambda.png").exists()
            # G2 must FAIL when the baseline differs in a single score by one unit in the last place
            bad = tmp / "phase6_bad"
            shutil.copytree(out, bad)
            b = p6t.read_csv(bad / "oof_baseline_rbfsvm.csv")
            b.loc[0, "score"] = np.nextafter(b.loc[0, "score"], np.inf)
            b.to_csv(bad / "oof_baseline_rbfsvm.csv", index=False)
            assert not p6k.gate_g2(bad)["passed"], "G2 accepted a baseline that differs by one ulp"

            # final test, mock: lock before any test score, refusal of a second run, T1 failure before the lock
            _fake_test_files(d, p3, p4, rng)
            p6f.load_features = lambda *a, **k: d
            p6f.EXPECTED_SPLIT = None
            cur = {}

            def guarded(fn):
                def wrapper(*a, **k):
                    assert cur["lock"].exists(), "test data were touched before the lock was written"
                    return fn(*a, **k)
                return wrapper
            p6f.read_saved_test_scores = guarded(saved["f_read"])
            p6f.scores = guarded(saved["f_scores"])
            ft = ["--phase3-dir", str(p3), "--phase4-dir", str(p4), "--tables-dir", str(tables), "--no-log",
                  "--n-jobs", "1", "--n-boot", "60"]
            cur["lock"] = tmp / "ft1" / "TEST_EVALUATED.json"
            p6f.main(ft + ["--out-dir", str(tmp / "ft1"), "--phase6-model", "R"])
            assert cur["lock"].exists() and (tmp / "ft1" / "test_scores.csv").exists()
            _expect_exit(lambda: p6f.main(ft + ["--out-dir", str(tmp / "ft1"), "--phase6-model", "R"]),
                         "a second run without --allow-rerun")
            cur["lock"] = tmp / "ft2" / "TEST_EVALUATED.json"
            p6f.main(ft + ["--out-dir", str(tmp / "ft2"), "--phase6-model", "kernel+R", "--kernel", "mtrbf_all"])
            ts = pd.read_csv(tmp / "ft2" / "test_scores.csv")
            assert {"score_kernel", "thr_K", "thr_K+R", "score_p4"} <= set(ts.columns)
            p3bad = tmp / "phase3_bad"
            shutil.copytree(p3, p3bad)
            run = json.loads((p3bad / "test_run.json").read_text())
            (p3bad / "test_run.json").write_text(json.dumps(dict(run, thr_tuned=run["thr_tuned"] + 0.01)))
            cur["lock"] = tmp / "ft3" / "TEST_EVALUATED.json"
            _expect_exit(lambda: p6f.main(ft[:0] + ["--phase3-dir", str(p3bad)] + ft[2:]
                                          + ["--out-dir", str(tmp / "ft3"), "--phase6-model", "G"]), "a T1 mismatch")
            assert not cur["lock"].exists(), "T1 failed but the lock was written"
        finally:
            p6t.load_train, p6t.EXPECTED_TRAIN = saved["t_load"], saved["t_exp"]
            p6k.load_train, p6k.EXPECTED_TRAIN = saved["k_load"], saved["k_exp"]
            p6f.load_features, p6f.EXPECTED_SPLIT = saved["f_load"], saved["f_exp"]
            p6f.read_saved_test_scores, p6f.scores = saved["f_read"], saved["f_scores"]


def main():
    t_all = time.time()
    for fn in CHECKS:
        t0 = time.time()
        print(f"... {fn.__name__}", flush=True)
        fn()
        print(f"PASS {fn.__name__} ({time.time() - t0:.1f} s)", flush=True)
    print(f"\nAll {len(CHECKS)} Phase 6 self-tests passed in {time.time() - t_all:.0f} s (synthetic data only).")


if __name__ == "__main__":
    main()
