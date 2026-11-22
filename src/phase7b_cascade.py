"""Phase 7b: two-stage cascade localizer. TRAINING IMAGES ONLY (protocol: docs/phase7b_improvement.md).

    python src/phase7b_cascade.py --gates            # real-data gates incl. leakage gates (before the CV)
    python src/phase7b_cascade.py --timing           # timing, runtime projection, fallback-grid decision
    python src/phase7b_cascade.py --stage1           # stage-1 models (5 outer + 10 pair) and cross-fitted proposals
    python src/phase7b_cascade.py --cv-fold 0        # one outer fold per call (0 .. 4): 4 stage-2 models, 8 variants
    python src/phase7b_cascade.py --select           # B1-B3 on the out-of-fold detections
    python src/phase7b_cascade.py --final-params     # final detector's stage-2 hyperparameters (inner 5-fold)
    python src/phase7b_cascade.py --threshold-cv 0   # only if B4 needs it
    python src/phase7b_cascade.py --summarize        # B4 threshold, metrics, comparisons, B6 verdict, tables, figures
    options: --n-jobs 4  --n-boot 2000  --boot-seed 0  --grid auto|full|fallback  --force  --no-log

Stage 1 = the Phase 7 round-0 linear detector (no mining); its top 1,000 windows per image are the proposals. Every
proposal used to train or tune stage 2 comes from a stage-1 model that did not see its image's fold (cross-fitting).
Stage 2 = HistGradientBoosting or random forest on window (366) + context (188) + window type (9) + stage-1 score (1)
features, trained on every positive window and 10 proposal negatives per positive, tuned by inner CV on a weighted
window-level AP among proposals; optional per-window-type Platt calibration; NMS 0.3, at most 10 boxes per image.

Test images and test boxes are never read here. The test set belongs to src/phase7b_final_test.py and its lock.
Outputs: features/phase7b/ (stage-1 models, proposals; not committed), results/phase7b/*.json / *.csv / *.log,
         results/tables/phase7b_*.csv, results/figures/phase7b_*.png
"""
import argparse
import contextlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import phase7_detect as P
import phase7_windows as W
from cv_protocol import tune_cv
from expt_log import log_run
from paths import REPO_ROOT
from phase4_regional import _fast_ap, _fast_auc, fold_paired_bootstrap
from phase5_viz import C as COLORS, REGION_LABEL, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

K_PROP = 1000                         # proposals per image (fixed by the protocol)
NEG_RATIO = 10                        # stage-2 training negatives per positive (fixed)
VAL_NEG = 100                         # non-positive proposals sampled per image for inner validation / calibration
N_WIN, N_CTX, N_TYPES = W.N_FEAT, 188, len(W.WINDOW_TYPES)
N_ALL = N_WIN + N_CTX + N_TYPES + 1   # 564
FEAT_COLS = {"wc": np.arange(N_ALL), "w": np.r_[np.arange(N_WIN), np.arange(N_WIN + N_CTX, N_ALL)]}
GRIDS = {"full": {"hgb": [dict(max_leaf_nodes=a, min_samples_leaf=b) for a in (15, 63) for b in (20, 100)],
                  "rf": [dict(max_features=a, min_samples_leaf=b) for a in ("sqrt", 0.2) for b in (1, 10)]},
         "fallback": {"hgb": [dict(max_leaf_nodes=a, min_samples_leaf=20) for a in (15, 63)],
                      "rf": [dict(max_features=a, min_samples_leaf=1) for a in ("sqrt", 0.2)]}}
CLFS, FEATS = ("hgb", "rf"), ("wc", "w")
LETTER = {"hgb": "H", "rf": "R"}
VARIANTS = [f"{LETTER[c]}-{f}{cal}" for cal in ("", "-cal") for f in FEATS for c in CLFS]
REGIONS = P.REGIONS
GROUPS = ["overall"] + REGIONS
FALLBACK_HOURS = 6.0
# Thread settings (engineering check, results/phase7b/engineering/threads_check.json; predictions are identical in
# every setting): one HGB fit at a time uses 4 OpenMP threads (8 threads on 4 physical cores were slower); HGB inner
# fits run 4 at a time in loky workers with 2 threads each (4 x 2 = 8 logical cores); RF fits run one at a time on
# all 8 logical cores.
HGB_THREADS, INNER_WORKERS, INNER_THREADS, RF_JOBS = 4, 4, 2, 8


def configure_threads(n_jobs):
    """Stage-2 parallelism from --n-jobs (4 = the setting of the engineering check; 3 = lighter load: 3 HGB inner
    workers x 2 threads, 3 HGB threads for single fits, RF on 6 cores). Fitted models do not depend on it."""
    global HGB_THREADS, INNER_WORKERS, RF_JOBS
    HGB_THREADS, INNER_WORKERS, RF_JOBS = min(4, n_jobs), n_jobs, 2 * n_jobs
    return dict(hgb_single=HGB_THREADS, inner_workers=INNER_WORKERS, inner_threads=INNER_THREADS, rf_jobs=RF_JOBS)


def vname(clf, feat, cal):
    return f"{LETTER[clf]}-{feat}{'-cal' if cal else ''}"


# ----------------------------------------------------------------------------- access log (leakage gate 1)

class AccessLog:
    """Records every image file and every cached map read while active (patched loaders), so a gate can check that
    only training images were touched."""

    def __init__(self):
        self.images, self.maps = [], []

    @contextlib.contextmanager
    def active(self):
        import preprocess
        saved = (preprocess.load_gray, W.load_gray, P.load_gray, W.load_maps, P.W.load_maps)

        def lg(rel, data_dir=None, _f=saved[0]):
            self.images.append(str(rel))
            return _f(rel, data_dir)

        def lm(path, _f=saved[3]):
            self.maps.append(Path(path).stem)
            return _f(path)
        preprocess.load_gray = W.load_gray = P.load_gray = lg
        W.load_maps = lm
        try:
            yield self
        finally:
            preprocess.load_gray, W.load_gray, P.load_gray, W.load_maps, _ = saved


# ----------------------------------------------------------------------------- context features

def _integral(A):
    return np.pad(np.cumsum(np.cumsum(A, 0), 1), ((1, 0), (1, 0), (0, 0)))


def _rect(I, r0, r1, c0, c1, ny, nx):
    """Sums over cells rows r0..r1, cols c0..c1 (inclusive), clipped to the grid; and the number of cells."""
    a0, a1 = np.clip(r0, 0, ny), np.clip(r1 + 1, 0, ny)
    b0, b1 = np.clip(c0, 0, nx), np.clip(c1 + 1, 0, nx)
    a1, b1 = np.maximum(a1, a0), np.maximum(b1, b0)
    return I[a1, b1] - I[a0, b1] - I[a1, b0] + I[a0, b0], (a1 - a0) * (b1 - b0)


def _type_context(M, j, i):
    """Context descriptor (188) of windows starting at cell (row j, col i) of one type's level (see the protocol)."""
    n = len(j)
    H = M["hog"].astype(np.float64)
    by, bx = H.shape[:2]
    Hrow = np.pad(np.cumsum(H, 1), ((0, 0), (1, 0), (0, 0)))
    Hcol = np.pad(np.cumsum(H, 0), ((1, 0), (0, 0), (0, 0)))

    def row_mean(r, c0, c1):
        a, b = np.clip(c0, 0, bx), np.clip(c1 + 1, 0, bx)
        ok = (r >= 0) & (r < by)
        cnt = np.where(ok, np.maximum(b - a, 0), 0)
        rr = np.clip(r, 0, by - 1)
        s = Hrow[rr, b] - Hrow[rr, a]
        return np.where(cnt[:, None] > 0, s / np.maximum(cnt, 1)[:, None], 0.0)

    def col_mean(c, r0, r1):
        a, b = np.clip(r0, 0, by), np.clip(r1 + 1, 0, by)
        ok = (c >= 0) & (c < bx)
        cnt = np.where(ok, np.maximum(b - a, 0), 0)
        cc = np.clip(c, 0, bx - 1)
        s = Hcol[b, cc] - Hcol[a, cc]
        return np.where(cnt[:, None] > 0, s / np.maximum(cnt, 1)[:, None], 0.0)

    hog = [row_mean(j - 2, i - 2, i + 4), row_mean(j + 4, i - 2, i + 4), col_mean(i - 2, j, j + 2),
           col_mean(i + 4, j, j + 2)]
    L, S = M["lbp"].astype(np.float64), M["inten"].astype(np.float64)
    cy, cx = L.shape[:2]
    IL, IS = _integral(L), _integral(S)
    sides = [(j - 2, j - 1, i - 2, i + 5), (j + 4, j + 5, i - 2, i + 5), (j, j + 3, i - 2, i - 1), (j, j + 3, i + 4, i + 5)]
    lbp, sx, sxx, cells = [], np.zeros(n), np.zeros(n), np.zeros(n)
    for r0, r1, c0, c1 in sides:
        s, cnt = _rect(IL, r0, r1, c0, c1, cy, cx)
        tot = s.sum(1, keepdims=True)
        lbp.append(np.divide(s, tot, out=np.zeros_like(s), where=tot > 0))
        si, _ = _rect(IS, r0, r1, c0, c1, cy, cx)
        sx, sxx, cells = sx + si[:, 0], sxx + si[:, 1], cells + cnt
    npix = cells * W.CELL ** 2
    mean = np.divide(sx, npix, out=np.zeros(n), where=npix > 0)
    std = np.sqrt(np.maximum(np.divide(sxx, npix, out=np.zeros(n), where=npix > 0) - mean ** 2, 0.0))
    sw, _ = _rect(IS, j, j + 3, i, i + 3, cy, cx)
    wmean = sw[:, 0] / (16 * W.CELL ** 2)
    out = np.concatenate(hog + lbp + [mean[:, None], std[:, None], (wmean - mean)[:, None], (cells / 48.0)[:, None]], 1)
    return out.astype(np.float32)


def context_features(maps, index, sel):
    """Context descriptors (n, 188) of the windows at global positions `sel`."""
    sel = np.asarray(sel, int)
    sizes = np.cumsum([0] + [len(jj) for jj, _, _ in index])
    out = np.zeros((len(sel), N_CTX), np.float32)
    for t, (M, (jj, ii, _)) in enumerate(zip(maps, index)):
        m = (sel >= sizes[t]) & (sel < sizes[t + 1])
        if m.any():
            loc = sel[m] - sizes[t]
            out[m] = _type_context(M, jj[loc].astype(int), ii[loc].astype(int))
    return out


def naive_context(img, w, h, i, j):
    """Context descriptor of one window computed directly from its level image (independent of the cached maps):
    skimage HOG on a cell-aligned crop with a one-cell margin, LBP codes counted pixel by pixel, intensity from pixels."""
    from skimage.feature import hog as sk_hog, local_binary_pattern
    from feature_extractors import _u8
    level = W.level_image(img, w, h)
    cy, cx = level.shape[0] // 8, level.shape[1] // 8
    by, bx = cy - 1, cx - 1
    ya, xa = max(0, 8 * (j - 3)), max(0, 8 * (i - 3))
    yb, xb = min(level.shape[0], 8 * (j + 7)), min(level.shape[1], 8 * (i + 7))
    Hc = sk_hog(level[ya:yb, xa:xb], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=False)
    Hc = Hc.reshape(Hc.shape[0], Hc.shape[1], 36)

    def blocks_mean(cells):
        v = [Hc[r - ya // 8, c - xa // 8] for r, c in cells if 0 <= r < by and 0 <= c < bx]
        return np.mean(v, 0) if v else np.zeros(36)
    hog = [blocks_mean([(j - 2, c) for c in range(i - 2, i + 5)]), blocks_mean([(j + 4, c) for c in range(i - 2, i + 5)]),
           blocks_mean([(r, i - 2) for r in range(j, j + 3)]), blocks_mean([(r, i + 4) for r in range(j, j + 3)])]
    pasted = np.zeros(level.shape, np.uint8)
    pasted[ya:yb, xa:xb] = _u8(level)[ya:yb, xa:xb]
    codes = local_binary_pattern(pasted, 8, 1, method="uniform").astype(int)
    lv = level.astype(np.float64)
    sides = [(j - 2, j - 1, i - 2, i + 5), (j + 4, j + 5, i - 2, i + 5), (j, j + 3, i - 2, i - 1), (j, j + 3, i + 4, i + 5)]
    lbp, pix = [], []
    for r0, r1, c0, c1 in sides:
        r0, r1, c0, c1 = max(r0, 0), min(r1, cy - 1), max(c0, 0), min(c1, cx - 1)
        if r1 < r0 or c1 < c0:
            lbp.append(np.zeros(10))
            continue
        cc = codes[8 * r0:8 * (r1 + 1), 8 * c0:8 * (c1 + 1)].ravel()
        lbp.append(np.bincount(cc, minlength=10)[:10] / len(cc))
        pix.append(lv[8 * r0:8 * (r1 + 1), 8 * c0:8 * (c1 + 1)].ravel())
    pix = np.concatenate(pix) if pix else np.zeros(0)
    mean = pix.mean() if len(pix) else 0.0
    std = pix.std() if len(pix) else 0.0
    wmean = lv[8 * j:8 * (j + 4), 8 * i:8 * (i + 4)].mean()
    return np.concatenate(hog + lbp + [[mean, std, wmean - mean, len(pix) / 64 / 48.0]])


def type_onehot(tt):
    out = np.zeros((len(tt), N_TYPES), np.float32)
    out[np.arange(len(tt)), np.asarray(tt, int)] = 1
    return out


def window_rows(maps, index, Tt, sel, s1):
    """Stage-2 feature rows (n, 564): window, context, type one-hot, stage-1 score."""
    sel = np.asarray(sel, int)
    return np.concatenate([W.maps_features(maps, index, sel), context_features(maps, index, sel),
                           type_onehot(Tt[sel]), np.asarray(s1, np.float32)[:, None]], 1).astype(np.float32)


# ----------------------------------------------------------------------------- stage 1

def tag_outer(k):
    return f"not{k}"


def tag_pair(j, k):
    a, b = sorted((j, k))
    return f"not{a}{b}"


def stage1_plan(folds=range(5)):
    """Every stage-1 model: tag -> (training folds, folds whose images it scores)."""
    fs = list(folds)
    plan = {tag_outer(k): ([f for f in fs if f != k], [k]) for k in fs}
    for a in fs:
        for b in fs:
            if a < b:
                plan[tag_pair(a, b)] = ([f for f in fs if f not in (a, b)], [a, b])
    return plan


def proposal_source(j, k):
    """Stage-1 model whose proposals on fold-j images are used: inside outer fold k (k=None: final detector)."""
    if k is None or j == k:
        return tag_outer(j)
    return tag_pair(j, k)


def fit_stage1(T, R0, folds, seed, n_jobs):
    use = T.fold.isin(folds).to_numpy()
    X, y, f, img = P.round0_set(R0, T, use)
    assert set(np.unique(f)) <= set(folds), "stage-1 training patches outside the model's folds"
    assert len(set(zip(img, f))) == len(set(img)), "an image's patches sit in two folds"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tc = tune_cv(X, y, f, "linsvm", "balanced", W.COL_WEIGHTS, grid={"C": P.C_GRID}, seed=seed, n_jobs=n_jobs)
    C = tc["params"]["C"]
    w, b = P.fit_linear(X, y, C, seed)
    return dict(folds=sorted(int(x) for x in folds), C=C, w=w, b=b, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                tune_pr_auc={str(c["C"]): a for c, a in zip(tc["configs"], tc["all_pr_auc"])})


def _prop_chunk(rows, cache_dir, w, b, k_prop):
    out = []
    for n, image_id, boxes in rows:
        maps, geom = W.load_maps(P.map_path(cache_dir, image_id))
        idx = W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        s = W.maps_scores(maps, idx, w, b)
        prop = np.argsort(-s, kind="stable")[:k_prop]
        gt = W.map_boxes(boxes, geom)
        if len(gt):
            I = W.iou_matrix(B, gt)
            iou = I[prop].max(1)
            box_iou = I[prop].max(0)
            r50 = [int((s > s[I[:, q] >= 0.5].max()).sum()) + 1 if (I[:, q] >= 0.5).any() else -1 for q in range(len(gt))]
            r30 = [int((s > s[I[:, q] >= 0.3].max()).sum()) + 1 if (I[:, q] >= 0.3).any() else -1 for q in range(len(gt))]
        else:
            iou, box_iou, r50, r30 = np.zeros(len(prop)), np.zeros(0), [], []
        out.append(dict(n=n, prop=prop.astype(np.int32), s1=s[prop].astype(np.float32), iou=iou.astype(np.float32),
                        box_iou=np.asarray(box_iou, np.float32), r50=np.asarray(r50, np.int64),
                        r30=np.asarray(r30, np.int64), n_win=len(s)))
    return out


def make_proposals(T, model, folds_scored, args):
    rows = [(n, r.image_id, r.boxes) for n, r in enumerate(T.itertuples()) if r.fold in folds_scored]
    assert not (set(folds_scored) & set(model["folds"])), "a stage-1 model scored images of its own training folds"
    res = sorted(P.run_parallel(_prop_chunk, rows, args.n_jobs, args.cache_dir, model["w"], model["b"], K_PROP),
                 key=lambda r: r["n"])
    nb = [len(r["box_iou"]) for r in res]
    return dict(n=np.array([r["n"] for r in res]), prop=np.stack([r["prop"] for r in res]),
                s1=np.stack([r["s1"] for r in res]), iou=np.stack([r["iou"] for r in res]),
                box_n=np.repeat([r["n"] for r in res], nb).astype(int),
                box_iou=np.concatenate([r["box_iou"] for r in res]) if sum(nb) else np.zeros(0),
                box_r50=np.concatenate([r["r50"] for r in res]) if sum(nb) else np.zeros(0, int),
                box_r30=np.concatenate([r["r30"] for r in res]) if sum(nb) else np.zeros(0, int),
                n_win=np.array([r["n_win"] for r in res]), model_folds=np.array(model["folds"]),
                folds_scored=np.array(sorted(folds_scored)))


def prop_path(args, tag):
    return args.work_dir / "proposals" / f"{tag}.npz"


def load_props(args, tag):
    with np.load(prop_path(args, tag)) as z:
        return {k: z[k] for k in z.files}


def load_stage1(args):
    with np.load(args.work_dir / "stage1_models.npz", allow_pickle=True) as z:
        return z["models"].item()


def run_stage1(T, R0, args):
    t0 = time.time()
    (args.work_dir / "proposals").mkdir(parents=True, exist_ok=True)
    plan = stage1_plan(sorted(T.fold.unique()))
    models, summ = {}, {}
    for tag, (train_folds, scored) in plan.items():
        m = fit_stage1(T, R0, train_folds, args.seed, args.n_jobs)
        models[tag] = m
        pr = make_proposals(T, m, scored, args)
        np.savez(prop_path(args, tag), **pr)
        summ[tag] = dict(train_folds=m["folds"], scores_folds=scored, C=m["C"], n_pos=m["n_pos"], n_neg=m["n_neg"],
                         n_images=int(len(pr["n"])), tune_pr_auc=m["tune_pr_auc"])
        print(f"  stage 1 {tag}: trained on folds {m['folds']}, C={m['C']}, scored folds {scored} "
              f"({len(pr['n'])} images)", flush=True)
    np.savez(args.work_dir / "stage1_models.npz", models=np.array(models, dtype=object))
    rec = proposal_recall_table(T, args)
    out = dict(models=summ, proposal_recall_heldout=rec, k_prop=K_PROP, seconds=time.time() - t0,
               date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "stage1.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"stage 1 done in {(time.time() - t0) / 60:.1f} min; held-out proposal recall (IoU 0.5 / 0.3): "
          f"{rec['overall']['iou50']:.3f} / {rec['overall']['iou30']:.3f}", flush=True)
    if not args.no_log:
        log_run(regions="all (pooled), training images", feature_set="Phase 7 window features",
                model="stage-1 linear SVM (Phase 7 round 0), 15 models", imbalance_strategy="balanced",
                key_hyperparameters=f"K={K_PROP} proposals; C per model {[m['C'] for m in models.values()]}",
                cv_score=f"held-out proposal recall IoU0.5={rec['overall']['iou50']:.3f}",
                test_metrics="not used (training images only)", notes="Phase 7b stage 1 and cross-fitted proposals")
    return out


def proposal_recall_table(T, args, ks=(250, 500, 1000, 2000)):
    """Held-out proposal recall (outer models, each image scored by the model that did not see its fold)."""
    region = T.region.to_numpy()
    rows = []
    for k in sorted(T.fold.unique()):
        pr = load_props(args, tag_outer(k))
        rows.append(pd.DataFrame(dict(region=region[pr["box_n"]], iou=pr["box_iou"], r50=pr["box_r50"],
                                      r30=pr["box_r30"], fold=k)))
    d = pd.concat(rows)
    out = {}
    for g in GROUPS:
        x = d if g == "overall" else d[d.region == g]
        if not len(x):
            continue
        out[g] = dict(boxes=int(len(x)), iou50=float((x.iou >= 0.5).mean()), iou30=float((x.iou >= 0.3).mean()),
                      **{f"rank_k{K}": float(((x.r50 > 0) & (x.r50 <= K)).mean()) for K in ks})
    return out


# ----------------------------------------------------------------------------- stage-2 sets

def _rows_chunk(rows, cache_dir):
    """rows: (n, image_id, boxes, w, b, neg_props, neg_s1, val_props, val_s1, want_pos). Returns feature rows."""
    out = []
    for n, image_id, boxes, w, b, neg_p, neg_s, val_p, val_s, want_pos in rows:
        maps, geom = W.load_maps(P.map_path(cache_dir, image_id))
        idx = W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        res = dict(n=n)
        if want_pos:
            gt = W.map_boxes(boxes, geom)
            pos = np.flatnonzero(W.iou_matrix(B, gt).max(1) >= P.POS_IOU) if len(gt) else np.zeros(0, int)
            Xw = W.maps_features(maps, idx, pos)
            res["pos"] = window_rows(maps, idx, Tt, pos, Xw.astype(np.float64) @ w + b)
            res["pos_idx"] = pos
        res["neg"] = window_rows(maps, idx, Tt, neg_p, neg_s) if len(neg_p) else np.zeros((0, N_ALL), np.float32)
        res["val"] = window_rows(maps, idx, Tt, val_p, val_s) if len(val_p) else np.zeros((0, N_ALL), np.float32)
        out.append(res)
    return out


def stage2_sets(T, R0, args, k, models=None, want_val=True):
    """Stage-2 training set (positives + 10:1 proposal negatives) and inner validation sets for outer fold k (k=None:
    all training images, for the final detector). Proposals on fold-j images come from proposal_source(j, k).
    Returns dict(train=..., val={j: ...}, provenance=...)."""
    models = models or load_stage1(args)
    fold = T.fold.to_numpy()
    folds = sorted(f for f in np.unique(fold) if f != k)
    n_pos_img = np.bincount(R0["img_pos"], minlength=len(T))
    pool, per_img, prov = [], {}, {}
    for j in folds:
        tag = proposal_source(j, k)
        m = models[tag]
        assert j not in m["folds"], f"proposals of fold {j} from {tag}, which saw fold {j}"
        assert k is None or k not in m["folds"], f"outer fold {k} inside the training of {tag}"
        pr = load_props(args, tag)
        prov[int(j)] = dict(source=tag, source_train_folds=m["folds"])
        for q, n in enumerate(pr["n"]):
            if fold[n] != j:
                continue
            per_img[n] = (pr["prop"][q], pr["s1"][q], pr["iou"][q], m["w"], m["b"])
            neg = np.flatnonzero(pr["iou"][q] < P.NEG_IOU)
            pool.append(np.c_[np.full(len(neg), n), neg])
    train_imgs = np.array(sorted(per_img))
    assert set(fold[train_imgs]) == set(folds) and (k is None or k not in set(fold[train_imgs]))
    pool = np.concatenate(pool)
    n_pos = int(n_pos_img[train_imgs].sum())
    n_neg = min(NEG_RATIO * n_pos, len(pool))
    pick = pool[np.random.default_rng(0).choice(len(pool), size=n_neg, replace=False)]
    neg_by_img = {}
    for n, q in pick:
        neg_by_img.setdefault(int(n), []).append(int(q))
    jobs = []
    for n in train_imgs:
        prop, s1, iou, w, b = per_img[n]
        q = np.array(sorted(neg_by_img.get(int(n), [])), int)
        vp, vs = np.zeros(0, int), np.zeros(0, np.float32)
        if want_val:
            posq = np.flatnonzero(iou >= P.POS_IOU)
            nonq = np.flatnonzero(iou < P.POS_IOU)
            sq = np.sort(np.random.default_rng([1, int(n)]).choice(nonq, size=min(VAL_NEG, len(nonq)), replace=False))
            vq = np.r_[posq, sq]
            vp, vs = prop[vq], s1[vq]
        r = T.iloc[n]
        jobs.append((int(n), r.image_id, r.boxes, w, b, prop[q], s1[q], vp, vs, n_pos_img[n] > 0))
    res = {r["n"]: r for r in P.run_parallel(_rows_chunk, jobs, args.n_jobs, args.cache_dir)}
    Xp = [res[n]["pos"] for n in train_imgs if "pos" in res[n]]
    Xn = [res[n]["neg"] for n in train_imgs]
    ip = np.concatenate([np.full(len(res[n]["pos"]), n) for n in train_imgs if "pos" in res[n]] or [np.zeros(0)])
    ineg = np.concatenate([np.full(len(res[n]["neg"]), n) for n in train_imgs])
    X = np.concatenate(Xp + Xn)
    y = np.r_[np.ones(sum(len(x) for x in Xp), int), np.zeros(sum(len(x) for x in Xn), int)]
    img = np.r_[ip, ineg].astype(int)
    assert len(Xp) == 0 or len(ip) == n_pos, "positive count differs from the round-0 patches"
    win = np.r_[np.concatenate([res[n]["pos_idx"] for n in train_imgs if "pos" in res[n]] or [np.zeros(0)]),
                np.concatenate([jobs[q][5] for q in range(len(jobs))] or [np.zeros(0)])].astype(int)
    train = dict(X=X, y=y, img=img, fold=fold[img], win=win)
    val = {}
    if want_val:
        for j in folds:
            ns = [n for n in train_imgs if fold[n] == j]
            Xv, yv, wv, iv = [], [], [], []
            for n in ns:
                prop, s1, iou, _, _ = per_img[n]
                posq = np.flatnonzero(iou >= P.POS_IOU)
                nonq = np.flatnonzero(iou < P.POS_IOU)
                ns_ = min(VAL_NEG, len(nonq))
                Xv.append(res[n]["val"])
                yv.append(np.r_[np.ones(len(posq), int), np.zeros(ns_, int)])
                wv.append(np.r_[np.ones(len(posq)), np.full(ns_, len(nonq) / max(ns_, 1))])
                iv.append(np.full(len(posq) + ns_, n))
            val[int(j)] = dict(X=np.concatenate(Xv), y=np.concatenate(yv), w=np.concatenate(wv),
                               img=np.concatenate(iv).astype(int))
    prov.update(outer_fold=k, train_folds=folds, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                n_train_images=int(len(train_imgs)), train_image_folds=sorted(int(f) for f in set(fold[img])))
    return dict(train=train, val=val, provenance=prov)


def check_sets(sets, T, k):
    """Leakage assertions on a stage-2 set (gates 2-4): one fold per image, held-out fold absent everywhere."""
    fold = T.fold.to_numpy()
    tr = sets["train"]
    assert (tr["fold"] == fold[tr["img"]]).all()
    assert len(set(zip(tr["img"], tr["fold"]))) == len(set(tr["img"])), "an image in two folds"
    if k is not None:
        assert k not in set(tr["fold"]), "held-out fold in the stage-2 training set"
    for j, v in sets["val"].items():
        assert set(fold[v["img"]]) == {j}, "inner validation set mixes folds"
        assert k is None or k != j
    for j, p in sets["provenance"].items():
        if isinstance(j, int):
            assert j not in p["source_train_folds"], f"fold {j} proposals from a model that saw fold {j}"
    return True


# ----------------------------------------------------------------------------- stage 2: models, tuning, calibration

def make_clf(name, cfg, seed, n_jobs):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    if name == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, early_stopping=False,
                                              l2_regularization=0.0, random_state=seed, **cfg)
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=n_jobs, **cfg)


def balanced_weights(y):
    n, npos = len(y), int(y.sum())
    return np.where(y == 1, n / (2 * max(npos, 1)), n / (2 * max(n - npos, 1)))


def fit_clf(name, cfg, X, y, seed, n_jobs=None, threads=None):
    from threadpoolctl import threadpool_limits
    m = make_clf(name, cfg, seed, RF_JOBS if n_jobs is None else n_jobs)
    if name == "hgb":
        with threadpool_limits(threads or HGB_THREADS, user_api="openmp"):
            m.fit(X, y, sample_weight=balanced_weights(y))
    else:
        m.fit(X, y, sample_weight=balanced_weights(y))
    return m


def _inner_job(name, cfg, X, y, Xv, seed, threads):
    return predict(fit_clf(name, cfg, X, y, seed, n_jobs=1, threads=threads), Xv)


def inner_predictions(name, cfgs, sets, cols, seed):
    """Inner-fold fits of every config: {(config index, inner fold): validation predictions}. HGB fits run
    INNER_WORKERS at a time with INNER_THREADS threads each; RF fits run one at a time on RF_JOBS cores."""
    tr, val = sets["train"], sets["val"]
    jobs = [(ci, j) for ci in range(len(cfgs)) for j in val]

    def job(ci, j):
        m = tr["fold"] != j
        assert j not in set(tr["fold"][m])
        return name, cfgs[ci], tr["X"][m][:, cols], tr["y"][m], val[j]["X"][:, cols], seed
    if name == "hgb":
        out = Parallel(n_jobs=INNER_WORKERS, backend="loky", inner_max_num_threads=INNER_THREADS)(
            delayed(_inner_job)(*job(ci, j), INNER_THREADS) for ci, j in jobs)
    else:
        out = []
        for ci, j in jobs:
            a = job(ci, j)
            out.append(predict(fit_clf(name, a[1], a[2], a[3], seed), a[4]))
    return dict(zip(jobs, out))


def predict(m, X):
    return m.predict_proba(X)[:, 1]


def weighted_ap(s, y, w):
    if y.sum() == 0:
        return np.nan
    return P.average_precision(s, y, float(y.sum()), w)


def tune(name, feat, sets, grid, args):
    """Inner CV over the outer training folds: per config, fit on the other inner folds, weighted AP on fold j's
    validation proposals. Refit the best config on the whole training set. Returns the model, the config, all inner
    scores and the best config's inner out-of-fold validation scores (for calibration)."""
    cols = FEAT_COLS[feat]
    tr, val = sets["train"], sets["val"]
    pred = inner_predictions(name, grid, sets, cols, args.seed)
    scores, oof = [], []
    for ci in range(len(grid)):
        o = {j: pred[(ci, j)] for j in val}
        aps = [weighted_ap(o[j], val[j]["y"], val[j]["w"]) for j in val]
        scores.append(float(np.nanmean(aps)) if np.isfinite(aps).any() else np.nan)
        oof.append(o)
    best = int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0
    model = fit_clf(name, grid[best], tr["X"][:, cols], tr["y"], args.seed)
    return dict(model=model, cfg=grid[best], cfg_index=best, inner_scores=scores, oof=oof[best])


def fit_fixed(name, feat, cfg, sets, args, want_oof=True):
    """Fixed config (threshold CV / final): refit on the whole training set, inner out-of-fold scores if needed."""
    cols = FEAT_COLS[feat]
    tr = sets["train"]
    oof = {}
    if want_oof:
        pred = inner_predictions(name, [cfg], sets, cols, args.seed)
        oof = {j: pred[(0, j)] for j in sets["val"]}
    model = fit_clf(name, cfg, tr["X"][:, cols], tr["y"], args.seed)
    return dict(model=model, cfg=cfg, oof=oof)


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibration(s, y, w, tt):
    """Per-window-type Platt scaling on the logit of the stage-2 score (weighted). Types without both classes use the
    pooled fit. Returns {type: (a, b)} and 'all'."""
    from sklearn.linear_model import LogisticRegression

    def fit(mask):
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(_logit(s[mask])[:, None], y[mask], sample_weight=w[mask])
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    cal = {"all": fit(np.ones(len(y), bool))}
    for t in range(N_TYPES):
        m = tt == t
        cal[str(t)] = fit(m) if (m.any() and 0 < y[m].sum() < m.sum()) else cal["all"]
    return cal


def apply_calibration(cal, s, tt):
    a = np.array([cal[str(t)][0] for t in range(N_TYPES)])[tt]
    b = np.array([cal[str(t)][1] for t in range(N_TYPES)])[tt]
    return 1 / (1 + np.exp(-(a * _logit(s) + b)))


def calibration_from_oof(oof, sets):
    val = sets["val"]
    js = sorted(oof)
    s = np.concatenate([oof[j] for j in js])
    y = np.concatenate([val[j]["y"] for j in js])
    w = np.concatenate([val[j]["w"] for j in js])
    tt = np.concatenate([val[j]["X"][:, N_WIN + N_CTX:N_WIN + N_CTX + N_TYPES].argmax(1) for j in js])
    return fit_calibration(s, y, w, tt)


# ----------------------------------------------------------------------------- held-out scoring

def _heldout_chunk(rows, cache_dir):
    out = []
    for n, image_id, prop, s1 in rows:
        maps, geom = W.load_maps(P.map_path(cache_dir, image_id))
        idx = W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        out.append(dict(n=n, X=window_rows(maps, idx, Tt, prop, s1), B=B[prop], Tt=Tt[prop]))
    return out


def detect_heldout(T, args, k, scorers, batch=64):
    """Fold k's images: proposals from M(not k); scorers: {variant: (model, cols, calibration or None)}.
    Returns detections (variant, image_id, rank, score, x, y, w, h, wtype)."""
    pr = load_props(args, tag_outer(k))
    fold = T.fold.to_numpy()
    rows = [(int(n), T.image_id[n], pr["prop"][q], pr["s1"][q]) for q, n in enumerate(pr["n"]) if fold[n] == k]
    assert len(rows) == int((fold == k).sum())
    dets = []
    for s0 in range(0, len(rows), batch):
        for r in sorted(P.run_parallel(_heldout_chunk, rows[s0:s0 + batch], args.n_jobs, args.cache_dir),
                        key=lambda r: r["n"]):
            cache = {}
            for v, (model, cols, cal) in scorers.items():
                key = id(model)
                if key not in cache:
                    cache[key] = predict(model, r["X"][:, cols])
                s = cache[key] if cal is None else apply_calibration(cal, cache[key], r["Tt"])
                keep = W.nms(r["B"], s, P.NMS_IOU, P.KEEP_PER_IMAGE)
                for rank, q in enumerate(keep):
                    dets.append((v, T.image_id[r["n"]], rank, float(s[q]), *r["B"][q], int(r["Tt"][q])))
    return pd.DataFrame(dets, columns=["variant", "image_id", "rank", "score", "x", "y", "w", "h", "wtype"])


# ----------------------------------------------------------------------------- one outer fold

def grid_choice(args):
    if args.grid in ("full", "fallback"):
        return args.grid
    p = args.out_dir / "timing.json"
    if p.exists():
        return "fallback" if json.loads(p.read_text()).get("use_fallback") else "full"
    return "full"


def run_cv_fold(T, R0, k, args):
    t0 = time.time()
    gname = grid_choice(args)
    grid = GRIDS[gname]
    models = load_stage1(args)
    sets = stage2_sets(T, R0, args, k, models)
    check_sets(sets, T, k)
    info = dict(fold=k, grid=gname, stage1_outer=dict(tag=tag_outer(k), C=models[tag_outer(k)]["C"],
                                                         train_folds=models[tag_outer(k)]["folds"]),
                provenance=sets["provenance"], models={})
    scorers = {}
    for clf in CLFS:
        for feat in FEATS:
            t1 = time.time()
            res = tune(clf, feat, sets, grid[clf], args)
            cal = calibration_from_oof(res["oof"], sets)
            cols = FEAT_COLS[feat]
            scorers[vname(clf, feat, False)] = (res["model"], cols, None)
            scorers[vname(clf, feat, True)] = (res["model"], cols, cal)
            info["models"][f"{clf}-{feat}"] = dict(cfg=res["cfg"], cfg_index=res["cfg_index"],
                                                   inner_scores=res["inner_scores"], calibration=cal,
                                                   seconds=time.time() - t1)
            print(f"    fold {k}: {clf}-{feat} cfg {res['cfg']} inner weighted AP {np.round(res['inner_scores'], 4)} "
                  f"({time.time() - t1:.0f} s)", flush=True)
    det = detect_heldout(T, args, k, scorers)
    info.update(n_heldout_images=int((T.fold == k).sum()), seconds=time.time() - t0,
                date=pd.Timestamp.now().isoformat(timespec="seconds"))
    det.to_csv(args.out_dir / f"cv_fold{k}_detections.csv", index=False)
    (args.out_dir / f"cv_fold{k}.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"fold {k}: {len(det)} detections over {len(VARIANTS)} variants, {info['seconds'] / 60:.1f} min", flush=True)
    if not args.no_log:
        log_run(regions="all (pooled), training images", feature_set="Phase 7b window + context features",
                model="cascade: linear stage 1 + HGB / RF stage 2", imbalance_strategy="balanced; 10:1 proposal negatives",
                key_hyperparameters=f"outer fold {k}; grid {gname}; " + "; ".join(
                    f"{a}: {b['cfg']}" for a, b in info["models"].items()),
                cv_score="see phase7b_cascade.py --select / --summarize", test_metrics="not used (CV only)",
                notes=f"Phase 7b CV fold {k}: out-of-fold detections of 8 variants")
    return info


# ----------------------------------------------------------------------------- evaluation

def load_det(args, prefix="cv"):
    parts, infos = [], {}
    for k in range(5):
        p = args.out_dir / f"{prefix}_fold{k}_detections.csv"
        if not p.exists():
            raise SystemExit(f"{p.name} missing")
        parts.append(pd.read_csv(p, float_precision="round_trip"))
        infos[k] = json.loads((args.out_dir / f"{prefix}_fold{k}.json").read_text())
    return pd.concat(parts, ignore_index=True), infos


def references(args):
    """S1 (Phase 7 round 0) and P7 (Phase 7 round 2) out-of-fold detections."""
    d, _ = P.load_fold_detections(args.p7_dir, "cv")
    return {"S1": d[d["round"] == 0], "P7": d[d["round"] == 2]}


def boot_ap(T, gt, dets, groups, n_boot, seed, extra_groups=None):
    """Fold-mean AP@50 and AP@50:95 of several detection sets with the SAME image multiplicities in every resample
    (paired). dets: {name: DataFrame}. Returns {(name, group, metric): (estimate, draws)}."""
    extra_groups = extra_groups or {}
    rng = np.random.default_rng(seed)
    fold = T.fold.to_numpy()
    region = dict(zip(T.image_id, T.region))
    es = {}
    for k in sorted(set(fold)):
        imgs = list(T.image_id[fold == k])
        for g in groups:
            if g in extra_groups:
                sub = [n for n, i in enumerate(imgs) if i in extra_groups[g]]
            else:
                sub = [n for n, i in enumerate(imgs) if g == "overall" or region[i] == g]
            for name, d in dets.items():
                es[(name, g, k)] = (np.array(sub, int), P.EvalSet([imgs[n] for n in sub], d, gt), len(imgs))
    out = {}
    for b in range(-1, n_boot):
        acc = {}
        ms = {k: (np.ones(len(T.image_id[fold == k])) if b < 0 else
                  rng.multinomial(int((fold == k).sum()), np.full(int((fold == k).sum()), 1 / (fold == k).sum())).astype(float))
              for k in sorted(set(fold))}
        for (name, g, k), (sub, e, _) in es.items():
            a = e.ap(ms[k][sub])
            acc.setdefault((name, g, "ap50"), []).append(a[0.5])
            acc.setdefault((name, g, "ap50_95"), []).append(float(np.mean([a[t] for t in P.AP_IOUS])))
        for key, v in acc.items():
            out.setdefault(key, []).append(float(np.nanmean(v)))
    return {key: (v[0], np.array(v[1:])) for key, v in out.items()}


def paired(res, a, b, g="overall", metric="ap50"):
    ea, da = res[(a, g, metric)]
    eb, db = res[(b, g, metric)]
    lo, hi = P.ci(da - db)
    return dict(diff=ea - eb, lo=lo, hi=hi)


def decide(res):
    """B1-B3 on fold-mean AP@50 (paired bootstrap differences)."""
    out = {}
    d1 = paired(res, "H-wc", "R-wc")
    clf = "hgb" if (d1["lo"] <= 0 <= d1["hi"]) else ("hgb" if d1["diff"] > 0 else "rf")
    out["B1"] = dict(compare="H-wc - R-wc", **d1, choice=clf)
    L = LETTER[clf]
    d2 = paired(res, f"{L}-wc", f"{L}-w")
    feat = "wc" if (d2["diff"] > 0 and d2["lo"] > 0) else "w"
    out["B2"] = dict(compare=f"{L}-wc - {L}-w", **d2, choice=feat)
    d3 = paired(res, f"{L}-{feat}-cal", f"{L}-{feat}")
    cal = bool(d3["diff"] > 0 and d3["lo"] > 0)
    out["B3"] = dict(compare=f"{L}-{feat}-cal - {L}-{feat}", **d3, choice=cal)
    out["selected"] = dict(clf=clf, feat=feat, cal=cal, variant=vname(clf, feat, cal))
    return out


def run_select(T, args):
    gt = P.gt_448(T, P.geometry_table(args.p7_dir))
    det, infos = load_det(args)
    dets = {v: det[det.variant == v] for v in VARIANTS}
    dets.update(references(args))
    res = boot_ap(T, gt, dets, ["overall"], args.n_boot, args.boot_seed)
    sel = decide(res)
    rows = [dict(variant=v, ap50=res[(v, "overall", "ap50")][0], ap50_lo=P.ci(res[(v, "overall", "ap50")][1])[0],
                 ap50_hi=P.ci(res[(v, "overall", "ap50")][1])[1], ap50_95=res[(v, "overall", "ap50_95")][0])
            for v in list(VARIANTS) + ["S1", "P7"]]
    tab = pd.DataFrame(rows)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    tab.round(5).to_csv(args.tables_dir / "phase7b_cv_variants.csv", index=False)
    sel.update(grid=infos[0]["grid"], n_boot=args.n_boot, boot_seed=args.boot_seed,
               date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "selection.json").write_text(json.dumps(sel, indent=1, default=str))
    print(tab.round(5).to_string(index=False))
    for b in ("B1", "B2", "B3"):
        x = sel[b]
        print(f"{b}: {x['compare']} = {x['diff']:+.5f} [{x['lo']:+.5f}, {x['hi']:+.5f}] -> {x['choice']}")
    print(f"selected: {sel['selected']['variant']}", flush=True)
    return sel


def run_final_params(T, R0, args):
    sel = json.loads((args.out_dir / "selection.json").read_text())["selected"]
    gname = json.loads((args.out_dir / "selection.json").read_text())["grid"]
    t0 = time.time()
    sets = stage2_sets(T, R0, args, None)
    check_sets(sets, T, None)
    res = tune(sel["clf"], sel["feat"], sets, GRIDS[gname][sel["clf"]], args)
    cal = calibration_from_oof(res["oof"], sets) if sel["cal"] else None
    out = dict(selected=sel, grid=gname, cfg=res["cfg"], cfg_index=res["cfg_index"], inner_scores=res["inner_scores"],
               calibration=cal, provenance=sets["provenance"], seconds=time.time() - t0,
               date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "final_params.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"final stage-2 hyperparameters ({sel['clf']}-{sel['feat']}): {res['cfg']} "
          f"(inner weighted AP {np.round(res['inner_scores'], 4)})", flush=True)
    if not args.no_log:
        log_run(regions="all (pooled), training images", feature_set=f"Phase 7b {sel['feat']} features",
                model=f"stage 2 {sel['clf']}", imbalance_strategy="balanced; 10:1 proposal negatives",
                key_hyperparameters=f"inner 5-fold tuning, grid {gname}", cv_score=f"cfg {res['cfg']}",
                test_metrics="not used (CV only)", notes="Phase 7b final detector's stage-2 hyperparameters")
    return out


def threshold_needed(args):
    fp = json.loads((args.out_dir / "final_params.json").read_text())
    final_c = json.loads((args.p7_dir / "final_c.json").read_text())["C"]
    key = f"{fp['selected']['clf']}-{fp['selected']['feat']}"
    cfgs, cs = [], []
    for k in range(5):
        info = json.loads((args.out_dir / f"cv_fold{k}.json").read_text())
        cfgs.append(info["models"][key]["cfg"])
        cs.append(info["stage1_outer"]["C"])
    same = all(c == fp["cfg"] for c in cfgs) and all(c == final_c for c in cs)
    return (not same), dict(fold_cfgs=cfgs, final_cfg=fp["cfg"], fold_C=cs, final_C=final_c)


def run_threshold_fold(T, R0, k, args):
    fp = json.loads((args.out_dir / "final_params.json").read_text())
    sel = fp["selected"]
    sets = stage2_sets(T, R0, args, k, want_val=sel["cal"])
    check_sets(sets, T, k)
    res = fit_fixed(sel["clf"], sel["feat"], fp["cfg"], sets, args, want_oof=sel["cal"])
    cal = calibration_from_oof(res["oof"], sets) if sel["cal"] else None
    det = detect_heldout(T, args, k, {sel["variant"]: (res["model"], FEAT_COLS[sel["feat"]], cal)})
    det.to_csv(args.out_dir / f"thrcv_fold{k}_detections.csv", index=False)
    (args.out_dir / f"thrcv_fold{k}.json").write_text(json.dumps(dict(fold=k, cfg=fp["cfg"], provenance=sets["provenance"],
                                                                       calibration=cal), indent=1, default=str))
    print(f"threshold CV fold {k}: {len(det)} detections", flush=True)


def _threshold_metrics(T, gt, det, thr, n_boot, seed):
    fold = T.fold.to_numpy()
    region = dict(zip(T.image_id, T.region))
    evt = {k: {g: P.EvalSet([i for i in T.image_id[fold == k] if g == "overall" or region[i] == g], det, gt)
               for g in GROUPS} for k in sorted(set(fold))}
    return P._threshold_boot(evt, thr, n_boot, seed)


def _proposal_boot(T, args, n_boot, seed):
    fold = T.fold.to_numpy()
    region = T.region.to_numpy()
    per = {}
    for k in sorted(set(fold)):
        pr = load_props(args, tag_outer(k))
        imgs = np.flatnonzero(fold == k)
        pos = {n: q for q, n in enumerate(imgs)}
        nb = np.zeros((len(imgs), 3))
        for n, iou in zip(pr["box_n"], pr["box_iou"]):
            nb[pos[n]] += [1, iou >= 0.5, iou >= 0.3]
        per[k] = (imgs, nb)
    rng = np.random.default_rng(seed)
    out = {}
    for b in range(-1, n_boot):
        acc = {}
        for k, (imgs, nb) in per.items():
            m = np.ones(len(imgs)) if b < 0 else rng.multinomial(len(imgs), np.full(len(imgs), 1 / len(imgs))).astype(float)
            for g in GROUPS:
                sel = np.ones(len(imgs), bool) if g == "overall" else region[imgs] == g
                tot = (m[sel, None] * nb[sel]).sum(0)
                for q, key in ((1, "iou50"), (2, "iou30")):
                    acc.setdefault((g, key), []).append(tot[q] / tot[0] if tot[0] else np.nan)
        for key, v in acc.items():
            out.setdefault(key, []).append(float(np.nanmean(v)))
    return {key: (v[0],) + P.ci(np.array(v[1:])) for key, v in out.items()}


def run_summarize(T, args):
    sel_all = json.loads((args.out_dir / "selection.json").read_text())
    sel = sel_all["selected"]
    v = sel["variant"]
    gt = P.gt_448(T, P.geometry_table(args.p7_dir))
    det, infos = load_det(args)
    refs = references(args)
    need, why = threshold_needed(args)
    if need:
        tdet, _ = load_det(args, "thrcv")
        src, thr_source = tdet[tdet.variant == v], f"threshold CV with the final hyperparameters ({why})"
    else:
        src, thr_source = det[det.variant == v], "detection CV (every outer fold chose the final configuration)"
    thr, thr_f1 = P.f1_threshold(P.EvalSet(list(T.image_id), src, gt))
    master = pd.read_csv(REPO_ROOT / "data" / "master.csv")[["image_id", "hardware"]]
    nohw = set(master.image_id[master.hardware == 0]) & set(T.image_id)
    dets = {v: det[det.variant == v], **refs}
    res = boot_ap(T, gt, dets, GROUPS + ["no_hardware"], args.n_boot, args.boot_seed, {"no_hardware": nohw})
    tm = _threshold_metrics(T, gt, src, thr, args.n_boot, args.boot_seed)
    prop = _proposal_boot(T, args, args.n_boot, args.boot_seed)
    reach = pd.read_csv(args.tables_dir / "phase7_reachability.csv").set_index("group")
    rows = []
    for g in GROUPS:
        row = dict(group=g, n_images=int((T.region == g).sum()) if g != "overall" else len(T),
                   reach_iou50=float(reach.loc["all" if g == "overall" else g, "reach_iou50"]),
                   proposal_recall50=prop[(g, "iou50")][0], proposal_recall50_lo=prop[(g, "iou50")][1],
                   proposal_recall50_hi=prop[(g, "iou50")][2], proposal_recall30=prop[(g, "iou30")][0])
        for name in (v, "S1", "P7"):
            e, d = res[(name, g, "ap50")]
            row.update({f"{name}_ap50": e, f"{name}_ap50_lo": P.ci(d)[0], f"{name}_ap50_hi": P.ci(d)[1]})
        e, d = res[(v, g, "ap50_95")]
        row.update(ap50_95=e, ap50_95_lo=P.ci(d)[0], ap50_95_hi=P.ci(d)[1])
        for key in ("recall", "precision", "f1", "recall_iou30"):
            if (g, key) in tm:
                row.update({key: tm[(g, key)][0], f"{key}_lo": tm[(g, key)][1], f"{key}_hi": tm[(g, key)][2]})
        for ref in ("S1", "P7"):
            p = paired(res, v, ref, g)
            row.update({f"d_{ref}": p["diff"], f"d_{ref}_lo": p["lo"], f"d_{ref}_hi": p["hi"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    nh = {name: res[(name, "no_hardware", "ap50")][0] for name in (v, "S1", "P7")}
    d_s1, d_p7 = paired(res, v, "S1"), paired(res, v, "P7")
    b6 = bool(d_s1["diff"] > 0 and d_s1["lo"] > 0 and d_p7["diff"] > 0 and d_p7["lo"] > 0)
    # image level: top box score of the selected variant against Phase 3 (fold-mean, paired)
    best = det[(det.variant == v) & (det["rank"] == 0)].set_index("image_id").score
    p3 = pd.read_csv(REPO_ROOT / "results" / "phase3" / "oof_rbfsvm__none.csv", float_precision="round_trip")
    j = T[["image_id", "region", "fold", "fractured"]].merge(p3[["image_id", "score"]], on="image_id")
    j["score_det"] = j.image_id.map(best).fillna(-np.inf)
    il = []
    for g in GROUPS:
        d = j if g == "overall" else j[j.region == g]
        if any(x.fractured.nunique() < 2 for _, x in d.groupby("fold")) or d.fold.nunique() < 5:
            il.append(dict(group=g))
            continue
        fb = fold_paired_bootstrap(d.fractured, d.score_det, d.score, d.fold, n_boot=args.n_boot, seed=args.boot_seed)
        per = lambda col, fn: np.mean([fn(x.fractured.to_numpy(), x[col].to_numpy()) for _, x in d.groupby("fold")])
        il.append(dict(group=g, det_pr_auc=per("score_det", _fast_ap), p3_pr_auc=per("score", _fast_ap),
                       d_pr_auc=fb["pr_auc"][0], d_pr_auc_lo=fb["pr_auc"][1], d_pr_auc_hi=fb["pr_auc"][2],
                       det_roc_auc=per("score_det", _fast_auc), p3_roc_auc=per("score", _fast_auc),
                       d_roc_auc=fb["roc_auc"][0], d_roc_auc_lo=fb["roc_auc"][1], d_roc_auc_hi=fb["roc_auc"][2]))
    iltab = pd.DataFrame(il)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(args.tables_dir / "phase7b_cv_localizer.csv", index=False)
    iltab.round(4).to_csv(args.tables_dir / "phase7b_cv_image_level.csv", index=False)
    summ = dict(selection=sel_all, threshold=thr, threshold_f1_pooled=thr_f1, threshold_source=thr_source,
                threshold_cv_needed=need, hyperparameter_check=why, d_vs_S1=d_s1, d_vs_P7=d_p7,
                B6=dict(passed=b6, rule="selected beats S1 AND P7 on fold-mean AP@50, each 95% CI excluding 0",
                        verdict="final test allowed (after approval)" if b6 else "7b reported on CV only; no test"),
                ap50_no_hardware=nh, n_boot=args.n_boot, boot_seed=args.boot_seed,
                date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "cv_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    fig_variants(pd.read_csv(args.tables_dir / "phase7b_cv_variants.csv"), sel_all, args.figures_dir)
    fig_examples(T, det[det.variant == v], gt, thr, args.figures_dir, v)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(table.round(4).to_string(index=False))
    print(iltab.round(3).to_string(index=False))
    print(f"threshold {thr} ({thr_source}); vs S1 {d_s1}; vs P7 {d_p7}; B6 {'PASSED' if b6 else 'FAILED'}")
    if not args.no_log:
        o = table.set_index("group").loc["overall"]
        log_run(regions="per region + overall, training images (out of fold)", feature_set=f"Phase 7b {sel['feat']}",
                model=f"cascade, selected {v}", imbalance_strategy="balanced; 10:1 proposal negatives",
                key_hyperparameters=f"threshold {thr}; B1-B3 {sel}",
                cv_score=f"AP@50={o[v + '_ap50']:.4f} [{o[v + '_ap50_lo']:.4f}, {o[v + '_ap50_hi']:.4f}]; B6 {b6}",
                test_metrics="not used (CV only)", notes="Phase 7b CV summary")
    return summ


# ----------------------------------------------------------------------------- figures

def fig_variants(tab, sel, figdir):
    setup()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    t = tab.reset_index(drop=True)
    col = [COLORS["unified"] if v == sel["selected"]["variant"] else (COLORS["muted"] if v in ("S1", "P7") else
                                                                      COLORS["region"]) for v in t.variant]
    ax.bar(np.arange(len(t)), t.ap50, color=col, yerr=np.vstack([t.ap50 - t.ap50_lo, t.ap50_hi - t.ap50]), capsize=3)
    ax.set_xticks(np.arange(len(t)), t.variant, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AP@50 (fold mean)")
    header(fig, "Phase 7b: AP@50 of every variant (training CV)", ax, ncols=1, top=0.86)
    save(fig, figdir / "phase7b_cv_variants.png", "Out-of-fold detections; error bars: 95% fold-stratified bootstrap. "
         "Orange: the variant B1-B3 select; grey: references (S1 = stage 1 alone, P7 = Phase 7 detector).")


def fig_examples(T, det, gt, thr, figdir, variant, n_good=4, n_bad=4):
    setup()
    top = det[det["rank"] == 0].set_index("image_id")
    hit, miss = [], []
    for r in T[T.boxes.map(len) > 0].itertuples():
        if r.image_id not in top.index:
            continue
        d = det[(det.image_id == r.image_id) & (det.score >= thr)]
        tp = P.match_flags(d[["x", "y", "w", "h"]].to_numpy(), d.score.to_numpy(), gt[r.image_id], 0.5)
        (hit if tp.any() else miss).append((top.loc[r.image_id, "score"], r.image_id))
    fp = sorted([(top.loc[i, "score"], i) for i in T.image_id[T.boxes.map(len) == 0] if i in top.index], reverse=True)
    picks = ([("good", i) for _, i in sorted(hit, reverse=True)[:n_good]]
             + [("missed", i) for _, i in sorted(miss)[:n_bad // 2]] + [("false alarm", i) for _, i in fp[:n_bad // 2]])
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.2))
    fpath, reg = dict(zip(T.image_id, T.file_path)), dict(zip(T.image_id, T.region))
    for ax, (kind, im) in zip(axes.ravel(), picks):
        img, _ = W.load_image(fpath[im])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for x, y, w, h in gt[im]:
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=COLORS["third"], lw=1.6))
        for rr in det[(det.image_id == im) & (det.score >= thr)].itertuples():
            ax.add_patch(plt.Rectangle((rr.x, rr.y), rr.w, rr.h, fill=False, ec=COLORS["unified"], lw=1.2))
        ax.set_title(f"{kind}: {im[3:-4]} ({reg[im]})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle(f"Phase 7b ({variant}) out-of-fold detections on training images (green: ground truth, orange: "
                 "detections at the threshold)", x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    save(fig, figdir / "phase7b_cv_examples.png")


# ----------------------------------------------------------------------------- gates and timing

def run_gates(T, R0, args):
    """Real-data gates on training images (before the CV). Leakage gates L1-L4 included."""
    res = {}
    rng = np.random.default_rng(0)
    log = AccessLog()
    t0 = time.time()
    with log.active():
        # context features from the cached maps = independent computation from the level image
        diffs = []
        for n in rng.choice(len(T), 10, replace=False):
            img, geom = W.load_image(T.file_path[n])
            maps, g2 = W.load_maps(P.map_path(args.cache_dir, T.image_id[n]))
            idx = W.window_index(g2["content"])
            B, Tt = W.index_boxes(idx)
            sel = rng.choice(len(B), 50, replace=False)
            sel[:5] = np.argsort(B[:, 0] + B[:, 1])[:5]                       # include windows at the image edge
            Xc = context_features(maps, idx, sel)
            sizes = np.cumsum([0] + [len(jj) for jj, _, _ in idx])
            for row, s in zip(Xc, sel):
                t = int(Tt[s])
                jj, ii, _ = idx[t]
                loc = s - sizes[t]
                ref = naive_context(img, *W.WINDOW_TYPES[t], int(ii[loc]), int(jj[loc]))
                diffs.append(float(np.abs(row - ref).max()))
        res["context_equals_naive"] = dict(windows=len(diffs), max_abs_diff=max(diffs), passed=max(diffs) <= 1e-5)
        print(f"  context = naive: max diff {max(diffs):.2e} over {len(diffs)} windows", flush=True)
        # stage 1 outer models = Phase 7 round 0 (all folds), proposal recall two ways
        saved, _ = P.load_fold_detections(args.p7_dir, "cv")
        worst, rec_ok, recs = 0.0, True, []
        for k in sorted(T.fold.unique()):
            m = fit_stage1(T, R0, [f for f in sorted(T.fold.unique()) if f != k], args.seed, args.n_jobs)
            pr = make_proposals(T, m, [k], argparse.Namespace(**{**vars(args), "n_jobs": 1}))   # in-process: logged
            sv = saved[saved["round"] == 0]
            for q, n in enumerate(pr["n"][::25]):
                im = T.image_id[n]
                maps, geom = W.load_maps(P.map_path(args.cache_dir, im))
                idx = W.window_index(geom["content"])
                B, _ = W.index_boxes(idx)
                s = W.maps_scores(maps, idx, m["w"], m["b"])
                keep = W.nms(B, s, P.NMS_IOU, P.KEEP_PER_IMAGE)
                d = sv[sv.image_id == im].sort_values("rank")
                worst = max(worst, float(np.abs(d.score.to_numpy() - s[keep]).max()),
                            float(np.abs(d[["x", "y", "w", "h"]].to_numpy() - B[keep]).max()))
            direct = pr["box_iou"] >= 0.5
            ranked = (pr["box_r50"] > 0) & (pr["box_r50"] <= K_PROP)
            direct30, ranked30 = pr["box_iou"] >= 0.3, (pr["box_r30"] > 0) & (pr["box_r30"] <= K_PROP)
            rec_ok &= bool((direct == ranked).all() and (direct30 == ranked30).all())
            recs.append((int(direct.sum()), int(direct30.sum()), len(direct)))
            print(f"  fold {k}: stage-1 C={m['C']}, proposal recall {direct.mean():.3f} / {direct30.mean():.3f}", flush=True)
        tot = np.sum(recs, 0)
        res["stage1_equals_phase7_round0"] = dict(max_abs_diff=worst, passed=worst == 0.0)
        res["proposal_recall_two_ways"] = dict(passed=rec_ok, iou50=tot[0] / tot[2], iou30=tot[1] / tot[2], boxes=int(tot[2]))
    # leakage gates
    res.update(leakage_gates(T, R0, args, log))
    res["all_passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    res["seconds"] = time.time() - t0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gates.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k}: {'PASSED' if v['passed'] else 'FAILED'}  " + ", ".join(
                f"{a}={b}" for a, b in v.items() if a != "passed" and not isinstance(b, (dict, list))), flush=True)
    if not res["all_passed"]:
        raise SystemExit("a Phase 7b gate FAILED (results/phase7b/gates.json)")
    return res


def test_images_last_access():
    """NTFS last-access times of the test images (read-only metadata; the images are not opened)."""
    import os
    from preprocess import image_path
    m = pd.read_csv(REPO_ROOT / "data" / "master.csv")[["image_id", "file_path"]].merge(
        pd.read_csv(REPO_ROOT / "results" / "splits.csv")[["image_id", "split"]], on="image_id")
    paths = m.file_path[m.split != "train"]
    return max(os.stat(image_path(p)).st_atime for p in paths), int(len(paths))


def leakage_gates(T, R0, args, log):
    out = {}
    fold = T.fold.to_numpy()
    train_ids, train_paths = set(T.image_id), set(T.file_path)
    # L1: no test image or test box anywhere before the 7b lock
    src = Path(__file__).read_text(encoding="utf-8")
    body = src.split("def test_images_last_access", 1)[0]        # the metadata check below is the only exception
    banned = ['split == "test"', "split=='test'", "test_table", "test_scores", "test_detections", "TEST_EVALUATED"]
    static = [b for b in banned if b in body]
    ft = (Path(__file__).parent / "phase7b_final_test.py").read_text(encoding="utf-8")
    lock_pos, read_pos = ft.find("lock.write_text("), ft.find("Te = test_table()")
    import datetime as dt
    p7_lock = json.loads((args.p7_dir / "test_run.json").read_text())
    p7_end = dt.datetime.fromisoformat(p7_lock["date"]).timestamp() + p7_lock["seconds"]   # naive = local time
    atime, n_test = test_images_last_access()
    out["L1_no_test_data_before_lock"] = dict(
        passed=bool(set(log.maps) <= {Path(i).stem for i in train_ids} and set(log.images) <= train_paths
                    and not static and 0 < lock_pos < read_pos and len(T) == P.EXPECTED_TRAIN
                    and atime <= p7_end + 60),
        maps_read=len(set(log.maps)), images_read=len(set(log.images)),
        all_reads_training=bool(set(log.maps) <= {Path(i).stem for i in train_ids} and set(log.images) <= train_paths),
        banned_strings_in_cascade=static, final_test_lock_before_test_rows=bool(0 < lock_pos < read_pos),
        train_table_rows=len(T), test_images=n_test,
        test_images_last_access=dt.datetime.fromtimestamp(atime).isoformat(timespec="seconds"),
        phase7_test_ended_at_latest=dt.datetime.fromtimestamp(p7_end).isoformat(timespec="seconds"))
    # L2: no image in more than one fold's patches (round-0 patches; stage-2 sets are asserted by check_sets)
    img = np.r_[R0["img_pos"], R0["img_pool"]]
    out["L2_one_fold_per_image"] = dict(passed=bool(len(set(zip(img, fold[img]))) == len(set(img))
                                                    and T.image_id.is_unique),
                                        images=int(len(set(img))), note="stage-2 sets: check_sets() at runtime")
    # L3: every stage-2 training proposal comes from a model that did not see that image's fold (plan, all folds)
    plan = stage1_plan(sorted(T.fold.unique()))
    viol = []
    for k in [None] + sorted(T.fold.unique()):
        for j in sorted(T.fold.unique()):
            if j == k:
                continue
            tr_folds, scored = plan[proposal_source(j, k)]
            if j in tr_folds or (k is not None and k in tr_folds) or j not in scored:
                viol.append((k, j))
    out["L3_cross_fitted_proposals"] = dict(passed=not viol, violations=viol, plan={t: v[0] for t, v in plan.items()})
    # L4: C, stage-2 hyperparameters, calibration and thresholds fitted on training-fold data only
    checks = []
    for k in sorted(T.fold.unique()):
        tr_folds, _ = plan[tag_outer(k)]
        checks.append(k not in tr_folds)
        for j in sorted(T.fold.unique()):
            if j != k:
                checks.append(k not in plan[proposal_source(j, k)][0])
    out["L4_fits_on_training_folds_only"] = dict(
        passed=bool(all(checks) and len(T) == P.EXPECTED_TRAIN),
        note="every fitted quantity uses T (training rows only); per-fold provenance asserted at runtime "
             "(fit_stage1, stage2_sets, check_sets, tune); thresholds from out-of-fold detections of training images")
    return out


def timing_set(T, R0, args, m=None):
    """A full-size stage-2 training set for timing (not used for any result): positives of folds 1-4 and 10:1 negatives
    from the stage-1 proposals of 600 of their images; plus a 65k-row validation-like block."""
    rng = np.random.default_rng(1)
    m = m or fit_stage1(T, R0, [1, 2, 3, 4], args.seed, args.n_jobs)
    use = np.flatnonzero(T.fold.to_numpy() != 0)
    frac = [n for n in use if T.boxes[n]]
    normal = [n for n in use if not T.boxes[n]]
    imgs = np.r_[frac, rng.choice(normal, max(0, 600 - len(frac)), replace=False)].astype(int)
    pr = P.run_parallel(_prop_chunk, [(n, T.image_id[n], T.boxes[n]) for n in imgs], args.n_jobs, args.cache_dir,
                        m["w"], m["b"], K_PROP)
    n_pos = int(np.bincount(R0["img_pos"], minlength=len(T))[use].sum())
    per_img = max(1, int(np.ceil(NEG_RATIO * n_pos / len(pr))))
    jobs = []
    for r in pr:
        neg = np.flatnonzero(r["iou"] < P.NEG_IOU)
        q = np.sort(rng.choice(neg, size=min(per_img, len(neg)), replace=False))
        jobs.append((r["n"], T.image_id[r["n"]], T.boxes[r["n"]], m["w"], m["b"], r["prop"][q], r["s1"][q],
                     r["prop"][:VAL_NEG], r["s1"][:VAL_NEG], bool(T.boxes[r["n"]])))
    t1 = time.time()
    rows = P.run_parallel(_rows_chunk, jobs, args.n_jobs, args.cache_dir)
    t_rows = time.time() - t1
    X = np.concatenate([r["pos"] for r in rows if "pos" in r] + [r["neg"] for r in rows])
    y = np.r_[np.ones(sum(len(r["pos"]) for r in rows if "pos" in r), int), np.zeros(sum(len(r["neg"]) for r in rows), int)]
    Xv = np.concatenate([r["val"] for r in rows])
    Xv = np.concatenate([Xv] * int(np.ceil(65000 / len(Xv))))[:65000]
    return X, y, Xv, t_rows, len(jobs)


def run_timing(T, R0, args):
    """Timing on real data: proposals + features + scoring for 10 training images; each stage-2 config fitted once on
    a full-size training set; projection of the CV runtime and the fallback-grid decision."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401  (import cost outside the timings)
    rng = np.random.default_rng(1)
    t0 = time.time()
    m = fit_stage1(T, R0, [1, 2, 3, 4], args.seed, args.n_jobs)
    t_stage1 = time.time() - t0
    picks = rng.choice(np.flatnonzero(T.fold.to_numpy() != 0), 10, replace=False)
    t_prop, t_feat = [], []
    for n in picks:
        t1 = time.time()
        r = _prop_chunk([(n, T.image_id[n], T.boxes[n])], args.cache_dir, m["w"], m["b"], K_PROP)[0]
        t2 = time.time()
        maps, geom = W.load_maps(P.map_path(args.cache_dir, T.image_id[n]))
        idx = W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        window_rows(maps, idx, Tt, r["prop"], r["s1"])
        t_prop.append(t2 - t1)
        t_feat.append(time.time() - t2)
    X, y, Xv, t_rows, n_jobs_rows = timing_set(T, R0, args, m)
    # stage-2 cost, measured on the code path the CV uses: per configuration, the 4 inner fits (each on 3/4 of the rows,
    # predicting a 65k-row validation block) through inner_predictions, then one refit on all rows
    fake = rng.integers(0, 4, len(y))
    tsets = dict(train=dict(X=X, y=y, fold=fake), val={j: dict(X=Xv) for j in range(4)})
    fits = []
    for clf in CLFS:
        for feat in FEATS:
            cols = FEAT_COLS[feat]
            for cfg in GRIDS["full"][clf]:
                t1 = time.time()
                inner_predictions(clf, [cfg], tsets, cols, args.seed)
                ti = time.time() - t1
                t1 = time.time()
                fit_clf(clf, cfg, X[:, cols], y, args.seed)
                fits.append(dict(clf=clf, cfg=cfg, feat=feat, rows=len(y), inner_4fits_s=ti, refit_s=time.time() - t1))
                print(f"  {clf} {cfg} {feat}: 4 inner fits {ti:.1f} s, refit {fits[-1]['refit_s']:.1f} s", flush=True)
    F = pd.DataFrame(fits)
    # the thread setting must not change any fitted model: CV-path predictions = single-threaded fits (real data)
    det_check = {}
    for clf in CLFS:
        cfg, cols = GRIDS["fallback"][clf][0], FEAT_COLS["wc"]
        par = inner_predictions(clf, [cfg], tsets, cols, args.seed)[(0, 0)]
        m0 = fake != 0
        ref = predict(fit_clf(clf, cfg, X[m0][:, cols], y[m0], args.seed, n_jobs=1, threads=1), Xv[:, cols])
        det_check[clf] = dict(cfg=cfg, max_abs_pred_diff=float(np.abs(par - ref).max()))
        print(f"  determinism {clf} {cfg}: CV path vs single-threaded fit, max |diff| = "
              f"{det_check[clf]['max_abs_pred_diff']:.2e}", flush=True)

    def stage2_hours(grid):
        tot = 0.0
        for clf in CLFS:
            for feat in FEATS:
                g = F[(F.clf == clf) & (F.feat == feat) & F.cfg.map(lambda c: c in grid[clf])]
                tot += g.inner_4fits_s.sum() + g.refit_s.mean()                 # inner tuning + one refit
        return 5 * tot / 3600
    per_img_prop, per_img_feat = float(np.mean(t_prop)), float(np.mean(t_feat))
    eff = max(1.0, args.n_jobs * 0.6)                                  # Phase 7: 4 workers gave about 2.5x
    h_stage2_full, h_stage2_fb = stage2_hours(GRIDS["full"]), stage2_hours(GRIDS["fallback"])
    other_h = (15 * t_stage1 + 16330 * per_img_prop / eff + 5 * t_rows * (len(T) * 0.8 / n_jobs_rows)
               + len(T) * per_img_feat / eff + 4 * len(T) * K_PROP / 65000 * 4.0) / 3600   # predict: ~4 s / 65k rows
    use_fb = h_stage2_full > FALLBACK_HOURS
    est = dict(stage1_fit_s=t_stage1, proposals_s_per_image=per_img_prop, features_s_per_image_1000=per_img_feat,
               train_rows=int(len(y)), train_rows_build_s=t_rows, fits=fits, stage2_fits_hours_full=h_stage2_full,
               stage2_fits_hours_fallback=h_stage2_fb, other_cv_hours=other_h,
               cv_total_hours_full=h_stage2_full + other_h, cv_total_hours_fallback=h_stage2_fb + other_h,
               fallback_rule=f"fallback grids if the stage-2 fits of the CV project more than {FALLBACK_HOURS} h",
               use_fallback=bool(use_fb), n_jobs=args.n_jobs, threads=dict(hgb_single=HGB_THREADS, inner_workers=INNER_WORKERS,
               inner_threads=INNER_THREADS, rf_jobs=RF_JOBS), determinism=det_check, date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / "timing.json").write_text(json.dumps(est, indent=1, default=str))
    print(f"projection: stage-2 fits {h_stage2_full:.1f} h (full grids) / {h_stage2_fb:.1f} h (fallback); other CV "
          f"steps {other_h:.1f} h -> {'FALLBACK grids' if use_fb else 'full grids'}", flush=True)
    return est


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    for flag in ("--gates", "--timing", "--stage1", "--select", "--final-params", "--summarize", "--force", "--no-log"):
        ap.add_argument(flag, action="store_true")
    ap.add_argument("--cv-fold", type=int, nargs="+", default=None)
    ap.add_argument("--threshold-cv", type=int, nargs="+", default=None)
    ap.add_argument("--grid", choices=["auto", "full", "fallback"], default="auto")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--debug-subset", type=int, default=0, help="DEBUG: about N training images (no log)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase7b")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "features" / "phase7")
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "features" / "phase7b")
    ap.add_argument("--p7-dir", type=Path, default=REPO_ROOT / "results" / "phase7")
    args = ap.parse_args(argv)
    threads = configure_threads(args.n_jobs)
    if args.debug_subset:
        args.no_log = True
        defaults = {REPO_ROOT / "results" / "phase7b", REPO_ROOT / "results" / "tables", REPO_ROOT / "results" / "figures",
                    REPO_ROOT / "features" / "phase7b", REPO_ROOT / "features" / "phase7", REPO_ROOT / "results" / "phase7"}
        if {args.out_dir, args.tables_dir, args.figures_dir, args.work_dir, args.cache_dir, args.p7_dir} & defaults:
            ap.error("--debug-subset needs its own output, work, cache and Phase 7 folders")
    T = P.train_table(args.debug_subset or None)
    R0 = P.load_round0(args.cache_dir, T)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(T)} training images ({int(T.fractured.sum())} fractured, {int(T.boxes.map(len).sum())} boxes); "
          f"stage-2 threads {threads}", flush=True)
    if args.gates:
        run_gates(T, R0, args)
    if args.timing:
        run_timing(T, R0, args)
    if args.stage1:
        if (args.work_dir / "stage1_models.npz").exists() and not args.force:
            print("[skip] stage 1 already done (use --force to redo)")
        else:
            run_stage1(T, R0, args)
    for k in args.cv_fold or []:
        if (args.out_dir / f"cv_fold{k}.json").exists() and not args.force:
            print(f"[skip] fold {k} already done (use --force to redo)")
            continue
        run_cv_fold(T, R0, k, args)
    if args.select:
        run_select(T, args)
    if args.final_params:
        run_final_params(T, R0, args)
    for k in args.threshold_cv or []:
        run_threshold_fold(T, R0, k, args)
    if args.summarize:
        run_summarize(T, args)


if __name__ == "__main__":
    main()
