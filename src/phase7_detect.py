"""Phase 7: patch classifier, hard-negative mining, sliding-window detection, NMS and AP. TRAINING IMAGES ONLY.

    python src/phase7_detect.py --gates            # box mapping figure + real-data gates (before anything is trained)
    python src/phase7_detect.py --timing           # full scoring of 10 training images, runtime estimates
    python src/phase7_detect.py --cache            # dense feature maps of every training image -> features/phase7/maps/
    python src/phase7_detect.py --patches          # round-0 positives and negative pools, from the cache
    python src/phase7_detect.py --cv-fold 0        # detection CV, one outer fold per call (0 .. 4)
    python src/phase7_detect.py --final-c          # tune_cv on all training round-0 patches (the final detector's C)
    python src/phase7_detect.py --threshold-cv 0   # only needed if the outer folds disagree on C (rule D2)
    python src/phase7_detect.py --summarize        # metrics, D1, threshold, tables, figures
    options: --n-jobs 4  --n-boot 2000  --boot-seed 0  --force  --no-log

Protocol and decision rules: docs/phase7_localization.md (written before any model was trained). Test images are never
loaded here; the test set belongs to src/phase7_final_test.py and its lock.

Detection CV, outer fold k: round-0 patches of the training images in the other four folds (positives: windows with
IoU >= 0.5 with a box; negatives: 3 per positive, IoU < 0.1 with every box, uniform over the eligible windows, seed 0),
C by tune_cv over their four folds (patch PR-AUC), a linear SVM (class_weight balanced), two rounds of hard-negative
mining on the same images (windows with IoU < 0.1 and score > 0, NMS 0.3, top 20 per image, negatives accumulate, same
C), and fold k's images scored by the round-0, 1 and 2 detectors (NMS 0.3, top 10 boxes per image). Patches inherit
their image's frozen fold, so an image's patches never sit in two folds.

Outputs: features/phase7/ (maps cache, round-0 patches; not committed), results/phase7/*.csv / *.json,
         results/tables/phase7_cv_*.csv, results/figures/phase7_box_mapping.png, phase7_cv_*.png
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import phase7_windows as W
from cv_protocol import tune_cv
from expt_log import log_run
from model_zoo import build
from paths import REPO_ROOT
from phase4_regional import _fast_ap, _fast_auc, fold_paired_bootstrap
from preprocess import load_gray, preprocess
from phase5_viz import C as COLORS, REGION_LABEL, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

REGIONS = ["hand", "leg", "hip", "shoulder"]
C_GRID = [0.001, 0.01, 0.1, 1]
POS_IOU, NEG_IOU, NMS_IOU = 0.5, 0.1, 0.3
NEG_PER_POS, POOL, MINE_PER_IMAGE, KEEP_PER_IMAGE, ROUNDS = 3, 40, 20, 10, 2
AP_IOUS = tuple(np.round(np.arange(0.50, 0.951, 0.05), 2))
EVAL_IOUS = (0.3,) + AP_IOUS
EXPECTED_TRAIN = 3266                                   # None skips the check (debug subsets)


# ----------------------------------------------------------------------------- data (training images only)

def train_table(subset=None):
    """The training images (test rows are dropped on reading): image_id, file_path, region, fold, boxes (original px)."""
    m = pd.read_csv(REPO_ROOT / "data" / "master.csv").merge(
        pd.read_csv(REPO_ROOT / "results" / "splits.csv")[["image_id", "split", "fold"]], on="image_id",
        validate="one_to_one")
    t = m[m.split == "train"].reset_index(drop=True)
    t = pd.DataFrame(dict(image_id=t.image_id, file_path=t.file_path, region=t.region_group, fold=t.fold.astype(int),
                          fractured=t.fractured.astype(int), height=t.height.astype(int), width=t.width.astype(int),
                          boxes=t.boxes_xywh.map(json.loads)))
    if subset:
        rng = np.random.default_rng(0)
        pick = []
        for (_, _), g in t.groupby(["fold", "fractured"]):
            pick += list(rng.choice(g.index, size=max(1, round(subset * len(g) / len(t))), replace=False))
        t = t.loc[sorted(pick)].reset_index(drop=True)
    elif EXPECTED_TRAIN:
        assert len(t) == EXPECTED_TRAIN, "unexpected training set; is this the frozen split?"
    return t


def map_path(cache_dir, image_id):
    return cache_dir / "maps" / f"{Path(image_id).stem}.npz"


def geometry_table(out_dir):
    g = pd.read_csv(out_dir / "train_geometry.csv")
    return {r.image_id: dict(scale=r.scale, off_x=r.off_x, off_y=r.off_y, content=(r.cx0, r.cy0, r.cx1, r.cy1))
            for r in g.itertuples()}


def chunks(items, n):
    n = max(1, min(n, len(items)))
    return [items[i::n] for i in range(n)]


def run_parallel(fn, items, n_jobs, *args):
    res = Parallel(n_jobs=n_jobs)(delayed(fn)(c, *args) for c in chunks(items, 4 * n_jobs))
    return [x for part in res for x in part]


# ----------------------------------------------------------------------------- cache

def _cache_chunk(rows, cache_dir, force):
    out = []
    for image_id, file_path, h0, w0, boxes in rows:
        p = map_path(cache_dir, image_id)
        gray = load_gray(file_path)
        img, geom = preprocess(gray, size=W.SIZE, mid=W.MID, norm="clahe")
        if force or not p.exists():
            W.save_maps(p, W.image_maps(img), geom)
        gt = W.map_boxes(boxes, geom)
        x0, y0, x1, y1 = geom["content"]
        inside = bool(((gt[:, 0] >= x0 - 1) & (gt[:, 1] >= y0 - 1) & (gt[:, 0] + gt[:, 2] <= x1 + 1)
                       & (gt[:, 1] + gt[:, 3] <= y1 + 1)).all()) if len(gt) else True
        out.append(dict(image_id=image_id, h=gray.shape[0], w=gray.shape[1], shape_ok=gray.shape == (h0, w0),
                        boxes_inside_content=inside, scale=geom["scale"], off_x=geom["off_x"], off_y=geom["off_y"],
                        cx0=x0, cy0=y0, cx1=x1, cy1=y1))
    return out


def build_cache(T, args):
    (args.cache_dir / "maps").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = [(r.image_id, r.file_path, r.height, r.width, r.boxes) for r in T.itertuples()]
    res = pd.DataFrame(run_parallel(_cache_chunk, rows, args.n_jobs, args.cache_dir, args.force))
    res = T[["image_id"]].merge(res, on="image_id", validate="one_to_one")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.drop(columns=["h", "w", "shape_ok", "boxes_inside_content"]).to_csv(args.out_dir / "train_geometry.csv",
                                                                             index=False)
    frac = T.boxes.map(len) > 0
    summ = dict(n_images=len(res), seconds=time.time() - t0,
                fractured_shape_mismatch=res.image_id[frac & ~res.shape_ok].tolist(),
                any_shape_mismatch=int((~res.shape_ok).sum()),
                boxes_outside_content=res.image_id[~res.boxes_inside_content].tolist(),
                cache_gb=sum(map_path(args.cache_dir, i).stat().st_size for i in res.image_id) / 1e9)
    (args.out_dir / "cache_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"cache: {len(res)} images in {summ['seconds'] / 60:.1f} min, {summ['cache_gb']:.1f} GB; fractured images "
          f"whose loaded shape differs from master: {len(summ['fractured_shape_mismatch'])}; boxes outside the content "
          f"rectangle: {len(summ['boxes_outside_content'])}", flush=True)
    if summ["fractured_shape_mismatch"] or summ["boxes_outside_content"]:
        raise SystemExit("box-mapping gate FAILED (see results/phase7/cache_summary.json)")
    if not args.no_log:
        log_run(regions="all (pooled), training images", feature_set="Phase 7 window maps (HOG 8 px, LBP, intensity)",
                model="n/a (feature cache, nothing fitted)", imbalance_strategy="n/a",
                key_hyperparameters=f"448 px CLAHE; window types {list(W.WINDOW_TYPES)}",
                cv_score=f"{len(res)} images, {summ['cache_gb']:.1f} GB", test_metrics="not used (training images only)",
                notes="Phase 7 cache of dense window feature maps (training images only)")
    return summ


# ----------------------------------------------------------------------------- round-0 patches

def _patch_chunk(rows, cache_dir):
    out = []
    for n, image_id, boxes in rows:
        maps, geom = W.load_maps(map_path(cache_dir, image_id))
        idx = W.window_index(geom["content"])
        B, _ = W.index_boxes(idx)
        gt = W.map_boxes(boxes, geom)
        best = W.iou_matrix(B, gt).max(1) if len(gt) else np.zeros(len(B))
        pos, elig = np.flatnonzero(best >= POS_IOU), np.flatnonzero(best < NEG_IOU)
        pool = np.random.default_rng([0, n]).choice(elig, size=min(POOL, len(elig)), replace=False)
        out.append((n, W.maps_features(maps, idx, pos), W.maps_features(maps, idx, pool), len(elig)))
    return out


def build_patches(T, args):
    t0 = time.time()
    res = sorted(run_parallel(_patch_chunk, [(n, r.image_id, r.boxes) for n, r in enumerate(T.itertuples())],
                              args.n_jobs, args.cache_dir), key=lambda x: x[0])
    R0 = dict(X_pos=np.concatenate([r[1] for r in res]), img_pos=np.concatenate([np.full(len(r[1]), r[0]) for r in res]),
              X_pool=np.concatenate([r[2] for r in res]),
              img_pool=np.concatenate([np.full(len(r[2]), r[0]) for r in res]),
              rank_pool=np.concatenate([np.arange(len(r[2])) for r in res]),
              n_elig=np.array([r[3] for r in res]), image_id=T.image_id.to_numpy())
    np.savez(args.cache_dir / "round0.npz", **R0)
    summ = dict(n_images=len(T), positives=int(len(R0["X_pos"])), pool_per_image=POOL, seconds=time.time() - t0)
    pos_by_region = pd.Series(T.region.to_numpy()[R0["img_pos"]]).value_counts().to_dict()
    summ.update(positives_by_region={k: int(v) for k, v in pos_by_region.items()})
    (args.out_dir / "patches_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"round-0 patches: {summ['positives']} positive windows {summ['positives_by_region']}; negative pools of "
          f"{POOL} per image ({len(R0['X_pool'])} windows); {summ['seconds']:.0f} s", flush=True)
    return R0


def load_round0(cache_dir, T):
    with np.load(cache_dir / "round0.npz", allow_pickle=True) as z:
        R0 = {k: z[k] for k in z.files}
    assert (R0["image_id"] == T.image_id.to_numpy()).all(), "round0.npz was built for another image list"
    return R0


def round0_set(R0, T, use, seed=0):
    """Round-0 patches of the images where `use` is True: every positive, and 3 negatives per positive drawn uniformly
    from all eligible windows of those images (image chosen in proportion to its eligible windows, then the next window
    of its pre-shuffled pool). Returns X, y, fold (each patch carries its image's fold), image index."""
    use = np.asarray(use, bool)
    pm = use[R0["img_pos"]]
    n_neg = NEG_PER_POS * int(pm.sum())
    elig = np.where(use, R0["n_elig"], 0).astype(float)
    avail = np.bincount(R0["img_pool"], minlength=len(T)) * use
    counts = np.random.default_rng(seed).multinomial(n_neg, elig / elig.sum())
    over = counts - np.minimum(counts, avail)
    counts = np.minimum(counts, avail)
    while over.sum():                                    # pools are 40 deep; this practically never runs
        spare = np.flatnonzero(counts < avail)
        add = np.random.default_rng(seed + 1).choice(spare, size=int(over.sum()))
        np.add.at(counts, add, 1)
        over = counts - np.minimum(counts, avail)
        counts = np.minimum(counts, avail)
    nm = R0["rank_pool"] < counts[R0["img_pool"]]
    fold = T.fold.to_numpy()
    X = np.concatenate([R0["X_pos"][pm], R0["X_pool"][nm]])
    img = np.concatenate([R0["img_pos"][pm], R0["img_pool"][nm]])
    y = np.r_[np.ones(pm.sum(), int), np.zeros(nm.sum(), int)]
    return X, y, fold[img], img


# ----------------------------------------------------------------------------- training, mining, detection

def fit_linear(X, y, C, seed=42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build("linsvm", "balanced", {"C": C}, W.COL_WEIGHTS, seed).fit(X, y)
    return W.fold_linear(pipe)


def _mine_chunk(rows, cache_dir, w, b):
    out = []
    for n, image_id, boxes in rows:
        maps, geom = W.load_maps(map_path(cache_dir, image_id))
        idx = W.window_index(geom["content"])
        B, _ = W.index_boxes(idx)
        s = W.maps_scores(maps, idx, w, b)
        gt = W.map_boxes(boxes, geom)
        best = W.iou_matrix(B, gt).max(1) if len(gt) else np.zeros(len(B))
        cand = np.flatnonzero((best < NEG_IOU) & (s > 0))
        keep = cand[W.nms(B[cand], s[cand], NMS_IOU, MINE_PER_IMAGE)] if len(cand) else cand
        out.append((n, W.maps_features(maps, idx, keep)))
    return out


def mine(T, use, args, w, b):
    rows = [(n, r.image_id, r.boxes) for n, r in enumerate(T.itertuples()) if use[n]]
    res = run_parallel(_mine_chunk, rows, args.n_jobs, args.cache_dir, w, b)
    X = np.concatenate([r[1] for r in res]) if res else np.zeros((0, W.N_FEAT), np.float32)
    img = np.concatenate([np.full(len(r[1]), r[0]) for r in res]) if res else np.zeros(0, int)
    return X, img


def _detect_chunk(rows, cache_dir, wbs):
    out = []
    for n, image_id in rows:
        maps, geom = W.load_maps(map_path(cache_dir, image_id))
        idx = W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        for r, (w, b) in enumerate(wbs):
            s = W.maps_scores(maps, idx, w, b)
            keep = W.nms(B, s, NMS_IOU, KEEP_PER_IMAGE)
            for rank, k in enumerate(keep):
                out.append(dict(image_id=image_id, round=r, rank=rank, score=s[k], x=B[k, 0], y=B[k, 1], w=B[k, 2],
                                h=B[k, 3], wtype=int(Tt[k])))
    return out


def detect(T, use, args, wbs):
    rows = [(n, r.image_id) for n, r in enumerate(T.itertuples()) if use[n]]
    return pd.DataFrame(run_parallel(_detect_chunk, rows, args.n_jobs, args.cache_dir, wbs))


def run_fold(T, R0, k, args, C_fixed=None, tag="cv"):
    """One outer fold of the detection CV (tag "cv": C tuned inside; tag "thrcv": C fixed, for the threshold rule)."""
    t0 = time.time()
    use = T.fold.to_numpy() != k
    X, y, f, img = round0_set(R0, T, use)
    assert len(set(zip(img, f))) == len(set(img)), "an image's patches sit in two folds"
    info = dict(fold=k, tag=tag, n_images=int(use.sum()), n_pos=int(y.sum()), n_neg_round0=int((y == 0).sum()))
    if C_fixed is None:
        tc = tune_cv(X, y, f, "linsvm", "balanced", W.COL_WEIGHTS, grid={"C": C_GRID}, seed=args.seed,
                     n_jobs=args.n_jobs)
        C = tc["params"]["C"]
        info.update(C=C, tune_pr_auc={str(c["C"]): a for c, a in zip(tc["configs"], tc["all_pr_auc"])})
    else:
        C = C_fixed
        info.update(C=C, tune_pr_auc=None)
    wbs = [fit_linear(X, y, C, args.seed)]
    hard = []
    for r in range(1, ROUNDS + 1):
        Xh, imgh = mine(T, use, args, *wbs[-1])
        hard.append(int(len(Xh)))
        X, y = np.concatenate([X, Xh]), np.r_[y, np.zeros(len(Xh), int)]
        wbs.append(fit_linear(X, y, C, args.seed))
        print(f"    fold {k} ({tag}): round {r} mined {len(Xh)} hard negatives -> {len(y)} patches", flush=True)
    det = detect(T, ~use, args, wbs)
    info.update(hard_negatives=hard, n_test_images=int((~use).sum()), seconds=time.time() - t0)
    det.to_csv(args.out_dir / f"{tag}_fold{k}_detections.csv", index=False)
    (args.out_dir / f"{tag}_fold{k}.json").write_text(json.dumps(info, indent=1))
    print(f"fold {k} ({tag}): C={C}, {info['n_pos']} positives, hard negatives {hard}, {info['seconds'] / 60:.1f} min",
          flush=True)
    if not args.no_log:
        log_run(regions="all (pooled), training images", feature_set="Phase 7 window features (HOG 8 px 324, LBP 40, "
                "intensity 2)", model="linear SVM patch classifier + sliding windows + NMS 0.3",
                imbalance_strategy="balanced; 3 negatives per positive + hard-negative mining",
                key_hyperparameters=f"outer fold {k} ({tag}); C={C}" + (" (fixed)" if C_fixed else " (tune_cv)")
                + f"; rounds {ROUNDS}; hard negatives {hard}",
                cv_score="see phase7_detect.py --summarize", test_metrics="not used (CV only)",
                notes=f"Phase 7 detection CV ({tag}), fold {k}: detections of fold {k} images for rounds 0-{ROUNDS}")
    return info


# ----------------------------------------------------------------------------- evaluation

def match_flags(det_boxes, det_scores, gt_boxes, thr):
    """True-positive flags of one image's detections: taken in descending score, each matched to the unmatched
    ground-truth box with the highest IoU, a true positive if that IoU >= thr."""
    det_scores = np.asarray(det_scores, float)
    tp = np.zeros(len(det_scores), bool)
    if len(det_scores) == 0 or len(gt_boxes) == 0:
        return tp
    I = W.iou_matrix(det_boxes, gt_boxes)
    used = np.zeros(I.shape[1], bool)
    for d in np.argsort(-det_scores, kind="stable"):
        cand = np.where(used, -1.0, I[d])
        g = int(cand.argmax())
        if cand[g] >= thr:
            tp[d], used[g] = True, True
    return tp


def average_precision(scores, tp, n_gt, weights=None):
    """All-point interpolated AP of detections (scores, true-positive flags) against n_gt boxes. `weights` (one per
    detection) implement image resampling: a weight of m counts the detection m times."""
    if n_gt <= 0:
        return np.nan
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores, float), kind="stable")
    t = np.asarray(tp, float)[o]
    wd = np.ones(len(t)) if weights is None else np.asarray(weights, float)[o]
    ctp, cfp = np.cumsum(wd * t), np.cumsum(wd * (1 - t))
    rec = ctp / n_gt
    prec = np.divide(ctp, ctp + cfp, out=np.zeros_like(ctp), where=(ctp + cfp) > 0)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * env))


class EvalSet:
    """Detections and ground truth of a set of images, with match flags precomputed per IoU threshold, so that AP and
    threshold metrics of any image weighting (bootstrap) are cumulative sums over one sorted list."""

    def __init__(self, images, det, gt):
        self.images = list(images)
        pos = {im: n for n, im in enumerate(self.images)}
        det = det[det.image_id.isin(pos)].copy()
        self.n_gt = np.array([len(gt.get(im, ())) for im in self.images], float)
        flags = np.zeros((len(det), len(EVAL_IOUS)), bool)
        det = det.reset_index(drop=True)
        for im, g in det.groupby("image_id", sort=False):
            gb = np.asarray(gt.get(im, np.zeros((0, 4))), float).reshape(-1, 4)
            for j, t in enumerate(EVAL_IOUS):
                flags[g.index, j] = match_flags(g[["x", "y", "w", "h"]].to_numpy(), g.score.to_numpy(), gb, t)
        o = np.argsort(-det.score.to_numpy(), kind="stable")
        self.score = det.score.to_numpy()[o]
        self.img = det.image_id.map(pos).to_numpy()[o].astype(int)
        self.flags = flags[o]

    def ap(self, m=None):
        """{iou: AP} for image multiplicities m (default 1 each)."""
        m = np.ones(len(self.images)) if m is None else m
        n_gt = float(m @ self.n_gt)
        wd = m[self.img]
        out = {}
        for j, t in enumerate(EVAL_IOUS):
            out[t] = average_precision(np.arange(len(wd))[::-1], self.flags[:, j], n_gt, wd)    # already sorted
        return out

    def at_threshold(self, thr, m=None):
        """Recall, precision, F1 at IoU 0.5 and recall at IoU 0.3, counting detections with score >= thr."""
        m = np.ones(len(self.images)) if m is None else m
        sel = self.score >= thr
        wd = m[self.img] * sel
        n_gt = float(m @ self.n_gt)
        tp50, tp30 = wd @ self.flags[:, EVAL_IOUS.index(0.5)], wd @ self.flags[:, EVAL_IOUS.index(0.3)]
        n_det = wd.sum()
        rec = tp50 / n_gt if n_gt else np.nan
        prec = tp50 / n_det if n_det else 0.0
        f1 = 2 * tp50 / (n_det + n_gt) if (n_det + n_gt) else np.nan
        return dict(recall=rec, precision=prec, f1=f1, recall_iou30=tp30 / n_gt if n_gt else np.nan)


def f1_threshold(es):
    """F1-optimal box-score threshold at IoU 0.5 (over the distinct scores of the pooled detections)."""
    t50 = es.flags[:, EVAL_IOUS.index(0.5)].astype(float)
    ctp, k = np.cumsum(t50), np.arange(1, len(t50) + 1)
    f1 = 2 * ctp / (k + es.n_gt.sum())
    last = np.r_[np.flatnonzero(np.diff(es.score) != 0), len(es.score) - 1]   # end of each block of equal scores
    best = last[np.argmax(f1[last])]
    return float(es.score[best]), float(f1[best])


def gt_448(T, geoms):
    return {r.image_id: W.map_boxes(r.boxes, geoms[r.image_id]) for r in T.itertuples()}


def load_fold_detections(out_dir, tag="cv"):
    parts, infos = [], {}
    for k in range(5):
        p = out_dir / f"{tag}_fold{k}_detections.csv"
        if not p.exists():
            raise SystemExit(f"{p.name} missing: run python src/phase7_detect.py --{'cv-fold' if tag == 'cv' else 'threshold-cv'} {k}")
        parts.append(pd.read_csv(p, float_precision="round_trip").assign(fold=k))
        infos[k] = json.loads((out_dir / f"{tag}_fold{k}.json").read_text())
    return pd.concat(parts, ignore_index=True), infos


def boot_folds(eval_by_fold, n_boot, seed, fn):
    """Fold-stratified image bootstrap: per resample, multiplicities drawn within each fold; fn(fold, evalset, m) -> dict
    of numbers, averaged over folds per group. Returns estimate and draws: {key: (estimate, array of draws)}."""
    rng = np.random.default_rng(seed)
    est = {}
    draws = {}
    for b in range(-1, n_boot):
        acc = {}
        for k, es_by_group in eval_by_fold.items():
            n = len(next(iter(es_by_group.values())).images)
            m = np.ones(n) if b < 0 else rng.multinomial(n, np.full(n, 1 / n)).astype(float)
            for key, v in fn(k, es_by_group, m).items():
                acc.setdefault(key, []).append(v)
        vals = {key: float(np.nanmean(v)) for key, v in acc.items()}
        if b < 0:
            est = vals
        else:
            for key, v in vals.items():
                draws.setdefault(key, []).append(v)
    return {key: (est[key], np.array(draws.get(key, []))) for key in est}


def ci(d):
    v = d[~np.isnan(d)]
    return (np.nan, np.nan) if len(v) == 0 else tuple(np.percentile(v, [2.5, 97.5]))


def summarize(T, args):
    out = args.out_dir
    geoms = geometry_table(out)
    gt = gt_448(T, geoms)
    det, infos = load_fold_detections(out, "cv")
    region = dict(zip(T.image_id, T.region))
    fold_of = dict(zip(T.image_id, T.fold))
    group_imgs = {g: (lambda g=g: [i for i in T.image_id if g == "overall" or region[i] == g]) for g in ["overall"] + REGIONS}
    rounds = sorted(det["round"].unique())

    # per fold, per round, per group evaluation sets (each on that fold's images only)
    ev = {r: {k: {g: EvalSet([i for i in group_imgs[g]() if fold_of[i] == k], det[(det["round"] == r)], gt)
                  for g in ["overall"] + REGIONS} for k in range(5)} for r in rounds}

    def ap_fn(k, esg, m):
        out = {}
        for g, es in esg.items():
            a = es.ap(m[_subset_index(esg, g)])
            out[f"{g}|AP0.5"] = a[0.5]
            out[f"{g}|AP50:95"] = float(np.mean([a[t] for t in AP_IOUS]))
        return out

    res_rounds = {r: boot_folds(ev[r], args.n_boot, args.boot_seed, ap_fn) for r in rounds}
    # D1: round 2 minus round 1, paired (same multiplicities: same seed and same image order)
    d1 = res_rounds[2]["overall|AP0.5"][1] - res_rounds[1]["overall|AP0.5"][1]
    d1_est = res_rounds[2]["overall|AP0.5"][0] - res_rounds[1]["overall|AP0.5"][0]
    d1_lo, d1_hi = ci(d1)
    final_rounds = 1 if d1_hi < 0 else 2
    R = final_rounds

    # threshold (D2): pooled out-of-fold detections of detectors sharing the final detector's configuration
    fc = json.loads((out / "final_c.json").read_text()) if (out / "final_c.json").exists() else None
    folds_C = [infos[k]["C"] for k in range(5)]
    thr_source = None
    if fc is None:
        thr, thr_f1, thr_source = np.nan, np.nan, "final C missing: run --final-c"
    elif all(c == fc["C"] for c in folds_C):
        pooled = EvalSet(list(T.image_id), det[det["round"] == R], gt)
        thr, thr_f1 = f1_threshold(pooled)
        thr_source = "detection CV (every outer fold chose the final C)"
    else:
        try:
            tdet, tinfo = load_fold_detections(out, "thrcv")
            assert all(tinfo[k]["C"] == fc["C"] for k in range(5))
            pooled = EvalSet(list(T.image_id), tdet[tdet["round"] == R], gt)
            thr, thr_f1 = f1_threshold(pooled)
            thr_source = f"threshold CV with C fixed at {fc['C']} (outer folds chose {folds_C})"
        except SystemExit as e:
            thr, thr_f1, thr_source = np.nan, np.nan, f"threshold CV required: {e}"

    # threshold metrics on the D1 round's out-of-fold detections, pooled over folds (CIs: within-fold resampling)
    rows = []
    tm = {}
    if not np.isnan(thr):
        src = det[det["round"] == R] if thr_source.startswith("detection CV") else tdet[tdet["round"] == R]
        evt = {k: {g: EvalSet([i for i in group_imgs[g]() if fold_of[i] == k], src, gt) for g in ["overall"] + REGIONS}
               for k in range(5)}
        tm = _threshold_boot(evt, thr, args.n_boot, args.boot_seed)
    reach = pd.read_csv(args.tables_dir / "phase7_reachability.csv").set_index("group")
    for g in ["overall"] + REGIONS:
        rr = res_rounds[R]
        row = dict(group=g, n_images=len(group_imgs[g]()), n_boxes=int(sum(len(gt[i]) for i in group_imgs[g]())),
                   reach_iou50=float(reach.loc["all" if g == "overall" else g, "reach_iou50"]),
                   ap50=rr[f"{g}|AP0.5"][0], ap50_lo=ci(rr[f"{g}|AP0.5"][1])[0], ap50_hi=ci(rr[f"{g}|AP0.5"][1])[1],
                   ap50_95=rr[f"{g}|AP50:95"][0], ap50_95_lo=ci(rr[f"{g}|AP50:95"][1])[0],
                   ap50_95_hi=ci(rr[f"{g}|AP50:95"][1])[1],
                   ap50_pooled=EvalSet(group_imgs[g](), det[det["round"] == R], gt).ap()[0.5])
        for key in ("recall", "precision", "f1", "recall_iou30"):
            if (g, key) in tm:
                row.update({key: tm[(g, key)][0], f"{key}_lo": tm[(g, key)][1], f"{key}_hi": tm[(g, key)][2]})
        rows.append(row)
    table = pd.DataFrame(rows)
    rtab = pd.DataFrame([dict(round=r, group=g, ap50=res_rounds[r][f"{g}|AP0.5"][0],
                              ap50_lo=ci(res_rounds[r][f"{g}|AP0.5"][1])[0], ap50_hi=ci(res_rounds[r][f"{g}|AP0.5"][1])[1])
                         for r in rounds for g in ["overall"] + REGIONS])
    rtab = pd.concat([rtab, pd.DataFrame([dict(round="2 - 1", group="overall", ap50=d1_est, ap50_lo=d1_lo,
                                               ap50_hi=d1_hi)])], ignore_index=True)

    # image level: best-window score of the D1 round against the Phase 3 unified classifier, same images and folds
    best = det[(det["round"] == R) & (det["rank"] == 0)].set_index("image_id").score
    p3 = pd.read_csv(REPO_ROOT / "results" / "phase3" / "oof_rbfsvm__none.csv", float_precision="round_trip")
    j = T[["image_id", "region", "fold", "fractured"]].merge(p3[["image_id", "score"]], on="image_id")
    j["score_det"] = j.image_id.map(best)
    j["score_det"] = j.score_det.fillna(-np.inf)
    il = []
    for g in ["overall"] + REGIONS:
        d = j if g == "overall" else j[j.region == g]
        fb = fold_paired_bootstrap(d.fractured, d.score_det, d.score, d.fold, n_boot=args.n_boot, seed=args.boot_seed)
        per = lambda col, fn: np.mean([fn(x.fractured.to_numpy(), x[col].to_numpy()) for _, x in d.groupby("fold")])
        il.append(dict(group=g, det_pr_auc=per("score_det", _fast_ap), p3_pr_auc=per("score", _fast_ap),
                       d_pr_auc=fb["pr_auc"][0], d_pr_auc_lo=fb["pr_auc"][1], d_pr_auc_hi=fb["pr_auc"][2],
                       det_roc_auc=per("score_det", _fast_auc), p3_roc_auc=per("score", _fast_auc),
                       d_roc_auc=fb["roc_auc"][0], d_roc_auc_lo=fb["roc_auc"][1], d_roc_auc_hi=fb["roc_auc"][2]))
    iltab = pd.DataFrame(il)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.tables_dir / "phase7_cv_detection.csv", index=False)
    rtab.round(4).to_csv(args.tables_dir / "phase7_cv_rounds.csv", index=False)
    iltab.round(4).to_csv(args.tables_dir / "phase7_cv_image_level.csv", index=False)
    summ = dict(D1=dict(diff=d1_est, lo=d1_lo, hi=d1_hi, final_rounds=final_rounds,
                        verdict="1 round (round 2 is worse)" if final_rounds == 1 else "2 rounds (plan)"),
                C_per_fold=folds_C, final_C=None if fc is None else fc["C"], threshold=thr, threshold_f1_pooled=thr_f1,
                threshold_source=thr_source, hard_negatives={k: infos[k]["hard_negatives"] for k in range(5)},
                n_boot=args.n_boot, boot_seed=args.boot_seed, date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (out / "cv_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    fig_rounds(rtab, args.figures_dir)
    if not np.isnan(thr):
        fig_examples(T, det[det["round"] == R], gt, thr, args.figures_dir)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== Detection CV (training images, out of fold), D1 round =", R, "===")
    print(table.round(3).to_string(index=False))
    print("\n=== AP@50 per mining round (fold mean, 95% CI) ===")
    print(rtab.round(3).to_string(index=False))
    print(f"D1: round 2 - round 1 = {d1_est:+.3f} [{d1_lo:+.3f}, {d1_hi:+.3f}] -> {summ['D1']['verdict']}")
    print("\n=== Image level: best-window score vs Phase 3 (fold-mean, paired) ===")
    print(iltab.round(3).to_string(index=False))
    print(f"\nC per outer fold {folds_C}; final C {summ['final_C']}; box-score threshold {thr} ({thr_source})")
    if not args.no_log:
        o = table.set_index("group").loc["overall"]
        log_run(regions="per region + overall, training images (out of fold)", feature_set="Phase 7 window features",
                model="linear SVM sliding-window detector", imbalance_strategy="balanced + hard-negative mining",
                key_hyperparameters=f"D1 rounds {final_rounds}; C per fold {folds_C}; threshold {thr}",
                cv_score=f"AP@50={o.ap50:.3f} [{o.ap50_lo:.3f}, {o.ap50_hi:.3f}]; AP@50:95={o.ap50_95:.3f}",
                test_metrics="not used (CV only)", notes="Phase 7 detection CV summary (fold-mean, within-fold bootstrap)")
    return summ


def _subset_index(esg, g):
    """Indices of group g's images inside the overall image list of the same fold (for shared multiplicities)."""
    pos = {im: n for n, im in enumerate(esg["overall"].images)}
    return np.array([pos[i] for i in esg[g].images], int)


def _threshold_boot(evt, thr, n_boot, seed):
    """Recall / precision / F1 at IoU 0.5 and recall at IoU 0.3 at one threshold, counts POOLED over the folds (one
    threshold, one configuration); CIs from images resampled within each fold."""
    rng = np.random.default_rng(seed)
    est, draws = {}, {}
    for b in range(-1, n_boot):
        tot = {}
        for k, esg in evt.items():
            n = len(esg["overall"].images)
            m = np.ones(n) if b < 0 else rng.multinomial(n, np.full(n, 1 / n)).astype(float)
            for g in esg:
                mm = m[_subset_index(esg, g)]
                e = esg[g]
                sel = e.score >= thr
                wd = mm[e.img] * sel
                acc = tot.setdefault(g, np.zeros(4))
                acc += [wd @ e.flags[:, EVAL_IOUS.index(0.5)], wd.sum(), float(mm @ e.n_gt),
                        wd @ e.flags[:, EVAL_IOUS.index(0.3)]]
        vals = {}
        for g, (tp, nd, ng, tp30) in tot.items():
            vals[(g, "recall")] = tp / ng if ng else np.nan
            vals[(g, "precision")] = tp / nd if nd else 0.0
            vals[(g, "f1")] = 2 * tp / (nd + ng) if (nd + ng) else np.nan
            vals[(g, "recall_iou30")] = tp30 / ng if ng else np.nan
        if b < 0:
            est = vals
        else:
            for key, v in vals.items():
                draws.setdefault(key, []).append(v)
    return {key: (est[key],) + ci(np.array(draws.get(key, [np.nan]))) for key in est}


# ----------------------------------------------------------------------------- figures

def fig_rounds(rtab, figdir):
    setup()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    r = rtab[rtab["round"] != "2 - 1"]
    for g, col in zip(["overall"] + REGIONS, [COLORS["ink"], COLORS["region"], COLORS["unified"], COLORS["third"],
                                             COLORS["muted"]]):
        d = r[r.group == g]
        ax.errorbar(d["round"].astype(int) + (REGIONS.index(g) + 1) * 0.04 if g != "overall" else d["round"].astype(int),
                    d.ap50, yerr=np.vstack([d.ap50 - d.ap50_lo, d.ap50_hi - d.ap50]), fmt="-o", color=col, ecolor=col,
                    capsize=3, label="Overall" if g == "overall" else REGION_LABEL[g])
    ax.set_xticks([0, 1, 2], ["round 0", "round 1", "round 2"])
    ax.set_ylabel("AP@50 (fold mean)")
    header(fig, "Phase 7: AP@50 after each hard-negative mining round (training CV)", ax, ncols=5, top=0.80)
    save(fig, figdir / "phase7_cv_rounds.png", "Out-of-fold detections of the training images; error bars: 95% "
         "fold-stratified bootstrap over images. Rule D1 compares rounds 2 and 1.")


def fig_examples(T, det, gt, thr, figdir, n_good=4, n_bad=4):
    """Out-of-fold detections on training images: good (highest-scoring true positives) and bad (missed boxes,
    confident false positives on normal images)."""
    setup()
    top = det[det["rank"] == 0].set_index("image_id")
    frac = T[T.boxes.map(len) > 0]
    hit, miss = [], []
    for r in frac.itertuples():
        if r.image_id not in top.index:
            continue
        d = det[(det.image_id == r.image_id) & (det.score >= thr)]
        tp = match_flags(d[["x", "y", "w", "h"]].to_numpy(), d.score.to_numpy(), gt[r.image_id], 0.5)
        (hit if tp.any() else miss).append((top.loc[r.image_id, "score"], r.image_id))
    normal = T[T.boxes.map(len) == 0]
    fp = sorted([(top.loc[i, "score"], i) for i in normal.image_id if i in top.index], reverse=True)
    picks = ([("good", i) for _, i in sorted(hit, reverse=True)[:n_good]]
             + [("missed", i) for _, i in sorted(miss)[:n_bad // 2]] + [("false alarm", i) for _, i in fp[:n_bad // 2]])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    fp_of = dict(zip(T.image_id, T.file_path))
    reg = dict(zip(T.image_id, T.region))
    for ax, (kind, im) in zip(axes.ravel(), picks):
        img, _ = W.load_image(fp_of[im])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for x, y, w, h in gt[im]:
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=COLORS["third"], lw=1.6))
        d = det[(det.image_id == im) & (det.score >= thr)]
        for rr in d.itertuples():
            ax.add_patch(plt.Rectangle((rr.x, rr.y), rr.w, rr.h, fill=False, ec=COLORS["unified"], lw=1.2))
        ax.set_title(f"{kind}: {im[3:-4]} ({reg[im]})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("Phase 7 out-of-fold detections on training images (green: ground truth, orange: detections at the "
                 "operating threshold)", x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, figdir / "phase7_cv_examples.png")


# ----------------------------------------------------------------------------- gates and timing

def naive_window_features(img, w, h, i, j):
    """Features of one window computed on its own (independent of the dense maps): crop the level image around the
    window with an 8 px margin (enough for the gradients and LBP), run skimage HOG and LBP on the crop, slice the
    window, and compute the intensity statistics directly."""
    from skimage.feature import hog as sk_hog, local_binary_pattern
    from feature_extractors import _u8
    level = W.level_image(img, w, h)
    y0, x0, m = W.CELL * j, W.CELL * i, W.CELL
    ya, xa = max(0, y0 - m), max(0, x0 - m)
    yb, xb = min(level.shape[0], y0 + W.CANON + m), min(level.shape[1], x0 + W.CANON + m)
    crop = level[ya:yb, xa:xb]
    Hc = sk_hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys",
                feature_vector=False)
    oy, ox = (y0 - ya) // 8, (x0 - xa) // 8
    f_hog = Hc[oy:oy + 3, ox:ox + 3].ravel()
    # skimage's LBP interpolates the diagonal neighbours with a rounding error that depends on the absolute pixel
    # position, so near-tie pixels can code differently in a crop. The crop is therefore pasted at its own position
    # into an otherwise empty level-sized image: the window still only sees its neighbourhood.
    pasted = np.zeros(level.shape, np.uint8)
    pasted[ya:yb, xa:xb] = _u8(crop)
    codes = local_binary_pattern(pasted, 8, 1, method="uniform")[y0:y0 + 32, x0:x0 + 32]
    f_lbp = np.concatenate([np.bincount(codes[dy:dy + 16, dx:dx + 16].astype(int).ravel(), minlength=10) / 256
                            for dy in (0, 16) for dx in (0, 16)])
    patch = crop[y0 - ya:y0 - ya + 32, x0 - xa:x0 - xa + 32].astype(np.float64)
    return np.concatenate([f_hog, f_lbp, [patch.mean(), patch.std()]])


def run_gates(T, args):
    """Real-data gates on training images (before any model): box mapping figure and shapes, window grid and
    reachability against the design analysis, dense = naive features, folded linear scoring."""
    import phase7_window_design as D
    res = {}
    # window grid: the implementation's windows per type equal the design grid for every training image shape
    mism = []
    for (h0, w0), _ in T.groupby(["height", "width"]):
        g = D.geometry(int(h0), int(w0))
        design = [int(D.content_ok(*D.grid(w, h), w, h, g["content"]).sum()) for w, h in W.WINDOW_TYPES]
        if design != W.window_counts(g["content"]):
            mism.append((int(h0), int(w0)))
    res["grid_equal_design"] = dict(passed=not mism, mismatched_shapes=mism)
    # reachability from the implementation's windows equals the design table
    reach = pd.read_csv(args.tables_dir / "phase7_reachability.csv").set_index("group")
    got = {}
    for r in T.itertuples():
        g = D.geometry(r.height, r.width)
        if not r.boxes:
            continue
        B, _ = W.index_boxes(W.window_index(g["content"]))
        best = W.iou_matrix(W.map_boxes(r.boxes, g), B).max(1)
        got.setdefault(r.region, []).extend(best >= POS_IOU)
    allv = [v for vs in got.values() for v in vs]
    cmp = {"all": float(np.mean(allv)), **{k: float(np.mean(v)) for k, v in got.items()}}
    res["reachability_equal_design"] = dict(passed=all(abs(cmp[k] - reach.loc[k, "reach_iou50"]) < 5e-5 for k in cmp),
                                            implementation=cmp, design=reach.reach_iou50.to_dict())
    # box mapping figure (3 training fractured images per region) + shape / content checks on 12 images
    rng = np.random.default_rng(0)
    picks = [i for g in REGIONS for i in rng.choice(T.index[(T.region == g) & (T.boxes.map(len) > 0)], 3, replace=False)]
    setup()
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    ok_shape, ok_inside = True, True
    for ax, n in zip(axes.T.ravel(), picks):
        r = T.loc[n]
        gray = load_gray(r.file_path)
        ok_shape &= gray.shape == (r.height, r.width)
        img, geom = preprocess(gray, size=W.SIZE, mid=W.MID, norm="clahe")
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        x0, y0, x1, y1 = geom["content"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=COLORS["muted"], lw=1, ls="--"))
        for x, y, w, h in W.map_boxes(r.boxes, geom):
            ok_inside &= x >= x0 - 1 and y >= y0 - 1 and x + w <= x1 + 1 and y + h <= y1 + 1
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=COLORS["unified"], lw=1.6))
        ax.set_title(f"{r.image_id[3:-4]} ({r.region})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Phase 7 gate: training boxes mapped through preprocess() at 448 px (dashed: content rectangle)",
                 x=0.005, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, args.figures_dir / "phase7_box_mapping.png")
    res["box_mapping_figure"] = dict(images=[T.image_id[n] for n in picks], shapes_ok=bool(ok_shape),
                                     boxes_inside_content=bool(ok_inside), passed=bool(ok_shape and ok_inside))
    # dense = naive: 500 random windows on 10 random training images
    diffs = []
    imgs = rng.choice(len(T), 10, replace=False)
    for n in imgs:
        img, geom = W.load_image(T.file_path[n])
        maps, idx = W.image_maps(img), W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        sel = rng.choice(len(B), 50, replace=False)
        X = W.maps_features(maps, idx, sel)
        sizes = np.cumsum([0] + [len(jj) for jj, _, _ in idx])
        for row, s in zip(X, sel):
            t = int(Tt[s])
            loc = s - sizes[t]
            jj, ii, _ = idx[t]
            ref = naive_window_features(img, *W.WINDOW_TYPES[t], int(ii[loc]), int(jj[loc]))
            diffs.append(float(np.abs(row - ref).max()))
    res["dense_equals_naive"] = dict(windows=len(diffs), max_abs_diff=max(diffs), passed=max(diffs) <= 1e-5)
    # folded linear scoring = pipeline.decision_function, and maps scoring = X @ w + b
    img, geom = W.load_image(T.file_path[imgs[0]])
    maps, idx = W.image_maps(img), W.window_index(geom["content"])
    X = W.maps_features(maps, idx)
    yy = (rng.random(len(X)) < 0.2).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build("linsvm", "balanced", {"C": 0.01}, W.COL_WEIGHTS).fit(X[::5], yy[::5])
    w, b = W.fold_linear(pipe)
    X64 = X.astype(np.float64)
    d_fold = float(np.abs(X64 @ w + b - pipe.decision_function(X64)).max())
    d_maps = float(np.abs(W.maps_scores(maps, idx, w, b) - (X64 @ w + b)).max())
    res["folded_scoring"] = dict(max_abs_diff_vs_pipeline=d_fold, max_abs_diff_maps_vs_features=d_maps,
                                 passed=d_fold <= 1e-6 and d_maps <= 1e-6)
    res["all_passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gates.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k}: {'PASSED' if v['passed'] else 'FAILED'}  " + ", ".join(
                f"{a}={b}" for a, b in v.items() if a != "passed" and not isinstance(b, (dict, list))), flush=True)
    if not res["all_passed"]:
        raise SystemExit("a Phase 7 gate FAILED (results/phase7/gates.json)")
    return res


def run_timing(T, args):
    """Full scoring of 10 training images (maps from pixels, then scoring + NMS + mining-style selection)."""
    rng = np.random.default_rng(1)
    picks = rng.choice(len(T), 10, replace=False)
    w, b = rng.normal(size=W.N_FEAT) * 0.01, -0.5
    t_maps, t_score = [], []
    for n in picks:
        t0 = time.perf_counter()
        img, geom = W.load_image(T.file_path[n])
        maps = W.image_maps(img)
        t1 = time.perf_counter()
        idx = W.window_index(geom["content"])
        B, _ = W.index_boxes(idx)
        s = W.maps_scores(maps, idx, w, b)
        W.nms(B, s, NMS_IOU, KEEP_PER_IMAGE)
        W.maps_features(maps, idx, W.nms(B, s, NMS_IOU, MINE_PER_IMAGE))
        t_maps.append(t1 - t0)
        t_score.append(time.perf_counter() - t1)
    tm, ts = float(np.mean(t_maps)), float(np.mean(t_score))
    n_tr, n_te, nj = 3266, 817, args.n_jobs
    per_fold_scorings = 2 * 0.8 * n_tr + 0.2 * n_tr                  # two mining passes + held-out fold
    est = dict(maps_s_per_image=tm, score_s_per_image=ts, n_jobs=nj,
               cache_min=n_tr * tm / nj / 60, cv_min=5 * per_fold_scorings * (ts + 0.03) / nj / 60,
               threshold_cv_min=5 * per_fold_scorings * (ts + 0.03) / nj / 60,
               final_test_min=(n_tr * 2 * (ts + 0.03) + n_te * (tm + ts)) / nj / 60, cache_gb=n_tr * 5.1e-3)
    (args.out_dir / "timing.json").write_text(json.dumps(est, indent=1))
    print(f"timing on 10 training images: maps {tm:.2f} s/image, scoring + NMS + selection {ts:.3f} s/image")
    print(f"estimates with {nj} workers: cache {est['cache_min']:.0f} min ({est['cache_gb']:.0f} GB), detection CV "
          f"{est['cv_min']:.0f} min (+ the same again if the threshold CV is needed), final test "
          f"{est['final_test_min']:.0f} min; plus tune_cv and LinearSVC fits", flush=True)
    return est


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--patches", action="store_true")
    ap.add_argument("--cv-fold", type=int, nargs="+", default=None)
    ap.add_argument("--final-c", action="store_true")
    ap.add_argument("--threshold-cv", type=int, nargs="+", default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42, help="model seed")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--debug-subset", type=int, default=0, help="DEBUG: use only about N training images (no log)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase7")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "features" / "phase7")
    args = ap.parse_args(argv)
    if args.debug_subset:
        args.no_log = True
        if args.out_dir == REPO_ROOT / "results" / "phase7":
            ap.error("--debug-subset needs its own --out-dir (debug runs never write into results/phase7)")
    T = train_table(args.debug_subset or None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(T)} training images ({int(T.fractured.sum())} fractured, {int(T.boxes.map(len).sum())} boxes)",
          flush=True)
    if args.gates:
        run_gates(T, args)
    if args.timing:
        run_timing(T, args)
    if args.cache:
        build_cache(T, args)
    if args.patches:
        R0 = build_patches(T, args)
        if not args.no_log:
            log_run(regions="all (pooled), training images", feature_set="Phase 7 window features",
                    model="n/a (round-0 patches, nothing fitted)", imbalance_strategy="n/a",
                    key_hyperparameters=f"positives IoU>={POS_IOU}; negative pools of {POOL} (IoU<{NEG_IOU}), seed 0",
                    cv_score=f"{len(R0['X_pos'])} positive windows", test_metrics="not used (training images only)",
                    notes="Phase 7 round-0 patches")
    if args.cv_fold is not None or args.final_c or args.threshold_cv is not None:
        R0 = load_round0(args.cache_dir, T)
    for k in args.cv_fold or []:
        if (args.out_dir / f"cv_fold{k}.json").exists() and not args.force:
            print(f"[skip] fold {k} already done (use --force to redo)")
            continue
        run_fold(T, R0, k, args)
    if args.final_c:
        X, y, f, _ = round0_set(R0, T, np.ones(len(T), bool))
        tc = tune_cv(X, y, f, "linsvm", "balanced", W.COL_WEIGHTS, grid={"C": C_GRID}, seed=args.seed,
                     n_jobs=args.n_jobs)
        fc = dict(C=tc["params"]["C"], tune_pr_auc={str(c["C"]): a for c, a in zip(tc["configs"], tc["all_pr_auc"])},
                  n_pos=int(y.sum()), n_neg=int((y == 0).sum()), date=pd.Timestamp.now().isoformat(timespec="seconds"))
        (args.out_dir / "final_c.json").write_text(json.dumps(fc, indent=1))
        print(f"final C = {fc['C']} (patch PR-AUC per C: {fc['tune_pr_auc']})", flush=True)
        if not args.no_log:
            log_run(regions="all (pooled), training images", feature_set="Phase 7 window features",
                    model="linear SVM patch classifier", imbalance_strategy="balanced",
                    key_hyperparameters=f"tune_cv on all training round-0 patches, grid {C_GRID}",
                    cv_score=f"C={fc['C']}; patch PR-AUC {fc['tune_pr_auc']}", test_metrics="not used (CV only)",
                    notes="Phase 7 final detector's C (needed by the threshold rule)")
    if args.threshold_cv is not None:
        fc = json.loads((args.out_dir / "final_c.json").read_text())
        for k in args.threshold_cv:
            if (args.out_dir / f"thrcv_fold{k}.json").exists() and not args.force:
                print(f"[skip] threshold-CV fold {k} already done")
                continue
            run_fold(T, R0, k, args, C_fixed=fc["C"], tag="thrcv")
    if args.summarize:
        summarize(T, args)


if __name__ == "__main__":
    main()
