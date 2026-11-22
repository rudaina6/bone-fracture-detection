"""READ-ONLY diagnosis of the Phase 7 detector for a possible Phase 7b. TRAINING IMAGES AND CV OUTPUTS ONLY.
Writes only into this scratch folder; nothing under the repository is written.

    python diag_phase7b.py a   # per outer fold: round-0 refit, held-out patch metrics, fracture-window ranks
    python diag_phase7b.py b   # fold 0: rounds 0-2 reproduced with window coordinates of the mined negatives
    python diag_phase7b.py c   # appearance of windows (bone / soft tissue / air fractions, edges, bright, padding)
    python diag_phase7b.py r   # report
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(r"D:\bone-fracture-detection")
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import phase7_detect as P
import phase7_windows as W
from preprocess import load_gray, preprocess

HERE = Path(__file__).resolve().parent
CACHE = REPO / "features" / "phase7"
RES = REPO / "results" / "phase7"
C = 0.1
NJ = 4
MINE_NJ = 4          # = the CV run's n_jobs: run_parallel's chunk order sets the order of mined negatives
FPRS = (1e-2, 1e-3, 1e-4)


def setup():
    T = P.train_table()
    R0 = P.load_round0(CACHE, T)
    gt = P.gt_448(T, P.geometry_table(RES))
    return T, R0, gt


def tpr_at_fpr(pos, neg, fprs=FPRS):
    neg = np.sort(neg)[::-1]
    return {f: float((pos > (neg[int(f * len(neg))] if int(f * len(neg)) < len(neg) else -np.inf)).mean()) for f in fprs}


def patch_metrics(w, b, R0, T, k):
    """Held-out patches of fold k's images: positives vs (i) the protocol's 1:3 negatives, (ii) all pool negatives."""
    fold = T.fold.to_numpy()
    Xh, yh, _, _ = P.round0_set(R0, T, fold == k)
    sh = Xh.astype(np.float64) @ w + b
    pos = R0["X_pos"][fold[R0["img_pos"]] == k].astype(np.float64) @ w + b
    neg = R0["X_pool"][fold[R0["img_pool"]] == k].astype(np.float64) @ w + b
    yp = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sp = np.r_[pos, neg]
    out = dict(fold=k, n_pos=len(pos), n_neg_13=int((yh == 0).sum()), n_neg_pool=len(neg),
               pr_auc_13=average_precision_score(yh, sh), roc_auc_13=roc_auc_score(yh, sh),
               pr_auc_pool=average_precision_score(yp, sp), roc_auc_pool=roc_auc_score(yp, sp),
               pos_score_gt0=float((pos > 0).mean()), pool_neg_score_gt0=float((neg > 0).mean()))
    out.update({f"tpr_at_fpr_{f:g}": v for f, v in tpr_at_fpr(pos, neg).items()})
    return out


def rank_chunk(rows, w, b):
    """Per ground-truth box of fold-k fractured images: rank of its best positive window among all windows of the image
    (overall and within each window type), and whether NMS's 10 boxes hit it."""
    out = []
    for image_id, region, boxes in rows:
        maps, geom = W.load_maps(P.map_path(CACHE, image_id))
        idx = W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        s = W.maps_scores(maps, idx, w, b)
        g = W.map_boxes(boxes, geom)
        I = W.iou_matrix(B, g)
        keep = W.nms(B, s, P.NMS_IOU, P.KEEP_PER_IMAGE)
        for j in range(len(g)):
            pos = np.flatnonzero(I[:, j] >= P.POS_IOU)
            row = dict(image_id=image_id, region=region, box=j, side=float(np.sqrt(g[j, 2] * g[j, 3])),
                       n_windows=len(s), n_pos=len(pos), nms_hit=bool((I[keep, j] >= P.POS_IOU).any()),
                       top_score=float(s.max()))
            if len(pos):
                kb = pos[np.argmax(s[pos])]
                row.update(best_type=int(Tt[kb]), best_score=float(s[kb]), rank=int((s > s[kb]).sum()) + 1)
                for t in range(len(W.WINDOW_TYPES)):
                    pt = pos[Tt[pos] == t]
                    if len(pt):
                        bt = s[pt].max()
                        row[f"rank_all_t{t}"] = int((s > bt).sum()) + 1
                        row[f"rank_in_t{t}"] = int((s[Tt == t] > bt).sum()) + 1
                        row[f"n_t{t}"] = int((Tt == t).sum())
            out.append(row)
    return out


def ranks(T, use_rows, w, b):
    rows = [(r.image_id, r.region, r.boxes) for r in T[use_rows].itertuples() if r.boxes]
    return pd.DataFrame(P.run_parallel(rank_chunk, rows, NJ, w, b))


def part_a():
    T, R0, gt = setup()
    fold = T.fold.to_numpy()
    pm, rk = [], []
    for k in range(5):
        t0 = time.time()
        X, y, f, img = P.round0_set(R0, T, fold != k)
        w, b = P.fit_linear(X, y, C, 42)
        np.savez(HERE / f"wb_fold{k}_r0.npz", w=w, b=b)
        tr = X.astype(np.float64) @ w + b
        m = patch_metrics(w, b, R0, T, k)
        m.update(train_roc_auc=roc_auc_score(y, tr), train_pr_auc=average_precision_score(y, tr), round=0)
        pm.append(m)
        r = ranks(T, fold == k, w, b).assign(fold=k, round=0)
        rk.append(r)
        print(f"fold {k}: held-out patch PR-AUC 1:3 {m['pr_auc_13']:.3f}, ROC-AUC pool {m['roc_auc_pool']:.4f}, "
              f"TPR@FPR1e-3 {m['tpr_at_fpr_0.001']:.3f}; median rank {r['rank'].median():.0f} ({time.time() - t0:.0f} s)",
              flush=True)
    pd.DataFrame(pm).to_csv(HERE / "a_patch_metrics.csv", index=False)
    pd.concat(rk).to_csv(HERE / "a_ranks.csv", index=False)


def mine_chunk(rows, w, b):
    out = []
    for n, image_id, boxes in rows:
        maps, geom = W.load_maps(P.map_path(CACHE, image_id))
        idx = W.window_index(geom["content"])
        B, Tt = W.index_boxes(idx)
        s = W.maps_scores(maps, idx, w, b)
        g = W.map_boxes(boxes, geom)
        best = W.iou_matrix(B, g).max(1) if len(g) else np.zeros(len(B))
        cand = np.flatnonzero((best < P.NEG_IOU) & (s > 0))
        keep = cand[W.nms(B[cand], s[cand], P.NMS_IOU, P.MINE_PER_IMAGE)] if len(cand) else cand
        out.append((n, W.maps_features(maps, idx, keep), B[keep], Tt[keep], s[keep], len(cand), len(s)))
    return out


def part_b():
    """Fold 0, exactly as run_fold (C 0.1, seed 42), keeping the mined windows' coordinates and per-round stats."""
    T, R0, gt = setup()
    fold = T.fold.to_numpy()
    k = 0
    use = fold != k
    X, y, f, img = P.round0_set(R0, T, use)
    src = np.zeros(len(y), int)                      # 0: round-0 patch, r: mined in round r
    wbs = [P.fit_linear(X, y, C, 42)]
    mined = []
    rows = [(n, r.image_id, r.boxes) for n, r in enumerate(T.itertuples()) if use[n]]
    stats = []
    for r in range(1, P.ROUNDS + 1):
        t0 = time.time()
        res = P.run_parallel(mine_chunk, rows, MINE_NJ, *wbs[-1])
        Xh = np.concatenate([q[1] for q in res])
        mined.append(pd.DataFrame(dict(n=np.concatenate([np.full(len(q[2]), q[0]) for q in res]),
                                       x=np.concatenate([q[2][:, 0] for q in res]),
                                       y=np.concatenate([q[2][:, 1] for q in res]),
                                       w=np.concatenate([q[2][:, 2] for q in res]),
                                       h=np.concatenate([q[2][:, 3] for q in res]),
                                       wtype=np.concatenate([q[3] for q in res]),
                                       score=np.concatenate([q[4] for q in res]))).assign(round=r))
        stats.append(dict(round=r, mined=len(Xh), cand_median=float(np.median([q[5] for q in res])),
                          cand_p10=float(np.percentile([q[5] for q in res], 10)),
                          cand_share_median=float(np.median([q[5] / q[6] for q in res]))))
        X, y = np.concatenate([X, Xh]), np.r_[y, np.zeros(len(Xh), int)]
        src = np.r_[src, np.full(len(Xh), r)]
        wbs.append(P.fit_linear(X, y, C, 42))
        print(f"round {r}: mined {len(Xh)} ({time.time() - t0:.0f} s); eligible windows with score > 0 per image: "
              f"median {stats[-1]['cand_median']:.0f}", flush=True)
    pd.concat(mined).merge(T[["image_id"]].reset_index().rename(columns={"index": "n"}), on="n").to_csv(
        HERE / "b_mined.csv", index=False)
    pd.DataFrame(stats).to_csv(HERE / "b_mining_stats.csv", index=False)
    # per-round model diagnostics on the final patch set (training) and on held-out fold 0
    rows_out, rk = [], []
    y_all = y
    for r, (w, b) in enumerate(wbs):
        np.savez(HERE / f"wb_fold0_r{r}.npz", w=w, b=b)
        seen = src <= r
        s_tr = X[seen].astype(np.float64) @ w + b
        yy = y_all[seen]
        npos, nneg = int(yy.sum()), int((yy == 0).sum())
        d = dict(round=r, n_pos=npos, n_neg=nneg, neg_per_pos=nneg / npos,
                 w_pos=len(yy) / (2 * npos), w_neg=len(yy) / (2 * nneg),
                 train_pos_score_gt0=float((s_tr[yy == 1] > 0).mean()),
                 train_pos_score_median=float(np.median(s_tr[yy == 1])),
                 train_r0neg_score_gt0=float((X[src == 0][y_all[src == 0] == 0].astype(np.float64) @ w + b > 0).mean()))
        for q in (1, 2):
            if (src == q).any():
                d[f"mined_r{q}_score_gt0"] = float((X[src == q].astype(np.float64) @ w + b > 0).mean())
        d.update({f"heldout_{a}": v for a, v in patch_metrics(w, b, R0, T, k).items() if a != "fold"})
        d["w_norm_hog"], d["w_norm_lbp"], d["w_mean"], d["w_std"] = (float(np.linalg.norm(w[:324])),
                                                                    float(np.linalg.norm(w[324:364])), w[364], w[365])
        d["bias"] = b
        if r:
            p = wbs[r - 1][0]
            d["cos_prev_w"] = float(w @ p / np.linalg.norm(w) / np.linalg.norm(p))
        rows_out.append(d)
        rk.append(ranks(T, fold == k, w, b).assign(fold=k, round=r))
        print(f"round {r}: {npos} pos / {nneg} neg, held-out ROC-AUC pool {d['heldout_roc_auc_pool']:.4f}, "
              f"median rank {rk[-1]['rank'].median():.0f}", flush=True)
    saved = pd.read_csv(RES / "cv_fold0_detections.csv", float_precision="round_trip")
    worst = 0.0
    for im in T.image_id[fold == k].to_numpy()[::30]:
        maps, geom = W.load_maps(P.map_path(CACHE, im))
        idx = W.window_index(geom["content"])
        B, _ = W.index_boxes(idx)
        for r, (w, b) in enumerate(wbs):
            s = W.maps_scores(maps, idx, w, b)
            kp = W.nms(B, s, P.NMS_IOU, P.KEEP_PER_IMAGE)
            sv = saved[(saved.image_id == im) & (saved["round"] == r)].sort_values("rank")
            worst = max(worst, float(np.abs(sv.score.to_numpy() - s[kp]).max()))
    print(f"reproduction of the saved fold-0 detections, rounds 0-2: max |score difference| = {worst:.2e}", flush=True)
    pd.DataFrame(rows_out).assign(repro_max_diff=worst).to_csv(HERE / "b_rounds.csv", index=False)
    pd.concat(rk).to_csv(HERE / "b_ranks.csv", index=False)


# ----------------------------------------------------------------------------- appearance

def appearance_chunk(rows):
    from skimage.feature import canny
    from skimage.filters import gaussian, sobel, threshold_multiotsu
    out = []
    for pos, image_id, file_path, wins in rows:
        gray = load_gray(file_path)
        cl, geom = preprocess(gray, size=W.SIZE, mid=W.MID, norm="clahe")
        raw, _ = preprocess(gray, size=W.SIZE, mid=W.MID, norm="none")
        x0, y0, x1, y1 = geom["content"]
        content = np.zeros(cl.shape, bool)
        content[y0:y1, x0:x1] = True
        sm = gaussian(raw, 2)
        t1, t2 = threshold_multiotsu(sm[content], classes=3)
        bone, tissue = (sm >= t2) & content, (sm >= t1) & (sm < t2) & content
        air = (sm < t1) & content
        edges = canny(cl, sigma=2) & content
        grad = sobel(cl) * content
        g_ref = float(grad[content].mean())
        bright = (raw >= 0.95) & content
        layers = dict(clahe=cl, raw=raw, bone=bone, tissue=tissue, air=air, edges=edges, grad=grad / g_ref,
                      bright=bright, content=content)
        II = {k: np.pad(np.cumsum(np.cumsum(v.astype(np.float64), 0), 1), ((1, 0), (1, 0))) for k, v in layers.items()}
        xs, ys, ws, hs = (np.round(wins[:, c]).astype(int) for c in range(4))
        xe, ye = np.minimum(xs + ws, W.SIZE), np.minimum(ys + hs, W.SIZE)
        area = (xe - xs) * (ye - ys)
        stats = {k: (A[ye, xe] - A[ys, xe] - A[ye, xs] + A[ys, xs]) / area for k, A in II.items()}
        inside = np.maximum(stats["content"], 1e-9)
        res = dict(_row=np.full(len(wins), pos), _k=np.arange(len(wins)), image_id=np.full(len(wins), image_id), t_air_tissue=np.full(len(wins), t1),
                   t_tissue_bone=np.full(len(wins), t2), pad_frac=1 - stats["content"])
        for k in ("clahe", "raw", "edges", "grad", "bright", "bone", "tissue", "air"):
            res[k] = stats[k] / inside                 # per content pixel of the window
        out.append(pd.DataFrame(res))
    return out


def used_counts(R0, T):
    """Per image, how many of its pool negatives the final detector's round 0 uses (round0_set's count logic)."""
    use = np.ones(len(T), bool)
    n_neg = P.NEG_PER_POS * len(R0["X_pos"])
    elig = R0["n_elig"].astype(float)
    avail = np.bincount(R0["img_pool"], minlength=len(T))
    counts = np.random.default_rng(0).multinomial(n_neg, elig / elig.sum())
    over = counts - np.minimum(counts, avail)
    counts = np.minimum(counts, avail)
    while over.sum():
        spare = np.flatnonzero(counts < avail)
        np.add.at(counts, np.random.default_rng(1).choice(spare, size=int(over.sum())), 1)
        over = counts - np.minimum(counts, avail)
        counts = np.minimum(counts, avail)
    return counts


def check_pools(T, gt, n_check=6):
    """The recomputed pool windows reproduce the cached round-0 pool features exactly."""
    R0 = P.load_round0(CACHE, T)
    sets = None
    worst = 0.0
    for n in np.random.default_rng(5).choice(len(T), n_check, replace=False):
        r = T.iloc[n]
        maps, geom = W.load_maps(P.map_path(CACHE, r.image_id))
        idx = W.window_index(geom["content"])
        B, _ = W.index_boxes(idx)
        g = gt[r.image_id]
        best = W.iou_matrix(B, g).max(1) if len(g) else np.zeros(len(B))
        elig = np.flatnonzero(best < P.NEG_IOU)
        pool = np.random.default_rng([0, n]).choice(elig, size=min(P.POOL, len(elig)), replace=False)
        ref = R0["X_pool"][R0["img_pool"] == n][np.argsort(R0["rank_pool"][R0["img_pool"] == n])]
        worst = max(worst, float(np.abs(W.maps_features(maps, idx, pool) - ref).max()))
    print(f"pool recomputation vs cached round-0 pool features ({n_check} images): max |difference| = {worst:.1e}")
    return worst


def window_sets(T, gt):
    """Every window to describe, per training image: ground-truth positive windows, the 40 pool negatives (recomputed
    exactly as _patch_chunk drew them), CV detections of rounds 0-2, fold-0 mined negatives of rounds 1-2."""
    geo = pd.read_csv(RES / "train_geometry.csv").set_index("image_id")
    det = pd.concat([pd.read_csv(RES / f"cv_fold{k}_detections.csv").assign(fold=k) for k in range(5)])
    mined = pd.read_csv(HERE / "b_mined.csv") if (HERE / "b_mined.csv").exists() else None
    used = used_counts(P.load_round0(CACHE, T), T)        # pool depth used by the final detector's round 0
    sets = {}
    for n, r in enumerate(T.itertuples()):
        q = geo.loc[r.image_id]
        content = (int(q.cx0), int(q.cy0), int(q.cx1), int(q.cy1))
        idx = W.window_index(content)
        B, Tt = W.index_boxes(idx)
        g = gt[r.image_id]
        best = W.iou_matrix(B, g).max(1) if len(g) else np.zeros(len(B))
        elig = np.flatnonzero(best < P.NEG_IOU)
        pool = np.random.default_rng([0, n]).choice(elig, size=min(P.POOL, len(elig)), replace=False)
        parts = [pd.DataFrame(dict(kind=np.where(np.arange(len(pool)) < used[n], "pool_neg_used", "pool_neg_unused"),
                                   wtype=Tt[pool], score=np.nan, rank=np.arange(len(pool)),
                                   x=B[pool, 0], y=B[pool, 1], w=B[pool, 2], h=B[pool, 3]))]
        pos = np.flatnonzero(best >= P.POS_IOU)
        if len(pos):
            parts.append(pd.DataFrame(dict(kind="gt_pos", wtype=Tt[pos], score=np.nan, rank=-1, x=B[pos, 0],
                                           y=B[pos, 1], w=B[pos, 2], h=B[pos, 3])))
        sets[r.image_id] = (n, parts)
    for (im, rnd), d in det.groupby(["image_id", "round"]):
        g = gt[im]
        tp = np.zeros(len(d), bool)
        if len(g):
            tp = P.match_flags(d[["x", "y", "w", "h"]].to_numpy(), d.score.to_numpy(), g, 0.5)
        sets[im][1].append(pd.DataFrame(dict(kind=np.where(tp, f"det_r{rnd}_tp", f"det_r{rnd}_fp"), wtype=d.wtype,
                                             score=d.score, rank=d["rank"], x=d.x, y=d.y, w=d.w, h=d.h)))
    if mined is not None:
        for (im, rnd), d in mined.groupby(["image_id", "round"]):
            sets[im][1].append(pd.DataFrame(dict(kind=f"mined_r{rnd}", wtype=d.wtype, score=d.score, rank=-1, x=d.x,
                                                 y=d.y, w=d.w, h=d.h)))
    return sets


def part_c():
    T, R0, gt = setup()
    assert check_pools(T, gt) == 0.0
    sets = window_sets(T, gt)
    fp = dict(zip(T.image_id, T.file_path))
    rows, meta = [], []
    for pos, (im, (n, parts)) in enumerate(sets.items()):
        d = pd.concat(parts, ignore_index=True).assign(image_id=im)
        meta.append(d)
        rows.append((pos, im, fp[im], d[["x", "y", "w", "h"]].to_numpy(float)))
    t0 = time.time()
    res = P.run_parallel(appearance_chunk, rows, NJ)
    A = pd.concat(res, ignore_index=True).sort_values(["_row", "_k"], kind="stable").reset_index(drop=True)
    A = A.drop(columns=["_row", "_k"])
    M = pd.concat(meta, ignore_index=True)
    assert len(A) == len(M) and (A.image_id.to_numpy() == M.image_id.to_numpy()).all()
    out = pd.concat([M, A.drop(columns="image_id")], axis=1)
    m = pd.read_csv(REPO / "data" / "master.csv")[["image_id", "hardware"]]
    out = out.merge(T[["image_id", "region", "fold", "fractured"]], on="image_id").merge(m, on="image_id")
    out.to_csv(HERE / "c_windows.csv.gz", index=False)
    print(f"appearance of {len(out)} windows on {len(sets)} images in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    {"a": part_a, "b": part_b, "c": part_c}[sys.argv[1]]()
