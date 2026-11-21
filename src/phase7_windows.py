"""Phase 7: sliding windows and their canonical features, computed densely (protocol: docs/phase7_localization.md).

    from phase7_windows import load_image, image_maps, window_index, maps_features, maps_scores, fold_linear
    img, geom = load_image(rel_path)                   # 448 px, CLAHE, mid 448 (the shared preprocessing)
    maps = image_maps(img)                             # per window type: HOG blocks, LBP cell counts, intensity sums
    idx = window_index(geom["content"])                # the kept windows of every type and their xywh at 448 px
    X = maps_features(maps, idx)                       # features (N, 366) of every kept window (or a selection)
    s = maps_scores(maps, idx, w, b)                   # linear scores of every kept window, without building X

Window set (adopted at the STOP 1 review): base sizes 16, 40 and 64 px at 448, each with aspect 1:1 (s x s), 1:2
(s x 2s) and 2:1 (2s x s) -> 9 types. Every window is resampled to a canonical 32 x 32 patch: per type, the 448 px
image is resized by (32/w, 32/h) into one pyramid level, the feature maps are computed once per level, and a window
is a slice of them. Window (i, j) of a type starts at cell (i, j) of its level, i.e. image pixels x = i * w/4,
y = j * h/4 (stride = one 8 px cell = window/4). Windows with less than half their area in the content rectangle
are dropped (the window-design rule).
    HOG        9 orientations, 8 px cells, 2 x 2 blocks, L2-Hys: the 3 x 3 blocks inside the window's 4 x 4 cells (324)
    LBP        uniform, P=8, R=1 (codes 0-9): 10-bin histograms on the window's 2 x 2 quarters, each summing to 1 (40)
    intensity  mean and standard deviation of the level image over the window (2)
Because windows sit on the cell grid, LBP quarters and intensity sums are sums of per-cell counts / sums (exact).
Family block weights 1/sqrt(dims) (COL_WEIGHTS) go into model_zoo.build like every other phase; fold_linear turns the
fitted pipeline into one weight vector, and maps_scores applies it to every window as a correlation over the maps.
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from feature_extractors import _u8
from preprocess import load_gray, preprocess

SIZE, MID, CANON, CELL = 448, 448, 32, 8
BASE_SIZES = (16, 40, 64)
ASPECTS = ((1, 1), (1, 2), (2, 1))                      # (w, h) multiples of the base size
WINDOW_TYPES = tuple((s * a, s * b) for s in BASE_SIZES for a, b in ASPECTS)
CONTENT_MIN = 0.5
FAMILY_DIMS = dict(hog=324, lbp=40, intensity=2)
N_FEAT = sum(FAMILY_DIMS.values())
COL_WEIGHTS = np.concatenate([np.full(d, d ** -0.5, np.float32) for d in FAMILY_DIMS.values()])
LBP_P, LBP_R, LBP_BINS = 8, 1, 10
QUARTERS = ((0, 0), (0, 2), (2, 0), (2, 2))            # quarter offsets in cells: TL, TR, BL, BR


def load_image(rel_path, data_dir=None):
    """The shared preprocessing at the Phase 7 working size: (448 x 448 float32 image, geometry)."""
    return preprocess(load_gray(rel_path, data_dir), size=SIZE, mid=MID, norm="clahe")


def map_boxes(boxes_xywh, geom):
    """Original-pixel boxes [x, y, w, h] -> 448 px coordinates with preprocess()'s geometry."""
    b = np.asarray(boxes_xywh, dtype=float).reshape(-1, 4)
    s = geom["scale"]
    return np.c_[b[:, 0] * s + geom["off_x"], b[:, 1] * s + geom["off_y"], b[:, 2] * s, b[:, 3] * s]


def level_shape(w, h):
    """(rows, cols) of the pyramid level of window type w x h."""
    return round(SIZE * CANON / h), round(SIZE * CANON / w)


def level_image(img, w, h):
    rows, cols = level_shape(w, h)
    interp = cv2.INTER_AREA if (CANON < w and CANON < h) else cv2.INTER_LINEAR
    return cv2.resize(img, (cols, rows), interpolation=interp)


def positions(w, h):
    """Top-left corners (448 px) of the type's windows along x and y: one per level cell that starts a 4 x 4 window."""
    rows, cols = level_shape(w, h)
    return np.arange(cols // CELL - 3) * (w / 4), np.arange(rows // CELL - 3) * (h / 4)


def content_mask(xs, ys, w, h, content):
    """(len(ys), len(xs)) mask of windows with at least CONTENT_MIN of their area inside the content rectangle."""
    x0, y0, x1, y1 = content
    fx = np.clip(np.minimum(xs + w, x1) - np.maximum(xs, x0), 0, None) / w
    fy = np.clip(np.minimum(ys + h, y1) - np.maximum(ys, y0), 0, None) / h
    return fy[:, None] * fx[None, :] >= CONTENT_MIN


def hog_blocks(level):
    """HOG of a whole level, unflattened: (block rows, block cols, 2, 2, 9)."""
    return hog(level, orientations=9, pixels_per_cell=(CELL, CELL), cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=False)


def lbp_codes(level):
    return local_binary_pattern(_u8(level), LBP_P, LBP_R, method="uniform").astype(np.uint8)


# ----------------------------------------------------------------------------- per-image maps

def type_maps(img, w, h):
    """Feature maps of one window type's level: HOG blocks (by, bx, 36) float32, LBP per-cell code counts
    (cy, cx, 10) uint8, intensity per-cell sums of x and x^2 (cy, cx, 2) float64."""
    level = level_image(img, w, h)
    cy, cx = level.shape[0] // CELL, level.shape[1] // CELL
    B = hog_blocks(level).reshape(cy - 1, cx - 1, 36).astype(np.float32)
    codes = lbp_codes(level)[:cy * CELL, :cx * CELL]
    cell = (np.arange(cy)[:, None] * cx + np.arange(cx)[None, :]).repeat(CELL, 0).repeat(CELL, 1)
    L = np.bincount((cell * LBP_BINS + codes).ravel(), minlength=cy * cx * LBP_BINS)
    L = L.reshape(cy, cx, LBP_BINS).astype(np.uint8)
    lv = level[:cy * CELL, :cx * CELL].astype(np.float64).reshape(cy, CELL, cx, CELL)
    S = np.stack([lv.sum((1, 3)), (lv * lv).sum((1, 3))], axis=-1)
    return dict(hog=B, lbp=L, inten=S)


def image_maps(img, types=WINDOW_TYPES):
    return [type_maps(img, w, h) for w, h in types]


def window_index(content, types=WINDOW_TYPES):
    """Per type: (jj, ii, xywh) of the kept windows; the global window order is by type, then row-major."""
    out = []
    for w, h in types:
        xs, ys = positions(w, h)
        jj, ii = np.nonzero(content_mask(xs, ys, w, h, content))
        out.append((jj, ii, np.c_[xs[ii], ys[jj], np.full(len(jj), w, float), np.full(len(jj), h, float)]))
    return out


def index_boxes(index):
    """(xywh (N, 4), type id (N,)) of every kept window in the global order."""
    return (np.concatenate([b for _, _, b in index]),
            np.concatenate([np.full(len(b), t, np.int16) for t, (_, _, b) in enumerate(index)]))


def window_counts(content, types=WINDOW_TYPES):
    """Number of kept windows per type for an image with this content rectangle (no pixels needed)."""
    return [len(jj) for jj, _, _ in window_index(content, types)]


def _pair_sums(A):
    """Sums over 2 x 2 neighbouring cells: (cy - 1, cx - 1, ...)."""
    return A[:-1, :-1] + A[1:, :-1] + A[:-1, 1:] + A[1:, 1:]


def _type_features(M, jj, ii):
    n = len(jj)
    V = np.lib.stride_tricks.sliding_window_view(M["hog"], (3, 3), axis=(0, 1))     # (ny, nx, 36, 3, 3)
    f_hog = V[jj, ii].transpose(0, 2, 3, 1).reshape(n, 324)
    Q = _pair_sums(M["lbp"].astype(np.int32))
    f_lbp = np.concatenate([Q[jj + dy, ii + dx] for dy, dx in QUARTERS], axis=1) / (CANON // 2) ** 2
    S2 = _pair_sums(M["inten"])
    S4 = sum(S2[jj + dy, ii + dx] for dy, dx in QUARTERS)
    mean = S4[:, 0] / CANON ** 2
    std = np.sqrt(np.maximum(S4[:, 1] / CANON ** 2 - mean ** 2, 0.0))
    return np.concatenate([f_hog, f_lbp, mean[:, None], std[:, None]], axis=1).astype(np.float32)


def maps_features(maps, index, sel=None):
    """Features (n, 366) float32 of every kept window, or of the windows at global positions `sel`."""
    sizes = np.cumsum([0] + [len(jj) for jj, _, _ in index])
    if sel is None:
        return np.concatenate([_type_features(M, jj, ii) for M, (jj, ii, _) in zip(maps, index)])
    sel = np.asarray(sel, int)
    out = np.zeros((len(sel), N_FEAT), np.float32)
    for t, (M, (jj, ii, _)) in enumerate(zip(maps, index)):
        m = (sel >= sizes[t]) & (sel < sizes[t + 1])
        if m.any():
            loc = sel[m] - sizes[t]
            out[m] = _type_features(M, jj[loc], ii[loc])
    return out


def maps_scores(maps, index, w, b):
    """X @ w + b for every kept window (global order), computed as a correlation over the maps (no X is built)."""
    w = np.asarray(w, np.float64)
    W_hog, W_lbp, w_mean, w_std = w[:324].reshape(3, 3, 36), w[324:364].reshape(4, LBP_BINS), w[364], w[365]
    out = []
    for M, (jj, ii, _) in zip(maps, index):
        H = M["hog"].astype(np.float64)
        s = np.zeros(len(jj))
        for dy in range(3):
            for dx in range(3):
                s += (H @ W_hog[dy, dx])[jj + dy, ii + dx]
        Q = _pair_sums(M["lbp"].astype(np.float64)) / (CANON // 2) ** 2
        for q, (dy, dx) in enumerate(QUARTERS):
            s += (Q @ W_lbp[q])[jj + dy, ii + dx]
        S2 = _pair_sums(M["inten"])
        S4 = sum(S2[jj + dy, ii + dx] for dy, dx in QUARTERS)
        mean = S4[:, 0] / CANON ** 2
        s += w_mean * mean + w_std * np.sqrt(np.maximum(S4[:, 1] / CANON ** 2 - mean ** 2, 0.0))
        out.append(s + b)
    return np.concatenate(out)


def save_maps(path, maps, geom):
    arrays = {f"{k}_{t}": M[k] for t, M in enumerate(maps) for k in ("hog", "lbp", "inten")}
    np.savez(path, content=np.asarray(geom["content"]), scale=geom["scale"], off_x=geom["off_x"],
             off_y=geom["off_y"], **arrays)


def load_maps(path):
    """(maps, geometry) saved by save_maps."""
    with np.load(path) as z:
        maps = [{k: z[f"{k}_{t}"] for k in ("hog", "lbp", "inten")} for t in range(len(WINDOW_TYPES))]
        geom = dict(content=tuple(int(v) for v in z["content"]), scale=float(z["scale"]), off_x=float(z["off_x"]),
                    off_y=float(z["off_y"]))
    return maps, geom


# ----------------------------------------------------------------------------- linear scoring, IoU, NMS

def fold_linear(pipe):
    """(weights, bias) with X @ weights + bias == pipe.decision_function(X), for the model_zoo pipeline
    StandardScaler -> BlockWeight -> linear classifier (scaler and block weights folded into one vector)."""
    scaler, block, clf = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[-1][1]
    w = clf.coef_.ravel().astype(np.float64) * np.asarray(block.weights, np.float64) / scaler.scale_
    return w, float(clf.intercept_.ravel()[0] - w @ scaler.mean_)


def iou_matrix(a, b):
    """IoU between every box of a (n, 4) and of b (m, 4), xywh -> (n, m)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
                 - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
                 - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    union = a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes, scores, thr=0.3, max_keep=None):
    """Greedy non-maximum suppression: indices of kept boxes in descending score; a box is suppressed when its IoU with
    an already kept box is above thr. Stops after max_keep boxes (identical to full NMS followed by the top max_keep)."""
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    b = boxes[order]
    alive = np.ones(len(order), bool)
    keep = []
    while max_keep is None or len(keep) < max_keep:
        rem = np.flatnonzero(alive)
        if len(rem) == 0:
            break
        i, rest = rem[0], rem[1:]
        keep.append(order[i])
        alive[i] = False
        if len(rest):
            alive[rest[iou_matrix(b[i], b[rest])[0] > thr]] = False
    return np.asarray(keep, int)
