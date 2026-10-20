"""Phase 2.2: compute the five handcrafted feature families from the cached images.

Run from the repo root (after preprocess.py):
    python src/extract_features.py --size 256 --norm clahe
    python src/extract_features.py --size 256 --norm zscore     # the EDA alternative
    python src/extract_features.py --size 256 --families hog    # one family only

Each family is saved to its OWN file. That is deliberate: the Phase 2.3 comparison
needs to try families alone and in combinations, and re-running one slow family
(Gabor) must not mean recomputing the others.

Contrast normalisation (docs/data_decisions.md section 9) is applied here rather
than in preprocess.py, because CLAHE vs per-image z-score is a variable being
compared, and both read the same cached geometry.

Output (one per family):
    features/<family>_<size>_<norm>.npy    float32, shape (4083, d)
    features/featureinfo_<size>_<norm>.json   dimensions and settings, for the report

Row order matches features/images_<size>_ids.csv. Never reorder.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

# --- settings from the project plan, section 2.2 -------------------------------
HOG_ORIENTATIONS = 9
HOG_CELL = (16, 16)
HOG_BLOCK = (2, 2)

LBP_SETTINGS = [(8, 1), (16, 2)]     # (P neighbours, R radius)
LBP_GRID = 4                          # 4x4 grid of image cells

GLCM_LEVELS = 32
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

GABOR_FREQUENCIES = [0.1, 0.2, 0.4]
GABOR_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GABOR_SIGMA_PREFACTOR = 0.5622       # bandwidth 1, same convention as skimage

INTENSITY_BINS = 16
CANNY_LOW, CANNY_HIGH = 50, 150


# --- contrast normalisation ----------------------------------------------------
def normalise(img, how):
    """img is uint8. Returns uint8, so every family sees the same input."""
    if how == "none":
        return img
    if how == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    if how == "zscore":
        f = img.astype(np.float32)
        s = f.std()
        if s < 1e-6:
            return img
        z = np.clip((f - f.mean()) / s, -3, 3)
        return (((z + 3) / 6) * 255).astype(np.uint8)
    raise ValueError(how)


# --- one feature function per family. Each takes uint8, returns a 1-D float array.
def f_hog(img):
    return hog(img.astype(np.float32) / 255.0,
               orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_CELL,
               cells_per_block=HOG_BLOCK, block_norm="L2-Hys", feature_vector=True)


def f_lbp(img):
    out, side = [], img.shape[0] // LBP_GRID
    for P, R in LBP_SETTINGS:
        codes = local_binary_pattern(img, P, R, method="uniform")
        nbins = P + 2
        for r in range(LBP_GRID):
            for c in range(LBP_GRID):
                tile = codes[r * side:(r + 1) * side, c * side:(c + 1) * side]
                h, _ = np.histogram(tile, bins=nbins, range=(0, nbins))
                out.append(h / max(h.sum(), 1))
    return np.concatenate(out)


def f_glcm(img):
    q = (img.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    m = graycomatrix(q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    return np.concatenate([graycoprops(m, p).ravel() for p in GLCM_PROPS])


def _gabor_kernels():
    """Built once. OpenCV's filter2D is ~11x faster than skimage.filters.gabor and
    ranks images identically per filter (verified by Spearman = 1.0), which is all
    that matters once StandardScaler removes the absolute scale."""
    ks = []
    for freq in GABOR_FREQUENCIES:
        sigma = GABOR_SIGMA_PREFACTOR / freq
        n = int(2 * np.ceil(3 * sigma) + 1)
        for th in GABOR_ANGLES:
            ks.append((
                cv2.getGaborKernel((n, n), sigma, th, 1.0 / freq, 1.0, 0, ktype=cv2.CV_32F),
                cv2.getGaborKernel((n, n), sigma, th, 1.0 / freq, 1.0, -np.pi / 2, ktype=cv2.CV_32F),
            ))
    return ks


GABOR_KERNELS = _gabor_kernels()


def f_gabor(img):
    f = img.astype(np.float32) / 255.0
    out = []
    for k_re, k_im in GABOR_KERNELS:
        mag = cv2.magnitude(cv2.filter2D(f, cv2.CV_32F, k_re),
                            cv2.filter2D(f, cv2.CV_32F, k_im))
        out += [mag.mean(), mag.var()]
    return np.array(out)


def f_intensity(img):
    f = img.astype(np.float32).ravel()
    h, _ = np.histogram(f, bins=INTENSITY_BINS, range=(0, 256))
    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)
    return np.concatenate([
        [f.mean(), f.std(), skew(f), kurtosis(f)],
        h / f.size,
        [float((edges > 0).mean())],
    ])


FAMILIES = {"hog": f_hog, "lbp": f_lbp, "glcm": f_glcm,
            "gabor": f_gabor, "intensity": f_intensity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--norm", choices=["clahe", "zscore", "none"], default="clahe")
    ap.add_argument("--families", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    ap.add_argument("--dir", default=Path("features"), type=Path)
    args = ap.parse_args()

    src = args.dir / f"images_{args.size}.npy"
    if not src.exists():
        raise SystemExit(f"{src} not found. Run: python src/preprocess.py --size {args.size}")
    images = np.load(src)
    n = len(images)
    print(f"{n} images, {args.size}x{args.size}, norm={args.norm}")
    print(f"families: {', '.join(args.families)}\n")

    # normalise once, reuse for every family
    t0 = time.time()
    norm = np.empty_like(images)
    for i in range(n):
        norm[i] = normalise(images[i], args.norm)
    print(f"normalised in {time.time() - t0:.0f}s\n")
    del images

    info_path = args.dir / f"featureinfo_{args.size}_{args.norm}.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    for name in args.families:
        fn, t0 = FAMILIES[name], time.time()
        first = fn(norm[0])
        X = np.empty((n, first.size), dtype=np.float32)
        X[0] = first
        for i in range(1, n):
            X[i] = fn(norm[i])
            if (i + 1) % 500 == 0 or i + 1 == n:
                el = time.time() - t0
                print(f"  {name}: {i + 1}/{n}  ({el:.0f}s elapsed, "
                      f"~{el / (i + 1) * (n - i - 1):.0f}s left)", flush=True)

        out = args.dir / f"{name}_{args.size}_{args.norm}.npy"
        np.save(out, X)
        bad = int(np.isnan(X).sum() + np.isinf(X).sum())
        const = int((X.std(axis=0) == 0).sum())
        info[name] = {"dim": int(X.shape[1]), "seconds": round(time.time() - t0),
                      "nan_or_inf": bad, "constant_columns": const}
        print(f"  -> {out}  dim={X.shape[1]}  {X.nbytes / 1e6:.0f} MB  "
              f"nan/inf={bad}  constant cols={const}\n")

    info_path.write_text(json.dumps(info, indent=2))
    total = sum(v["dim"] for v in info.values())
    print(f"Wrote {info_path}")
    print(f"Total dimension across all saved families: {total}")


if __name__ == "__main__":
    main()