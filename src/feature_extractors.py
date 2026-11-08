"""Phase 2.2: handcrafted feature families. Every extractor takes a preprocessed float32 image in
[0, 1] (square) and returns a 1-D float32 vector. Starting values follow Fracture_Project_Phases.md.

    family     dims at 256 px   captures
    hog        8100  (c=16)     edges / shape
    lbp         448             local texture (uniform LBP, two radii, 4x4 grid of cells)
    glcm         48             texture statistics
    gabor        24             frequency / orientation
    intensity    21             global intensity and edge density
"""
from functools import lru_cache

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor, gabor_kernel

FAMILIES = ("hog", "lbp", "glcm", "gabor", "intensity")


def _u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def hog_features(img, cell=16, orientations=9, block=2):
    return hog(img, orientations=orientations, pixels_per_cell=(cell, cell),
               cells_per_block=(block, block), block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


def lbp_features(img, settings=((8, 1), (16, 2)), grid=4):
    g = _u8(img)
    h, w = g.shape
    feats = []
    for P, R in settings:
        codes = local_binary_pattern(g, P, R, method="uniform")      # values 0..P+1
        n_bins = P + 2
        for i in range(grid):
            for j in range(grid):
                cell = codes[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
                hist = np.bincount(cell.astype(np.int64).ravel(), minlength=n_bins).astype(np.float32)
                feats.append(hist / hist.sum())
    return np.concatenate(feats)


def glcm_features(img, levels=32, distances=(1, 3), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    q = np.minimum((img * levels).astype(np.uint8), levels - 1)
    m = graycomatrix(q, distances=list(distances), angles=list(angles), levels=levels,
                     symmetric=True, normed=True)
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    return np.concatenate([graycoprops(m, p).ravel() for p in props]).astype(np.float32)


@lru_cache(maxsize=None)
def _gabor_kernels(frequencies, n_theta):
    return [(gabor_kernel(f, theta=k * np.pi / n_theta).real.astype(np.float32),
             gabor_kernel(f, theta=k * np.pi / n_theta).imag.astype(np.float32))
            for f in frequencies for k in range(n_theta)]


def gabor_features(img, frequencies=(0.1, 0.2, 0.4), n_theta=4):
    """Mean and variance of the Gabor response magnitude for each frequency x orientation.
    Same numbers as skimage.filters.gabor (up to float precision) but computed with cv2.filter2D,
    which is ~50x faster than skimage's convolution."""
    x = img.astype(np.float32)
    feats = []
    for kr, ki in _gabor_kernels(tuple(frequencies), n_theta):
        re = cv2.filter2D(x, -1, kr, borderType=cv2.BORDER_REFLECT)
        im = cv2.filter2D(x, -1, ki, borderType=cv2.BORDER_REFLECT)
        mag = np.hypot(re, im)
        feats += [mag.mean(), mag.var()]
    return np.asarray(feats, np.float32)


def intensity_features(img, bins=16):
    v = img.ravel()
    hist = np.histogram(v, bins=bins, range=(0, 1))[0].astype(np.float32)
    edges = cv2.Canny(_u8(img), 50, 150)
    stats = [v.mean(), v.std(), skew(v), kurtosis(v)]
    return np.concatenate([stats, hist / hist.sum(), [(edges > 0).mean()]]).astype(np.float32)


def extract(img, families=FAMILIES, hog_cell=16):
    """Dict family -> feature vector for one preprocessed image."""
    fns = dict(hog=lambda x: hog_features(x, cell=hog_cell), lbp=lbp_features, glcm=glcm_features,
               gabor=gabor_features, intensity=intensity_features)
    return {f: fns[f](img) for f in families}
