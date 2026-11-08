"""Phase 2 hand-off: load the saved features in the agreed form, for Phases 3-7.

    from feature_utils import load_features, make_preprocessor
    d = load_features()                    # 256 px / CLAHE, all five families (the Phase 2 choice)
    train = d["split"] == "train"
    X, y, fold = d["X"][train], d["y"][train], d["fold"][train]
    model = make_pipeline(make_preprocessor(d["col_weights"]), LinearSVC(C=0.1, class_weight="balanced"))

IMPORTANT: X is the RAW concatenated feature matrix (nothing fitted on it). Standardising must happen
INSIDE each CV fold, which is what the pipeline from `make_preprocessor` does (StandardScaler, then the
equal-family weights). Never standardise the whole matrix before splitting.
The test rows (split == "test") must only be used for the final evaluation of a stage.
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractors import FAMILIES
from paths import REPO_ROOT

DEFAULT_TAG = "s256_m448_clahe_c16"     # 256 px working size, CLAHE, HOG 16 px cells (Phase 2 choice)


class BlockWeight(BaseEstimator, TransformerMixin):
    """Multiply each column by a fixed weight (gives every feature family equal total weight)."""

    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]          # marks the step as fitted (needed inside nested pipelines)
        return self

    def transform(self, X):
        return X * self.weights


def make_preprocessor(col_weights=None):
    """StandardScaler (+ equal-family weights). Put it in the pipeline in front of the model."""
    steps = [StandardScaler()]
    if col_weights is not None:
        steps.append(BlockWeight(col_weights))
    return make_pipeline(*steps)


def load_features(tag=DEFAULT_TAG, families=FAMILIES, features_dir=None, master=None, splits=None):
    """Returns a dict:
        X            float32 (n_images, n_features)  raw concatenated features, rows in master order
        col_weights  float32 (n_features,)           1/sqrt(dims of the column's family) -> equal family weight
        slices       {family: slice}                 which columns belong to which family
        y            fractured (0/1)
        region       region_group (hand / leg / hip / shoulder / other)
        split        "train" / "test"   (frozen, results/splits.csv)
        fold         0-4 for training rows, -1 for test rows
        image_id     image ids (same order as X)
        meta         the master table rows (boxes, width, height, hardware, ...)
    """
    fdir = (features_dir or REPO_ROOT / "features") / tag
    ids = pd.read_csv(fdir / "ids.csv")["image_id"]
    blocks = {f: np.load(fdir / f"{f}.npy") for f in families}
    X = np.hstack(list(blocks.values())).astype(np.float32)
    slices, start = {}, 0
    for f, b in blocks.items():
        slices[f] = slice(start, start + b.shape[1])
        start += b.shape[1]
    w = np.concatenate([np.full(b.shape[1], b.shape[1] ** -0.5, np.float32) for b in blocks.values()])
    meta = pd.read_csv(master or REPO_ROOT / "data" / "master.csv").set_index("image_id").loc[ids].reset_index()
    sp = pd.read_csv(splits or REPO_ROOT / "results" / "splits.csv").set_index("image_id").loc[ids].reset_index()
    assert (meta["image_id"] == sp["image_id"]).all() and len(ids) == len(X)
    return dict(X=X, col_weights=w, slices=slices, y=meta["fractured"].to_numpy(),
                region=meta["region_group"].to_numpy(), split=sp["split"].to_numpy(),
                fold=sp["fold"].to_numpy(), image_id=ids.to_numpy(), meta=meta,
                config=json.load(open(fdir / "config.json")))
