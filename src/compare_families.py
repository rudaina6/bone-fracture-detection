"""Phase 2.3: feature-family comparison with ONE fixed model (linear SVM), CV only.

    python src/compare_families.py                         # tag s256_m448_clahe_c16, no reduction
    python src/compare_families.py --tag s448_m448_clahe_c28
    python src/compare_families.py --reduction pca         # none | pca | l1 | tree
    python src/compare_families.py --C 0.0001 0.0003 0.001  # several C values in one go
    python src/compare_families.py --block-weight --C 0.01 0.1 1   # equal weight per family
    python src/compare_families.py --exclude-hires         # drop the 351 high-resolution images
    python src/compare_families.py --exclude-hardware      # drop the 99 hardware images

Uses ONLY the training images and the frozen folds in results/splits.csv (the test set is never
loaded). The scaler (and PCA / selector) is a pipeline step, so it is fitted on the training part of
each fold only. Class imbalance is handled with class_weight="balanced" for every candidate, so
differences come from the features.

Candidates: each family alone; then forward selection (start from the best single family, add the
family that most improves mean CV PR-AUC, stop when nothing helps); plus "all" and "all minus
intensity" (intensity statistics identify the region and may act as a shortcut, see
docs/data_decisions.md section 9).

Output: results/tables/feature_family_<tag>_<reduction>_C<C>[_flags].csv and one row per candidate in
results/experiment_log.csv.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import average_precision_score, fbeta_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from expt_log import log_run
from feature_utils import BlockWeight
from feature_extractors import FAMILIES
from paths import REPO_ROOT

REGIONS = ["hand", "leg", "hip", "shoulder"]


def make_model(reduction, C, seed, weights=None):
    steps = [StandardScaler()]
    if weights is not None:
        steps.append(BlockWeight(weights))
    if reduction == "pca":
        steps.append(PCA(n_components=0.95, svd_solver="full", random_state=seed))
    elif reduction == "l1":
        steps.append(SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.01, class_weight="balanced",
                                               max_iter=5000, random_state=seed)))
    elif reduction == "tree":
        steps.append(SelectFromModel(ExtraTreesClassifier(n_estimators=200, class_weight="balanced",
                                                          n_jobs=1, random_state=seed), threshold="median"))
    steps.append(LinearSVC(C=C, class_weight="balanced", dual="auto", max_iter=20000, random_state=seed))
    return make_pipeline(*steps)


def fit_fold(X, y, tr, va, reduction, C, seed, weights=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_model(reduction, C, seed, weights).fit(X[tr], y[tr])
        return va, model.decision_function(X[va])


def evaluate(X, y, folds, region, reduction, C, seed, n_jobs, weights=None):
    """Per-fold metrics (mean/std over folds) and pooled out-of-fold PR-AUC per region."""
    ks = sorted(set(folds))
    res = Parallel(n_jobs=n_jobs)(
        delayed(fit_fold)(X, y, np.flatnonzero(folds != k), np.flatnonzero(folds == k), reduction, C, seed, weights)
        for k in ks)
    oof = np.zeros(len(y))
    per = []
    for va, s in res:
        oof[va] = s
        pred = (s > 0).astype(int)
        per.append(dict(roc_auc=roc_auc_score(y[va], s), pr_auc=average_precision_score(y[va], s),
                        recall=recall_score(y[va], pred), f2=fbeta_score(y[va], pred, beta=2)))
    per = pd.DataFrame(per)
    out = {f"{m}": per[m].mean() for m in per} | {f"{m}_sd": per[m].std() for m in per}
    for g in REGIONS:
        idx = region == g
        out[f"pr_auc_{g}"] = average_precision_score(y[idx], oof[idx]) if y[idx].sum() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s256_m448_clahe_c16")
    ap.add_argument("--master", default=REPO_ROOT / "data" / "master.csv", type=Path)
    ap.add_argument("--splits", default=REPO_ROOT / "results" / "splits.csv", type=Path)
    ap.add_argument("--features-dir", default=REPO_ROOT / "features", type=Path)
    ap.add_argument("--reduction", choices=["none", "pca", "l1", "tree"], default="none")
    ap.add_argument("--C", type=float, nargs="+", default=[0.01],
                    help="linear SVM C, one or several values (features are standardised)")
    ap.add_argument("--block-weight", action="store_true",
                    help="after standardising, scale each family by 1/sqrt(dims) so every family has equal weight")
    ap.add_argument("--exclude-hires", action="store_true")
    ap.add_argument("--exclude-hardware", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=5, help="folds run in parallel")
    ap.add_argument("--no-log", action="store_true", help="do not append to the experiment log")
    args = ap.parse_args()

    fdir = args.features_dir / args.tag
    ids = pd.read_csv(fdir / "ids.csv")["image_id"]
    feats = {f: np.load(fdir / f"{f}.npy") for f in FAMILIES if (fdir / f"{f}.npy").exists()}
    master = pd.read_csv(args.master).set_index("image_id").loc[ids].reset_index()
    splits = pd.read_csv(args.splits).set_index("image_id").loc[ids].reset_index()
    keep = (splits["split"] == "train").to_numpy()                    # TEST IMAGES ARE NEVER USED HERE
    if args.exclude_hires:
        keep = keep & ((master.width * master.height) <= 1e6).to_numpy()
    if args.exclude_hardware:
        keep = keep & (master.hardware == 0).to_numpy()
    assert (splits.loc[keep, "split"] == "train").all() and (splits.loc[keep, "fold"] >= 0).all()
    y = master.loc[keep, "fractured"].to_numpy()
    folds = splits.loc[keep, "fold"].to_numpy()
    region = master.loc[keep, "region_group"].to_numpy()
    feats = {f: a[keep] for f, a in feats.items()}
    flags = ("_nohires" if args.exclude_hires else "") + ("_nohw" if args.exclude_hardware else "")
    print(f"{keep.sum()} training images, {y.sum()} fractured; families: {list(feats)}; "
          f"reduction={args.reduction}; C={args.C}; block_weight={args.block_weight}")
    flags += "_bw" if args.block_weight else ""
    for C in args.C:
        analyse(args, C, feats, y, folds, region, flags)


def analyse(args, C, feats, y, folds, region, flags):
    print(f"\n=== C={C:g} ===")

    cache = {}

    def run(fams):
        key = "+".join(fams)
        if key not in cache:
            t0 = time.time()
            X = np.hstack([feats[f] for f in fams])
            w = (np.concatenate([np.full(feats[f].shape[1], feats[f].shape[1] ** -0.5) for f in fams])
                 if args.block_weight else None)
            cache[key] = dict(features=key, n_dims=X.shape[1],
                              **evaluate(X, y, folds, region, args.reduction, C, args.seed, args.n_jobs, w))
            r = cache[key]
            print(f"  {key:<34} dims={r['n_dims']:>5}  PR-AUC={r['pr_auc']:.3f}+-{r['pr_auc_sd']:.3f}  "
                  f"ROC-AUC={r['roc_auc']:.3f}  F2={r['f2']:.3f}  ({time.time() - t0:.0f}s)")
        return cache[key]

    names = list(feats)
    print("Single families:")
    singles = {f: run([f]) for f in names}
    print("Forward selection:")
    chosen = [max(names, key=lambda f: singles[f]["pr_auc"])]
    best = singles[chosen[0]]["pr_auc"]
    while len(chosen) < len(names):
        cand = {f: run(chosen + [f]) for f in names if f not in chosen}
        f_best = max(cand, key=lambda f: cand[f]["pr_auc"])
        if cand[f_best]["pr_auc"] <= best:
            break
        chosen.append(f_best)
        best = cand[f_best]["pr_auc"]
    print("Reference sets:")
    run(names)
    if "intensity" in names and len(names) > 1:
        run([f for f in names if f != "intensity"])

    table = pd.DataFrame(cache.values()).sort_values("pr_auc", ascending=False)
    out = REPO_ROOT / "results" / "tables" / f"feature_family_{args.tag}_{args.reduction}_C{C:g}{flags}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)
    print(f"\nBest by mean CV PR-AUC: {table.iloc[0]['features']} ({table.iloc[0]['pr_auc']:.3f})\nWrote {out}")

    if not args.no_log:
        for r in table.itertuples():
            log_run(regions="all (pooled)", feature_set=f"{r.features} [{args.tag}]", model="LinearSVC",
                    imbalance_strategy="class_weight=balanced",
                    key_hyperparameters=f"C={C:g}; reduction={args.reduction}; block_weight={args.block_weight}{flags}",
                    cv_score=f"PR-AUC={r.pr_auc:.3f}+-{r.pr_auc_sd:.3f}; ROC-AUC={r.roc_auc:.3f}; "
                             f"recall={r.recall:.3f}; F2={r.f2:.3f}",
                    test_metrics="not used (CV only)", notes="Phase 2 feature-family comparison")


if __name__ == "__main__":
    main()
