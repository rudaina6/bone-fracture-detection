"""The CV protocol shared by Phase 3 (unified) and Phase 4 (region-specific): same code, same folds, same rules.

Only TRAINING images are ever used here. The frozen folds (results/splits.csv, folds 0-4) are the outer folds.

nested_cv(...)   honest estimate. For each outer fold k:
                   1. every grid configuration is scored by an inner CV over the other four folds
                      (fit on three, validate on one; the config with the best mean inner PR-AUC wins),
                   2. the F2-optimal threshold is chosen from the inner out-of-fold scores of the winning config,
                   3. the winner is refitted on the four folds and scored on fold k (never seen in 1-2).
                 The outer scores (default threshold, and the threshold chosen in 2) are therefore free of the
                 optimism that comes from tuning C, the model and the threshold on the same folds.
tune_cv(...)     plain grid search over the 5 frozen folds (used once, to pick the final model's settings
                 before the test evaluation). Its scores ARE slightly optimistic.
"""
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate import best_threshold, threshold_metrics
from feature_utils import load_features
from model_zoo import build, default_threshold, expand_grid, scores


def load_train(tag=None):
    """Training rows only (test rows are dropped immediately and never returned)."""
    d = load_features(tag) if tag else load_features()
    tr = d["split"] == "train"
    assert tr.sum() > 0 and (d["fold"][tr] >= 0).all() and (d["fold"][~tr] == -1).all()
    out = dict(X=d["X"][tr], y=d["y"][tr].astype(int), fold=d["fold"][tr], region=d["region"][tr],
               image_id=d["image_id"][tr], meta=d["meta"][tr].reset_index(drop=True),
               col_weights=d["col_weights"], config=d["config"])
    del d
    return out


def _fit_score(model, imbalance, params, col_weights, seed, X, y, tr, va, kbest="spec"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build(model, imbalance, params, col_weights, seed, kbest).fit(X[tr], y[tr])
        return scores(pipe, X[va])


def _ap(y, s):
    return average_precision_score(y, s)


def nested_cv(X, y, fold, model, imbalance, col_weights, grid=None, seed=42, n_jobs=4, beta=2.0, kbest="spec",
              verbose=0):
    """Nested CV on the given rows (see module doc). Returns a dict:
        oof          outer out-of-fold score for every row
        thr_default  scalar default threshold of the model; thr_tuned  per-row threshold (its fold's tuned value)
        folds        list of per-outer-fold dicts: params, inner_pr_auc, thr_tuned, outer metrics
        seconds      wall time"""
    t0 = time.time()
    ks = sorted(set(fold))
    configs = expand_grid(model, grid)
    n_cfg = len(configs)
    jobs = [(k, ci, j) for k in ks for j in ks if j != k for ci in range(n_cfg)]
    print(f"    {n_cfg} configs x {len(ks)} outer x {len(ks) - 1} inner = {len(jobs)} inner fits, "
          f"then {len(ks)} outer refits", flush=True)
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_score)(model, imbalance, configs[ci], col_weights, seed, X, y,
                            (fold != k) & (fold != j), fold == j, kbest) for k, ci, j in jobs)
    inner = {(k, ci, j): s for (k, ci, j), s in zip(jobs, res)}

    best = {}
    for k in ks:
        js = [j for j in ks if j != k]
        mean_ap = [np.mean([_ap(y[fold == j], inner[(k, ci, j)]) for j in js]) for ci in range(n_cfg)]
        ci = int(np.argmax(mean_ap))                       # ties -> the first (simplest) configuration
        yy = np.concatenate([y[fold == j] for j in js])
        ss = np.concatenate([inner[(k, ci, j)] for j in js])
        thr, f_in = best_threshold(yy, ss, beta)
        best[k] = dict(ci=ci, params=configs[ci], inner_pr_auc=float(mean_ap[ci]), thr_tuned=thr, inner_f2=f_in,
                       inner_all=[float(m) for m in mean_ap])

    outer = Parallel(n_jobs=min(n_jobs, len(ks)), verbose=verbose)(
        delayed(_fit_score)(model, imbalance, best[k]["params"], col_weights, seed, X, y, fold != k, fold == k, kbest)
        for k in ks)
    oof = np.zeros(len(y))
    thr_tuned = np.zeros(len(y))
    thr_def = default_threshold(model)
    folds = []
    for k, s in zip(ks, outer):
        va = fold == k
        oof[va] = s
        thr_tuned[va] = best[k]["thr_tuned"]
        m_def = threshold_metrics(y[va], s, thr_def)
        m_tun = threshold_metrics(y[va], s, best[k]["thr_tuned"])
        folds.append(dict(fold=int(k), params=best[k]["params"], inner_pr_auc=best[k]["inner_pr_auc"],
                          thr_tuned=best[k]["thr_tuned"], pr_auc=float(_ap(y[va], s)),
                          roc_auc=float(roc_auc_score(y[va], s)),
                          **{f"def_{m}": m_def[m] for m in ("recall", "precision", "specificity", "f2")},
                          **{f"tuned_{m}": m_tun[m] for m in ("recall", "precision", "specificity", "f2")}))
    return dict(oof=oof, thr_default=thr_def, thr_tuned=thr_tuned, folds=folds, configs=configs,
                seconds=time.time() - t0)


def tune_cv(X, y, fold, model, imbalance, col_weights, grid=None, seed=42, n_jobs=4, beta=2.0, kbest="spec",
            verbose=0):
    """Grid search over the frozen folds (non-nested). Returns best params, its mean CV PR-AUC, and the
    out-of-fold scores of the best config (used to choose the F2 threshold for the final model)."""
    ks = sorted(set(fold))
    configs = expand_grid(model, grid)
    jobs = [(ci, k) for ci in range(len(configs)) for k in ks]
    print(f"    {len(configs)} configs x {len(ks)} folds = {len(jobs)} fits", flush=True)
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_score)(model, imbalance, configs[ci], col_weights, seed, X, y, fold != k, fold == k, kbest)
        for ci, k in jobs)
    sc = {(ci, k): s for (ci, k), s in zip(jobs, res)}
    mean_ap = [np.mean([_ap(y[fold == k], sc[(ci, k)]) for k in ks]) for ci in range(len(configs))]
    ci = int(np.argmax(mean_ap))
    oof = np.zeros(len(y))
    for k in ks:
        oof[fold == k] = sc[(ci, k)]
    thr, f2 = best_threshold(y, oof, beta)
    return dict(params=configs[ci], cv_pr_auc=float(mean_ap[ci]), oof=oof, thr_tuned=thr, f2_at_thr=f2,
                configs=configs, all_pr_auc=[float(m) for m in mean_ap])


def edge_warnings(folds, grid):
    """Names of numeric hyperparameters whose chosen value was the smallest/largest grid value in >= 3 of 5 folds."""
    warn = []
    for key, vals in grid.items():
        num = [v for v in vals if isinstance(v, (int, float)) and v is not None]
        if len(num) < 3 or len(num) != len(vals):
            continue
        lo, hi = min(num), max(num)
        chosen = [f["params"][key] for f in folds]
        if sum(c == lo for c in chosen) >= 3:
            warn.append(f"{key} at LOWER edge ({lo}) in {sum(c == lo for c in chosen)}/{len(chosen)} folds")
        if sum(c == hi for c in chosen) >= 3:
            warn.append(f"{key} at UPPER edge ({hi}) in {sum(c == hi for c in chosen)}/{len(chosen)} folds")
    return warn
