"""Shared evaluation code for every phase (Phase 3 onwards). One metric definition, used everywhere.

    from evaluate import (threshold_metrics, ranking_metrics, all_metrics, best_threshold,
                          metrics_by_group, bootstrap_ci, paired_bootstrap, never_fracture_metrics)

Conventions
- `score` is any real-valued fracture score where larger = more likely fractured (SVM margin, logistic
  decision value or a probability). A prediction is "fracture" when score >= thr.
- Threshold metrics: recall (= sensitivity), precision, specificity, F1, F2, accuracy and the confusion
  counts tn/fp/fn/tp. precision is 0 when nothing is predicted positive.
- Ranking metrics (threshold-free): roc_auc and pr_auc. pr_auc is sklearn's average_precision_score, the
  same quantity as in Phase 2. Chance level of pr_auc is the fracture rate, not 0.5.
- best_threshold picks the threshold that maximises F-beta (default beta=2, recall counts 4x precision)
  on the scores you give it. Pass it training/CV scores only, never test scores.
- Bootstrap CIs resample IMAGES with replacement (percentile interval). The threshold is held fixed, so the
  interval reflects test-set sampling noise only, not the noise of having chosen the threshold.
  Groups with fewer than `min_pos` fractured images get no interval (NaN): with ~2 positives it is meaningless.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

REGIONS = ["hand", "leg", "hip", "shoulder"]          # "other" (9 fractured images) is only in the overall numbers
THRESHOLD_METRICS = ["recall", "precision", "specificity", "f1", "f2", "accuracy"]
RANKING_METRICS = ["roc_auc", "pr_auc"]
METRICS = ["recall", "precision", "f1", "f2", "specificity", "roc_auc", "pr_auc"]


def _fbeta(p, r, beta):
    b2 = beta * beta
    d = b2 * p + r
    return (1 + b2) * p * r / d if d > 0 else 0.0


def threshold_metrics(y, score, thr):
    """Confusion matrix and threshold-dependent metrics at `score >= thr`."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(score) >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = _fbeta(precision, recall, 1.0) if tp + fn else np.nan
    f2 = _fbeta(precision, recall, 2.0) if tp + fn else np.nan
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, recall=recall, precision=precision, specificity=spec,
                f1=f1, f2=f2, accuracy=(tp + tn) / len(y))


def ranking_metrics(y, score):
    """ROC-AUC and PR-AUC (NaN if only one class is present)."""
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return dict(roc_auc=np.nan, pr_auc=np.nan)
    return dict(roc_auc=roc_auc_score(y, score), pr_auc=average_precision_score(y, score))


def all_metrics(y, score, thr):
    return {**threshold_metrics(y, score, thr), **ranking_metrics(y, score)}


def best_threshold(y, score, beta=2.0):
    """Threshold maximising F-beta. Returns (threshold, fbeta). Predict fracture when score >= threshold."""
    y = np.asarray(y).astype(int)
    p, r, t = precision_recall_curve(y, score)          # p, r have one more entry than t
    p, r = p[:-1], r[:-1]
    b2 = beta * beta
    denom = b2 * p + r
    f = np.divide((1 + b2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f))
    return float(t[i]), float(f[i])


def never_fracture_metrics(y):
    """The trivial baseline "always predict no fracture": recall 0, accuracy = 1 - fracture rate,
    ROC-AUC 0.5, PR-AUC = fracture rate. Shows why accuracy alone is misleading."""
    y = np.asarray(y).astype(int)
    return all_metrics(y, np.zeros(len(y)), 1.0)


def thr_vector(thr, groups, n):
    """Per-image threshold array from a number, a dict {region: threshold} or an array of length n."""
    if isinstance(thr, dict):
        groups = np.asarray(groups)
        default = thr.get("overall", 0.0)
        return np.array([thr.get(g, default) for g in groups], dtype=float)
    thr = np.asarray(thr, dtype=float)
    return np.full(n, float(thr)) if thr.ndim == 0 else thr


def metrics_by_group(y, score, groups, thr, group_names=REGIONS, include_overall=True):
    """One row per group (plus "overall"): n, n_pos, prevalence, and all metrics at threshold `thr`.
    `thr` is a number, a dict {region: threshold} (per-region thresholds, Phase 6), or an array with one
    threshold per image (e.g. the fold-specific thresholds of a cross-validated run)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups)
    t_all = thr_vector(thr, groups, len(y))
    rows = []
    for g in (["overall"] if include_overall else []) + list(group_names):
        idx = np.ones(len(y), bool) if g == "overall" else groups == g
        if not idx.any():
            continue
        rows.append(dict(group=g, n=int(idx.sum()), n_pos=int(y[idx].sum()), prevalence=y[idx].mean(),
                         **all_metrics(y[idx], score[idx], t_all[idx])))
    return pd.DataFrame(rows)


def _metric_arrays(y, score, thr, names):
    m = all_metrics(y, score, thr)
    return np.array([m[k] for k in names], dtype=float)


def bootstrap_ci(y, score, thr, groups=None, group_names=REGIONS, n_boot=2000, seed=0, level=0.95,
                 min_pos=10, metrics=METRICS):
    """Percentile bootstrap CI for every metric, overall and per group.
    Returns a long DataFrame: group, metric, estimate, lo, hi, n, n_pos, n_valid (resamples with a defined value)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups) if groups is not None else np.array(["overall"] * len(y))
    rng = np.random.default_rng(seed)
    a = (1 - level) / 2
    out = []
    t_all = thr_vector(thr, groups, len(y))
    for g in ["overall"] + [g for g in group_names if g in set(groups)]:
        idx = np.flatnonzero(np.ones(len(y), bool) if g == "overall" else groups == g)
        yy, ss, tt = y[idx], score[idx], t_all[idx]
        est = _metric_arrays(yy, ss, tt, metrics)
        ci = np.full((2, len(metrics)), np.nan)
        n_valid = np.zeros(len(metrics), int)
        if yy.sum() >= min_pos:
            draws = np.full((n_boot, len(metrics)), np.nan)
            for b in range(n_boot):
                j = rng.integers(0, len(idx), len(idx))
                if yy[j].sum() == 0 or yy[j].sum() == len(j):
                    continue
                draws[b] = _metric_arrays(yy[j], ss[j], tt[j], metrics)
            n_valid = np.sum(~np.isnan(draws), axis=0)
            with np.errstate(all="ignore"):
                ci = np.nanpercentile(draws, [100 * a, 100 * (1 - a)], axis=0)
        for k, m in enumerate(metrics):
            out.append(dict(group=g, metric=m, estimate=est[k], lo=ci[0, k], hi=ci[1, k], n=len(idx),
                            n_pos=int(yy.sum()), n_valid=int(n_valid[k])))
    return pd.DataFrame(out)


def paired_bootstrap(y, score_a, score_b, thr_a, thr_b, groups=None, group_names=REGIONS, n_boot=2000,
                     seed=0, level=0.95, min_pos=10, metrics=("recall", "precision", "f1", "f2", "pr_auc")):
    """Paired bootstrap of metric(A) - metric(B) on the SAME images (for unified vs region-specific, Phase 5).
    Returns group, metric, diff, lo, hi, p_two_sided (share of resamples on the other side of 0, doubled)."""
    y = np.asarray(y).astype(int)
    sa, sb = np.asarray(score_a, float), np.asarray(score_b, float)
    groups = np.asarray(groups) if groups is not None else np.array(["overall"] * len(y))
    rng = np.random.default_rng(seed)
    al = (1 - level) / 2
    out = []
    ta, tb_ = thr_vector(thr_a, groups, len(y)), thr_vector(thr_b, groups, len(y))
    for g in ["overall"] + [g for g in group_names if g in set(groups)]:
        idx = np.flatnonzero(np.ones(len(y), bool) if g == "overall" else groups == g)
        yy = y[idx]
        d0 = _metric_arrays(yy, sa[idx], ta[idx], metrics) - _metric_arrays(yy, sb[idx], tb_[idx], metrics)
        lo = hi = p = np.full(len(metrics), np.nan)
        if yy.sum() >= min_pos:
            draws = np.full((n_boot, len(metrics)), np.nan)
            for b in range(n_boot):
                j = rng.integers(0, len(idx), len(idx))
                if yy[j].sum() == 0 or yy[j].sum() == len(j):
                    continue
                draws[b] = (_metric_arrays(yy[j], sa[idx][j], ta[idx][j], metrics)
                            - _metric_arrays(yy[j], sb[idx][j], tb_[idx][j], metrics))
            with np.errstate(all="ignore"):
                lo, hi = np.nanpercentile(draws, [100 * al, 100 * (1 - al)], axis=0)
                p = np.minimum(1.0, 2 * np.minimum(np.nanmean(draws <= 0, axis=0), np.nanmean(draws >= 0, axis=0)))
        for k, m in enumerate(metrics):
            out.append(dict(group=g, metric=m, diff=d0[k], lo=lo[k], hi=hi[k], p_two_sided=p[k]))
    return pd.DataFrame(out)


def format_ci_table(ci, metrics=METRICS, digits=3):
    """Wide table of "estimate [lo, hi]" strings, one row per group (for reports and logs)."""
    def cell(r):
        if np.isnan(r.estimate):
            return "n/a"
        s = f"{r.estimate:.{digits}f}"
        return s if np.isnan(r.lo) else f"{s} [{r.lo:.{digits}f}, {r.hi:.{digits}f}]"
    ci = ci.assign(text=[cell(r) for r in ci.itertuples()])
    wide = ci.pivot(index="group", columns="metric", values="text")
    meta = ci.drop_duplicates("group").set_index("group")[["n", "n_pos"]]
    order = [g for g in ["overall"] + REGIONS if g in wide.index]
    return meta.join(wide[[m for m in metrics if m in wide.columns]]).loc[order]


if __name__ == "__main__":      # python src/evaluate.py  -> quick self-test on random scores
    rng = np.random.default_rng(0)
    yy = (rng.random(800) < 0.18).astype(int)
    ss = rng.normal(size=800) + 1.2 * yy
    reg = rng.choice(REGIONS, 800)
    t, f = best_threshold(yy, ss)
    print(f"best F2 threshold {t:.3f} (F2={f:.3f})")
    print(metrics_by_group(yy, ss, reg, t).round(3).to_string(index=False))
    print(format_ci_table(bootstrap_ci(yy, ss, t, reg, n_boot=200)).to_string())
    print("never fracture:", {k: round(v, 3) for k, v in never_fracture_metrics(yy).items()})
