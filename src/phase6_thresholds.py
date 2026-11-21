"""Phase 6, lever 1: per-region decision thresholds on the single unified model. CROSS-VALIDATION ONLY, no test images.

    python src/phase6_thresholds.py                 # gate G1 (fresh unified baseline, ~2.5 min) + lever 1
    python src/phase6_thresholds.py --summarize     # rebuild tables, figure and verdicts from the saved baseline
    options: --n-jobs 4  --n-boot 2000  --boot-seed 0  --force (re-run the baseline)  --no-log

Protocol and decision rules: docs/phase6_gap_closing.md (written before any Phase 6 result).

Gate G1: the Phase 3 model (rbfsvm, imbalance none, full grid, seed 42) is re-run under the same nested CV on this
machine, with nested_cv(return_inner=True). It must reproduce results/phase3/oof_rbfsvm__none.csv within the G1
tolerances (same parameters in every outer fold, scores within 2e-3, thresholds within 1e-3, fold PR-AUC within
5e-4), or the script stops. The run is saved as results/phase6/oof_baseline_rbfsvm.csv and is the "unified" arm of
every Phase 6 comparison (G3).

Lever 1 arms, all on the baseline's outer scores. Every threshold is chosen ONLY on the inner out-of-fold scores of
its own outer fold (the scores Phase 3 chose its global threshold on), never on pooled outer scores: in Phase 3,
fold 0 chose gamma 0.2 and folds 1-4 gamma 0.05, so outer margins are on different scales.
  G          one global F2 threshold per fold (recomputed here and asserted equal to the fold's thr_tuned)
  R          one F2 threshold per region (hand, leg, hip, shoulder); "other" falls back to G
  R-shrunk   w * thr_R + (1 - w) * thr_G with w = npos / (npos + 40), npos = the region's fractured images in the
             inner set (k = 40 is fixed by the protocol and never tuned)
  P4         the routed Phase 4 region models at their own thresholds (results/phase4/oof_rbfsvm__none.csv)
Verdicts L1-L5 are printed and saved. Nothing here changes a model or touches a test image.

Outputs: results/phase6/oof_baseline_rbfsvm.csv, inner_baseline_rbfsvm.csv, run_baseline_rbfsvm.json, gate_G1.json,
         thresholds_per_fold.csv, lever1_verdicts.json,
         results/tables/phase6_thresholds_metrics.csv, phase6_thresholds_paired.csv, phase6_threshold_stability.csv,
         results/figures/phase6_thresholds.png
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cv_protocol import load_train, nested_cv
from evaluate import REGIONS, best_threshold, metrics_by_group, paired_bootstrap, threshold_metrics
from expt_log import log_run
from model_zoo import REGION_CODES, SPECS, grid_for
from paths import REPO_ROOT
from phase5_viz import C, REGION_LABEL, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

MODEL, IMB = "rbfsvm", "none"
BASE = "baseline_rbfsvm"                  # results/phase6/{oof,inner,run}_baseline_rbfsvm.*
SHRINK_K = 40                             # R-shrunk: fixed by the protocol, never tuned
G1_TOL = dict(score=2e-3, thr=1e-3, fold_pr_auc=5e-4)
EXPECTED_TRAIN = (3266, 576)              # the frozen training split (images, fractured); None skips the check
ARMS = ("G", "R", "R-shrunk")
THR_METRICS = ("recall", "precision", "specificity", "f2")
PAIRS = (("R", "G"), ("R-shrunk", "G"), ("R", "P4"), ("R-shrunk", "P4"), ("P4", "G"))


def read_csv(path):
    """CSV with an exact float round trip (pandas' default parser can be one unit off in the last place)."""
    return pd.read_csv(path, float_precision="round_trip")


def fmt_ci(d, lo, hi):
    return "n/a" if np.isnan(d) else f"{d:+.3f}" + ("" if np.isnan(lo) else f" [{lo:+.3f}, {hi:+.3f}]")


# ----------------------------------------------------------------------------- gate G1

def run_baseline(args):
    """Gate G1's run: the Phase 3 nested CV on this machine, keeping each outer fold's inner out-of-fold scores."""
    tr = load_train()
    if EXPECTED_TRAIN:
        assert (len(tr["y"]), int(tr["y"].sum())) == EXPECTED_TRAIN, "unexpected training set; is this the frozen split?"
    grid = grid_for(MODEL)
    print(f"{len(tr['y'])} training images, {int(tr['y'].sum())} fractured, {tr['X'].shape[1]} features")
    print(f"\n=== Gate G1 run: {SPECS[MODEL].label}, imbalance={IMB}, grid={grid}, seed={args.seed} ===", flush=True)
    r = nested_cv(tr["X"], tr["y"], tr["fold"], MODEL, IMB, tr["col_weights"], grid=grid, seed=args.seed,
                  n_jobs=args.n_jobs, return_inner=True)
    oof = pd.DataFrame(dict(image_id=tr["image_id"], fold=tr["fold"], y=tr["y"], region=tr["region"], score=r["oof"],
                            thr_default=r["thr_default"], thr_tuned=r["thr_tuned"]))
    inner = pd.concat([pd.DataFrame(dict(outer_fold=k, image_id=tr["image_id"][idx], fold=tr["fold"][idx],
                                         y=tr["y"][idx], region=tr["region"][idx], inner_score=s))
                       for k, (idx, s) in sorted(r["inner"].items())], ignore_index=True)
    chosen = Counter(json.dumps(x["params"], sort_keys=True) for x in r["folds"]).most_common()
    meta = dict(model=MODEL, imbalance=IMB, grid={k: [str(v) for v in vs] for k, vs in grid.items()},
                kbest=SPECS[MODEL].kbest, seed=args.seed, n_jobs=args.n_jobs, tag=tr["config"],
                n_train=int(len(tr["y"])), n_pos=int(tr["y"].sum()), seconds=r["seconds"], folds=r["folds"],
                modal_params=json.loads(chosen[0][0]), date=pd.Timestamp.now().isoformat(timespec="seconds"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.out_dir / f"oof_{BASE}.csv", index=False)
    inner.to_csv(args.out_dir / f"inner_{BASE}.csv", index=False)
    (args.out_dir / f"run_{BASE}.json").write_text(json.dumps(meta, indent=1, default=str))
    f = pd.DataFrame(r["folds"])
    print(f"    outer-fold PR-AUC {f.pr_auc.mean():.3f} +- {f.pr_auc.std():.3f} | {r['seconds'] / 60:.1f} min\n"
          f"    chosen per fold: {[x['params'] for x in r['folds']]}", flush=True)


def load_baseline(out_dir):
    paths = [out_dir / f"oof_{BASE}.csv", out_dir / f"inner_{BASE}.csv", out_dir / f"run_{BASE}.json"]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"baseline files missing in {out_dir}: {missing}. Run python src/phase6_thresholds.py first.")
    return read_csv(paths[0]), read_csv(paths[1]), json.loads(paths[2].read_text())


def gate_g1(oof, meta, phase3_dir):
    """G1: the fresh baseline against Phase 3's saved nested run (same images, folds and grid)."""
    p3 = read_csv(phase3_dir / f"oof_{MODEL}__{IMB}.csv")
    p3_run = json.loads((phase3_dir / f"run_{MODEL}__{IMB}.json").read_text())
    m = p3.merge(oof, on="image_id", suffixes=("_p3", ""), validate="one_to_one")
    assert len(m) == len(oof) == len(p3), "the baseline and Phase 3 out-of-fold files cover different images"
    assert (m.fold == m.fold_p3).all() and (m.y == m.y_p3).all(), "the baseline and Phase 3 disagree on folds or labels"
    par, par3 = [f["params"] for f in meta["folds"]], [f["params"] for f in p3_run["folds"]]
    res = dict(same_params=par == par3, params_baseline=par, params_phase3=par3,
               max_score_diff=float((m.score - m.score_p3).abs().max()),
               max_thr_diff=float((m.thr_tuned - m.thr_tuned_p3).abs().max()),
               max_fold_pr_auc_diff=float(max(abs(a["pr_auc"] - b["pr_auc"])
                                              for a, b in zip(meta["folds"], p3_run["folds"]))),
               tolerances=G1_TOL)
    res["passed"] = bool(res["same_params"] and res["max_score_diff"] <= G1_TOL["score"]
                         and res["max_thr_diff"] <= G1_TOL["thr"]
                         and res["max_fold_pr_auc_diff"] <= G1_TOL["fold_pr_auc"])
    return res


# ----------------------------------------------------------------------------- thresholds

def fold_thresholds(inner, folds, k_shrink=SHRINK_K):
    """G, R and R-shrunk thresholds of every outer fold, each chosen on that fold's inner out-of-fold scores only.
    `folds` are the run's per-fold dicts: their thr_tuned must equal G recomputed here (asserted, exactly)."""
    thr_tuned = {int(f["fold"]): f["thr_tuned"] for f in folds}
    rows = []
    for k, g in inner.groupby("outer_fold", sort=True):
        k = int(k)
        y, s, reg = g.y.to_numpy().astype(int), g.inner_score.to_numpy(dtype=float), g.region.to_numpy()
        thr_g = best_threshold(y, s)[0]
        assert thr_g == thr_tuned[k], (f"fold {k}: the global threshold recomputed from the inner scores ({thr_g!r}) "
                                       f"is not the run's thr_tuned ({thr_tuned[k]!r})")
        rows.append(dict(fold=k, region="all", arm="G", thr=thr_g, npos=int(y.sum()), w=np.nan))
        for r in REGIONS:
            mr = reg == r
            npos = int(y[mr].sum())
            assert 0 < npos < mr.sum(), f"fold {k}: the inner set of {r} has a single class"
            thr_r = best_threshold(y[mr], s[mr])[0]
            w = npos / (npos + k_shrink)
            rows += [dict(fold=k, region=r, arm="R", thr=thr_r, npos=npos, w=np.nan),
                     dict(fold=k, region=r, arm="R-shrunk", thr=w * thr_r + (1 - w) * thr_g, npos=npos, w=w)]
    return pd.DataFrame(rows)


def warn_unknown_regions(region, where):
    """Per-region rules fall back to the global threshold outside hand/leg/hip/shoulder. That is intended for "other";
    any other label is probably a data error, so it is reported rather than handled silently."""
    unknown = sorted(set(map(str, region)) - set(REGION_CODES))
    if unknown:
        print(f"    WARNING ({where}): region labels {unknown} are not region groups; they fall back to the global "
              "threshold", flush=True)


def image_thresholds(df, thr_tab, arm):
    """Each image's threshold under an arm: its own outer fold's value; per region for R / R-shrunk, where "other"
    (or any region without a value) falls back to the fold's global threshold."""
    g = thr_tab[thr_tab.arm == "G"].set_index("fold").thr
    t = g.reindex(df.fold.to_numpy()).to_numpy(dtype=float)
    if arm != "G":
        warn_unknown_regions(df.region, f"arm {arm}")
        reg_thr = thr_tab[thr_tab.arm == arm].set_index(["fold", "region"]).thr.to_dict()
        t = np.array([reg_thr.get((k, r), tg) for k, r, tg in zip(df.fold, df.region, t)], dtype=float)
    assert not np.isnan(t).any(), "an image has no threshold"
    return t


# ----------------------------------------------------------------------------- tables

def metrics_table(base, p4j, thr, thr4):
    """Pooled out-of-fold threshold metrics. G / R / R-shrunk on all training images ("overall") and on the four
    modelled regions ("4 regions", P4's population); P4's "4 regions" row is the routed Phase 4 system."""
    cols = ["arm", "group", "n", "n_pos", "tn", "fp", "fn", "tp", *THR_METRICS]
    rows = []
    for arm in ARMS:
        rows.append(metrics_by_group(base.y, base.score, base.region, thr[arm]).assign(arm=arm))
        rows.append(metrics_by_group(p4j.y, p4j.score, p4j.region, thr4[arm], group_names=[]).assign(
            arm=arm, group="4 regions"))
    t = metrics_by_group(p4j.y, p4j.score_p4, p4j.region, p4j.thr_p4.to_numpy())
    rows.append(t.assign(arm="P4", group=t.group.replace({"overall": "4 regions"})))
    return pd.concat(rows, ignore_index=True)[cols]


def paired_table(base, p4j, thr, thr4, n_boot, seed):
    """Paired bootstrap of every pre-registered pair (evaluate.paired_bootstrap, thresholds held fixed), plus the
    number of outer folds in which arm A's pooled F2 is higher (L1 reports it beside the CI)."""
    out = []
    for a, b in PAIRS:
        if "P4" in (a, b):
            d, pop = p4j, "4 regions"
            sc = {**{arm: d.score.to_numpy() for arm in ARMS}, "P4": d.score_p4.to_numpy()}
            th = {**thr4, "P4": d.thr_p4.to_numpy()}
        else:
            d, pop = base, "overall"
            sc, th = {arm: d.score.to_numpy() for arm in ARMS}, thr
        y, reg, fold = d.y.to_numpy().astype(int), d.region.to_numpy(), d.fold.to_numpy()
        pb = paired_bootstrap(y, sc[a], sc[b], th[a], th[b], groups=reg, n_boot=n_boot, seed=seed,
                              metrics=THR_METRICS)
        pb["group"] = pb.group.replace({"overall": pop})
        f2 = np.array([threshold_metrics(y[fold == k], sc[a][fold == k], th[a][fold == k])["f2"]
                       - threshold_metrics(y[fold == k], sc[b][fold == k], th[b][fold == k])["f2"]
                       for k in sorted(set(fold))])
        pb["folds_a_better"] = np.where((pb.group == pop) & (pb.metric == "f2"), int((f2 > 0).sum()), np.nan)
        pb["n_folds"] = len(f2)
        out.append(pb.assign(pair=f"{a} - {b}", a=a, b=b))
    return pd.concat(out, ignore_index=True)


def stability_table(thr_tab):
    """L4: each region's threshold over the outer folds, against the spread (SD) of the global threshold."""
    g = thr_tab[thr_tab.arm == "G"].sort_values("fold").thr
    rows = [dict(arm="G", region="all", sd=g.std(ddof=1), range=g.max() - g.min(),
                 per_fold="; ".join(f"{v:.4f}" for v in g), noisy="")]
    for arm in ("R", "R-shrunk"):
        for r in REGIONS:
            t = thr_tab[(thr_tab.arm == arm) & (thr_tab.region == r)].sort_values("fold").thr
            rows.append(dict(arm=arm, region=r, sd=t.std(ddof=1), range=t.max() - t.min(),
                             per_fold="; ".join(f"{v:.4f}" for v in t), noisy=bool(t.std(ddof=1) > g.std(ddof=1))))
    return pd.DataFrame(rows)


def verdict(lo, hi, name):
    if lo > 0:
        return f"{name} helps"
    if hi < 0:
        return f"{name} hurts"
    return "no detectable difference"


def lever1_verdicts(paired, stab):
    """L1 (primary), L2 (secondary), L4 flags, and the threshold rule that rule S2 selects."""
    def overall_f2(pair):
        r = paired[(paired.pair == pair) & (paired.group == "overall") & (paired.metric == "f2")].iloc[0]
        return dict(diff=float(r["diff"]), lo=float(r.lo), hi=float(r.hi), p_two_sided=float(r.p_two_sided),
                    folds_better=int(r.folds_a_better), n_folds=int(r.n_folds))
    l1, l2 = overall_f2("R - G"), overall_f2("R-shrunk - G")
    l1["verdict"], l2["verdict"] = verdict(l1["lo"], l1["hi"], "R"), verdict(l2["lo"], l2["hi"], "R-shrunk")
    sd_g = float(stab[stab.arm == "G"].sd.iloc[0])
    l4 = {r: dict(sd_R=float(stab[(stab.arm == "R") & (stab.region == r)].sd.iloc[0]),
                  sd_R_shrunk=float(stab[(stab.arm == "R-shrunk") & (stab.region == r)].sd.iloc[0]), sd_G=sd_g,
                  noisy=bool(stab[(stab.arm == "R") & (stab.region == r)].noisy.iloc[0])) for r in REGIONS}
    rule = "R" if l1["verdict"] == "R helps" else ("R-shrunk" if l2["verdict"] == "R-shrunk helps" else "G")
    return dict(L1=l1, L2=l2, L4=l4, rule_S2_threshold_rule=rule,
                L5="The bootstrap holds every threshold fixed and ignores the noise of choosing it, which is largest "
                   "for hip and shoulder (36-39 fractured images per inner set).")


# ----------------------------------------------------------------------------- figure

def fig_forest(paired, figdir, n_boot):
    setup()
    series = (("R - G", C["region"], "R minus G"), ("R-shrunk - G", C["third"], "R-shrunk minus G"),
              ("R - P4", C["unified"], "R minus P4 (routed Phase 4 models)"))
    groups = ["overall"] + REGIONS
    metrics = (("recall", "Recall"), ("specificity", "Specificity"), ("f2", "F2"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    ypos = np.arange(len(groups))[::-1]
    for ax, (m, lab) in zip(axes, metrics):
        for (pair, col, name), off in zip(series, (0.22, 0.0, -0.22)):
            sub = paired[(paired.pair == pair) & (paired.metric == m)].copy()
            sub = sub.assign(g=sub.group.replace({"4 regions": "overall"})).set_index("g").reindex(groups)
            d, lo, hi = sub["diff"].to_numpy(float), sub.lo.to_numpy(float), sub.hi.to_numpy(float)
            ax.errorbar(d, ypos + off, xerr=np.vstack([d - lo, hi - d]), fmt="o", color=col, ecolor=col,
                        elinewidth=1.6, capsize=3, markersize=5, markeredgecolor=C["surface"], markeredgewidth=1.0,
                        label=name)
        ax.axvline(0, color=C["muted"], lw=1.2, ls=(0, (3, 2)), zorder=1)
        ax.set_title(lab)
        ax.set_xlabel("difference")
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, ["Overall"] + [REGION_LABEL[r] for r in REGIONS])
    header(fig, "Lever 1: per-region thresholds on the unified RBF SVM (paired difference, 95% CI)", axes[0],
           ncols=3, top=0.83)
    save(fig, figdir / "phase6_thresholds.png",
         f"Pooled out-of-fold over the 3,266 training images, each image at its own fold's thresholds, all chosen on "
         f"inner out-of-fold scores. 'Overall' is all images for the comparisons with G and the four modelled regions "
         f"for the comparison with P4. Paired bootstrap, {n_boot:,} resamples, thresholds held fixed (so the noise "
         f"of choosing them is not in the interval). The ranking is the same in G, R and R-shrunk.")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true", help="only rebuild tables, figure and verdicts")
    ap.add_argument("--force", action="store_true", help="re-run the baseline even if it is saved")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42, help="model seed")
    ap.add_argument("--boot-seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase6")
    ap.add_argument("--phase3-dir", type=Path, default=REPO_ROOT / "results" / "phase3")
    ap.add_argument("--phase4-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args(argv)
    if args.out_dir.resolve() in (args.phase3_dir.resolve(), args.phase4_dir.resolve()):
        raise SystemExit("--out-dir must not be the Phase 3 or Phase 4 results folder (their --summarize reads it)")

    have = (args.out_dir / f"oof_{BASE}.csv").exists()
    if args.summarize and not have:
        raise SystemExit(f"no saved baseline in {args.out_dir}; run python src/phase6_thresholds.py first")
    ran = not have or (args.force and not args.summarize)
    if ran:
        run_baseline(args)
    else:
        print(f"[skip] the G1 baseline is already saved in {args.out_dir} (use --force to re-run it)", flush=True)
    oof, inner, meta = load_baseline(args.out_dir)

    g1 = gate_g1(oof, meta, args.phase3_dir)
    (args.out_dir / "gate_G1.json").write_text(json.dumps(g1, indent=1, default=str))
    print(f"\n=== Gate G1 against {args.phase3_dir / f'oof_{MODEL}__{IMB}.csv'} ===\n"
          f"    same parameters in every fold: {g1['same_params']}  {g1['params_baseline']}\n"
          f"    max |score diff| {g1['max_score_diff']:.2e} (tol {G1_TOL['score']:g}) | "
          f"max |threshold diff| {g1['max_thr_diff']:.2e} (tol {G1_TOL['thr']:g}) | "
          f"max |fold PR-AUC diff| {g1['max_fold_pr_auc_diff']:.2e} (tol {G1_TOL['fold_pr_auc']:g})\n"
          f"    G1 {'PASSED' if g1['passed'] else 'FAILED'}", flush=True)
    if ran and not args.no_log:
        f = pd.DataFrame(meta["folds"])
        log_run(regions="all (pooled)", feature_set="all five families [default]",
                model=SPECS[MODEL].label, imbalance_strategy=f"{IMB} + F2 threshold tuning",
                key_hyperparameters=f"Phase 6 gate G1: nested 5x4 CV of the Phase 3 model on this machine; "
                                    f"params per fold={json.dumps([x['params'] for x in meta['folds']])}; "
                                    f"seed={meta['seed']}; n_jobs={meta['n_jobs']}",
                cv_score=f"PR-AUC={f.pr_auc.mean():.3f}+-{f.pr_auc.std():.3f}; ROC-AUC={f.roc_auc.mean():.3f}; "
                         f"tuned-thr recall={f.tuned_recall.mean():.3f} F2={f.tuned_f2.mean():.3f}",
                test_metrics="not used (CV only)",
                notes=f"Phase 6 gate G1 {'passed' if g1['passed'] else 'FAILED'}: vs Phase 3 max score diff "
                      f"{g1['max_score_diff']:.1e}, max thr diff {g1['max_thr_diff']:.1e}, max fold PR-AUC diff "
                      f"{g1['max_fold_pr_auc_diff']:.1e}; saved as results/phase6/oof_{BASE}.csv")
    if not g1["passed"]:
        raise SystemExit("Gate G1 FAILED: the fresh baseline does not reproduce Phase 3 within the protocol's "
                         "tolerances. Phase 6 stops here (docs/phase6_gap_closing.md, G1).")

    # ---- lever 1
    thr_tab = fold_thresholds(inner, meta["folds"])
    thr = {arm: image_thresholds(oof, thr_tab, arm) for arm in ARMS}
    p4 = read_csv(args.phase4_dir / f"oof_{MODEL}__{IMB}.csv")
    p4j = oof.merge(p4[["image_id", "y", "fold", "region", "score", "thr_tuned"]].rename(
        columns={"y": "y_p4", "fold": "fold_p4", "region": "region_p4", "score": "score_p4", "thr_tuned": "thr_p4"}),
        on="image_id", validate="one_to_one")
    assert len(p4j) == len(p4) == int(oof.region.isin(REGIONS).sum()), "P4 does not cover the four modelled regions"
    assert ((p4j.y == p4j.y_p4) & (p4j.fold == p4j.fold_p4) & (p4j.region == p4j.region_p4)).all(), \
        "the baseline and the Phase 4 out-of-fold file disagree on labels, folds or regions"
    thr4 = {arm: image_thresholds(p4j, thr_tab, arm) for arm in ARMS}
    met = metrics_table(oof, p4j, thr, thr4)
    paired = paired_table(oof, p4j, thr, thr4, args.n_boot, args.boot_seed)
    stab = stability_table(thr_tab)
    ver = lever1_verdicts(paired, stab)
    ver.update(n_boot=args.n_boot, boot_seed=args.boot_seed, shrink_k=SHRINK_K, gate_G1_passed=g1["passed"],
               date=pd.Timestamp.now().isoformat(timespec="seconds"))

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    thr_tab.to_csv(args.out_dir / "thresholds_per_fold.csv", index=False)
    met.round(4).to_csv(args.tables_dir / "phase6_thresholds_metrics.csv", index=False)
    paired.round(4).to_csv(args.tables_dir / "phase6_thresholds_paired.csv", index=False)
    stab.round(4).to_csv(args.tables_dir / "phase6_threshold_stability.csv", index=False)
    (args.out_dir / "lever1_verdicts.json").write_text(json.dumps(ver, indent=1, default=str))
    fig_forest(paired, args.figures_dir, args.n_boot)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== Thresholds per outer fold (margin), chosen on the inner out-of-fold scores ===")
    print(thr_tab.pivot_table(index=["arm", "region"], columns="fold", values="thr", sort=False).round(4).to_string())
    print("\n=== Pooled out-of-fold threshold metrics ===")
    show = met[met.group.isin(["overall", "4 regions"] + REGIONS)]
    print(show.pivot_table(index="group", columns="arm", values="f2", sort=False).round(3).to_string())
    print("\n=== Paired differences (95% CI, thresholds fixed) ===")
    pv = paired.assign(text=[fmt_ci(r["diff"], r.lo, r.hi) for _, r in paired.iterrows()])
    print(pv.pivot_table(index=["pair", "group"], columns="metric", values="text", aggfunc="first",
                         sort=False)[list(THR_METRICS)].to_string())
    print(f"\nL1 (primary)  R - G overall F2 {fmt_ci(ver['L1']['diff'], ver['L1']['lo'], ver['L1']['hi'])}, "
          f"R better in {ver['L1']['folds_better']}/{ver['L1']['n_folds']} folds -> {ver['L1']['verdict'].upper()}")
    print(f"L2 (secondary) R-shrunk - G overall F2 {fmt_ci(ver['L2']['diff'], ver['L2']['lo'], ver['L2']['hi'])}, "
          f"better in {ver['L2']['folds_better']}/{ver['L2']['n_folds']} folds -> {ver['L2']['verdict'].upper()}")
    print("L4 threshold stability (SD over folds; noisy if above the global threshold's SD "
          f"{ver['L4']['hand']['sd_G']:.4f}): " + ", ".join(f"{r} {v['sd_R']:.4f}{' NOISY' if v['noisy'] else ''}"
                                                            for r, v in ver["L4"].items()))
    print(f"L5: {ver['L5']}")
    print(f"Rule S2 (threshold rule if adopted): {ver['rule_S2_threshold_rule']}")
    if not args.no_log:
        log_run(regions="per region (hand, leg, hip, shoulder) + overall", feature_set="n/a (G1 baseline scores)",
                model="RBF SVM (fresh unified baseline), per-region thresholds",
                imbalance_strategy=f"{IMB} + F2 thresholds G / R / R-shrunk (k={SHRINK_K})",
                key_hyperparameters=f"Phase 6 lever 1; thresholds from inner out-of-fold scores; paired bootstrap "
                                    f"n_boot={args.n_boot}; seed={args.boot_seed}",
                cv_score=f"overall F2 R-G {fmt_ci(ver['L1']['diff'], ver['L1']['lo'], ver['L1']['hi'])}; "
                         f"R-shrunk-G {fmt_ci(ver['L2']['diff'], ver['L2']['lo'], ver['L2']['hi'])}",
                test_metrics="not used (CV only)",
                notes=f"Phase 6 lever 1: L1 {ver['L1']['verdict']}; L2 {ver['L2']['verdict']}; nothing fitted "
                      f"beyond gate G1")
    print(f"\nWrote {args.out_dir / 'lever1_verdicts.json'} and 3 tables in {args.tables_dir}")


if __name__ == "__main__":
    main()
