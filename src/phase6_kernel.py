"""Phase 6, lever 2: a shared-plus-region-specific RBF kernel on the unified model. CROSS-VALIDATION ONLY.

    python src/phase6_kernel.py --gate                             # G2: mtrbf_all at lambda = 0 vs the G1 baseline
    python src/phase6_kernel.py --kernels mtrbf_all mtrbf_hip      # primary + secondary nested CV, full grid
    python src/phase6_kernel.py --diagnostic --kernels mtrbf_all   # one nested run per fixed lambda in {0.25, 1, 4}
    python src/phase6_kernel.py --summarize                        # tables, figure, verdicts K1-K4 and rule S
    options: --n-jobs 4 (must match the G1 baseline)  --n-boot 2000  --boot-seed 0  --force  --no-log

Kernel (src/model_zoo.py, RegionKernelSVC):  K((x, r), (x', r')) = rbf(x, x') * (1 + lambda * M(r, r'))
  mtrbf_all (primary)    M = [r == r'] over the five region groups
  mtrbf_hip (secondary)  M = [r == r' == hip]
Grid: Phase 3's C x gamma x lambda {0, 0.25, 1, 4}, in the same nested CV as Phase 3 (inner selection by mean overall
PR-AUC, F2 threshold from the winner's inner out-of-fold scores). lambda = 0 is the unified model, so the region term
has to win the inner CV to be used. The region comes from the metadata and travels as the last column of X.

Gate G2: mtrbf_all with lambda fixed at 0 must be BIT-IDENTICAL to the G1 baseline written by phase6_thresholds.py
(same parameters, inner PR-AUCs, outer scores and thresholds in every fold). --kernels and --diagnostic refuse to run
until G2 has passed.

Comparisons (protocol: docs/phase6_gap_closing.md). Partners on the same images and folds: the G1 baseline ("unified",
G3) and the Phase 4 region models (results/phase4/oof_rbfsvm__none.csv). PR-AUC / ROC-AUC: mean of the per-fold
differences, images resampled within folds (phase4_regional.fold_paired_bootstrap); threshold metrics pooled
out-of-fold at each model's fold-wise thresholds (evaluate.paired_bootstrap). Verdicts K1-K4 for mtrbf_all; the same
statistics for mtrbf_hip (secondary, cannot be adopted). The fixed-lambda diagnostics are never used to choose lambda.
With lever 1's verdicts (results/phase6/lever1_verdicts.json) rule S is applied, including the fifth arm kernel+R and
its own L1 check when both levers are adopted (S3).

Outputs: results/phase6/oof_<run>.csv, run_<run>.json, inner_<kernel>.csv, gate_G2.json, lever2_verdicts.json,
         results/tables/phase6_kernel_summary.csv, phase6_kernel_paired.csv, phase6_kernel_lambda.csv,
         (phase6_kernel_s3.csv when S3 applies), results/figures/phase6_kernel_lambda.png
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cv_protocol import edge_warnings, load_train, nested_cv
from evaluate import REGIONS, metrics_by_group, paired_bootstrap
from expt_log import log_run
from model_zoo import LAMBDA_GRID, SPECS, grid_for, with_region_tag
from paths import REPO_ROOT
from phase4_regional import _fast_ap, _fast_auc, fold_paired_bootstrap
from phase6_thresholds import (BASE, IMB, THR_METRICS, fmt_ci, fold_thresholds, image_thresholds, read_csv,
                               verdict)
from phase6_thresholds import MODEL as BASE_MODEL
from phase5_viz import C, REGION_LABEL, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

PRIMARY, SECONDARY = "mtrbf_all", "mtrbf_hip"
ROLE = {PRIMARY: "primary", SECONDARY: "secondary"}
GATE = f"{PRIMARY}__lam0"
DIAG_LAMBDAS = (0.25, 1, 4)
EXPECTED_TRAIN = (3266, 576)              # the frozen training split; None skips the check
UNI, P4 = "unified (G1 baseline)", "P4 region models"


def diag_name(lam):
    return f"{PRIMARY}__lam{lam:g}"


def fixed_grid(model, lam):
    g = {k: list(v) for k, v in grid_for(model).items()}
    g["lam"] = [lam]
    return g


# ----------------------------------------------------------------------------- runs

def run_kernel(tr, model, grid, name, role, args, return_inner=False):
    """Nested CV of a region-kernel model on all training images (region tag appended as the last column of X)."""
    mode = SPECS[model].extra["region_mode"]
    X, w = with_region_tag(tr["X"], tr["col_weights"], tr["region"], mode)
    print(f"\n=== {role}: {SPECS[model].label} -> {name} | grid={grid} ===", flush=True)
    r = nested_cv(X, tr["y"], tr["fold"], model, IMB, w, grid=grid, seed=args.seed, n_jobs=args.n_jobs,
                  return_inner=return_inner)
    del X
    warns = edge_warnings(r["folds"], grid)
    f = pd.DataFrame(r["folds"])
    lams = [x["params"]["lam"] for x in r["folds"]]
    print(f"    outer-fold PR-AUC {f.pr_auc.mean():.3f} +- {f.pr_auc.std():.3f} | lambda per fold {lams} | "
          f"{r['seconds'] / 60:.1f} min\n    chosen per fold: {[x['params'] for x in r['folds']]}", flush=True)
    for wm in warns:
        print(f"    grid edge: {wm} (reported; the grid is not extended, K3)", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(image_id=tr["image_id"], fold=tr["fold"], y=tr["y"], region=tr["region"], score=r["oof"],
                      thr_default=r["thr_default"], thr_tuned=r["thr_tuned"])).to_csv(
        args.out_dir / f"oof_{name}.csv", index=False)
    if return_inner:
        pd.concat([pd.DataFrame(dict(outer_fold=k, image_id=tr["image_id"][idx], fold=tr["fold"][idx],
                                     y=tr["y"][idx], region=tr["region"][idx], inner_score=s))
                   for k, (idx, s) in sorted(r["inner"].items())], ignore_index=True).to_csv(
            args.out_dir / f"inner_{name}.csv", index=False)
    chosen = Counter(json.dumps(x["params"], sort_keys=True) for x in r["folds"]).most_common()
    meta = dict(run=name, role=role, model=model, region_mode=mode, imbalance=IMB,
                grid={k: [str(v) for v in vs] for k, vs in grid.items()}, seed=args.seed, n_jobs=args.n_jobs,
                tag=tr["config"], n_train=int(len(tr["y"])), n_pos=int(tr["y"].sum()), seconds=r["seconds"],
                edge_warnings=warns, lam_per_fold=lams, folds=r["folds"], modal_params=json.loads(chosen[0][0]),
                date=pd.Timestamp.now().isoformat(timespec="seconds"))
    (args.out_dir / f"run_{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    if not args.no_log:
        log_run(regions="all (pooled)", feature_set="all five families [default] + region tag",
                model=SPECS[model].label, imbalance_strategy=f"{IMB} + F2 threshold tuning",
                key_hyperparameters=f"nested 5x4 CV; grid C x gamma x lambda {grid['lam']}; lambda per fold={lams}; "
                                    f"modal params={chosen[0][0]} ({chosen[0][1]}/5 folds); seed={args.seed}",
                cv_score=f"PR-AUC={f.pr_auc.mean():.3f}+-{f.pr_auc.std():.3f}; ROC-AUC={f.roc_auc.mean():.3f}; "
                         f"tuned-thr recall={f.tuned_recall.mean():.3f} F2={f.tuned_f2.mean():.3f}",
                test_metrics="not used (CV only)",
                notes=f"Phase 6 lever 2, {role} ({name})" + ("; " + "; ".join(warns) if warns else ""))
    return meta


def gate_g2(out_dir):
    """G2: mtrbf_all at lambda = 0 against the G1 baseline, from the saved files, with no tolerance at all."""
    base, base_meta = read_csv(out_dir / f"oof_{BASE}.csv"), json.loads((out_dir / f"run_{BASE}.json").read_text())
    gate, gate_meta = read_csv(out_dir / f"oof_{GATE}.csv"), json.loads((out_dir / f"run_{GATE}.json").read_text())
    same_rows = len(gate) == len(base) and bool((gate.image_id.to_numpy() == base.image_id.to_numpy()).all())
    res = dict(
        same_rows=same_rows,
        same_params=[{k: f["params"][k] for k in ("C", "gamma")} for f in gate_meta["folds"]]
        == [f["params"] for f in base_meta["folds"]],
        lam_all_zero=all(f["params"]["lam"] == 0 for f in gate_meta["folds"]),
        inner_pr_auc_identical=[f["inner_pr_auc"] for f in gate_meta["folds"]]
        == [f["inner_pr_auc"] for f in base_meta["folds"]],
        scores_identical=same_rows and bool(np.array_equal(gate.score.to_numpy(), base.score.to_numpy())),
        thresholds_identical=same_rows and bool(np.array_equal(gate.thr_tuned.to_numpy(), base.thr_tuned.to_numpy())),
        max_abs_score_diff=float(np.abs(gate.score.to_numpy() - base.score.to_numpy()).max()) if same_rows else None,
        n_jobs=dict(gate=gate_meta.get("n_jobs"), baseline=base_meta.get("n_jobs")))
    res["passed"] = all(res[k] for k in ("same_rows", "same_params", "lam_all_zero", "inner_pr_auc_identical",
                                         "scores_identical", "thresholds_identical"))
    return res


def require_gate(out_dir):
    p = out_dir / "gate_G2.json"
    if not p.exists() or not json.loads(p.read_text()).get("passed"):
        raise SystemExit("Gate G2 has not passed (results/phase6/gate_G2.json). Run python src/phase6_kernel.py --gate "
                         "first; the kernel runs stay blocked until it passes.")


# ----------------------------------------------------------------------------- tables

def load_runs(out_dir):
    runs = {}
    for n in [GATE, PRIMARY, SECONDARY] + [diag_name(l) for l in DIAG_LAMBDAS]:
        if (out_dir / f"oof_{n}.csv").exists():
            runs[n] = (read_csv(out_dir / f"oof_{n}.csv"), json.loads((out_dir / f"run_{n}.json").read_text()))
    return runs


def per_fold(d, col, fn):
    return np.array([fn(g.y.to_numpy().astype(int), g[col].to_numpy(dtype=float))
                     for _, g in d.groupby("fold", sort=True)])


def summary_rows(arm, d, groups, score="score", thr="thr_tuned"):
    """Fold-mean ranking metrics and pooled threshold metrics of one arm, per group."""
    mt = metrics_by_group(d.y, d[score], d.region, d[thr].to_numpy()).set_index("group")
    rows = []
    for g in groups:
        mg = "overall" if g in ("overall", "4 regions") else g
        row = dict(arm=arm, group=g, n=int(mt.loc[mg, "n"]), n_pos=int(mt.loc[mg, "n_pos"]),
                   chance=float(mt.loc[mg, "prevalence"]), **{m: float(mt.loc[mg, m]) for m in THR_METRICS})
        if g != "4 regions":          # the routed Phase 4 system mixes four models' scales: no ranking metric
            sub = d if g == "overall" else d[d.region == g]
            pr, roc = per_fold(sub, score, _fast_ap), per_fold(sub, score, _fast_auc)
            row.update(pr_auc_fold=pr.mean(), pr_auc_fold_sd=pr.std(ddof=1), roc_auc_fold=roc.mean(),
                       roc_auc_fold_sd=roc.std(ddof=1))
        rows.append(row)
    return rows


def summary_table(base, p4, runs):
    rows = summary_rows(UNI, base, ["overall"] + REGIONS)
    for name, (oof, _) in runs.items():
        rows += summary_rows(name, oof, ["overall"] + REGIONS)
    rows += summary_rows(P4, p4, ["4 regions"] + REGIONS)
    return pd.DataFrame(rows)


def paired_table(base, p4, runs, n_boot, seed):
    """Kernel minus unified and kernel minus P4 (descriptive), plus P4 minus unified (the gap being closed)."""
    rows = []

    def add(run, partner, group, metric, vals):
        d, lo, hi, p = vals
        rows.append(dict(run=run, partner=partner, group=group, metric=metric, diff=d, lo=lo, hi=hi, p_two_sided=p))

    p4s = p4[["image_id", "score", "thr_tuned"]].rename(columns={"score": "score_p4", "thr_tuned": "thr_p4"})
    j4 = base.merge(p4s, on="image_id", validate="one_to_one")
    for g in REGIONS:
        d = j4[j4.region == g]
        for m, v in fold_paired_bootstrap(d.y, d.score_p4, d.score, d.fold, n_boot=n_boot, seed=seed).items():
            add("P4", "unified", g, m, v)
    for name in (PRIMARY, SECONDARY):
        if name not in runs:
            continue
        k = runs[name][0][["image_id", "score", "thr_tuned"]]
        j = base.merge(k, on="image_id", suffixes=("_u", ""), validate="one_to_one")
        for g in ["overall"] + REGIONS:
            d = j if g == "overall" else j[j.region == g]
            for m, v in fold_paired_bootstrap(d.y, d.score, d.score_u, d.fold, n_boot=n_boot, seed=seed).items():
                add(name, "unified", g, m, v)
        pb = paired_bootstrap(j.y, j.score, j.score_u, j.thr_tuned.to_numpy(), j.thr_tuned_u.to_numpy(),
                              groups=j.region, n_boot=n_boot, seed=seed, metrics=THR_METRICS)
        for r in pb.to_dict("records"):
            add(name, "unified", r["group"], r["metric"], (r["diff"], r["lo"], r["hi"], r["p_two_sided"]))
        jp = j.merge(p4s, on="image_id", validate="one_to_one")
        for g in REGIONS:
            d = jp[jp.region == g]
            for m, v in fold_paired_bootstrap(d.y, d.score, d.score_p4, d.fold, n_boot=n_boot, seed=seed).items():
                add(name, "P4", g, m, v)
        pb = paired_bootstrap(jp.y, jp.score, jp.score_p4, jp.thr_tuned.to_numpy(), jp.thr_p4.to_numpy(),
                              groups=jp.region, n_boot=n_boot, seed=seed, metrics=THR_METRICS)
        for r in pb.to_dict("records"):
            add(name, "P4", "4 regions" if r["group"] == "overall" else r["group"], r["metric"],
                (r["diff"], r["lo"], r["hi"], r["p_two_sided"]))
    return pd.DataFrame(rows, columns=["run", "partner", "group", "metric", "diff", "lo", "hi", "p_two_sided"])


def lambda_table(runs):
    rows = []
    for name, (_, meta) in runs.items():
        for f in meta["folds"]:
            rows.append(dict(run=name, role=meta["role"], fold=f["fold"], C=f["params"]["C"],
                             gamma=f["params"]["gamma"], lam=f["params"]["lam"], inner_pr_auc=f["inner_pr_auc"],
                             outer_pr_auc=f["pr_auc"], thr_tuned=f["thr_tuned"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- verdicts

def _ci(row):
    return dict(diff=float(row["diff"]), lo=float(row["lo"]), hi=float(row["hi"]), p_two_sided=float(row["p_two_sided"]))


def k_verdicts(summ, paired, runs):
    s = summ.set_index(["arm", "group"])
    p4u = paired[(paired.run == "P4") & (paired.metric == "pr_auc")].set_index("group")
    out = {}
    for name in (PRIMARY, SECONDARY):
        if name not in runs:
            continue
        meta = runs[name][1]
        p = paired[(paired.run == name) & (paired.partner == "unified") & (paired.metric == "pr_auc")].set_index("group")
        hip = p.loc["hip"]
        worse = [g for g in ("hand", "leg", "shoulder", "overall") if p.loc[g, "hi"] < 0]
        if hip.lo > 0:
            k1 = "trades regions" if worse else "closes part of the hip gap"
        elif hip.hi < 0:
            k1 = "hip worse"
        else:
            k1 = "no detectable difference"
        hk, hu, hp = (float(s.loc[(a, "hip"), "pr_auc_fold"]) for a in (name, UNI, P4))
        lams = meta["lam_per_fold"]
        n0, ntop = sum(l == 0 for l in lams), sum(l == max(LAMBDA_GRID) for l in lams)
        final = "inner CV does not select the region term" if n0 >= 4 else k1
        out[name] = dict(
            role=ROLE[name],
            K1=dict(verdict=k1, hip=_ci(hip), worse_in=worse, by_group={g: _ci(p.loc[g]) for g in p.index}),
            K2=dict(gap_closed=(hk - hu) / (hp - hu) if hp > hu else None,
                    hip_pr_auc_fold=dict(kernel=hk, unified=hu, P4=hp),
                    kernel_minus_unified=_ci(hip), P4_minus_unified=_ci(p4u.loc["hip"])),
            K3=dict(lam_per_fold=lams, n_lambda_zero=n0, overrides_K1=n0 >= 4, n_lambda_top=ntop,
                    grid_edge_top=ntop >= 3, edge_warnings=meta["edge_warnings"]),
            K4="Hip was singled out by Phase 5 on the same training folds: nested CV makes the score honest, but a "
               "positive hip result is weaker evidence than a pre-specified one.",
            verdict=final, adopted=name == PRIMARY and final == "closes part of the hip gap")
    return out


def s3_fifth_arm(out_dir, base, runs, rule, n_boot, seed):
    """Rule S3 (both levers adopted): kernel+rule, per-region thresholds chosen on the kernel's own inner out-of-fold
    scores, and its own L1 check (overall F2, kernel+rule minus kernel at its global threshold)."""
    k_oof, k_meta = runs[PRIMARY]
    tab = fold_thresholds(read_csv(out_dir / f"inner_{PRIMARY}.csv"), k_meta["folds"])
    t_rule, t_g = image_thresholds(k_oof, tab, rule), image_thresholds(k_oof, tab, "G")
    y, sc, reg = k_oof.y.to_numpy().astype(int), k_oof.score.to_numpy(), k_oof.region.to_numpy()
    pb = paired_bootstrap(y, sc, sc, t_rule, t_g, groups=reg, n_boot=n_boot, seed=seed, metrics=THR_METRICS)
    met = pd.concat([metrics_by_group(y, sc, reg, t_rule).assign(arm=f"kernel+{rule}"),
                     metrics_by_group(y, sc, reg, t_g).assign(arm="kernel")], ignore_index=True)
    f2 = pb[(pb.group == "overall") & (pb.metric == "f2")].iloc[0]
    check = dict(_ci(f2), verdict=verdict(f2["lo"], f2["hi"], f"kernel+{rule}"))
    tab.to_csv(out_dir / f"thresholds_per_fold_{PRIMARY}.csv", index=False)
    return met, pb.assign(pair=f"kernel+{rule} - kernel"), check


def rule_s(kv, lever1, s3_check=None):
    if PRIMARY not in kv:
        return dict(phase6_model=None, reason="mtrbf_all has not been run")
    kernel = kv[PRIMARY]["adopted"]
    if lever1 is None:
        return dict(phase6_model=None, kernel_adopted=kernel,
                    reason="lever 1 verdicts missing: run python src/phase6_thresholds.py")
    rule = lever1["rule_S2_threshold_rule"]
    if not kernel:
        model = rule
    elif rule == "G":
        model = "kernel"
    else:
        model = f"kernel+{rule}" if s3_check and s3_check["verdict"].endswith("helps") else "kernel"
    return dict(phase6_model=model, kernel_adopted=kernel, threshold_rule=rule, L1=lever1["L1"]["verdict"],
                L2=lever1["L2"]["verdict"], lever2=kv[PRIMARY]["verdict"], S3_check=s3_check)


# ----------------------------------------------------------------------------- figure

def fig_lambda(summ, runs, figdir):
    curve = [(l, n) for l, n in [(0, GATE)] + [(l, diag_name(l)) for l in DIAG_LAMBDAS] if n in runs]
    if len(curve) < 2:
        print("    [figure] fewer than two fixed-lambda runs saved: phase6_kernel_lambda.png skipped", flush=True)
        return
    setup()
    s = summ.set_index(["arm", "group"])
    groups = ["overall"] + REGIONS
    fig, axes = plt.subplots(1, len(groups), figsize=(12.5, 3.2))
    x = np.arange(len(curve))
    for ax, g in zip(axes, groups):
        v = np.array([s.loc[(n, g), "pr_auc_fold"] for _, n in curve])
        sd = np.array([s.loc[(n, g), "pr_auc_fold_sd"] for _, n in curve])
        ax.errorbar(x, v, yerr=sd, fmt="-o", color=C["third"], ecolor=C["third"], elinewidth=1.2, capsize=3,
                    label="kernel at a fixed lambda (C, gamma re-tuned)")
        if g in REGIONS:
            ax.axhline(s.loc[(P4, g), "pr_auc_fold"], color=C["region"], ls=(0, (4, 2)), lw=1.4,
                       label="Phase 4 region model")
        ax.axhline(s.loc[(UNI, g), "chance"], color=C["muted"], ls=(0, (1, 2)), lw=1.2, label="chance (fracture rate)")
        ax.set_xticks(x, ["0\n(unified)" if l == 0 else f"{l:g}" for l, _ in curve])
        ax.set_title("Overall" if g == "overall" else REGION_LABEL[g])
        ax.set_xlabel("lambda")
    axes[0].set_ylabel("fold-mean PR-AUC")
    header(fig, "Lever 2 diagnostic: PR-AUC per region at a fixed lambda (mtrbf_all)", axes[1], ncols=3, top=0.80)
    save(fig, figdir / "phase6_kernel_lambda.png",
         "DIAGNOSTIC ONLY: choosing lambda from this figure would be tuning on the outer folds. Each point is a full "
         "nested CV with lambda fixed and C, gamma tuned inside; lambda = 0 is the gate run G2 (= the unified model). "
         "Error bars: SD over the 5 outer folds. PR-AUC is comparable within a panel, not across regions.")


# ----------------------------------------------------------------------------- main

def summarize(args):
    out = args.out_dir
    base = read_csv(out / f"oof_{BASE}.csv")
    p4 = read_csv(args.phase4_dir / f"oof_{BASE_MODEL}__{IMB}.csv")
    runs = load_runs(out)
    if not runs:
        raise SystemExit(f"no kernel runs saved in {out}")
    for name, (oof, _) in runs.items():
        assert len(oof) == len(base) and (oof.image_id.to_numpy() == base.image_id.to_numpy()).all() \
            and (oof.y.to_numpy() == base.y.to_numpy()).all() and (oof.fold.to_numpy() == base.fold.to_numpy()).all(), \
            f"{name}: rows, labels or folds differ from the G1 baseline"
    summ = summary_table(base, p4, runs)
    paired = paired_table(base, p4, runs, args.n_boot, args.boot_seed)
    lam_tab = lambda_table(runs)
    kv = k_verdicts(summ, paired, runs)
    lever1 = json.loads((out / "lever1_verdicts.json").read_text()) if (out / "lever1_verdicts.json").exists() else None
    s3 = None
    if kv.get(PRIMARY, {}).get("adopted") and lever1 and lever1["rule_S2_threshold_rule"] != "G":
        s3 = s3_fifth_arm(out, base, runs, lever1["rule_S2_threshold_rule"], args.n_boot, args.boot_seed)
    rs = rule_s(kv, lever1, s3[2] if s3 else None)
    g2 = json.loads((out / "gate_G2.json").read_text()) if (out / "gate_G2.json").exists() else None

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    summ.round(4).to_csv(args.tables_dir / "phase6_kernel_summary.csv", index=False)
    paired.round(4).to_csv(args.tables_dir / "phase6_kernel_paired.csv", index=False)
    lam_tab.round(4).to_csv(args.tables_dir / "phase6_kernel_lambda.csv", index=False)
    if s3:
        pd.concat([s3[0], s3[1]], ignore_index=True).round(4).to_csv(args.tables_dir / "phase6_kernel_s3.csv",
                                                                       index=False)
    (out / "lever2_verdicts.json").write_text(json.dumps(
        dict(kernels=kv, rule_S=rs, gate_G2_passed=bool(g2 and g2["passed"]), n_boot=args.n_boot,
             boot_seed=args.boot_seed, date=pd.Timestamp.now().isoformat(timespec="seconds")), indent=1, default=str))
    fig_lambda(summ, runs, args.figures_dir)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== Fold-mean PR-AUC (training images, out-of-fold) ===")
    print(summ.pivot_table(index="arm", columns="group", values="pr_auc_fold", sort=False)
          .reindex(columns=["overall"] + REGIONS).round(3).to_string())
    if len(paired):
        print("\n=== Paired differences (95% CI) ===")
        pv = paired.assign(text=[fmt_ci(r["diff"], r["lo"], r["hi"]) for r in paired.to_dict("records")])
        print(pv.pivot_table(index=["run", "partner", "group"], columns="metric", values="text", aggfunc="first",
                             sort=False).to_string())
    for name, v in kv.items():
        print(f"\n{name} ({v['role']}): lambda per fold {v['K3']['lam_per_fold']}"
              f"{' - grid edge at the top' if v['K3']['grid_edge_top'] else ''}")
        print(f"  K1 hip PR-AUC kernel - unified {fmt_ci(v['K1']['hip']['diff'], v['K1']['hip']['lo'], v['K1']['hip']['hi'])}"
              f" -> {v['K1']['verdict']}" + (f" (worse in {v['K1']['worse_in']})" if v["K1"]["worse_in"] else ""))
        gap = v["K2"]["gap_closed"]
        print(f"  K2 gap closed: {'n/a' if gap is None else f'{gap:.2f}'}  (hip fold-mean PR-AUC kernel "
              f"{v['K2']['hip_pr_auc_fold']['kernel']:.3f}, unified {v['K2']['hip_pr_auc_fold']['unified']:.3f}, "
              f"P4 {v['K2']['hip_pr_auc_fold']['P4']:.3f})")
        print(f"  K3 lambda = 0 in {v['K3']['n_lambda_zero']}/5 folds{' -> OVERRIDES K1' if v['K3']['overrides_K1'] else ''}")
        print(f"  VERDICT: {v['verdict']}" + ("" if v["role"] == "primary" else " (secondary, cannot be adopted)"))
    if s3:
        print(f"\nS3 fifth arm: kernel+{lever1['rule_S2_threshold_rule']} minus kernel, overall F2 "
              f"{fmt_ci(s3[2]['diff'], s3[2]['lo'], s3[2]['hi'])} -> {s3[2]['verdict']}")
    print(f"\nRule S: Phase 6 model = {rs.get('phase6_model')}  ({rs.get('reason', '')})")
    if not args.no_log:
        log_run(regions="per region (hand, leg, hip, shoulder) + overall", feature_set="n/a (saved Phase 6 scores)",
                model="multi-task RBF SVM vs G1 baseline and Phase 4 region models",
                imbalance_strategy=f"{IMB} + F2 threshold tuning",
                key_hyperparameters=f"Phase 6 lever 2 summary; paired bootstrap n_boot={args.n_boot}; "
                                    f"seed={args.boot_seed}",
                cv_score="; ".join(f"{n}: {v['verdict']}" for n, v in kv.items()) or "no kernel runs",
                test_metrics="not used (CV only)", notes=f"rule S: {rs.get('phase6_model')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="G2: mtrbf_all with lambda fixed at 0 vs the G1 baseline")
    ap.add_argument("--kernels", nargs="+", choices=[PRIMARY, SECONDARY], default=None)
    ap.add_argument("--diagnostic", action="store_true", help="fixed-lambda runs of mtrbf_all (lambda 0.25, 1, 4)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run a saved run")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42, help="model seed")
    ap.add_argument("--boot-seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase6")
    ap.add_argument("--phase4-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args(argv)
    if not (args.gate or args.kernels or args.summarize):
        ap.error("choose --gate, --kernels ..., --diagnostic --kernels mtrbf_all or --summarize")
    if args.diagnostic and args.kernels != [PRIMARY]:
        ap.error("the diagnostics are pre-registered for mtrbf_all only: --diagnostic --kernels mtrbf_all")
    if args.out_dir.resolve() in (args.phase4_dir.resolve(), (REPO_ROOT / "results" / "phase3").resolve()):
        raise SystemExit("--out-dir must not be the Phase 3 or Phase 4 results folder (their --summarize reads it)")

    if args.gate or args.kernels:
        g1 = args.out_dir / "gate_G1.json"
        if not g1.exists() or not json.loads(g1.read_text()).get("passed"):
            raise SystemExit("Gate G1 has not passed: run python src/phase6_thresholds.py first")
        base_meta = json.loads((args.out_dir / f"run_{BASE}.json").read_text())
        if base_meta.get("n_jobs") != args.n_jobs:
            raise SystemExit(f"use --n-jobs {base_meta.get('n_jobs')}, as in the G1 baseline, so that both runs have "
                             "the same parallel layout (G2 compares them bit for bit)")
        tr = load_train()
        if EXPECTED_TRAIN:
            assert (len(tr["y"]), int(tr["y"].sum())) == EXPECTED_TRAIN, "unexpected training set"
    if args.gate:
        if (args.out_dir / f"oof_{GATE}.csv").exists() and not args.force:
            print(f"[skip] {GATE} already saved (use --force to re-run it)", flush=True)
        else:
            run_kernel(tr, PRIMARY, fixed_grid(PRIMARY, 0), GATE, "gate G2 (lambda fixed at 0)", args)
        g2 = gate_g2(args.out_dir)
        (args.out_dir / "gate_G2.json").write_text(json.dumps(g2, indent=1, default=str))
        print(f"\n=== Gate G2: {GATE} vs the G1 baseline, no tolerance ===\n    "
              + "  ".join(f"{k}={g2[k]}" for k in ("same_params", "lam_all_zero", "inner_pr_auc_identical",
                                                   "scores_identical", "thresholds_identical"))
              + f"\n    G2 {'PASSED' if g2['passed'] else 'FAILED'}", flush=True)
        if not g2["passed"]:
            raise SystemExit("Gate G2 FAILED: the kernel at lambda = 0 is not bit-identical to the unified baseline. "
                             "Phase 6 stops here (G2). Re-run G1 once only to tell non-determinism from a code-path "
                             "difference; the gate still counts as failed.")
    if args.kernels:
        require_gate(args.out_dir)
        todo = ([(PRIMARY, fixed_grid(PRIMARY, l), diag_name(l),
                  f"DIAGNOSTIC (lambda fixed at {l:g}; never used to choose lambda)", False) for l in DIAG_LAMBDAS]
                if args.diagnostic else
                [(m, grid_for(m), m, ROLE[m], True) for m in args.kernels])
        for model, grid, name, role, inner in todo:
            if (args.out_dir / f"oof_{name}.csv").exists() and not args.force:
                print(f"[skip] {name} already saved (use --force to re-run it)", flush=True)
                continue
            run_kernel(tr, model, grid, name, role, args, return_inner=inner)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
