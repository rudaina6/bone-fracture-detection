"""Every diagnosis number quoted in docs/phase7b_improvement.md that is not printed by diag_phase7b.py itself, recomputed
from the saved outputs (training images only). Needs a_patch_metrics.csv, a_ranks.csv, b_rounds.csv, visual_counts.csv
next to this file, and c_windows.csv.gz (regenerate with `diag_phase7b.py b` then `diag_phase7b.py c` if absent).

    python summary_numbers.py
"""
import sys
from pathlib import Path

REPO = Path(r"D:\bone-fracture-detection")
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

import phase7_detect as P

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 250, "display.max_columns", 40)


def cp(k, n):
    return (beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)


def main():
    print("== 1. patch level, round 0, held-out per outer fold (a_patch_metrics.csv)")
    pm = pd.read_csv(HERE / "a_patch_metrics.csv")
    cols = ["train_roc_auc", "pr_auc_13", "roc_auc_pool", "tpr_at_fpr_0.01", "tpr_at_fpr_0.001", "tpr_at_fpr_0.0001",
            "pool_neg_score_gt0"]
    print(pm[["fold"] + cols].round(4).to_string(index=False))
    print("min:", pm[cols].min().round(4).to_dict())
    print("max:", pm[cols].max().round(4).to_dict())
    print("mean:", pm[cols].mean().round(4).to_dict())

    print("\n== 2. rank of each box's best positive window among the image's windows, round 0 (a_ranks.csv)")
    r = pd.read_csv(HERE / "a_ranks.csv")
    q = r["rank"]
    print(f"all boxes {len(r)}: median {q.median()}, quartiles {q.quantile(.25)} / {q.quantile(.75)}")
    print("per region median:", r.groupby("region")["rank"].median().to_dict())
    side = pd.cut(r.side, [0, 20, 28, 40, 1000], labels=["<20", "20-28", "28-40", ">=40"])
    print("per box side median:", r.groupby(side, observed=True)["rank"].median().to_dict())
    print("share of all boxes with best positive window in the top K (= proposal recall at IoU 0.5):",
          {K: round(float((q <= K).mean()), 3) for K in (100, 250, 500, 1000, 2000)})
    pct = []
    for t in range(9):
        a, n = r.get(f"rank_in_t{t}"), r.get(f"n_t{t}")
        m = a.notna()
        pct.append(float((a[m] / n[m]).median()))
    print("within-type percentile (median) per type:", np.round(pct, 4).tolist())

    print("\n== 3. appearance, mask-free (c_windows.csv.gz)")
    d = pd.read_csv(HERE / "c_windows.csv.gz")
    G = {"fracture windows": d.kind == "gt_pos", "round-0 negatives used": d.kind == "pool_neg_used",
         "round-0 top false box": (d.kind == "det_r0_fp") & (d["rank"] == 0), "round-0 false boxes": d.kind == "det_r0_fp",
         "round-1 false boxes": d.kind == "det_r1_fp", "mined r1 (fold 0)": d.kind == "mined_r1",
         "mined r2 (fold 0)": d.kind == "mined_r2"}
    rows = []
    for k, m in G.items():
        x = d[m]
        rows.append(dict(group=k, n=len(x), clahe_median=x.clahe.median(), edge_density_median=x.edges.median(),
                         no_edge_pixel=(x.edges == 0).mean(), touches_padding=(x.pad_frac > 0).mean(),
                         any_saturated=(x.bright > 0).mean(), on_hardware_image=x.hardware.mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    f = d[G["fracture windows"]]
    for k in ("round-0 negatives used", "round-0 top false box"):
        x = d[G[k]]
        y = np.r_[np.ones(len(f)), np.zeros(len(x))]
        print(f"univariate ROC-AUC, fracture windows vs {k}:",
              {c: round(roc_auc_score(y, np.r_[f[c], x[c]]), 3) for c in ("clahe", "edges", "grad")})

    print("\n== 4. visual counts (one rater; visual_counts.csv) with exact binomial 95% CIs")
    v = pd.read_csv(HERE / "visual_counts.csv")
    for s, g in v.groupby("sample", sort=False):
        n = len(g)
        print(f"{s} (n={n}):", ", ".join(f"{c} {k} [{cp(k, n)[0]:.1%}, {cp(k, n)[1]:.1%}]"
                                         for c, k in g["class"].value_counts().items()))
    fp = v[v["sample"] == "round0_false_boxes_random"]
    k = int((~fp["class"].isin(["soft tissue", "border or artifact", "air"])).sum())
    print(f"round-0 false boxes on bone (any): {k}/40 [{cp(k, 40)[0]:.1%}, {cp(k, 40)[1]:.1%}]")

    print("\n== 5. mining, fold 0 (b_rounds.csv)")
    b = pd.read_csv(HERE / "b_rounds.csv")
    print(b[["round", "n_pos", "n_neg", "heldout_roc_auc_pool", "heldout_tpr_at_fpr_0.001", "heldout_pool_neg_score_gt0",
             "train_r0neg_score_gt0", "cos_prev_w", "repro_max_diff"]].round(4).to_string(index=False))
    br = pd.read_csv(HERE / "b_ranks.csv")
    print("median rank per round:", br.groupby("round")["rank"].median().to_dict(),
          "| top-10 NMS hits per round:", br.groupby("round").nms_hit.sum().to_dict())

    print("\n== 6. oracle image gating on the saved Phase 7 CV detections (boxes on normal images removed)")
    T = P.train_table()
    gt = P.gt_448(T, P.geometry_table(REPO / "results" / "phase7"))
    det, _ = P.load_fold_detections(REPO / "results" / "phase7", "cv")
    frac = set(T.image_id[T.fractured == 1])
    for rnd in (0, 2):
        a, o = [], []
        for k in range(5):
            imgs = list(T.image_id[T.fold == k])
            dk = det[det["round"] == rnd]
            a.append(P.EvalSet(imgs, dk, gt).ap()[0.5])
            o.append(P.EvalSet(imgs, dk[dk.image_id.isin(frac)], gt).ap()[0.5])
        print(f"round {rnd}: fold-mean AP@50 {np.mean(a):.5f} -> oracle gating {np.mean(o):.5f}")


if __name__ == "__main__":
    main()
