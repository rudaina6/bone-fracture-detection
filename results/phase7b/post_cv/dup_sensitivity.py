"""Phase 7b post-CV sensitivity (descriptive, training images, existing out-of-fold detections; nothing refitted or
re-selected): fold-mean AP@50 of the selected variant, S1 and P7 on the images that are NOT part of a near-duplicate
pair (near_duplicates.csv, full-image correlation > 0.999), with paired bootstrap differences (2000, seed 0).
    python results/phase7b/post_cv/dup_sensitivity.py
"""
import json, sys
from pathlib import Path
import argparse
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import phase7b_cascade as C
P = C.P
HERE = Path(__file__).resolve().parent
args = argparse.Namespace(out_dir=REPO / "results/phase7b", p7_dir=REPO / "results/phase7", n_boot=2000, boot_seed=0)
T = P.train_table(None)
D = pd.read_csv(HERE / "near_duplicates.csv")
D = D[D.full_corr > 0.999]
dup = set(D.a) | set(D.b)
cross = D[D.fold_a != D.fold_b]
dup_cross = set(cross.a) | set(cross.b)
keep_all = set(T.image_id) - dup
keep_cross = set(T.image_id) - dup_cross
v = json.loads((args.out_dir / "selection.json").read_text())["selected"]["variant"]
gt = P.gt_448(T, P.geometry_table(args.p7_dir))
det, _ = C.load_det(args)
dets = {v: det[det.variant == v], **C.references(args)}
extra = {"no_dup": keep_all, "no_crossfold_dup": keep_cross}
res = C.boot_ap(T, gt, dets, ["overall"] + list(extra), args.n_boot, args.boot_seed, extra)
out = {}
for g in ["overall"] + list(extra):
    out[g] = dict(n_images=len(T) if g == "overall" else len(extra[g]),
                  ap50={n: res[(n, g, "ap50")][0] for n in dets},
                  d_vs_S1=C.paired(res, v, "S1", g), d_vs_P7=C.paired(res, v, "P7", g))
(HERE / "dup_sensitivity.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
