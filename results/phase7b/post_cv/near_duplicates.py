"""Phase 7b post-CV observation (read-only, TRAINING images only): near-duplicate images across folds.
Found while checking the CV example figure (IMG0002935 / IMG0004217 look identical). Not part of the protocol;
nothing is refitted or re-selected from it.
    python results/phase7b/post_cv/near_duplicates.py     # writes near_duplicates.csv / .json next to itself
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import phase7_detect as P
HERE = Path(__file__).resolve().parent
T = P.train_table(None).reset_index(drop=True)          # training rows only
thumbs = []
for fp in T.file_path:
    im = Image.open(REPO / "data" / "FracAtlas" / fp).convert("L").resize((32, 32), Image.BILINEAR)
    v = np.asarray(im, dtype=np.float32).ravel()
    v = (v - v.mean()) / (v.std() + 1e-6)
    thumbs.append(v / np.sqrt(len(v)))
X = np.stack(thumbs)
S = X @ X.T
np.fill_diagonal(S, -1)
cand = np.argwhere(np.triu(S > 0.99, 1))
rows = []
for i, j in cand:
    a = np.asarray(Image.open(REPO / "data" / "FracAtlas" / T.file_path[i]).convert("L"), dtype=np.float32)
    b = np.asarray(Image.open(REPO / "data" / "FracAtlas" / T.file_path[j]).convert("L"), dtype=np.float32)
    full = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.shape == b.shape else np.nan
    rows.append(dict(a=T.image_id[i], b=T.image_id[j], fold_a=int(T.fold[i]), fold_b=int(T.fold[j]),
                     region_a=T.region[i], region_b=T.region[j], fr_a=int(T.fractured[i]), fr_b=int(T.fractured[j]),
                     thumb_corr=float(S[i, j]), full_corr=full))
import pandas as pd
D = pd.DataFrame(rows)
D.to_csv(HERE / "near_duplicates.csv", index=False)
dup = D[D.full_corr > 0.999] if len(D) else D
imgs = set(dup.a) | set(dup.b) if len(dup) else set()
fr_ids = set(T.image_id[T.fractured == 1])
summ = dict(training_images=len(T), candidate_pairs_thumb_gt_0_99=len(D), pairs_full_corr_gt_0_999=len(dup),
            pairs_cross_fold=int((dup.fold_a != dup.fold_b).sum()) if len(dup) else 0,
            pairs_both_fractured=int(((dup.fr_a == 1) & (dup.fr_b == 1)).sum()) if len(dup) else 0,
            images_involved=len(imgs), fractured_images_involved=len(imgs & fr_ids),
            fractured_images_total=len(fr_ids))
(HERE / "near_duplicates.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
