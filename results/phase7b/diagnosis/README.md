# Phase 7 diagnosis (basis of Phase 7b)

Read-only diagnosis of the Phase 7 detector, run 2026-11-22 after the Phase 7 final test, on **training images and
saved CV outputs only**. It changed no Phase 7 file. It was run from a scratch folder; the scripts are copied here
unchanged (they find their inputs next to themselves and the repository at `D:\bone-fracture-detection`). Summary and
consequences: `docs/phase7b_improvement.md`, "Starting point".

| File | What it is |
|---|---|
| `diag_phase7b.py` | Part `a`: round-0 refit per outer fold, held-out patch metrics, rank of each box's best positive window. Part `b`: fold 0 rounds 0-2 reproduced exactly (`repro_max_diff` = 0), mined windows, per-round class balance and scores. Part `c`: per-window appearance statistics. |
| `summary_numbers.py`, `summary_numbers.log` | Every other number quoted in the 7b doc, recomputed from the saved outputs. |
| `a_patch_metrics.csv`, `a_ranks.csv`, `a.log` | Part `a` outputs. |
| `b_rounds.csv`, `b_mining_stats.csv`, `b_ranks.csv`, `b.log` | Part `b` outputs. |
| `c.log` | Part `c` log. Its table `c_windows.csv.gz` (16 MB) and `b_mined.csv` (7 MB) are not committed; rerun `python diag_phase7b.py b` then `c` (about 10 min) to regenerate them. |
| `visual_counts.csv` | Per-tile visual classes (one rater) for the montages below. |
| `m_neg.png`, `m_fp0.png`, `m_pos.png` | Numbered montages: 40 random round-0 negatives used in training, 40 random round-0 false boxes, 24 random fracture windows (`montage.py`). |
| `r_top_fp_round0.png`, `r_top_fp_round1.png` | The 24 highest-scoring false windows on normal images, rounds 0 and 1 (`diag_report.py`). |
| `r_mask_check.png`, `r_mask2_check.png` | Why automatic bone masks were rejected (`diag_report.py`, `mask2.py`). |

**Caveat.** The `bone`, `tissue` and `air` columns computed by part `c`, and the categories in `diag_report.py`, come
from a global 3-class Otsu mask that the mask check showed to be wrong (bright strips and labels become "bone", real
bone becomes "tissue"); they are not used anywhere. The mask-free columns (`clahe`, `raw`, `edges`, `grad`, `bright`,
`pad_frac`) and the visual counts are what the 7b doc quotes.
