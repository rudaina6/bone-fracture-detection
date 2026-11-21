# Phase 6: gap-closing (protocol written 2026-11-21, before any result)

Task: bring the single unified model closer to what the region-specific models achieve, without training four
separate models (plan, Phase 6). Phase 5 located the gap in two different places, and each gets its own lever: the
**decision threshold** (5.1: specialising moved where the F2 threshold landed far more than it changed the ranking)
and the **hip ranking** (5.2: hip is a genuine region effect that the pooled model does not learn). Status:
`docs/PROJECT_STATUS.md`.

**This file records the protocol and the decision rules before any Phase 6 result was seen.** It was committed
before any Phase 6 code ran on the real features. Anything added after results are in is marked
`[added after results]` with the reason.

**The test set is used once**, by `src/phase6_final_test.py` with its own lock (`results/phase6/TEST_EVALUATED.json`),
after the CV results are written into this file and the choice of the Phase 6 model has been approved. Every run
before that uses the 3,266 training images and the frozen folds 0-4 only. The Phase 3 and Phase 4 locks stay as they
are; `phase3_final_test.py` and `phase4_final_test.py` are not run. **The Phase 4 hip test result (11 fractured
images) is not cited as evidence for anything in this phase** (`docs/phase5_analysis.md`, "The hip question").

## Starting point

| Finding | Numbers | Source |
|---|---|---|
| Unified model: RBF SVM, imbalance none, F2 threshold | nested-CV PR-AUC 0.602 +/- 0.036; C=10 in every outer fold, gamma 0.2 in fold 0 and 0.05 in folds 1-4 | Phase 3, `results/phase3/run_rbfsvm__none.json` |
| Region models change decisions more than rankings | CV PR-AUC differences hand -0.018, leg +0.023, shoulder -0.077, all CIs include 0; out-of-fold specificity hand 0.369 (region) vs 0.514 (unified), leg 0.899 vs 0.776 | Phase 4, 5.1 |
| Hip is a genuine region effect | CV PR-AUC +0.223 [+0.105, +0.324], better in 5/5 folds; a model trained without hip scores 0.267 on it (chance 0.18); transfer ratios into and out of hip -0.03 to +0.16 | Phase 4, 5.2 |
| The region is readable from the features | region classifier accuracy 0.94; macro-F1 0.765 ("other" recovered 1 of 19) | 5.2 |
| Routed Phase 4 system vs unified, CV | F2 +0.027 [+0.003, +0.050]; specificity +0.047 [+0.031, +0.065]; recall +0.005 [-0.026, +0.037] | Phase 4 |

## Mapping to the plan

| Plan item | Phase 6 | Why |
|---|---|---|
| 2. Per-region decision thresholds | **Lever 1**: one F2 threshold per region on the single unified model | 5.1: the routed system's recall/specificity trade came from where each region's threshold landed |
| 1. Region as an input feature | **Lever 2**: a shared-plus-region-specific kernel, which is this idea with the region weight decoupled from gamma (below) | 5.2: hip needs its own decision function and gets too little weight in the pooled one |
| 3. Per-region class weighting or resampling | **dropped** | Phase 3 found none, class weights and SMOTE equivalent once the threshold is tuned; a per-region class weight moves that region's operating point, which lever 1 does directly |

**The kernel.** For an image `x` from region `r`:

    K((x, r), (x', r')) = rbf(x, x') * (1 + lambda * M(r, r')),    lambda >= 0

where `rbf(x, x') = exp(-gamma * ||x - x'||^2)` on the Phase 3 features and preprocessing, and `M` marks pairs from
the same region (two versions below).

- **It is plan item 1 done properly.** A one-hot region column scaled by `s` adds `2 s^2` to the squared distance of
  any two images from different regions, so the RBF kernel becomes `rbf(x, x') * exp(-2 gamma s^2 [r != r'])`
  `= c * rbf(x, x') * (1 + lambda [r == r'])` with `lambda = exp(2 gamma s^2) - 1` and `c = exp(-2 gamma s^2)`. An
  SVM with kernel `c * K` and penalty `C` has the same decision function as kernel `K` with penalty `c * C`, so the
  constant is absorbed by C. The kernel is therefore the region-as-feature idea with lambda decoupled from gamma: with
  a feature column, every gamma tried during tuning would silently change the region weight too. (A column passed
  through the pipeline's `StandardScaler` would also be rescaled by its region's frequency, giving every pair of
  regions a different weight; the explicit kernel avoids that as well.)
- **It is a multi-task kernel** (Evgeniou and Pontil, *Regularized multi-task learning*, KDD 2004; with a non-linear
  base kernel as in Evgeniou, Micchelli and Pontil, JMLR 2005): the SVM learns `f(x, r) = f_shared(x) + f_r(x)`, and
  lambda sets how cheap the region-specific part is relative to the shared one. `lambda = 0` is exactly the unified
  model. As lambda grows the kernel approaches a block-diagonal one, i.e. separate per-region SVMs (each with an
  effective penalty `(1 + lambda) C`, sharing one intercept).
- **It is a valid kernel** for `lambda >= 0`: `rbf` and `1 + lambda * M` are both positive semi-definite (`M` is a
  block indicator) and so is their elementwise product. Negative lambda is not PSD in general and is excluded.

## Code

| File | Role |
|---|---|
| `src/cv_protocol.py` | `nested_cv(..., return_inner=False)`. With `return_inner=True` it also returns a separate top-level key `inner` = {outer fold: (row indices, inner out-of-fold scores of the winning configuration)}, the scores each fold's F2 threshold is chosen on. Nothing is added to `folds` (Phases 3 and 4 write it to JSON) and the default output is unchanged byte for byte. |
| `src/model_zoo.py` | `RegionKernelSVC` (a subclass of `PrecomputedRBFSVC`) and two specs, `mtrbf_all` and `mtrbf_hip`; existing specs and `DEFAULT_MODELS` are not touched. The region travels as a tag in the **last column of X** (`mtrbf_all`: one code per region group; `mtrbf_hip`: 1 = hip, 0 = rest), and `col_weights` gets one more entry of 1.0. `StandardScaler` and the block weight are increasing affine maps per column, so equal tags stay equal and the hip tag stays the larger one. The classifier strips the tag before the RBF, uses only tag equality (and, in hip mode, which value is larger, read at fit), and stores the support vectors' tags. With `lambda = 0` it skips the multiplication, so the kernel is bit-identical to the parent class. ValueError for `lambda < 0`, a non-binary tag in hip mode, or more than 5 distinct tags in all mode. |
| `src/phase4_regional.py` | `_fast_ap` / `_fast_auc` return NaN on a single-class subset instead of dividing by zero. No current caller passes one (they either check first or work on stratified folds that always hold both classes), so no Phase 4 or Phase 5 output changes. |
| `src/phase6_thresholds.py` | Gate G1 (the fresh unified baseline, run with `return_inner=True`), then lever 1: arms G, R, R-shrunk and the comparator P4; tables, JSON, verdicts L1-L5. |
| `src/phase6_kernel.py` | Lever 2: `--gate` (G2), `--kernels mtrbf_all mtrbf_hip` (nested CV), `--diagnostic` (fixed-lambda runs), `--summarize` (tables, figure, verdicts K1-K4). |
| `src/phase6_final_test.py` | The single test evaluation, T1-T6 below. Lock `results/phase6/TEST_EVALUATED.json`; refuses a second run unless `--allow-rerun`. |
| `tests/test_phase6.py` | Synthetic self-tests (list below), run before any real-data run: `python tests/test_phase6.py` (plain asserts; pytest is not installed). |
| reused unchanged | `evaluate.py` (`best_threshold`, `thr_vector`, `paired_bootstrap`, `bootstrap_ci`), `phase4_regional.fold_paired_bootstrap`, `cv_protocol.load_train` / `tune_cv`, `feature_utils.load_features`, `expt_log.log_run` |

Outputs: `results/phase6/` (per-run CSV/JSON), `results/tables/phase6_*.csv`, `results/figures/phase6_*.png`.
Nothing is written to `results/phase3/` or `results/phase4/`, whose `--summarize` would pick it up. Every fitting run
(and the threshold analysis) gets a row in `results/experiment_log.csv` with `test_metrics="not used (CV only)"` until
the final test. Seed 42 for models, seed 0 for bootstraps.

## Index: why each analysis is run, and what decides its answer

| Analysis | Why it is run | What decides the answer | Fixed when |
|---|---|---|---|
| Gate G1, fresh unified baseline | This machine reproduces Phase 3 to about 1e-3, not exactly, so every comparison needs a unified arm from the same machine | G1 tolerances | before |
| Lever 1, R vs G | Can per-region thresholds on one model deliver the operating-point gain of the routed system? | L1: paired CI of the overall F2 difference | before |
| Lever 1, R-shrunk vs G | Hip and shoulder thresholds rest on 36-39 fractured images per inner set; shrinkage trades that noise for bias | L2, same test, secondary | before |
| Per-region R vs G and R vs P4 | Does one model with per-region thresholds reproduce the Phase 4 decisions? | L3: descriptive, no verdict | before |
| Threshold stability | Is a region's threshold estimate stable enough to be worth having? | L4: fold SD against the global threshold's | before |
| Gate G2, kernel at lambda = 0 | At lambda = 0 the kernel must be the unified model, or any difference could be a code artefact | Bit-identical to G1 | before |
| Lever 2, `mtrbf_all` | Can one model with a region term close part of the hip gap without costing another region? | K1, with K3 taking precedence | before |
| Gap closed | How much of the Phase 4 hip gain the kernel recovers | K2: point estimate, not a verdict | before |
| Lever 2, `mtrbf_hip` | A region term for hip alone | K1-K3 statistics, secondary; cannot be adopted | before |
| lambda diagnostics | How per-region PR-AUC moves with lambda when C and gamma are re-tuned | Descriptive only; choosing lambda from them would be tuning on outer folds | before |
| Choosing the Phase 6 model | The test set gets one pre-registered model | Rule S | before |
| Final test | One honest number per arm | T1-T6 | before |

## Lever 1: per-region thresholds on the unified model (plan item 2)

**Arms**, all on the unified RBF SVM's scores (the fresh baseline, G3):
- **G**: one global F2 threshold per outer fold (Phase 3 as is).
- **R**: one F2 threshold per region (hand, leg, hip, shoulder), each maximising that region's own F2. "Other"
  (19 training images) falls back to the global threshold; `evaluate.thr_vector` already accepts
  `{region: thr, "overall": default}`.
- **R-shrunk** (secondary): `thr_r = w * thr_region + (1 - w) * thr_global` with `w = npos_r / (npos_r + 40)`, where
  `npos_r` is the number of fractured images of region r in the set the threshold is chosen on. **k = 40 is fixed here
  and never tuned.** It is about the number of fractured hip or shoulder images in an inner set (36-39), so those two
  regions get roughly half their own threshold and half the global one, while hand (about 244) and leg (about 136)
  keep 86% and 77% of theirs.
- **P4** (comparator): the routed Phase 4 region models at their own thresholds (`results/phase4/oof_rbfsvm__none.csv`).

**Where the thresholds come from.** Only from the **inner** out-of-fold scores inside the nested CV: the scores
Phase 3 already chose its global threshold on. For outer fold k, `nested_cv(..., return_inner=True)` returns the rows
of the other four folds and the winning configuration's inner out-of-fold scores on them; G, R and R-shrunk are all
computed from those and applied to fold k's outer scores. The G threshold recomputed this way must equal the fold's
`thr_tuned` exactly (asserted). Thresholds are **not** chosen from the pooled saved outer scores: in Phase 3, outer
fold 0 chose gamma = 0.2 and folds 1-4 gamma = 0.05, so margins from different folds are on different scales.

**Metrics.** Recall, precision, specificity and F2, pooled out-of-fold (each image at its own fold's thresholds),
overall and per region. "Overall" is all 3,266 training images, the population Phase 3 reported on; "other" gets the
global threshold in every lever-1 arm, so its 19 images contribute identically to G, R and R-shrunk. Comparisons with
P4 use the 3,247 images of the four modelled regions (P4 has no "other" model), as in Phase 4's routed system. No
PR-AUC comparison of G vs R: a threshold cannot change a ranking.

### Decision rules for lever 1

L1. **Primary:** overall F2, R minus G, paired bootstrap over images (`evaluate.paired_bootstrap`, 2,000 resamples,
    seed 0, thresholds fixed), reported with the number of outer folds in which R's F2 is higher. **"R helps"** if
    the 95% CI excludes 0 above, **"R hurts"** if it excludes 0 below, otherwise **"no detectable difference"**.
L2. R-shrunk minus G, judged the same way, reported as secondary.
L3. Descriptive, no verdict: per region, R vs G and R vs P4 (recall, specificity, F2, with paired CIs). The question
    is whether one model with per-region thresholds reproduces the Phase 4 decisions (5.1: hand more liberal, leg
    more conservative). Hip is expected **not** to be reproducible by thresholds, because its Phase 4 gain is in the
    ranking.
L4. Threshold stability: each region's threshold over the 5 outer folds. A region is flagged **noisy** if its spread
    (SD over the 5 folds) exceeds the spread of the global threshold. Both include the scale change of any fold
    whose model chose another gamma, so the comparison is like for like, but absolute threshold values are not
    comparable across folds.
L5. Stated beside every per-region number: the bootstrap holds thresholds fixed and ignores the noise of choosing
    them, which is largest for hip and shoulder (36-39 fractured images per inner set).

## Lever 2: shared-plus-region-specific kernel (plan item 1)

**Models**, both trained on all 3,266 training images, with the region read from the metadata (`region_group`; the
same assumption as Phase 4: the region is known when an image arrives):
- **`mtrbf_all` (primary)**: `M(r, r') = [r == r']` over the five region groups (hand, leg, hip, shoulder, other).
- **`mtrbf_hip` (secondary)**: `M(r, r') = [r == r' == hip]`; every other region shares fully.

**Grid and selection.** Phase 3's grid times lambda: C {0.1, 1, 10, 100} x gamma {0.8, 0.2, 0.05, 0.0125} x lambda
{0, 0.25, 1, 4}, 64 configurations, in the same nested CV as Phase 3 (`cv_protocol.nested_cv`, frozen folds, seed 42),
imbalance none, F2 threshold from the winner's inner out-of-fold scores. The winner of each outer fold is the
configuration with the best mean inner **overall** PR-AUC (the Phase 3 rule). lambda = 0 is the unified model, so the
region term has to win the inner CV to be used at all. Within each (C, gamma), lambda = 0 is listed first, so an exact
tie at the same C and gamma goes to the unified model.

**Runs**
- (a) **Gate G2**: `mtrbf_all` with lambda fixed at 0, C and gamma tuned over the full Phase 3 grid.
- (b) **Primary**: nested CV of `mtrbf_all`, full 64-configuration grid. Run with `return_inner=True` so that rule S3
  can be applied if needed.
- (c) **Secondary**: nested CV of `mtrbf_hip`, full grid.
- (d) **Diagnostics**: one nested run of `mtrbf_all` for each fixed lambda in {0.25, 1, 4}, C and gamma still tuned
  inside. With (a) as the lambda = 0 point they give the per-region PR-AUC-vs-lambda curve
  (`results/figures/phase6_kernel_lambda.png`, with the Phase 4 region model as a reference line). They are labelled
  as diagnostics everywhere: choosing lambda from them would be tuning on the outer folds.

Expected cost on this machine (4 cores, `n_jobs=4`; Phase 3's 16-configuration nested run took 141 s here): about
10 min each for (b) and (c), about 2.5 min for (a) and for each run in (d).

**Metrics**, as in Phase 4. Partners on the same images and folds: the **unified model, meaning the fresh baseline
G1** (G3), and the Phase 4 region models (`results/phase4/oof_rbfsvm__none.csv`). PR-AUC and ROC-AUC: mean of the
per-fold differences, images resampled within folds (`phase4_regional.fold_paired_bootstrap`, 2,000 resamples,
seed 0), per region and overall. The kernel is one model per fold, so its overall PR-AUC is meaningful (unlike the
routed Phase 4 system, which gets no overall ranking metric). Threshold metrics: pooled out-of-fold at each model's
fold-wise global F2 threshold (`evaluate.paired_bootstrap`). Kernel vs Phase 4 region model, per region, is reported
descriptively.

### Decision rules for lever 2

K1. **Primary:** hip PR-AUC, `mtrbf_all` minus unified. **"Closes part of the hip gap"** if the 95% CI excludes 0
    above **and** no other region (hand, leg, shoulder) and not the overall PR-AUC has a CI excluding 0 below. Hip
    better but another region or overall worse: **"trades regions"**, not adopted. Otherwise **"no detectable
    difference"**.
K2. **Gap closed** = (hip kernel - unified) / (hip Phase 4 region model - unified), fold-mean PR-AUC on hip's
    training images, reported as a point estimate beside both differences and their CIs. Not a verdict on its own.
K3. The lambda chosen in each outer fold is reported. If lambda = 0 wins in 4 or 5 of the 5 folds, the verdict is
    **"inner CV does not select the region term"**, whatever K1 and the diagnostics show; K3 takes precedence over K1.
    If lambda = 4 (the top of the grid) is chosen in at least 3 folds, that is reported as a grid edge and the grid is
    not extended (as Phase 4 rule 3); an extended-grid run, if any, is a labelled sensitivity run and never goes to
    the test set.
K4. Circularity caveat, stated with every hip result: hip was singled out by Phase 5 on the same training folds.
    Nested CV makes the score honest, but a positive hip result is weaker evidence than a pre-specified one would have
    been. That is why `mtrbf_all`, which treats every region alike, is primary and `mtrbf_hip` is secondary.

The same statistics are reported for `mtrbf_hip`, labelled secondary; it cannot become the Phase 6 model (rule S1).

## Reproduction gates

A preliminary, unlogged re-run on 2026-11-21, made during the read-only check of the repository before this protocol
was written, reproduced `results/phase3/oof_rbfsvm__none.csv` with the same chosen parameters in every fold, but not
bit for bit: largest score difference 8.8e-4, largest threshold difference 3.0e-4, largest fold PR-AUC difference
8.1e-5 (141 s). `tune_cv` gave C=10, gamma=0.05 and threshold -0.844039 (Phase 3: -0.843974). The cause is not
established: the features were regenerated on 2026-11-12, and float32 BLAS rounding may also differ between machines.
The tolerances below are 2-6 times those differences.

G1. **Fresh baseline.** Re-run `nested_cv` for `rbfsvm` / none, full grid, on this machine (`return_inner=True`), and
    save it as `results/phase6/oof_baseline_rbfsvm.csv`. Check against `results/phase3/oof_rbfsvm__none.csv`:
    identical chosen parameters in every fold, absolute score difference <= 2e-3, absolute threshold difference
    <= 1e-3, absolute fold PR-AUC difference <= 5e-4. Phase 6 stops if any check fails.
G2. **Kernel gate.** `mtrbf_all` with lambda fixed at 0 must be **bit-identical** to G1 (same machine, same code
    path): identical chosen parameters, outer scores and thresholds in every fold, compared with no tolerance. Phase 6
    stops if not. Two runs on one machine have not been compared yet; if G2 fails, G1 is re-run once, only to tell
    non-determinism from a code-path difference, and the gate still counts as failed.
G3. Every Phase 6 comparison against "unified" uses the fresh baseline G1, so both arms come from the same machine.
    Phase 3's saved out-of-fold file is used only for the G1 check. (The Phase 4 region scores were computed on the
    earlier machine; differences of order 1e-3 are far below any effect this phase can detect.)

## Choosing the Phase 6 model (rule S)

S1. Ranking model: the kernel `mtrbf_all` if K1 says "closes part of the hip gap" and K3 does not apply; otherwise the
    unified RBF SVM. `mtrbf_hip` and the diagnostic runs cannot be adopted.
S2. Threshold rule: R if L1 says "R helps"; otherwise R-shrunk if L2 says "R-shrunk helps"; otherwise G.
S3. The Phase 6 model is the pair. With the unified model that is G, R or R-shrunk; with the kernel and G it is
    "kernel" (one global F2 threshold). If both levers are adopted, a **fifth arm, "kernel+R"** (or kernel+R-shrunk,
    whichever S2 selected), is built: per-region thresholds chosen on the kernel's own **inner** out-of-fold scores
    (saved by run (b)) exactly as in lever 1, applied to the kernel's outer scores. It gets **its own L1 check**:
    overall F2, kernel+R minus kernel (global threshold), paired bootstrap as in L1. The Phase 6 model is kernel+R
    only if that check says "helps"; otherwise it is "kernel". Both numbers are reported beside the single-lever
    arms. *[Revised 2026-11-21 at the STOP 1 review, before any result: this replaces "reported with no further
    verdict".]*
S4. If neither lever is adopted, the Phase 6 model is G, the Phase 3 model unchanged; the test run still reports every
    arm (T6).

The CV verdicts and the model they select are written into this file and approved before the test run.

## Final test (run once, after approval)

All in `src/phase6_final_test.py`, outputs in `results/phase6/`.

T1. `tune_cv` for `rbfsvm` / none on the training images: assert C=10, gamma=0.05 and a threshold within 1e-3 of
    -0.843974 (`results/phase3/test_run.json`). Stop if not.
T2. Per-region thresholds (R, R-shrunk) from that configuration's `tune_cv` out-of-fold scores; `npos_r` = all
    fractured training images of the region (w = 0.88 hand, 0.81 leg, 0.53 hip, 0.55 shoulder). R-shrunk shrinks
    towards Phase 3's saved threshold (-0.843974), the G threshold on test, so that G stays exactly the Phase 3 result.
T3. Only if a kernel was adopted: `tune_cv` of the adopted kernel on the training images, refit on all 3,266, global
    F2 threshold (and the per-region thresholds under S3) from its `tune_cv` out-of-fold scores. If `tune_cv` picks
    lambda = 0, the kernel arm is the unified model refitted on this machine, and is reported as that.
T4. `results/phase6/TEST_EVALUATED.json` is written with every setting, every threshold and the CV-chosen Phase 6
    model **before** any test image is scored.
T5. G, R and R-shrunk reuse the Phase 3 test scores (`results/phase3/test_scores.csv`, no refit); G uses Phase 3's
    saved threshold, so G is exactly the Phase 3 test result. The kernel arm scores the 817 test images once. P4 comes
    from `results/phase4/test_scores.csv` (812 images of the four regions).
T6. Every pre-registered arm is reported whatever its CV verdict was (G, R, R-shrunk and P4 always; the kernel if it
    was adopted; `mtrbf_hip` and the diagnostics never touch the test set), with bootstrap CIs (2,000 resamples,
    seed 0) and paired differences R - G, R-shrunk - G, K - G (including PR-AUC), R - P4 and K - P4. For hip and
    shoulder the CI half-width is reported as the smallest detectable effect (an effect of that size would be detected
    about half the time); a null there is "not resolvable with this test set". No model or threshold is changed after
    the test numbers are seen.

## Correctness checks (synthetic data, before any real run)

`tests/test_phase6.py` must pass before any real-data run. It checks that:
- lambda = 0 is bit-identical to `PrecomputedRBFSVC` through `build()`, in both modes;
- the kernel matrix is PSD and equals `rbf * (1 + lambda * M)` elementwise, in both modes;
- region tags survive `StandardScaler` and the block weight on held-out rows;
- with a very large lambda, `mtrbf_hip`'s hip scores are highly correlated with those of a hip-only model;
- `sklearn.base.clone` works on `RegionKernelSVC`;
- `nested_cv(return_inner=False)` output is unchanged, and the global threshold recomputed from `inner` equals each
  fold's `thr_tuned`;
- `_fast_ap` / `_fast_auc` agree with sklearn on two-class data and return NaN on single-class data.

A mock run of `phase6_final_test.py` on synthetic data checks the lock logic; the script is not run on the real data
before approval.

## Known threats (carry into the report)

- **Threshold noise in small regions.** Hip and shoulder thresholds rest on 36-39 fractured images per inner set, and
  the bootstrap ignores the noise of choosing them (L5). R-shrunk is there to show how much of that noise shrinkage
  removes, at the price of pulling small regions towards the global operating point.
- **R is not a superset of G.** G's threshold maximises the pooled F2 on the inner scores; R's thresholds maximise
  each region's own F2. R can therefore have a lower overall F2 than G even on the scores it was chosen on, and L1 can
  come out "R hurts" without anything having gone wrong. L1 judges the pooled F2 because that is what the system is
  scored on; R is defined per region because that is what the Phase 4 region models did, which L3 compares against.
- **Overall PR-AUC selection can reward a region shift.** Overall PR-AUC pools regions with different fracture rates
  (hand 0.30, leg 0.10), so a region term can raise it by moving whole regions up or down without ranking better
  within any region. That is why K1 judges within-region PR-AUC; the inner selection still uses the overall rule, as
  in Phase 3.
- **Circularity of the hip focus** (K4).
- **Effective regularisation moves with lambda.** The within-region part of the kernel carries `(1 + lambda)` times
  the weight, so at a fixed C a larger lambda also regularises that part less. C and gamma are re-tuned at every
  lambda, which partly compensates, but the diagnostic curve mixes the two.
- **"Other" is its own task in `mtrbf_all`**, with 19 training images (7 fractured; about 15 images per inner training
  set). It is not reported as a region, but it is part of the overall numbers and of the inner selection.
- **Machines.** The fresh baseline and the kernel runs come from this machine; the saved Phase 3 and Phase 4 scores,
  including the Phase 3 test scores reused in T5, come from the earlier one. Differences of order 1e-3 are far below
  the effects the test can detect (the Phase 4 paired per-region test CIs spanned roughly 0.15-0.4).
- **No patient ID**, with an uneven multiscan share (hand 8.6%, leg 13.5%, hip and shoulder 0%), as in every earlier
  phase.
- **Region label assumed known** at prediction time, as in Phase 4.
- **Small test regions.** Hip and shoulder have 11 and 12 fractured test images; T6 states their power.

---

## Results

*(Empty until the runs. The protocol above was fixed before any result.)*

### Reproduce (planned)

Run with the repository's `.venv` Python on the Phase 6 machine (its bare `python` is 3.14 without imbalanced-learn).
```
python tests/test_phase6.py                                   # synthetic self-tests, must pass first
python src/phase6_thresholds.py                               # gate G1 + lever 1, ~2.5 min
python src/phase6_kernel.py --gate                            # gate G2, ~2.5 min
python src/phase6_kernel.py --kernels mtrbf_all mtrbf_hip     # ~10 min each
python src/phase6_kernel.py --diagnostic --kernels mtrbf_all  # 3 runs, ~2.5 min each
python src/phase6_kernel.py --summarize                       # tables, figure, verdicts K1-K4
python src/phase6_final_test.py ...                           # test set, ONCE, after approval (lock: results/phase6/TEST_EVALUATED.json)
```
