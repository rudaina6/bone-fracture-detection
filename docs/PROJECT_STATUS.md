# Project status (living document)

**Last updated:** 2026-11-23 (Phase 7 and Phase 7b complete and closed; Phase 8 next) · **Read this first** if you are picking up the project. The plan is in
`Fracture_Project_Phases.md`; this file says where we actually are, what was decided, and how to reproduce it.
Percentages are rough, by planned effort (weeks in the plan), not by calendar.

## Dashboard

| Phase | What | Status | Done | Where the work is |
|---|---|---|---|---|
| 0 | Setup | Complete | 100% | `README.md`, `requirements.txt`, `docs/design_note.md`, `results/experiment_log.csv` |
| 1 | Data preparation and EDA | Complete | 100% | `data/master.csv`, `results/splits.csv`, `notebooks/01_eda.ipynb`, `docs/data_decisions.md` |
| 2 | Preprocessing and features | **Complete** | 100% | `docs/phase2_features.md`, `src/preprocess.py`, `src/feature_extractors.py`, `src/extract_features.py`, `src/compare_families.py`, `src/feature_utils.py` |
| 3 | Unified classifier | **Complete** | 100% | `docs/phase3_unified.md`, `src/evaluate.py`, `src/model_zoo.py`, `src/cv_protocol.py`, `src/phase3_unified.py`, `src/phase3_final_test.py`, `results/phase3/`, `results/tables/phase3_*.csv` |
| 4 | Region-specific classifiers | **Complete** | 100% | `docs/phase4_regional.md`, `src/phase4_regional.py`, `src/phase4_final_test.py`, `results/phase4/`, `results/tables/phase4_*.csv` |
| 5 | Comparative analysis | **Complete** | 100% | `docs/phase5_analysis.md`, `src/phase5_main.py`, `src/phase5_curves.py`, `src/phase5_features.py`, `src/phase5_errors.py`, `src/phase5_box_contrast.py`, `src/phase5_viz.py`, `results/phase5/`, `results/tables/phase5_*.csv`, `results/figures/phase5_*.png` |
| 6 | Gap-closing | **Complete** | 100% | `docs/phase6_gap_closing.md`, `src/phase6_thresholds.py`, `src/phase6_kernel.py`, `src/phase6_final_test.py`, `tests/test_phase6.py`, `results/phase6/`, `results/tables/phase6_*.csv`, `results/figures/phase6_*.png` |
| 7 | Localization | **Complete** | 100% | `docs/phase7_localization.md`, `src/phase7_window_design.py`, `src/phase7_windows.py`, `src/phase7_detect.py`, `src/phase7_final_test.py`, `results/phase7/`, `results/tables/phase7_*.csv`, `results/figures/phase7_*.png` |
| 7b | Localization improvement (two-stage cascade, post-hoc follow-up) | **Complete, closed** | 100% | `docs/phase7b_improvement.md`, `src/phase7b_cascade.py`, `src/phase7b_final_test.py`, `tests/test_phase7b.py`, `results/phase7b/`, `results/tables/phase7b_*.csv`, `results/figures/phase7b_*.png` |
| 8 | Report and presentation | Not started | 0% | |
| | **Whole project** | | **~90%** | |

## Phase 2 summary (full detail: `docs/phase2_features.md`)
- **Choice:** 256 px, CLAHE, all five handcrafted families (8,641 dims), `StandardScaler` inside the CV folds, each family weighted 1/sqrt(dims), no dimensionality reduction. Reference linear SVM: C=0.1, `class_weight="balanced"`.
- **Score of the reference model** (5-fold CV on the 3,266 training images): PR-AUC 0.552 +/- 0.041 (chance 0.176), ROC-AUC 0.813, recall about 0.68 and F2 about 0.60 at the default threshold.
- **Compared and not better:** z-score instead of CLAHE (0.550), 448 px (0.537), PCA (0.541), tree selection (0.544); L1 selection failed (selected nothing after block weighting). Removing high-resolution images (0.540) or hardware images (0.525) changes the score only within noise.
- **Two pipelines, one conclusion:** Aiman independently built Phase 2 on `aiman_branch` (Oct 17-21). Both are merged: the shared pipeline in `src/` is the one to use from Phase 3 on; hers is kept in `src/aiman/`, `src/Ablations.py` and `notebooks/02_features.ipynb` (see `src/aiman/README.md` and the cross-check table in `docs/phase2_features.md`).
- **Hand-off:** `from feature_utils import load_features, make_preprocessor` (see the top of `docs/phase2_features.md`). Features are regenerated with `python src/extract_features.py` (not committed).
- These CV scores also guided the choices, so they are slightly optimistic; the test set is untouched.
- **Re-extraction check (2026-11-12):** the features were regenerated (`python src/extract_features.py`) and the family comparison was re-run on them (`python src/compare_families.py --block-weight --C 0.1`). All five families again score PR-AUC 0.552 +/- 0.041, and `results/tables/feature_family_s256_m448_clahe_c16_none_C0.1_bw.csv` came out identical to the committed table. The re-run's 17 rows (dated 2026-11-12) are in `results/experiment_log.csv`; before it, the chosen configuration had no rows in the log.

## Phase 3 summary (full detail: `docs/phase3_unified.md`)
- **Chosen model:** RBF SVM (C=10, gamma=0.05), no class weights or SMOTE, plus an F2-tuned threshold (-0.844 on the margin). Compared with logistic regression L1/L2, linear SVM, random forest, gradient boosting under nested CV; imbalance strategies none / class weights / SMOTE compared for the RBF and linear SVMs (differences within noise; none is cheapest).
- **Nested CV (training images):** PR-AUC 0.602 +/- 0.036 (chance 0.176), tuned recall 0.80, precision 0.35, F2 0.63. The Phase 2 linear SVM scores 0.548 under the same protocol (Phase 2 reported 0.552).
- **Test set (used once):** recall 0.755 [0.684, 0.830], precision 0.411, F2 0.647 [0.585, 0.707], specificity 0.770, ROC-AUC 0.834, PR-AUC 0.626 [0.544, 0.711]; confusion TN/FP/FN/TP 519/155/35/108. Per region: hand F2 0.69, leg 0.58, hip 0.65, shoulder 0.69, all CIs overlapping. No sign of implants driving the score (recall 0.756 without hardware, 0.750 with).
- **Phase 4 must use:** `model_zoo.build/SPECS`, `cv_protocol.nested_cv`, `evaluate.*` and the same imbalance strategy (none + tuned threshold). `results/phase3/test_scores.csv` holds the unified model's per-image test scores for the Phase 5 comparison (paired bootstrap in `evaluate.paired_bootstrap`). Do not run another model selection on the test set.
- Re-run lock: `results/phase3/TEST_EVALUATED.json`.

## Phase 4 summary (full detail: `docs/phase4_regional.md`)
- **Setup:** one RBF SVM per region (hand, leg, hip, shoulder) with the Phase 3 features, grid, nested CV and imbalance strategy (none + F2 threshold), trained on that region's training rows only. "Other" (7 fractured) not modelled. Compared with the unified model on the same images (paired bootstrap).
- **Nested CV:** hip region model clearly better (PR-AUC +0.223 [+0.105, +0.324], 5/5 folds), hand/leg/shoulder no detectable difference (shoulder borderline worse, mostly hyperparameter-tuning noise on 9-10 positives per fold). Linear SVM per region gives the same picture; no region prefers another family.
- **Test set (used once):** no detectable difference in any region (PR-AUC differences hand +0.045, leg +0.013, hip -0.096, shoulder -0.044, all CIs include 0). The CV hip gain did not replicate (11 fractured hip test images). Routed system (each image to its region's model): recall 0.844 vs 0.759, specificity 0.708 vs 0.773, F2 0.677 vs 0.650 (difference within noise); the hand model's F2 threshold is very liberal (specificity 0.26).
- **For Phase 5 (all answered - see `docs/phase5_analysis.md`):** the inputs were `results/phase4/test_scores.csv` (both models' per-image test scores and thresholds), `results/phase4/oof_*.csv` and `results/phase3/oof_*.csv` (same folds). The hip CV-vs-test question is resolved: the multiscan-leakage hypothesis was **wrong** (hip has 0% multiscan), the CV effect is **real and mechanistically explained**, and whether it generalises is **unprovable with 11 test positives**. Per-region thresholds moved from an aside to one of the two Phase 6 levers.
- Re-run lock: `results/phase4/TEST_EVALUATED.json`.

## Phase 5 summary (full detail: `docs/phase5_analysis.md`)
- **Discipline:** Phase 5 fits nothing on the test set and chooses no model or threshold. It re-reads the per-image scores Phases 3 and 4 already wrote; the test locks are untouched. Everything that needs fitting (learning curves, transfer, family ablation) runs on the training folds only. Protocol and decision rules were written before any result.
- **5.1 (done):** the Phase 4 PR-AUC verdicts are reproduced exactly by an independent recomputation (asserted in `phase5_main.py`). **New result:** the two approaches rank almost identically but *decide* very differently. McNemar on the test decisions is highly significant in hand (only-unified-right 71 vs 20, p=7e-08) and in leg (only-region-right 30 vs 11, p=0.004), in opposite directions. The cause is threshold calibration, not the learned ranking: the hand region model's F2 threshold gives specificity 0.26, the leg region model's is more conservative than the unified one (0.898 vs 0.843). Splitting per region changes the threshold more than it changes the model.
- **5.3 (done):** the median fracture is **14.0 px** (hand) and 15.3 px (shoulder) in the 256 px input, against a 16 px HOG cell — the typical lesion is smaller than the cell its gradient histogram is pooled over. Hip's are twice as large (31.6 px). "Small fractures are missed" holds in **leg only** (rank AUC 0.69 [0.60, 0.78]; recall 0.36 -> 0.80 across size bins), not in hand (0.48) or shoulder. Multi-scan images are detected more often (+0.14 [+0.06, +0.21]), which is the no-patient-ID leakage threat made measurable — and hip has 0% multiscan, so **leakage cannot explain hip's CV advantage**.
- **5.2 (done):** the correctness gate passed exactly (region arm at full size reproduces `oof_rbfsvm__none__fixed.csv`, largest difference 0.00e+00). **Hip is a genuine region effect; hand, leg and shoulder are not.** A unified model trained on *zero* images of a region reaches 0.528 on hand, 0.430 on leg and 0.523 on shoulder — but only **0.267 on hip against a 0.18 chance level**, while hip's own 204 images reach **0.733**. Hip's advantage *widens* with data (+0.083 -> +0.209), the opposite of a scarcity signature. Transfer agrees: every off-diagonal ratio in hip's row and column is between -0.03 and +0.16, while hand<->shoulder share (+0.52, +0.51) and leg->hand is +0.99. Mechanism: the region can be read from the features (a region classifier reaches accuracy 0.94; its macro-F1 is 0.765 because "other" is recovered 1 of 19 times, and hand/leg are confused 73 and 50 times), and hip's linear fracture direction is near-orthogonal to the pooled one (cosine 0.193 vs hand 0.625) while hip supplies only 7.8% of the images. Caveat recorded: shoulder's "flat" verdict reflects a large fold SD (0.123), not a genuinely flat curve.
- **For Phase 6:** two separate levers, both pointed at by the evidence. (1) Per-region **thresholds** on the single unified model — 5.1 shows specialising changes the threshold far more than the ranking, so most of the routed system's recall/specificity trade should be reachable without four models. (2) A **shared-plus-region-specific kernel** has real headroom specifically for hip: the region information is in the features (region classifier accuracy 0.94, macro-F1 0.765) but the unified model is not exploiting it. The hip CV effect is settled as real and mechanistically explained; whether it **generalises** is not established and the 11-positive test set could not establish it — Phase 6 must not cite the Phase 4 hip test number as support.

## Phase 6 summary (full detail: `docs/phase6_gap_closing.md`)
- **Discipline:** protocol, decision rules, gates and the rule that picks "the Phase 6 model" were committed before any result (`0922a08`, revised at STOP 1 in `fabc80e`). Test set used once, own lock `results/phase6/TEST_EVALUATED.json`.
- **Levers:** (1) per-region F2 thresholds on the unified RBF SVM (plan item 2; arms G, R, R-shrunk with k=40); (2) a shared-plus-region-specific kernel `rbf * (1 + lambda [same region])` (plan item 1 done properly; Evgeniou-Pontil multi-task kernel), `mtrbf_all` primary, `mtrbf_hip` secondary. Plan item 3 dropped (imbalance strategies equivalent after threshold tuning).
- **Gates:** this machine reproduces Phase 3 to ~1e-3, not bit for bit (G1 passed: max score diff 8.8e-4); the kernel at lambda=0 is bit-identical to the unified baseline (G2 passed). All Phase 6 "unified" numbers use the fresh baseline.
- **CV (training images):** per-region thresholds: overall F2 R-G +0.015 [-0.000, +0.031] (no detectable difference), R-shrunk-G +0.015 [+0.001, +0.029] (helps); every region's threshold is noisy across folds (hip SD 0.33 vs global 0.06). Kernel `mtrbf_all`: hip PR-AUC +0.117 [+0.026, +0.208], no region significantly worse, lambda>0 chosen in 5/5 folds, closes 52% of the hip gap to the Phase 4 hip model. Rule S selected **kernel+R-shrunk** (fifth-arm check vs kernel +0.024 [+0.009, +0.040]); CV F2 0.659 vs 0.631.
- **Test (used once):** Phase 6 model F2 0.665 [0.612, 0.714] vs Phase 3 0.647: difference **+0.018 [-0.026, +0.067], the CV gain is not confirmed**; it buys recall (+0.084 [+0.028, +0.146]) with specificity (-0.082). Hip PR-AUC -0.114 [-0.321, +0.093]: **not resolvable** with 11 fractured hip images. **Shoulder is worse:** F2 -0.129 [-0.275, -0.034] (specificity), the one clear test effect. Against the routed Phase 4 system: -0.013 [-0.036, +0.007], no detectable difference.
- **Answer to the plan's question:** one model with a region term and per-region thresholds matches the four region models (no measurable gap left) but does not measurably beat the unified model on test. The hip region effect is real in CV and reachable by a shared kernel; whether it generalises still needs more hip test data.
- **Secondary finding:** the hip-only kernel `mtrbf_hip` recovered almost as much hip (+0.101 [+0.030, +0.174]) with no region pointing down in CV; it was not adoptable under rule S1 and was not tested.

## Phase 7 summary (full detail: `docs/phase7_localization.md`)
- **Design:** sliding windows {16, 40, 64} px (nine types) on the 448 px CLAHE image, handcrafted window features, linear SVM (C 0.1), 2 rounds of hard-negative mining, NMS 0.3, at most 10 boxes per image; 93.9% of training boxes reachable at IoU 0.5.
- **CV (training images):** AP@50 0.0001 [0.0001, 0.0005]; round 0 (no mining) scored higher (0.0011) but D1 kept the planned 2 rounds. Best-window score as an image classifier: ROC-AUC 0.511 against Phase 3's 0.822.
- **Test (used once, CV failure known beforehand, nothing changed):** AP@50 0.0003 [0.0001, 0.0009]; 6 of 176 test boxes hit at IoU 0.5. **The sliding-window detector does not localize fractures.** Lock: `results/phase7/TEST_EVALUATED.json`.
- A read-only diagnosis on training images (`results/phase7b/diagnosis/`) found the false boxes are bone and joints as edge-rich as fractures, and that mining collapsed the detector; it motivated Phase 7b.

## Phase 7b summary (full detail: `docs/phase7b_improvement.md`)
- **Design:** two-stage cascade. Stage 1 = Phase 7 round-0 linear SVM, top 1,000 windows per image as proposals (cross-fitted: no stage-2 training proposal comes from a model that saw its image's fold). Stage 2 = HistGradientBoosting on window + context-ring features + window type + stage-1 score, trained on positives and 10:1 proposal negatives. Protocol, decision rules B1-B6 and leakage gates fixed before any 7b model (random forest dropped for runtime at STOP 2, before any CV run).
- **CV (training images):** selected **H-wc** (context kept: +0.022 [+0.014, +0.034]; calibration not kept). AP@50 **0.032** [0.025, 0.048] against 0.0011 for stage 1 alone and 0.0001 for Phase 7; recall 11% at IoU 0.5 against an 81% proposal ceiling. B6 passed (beats both, CIs excluding 0). No implant shortcut (AP@50 without hardware 0.0325).
- **Near-duplicates (new caveat):** 67 near-identical training image pairs (58 across folds, 34 fractured images involved) inflate the cascade's CV AP@50 (0.024 without them); stage 1 unaffected. On test, 26 of 817 images (5 of 176 boxes) have a near-duplicate in training.
- **Test (second use of the test set, post-hoc follow-up; lock `results/phase7b/TEST_EVALUATED.json`):** AP@50 **0.021** [0.008, 0.043], against Phase 7 **+0.021 [+0.008, +0.043]**; without near-duplicates 0.020 [0.007, 0.045], +0.020 [+0.007, +0.045]. Recall 7.4% at IoU 0.5 (17.6% at IoU 0.3), precision 6.5%. Hand carries it (10 of 13 hits); leg weaker than in CV; hip and shoulder not resolvable. Image level still clearly below Phase 3 (PR-AUC 0.38 against 0.63).
- **Answer:** a classical two-stage cascade gives a real, replicated localization gain over the Phase 7 detector, but it remains far from usable localization; Phase 3 stays the image-level model. Phase 7b is closed; the test set is not used again.

## Decisions to keep (details in `docs/data_decisions.md`)
- Images: 4,083 total, 719 fractured (the plan says 717; 719 is correct, see `docs/data_decisions.md`).
- Region groups: hand, leg (+leg-only), hip (+leg+hip), shoulder (+hand+shoulder), other (9 fractured images).
- Frozen split: `results/splits.csv` (80/20, stratified on region x fractured, 5 folds inside train). **Never regenerate it.**
- Image paths in `master.csv` are relative to the FracAtlas folder (`data/FracAtlas/` or `$FRACATLAS_DIR`).
- EXIF rotation is applied to every image. 21 normal images (20 hip, 1 shoulder) store unrotated width/height in the metadata; harmless (no boxes), and `check_images.py` reports them (80 problem files = 59 truncated + 21 rotated).
- Class imbalance: Phase 2 used `class_weight="balanced"`; Phase 3 compared strategies and chose none + F2-tuned threshold (class weights and SMOTE are equivalent after tuning).

## Known caveats (carry into the report)
- Image-level split: the dataset has no patient ID, so views of one patient may fall in both train and test (396 images carry the `multiscan` flag).
- Hip and shoulder have only about 46-48 fractured training images and 11-12 fractured test images; their metrics will have wide confidence intervals.
- Phase 2 scores tuned C and the feature set on the same folds and are slightly optimistic; Phase 3 CV numbers are nested (tuning inside the training folds). The Phase 3 test CIs hold the threshold fixed, so they ignore the uncertainty of choosing it.
- Reproduction across machines is ~1e-3, not exact (Phase 6 gate G1); exact reproductions only hold on one machine with the same `--n-jobs`. Use the repo's `.venv` (Python 3.11.9); on the Phase 6 machine the bare `python` is 3.14 without imbalanced-learn.
- Phase 6 shoulder loss on test (F2 -0.129 [-0.275, -0.034]) rests on 12 fractured images and was not significant in CV; report it as a single finding.
- Near-duplicate images: 67 near-identical pairs in training (58 across folds) and 26 test images with a near-duplicate in training (pixel correlation > 0.999; found in Phase 7b). They inflate the Phase 7b cascade's CV AP@50 by about a third and barely change its test result; earlier phases' CV numbers may carry a smaller inflation of the same kind.
- Phase 7b used the test set a second time (after Phase 7); its test numbers are a post-hoc follow-up.
- Surgical hardware: 97 of 99 hardware images are fractured. Phase 3 test recall is the same with and without hardware, so no sign of a shortcut, but only 21 test images have hardware.

## Open items (from the plan)
- [ ] Actual deadlines for remaining deliverables and the final presentation
- [ ] Ask the instructor whether pretrained CNN features used only as feature extractors are allowed (assume not)
- [x] FracAtlas metadata column names and treatment of multi-region / hardware images (Phase 1)
- [x] Patch sizes and window stride for Phase 7 (done: {16, 40, 64} px windows, `docs/phase7_localization.md`) (box-size percentiles are in `results/tables/eda_box_side_percentiles_256.csv`)
- [x] Which gap-closing strategies to prioritise (done in Phase 6; see the Phase 6 summary). Phase 5 had narrowed this to two, in order of cost: **per-region thresholds** on the single unified model (5.1: specialisation changes the threshold far more than the ranking), and a **shared-plus-region-specific kernel** aimed at hip (5.2: a region classifier reaches accuracy 0.94 from the features, macro-F1 0.765, but the unified model is not exploiting it). Both need validating in Phase 6 on their own terms.

## When a phase is finished (checklist)
1. Update the dashboard and the phase detail in this file (status, percentage, date, key numbers).
2. Write or update `docs/phaseN_<topic>.md` with the decisions, the final numbers and the reasons.
3. Add the commands to reproduce it to `README.md`.
4. Make sure every run is in `results/experiment_log.csv` and every table/figure is in `results/`.
5. Re-pin `requirements.txt` if libraries changed (`python src/freeze_requirements.py`).
6. Commit and push, then tell the team what changed and what the next phase can now use.
