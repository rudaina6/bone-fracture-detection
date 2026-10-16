# Phase 1: Data Decisions

Record of every data-handling decision made in Phase 1, with the evidence. Reuse in the report's dataset section.
Reproduce with `src/build_master.py`, `src/make_splits.py`, `src/check_images.py` and `notebooks/01_eda.ipynb`.

## 1. Dataset totals
- 4,083 images; **719 fractured** (17.6%). The plan said 717; the `Fractured` folder, `dataset.csv` and the COCO boxes all agree on 719, so 719 is used.
- Every fractured image has at least one box; no normal image has a box. No image sits in the wrong folder for its label.
- Two file names (`IMG0003375.jpg`, `IMG0003376.jpg`) exist in both folders; both are fractured, so the copy in `Fractured/` is used.

## 2. Region grouping
Single-region hip and shoulder images contain only 10 fractures each, too few to train or test on. The two large multi-region combinations are folded in:

| Group | Made of | Images | Fractured |
|---|---|---|---|
| hand | hand | 1,281 | 381 |
| leg | leg | 2,124 | 212 |
| hip | hip + leg+hip | 320 | 57 |
| shoulder | shoulder + hand+shoulder | 334 | 60 |
| other | 24 images with unusual combinations | 24 | 9 |

- Visual check (`results/figures/eda_merge_check.png`): `leg+hip` images are pelvis/femur views and `hand+shoulder` images are chest/shoulder views with the arm in frame, so the grouping is anatomically sensible.
- "other" stays in the unified model only. The exact combination is kept in `region_detail`.
- **Caveat for error analysis:** fracture rate inside the hip group differs a lot by sub-group (hip alone 6%, leg+hip 33%), and leg+hip includes paediatric and femur-only images. A model could learn the sub-group instead of the fracture. Report hip results by `region_detail` too.

## 3. Hardware (implants)
99 images contain hardware, and 97 of them are fractured (13.5% of all fractures). Decision: keep them, flag them, report results with and without them, and inspect them separately in error analysis (shortcut risk: "implant means fracture").

## 4. Multiple fractures
171 images have more than one box. Kept. The image-level label is unaffected and the localization phase uses every box.

## 5. Splits (frozen)
- `results/splits.csv`: stratified 80/20 train/test on region group x fractured, seed 42; 5 stratified CV folds inside the training set. Do not regenerate.
- The "other" fractured stratum had fewer than 10 images and was merged with the other fractured strata for stratification only.
- FracAtlas's own `Utilities/Fracture Split` was not used because it is not stratified by region group and the project needs a split built for the region comparison.
- Test fractured images: hand 76, leg 42, shoulder 12, hip 11, other 2. Hip and shoulder test sets are small, so report bootstrap confidence intervals.

## 6. Image integrity (`src/check_images.py`)
- **59 truncated JPEGs**, all normal images (IMG0004028 to IMG0004347), each missing 0 to 67 bytes at the end of the file. Decision: keep all; load with `ImageFile.LOAD_TRUNCATED_IMAGES = True` (Pillow). OpenCV tolerates them.
- **1 rotated image** (`IMG0002628.jpg`, fractured shoulder): the file has EXIF orientation tag 8, so its stored size is 373x454 but the COCO size and box are for the rotated view (454x373). Verified visually: after `ImageOps.exif_transpose` the box sits on the shoulder/upper arm. Decision: every image loader applies `ImageOps.exif_transpose` (no effect on images without the tag). OpenCV's `cv2.imread` does this by default.

## 7. Box sizes (training boxes, side = sqrt(w*h) in the 256x256 padded space)

| | all | hand | leg | hip | shoulder |
|---|---|---|---|---|---|
| p25 | 11.3 | 10.6 | 12.9 | 19.9 | 11.1 |
| p50 | 14.7 | 13.3 | 16.7 | 30.8 | 15.3 |
| p75 | 20.1 | 16.9 | 23.8 | 37.4 | 18.9 |

- Median box area is about 0.2 to 0.5% of the image (hip about 1.8%). Fractures are tiny relative to the whole image.
- Consequence for Phase 2: global features at 256x256 may dilute the signal. Compare 256 against about 448 (the native size of most images; see section 8) in the feature-family comparison.
- Consequence for Phase 7: at 256x256 the boxes are only 11 to 37 pixels wide, too small for 16x16-pixel HOG cells. The localization phase needs a higher working resolution, and hip needs larger patches than hand.

## 8. Image sizes
- 11 distinct sizes. 3,727 images (91%) are 373x454 (2,703) or 454x373 (1,024). About 350 are high resolution: 2304x2880 (303), 2880x2304 (23), 1760x2140 (24), roughly 6 MP against 0.17 MP for the rest. Six images are odd crops (e.g. 181x454).
- All aspect ratios are about 0.8 or 1.25, so padding to square adds about 20% border.
- Resizing to 256 shrinks the small images about 1.8x but the large ones 8 to 11x, so the detail kept differs by image source.
- Most images are only 454 px on the long side, so a 512 working size would mostly upsample. Test 256 against about 448.
- Checked: the 351 high-resolution images (> 1 MP) are spread evenly over regions (hand 7.9%, hip 6.6%, leg 9.2%, shoulder 9.6%, other 2 of 24), but they are fractured far more often: 29.1% vs 16.5% for the rest. They are 8.6% of the data and hold about 14% of the fractures.
- Risk: after resizing, large images are sharper and cleaner than small ones, and texture features (LBP, Gabor, fine HOG) respond to that, so a model could partly learn "crisp image, probably fractured".
- Decision: in Phase 2 first resize every image to the same intermediate long side (about 448) with area interpolation, then pad and resize to the working size. Report results with and without the high-resolution images, and review them as a separate group in the error analysis.

## 9. Intensity
- Median per-image mean intensity: hand 43, leg 44, hip 59, other 72, shoulder 90. Median contrast (std): hand 34, leg 41, hip 47, other 57, shoulder 68. The spread within each region is also wide.
- Decision: keep CLAHE; also compare a per-image z-score in Phase 2.
- Global intensity features identify the region, and fracture rates differ by region (hand 29.7%, leg 10.0%, hip 17.8%, shoulder 18.0%), so a pooled model can use them as a shortcut. Compare results with and without the intensity-statistics feature family, and tune per-region thresholds in Phase 6.

## Open items
- [ ] Confirm with the instructor that the hip/shoulder grouping is acceptable.
- [ ] Confirm with the instructor whether pretrained CNN features as feature extractors are allowed.
- [ ] Choose the image resolution(s) to test in Phase 2.