"""Phase 1.1: build the master label table (one row per image) from FracAtlas.

Run from anywhere:
    python src/build_master.py --data-dir data/FracAtlas

The script searches --data-dir for the metadata CSV, the COCO annotation JSON and
the images, so the exact folder layout does not matter. It prints the CSV columns it
finds and runs sanity checks against the numbers in the project plan
(4,083 images, 717 fractured, ~17.6%). Check those printouts before trusting the output.

Output: data/master.csv  (file_path is relative to --data-dir, e.g. images/Fractured/IMG0000019.jpg)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from paths import REPO_ROOT

REGIONS = ["hand", "leg", "hip", "shoulder"]
IMG_EXT = {".jpg", ".jpeg", ".png"}

# Modelling groups. Hip and shoulder have only ~10 fractured images each as single-region
# images, so the two large multi-region combinations are folded into them:
#   leg+hip -> hip,  hand+shoulder -> shoulder.
# Everything else (24 images with unusual combinations) goes to "other": kept in the unified
# model but excluded from region-specific models. The exact combination is kept in region_detail.
GROUP_MAP = {
    "hand": "hand", "leg": "leg", "hip": "hip", "shoulder": "shoulder",
    "leg+hip": "hip", "hand+shoulder": "shoulder",
}


def find_metadata_csv(root):
    # FracAtlas ships its metadata as dataset.csv at the top level; Utilities/Fracture Split/*.csv
    # are the authors' train/valid/test lists and must not be mistaken for it.
    if (root / "dataset.csv").exists():
        return root / "dataset.csv"
    hits = [h for h in sorted(root.rglob("*.csv")) if "Utilities" not in h.parts]
    # otherwise take a CSV that has the region columns
    for h in hits:
        try:
            cols = {c.lower().strip() for c in pd.read_csv(h, nrows=1).columns}
        except Exception:
            continue
        if set(REGIONS) <= cols and "fractured" in cols:
            return h
    raise SystemExit(f"No metadata CSV with {REGIONS} + 'fractured' columns found under {root}")


def find_coco_json(root):
    for h in sorted(root.rglob("*.json")):
        try:
            with open(h) as f:
                d = json.load(f)
        except Exception:
            continue
        if isinstance(d, dict) and "images" in d and "annotations" in d:
            return h, d
    raise SystemExit(f"No COCO-style JSON (with 'images' and 'annotations') found under {root}")


def index_images(root):
    """Map file name -> list of paths (a name can exist in more than one folder)."""
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx.setdefault(p.name, []).append(p)
    dupes = sorted(n for n, ps in idx.items() if len(ps) > 1)
    if dupes:
        print(f"WARNING: {len(dupes)} image names exist in more than one folder: {dupes}")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "data" / "master.csv", type=Path)
    args = ap.parse_args()
    root = args.data_dir

    csv_path = find_metadata_csv(root)
    print(f"Metadata CSV : {csv_path}")
    meta = pd.read_csv(csv_path)
    print(f"CSV columns  : {list(meta.columns)}")
    print(meta.head(3).to_string(), "\n")

    cmap = {c.lower().strip(): c for c in meta.columns}
    id_col = cmap.get("image_id", meta.columns[0])
    df = pd.DataFrame({"image_id": meta[id_col].astype(str)})
    for r in REGIONS:
        df[r] = meta[cmap[r]].astype(int)
    df["fractured"] = meta[cmap["fractured"]].astype(int)
    for opt in ["mixed", "hardware", "multiscan", "fracture_count"]:
        df[opt] = meta[cmap[opt]] if opt in cmap else pd.NA

    # region label: exactly one region flag -> that region; several -> "multi"; none -> "none"
    n_flags = df[REGIONS].sum(axis=1)
    df["region"] = df[REGIONS].idxmax(axis=1).where(n_flags == 1, other="multi")
    df.loc[n_flags == 0, "region"] = "none"

    # exact region combination, and the modelling group used for splits / region-specific models
    df["region_detail"] = df[REGIONS].apply(
        lambda r: "+".join(c for c in REGIONS if r[c]) or "none", axis=1)
    df["region_group"] = df["region_detail"].map(GROUP_MAP).fillna("other")

    # file paths
    imgs = index_images(root)

    def pick_path(row):
        ps = imgs.get(row["image_id"])
        if not ps:
            return pd.NA
        if len(ps) == 1:
            return ps[0].relative_to(root).as_posix()
        # name exists in several folders: take the copy whose folder matches the label
        want = "Fractured" if row["fractured"] == 1 else "Non_fractured"
        match = [p for p in ps if p.parent.name == want]
        return (match or ps)[0].relative_to(root).as_posix()

    # file_path is stored RELATIVE to --data-dir (posix slashes) so master.csv works on every machine;
    # resolve it with src/paths.py::image_path().
    df["file_path"] = df.apply(pick_path, axis=1)
    df["in_both_folders"] = df["image_id"].map(lambda n: int(len(imgs.get(n, [])) > 1))
    n_missing = df["file_path"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} CSV rows have no matching image file, e.g. "
              f"{df.loc[df.file_path.isna(), 'image_id'].head(3).tolist()}")

    # bounding boxes from COCO (bbox = [x, y, w, h] in pixels)
    coco_path, coco = find_coco_json(root)
    print(f"COCO JSON    : {coco_path}")
    id2img = {im["id"]: im for im in coco["images"]}
    boxes = {}
    for a in coco["annotations"]:
        name = Path(id2img[a["image_id"]]["file_name"]).name
        boxes.setdefault(name, []).append([round(float(v), 1) for v in a["bbox"]])
    df["n_boxes"] = df["image_id"].map(lambda n: len(boxes.get(n, [])))
    df["boxes_xywh"] = df["image_id"].map(lambda n: json.dumps(boxes.get(n, [])))

    # image width / height (COCO where available, otherwise read the file)
    coco_wh = {Path(im["file_name"]).name: (im.get("width"), im.get("height")) for im in coco["images"]}

    def get_wh(row):
        w, h = coco_wh.get(row["image_id"], (None, None))
        if w and h:
            return pd.Series([w, h])
        if pd.notna(row["file_path"]):
            with Image.open(row["file_path"]) as im:
                return pd.Series(im.size)
        return pd.Series([pd.NA, pd.NA])

    df[["width", "height"]] = df.apply(get_wh, axis=1)

    # ---- sanity checks against the plan ----
    n, nf = len(df), int(df["fractured"].sum())
    print(f"\nTotal images : {n}  (plan: ~4,083)")
    print(f"Fractured    : {nf} = {nf / n:.1%}  (plan: 717, ~17.6%)")
    print("\nRegion x fractured:")
    print(pd.crosstab(df["region"], df["fractured"], margins=True))
    print("\nRegion group x fractured (used for splits and region-specific models):")
    print(pd.crosstab(df["region_group"], df["fractured"], margins=True))
    print(f"\nFractured images with 0 boxes : {int(((df.fractured == 1) & (df.n_boxes == 0)).sum())}")
    print(f"Non-fractured images with boxes: {int(((df.fractured == 0) & (df.n_boxes > 0)).sum())}")
    for flag in ["mixed", "hardware", "multiscan"]:
        if df[flag].notna().any():
            print(f"Images with {flag}=1: {int((df[flag] == 1).sum())}")
    print(f"Images with >1 fracture box   : {int((df.n_boxes > 1).sum())}")

    print("\nMulti-region images by region combination x fractured:")
    multi = df[df["region"] == "multi"].copy()
    multi["combo"] = multi[REGIONS].apply(lambda r: "+".join(c for c in REGIONS if r[c]), axis=1)
    print(pd.crosstab(multi["combo"], multi["fractured"], margins=True))

    print("\nHardware x fractured:")
    print(pd.crosstab(df["hardware"], df["fractured"], margins=True))

    folder = df["file_path"].map(lambda p: Path(p).parent.name if pd.notna(p) else "")
    wrong = int(((folder == "Fractured") != (df["fractured"] == 1)).sum())
    print(f"\nImages whose folder disagrees with the fractured label: {wrong}")
    print(f"Image names present in both folders: {int(df['in_both_folders'].sum())}")
    print(df.loc[df["in_both_folders"] == 1, ["image_id", "fractured", "n_boxes", "region"]].to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()