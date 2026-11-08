"""Phase 1: image integrity check. Run once from the repo root (after build_master.py):

    python src/check_images.py

Fully decodes every image listed in data/master.csv and reports files that are truncated,
unreadable, or whose real size differs from the width/height recorded in master.csv.

Output: results/tables/image_integrity.csv (only problem files) and a printed summary.
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from paths import DATA_DIR, REPO_ROOT, image_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=REPO_ROOT / "data" / "master.csv", type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "results" / "tables" / "image_integrity.csv", type=Path)
    ap.add_argument("--data-dir", default=DATA_DIR, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    problems = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        issue = None
        try:
            with Image.open(image_path(r.file_path, args.data_dir)) as im:
                im.load()                       # full decode; raises on truncated data
                size = ImageOps.exif_transpose(im).size   # COCO sizes/boxes assume the EXIF rotation is applied
                if size != (int(r.width), int(r.height)):
                    issue = f"size mismatch: file {size} vs master ({r.width}, {r.height})"
        except Exception as e:                  # noqa: BLE001
            issue = f"{type(e).__name__}: {e}"
        if issue:
            problems.append(dict(image_id=r.image_id, region_group=r.region_group,
                                 fractured=r.fractured, file_path=r.file_path, issue=issue))
        if i % 500 == 0:
            print(f"  checked {i}/{len(df)}")

    out = pd.DataFrame(problems)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n{len(out)} problem image(s) out of {len(df)}")
    if len(out):
        print(out[["image_id", "region_group", "fractured", "issue"]].to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()