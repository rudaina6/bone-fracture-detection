"""Phase 2.1: image loading and preprocessing (identical for every image).

Pipeline (see docs/data_decisions.md sections 6, 8, 9):
    load (EXIF rotation applied, truncated JPEGs tolerated) -> grayscale
    -> resize to a common long side `mid` (area interpolation; removes the sharpness gap between
       the ~6 MP and the ~0.17 MP images) -> pad to a square -> resize to `size`
    -> contrast normalisation (`norm`) -> float32 in [0, 1]

`preprocess` also returns the geometry (scale + offsets) that maps ORIGINAL pixel coordinates to
the preprocessed image, so Phase 7 can place ground-truth boxes in the same space:
    x_out = x_in * scale + off_x      y_out = y_in * scale + off_y
"""
import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from paths import image_path

ImageFile.LOAD_TRUNCATED_IMAGES = True          # 59 normal images are cut off by up to 67 bytes

NORMS = ("clahe", "zscore", "none")


def load_gray(rel_path, data_dir=None):
    """Load an image as uint8 grayscale, applying the EXIF rotation tag (COCO boxes assume it)."""
    with Image.open(image_path(rel_path, data_dir)) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("L"))


def preprocess(gray, size=256, mid=448, norm="clahe", clip=2.0, tile=8):
    """gray: uint8 2-D array. Returns (img float32 in [0,1] of shape (size, size), geometry dict).

    mid  : long side of the intermediate resize; 0 or None skips it.
    norm : "clahe"  CLAHE (clip limit `clip`, `tile` x `tile` grid), applied after padding/resizing
           "zscore" per-image z-score over the image content (not the padding), clipped to +-3 sd
           "none"   no contrast normalisation
    """
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}")
    h0, w0 = gray.shape
    img, s1 = gray, 1.0
    if mid:
        s1 = mid / max(h0, w0)
        interp = cv2.INTER_AREA if s1 < 1 else cv2.INTER_LINEAR
        img = cv2.resize(gray, (round(w0 * s1), round(h0 * s1)), interpolation=interp)

    h1, w1 = img.shape
    side = max(h1, w1)
    py, px = (side - h1) // 2, (side - w1) // 2          # pad with black, image centred
    sq = np.zeros((side, side), np.uint8)
    sq[py:py + h1, px:px + w1] = img
    s2 = size / side
    out = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)

    scale = s1 * s2                                       # original px -> output px
    geom = dict(scale=scale, off_x=px * s2, off_y=py * s2,
                content=(round(px * s2), round(py * s2), round((px + w1) * s2), round((py + h1) * s2)))

    if norm == "clahe":
        out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(out)
        return out.astype(np.float32) / 255.0, geom
    out = out.astype(np.float32)
    if norm == "zscore":
        x0, y0, x1, y1 = geom["content"]
        region = out[y0:y1, x0:x1]
        z = (out - region.mean()) / (region.std() + 1e-6)
        return (np.clip(z, -3, 3) + 3) / 6, geom          # map +-3 sd to [0, 1]
    return out / 255.0, geom
