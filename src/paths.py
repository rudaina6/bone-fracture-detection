"""Where the FracAtlas files live, so no absolute path is ever stored in a CSV.

`data/master.csv` stores `file_path` relative to the FracAtlas folder,
e.g. `images/Fractured/IMG0000019.jpg`. Resolve it with `image_path()`.

The FracAtlas folder defaults to `<repo>/data/FracAtlas`. To keep it elsewhere, set the
environment variable FRACATLAS_DIR (Windows: `setx FRACATLAS_DIR "D:\\datasets\\FracAtlas"`,
then restart the terminal / Jupyter).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FRACATLAS_DIR", REPO_ROOT / "data" / "FracAtlas"))


def image_path(rel_path, data_dir=None):
    """Absolute path of an image from the `file_path` column of master.csv."""
    return Path(data_dir or DATA_DIR) / rel_path
