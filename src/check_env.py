"""Phase 0 environment check. Run from the repo root:  python src/check_env.py"""
import importlib
import sys
from pathlib import Path

REQUIRED = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "imblearn": "imbalanced-learn",
}
OPTIONAL = {"xgboost": "xgboost", "lightgbm": "lightgbm"}
FOLDERS = ["data", "notebooks", "src", "features", "results", "docs"]


def check(modules, label):
    missing = []
    for mod, pkg in modules.items():
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {pkg:<18} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  MISSING  {pkg}")
            missing.append(pkg)
    return missing


def main():
    print(f"Python {sys.version.split()[0]}")
    print("\nRequired packages:")
    missing = check(REQUIRED, "required")
    print("\nOptional packages:")
    check(OPTIONAL, "optional")

    print("\nFolders:")
    root = Path(__file__).resolve().parent.parent
    for f in FOLDERS:
        print(f"  {'ok     ' if (root / f).is_dir() else 'MISSING'}  {f}/")

    log = root / "results" / "experiment_log.csv"
    print(f"\nExperiment log: {'ok' if log.exists() else 'MISSING'}")

    if missing:
        print("\nInstall missing packages with:  pip install -r requirements.txt")
        sys.exit(1)
    print("\nPhase 0 environment looks good.")


if __name__ == "__main__":
    main()
