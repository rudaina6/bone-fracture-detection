"""Pin requirements.txt to the versions installed in YOUR environment, so everyone on the team
runs the same library versions.  Run once, then commit requirements.txt:

    python src/freeze_requirements.py

Keeps the package list from requirements.txt; packages that are not installed stay unpinned
(optional ones such as xgboost / lightgbm) and are reported.
"""
import re
from importlib import metadata

from paths import REPO_ROOT

REQ = REPO_ROOT / "requirements.txt"


def main():
    out, missing = [], []
    for line in REQ.read_text(encoding="utf-8").splitlines():
        name = re.split(r"[<>=!~ ]", line.strip(), maxsplit=1)[0]
        if not name or line.lstrip().startswith("#"):
            out.append(line)
            continue
        try:
            out.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            out.append(name)
            missing.append(name)
    REQ.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    if missing:
        print(f"\nNot installed (left unpinned): {', '.join(missing)}")


if __name__ == "__main__":
    main()
