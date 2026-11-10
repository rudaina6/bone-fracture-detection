"""Append one row to the shared experiment log (results/experiment_log.csv). Writes the header if needed.

If the log cannot be written - on Windows it is locked while the file is open in Excel - the row goes to
results/experiment_log_pending.csv instead and a warning is printed, rather than raising. Logging must never
destroy the results of a run that has already finished: a Phase 5 learning-curve run is eleven minutes of
fitting, and losing it because a spreadsheet was open would be far worse than a log row arriving late.
Merge the pending file into the main log and delete it once the main log is closed.
"""
import csv
import datetime as dt
import time
import uuid

from paths import REPO_ROOT

LOG = REPO_ROOT / "results" / "experiment_log.csv"
PENDING = REPO_ROOT / "results" / "experiment_log_pending.csv"
COLUMNS = ["date", "run_id", "regions", "feature_set", "model", "imbalance_strategy",
           "key_hyperparameters", "cv_score", "test_metrics", "notes"]


def _append(path, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\r\n")
        if new:
            w.writeheader()
        w.writerow(row)


def log_run(**kw):
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown log columns: {unknown}")
    row = {c: "" for c in COLUMNS}
    row.update(date=dt.date.today().isoformat(), run_id=uuid.uuid4().hex[:8], **kw)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(3):
        try:
            _append(LOG, row)
            return row["run_id"]
        except OSError as e:          # PermissionError on Windows while the file is open elsewhere
            err = e
            if attempt < 2:
                time.sleep(1.0)
    try:
        _append(PENDING, row)
        where = f"written to {PENDING.name} instead"
    except OSError:
        where = "NOT saved - copy the row printed below into the log by hand"
    print(f"\n[!] could not write {LOG.name} ({type(err).__name__}: {err})."
          f"\n    The run's results and figures are saved; only the log row is affected - {where}."
          f"\n    If the log is open in Excel, close it and merge {PENDING.name} into it, then delete it."
          f"\n    row: {row}\n", flush=True)
    return row["run_id"]
