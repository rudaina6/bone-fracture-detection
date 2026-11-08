"""Append one row to the shared experiment log (results/experiment_log.csv). Writes the header if needed."""
import csv
import datetime as dt
import uuid

from paths import REPO_ROOT

LOG = REPO_ROOT / "results" / "experiment_log.csv"
COLUMNS = ["date", "run_id", "regions", "feature_set", "model", "imbalance_strategy",
           "key_hyperparameters", "cv_score", "test_metrics", "notes"]


def log_run(**kw):
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown log columns: {unknown}")
    row = {c: "" for c in COLUMNS}
    row.update(date=dt.date.today().isoformat(), run_id=uuid.uuid4().hex[:8], **kw)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists() or LOG.stat().st_size == 0
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\r\n")
        if new:
            w.writeheader()
        w.writerow(row)
    return row["run_id"]
