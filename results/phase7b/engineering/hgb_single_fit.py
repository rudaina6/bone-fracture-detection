"""Phase 7b engineering: ONE HistGradientBoosting fit (fallback config 1: max_leaf_nodes 15, min_samples_leaf 20;
window + context features) on the full-size 30k-row timing set, single process, 4 OpenMP threads, while sampling the
actual CPU performance / frequency (Windows performance counters), CPU use, and the top competing processes.

    python results/phase7b/engineering/hgb_single_fit.py      # writes hgb_single_fit.json next to itself
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
SET = REPO / "features" / "phase7b" / "engineering_set.npz"
CFG = dict(max_leaf_nodes=15, min_samples_leaf=20)
COUNTERS = [r"\Processor Information(_Total)\% Processor Performance",
            r"\Processor Information(_Total)\Processor Frequency",
            r"\Processor Information(_Total)\% Processor Time"]


def sample_counters(samples, stop):
    while not stop.is_set():
        try:
            out = subprocess.run(["typeperf"] + COUNTERS + ["-sc", "1"], capture_output=True, text=True, timeout=20).stdout
            row = [l for l in out.splitlines() if l.startswith('"') and "/" in l.split(",")[0]][-1]
            vals = [float(v.strip('"')) for v in row.split(",")[1:]]
            others = sorted(((p.info["cpu_percent"] or 0, p.info["name"]) for p in psutil.process_iter(["name", "cpu_percent"])
                             if p.pid != psutil.Process().pid), reverse=True)[:4]
            samples.append(dict(t=time.time(), perf_pct=vals[0], freq_mhz=vals[1], cpu_time_pct=vals[2],
                                top_procs=[(n, round(c, 1)) for c, n in others]))
        except Exception as e:                                  # a missed sample is not fatal
            samples.append(dict(t=time.time(), error=repr(e)))
        stop.wait(2.0)


def main():
    import phase7b_cascade as C
    z = np.load(SET)
    X, y = z["X"], z["y"]
    for p in psutil.process_iter(["name"]):
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    idle = []
    stop = threading.Event()
    th = threading.Thread(target=sample_counters, args=(idle, stop), daemon=True)
    th.start()
    time.sleep(12)                                              # idle baseline
    stop.set()
    th.join()
    load, stop = [], threading.Event()
    th = threading.Thread(target=sample_counters, args=(load, stop), daemon=True)
    th.start()
    t0 = time.time()
    with threadpool_limits(4, user_api="openmp"):
        m = C.make_clf("hgb", CFG, 42, 1)
        m.fit(X, y, sample_weight=C.balanced_weights(y))
        threads = [(d["internal_api"], d["num_threads"]) for d in threadpool_info()]
    fit_s = time.time() - t0
    stop.set()
    th.join()

    def summ(s):
        v = [x for x in s if "perf_pct" in x]
        return dict(n=len(v), perf_pct_mean=float(np.mean([x["perf_pct"] for x in v])) if v else None,
                    perf_pct_min=float(np.min([x["perf_pct"] for x in v])) if v else None,
                    freq_mhz_mean=float(np.mean([x["freq_mhz"] for x in v])) if v else None,
                    freq_mhz_min=float(np.min([x["freq_mhz"] for x in v])) if v else None,
                    cpu_time_pct_mean=float(np.mean([x["cpu_time_pct"] for x in v])) if v else None)
    res = dict(cfg=CFG, rows=int(len(y)), features=int(X.shape[1]), positives=int(y.sum()), fit_s=fit_s,
               n_iter=int(m.n_iter_), max_iter=300, early_stopping=bool(m.early_stopping), n_bins=int(m.max_bins + 1),
               threads=threads, cpu_freq_psutil=psutil.cpu_freq()._asdict(), idle=summ(idle), during_fit=summ(load),
               samples_during_fit=load, power_plan=subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                                                   text=True).stdout.strip())
    (HERE / "hgb_single_fit.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "samples_during_fit"}, indent=1, default=str))


if __name__ == "__main__":
    main()
