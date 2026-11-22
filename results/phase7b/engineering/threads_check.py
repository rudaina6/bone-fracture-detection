"""Phase 7b engineering check (no protocol change, training images only): why are stage-2 fits slow, and which thread
setting is fastest and safe? One HGB and one RF configuration (both in the fallback grid) on the same full-size
30k-row training set, timed (a) in a fresh process, (b) inside the pipeline as now, (c) inside the pipeline with thread
limits, including 4 concurrent fits through joblib (loky) with 1 or 2 OpenMP threads each. CPU usage is sampled
during every fit (system-wide and this process tree). Predictions are compared across settings.

    python results/phase7b/engineering/threads_check.py          # writes threads_check.json next to itself
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
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
SET = REPO / "features" / "phase7b" / "engineering_set.npz"
CFG = {"hgb": {"max_leaf_nodes": 63, "min_samples_leaf": 20}, "rf": {"max_features": "sqrt", "min_samples_leaf": 1}}
N_PRED = 20000


class CPU:
    """Samples system-wide CPU % and this process tree's CPU % (in units of logical cores x 100) once a second."""

    def __enter__(self):
        self.sys, self.own, self.stop = [], [], False
        me = psutil.Process()
        procs = {}

        def run():
            psutil.cpu_percent(None)
            while not self.stop:
                time.sleep(1.0)
                self.sys.append(psutil.cpu_percent(None))
                tot = 0.0
                for p in [me] + me.children(recursive=True):
                    if p.pid not in procs:
                        procs[p.pid] = p
                        p.cpu_percent(None)
                        continue
                    try:
                        tot += procs[p.pid].cpu_percent(None)
                    except psutil.Error:
                        pass
                self.own.append(tot)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()

    def summary(self):
        n = psutil.cpu_count()
        s = float(np.mean(self.sys)) if self.sys else np.nan
        o = float(np.mean(self.own)) / n if self.own else np.nan                 # as % of the whole machine
        return dict(system_cpu_pct=s, own_cpu_pct=o, others_cpu_pct=s - o)


def fit(clf, X, y, n_jobs=4):
    import phase7b_cascade as C
    return C.fit_clf(clf, CFG[clf], X, y, 42, n_jobs)


def _worker(clf, X, y, Xv, threads, rf_jobs):
    with threadpool_limits(threads, user_api="openmp"):
        t = time.time()
        m = fit(clf, X, y, rf_jobs)
        return time.time() - t, m.predict_proba(Xv)[:, 1], threadpool_info()


def fresh(clf):
    z = np.load(SET)
    t = time.time()
    m = fit(clf, z["X"], z["y"])
    out = dict(seconds=time.time() - t, threads=[(d["internal_api"], d["num_threads"]) for d in threadpool_info()])
    np.save(HERE / f"_fresh_{clf}.npy", m.predict_proba(z["Xv"][:N_PRED])[:, 1])
    print(json.dumps(out))


def main():
    import phase7_detect as P
    import phase7b_cascade as C
    import argparse
    args = argparse.Namespace(n_jobs=4, seed=42, cache_dir=REPO / "features" / "phase7")
    T = P.train_table()
    R0 = P.load_round0(args.cache_dir, T)
    X, y, Xv, _, _ = C.timing_set(T, R0, args)            # run_parallel inside: loky workers alive, as in the CV
    Xv = Xv[:N_PRED]
    SET.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SET, X=X, y=y, Xv=Xv)
    res = dict(rows=int(len(y)), features=int(X.shape[1]), logical_cores=psutil.cpu_count(),
               physical_cores=psutil.cpu_count(logical=False), cfg=CFG,
               main_threadpools=[(d["internal_api"], d["num_threads"]) for d in threadpool_info()], runs={})
    preds = {}

    def record(name, fn):
        with CPU() as c:
            t = time.time()
            out = fn()
            wall = time.time() - t
        r = dict(wall_s=wall, **c.summary())
        if isinstance(out, dict):
            r.update(out)
        res["runs"][name] = r
        print(f"  {name}: {wall:.1f} s wall; " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                                        for k, v in r.items() if k != "wall_s"), flush=True)

    for clf in ("hgb", "rf"):
        # (a) fresh process
        def run_fresh():
            o = subprocess.run([sys.executable, __file__, "fresh", clf], capture_output=True, text=True, check=True)
            preds[f"{clf}_fresh"] = np.load(HERE / f"_fresh_{clf}.npy")
            (HERE / f"_fresh_{clf}.npy").unlink()
            return dict(fit_s=json.loads(o.stdout.strip().splitlines()[-1])["seconds"])
        record(f"{clf}: (a) fresh process", run_fresh)

        # (b) inside the pipeline, as now (HGB: all OpenMP threads; RF: n_jobs=4)
        def run_now():
            m = fit(clf, X, y, 4)
            preds[f"{clf}_now"] = m.predict_proba(Xv)[:, 1]
        record(f"{clf}: (b) in pipeline, as now", run_now)

        # (c) inside the pipeline with thread limits, one fit at a time
        for th in ((1, 2, 4) if clf == "hgb" else (1, 2, 8)):
            def run_lim(th=th):
                if clf == "hgb":
                    with threadpool_limits(th, user_api="openmp"):
                        m = fit(clf, X, y)
                else:
                    m = fit(clf, X, y, th)
                preds[f"{clf}_lim{th}"] = m.predict_proba(Xv)[:, 1]
            record(f"{clf}: (c) in pipeline, {'OpenMP threads' if clf == 'hgb' else 'n_jobs'}={th}", run_lim)

        # (c) 4 concurrent fits through joblib (loky), 1 or 2 threads each (workers x threads vs 8 logical cores)
        for th in (1, 2):
            def run_conc(th=th):
                out = Parallel(n_jobs=4, backend="loky", inner_max_num_threads=th)(
                    delayed(_worker)(clf, X, y, Xv, th, th) for _ in range(4))
                preds[f"{clf}_conc{th}"] = out[0][1]
                same = max(float(np.abs(o[1] - out[0][1]).max()) for o in out)
                return dict(per_fit_s=[round(o[0], 1) for o in out], effective_s_per_fit=None,
                            worker_threadpools=[(d["internal_api"], d["num_threads"]) for d in out[0][2]],
                            workers_x_threads=4 * th, max_diff_between_workers=same)
            record(f"{clf}: (c) 4 concurrent fits, {th} thread(s) each", run_conc)
            r = res["runs"][f"{clf}: (c) 4 concurrent fits, {th} thread(s) each"]
            r["effective_s_per_fit"] = r["wall_s"] / 4

    ref = {c: preds[f"{c}_now"] for c in ("hgb", "rf")}
    res["max_abs_pred_diff_vs_as_now"] = {k: float(np.abs(v - ref[k.split("_")[0]]).max()) for k, v in preds.items()}
    (HERE / "threads_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["max_abs_pred_diff_vs_as_now"], indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "fresh":
        fresh(sys.argv[2])
    else:
        main()
