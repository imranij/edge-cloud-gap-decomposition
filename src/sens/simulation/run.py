"""POST-PLAN: resumable runner, one CSV per scenario x replicate.
Post-plan, not pre-specified.

    OMP_NUM_THREADS=1 nohup nice -n 19 python src/sens/simulation/run.py --reps 0:100 \
        > logs/sens/sim_run.log 2>&1 &

One replicate = 4 independent synthetic datasets of the same scenario (seed of dataset k of replicate
r of scenario i: SeedSequence([20261012, i, r, k])). Replicates are run replicate-major (all
scenarios of replicate r before r + 1), so a partial run stays balanced across scenarios. A file
results_sens/simulation/reps/<scenario>_r<rep>.csv marks a finished scenario x replicate; existing
files are skipped.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generator import SCENARIOS, generate      # noqa: E402
from pipeline import run_sim_dataset          # noqa: E402

OUT = HERE.parents[2] / "results_sens" / "simulation"
BASE_SEED = 20261012
N_DATASETS = 4
SCEN_ORDER = ["S0", "S1", "S1b", "S2", "S3"]


def dataset_seed(scen: str, rep: int, k: int) -> int:
    return int(np.random.SeedSequence([BASE_SEED, SCEN_ORDER.index(scen), rep, k]).generate_state(1)[0])


def run_replicate(scen: str, rep: int, n_splits: int, null_draws: int) -> pd.DataFrame:
    rows = []
    for k in range(N_DATASETS):
        s = dataset_seed(scen, rep, k)
        ds = generate(scen, s)
        for r in run_sim_dataset(ds, s, n_splits=n_splits, null_draws=null_draws):
            rows.append(dict(scenario=scen, rep=rep, dataset=k, ds_seed=s, **r))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:1")
    ap.add_argument("--scenarios", default=",".join(SCEN_ORDER))
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--null-draws", type=int, default=1)
    a = ap.parse_args(argv)
    lo, hi = (int(v) for v in a.reps.split(":"))
    scen = a.scenarios.split(",")
    od = OUT / "reps"; od.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "run_log.jsonl", "a")
    for rep in range(lo, hi):
        for sc in scen:
            fn = od / f"{sc}_r{rep:03d}.csv"
            if fn.exists():
                continue
            t0, c0 = time.time(), time.process_time()
            df = run_replicate(sc, rep, a.n_splits, a.null_draws)
            tmp = fn.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False); os.replace(tmp, fn)
            rec = dict(scenario=sc, rep=rep, wall_s=round(time.time() - t0, 1),
                       cpu_s=round(time.process_time() - c0, 1), n_rows=len(df),
                       utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
    log.close()


if __name__ == "__main__":
    main()
