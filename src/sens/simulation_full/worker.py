"""POST-PLAN: runner of the full validation (one process pool per call; resumable; deterministic).
Post-plan, not pre-specified. R per scenario is FIXED here and hashed before the run.

    python src/sens/simulation_full/worker.py --task 0 --ntasks 1 --procs 64 --out results_sens/simulation_full
    python src/sens/simulation_full/worker.py --list                  # unit counts and estimated CPU cost

A unit is one scenario x replicate (4 independent synthetic datasets; seed of dataset k of replicate r of scenario i:
SeedSequence([20261013, i, r, k])) or one chunk of 10 truth draws. Units are sorted by estimated cost and dealt
round-robin to the --ntasks tasks of a job array; a finished unit is a file (written atomically) and is skipped
on restart, so tasks can be resubmitted freely and results do not depend on how units are distributed.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch_stub                                                                    # noqa: E402
TORCH_STUB = torch_stub.ensure()
from gen_full import generate, BASE_SCENARIOS, T_SCENARIOS, C_SCENARIOS, family   # noqa: E402

BASE_SEED = 20261013
N_DATASETS = 4
ALL_ORDER = list(BASE_SCENARIOS) + list(T_SCENARIOS) + list(C_SCENARIOS)
R_FIXED = {"S0": 1000, "S1": 1000, "S1b": 500, "S2": 500, "S3": 500,
           "T0_0.0": 1000, "T0_0.3": 1000, "T0_0.6": 1000, "T0_0.9": 1000, "T0s_0.9": 1000,
           "T1_0.0": 500, "T1_0.3": 500, "T1_0.6": 500, "T1_0.9": 500,
           "C_clock": 1000, "C_offset": 1000, "C_ref": 500}
TRUTH_DRAWS = 200
TRUTH_CHUNK = 10
# CPU seconds per unit, measured in the local verification (plan/postplan/simulation_full.md); used only to order units
COST = {"base": 90.0, "T": 2.5, "C": 5.5, "truth_base": 17.0, "truth_T": 2.0, "truth_C": 4.0}
DEFAULT_OUT = HERE.parents[2] / "results_sens" / "simulation_full"


def dataset_seed(scen: str, rep: int, k: int) -> int:
    return int(np.random.SeedSequence([BASE_SEED, ALL_ORDER.index(scen), rep, k]).generate_state(1)[0])


def all_units() -> list[tuple[str, str, int, float]]:
    """(kind, scenario, index, cost) for every unit, most expensive first (stable)."""
    u = []
    for sc in ALL_ORDER:
        fam = family(sc)
        u += [("rep", sc, r, COST[fam]) for r in range(R_FIXED[sc])]
        u += [("truth", sc, c, COST["truth_" + fam]) for c in range(TRUTH_DRAWS // TRUTH_CHUNK)]
    order = sorted(range(len(u)), key=lambda i: (-u[i][3], i))
    return [u[i] for i in order]


def unit_path(out: Path, kind: str, sc: str, idx: int) -> Path:
    return out / ("reps" if kind == "rep" else "truth") / (f"{sc}_r{idx:04d}.csv" if kind == "rep" else f"{sc}_c{idx:02d}.csv")


def run_unit(args) -> dict:
    kind, sc, idx, out = args
    fn = unit_path(Path(out), kind, sc, idx)
    if fn.exists():
        return dict(kind=kind, scenario=sc, idx=idx, skipped=True)
    t0, c0 = time.time(), time.process_time()
    try:
        if kind == "rep":
            from pipe_full import run_dataset
            rows = []
            for k in range(N_DATASETS):
                s = dataset_seed(sc, idx, k)
                ds = generate(sc, s)
                rows += [dict(scenario=sc, rep=idx, dataset=k, ds_seed=s, **r) for r in run_dataset(ds, s)]
        else:
            from truth_full import one_draw
            rows = [one_draw(sc, d) for d in range(idx * TRUTH_CHUNK, (idx + 1) * TRUTH_CHUNK)]
        df = pd.DataFrame(rows)
        fn.parent.mkdir(parents=True, exist_ok=True)
        tmp = fn.with_name(fn.name + f".tmp{os.getpid()}")
        df.to_csv(tmp, index=False)
        os.replace(tmp, fn)
        return dict(kind=kind, scenario=sc, idx=idx, wall_s=round(time.time() - t0, 2),
                    cpu_s=round(time.process_time() - c0, 2), n_rows=len(df))
    except Exception as e:                                   # record and continue; the unit stays missing
        return dict(kind=kind, scenario=sc, idx=idx, error=repr(e), tb=traceback.format_exc()[-2000:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", default="", help="comma-separated scenarios (verification runs)")
    ap.add_argument("--max-index", type=int, default=-1, help="only units with index < this (verification runs)")
    ap.add_argument("--kinds", default="rep,truth")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="process this task's units last-first (several machines "
                    "sharing one output folder meet in the middle; a unit is re-checked just before it runs)")
    a = ap.parse_args(argv)
    units = all_units()
    if a.list:
        df = pd.DataFrame(units, columns=["kind", "scenario", "idx", "cost"])
        s = df.groupby(["scenario", "kind"]).agg(n=("idx", "size"), cpu_h=("cost", lambda c: c.sum() / 3600))
        print(s.to_string()); print("total CPU-hours (estimate):", round(df.cost.sum() / 3600, 1)); return
    kinds = set(a.kinds.split(","))
    only = set(a.only.split(",")) if a.only else None
    mine = [u for i, u in enumerate(units) if i % a.ntasks == a.task and u[0] in kinds
            and (only is None or u[1] in only) and (a.max_index < 0 or u[2] < a.max_index)]
    out = Path(a.out); (out / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(k, sc, i, str(out)) for k, sc, i, _ in mine if not unit_path(out, k, sc, i).exists()]
    if a.reverse:
        todo = todo[::-1]
    log = open(out / "logs" / f"task_{a.task:03d}_of_{a.ntasks:03d}.jsonl", "a")
    head = dict(event="start", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node(),
                python=platform.python_version(), procs=a.procs, n_mine=len(mine), n_todo=len(todo), torch_stub=TORCH_STUB)
    try:
        import sklearn, scipy
        head.update(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
    except Exception:
        pass
    log.write(json.dumps(head) + "\n"); log.flush(); print(json.dumps(head), flush=True)
    t0 = time.time()
    if TORCH_STUB:        # import once before forking; NOT with a real PyTorch loaded (its OpenMP runtime made
        import pipe_full, truth_full   # noqa: F401  forked HGB fits spin, observed 2026-10-13 on the cloud machine)
    if a.procs <= 1:
        it = map(run_unit, todo)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")   # spawn on Windows
        pool = ctx.Pool(a.procs, maxtasksperchild=50)
        it = pool.imap_unordered(run_unit, todo, chunksize=1)
    n = 0
    for rec in it:
        n += 1
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log.write(json.dumps(rec) + "\n"); log.flush()
        if "error" in rec or n % 25 == 0 or n == len(todo):
            print(f"[{n}/{len(todo)} {time.time() - t0:.0f}s] {json.dumps({k: v for k, v in rec.items() if k != 'tb'})}",
                  flush=True)
    log.write(json.dumps(dict(event="end", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              wall_s=round(time.time() - t0, 1), n_done=n)) + "\n")
    log.close()


if __name__ == "__main__":
    main()
