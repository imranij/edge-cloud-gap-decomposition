"""POST-PLAN: redesigned level and session hold-outs.

Outer split: leave one unit out (SKAB: recording session, which coincides with a disturbance family;
hydraulic: cooler level or accumulator level). Fit + validation = ALL other units. The validation part is
carved out of them by StratifiedGroupKFold over the ORIGINAL groups (SKAB files / anomaly-free chunks;
hydraulic runs or run chunks), K_in = min(4, positive groups, negative groups), shuffle, seed 2000 + fold,
first inner fold = validation, exactly as escalation.splits.inner_split does for the main run. Each fold is
then run through the unchanged run_dataset.run_split (same models, thresholds, routers, audit), and written
in the same format to results_sens/holdout2/<design>/<pair>/.

    python src/sens/designs/run_custom_holdout.py <design> <pair>
design in {skab_lso2, hyd_cooler2, hyd_accum2}
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import run_dataset as RD                                         # noqa: E402
from escalation.config import PLAN                               # noqa: E402
from escalation.data import load_dataset                         # noqa: E402
from escalation.splits import inner_split, check_split           # noqa: E402

DESIGNS = {"skab_lso2": ("skab", "session"), "hyd_cooler2": ("hydraulic", "cooler"),
           "hyd_accum2": ("hydraulic", "accumulator"),
           # the same session hold-out on the causal-offset features
           "skab_lso2_base": ("skab_base", "session"),
           # the cooler hold-out with the cloud view = edge sensors + FS1
           "hyd_cooler2_fs1": ("hyd_e_fs1", "cooler")}
ROOTS = {"skab_base": "data/processed_sens", "hyd_e_fs1": "data/processed_sens"}


def main():
    design, pair = sys.argv[1], sys.argv[2]
    parent, unit_col = DESIGNS[design]
    ds = load_dataset(parent, ROOT / ROOTS.get(parent, "data/processed"))
    ds["name"] = design
    y, groups = ds["y"], ds["groups"]
    units = ds["meta"][unit_col].astype(str).to_numpy()
    levels = sorted(np.unique(units))
    od = ROOT / "results_sens" / "holdout2" / design / pair
    od.mkdir(parents=True, exist_ok=True)
    log = open(od / "log.jsonl", "a")
    for j, u in enumerate(levels):
        fn = od / f"split_{j}.parquet"
        if fn.exists():
            continue
        test = np.flatnonzero(units == u)
        train = np.flatnonzero(units != u)
        fit, val, K_in, reason = inner_split(y, groups, train, j, PLAN)
        if reason is None:
            reason = check_split(y, groups, fit, val, test)
        sp = dict(split_id=j, repeat=0, fold=j, K=len(levels), K_in=K_in, fit=fit, val=val, test=np.sort(test),
                  skip=reason)
        t0 = time.time()
        if reason:
            rec = dict(split=j, held_out=u, skipped=reason)
        else:
            try:
                rows, arrays, info = RD.run_split(ds, pair, sp, PLAN)
            except Exception as e:                                      # e.g. too few fit groups for R8
                rec = dict(split=j, held_out=u, error=repr(e)[:300])
                log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
                continue
            df = pd.DataFrame(rows)
            df["held_out"] = u
            RD._atomic_npz(od / f"arrays_{j}.npz", **arrays)
            RD._atomic_parquet(df, fn)
            rec = dict(split=j, held_out=u, seconds=round(time.time() - t0, 1), n_fit=len(fit), n_val=len(val),
                       n_test=len(test), K_in=K_in,
                       pos_groups_fit=int(len(np.unique(groups[fit][y[fit] == 1]))))
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    log.close()


if __name__ == "__main__":
    main()
