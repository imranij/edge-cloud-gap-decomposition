"""POST-PLAN: check that pipeline.run_audit_fast reproduces escalation.audit.run_audit exactly.

    OMP_NUM_THREADS=1 nice -n 19 python src/sens/simulation/verify_audit.py
Writes results_sens/simulation/verify_audit.csv (one row per cell, max |difference| of C, lower and
upper bound, and whether the GO calls agree) and exits 2 on any disagreement beyond 1e-12.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generator import generate                                  # noqa: E402
from pipeline import run_audit_fast, PLAN, RT, M, fit_tiers     # noqa: E402
from escalation.audit import run_audit                          # noqa: E402
from escalation.splits import make_splits                       # noqa: E402

OUT = HERE.parents[2] / "results_sens" / "simulation"


def main():
    rows = []
    for sc, seed in (("S0", 11), ("S1", 12), ("S2", 13), ("S3", 14)):
        ds = generate(sc, seed)
        y, groups, Xe = ds["y"], ds["groups"], ds["X_edge"]
        for sp in make_splits(y, groups, "tabular", PLAN)[:10]:
            sid = sp["split_id"]; fit, val, test = sp["fit"], sp["val"], sp["test"]
            T = fit_tiers(ds, fit, val, test, PLAN.model_seed + sid)
            (pe_v, _, tE), (pc_v, _, tC) = T["e"], T["c"]
            Z = RT.router_inputs(Xe[val], pe_v)
            tg = RT.router_targets(y[val], pe_v, pc_v, tE, tC)
            a = run_audit(Z, tg["g"], tg["w"], groups[val], sid, PLAN)
            b = run_audit_fast(Z, tg["g"], tg["w"], groups[val], sid, PLAN)
            rows.append(dict(scenario=sc, seed=seed, split=sid,
                             d_C=abs(a["audit_C_val"] - b["audit_C_val"]), d_lo=abs(a["audit_C_lo"] - b["audit_C_lo"]),
                             d_hi=abs(a["audit_C_hi"] - b["audit_C_hi"]), go_frozen=a["audit_go"], go_fast=b["audit_go"],
                             C_lo=a["audit_C_lo"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "verify_audit.csv", index=False)
    bad = (df[["d_C", "d_lo", "d_hi"]].max(axis=1) > 1e-12) | (df.go_frozen != df.go_fast)
    print(df.round(6).to_string())
    print(f"{len(df)} cells, max diff {df[['d_C', 'd_lo', 'd_hi']].to_numpy().max():.2e}, disagreements {int(bad.sum())}")
    if bad.any():
        sys.exit(2)


if __name__ == "__main__":
    main()
