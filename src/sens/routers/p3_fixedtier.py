"""POST-PLAN: fixed-tier H2 variant.

The main-run tiers (q = 1 edge and cloud models, thresholds, test probabilities, g_dep) are frozen.
For q in 0.1, 0.25, 0.5 the validation subsample of the main H2 run is reproduced with
splits.h2_subsample (validation output only; checked against common.reconstruct_split(q)), the stored
q = 1 validation arrays are restricted to the kept windows (np.isin(val, val_kept)), the router
targets are recomputed on the kept windows, R6 is refitted (seed PLAN.model_seed + split) and scored
on the full q = 1 test fold. q = 1 is the stored R6 (refitted here and checked).
Writes results_sens/routers/p3_cells.parquet.

    OMP_NUM_THREADS=1 nice -n 10 python src/sens/routers/p3_fixedtier.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from util import CM, PLAN, OUT, DATASETS, load_cell
from escalation import metrics as M
from escalation import routers as RT
from escalation.splits import h2_subsample, groups_per_class


def main():
    recs = []
    for ds in DATASETS:
        D = CM.dataset(ds)
        y, groups, Xe = D["y"], D["groups"], D["X_edge"]
        for sid in CM.stored_split_ids(ds, "A", None):
            sp, arr, rows = load_cell(ds, "A", 1.0, sid)
            fit, val, test = sp["fit"], sp["val"], sp["test"]
            te_, tc_ = float(arr["t_ef_ev"]), float(arr["t_cf_cv"])
            Z_te = RT.router_inputs(Xe[test], arr["p_ef_ev"])
            g = arr["g_dep"]
            C1 = M.weighted_concordance(g, arr["score_R1"])
            seed = PLAN.model_seed + sid
            for q in (1.0, 0.1, 0.25, 0.5):
                if q == 1.0:
                    vk = val
                    check = "main"
                else:
                    _, vk = h2_subsample(y, groups, fit, val, q, sid, PLAN)
                    h2sp = CM.reconstruct_split(ds, "A", sid, q)
                    check = "match" if (h2sp is not None and np.array_equal(h2sp["val"], vk)) else \
                        ("h2_split_absent" if h2sp is None else "MISMATCH")
                pos = np.isin(val, vk)
                yk = y[val][pos]
                pe_v, pc_v = arr["pval_ef_ev"][pos], arr["pval_cf_cv"][pos]
                tg = RT.router_targets(yk, pe_v, pc_v, te_, tc_)
                Z = RT.router_inputs(Xe[val][pos], pe_v)
                s = RT.fit_regressor_router(Z, tg["g"], seed, PLAN, sample_weight=tg["w"]).predict(Z_te)
                C6 = M.weighted_concordance(g, s)
                if q == 1.0:
                    assert abs(C6 - M.weighted_concordance(g, arr["score_R6"])) < 1e-9, (ds, sid)
                recs.append(dict(dataset=ds, split=sid, q=q, check=check, n_val_kept=int(pos.sum()),
                                 pos_win_val_kept=int(yk.sum()),
                                 pos_groups_val_kept=groups_per_class(yk, groups[val][pos])[0],
                                 nonzero_g_kept=int((tg["g"] != 0).sum()), C_R6=C6, C_R1=C1, dC=C6 - C1,
                                 capture_R6=M.capture(g, s, 0.05, 0.30), n_train=len(fit) + len(val),
                                 n_test=len(test)))
            print(ds, sid, flush=True)
    pd.DataFrame(recs).to_parquet(OUT / "p3_cells.parquet", index=False)


if __name__ == "__main__":
    main()
