"""POST-PLAN, evaluation only. Refits no model.

Realised escalation rate per unit (hydraulic run or run chunk, SKAB file or chunk, SEU recording,
3W episode) under the causal streaming thresholds of the cross-fitted router run: the threshold is
the (1 - f) quantile of the out-of-fold training scores, applied window by window to the test fold
(src/sens/routers/p2_crossfit.py). The escalation decisions are reconstructed exactly from stored
arrays: R1's training scores are r1_confidence(pE, t_ef_ev) with the out-of-fold edge probabilities
pE in results_sens/routers/p2_arrays/<ds>_<pair>_<j>.npz and the stored deployed threshold; R6x's are
the stored out-of-fold scores r6_oof; the test scores are stored in the same file. The test units come
from results/<ds>/<pair>/arrays_<j>.npz (groups_test). The reconstructed per-fold fraction is checked
against results_sens/routers/p2_stream.parquet.

Outputs (results/extra/):
  per_unit_escalation_units.csv  one row per dataset x pair x router x split x test unit
  per_unit_escalation.csv        per dataset x pair x router at f = 0.10: per-fold range, and over all
                                 (split, unit) cells the median, 90th and 95th percentile and maximum of
                                 the per-unit rate, also separately for units with and without faults,
                                 plus the provisioning factor (95th percentile / nominal f)
    python src/extra/per_unit_escalation.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X
from escalation.routers import r1_confidence

P2 = X.ROOT / "results_sens" / "routers" / "p2_arrays"
F = 0.10


def main():
    st = pd.read_parquet(X.ROOT / "results_sens" / "routers" / "p2_stream.parquet")
    rows = []
    for ds in X.DATASETS:
        for pair in ("A", "B"):
            for j in range(20):
                f2 = P2 / f"{ds}_{pair}_{j}.npz"
                if not f2.exists():
                    continue
                z = X.arrays(ds, pair, j)
                p = np.load(f2)
                te = float(z["t_ef_ev"])
                train = {"R1": r1_confidence(p["pE"], te), "R6x": p["r6_oof"]}
                test = {"R1": p["score_R1"], "R6x": p["score_R6x"]}
                units = z["groups_test"]; y = z["y_test"].astype(int)
                for r in ("R1", "R6x"):
                    thr = float(np.quantile(train[r], 1 - F))
                    esc = test[r] >= thr
                    ref = st[(st.dataset == ds) & (st.pair == pair) & (st.split == j) & (st.router == r)
                             & np.isclose(st.f, F)].realised_frac.iloc[0]
                    assert abs(esc.mean() - ref) < 1e-12, (ds, pair, j, r, esc.mean(), ref)
                    for u in np.unique(units):
                        ix = units == u
                        rows.append(dict(dataset=ds, pair=pair, router=r, split=j, unit=str(u), n_windows=int(ix.sum()),
                                         unit_has_fault=bool(y[ix].any()), unit_fault_share=float(y[ix].mean()),
                                         rate=float(esc[ix].mean()), fold_rate=float(esc.mean())))
    u = pd.DataFrame(rows)
    u.to_csv(X.OUT / "per_unit_escalation_units.csv", index=False)
    out = []
    for (ds, pair, r), g in u.groupby(["dataset", "pair", "router"]):
        folds = g.drop_duplicates("split").fold_rate
        rec = dict(dataset=ds, pair=pair, router=r, f=F, J=g.split.nunique(), n_unit_cells=len(g),
                   fold_min=folds.min(), fold_mean=folds.mean(), fold_max=folds.max(),
                   n_folds_zero=int((folds == 0).sum()),
                   unit_median=g.rate.median(), unit_q90=g.rate.quantile(0.90), unit_q95=g.rate.quantile(0.95),
                   unit_max=g.rate.max(), share_units_zero=float((g.rate == 0).mean()))
        for lab, m in (("fault", g.unit_has_fault), ("normal", ~g.unit_has_fault)):
            gg = g[m]
            rec[f"{lab}_units"] = len(gg)
            rec[f"{lab}_unit_mean"] = gg.rate.mean() if len(gg) else np.nan
            rec[f"{lab}_unit_q95"] = gg.rate.quantile(0.95) if len(gg) else np.nan
        rec["provision_factor_q95"] = rec["unit_q95"] / F
        out.append(rec)
    out = pd.DataFrame(out)
    out.to_csv(X.OUT / "per_unit_escalation.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
