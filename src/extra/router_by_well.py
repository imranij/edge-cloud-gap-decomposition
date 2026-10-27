"""POST-PLAN, evaluation only. Refits no model.

Do the 3W router advantages over confidence (R1) survive when routers are scored only on pairs of
windows from the same well? A router whose edge features encode well identity can rank whole wells
by how often the cloud is right; within one well that information is useless.

For every 3W split of pairs A and B: the gains g and test indices come from the stored main run
(results/threew/<pair>/arrays_<j>.npz), the router scores from the cross-fitted router run
(results_sens/routers/p2_arrays/threew_<pair>_<j>.npz; R1 and R6 are checked to be identical in both),
and the well of each test window from data/processed/threew/meta.parquet. The well-stratified
concordance is
    C_w(s) = sum_wells D_k C_k(s) / sum_wells D_k,
with C_k the weighted concordance (Eq. 6) inside well k and D_k its denominator, i.e. the weighted
concordance restricted to pairs of windows in the same well. Per-split values and paired differences are
summarised by the Nadeau-Bengio corrected mean and 95% CI (3W sizes of the main run).
Outputs: results/extra/router_by_well.csv (summary), router_by_well_splits.csv (per split)
    python src/extra/router_by_well.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import common as X

ROUTERS = ["R1", "R6", "R4x", "R6x", "HyR"]
P2 = X.ROOT / "results_sens" / "routers" / "p2_arrays"


def S_D(g, s):
    n = len(g)
    gs = np.sort(g)
    D = float((gs * (2 * np.arange(1, n + 1) - n - 1)).sum())
    R = rankdata(s, method="average")
    S = float((g * (R - (n + 1) / 2.0)).sum())
    return S, D


def main():
    meta = pd.read_parquet(X.ROOT / "data" / "processed" / "threew" / "meta.parquet")
    rows = []
    for pair in ("A", "B"):
        for j in range(20):
            z = X.arrays("threew", pair, j)
            p = np.load(P2 / f"threew_{pair}_{j}.npz")
            assert np.allclose(z["score_R1"], p["score_R1"]) and np.allclose(z["score_R6"], p["score_R6"])
            g = z["g_dep"]; wells = meta.well.to_numpy()[z["test_idx"]]
            rec = dict(pair=pair, split=j, n_wells=len(set(wells)))
            for r in ROUTERS:
                s = p[f"score_{r}"]
                rec[f"C_{r}"] = X.M.weighted_concordance(g, s)
                Ssum = Dsum = 0.0
                for w in np.unique(wells):
                    ix = wells == w
                    if ix.sum() < 2:
                        continue
                    Sk, Dk = S_D(g[ix], s[ix])
                    if Dk > 0:
                        Ssum += Sk; Dsum += Dk
                rec[f"Cw_{r}"] = 0.5 + Ssum / Dsum if Dsum > 0 else np.nan
            # share of the oracle-ranking weight that lies within wells
            S0, D0 = S_D(g, g)
            Dw = sum(S_D(g[wells == w], g[wells == w])[1] for w in np.unique(wells) if (wells == w).sum() > 1)
            rec["share_D_within_wells"] = Dw / D0 if D0 > 0 else np.nan
            rows.append(rec)
    per = pd.DataFrame(rows)
    per.to_csv(X.OUT / "router_by_well_splits.csv", index=False)
    out = []
    for pair, g in per.groupby("pair"):
        def add(name, v):
            t = X.nb(v.to_numpy(), "threew", pair)
            out.append(dict(pair=pair, quantity=name, J=t["J"], mean=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"],
                            p_two_sided=t["p"]))
        for r in ROUTERS:
            add(f"C({r}) pooled over wells", g[f"C_{r}"])
            add(f"C({r}) within wells", g[f"Cw_{r}"])
        for r in ("R6x", "HyR", "R4x"):
            add(f"C({r}) - C(R1) pooled over wells", g[f"C_{r}"] - g["C_R1"])
            add(f"C({r}) - C(R1) within wells", g[f"Cw_{r}"] - g["Cw_R1"])
        add("share of oracle weight D within wells", g["share_D_within_wells"])
    out = pd.DataFrame(out)
    out.to_csv(X.OUT / "router_by_well.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
