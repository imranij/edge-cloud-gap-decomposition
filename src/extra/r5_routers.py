"""POST-PLAN, evaluation only. Refits no model.

Adds the cross-fitted R4x and R5x to the post-plan paired router contrasts against confidence routing R1:
C(R4x) - C(R1) and C(R5x) - C(R1) per dataset x pair (pre-specified splits, full training data), Nadeau-Bengio
corrected CIs with the same sizes as results_sens/routers/p2_diffs.csv, and Holm adjustment within each contrast
over the eight cells (as src/extra/router_holm.py). The stored per-split C values come from
results_sens/routers/p2_rows.parquet. Check: C(R6x) - C(R1) recomputed here reproduces p2_diffs.csv in every cell.
Output: results/extra/r5_router_r4x_r5x.csv
    python src/extra/r5_routers.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

RS = X.ROOT / "results_sens" / "routers"


def main():
    rows = pd.read_parquet(RS / "p2_rows.parquet")
    C = rows[rows.quantity == "C"].pivot_table(index=["dataset", "pair", "split"], columns="router", values="value")
    ref = pd.read_csv(RS / "p2_diffs.csv")
    out = []
    for a in ("R6x", "R4x", "R5x"):
        for pair in ("A", "B"):
            for ds in X.DATASETS:
                if (ds, pair) not in {(i[0], i[1]) for i in C.index}:
                    continue
                d = (C.loc[(ds, pair)][a] - C.loc[(ds, pair)]["R1"]).dropna()
                t = X.nb(d.to_numpy(float), ds, pair)
                out.append(dict(contrast=f"C({a}) - C(R1)", pair=pair, dataset=ds, J=t["J"], mean=t["mean"],
                                ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], p=t["p"],
                                C_a=float(C.loc[(ds, pair)][a].mean()), C_R1=float(C.loc[(ds, pair)]["R1"].mean())))
    o = pd.DataFrame(out)
    for con, g in o.groupby("contrast"):
        o.loc[g.index, "p_holm_within_contrast"] = X.S.holm(g.p.to_numpy())
    # reproduce the stored R6x contrast
    chk = o[o.contrast == "C(R6x) - C(R1)"].merge(ref[ref.contrast == "C(R6x) - C(R1)"], on=["pair", "dataset"])
    assert len(chk) == 8 and np.allclose(chk.mean_x, chk.mean_y) and np.allclose(chk.p_x, chk.p_y)
    o.to_csv(X.OUT / "r5_router_r4x_r5x.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.round(4).to_string())


if __name__ == "__main__":
    main()
