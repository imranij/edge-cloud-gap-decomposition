"""Paired Nadeau-Bengio tests of C(R6) - C(R1)
and C(R6) - C(R2) for the pre-specified routers, per dataset and pair, at full training data.
Post-plan analysis, not pre-specified. Reads the stored per-split C values; no model is
refitted. The cross-fitted routers and the R1 + R6 hybrid need new fits and are in src/sens/routers/.

    python src/extra/router_paired.py   -> results/extra/router_paired.csv
"""
from __future__ import annotations
import pandas as pd

import common as X


def main():
    df = X.long_results()
    d = df[df.dataset.isin(X.DATASETS) & (df.h2_q == 1.0) & (df.quantity == "C")]
    w = d.pivot_table(index=["dataset", "pair", "split"], columns="router", values="value").reset_index()
    rows = []
    for (ds, pair), g in w.groupby(["dataset", "pair"]):
        for other in ("R1", "R2"):
            t = X.nb((g["R6"] - g[other]).to_numpy(float), ds, pair)
            rows.append(dict(dataset=ds, pair=pair, comparison=f"C(R6) - C({other})", J=t["J"], mean=t["mean"],
                             ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], p_two_sided=t["p"]))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "router_paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.round(3).to_string())


if __name__ == "__main__":
    main()
