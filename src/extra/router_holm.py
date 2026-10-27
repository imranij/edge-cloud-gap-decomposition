"""POST-PLAN, evaluation only. Refits no model.

Holm adjustment of the post-plan paired router comparisons within each contrast, over the eight
dataset x pair cells, from results_sens/routers/p2_diffs.csv (Nadeau-Bengio corrected two-sided p per
cell). The HKSJ-pooled rows are kept beside them unadjusted.
Output: results/extra/router_holm.csv
    python src/extra/router_holm.py
"""
from __future__ import annotations
import pandas as pd

import common as X


def main():
    d = pd.read_csv(X.ROOT / "results_sens" / "routers" / "p2_diffs.csv")
    cells = d[~d.dataset.str.startswith("POOLED")].copy()
    cells["p_holm_within_contrast"] = float("nan")
    for con, g in cells.groupby("contrast"):
        if len(g) >= 2:
            cells.loc[g.index, "p_holm_within_contrast"] = X.S.holm(g.p.to_numpy())
        cells.loc[g.index, "n_cells_in_family"] = len(g)
    out = pd.concat([cells, d[d.dataset.str.startswith("POOLED")]], ignore_index=True)
    out.to_csv(X.OUT / "router_holm.csv", index=False)
    keep = out.contrast.isin(["C(R6x) - C(R1)", "C(HyR) - C(R1)", "C(R6x) - C(R6)"])
    print(out[keep].round(4).to_string())


if __name__ == "__main__":
    main()
