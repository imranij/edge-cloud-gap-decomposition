"""Does our physical reading of where each fault's signature reaches ('Reaches' in the
per-fault-type table) predict the recall part of the information step?
Post-plan analysis, not pre-specified. Reads results/extra/info_by_fault_type_cells.csv
(written by src/extra/by_fault_type.py from stored predictions); no model is refitted.

    python src/extra/fault_reach_test.py

The reading was written down before the per-type numbers were computed. If it were right, the
cloud-only sensors should raise recall more on faults read as acting away from the edge node
('loop', 'topside', 'both') than on faults read as reaching it ('edge'). Per split we take the mean
recall part over the non-edge types present in the test fold minus the mean over the edge types
present, and test the mean of that contrast over splits with the Nadeau-Bengio correction
(two-sided). Splits whose test fold lacks either kind are skipped. The hydraulic rig has no
contrast (both of its types are read as 'partly'), and SEU was not classified.
Output: results/extra/fault_reach_test.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X
from by_fault_type import EDGE_REACH


def kind(ds, t):
    s = EDGE_REACH.get(ds, {}).get(t, "")
    return "edge" if s.startswith("edge") else ("away" if s else "")


def main():
    c = pd.read_csv(X.OUT / "info_by_fault_type_cells.csv")
    rows = []
    for ds in ("skab", "threew"):
        for pair in ("A", "B", "C"):
            g = c[(c.dataset == ds) & (c.pair == pair)].copy()
            if g.empty:
                continue
            g["kind"] = [kind(ds, t) for t in g.fault_type]
            con = []
            for s, h in g.groupby("split"):
                a = h[h.kind == "away"].info_recall_part
                e = h[h.kind == "edge"].info_recall_part
                if len(a) and len(e):
                    con.append(a.mean() - e.mean())
            t = X.nb(np.array(con), ds, pair)
            rows.append(dict(dataset=ds, pair=pair, J=t["J"], contrast_away_minus_edge=t["mean"], ci_lo=t["ci_lo"],
                             ci_hi=t["ci_hi"], p_two_sided=t["p"],
                             mean_recall_part_away=g[g.kind == "away"].info_recall_part.mean(),
                             mean_recall_part_edge=g[g.kind == "edge"].info_recall_part.mean(),
                             n_types_away=g[g.kind == "away"].fault_type.nunique(),
                             n_types_edge=g[g.kind == "edge"].fault_type.nunique()))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "fault_reach_test.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.round(3).to_string())


if __name__ == "__main__":
    main()
