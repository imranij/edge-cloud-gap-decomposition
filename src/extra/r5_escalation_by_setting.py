"""POST-PLAN, evaluation only. Refits no model.

Per-unit escalation rate of R1 with the causal threshold at a nominal 10% on the hydraulic rig, by internal pump
leakage level (0 normal, 1 weak, 2 severe) and by cooler level (3, 20, 100 %), for pairs A and B. The per-unit
rates are those of results/extra/per_unit_escalation_units.csv (reconstructed exactly from stored arrays by
src/extra/per_unit_escalation.py); the unit is a run or run chunk with constant component settings, so its
settings are read from data/processed/hydraulic/meta.parquet (group_id = unit). Two summaries per level: the mean
of the per-unit rates over all (split, unit) cells (as in Sec. 5.3), and the window-weighted rate.
Output: results/extra/r5_escalation_by_setting.csv
    python src/extra/r5_escalation_by_setting.py
"""
from __future__ import annotations
import pandas as pd

import common as X


def main():
    u = pd.read_csv(X.OUT / "per_unit_escalation_units.csv")
    u = u[(u.dataset == "hydraulic") & (u.router == "R1")]
    m = pd.read_parquet(X.ROOT / "data" / "processed" / "hydraulic" / "meta.parquet")
    s = m.groupby("group_id")[["pump", "cooler", "valve", "accumulator"]].agg(["min", "max"])
    for c in ("pump", "cooler", "valve", "accumulator"):
        assert (s[(c, "min")] == s[(c, "max")]).all(), f"{c} not constant within a unit"
    s = m.groupby("group_id")[["pump", "cooler", "valve", "accumulator"]].first()
    u = u.join(s, on="unit")
    assert u.pump.notna().all()
    # check: the pre-existing fault/normal split of Sec. 5.3 (0.149 normal, 0.088 faulty for pair A)
    a = u[u.pair == "A"]
    print("pair A normal/faulty unit means:", round(a[~a.unit_has_fault].rate.mean(), 3), round(a[a.unit_has_fault].rate.mean(), 3))
    rows = []
    lab = {"pump": {0: "normal", 1: "weak leakage", 2: "severe leakage"}}
    for pair, g in u.groupby("pair"):
        for col in ("pump", "cooler", "valve", "accumulator"):
            for lev, h in g.groupby(col):
                rows.append(dict(pair=pair, setting=col, level=lev, level_name=lab.get(col, {}).get(lev, str(lev)),
                                 n_unit_cells=len(h), mean_unit_rate=h.rate.mean(),
                                 window_weighted_rate=(h.rate * h.n_windows).sum() / h.n_windows.sum(),
                                 share_units_zero=(h.rate == 0).mean()))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "r5_escalation_by_setting.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.round(3).to_string())


if __name__ == "__main__":
    main()
