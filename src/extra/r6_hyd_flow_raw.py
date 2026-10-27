"""POST-PLAN, descriptive. Reads the raw hydraulic files only; fits nothing and is
not used for sensor placement (the edge view was fixed in plan/ADDENDUM_A.md before any run).

Per-cycle means of the two flow sensors FS1 and FS2 (10 Hz, l/min) and the motor power EPS1 (100 Hz, W), from
data/raw/hydraulic/{FS1,FS2,EPS1}.txt (2,205 cycles of 60 s, one row per cycle), joined to profile.txt (cooler
efficiency %, valve %, internal pump leakage 0/1/2, accumulator bar, stable flag). Outputs:
  results/extra/r6_hyd_flow_raw.csv        mean, within-level SD and n of the per-cycle means by setting and level,
                                           and by leakage level within each cooler level
  results/extra/r6_hyd_flow_raw_corr.csv   Pearson correlation of each sensor's per-cycle mean with the cycle index,
                                           the cooler setting and the leakage level
    python src/extra/r6_hyd_flow_raw.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "hydraulic"
OUT = ROOT / "results" / "extra"


def main():
    prof = pd.read_csv(RAW / "profile.txt", sep="\t", header=None,
                       names=["cooler", "valve", "pump", "accumulator", "stable"])
    d = prof.copy()
    d["cycle"] = np.arange(len(d))
    for s in ("FS1", "FS2", "EPS1"):
        x = np.loadtxt(RAW / f"{s}.txt")
        assert x.shape[0] == len(d)
        d[s] = x.mean(axis=1)
    rows = []
    for s in ("FS1", "FS2", "EPS1"):
        for col in ("cooler", "pump"):
            for lev, g in d.groupby(col):
                rows.append(dict(sensor=s, by=col, level=lev, within="all", n=len(g), mean=g[s].mean(), sd=g[s].std(ddof=1)))
        for c, gc in d.groupby("cooler"):
            for lev, g in gc.groupby("pump"):
                rows.append(dict(sensor=s, by="pump", level=lev, within=f"cooler {c}", n=len(g), mean=g[s].mean(),
                                 sd=g[s].std(ddof=1)))
    o = pd.DataFrame(rows)
    # within-cooler drop from normal to weak leakage, in l/min, % of reading and within-level SDs
    add = []
    for s in ("FS1", "FS2"):
        for c in sorted(d.cooler.unique()):
            g = o[(o.sensor == s) & (o.within == f"cooler {c}")].set_index("level")
            drop = g.loc[0, "mean"] - g.loc[1, "mean"]
            add.append(dict(sensor=s, by="weak-leak drop", level=1, within=f"cooler {c}", n=int(g.loc[[0, 1], "n"].sum()),
                            mean=drop, sd=max(g.loc[0, "sd"], g.loc[1, "sd"]),
                            pct_of_reading=100 * drop / g.loc[0, "mean"], drop_in_sd=drop / max(g.loc[0, "sd"], g.loc[1, "sd"])))
    o = pd.concat([o, pd.DataFrame(add)], ignore_index=True)
    o.to_csv(OUT / "r6_hyd_flow_raw.csv", index=False)
    corr = pd.DataFrame([dict(sensor=s, r_cycle=np.corrcoef(d[s], d.cycle)[0, 1], r_cooler=np.corrcoef(d[s], d.cooler)[0, 1],
                              r_leakage=np.corrcoef(d[s], d.pump)[0, 1]) for s in ("FS1", "FS2", "EPS1")])
    corr.to_csv(OUT / "r6_hyd_flow_raw_corr.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.round(4).to_string()); print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
