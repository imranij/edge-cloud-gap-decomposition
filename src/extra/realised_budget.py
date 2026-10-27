"""Realised escalation fraction and balanced accuracy when the budget is fixed on
validation data and applied window by window on the test fold (no look-ahead).
Post-plan analysis, not pre-specified. No model is refitted.

    python src/extra/realised_budget.py

For router r and nominal budget f the threshold is the (1 - f) quantile of r's scores on the
validation windows of the split; a test window is escalated when its score is strictly above it.
Validation scores are recomputed from stored quantities only:
  R0  a fresh uniform draw per validation window (seed 6000 + split);
  R1  -|p~ - 0.5| from the stored validation edge probabilities (pval_ef_ev) and threshold t_ef_ev;
  R2  binary entropy of pval_ef_ev;
  R3  split-conformal score of each validation window against the validation set itself
      (in-sample calibration, the only version a deployed node could compute from these data);
  R9  the periodic schedule from each validation window's position in its unit.
Test scores are the stored score_R* arrays; the recomputed R1 test score is checked against the
stored one. R4 to R8 have no out-of-fold validation scores in the stored outputs, so they are not
covered. Realised BA = BA(edge) + sum of the stored gains g_dep over the escalated test windows.
The look-ahead BA at the same nominal budget (top-k of the whole test fold) is reported beside it.
Uplink and cloud calls per hour scale the realised fraction by the window rates and the cloud
feature payload of results/extra/uplink_per_hour.csv (pull flow, features).
Outputs: results/extra/realised_budget_cells.csv, results/extra/realised_budget.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X
from escalation import routers as RT
from escalation.metrics import ba_at_budget

BUDGETS = (0.05, 0.10, 0.30)
ROUTERS = ("R0", "R1", "R2", "R3", "R9")


def val_scores(z, yv, order_v, split):
    pv = z["pval_ef_ev"].astype(float)
    t = float(z["t_ef_ev"])
    return {"R0": np.random.default_rng(6000 + split).random(len(pv)),
            "R1": RT.r1_confidence(pv, t),
            "R2": RT.r2_entropy(pv),
            "R3": RT.r3_conformal(pv, yv, t, pv),
            "R9": RT.r9_periodic(order_v)}


def main():
    up = pd.read_csv(X.OUT / "uplink_per_hour.csv").drop_duplicates("dataset").set_index("dataset")
    sz = X.sizes()
    rows, maxdiff = [], 0.0
    for ds in X.DATASETS:
        d = X.dataset(ds)
        sp = X.splits(ds)
        for pair in ("A", "B", "C"):
            J = int(sz[(sz.dataset == ds) & (sz.pair == pair)].J.iloc[0])
            for s in range(J):
                z = X.arrays(ds, pair, s)
                val = sp[s]["val"]
                if len(val) != len(z["pval_ef_ev"]):
                    raise RuntimeError(f"validation length mismatch {ds} {pair} {s}")
                yv = d["y"][val].astype(int)
                vs = val_scores(z, yv, d["order"][val], s)
                y = z["y_test"].astype(int)
                pe = (z["p_ef_ev"] >= float(z["t_ef_ev"])).astype(int)
                pc = (z["p_cf_cv"] >= float(z["t_cf_cv"])).astype(int)
                g = z["g_dep"]
                ba_e = float((np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum())) * (pe == y)).sum())
                r1_test = RT.r1_confidence(z["p_ef_ev"].astype(float), float(z["t_ef_ev"]))
                maxdiff = max(maxdiff, float(np.max(np.abs(r1_test - z["score_R1"]))))
                for r in ROUTERS:
                    st = z[f"score_{r}"].astype(float)
                    for f in BUDGETS:
                        thr = float(np.quantile(vs[r], 1 - f))
                        esc = st > thr
                        rows.append(dict(dataset=ds, pair=pair, split=s, router=r, budget=f,
                                         realised_fraction=float(esc.mean()),
                                         ba_realised=ba_e + float(g[esc].sum()),
                                         ba_lookahead=ba_at_budget(y, pe, pc, st, f), ba_edge=ba_e))
    c = pd.DataFrame(rows)
    c.to_csv(X.OUT / "realised_budget_cells.csv", index=False)
    print("max |recomputed R1 - stored score_R1| on test:", maxdiff)
    out = []
    for (ds, pair, r, f), g in c.groupby(["dataset", "pair", "router", "budget"]):
        fr = g.realised_fraction.to_numpy()
        a = X.nb(g.ba_realised.to_numpy(), ds, pair)
        b = X.nb(g.ba_lookahead.to_numpy(), ds, pair)
        dd = X.nb((g.ba_realised - g.ba_lookahead).to_numpy(), ds, pair)
        u = up.loc[ds]
        wph, fb = float(u.windows_per_hour), float(u.features_cloud_B)
        out.append(dict(dataset=ds, pair=pair, router=r, budget=f, J=len(g),
                        frac_mean=fr.mean(), frac_sd=fr.std(ddof=1), frac_min=fr.min(), frac_max=fr.max(),
                        ba_realised=a["mean"], ba_realised_lo=a["ci_lo"], ba_realised_hi=a["ci_hi"],
                        ba_lookahead=b["mean"], diff_realised_minus_lookahead=dd["mean"],
                        diff_lo=dd["ci_lo"], diff_hi=dd["ci_hi"],
                        uplink_nominal_B_per_h=f * wph * fb,
                        uplink_min_B_per_h=fr.min() * wph * fb, uplink_max_B_per_h=fr.max() * wph * fb,
                        cloud_calls_min_per_h=fr.min() * wph, cloud_calls_max_per_h=fr.max() * wph))
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "realised_budget.csv", index=False)
    pd.set_option("display.width", 250)
    print(o[o.router.isin(["R1", "R0"]) & (o.pair != "C")].drop(columns=["uplink_nominal_B_per_h"]).round(3).to_string())


if __name__ == "__main__":
    main()
