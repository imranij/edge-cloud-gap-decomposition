"""How much of the oracle headroom independent disagreement noise alone would produce, and
the stacker against the better tier. Post-plan analysis, not pre-specified. No model refitted.

    python src/extra/oracle_reference.py

For every dataset x pair x split (main run) and the 3W by-well sensitivity run:
  independence reference  the oracle BA expected if edge and cloud erred independently at their
                          observed class-conditional error rates e_c and k_c on the test fold,
                          BA_ind = 1 - (e_1 k_1 + e_0 k_0)/2
  excess                  observed oracle BA - BA_ind (> 0: the tiers err on different windows more
                          often than independence implies; < 0: their errors overlap more)
  stacker vs best tier    BA(stacker) - max(B_eE, B_cC) on test (post hoc best tier), and
                          BA(stacker) - BA(tier selected on validation BA), the selection using the
                          stored validation probabilities, thresholds and the reconstructed
                          validation labels
Outputs: results/extra/oracle_reference.csv (NB-corrected means and 95% CIs per dataset x pair),
results/extra/oracle_reference_cells.csv (per split)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X
from escalation.metrics import balanced_accuracy

RUNS = [(ds, p) for ds in X.DATASETS for p in ("A", "B", "C")] + [("threew__by_well", "A")]


def val_labels(ds, split):
    if ds == "threew__by_well":
        from escalation.data import load_dataset
        from escalation.splits import make_splits
        if not hasattr(val_labels, "_bw"):
            d = load_dataset("threew", X.ROOT / "data" / "processed", "well")
            val_labels._bw = (d, {s["split_id"]: s for s in make_splits(d["y"], d["groups"])})
        d, sp = val_labels._bw
        return d["y"][sp[split]["val"]].astype(int)
    return X.dataset(ds)["y"][X.splits(ds)[split]["val"]].astype(int)


def main():
    df = X.long_results()
    rows = []
    for ds, pair in RUNS:
        sz = X.sizes()
        if ds == "threew__by_well":
            J = 20
        else:
            J = int(sz[(sz.dataset == ds) & (sz.pair == pair)].J.iloc[0])
        st = X.wide(df[(df.dataset == ds) & (df.pair == pair) & (df.h2_q == 1.0)],
                    ["ba_stacker_dep", "ba_oracle_dep", "ba_ef_ev", "ba_cf_cv"]).set_index("split")
        for s in range(J):
            z = X.arrays(ds, pair, s)
            y = z["y_test"].astype(int)
            pe = (z["p_ef_ev"] >= float(z["t_ef_ev"])).astype(int)
            pc = (z["p_cf_cv"] >= float(z["t_cf_cv"])).astype(int)
            e = [float((pe[y == c] != c).mean()) for c in (0, 1)]
            k = [float((pc[y == c] != c).mean()) for c in (0, 1)]
            ind = 1.0 - 0.5 * (e[0] * k[0] + e[1] * k[1])
            both_wrong = [float(((pe != y) & (pc != y))[y == c].mean()) for c in (0, 1)]
            yv = val_labels(ds, s)
            pev, pcv = z["pval_ef_ev"], z["pval_cf_cv"]
            assert len(pev) == len(yv), (ds, pair, s)
            bev = balanced_accuracy(yv, (pev >= float(z["t_ef_ev"])).astype(int))
            bcv = balanced_accuracy(yv, (pcv >= float(z["t_cf_cv"])).astype(int))
            r = st.loc[s]
            sel = r.ba_cf_cv if bcv >= bev else r.ba_ef_ev
            rows.append(dict(dataset=ds, pair=pair, split=s, oracle=r.ba_oracle_dep, oracle_indep=ind,
                             excess_over_indep=r.ba_oracle_dep - ind,
                             both_wrong_pos=both_wrong[1], both_wrong_neg=both_wrong[0],
                             indep_both_wrong_pos=e[1] * k[1], indep_both_wrong_neg=e[0] * k[0],
                             headroom_over_best_tier=r.ba_oracle_dep - max(r.ba_ef_ev, r.ba_cf_cv),
                             indep_headroom_over_best_tier=ind - max(r.ba_ef_ev, r.ba_cf_cv),
                             stacker_minus_best_test_tier=r.ba_stacker_dep - max(r.ba_ef_ev, r.ba_cf_cv),
                             stacker_minus_val_selected_tier=r.ba_stacker_dep - sel,
                             val_selected_cloud=float(bcv >= bev)))
    c = pd.DataFrame(rows)
    c.to_csv(X.OUT / "oracle_reference_cells.csv", index=False)
    out = []
    qs = ["oracle", "oracle_indep", "excess_over_indep", "headroom_over_best_tier", "indep_headroom_over_best_tier",
          "stacker_minus_best_test_tier", "stacker_minus_val_selected_tier", "val_selected_cloud"]
    for (ds, pair), g in c.groupby(["dataset", "pair"], sort=False):
        for q in qs:
            nbp = "A" if ds == "threew__by_well" else pair
            t = X.nb(g[q].to_numpy(), "threew" if ds == "threew__by_well" else ds, nbp)
            out.append(dict(dataset=ds, pair=pair, quantity=q, J=t["J"], mean=t["mean"], ci_lo=t["ci_lo"],
                            ci_hi=t["ci_hi"], p_two_sided=t["p"]))
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "oracle_reference.csv", index=False)
    pd.set_option("display.width", 220)
    print(o.pivot_table(index=["dataset", "pair"], columns="quantity", values="mean").round(3).to_string())
    print(o[o.quantity.isin(["excess_over_indep", "stacker_minus_val_selected_tier", "stacker_minus_best_test_tier"])].round(3).to_string())


if __name__ == "__main__":
    main()
