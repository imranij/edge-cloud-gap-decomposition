"""The information step by fault or disturbance
type, and the hydraulic results with transitional cycles excluded at test time.
Post-plan analysis, not pre-specified. No model is refitted: stored test predictions and
thresholds are re-scored on subsets of the test fold defined by data/processed/<ds>/meta.parquet.

    python src/extra/by_fault_type.py

For fault type t, BA_t of a model is (recall on the test positives of type t + specificity on all
test negatives) / 2, and the Shapley information step for t is the mean of
BA_t(ef_cv) - BA_t(ef_ev) and BA_t(cf_cv) - BA_t(cf_ev). Because the specificity part is shared by
every type, the recall part (the Shapley-averaged recall difference on type t, halved) carries all
differences between types; both are reported. Means are over the splits whose test fold contains
type t, with a Nadeau-Bengio corrected 95% CI when at least three such splits exist.

EDGE_REACH is our physical reading, written down before the per-type numbers were computed, of
whether the signature of each fault reaches the edge node's sensors. It was not tested.
Outputs: results/extra/info_by_fault_type.csv, results/extra/hydraulic_stable_only.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

EDGE_REACH = {
    "skab": {
        "rotor_imbalance": "edge (vibration of the motor-pump set)",
        "cavitation": "edge (pump vibration and motor load)",
        "valve1_inlet_closure": "mainly loop (flow, pressure); motor load secondary",
        "valve2_outlet_closure": "mainly loop (flow, pressure); motor load secondary",
        "fluid_leak_or_addition": "loop (fluid volume and flow)",
        "water_increase": "loop (fluid volume and flow)",
        "warm_water_supply": "loop (fluid temperature)",
    },
    "threew": {
        "abrupt_increase_of_bsw": "edge (pressure and temperature at the tree)",
        "spurious_closure_of_dhsv": "edge (pressure at the tree)",
        "severe_slugging": "edge (pressure oscillation along the line, tree included)",
        "flow_instability": "edge (pressure oscillation along the line, tree included)",
        "rapid_productivity_loss": "edge (reservoir-side pressure at the tree)",
        "hydrate_in_production_line": "both (between tree and platform)",
        "scaling_in_pck": "mainly topside (production choke)",
        "hydrate_in_service_line": "mainly topside (gas-lift line)",
    },
    "hydraulic": {"pump_leak_weak": "partly edge (motor power); flow FS1 is cloud-only",
                  "pump_leak_severe": "partly edge (motor power); flow FS1 is cloud-only"},
    "seu": {},
}
MODELS = ["ef_ev", "cf_ev", "ef_cv", "cf_cv"]


def preds(z):
    return {k: (z[f"p_{k}"] >= float(z[f"t_{k}"])).astype(int) for k in MODELS}


def main():
    rows, halves = [], []
    for ds in X.DATASETS:
        meta = X.dataset(ds)["meta"]
        sz = X.sizes()
        for pair in ("A", "B", "C"):
            J = int(sz[(sz.dataset == ds) & (sz.pair == pair)].J.iloc[0])
            for s in range(J):
                z = X.arrays(ds, pair, s)
                y = z["y_test"].astype(int)
                ft = meta.fault_type.to_numpy()[z["test_idx"]]
                P = preds(z)
                spec = {k: float((P[k][y == 0] == 0).mean()) for k in MODELS}
                recall_all = {k: float((P[k][y == 1] == 1).mean()) for k in MODELS}
                halves.append(dict(dataset=ds, pair=pair, split=s,
                                   info_spec_half=0.25 * ((spec["ef_cv"] - spec["ef_ev"]) + (spec["cf_cv"] - spec["cf_ev"])),
                                   info_recall_half=0.25 * ((recall_all["ef_cv"] - recall_all["ef_ev"]) + (recall_all["cf_cv"] - recall_all["cf_ev"])),
                                   model_spec_half=0.25 * ((spec["cf_ev"] - spec["ef_ev"]) + (spec["cf_cv"] - spec["ef_cv"])),
                                   model_recall_half=0.25 * ((recall_all["cf_ev"] - recall_all["ef_ev"]) + (recall_all["cf_cv"] - recall_all["ef_cv"]))))
                for t in np.unique(ft[y == 1]):
                    m = (y == 1) & (ft == t)
                    rec = {k: float((P[k][m] == 1).mean()) for k in MODELS}
                    ba = {k: 0.5 * (rec[k] + spec[k]) for k in MODELS}
                    rows.append(dict(dataset=ds, pair=pair, split=s, fault_type=t, n_pos=int(m.sum()),
                                     info_step=0.5 * ((ba["ef_cv"] - ba["ef_ev"]) + (ba["cf_cv"] - ba["cf_ev"])),
                                     info_recall_part=0.25 * ((rec["ef_cv"] - rec["ef_ev"]) + (rec["cf_cv"] - rec["cf_ev"])),
                                     info_spec_part=0.25 * ((spec["ef_cv"] - spec["ef_ev"]) + (spec["cf_cv"] - spec["cf_ev"])),
                                     model_step=0.5 * ((ba["cf_ev"] - ba["ef_ev"]) + (ba["cf_cv"] - ba["ef_cv"])),
                                     **{f"recall_{k}": rec[k] for k in MODELS}))
    c = pd.DataFrame(rows)
    c.to_csv(X.OUT / "info_by_fault_type_cells.csv", index=False)
    out = []
    for (ds, pair, t), g in c.groupby(["dataset", "pair", "fault_type"]):
        r = dict(dataset=ds, pair=pair, fault_type=t, edge_reach=EDGE_REACH.get(ds, {}).get(t, "not classified"),
                 J_present=len(g), mean_n_pos=g.n_pos.mean())
        for q in ("info_step", "info_recall_part", "model_step", "recall_ef_ev", "recall_cf_ev", "recall_ef_cv", "recall_cf_cv"):
            if len(g) >= 3:
                tt = X.nb(g[q].to_numpy(), ds, pair)
                r.update({q: tt["mean"], f"{q}_lo": tt["ci_lo"], f"{q}_hi": tt["ci_hi"]})
            else:
                r.update({q: g[q].mean(), f"{q}_lo": np.nan, f"{q}_hi": np.nan})
        out.append(r)
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "info_by_fault_type.csv", index=False)
    hv = pd.DataFrame(halves)
    hsumm = []
    for (ds, pair), g in hv.groupby(["dataset", "pair"]):
        for q in ("info_spec_half", "info_recall_half", "model_spec_half", "model_recall_half"):
            tt = X.nb(g[q].to_numpy(), ds, pair)
            hsumm.append(dict(dataset=ds, pair=pair, quantity=q, mean=tt["mean"], ci_lo=tt["ci_lo"], ci_hi=tt["ci_hi"], J=tt["J"]))
    pd.DataFrame(hsumm).to_csv(X.OUT / "info_step_halves.csv", index=False)
    print(pd.DataFrame(hsumm).round(3).to_string())

    # hydraulic: drop the cycles flagged as possibly not yet stable (stable = 1) from each test fold
    meta = X.dataset("hydraulic")["meta"]
    from escalation.metrics import balanced_accuracy, escalation_gain, weighted_concordance
    hs = []
    for pair in ("A", "B"):
        for s in range(20):
            z = X.arrays("hydraulic", pair, s)
            keep = meta.stable.to_numpy()[z["test_idx"]] == 0
            y = z["y_test"].astype(int)
            P = preds(z)
            b = {k: balanced_accuracy(y[keep], P[k][keep]) for k in MODELS}
            ball = {k: balanced_accuracy(y, P[k]) for k in MODELS}
            g = escalation_gain(y[keep], P["ef_ev"][keep], P["cf_cv"][keep])
            hs.append(dict(pair=pair, split=s, n_kept=int(keep.sum()), n_test=len(y),
                           info_stable=0.5 * ((b["ef_cv"] - b["ef_ev"]) + (b["cf_cv"] - b["cf_ev"])),
                           info_all=0.5 * ((ball["ef_cv"] - ball["ef_ev"]) + (ball["cf_cv"] - ball["cf_ev"])),
                           model_stable=0.5 * ((b["cf_ev"] - b["ef_ev"]) + (b["cf_cv"] - b["ef_cv"])),
                           edge_stable=b["ef_ev"], cloud_stable=b["cf_cv"],
                           C_R1_stable=weighted_concordance(g, z["score_R1"][keep]),
                           C_R6_stable=weighted_concordance(g, z["score_R6"][keep])))
    h = pd.DataFrame(hs)
    hsum = []
    for pair, g in h.groupby("pair"):
        for q in ("info_stable", "info_all", "model_stable", "edge_stable", "cloud_stable", "C_R1_stable", "C_R6_stable", "n_kept"):
            tt = X.nb(g[q].to_numpy(), "hydraulic", pair)
            hsum.append(dict(pair=pair, quantity=q, mean=tt["mean"], ci_lo=tt["ci_lo"], ci_hi=tt["ci_hi"], J=tt["J"]))
    pd.DataFrame(hsum).to_csv(X.OUT / "hydraulic_stable_only.csv", index=False)
    pd.set_option("display.width", 250)
    print(o[["dataset", "pair", "fault_type", "edge_reach", "J_present", "mean_n_pos", "info_step", "info_step_lo",
             "info_step_hi", "info_recall_part", "recall_ef_ev", "recall_cf_cv"]].round(3).to_string())
    print(pd.DataFrame(hsum).round(3).to_string())


if __name__ == "__main__":
    main()
