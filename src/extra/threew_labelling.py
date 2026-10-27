"""3W labelling disclosure and an evaluation-only sensitivity analysis.
Post-plan analysis, not pre-specified. No model is refitted.

    python src/extra/threew_labelling.py

1. Event-type table of the 445-instance confirmatory subset (data/processed/threew/meta.parquet):
   per instance event type the numbers of instances, episodes, wells and windows, the fault-window
   count, how many instances carry no fault window, and how many instances hold both classes
   (a within-instance normal contrast). Output: threew_event_table.csv, and the per-well detail
   threew_event_by_well.csv.
2. Sensitivity: the windows of event instances that carry no fault window (all of them labelled
   normal; 'hydrate_in_service_line' instances of three wells) are removed from every test fold, and
   the stored predictions and thresholds are re-scored on the rest. For pairs A, B and C (grouped by
   episode) and pair A grouped by well we report the four factorial balanced accuracies, the
   Shapley model and information steps, the oracle, C of R1 and R6 and capture of R6 over budgets
   5% to 30%, each against the same quantity on the full test fold. The per-window gains are
   recomputed on the reduced fold because the balanced-accuracy weights change.
   Output: threew_unlabelled_sensitivity.csv (mean and NB 95% CI; for n_removed_per_fold the ci
   columns hold the minimum and maximum, and spec_on_removed_<model> is the mean specificity of each
   model on the removed windows over the splits that contain any) and ..._cells.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X
from escalation.metrics import balanced_accuracy, escalation_gain, weighted_concordance, capture, oracle_predictions

MODELS = ["ef_ev", "cf_ev", "ef_cv", "cf_cv"]


def event_tables(meta):
    inst = meta.groupby("instance").agg(ev=("instance_event_name", "first"), well=("well", "first"),
                                        ep=("episode_id", "first"), npos=("label", "sum"), n=("label", "size"))
    inst["both"] = (inst.npos > 0) & (inst.npos < inst.n)
    inst["nofault"] = inst.npos == 0
    t = inst.groupby("ev").agg(instances=("n", "size"), episodes=("ep", "nunique"), wells=("well", "nunique"),
                               windows=("n", "sum"), fault_windows=("npos", "sum"),
                               instances_without_fault=("nofault", "sum"),
                               instances_with_both_classes=("both", "sum")).reset_index()
    t["well_list"] = [", ".join(sorted(inst[inst.ev == e].well.unique())) for e in t.ev]
    t = t.rename(columns={"ev": "instance_event_name"}).sort_values("windows", ascending=False)
    t.to_csv(X.OUT / "threew_event_table.csv", index=False)
    w = inst.groupby(["ev", "well"]).agg(instances=("n", "size"), episodes=("ep", "nunique"), windows=("n", "sum"),
                                         fault_windows=("npos", "sum"),
                                         instances_without_fault=("nofault", "sum")).reset_index()
    w.to_csv(X.OUT / "threew_event_by_well.csv", index=False)
    drop_inst = inst[(inst.ev != "normal") & inst.nofault].index
    return t, w, set(drop_inst)


def scores(z, keep):
    y = z["y_test"].astype(int)[keep]
    P = {k: (z[f"p_{k}"][keep] >= float(z[f"t_{k}"])).astype(int) for k in MODELS}
    b = {k: balanced_accuracy(y, P[k]) for k in MODELS}
    g = escalation_gain(y, P["ef_ev"], P["cf_cv"])
    o = balanced_accuracy(y, oracle_predictions(y, P["ef_ev"], P["cf_cv"]))
    return dict(B_eE=b["ef_ev"], B_cE=b["cf_ev"], B_eC=b["ef_cv"], B_cC=b["cf_cv"],
                model=0.5 * ((b["cf_ev"] - b["ef_ev"]) + (b["cf_cv"] - b["ef_cv"])),
                info=0.5 * ((b["ef_cv"] - b["ef_ev"]) + (b["cf_cv"] - b["cf_ev"])),
                oracle=o, C_R1=weighted_concordance(g, z["score_R1"][keep]),
                C_R6=weighted_concordance(g, z["score_R6"][keep]),
                cap_R6=capture(g, z["score_R6"][keep], 0.05, 0.30), n=int(keep.sum()),
                n_neg=int((y == 0).sum()))


def main():
    meta = X.dataset("threew")["meta"]
    t, w, drop = event_tables(meta)
    print(t.to_string())
    print("instances without fault label, by well:")
    print(w[(w.ev != "normal") & (w.instances_without_fault > 0)].to_string())
    inst = meta.instance.to_numpy()
    sz = X.sizes()
    rows = []
    for name, pair in (("threew", "A"), ("threew", "B"), ("threew", "C"), ("threew__by_well", "A")):
        J = int(sz[(sz.dataset == name) & (sz.pair == pair)].J.iloc[0])
        for s in range(J):
            z = X.arrays(name, pair, s)
            full = np.ones(len(z["y_test"]), bool)
            keep = ~np.isin(inst[z["test_idx"]], list(drop))
            a = scores(z, full)
            r = dict(dataset=name, pair=pair, split=s, n_removed=int((~keep).sum()))
            for k in MODELS:   # specificity of each model on the removed (all-normal) windows
                rm = ~keep
                r[f"spec_removed_{k}"] = (float(((z[f"p_{k}"][rm] >= float(z[f"t_{k}"])) == 0).mean())
                                          if rm.any() else np.nan)
            if len(np.unique(z["y_test"][keep])) == 2:
                b = scores(z, keep)
            else:
                b = {k: np.nan for k in a}
            r.update({f"{k}_full": v for k, v in a.items()})
            r.update({f"{k}_excl": v for k, v in b.items()})
            rows.append(r)
    c = pd.DataFrame(rows)
    c.to_csv(X.OUT / "threew_unlabelled_sensitivity_cells.csv", index=False)
    out = []
    qs = ["B_eE", "B_cE", "B_eC", "B_cC", "model", "info", "oracle", "C_R1", "C_R6", "cap_R6"]
    for (name, pair), g in c.groupby(["dataset", "pair"]):
        for q in qs:
            for ver in ("full", "excl", "diff"):
                v = (g[f"{q}_excl"] - g[f"{q}_full"]) if ver == "diff" else g[f"{q}_{ver}"]
                tt = X.nb(v.to_numpy(float), name, pair)
                out.append(dict(dataset=name, pair=pair, quantity=q, version=ver, J=tt["J"], mean=tt["mean"],
                                ci_lo=tt["ci_lo"], ci_hi=tt["ci_hi"]))
        out.append(dict(dataset=name, pair=pair, quantity="n_removed_per_fold", version="excl", J=len(g),
                        mean=g.n_removed.mean(), ci_lo=g.n_removed.min(), ci_hi=g.n_removed.max()))
        for k in MODELS:
            v = g[f"spec_removed_{k}"].dropna()
            out.append(dict(dataset=name, pair=pair, quantity=f"spec_on_removed_{k}", version="excl", J=len(v),
                            mean=v.mean(), ci_lo=np.nan, ci_hi=np.nan))
        out.append(dict(dataset=name, pair=pair, quantity="splits_with_removal", version="excl", J=len(g),
                        mean=float((g.n_removed > 0).sum()), ci_lo=np.nan, ci_hi=np.nan))
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "threew_unlabelled_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.pivot_table(index=["dataset", "pair", "quantity"], columns="version", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
