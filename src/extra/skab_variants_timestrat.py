"""POST-PLAN, evaluation only. Refits no model.

The SKAB information step at matched elapsed time (balanced accuracy within 120-s bins of the time since
file start, experiment files only, bins with at least 5 windows of each class; the rule of
src/extra/skab_breakdown.py) computed for the pre-specified run and for every complete SKAB control
variant, from the stored test probabilities and thresholds of the four factorial models
(results/skab/<pair>/arrays_<j>.npz and results_sens/hydskab/<variant>/<pair>/arrays_<j>.npz). Also checks
that a variant's test folds equal the pre-specified ones (same splits, only features changed), so the
pre-specified shortcut baselines apply to it unchanged.
Also the single-loop-channel and time-in-file SKAB runs on the first 10 pre-specified splits
(results_sens/round3/skab_e_{flow,press,thermo}, cloud = drive channels plus one loop channel; skab_sc, time in
file added to both views), each beside the pre-specified run restricted to the same 10 splits.
Output: results/extra/skab_timestrat_variants.csv
    python src/extra/skab_variants_timestrat.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

MODELS = {"eE": "ef_ev", "cE": "cf_ev", "eC": "ef_cv", "cC": "cf_cv"}
BIN_S, MIN_PER_CLASS = 120, 5
RS = X.ROOT / "results_sens" / "hydskab"
VARIANTS = ["skab_filez", "skab_base", "skab_notemp", "skab_edgeP", "skab_cm_flow", "skab_cm_thermo",
            "skab_cm_pressure"]
RS3 = X.ROOT / "results_sens" / "round3"
VARIANTS10 = ["skab_e_flow", "skab_e_press", "skab_e_thermo", "skab_sc"]   # first 10 pre-specified splits


def ba(y, pred):
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def timestrat_info(z, meta):
    idx = z["test_idx"]; y = z["y_test"].astype(int)
    infile = (meta.subset.to_numpy()[idx] != "anomaly-free")
    bins = (meta.t_start_s.to_numpy()[idx] // BIN_S).astype(int)
    ok = [b for b in np.unique(bins[infile])
          if ((bins == b) & infile & (y == 0)).sum() >= MIN_PER_CLASS and ((bins == b) & infile & (y == 1)).sum() >= MIN_PER_CLASS]
    if not ok:
        return np.nan, {}
    out = {}
    for s, k in MODELS.items():
        pred = (z[f"p_{k}"] >= float(z[f"t_{k}"])).astype(int)
        out[s] = float(np.mean([ba(y[(bins == b) & infile], pred[(bins == b) & infile]) for b in ok]))
    return 0.5 * ((out["eC"] - out["eE"]) + (out["cC"] - out["cE"])), out


def main():
    meta = X.dataset("skab")["meta"]
    rows = []
    jobs = [(v, 20) for v in ["pre-specified"] + VARIANTS] + [("pre-specified, splits 0-9", 10)] + \
        [(v, 10) for v in VARIANTS10]
    for v, jn in jobs:
        for pair in ("A", "B"):
            if v.startswith("pre-specified"):
                d = X.RES / "skab" / pair
            else:
                d = (RS3 / v / pair) if v in VARIANTS10 else (RS / v / pair)
            fs = sorted(f for f in d.glob("arrays_*.npz") if "_q" not in f.stem) if d.exists() else []
            fs = [f for f in fs if int(f.stem.split("_")[1]) < jn]
            if len(fs) < jn:
                continue
            vals, eE, cC, same = [], [], [], True
            ntr = nte = None
            for f in fs:
                j = int(f.stem.split("_")[1])
                z = np.load(f)
                same &= bool(np.array_equal(z["test_idx"], X.arrays("skab", pair, j)["test_idx"]))
                info, m = timestrat_info(z, meta)
                vals.append(info); eE.append(m.get("eE", np.nan)); cC.append(m.get("cC", np.nan))
            sz = X.sizes(); s = sz[(sz.dataset == "skab") & (sz.pair == pair)].iloc[0]
            t = X.S.nb_test(np.array(vals), s.n_train, s.n_test)
            rows.append(dict(variant=v, pair=pair, J=t["J"], timestrat_info=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"],
                             edge_timestrat=np.nanmean(eE), cloud_timestrat=np.nanmean(cC), same_test_folds_as_main=same))
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "skab_timestrat_variants.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
