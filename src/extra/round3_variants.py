"""POST-PLAN, evaluation only. Refits no model.

Summarises the shortcut-augmented, single-channel and sensor-family ablation runs from their stored
per-split outputs, each beside the run it modifies on the SAME splits (test folds checked equal):

  results_sens/round3/hyd_blocks_sc    edge + cycle index (both views), block-grouped splits     ref hyd_blocks
  results_sens/round3/hyd_blocks_sc2   edge + cycle index + component settings, block-grouped    ref hyd_blocks
  results_sens/round3/skab_sc          edge + time in file (t_start_s), pre-specified splits     ref results/skab
  results_sens/round3/skab_e_{flow,press,thermo}   cloud = drive channels + ONE loop channel     ref results/skab
  results_sens/hydabl/hyd_blk_e_{flow,press,temp}  cloud = edge + one sensor family              ref hyd_blocks
  results_sens/hydabl/hyd_blk_cm_{flow,press,temp} cloud = all minus one sensor family           ref hyd_blocks
  results_sens/hydabl/hyd_blk_e_fs{1,2}            cloud = edge + ONE flow sensor (FS1 or FS2)      ref hyd_blocks
      (also compared on the same splits with hyd_blk_e_flow, the edge plus both flow sensors: info_minus_both)

Per variant x pair (h2_q = 1 rows): J, the deployed edge B_eE, the cloud B_cC, the Shapley information step,
C(R1), C(R6), each as a Nadeau-Bengio corrected mean and 95% CI (escalation.stats.nb_test with the run's mean
n_train and n_test); the reference run's information step and cloud on the same split ids; the paired
difference variant minus reference of the information step; and, for the shortcut-augmented variants, the
reference cloud minus the augmented edge (the stronger form of survival criterion (ii)).
A variant x pair is complete only with its expected number of splits (EXPECTED); otherwise complete = False and
the table generators print PENDING.
Output: results/extra/round3_variants.csv
    python src/extra/round3_variants.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

RSS = X.ROOT / "results_sens"
REF_HYD = RSS / "hydskab" / "hyd_blocks"
REF_SKAB = X.RES / "skab"
VARIANTS = {   # name: (directory, reference directory, expected J, kind)
    "hyd_blocks_sc": (RSS / "round3" / "hyd_blocks_sc", REF_HYD, 20, "shortcut-augmented"),
    "hyd_blocks_sc2": (RSS / "round3" / "hyd_blocks_sc2", REF_HYD, 20, "shortcut-augmented"),
    "skab_sc": (RSS / "round3" / "skab_sc", REF_SKAB, 10, "shortcut-augmented"),
    "skab_e_flow": (RSS / "round3" / "skab_e_flow", REF_SKAB, 10, "single loop channel"),
    "skab_e_press": (RSS / "round3" / "skab_e_press", REF_SKAB, 10, "single loop channel"),
    "skab_e_thermo": (RSS / "round3" / "skab_e_thermo", REF_SKAB, 10, "single loop channel"),
    "hyd_blk_e_flow": (RSS / "hydabl" / "hyd_blk_e_flow", REF_HYD, 10, "edge plus one family"),
    "hyd_blk_e_press": (RSS / "hydabl" / "hyd_blk_e_press", REF_HYD, 10, "edge plus one family"),
    "hyd_blk_e_temp": (RSS / "hydabl" / "hyd_blk_e_temp", REF_HYD, 10, "edge plus one family"),
    "hyd_blk_cm_flow": (RSS / "hydabl" / "hyd_blk_cm_flow", REF_HYD, 10, "all minus one family"),
    "hyd_blk_cm_press": (RSS / "hydabl" / "hyd_blk_cm_press", REF_HYD, 10, "all minus one family"),
    "hyd_blk_cm_temp": (RSS / "hydabl" / "hyd_blk_cm_temp", REF_HYD, 10, "all minus one family"),
    "hyd_blk_e_fs1": (RSS / "hydabl" / "hyd_blk_e_fs1", REF_HYD, 10, "edge plus one flow sensor"),
    "hyd_blk_e_fs2": (RSS / "hydabl" / "hyd_blk_e_fs2", REF_HYD, 10, "edge plus one flow sensor"),
}
BOTH_FLOW = RSS / "hydabl" / "hyd_blk_e_flow"   # comparator for the single-flow-sensor variants
QS = [("edge", "ba_ef_ev", ""), ("cloud", "ba_cf_cv", ""), ("info", "shapley_info", ""), ("C_R1", "C", "R1"),
      ("C_R6", "C", "R6")]


def load(d):
    fs = sorted(f for f in d.glob("split_*.parquet") if "_q" not in f.stem) if d.exists() else []
    if not fs:
        return None
    x = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    x = x[x.h2_q == 1.0].copy()
    x["router"] = x["router"].fillna("")
    return x


def per_split(x, q, r=""):
    m = (x.quantity == q) & (x.router == r) & x.budget.isna()
    return x[m].set_index("split").value.sort_index()


def nb(v, ntr, nte):
    return X.S.nb_test(np.asarray(v, float), ntr, nte)


def main():
    rows = []
    for name, (vd, rd, J_exp, kind) in VARIANTS.items():
        for pair in ("A", "B"):
            x = load(vd / pair)
            rec = dict(variant=name, kind=kind, pair=pair, J_expected=J_exp,
                       reference="hyd_blocks" if rd == REF_HYD else "skab (pre-specified)")
            if x is None:
                rows.append(dict(rec, J=0, complete=False))
                continue
            ref = load(rd / pair)
            splits = sorted(x.split.unique())
            same = all(np.array_equal(np.load(vd / pair / f"arrays_{j}.npz")["test_idx"],
                                      np.load(rd / pair / f"arrays_{j}.npz")["test_idx"]) for j in splits)
            ntr, nte = float(x.n_train.mean()), float(x.n_test.mean())
            rec.update(J=len(splits), complete=len(splits) >= J_exp, same_test_folds_as_reference=same)
            for lab, q, r in QS:
                t = nb(per_split(x, q, r), ntr, nte)
                rec[lab], rec[f"{lab}_lo"], rec[f"{lab}_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            ri = per_split(ref, "shapley_info").reindex(splits)
            rc = per_split(ref, "ba_cf_cv").reindex(splits)
            re_ = per_split(ref, "ba_ef_ev").reindex(splits)
            for lab, s in (("ref_info", ri), ("ref_cloud", rc), ("ref_edge", re_)):
                t = nb(s, ntr, nte)
                rec[lab], rec[f"{lab}_lo"], rec[f"{lab}_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            t = nb(per_split(x, "shapley_info").reindex(splits) - ri, ntr, nte)
            rec["info_minus_ref"], rec["imr_lo"], rec["imr_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            if kind == "edge plus one flow sensor":
                bf = load(BOTH_FLOW / pair)
                assert all(np.array_equal(np.load(vd / pair / f"arrays_{j}.npz")["test_idx"],
                                          np.load(BOTH_FLOW / pair / f"arrays_{j}.npz")["test_idx"]) for j in splits)
                bi = per_split(bf, "shapley_info").reindex(splits)
                t = nb(bi, ntr, nte)
                rec["both_info"], rec["both_info_lo"], rec["both_info_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
                t = nb(per_split(x, "shapley_info").reindex(splits) - bi, ntr, nte)
                rec["info_minus_both"], rec["imb_lo"], rec["imb_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            if kind == "shortcut-augmented":
                t = nb(rc - per_split(x, "ba_ef_ev").reindex(splits), ntr, nte)
                rec["ref_cloud_minus_aug_edge"], rec["rcae_lo"], rec["rcae_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
                t = nb(per_split(x, "ba_ef_ev").reindex(splits) - re_, ntr, nte)
                rec["aug_edge_minus_edge"], rec["aee_lo"], rec["aee_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "round3_variants.csv", index=False)
    pd.set_option("display.width", 320)
    cols = ["variant", "pair", "J", "complete", "same_test_folds_as_reference", "edge", "cloud", "info", "info_lo",
            "info_hi", "ref_info", "info_minus_ref", "imr_lo", "imr_hi", "C_R1", "C_R6", "ref_cloud_minus_aug_edge",
            "rcae_lo", "rcae_hi", "info_minus_both", "imb_lo", "imb_hi"]
    print(out[[c for c in cols if c in out]].round(3).to_string())


if __name__ == "__main__":
    main()
