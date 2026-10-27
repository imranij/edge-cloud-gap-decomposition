"""POST-PLAN, evaluation only. Refits no model.

Per-fold diagnostics of the leave-one-level-out and leave-session-out controls, read from the stored
per-split outputs results_sens/hydskab/{skab_lso,hyd_cooler,hyd_accum}/<pair>/{split_*.parquet,arrays_*.npz}
and the variant meta data/processed_sens/<variant>/meta.parquet:

  n_fit, n_val, n_test, positive groups in the fit part, the held-out level or session,
  per-fold balanced accuracy of the deployed edge (ef_ev) and the cloud (cf_cv),
  predicted-positive rate on the test fold (p >= stored threshold),
  threshold-free ROC AUC of the edge and cloud probabilities, and C(R1), C(R6).

For SKAB it also gives, per held-out session, the recall of every disturbance type present and the
specificity on the held-out normal windows (anomaly-free recording and experiment-file normals
separately). Outputs:
  results/extra/holdout_diagnostics.csv        one row per variant x pair x fold
  results/extra/holdout_skab_by_type.csv       SKAB per held-out session x disturbance type
  results/extra/skab_session_composition.csv   SKAB windows by session x disturbance family (meta only)

The same diagnostics for the REDESIGNED hold-outs, run
with src/sens/designs/run_custom_holdout.py (fit + validation = every non-held-out level or
session, validation carved out by file or run): results_sens/holdout2/{skab_lso2,hyd_cooler2,hyd_accum2}/<pair>/,
with the parent dataset's meta (data/processed/<parent>/meta.parquet). Rows carry design = "plan" or
"redesign (R1)"; the balanced accuracy of all four factorial models and the Shapley information step are added.

The same diagnostics for the redesigned
session hold-out on the causal-offset features, design skab_lso2_base (results_sens/holdout2/skab_lso2_base/<pair>/,
meta data/processed_sens/skab_base/meta.parquet, design = "redesign, causal offset (R1 r4)"). The per-fold time-in-file
shortcut re-scored on the same folds (results_sens/shortcuts_controls/per_split.csv / summary.csv, variant
skab_lso2_base, shortcut a_t_start_tree3) is attached as shortcut_t_start_ba.

    python src/extra/holdout_diagnostics.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RS = ROOT / "results_sens" / "hydskab"
OUT = ROOT / "results" / "extra"
VARIANTS = ["skab_lso", "hyd_cooler", "hyd_accum"]
RS2 = ROOT / "results_sens" / "holdout2"
VARIANTS2 = {"skab_lso2": "skab", "hyd_cooler2": "hydraulic", "hyd_accum2": "hydraulic"}
# redesigned session hold-out on the causal-offset features, sens dataset root
VARIANTS3 = {"skab_lso2_base": "skab_base"}
SC = ROOT / "results_sens" / "shortcuts_controls"
SKAB_V = ("skab_lso", "skab_lso2", "skab_lso2_base")


def q(df, name, router=""):
    r = df[(df.quantity == name) & (df.router.fillna("") == router) & df.budget.isna()]
    return float(r.value.iloc[0]) if len(r) else np.nan


def auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def attach_shortcut(out):
    """Per-fold test BA of the time-in-file shortcut (a_t_start_tree3, a depth-3 tree on the window start time)
    on the same hold-out folds, from results_sens/shortcuts_controls/per_split.csv. The shortcut uses neither the
    edge nor the cloud view, so the value is the same for pairs A and B."""
    out["shortcut_t_start_ba"] = np.nan
    f = SC / "per_split.csv"
    if not f.exists():
        return out
    sc = pd.read_csv(f)
    sc = sc[sc.shortcut == "a_t_start_tree3"].drop_duplicates(["variant", "split"])
    for _, r in sc.iterrows():
        m = (out.variant == r.variant) & (out.split == int(r.split))
        if m.any():
            assert (out.loc[m, "n_test"] == int(r.n_test)).all(), (r.variant, r.split)
            out.loc[m, "shortcut_t_start_ba"] = float(r.ba_test)
    return out


def main():
    rows, by_type = [], []
    jobs = [(v, RS / v, ROOT / "data" / "processed_sens" / v / "meta.parquet", "plan") for v in VARIANTS]
    jobs += [(v, RS2 / v, ROOT / "data" / "processed" / par / "meta.parquet", "redesign (R1)")
             for v, par in VARIANTS2.items()]
    jobs += [(v, RS2 / v, ROOT / "data" / "processed_sens" / par / "meta.parquet", "redesign, causal offset (R1 r4)")
             for v, par in VARIANTS3.items()]
    for v, vdir, meta_path, design in jobs:
        meta = pd.read_parquet(meta_path)
        for pair in ("A", "B"):
            d = vdir / pair
            for f in sorted(d.glob("split_*.parquet")) if d.exists() else []:
                if "_q" in f.stem:
                    continue
                j = f.stem.split("_")[1]
                s = pd.read_parquet(f)
                s = s[s.h2_q == 1.0]
                z = np.load(d / f"arrays_{j}.npz")
                y = z["y_test"].astype(int)
                if "held_out" in s.columns:
                    held = [str(s.held_out.iloc[0])]
                    fit_groups = []
                else:
                    held = sorted(set(meta.group_id.to_numpy()[z["test_idx"]]))
                    fit_groups = sorted(set(meta.group_id) - set(held))
                pe, pc = z["p_ef_ev"], z["p_cf_cv"]
                te, tc = float(z["t_ef_ev"]), float(z["t_cf_cv"])
                rows.append(dict(
                    variant=v, design=design, pair=pair, split=int(j), held_out="+".join(held),
                    n_train=int(s.n_train.iloc[0]), n_fit=q(s, "n_fit"), n_val=q(s, "n_val"), n_test=len(y),
                    n_pos_groups_fit=q(s, "n_pos_groups_fit"), n_pos_fit=q(s, "n_pos_fit"),
                    n_pos_test=int(y.sum()), prev_test=float(y.mean()),
                    ba_edge=q(s, "ba_ef_ev"), ba_cloud=q(s, "ba_cf_cv"),
                    ba_upgraded_edge=q(s, "ba_cf_ev"), ba_gateway=q(s, "ba_ef_cv"),
                    ppr_edge=float((pe >= te).mean()), ppr_cloud=float((pc >= tc).mean()),
                    auc_edge=auc(y, pe), auc_cloud=auc(y, pc),
                    auc_gateway=auc(y, z["p_ef_cv"]), auc_upgraded_edge=auc(y, z["p_cf_ev"]),
                    C_R1=q(s, "C", "R1"), C_R6=q(s, "C", "R6"), shapley_info=q(s, "shapley_info"),
                    fit_groups_other="+".join(fit_groups)))
                if v in SKAB_V:
                    mt = meta.iloc[z["test_idx"]].reset_index(drop=True)
                    ye, yc = (pe >= te).astype(int), (pc >= tc).astype(int)
                    af = (mt.subset == "anomaly-free").to_numpy()
                    neg_af, neg_ex = af & (y == 0), ~af & (y == 0)
                    rows[-1].update(
                        spec_edge_anomaly_free=1 - ye[neg_af].mean() if neg_af.any() else np.nan,
                        spec_cloud_anomaly_free=1 - yc[neg_af].mean() if neg_af.any() else np.nan,
                        spec_edge_exp_normals=1 - ye[neg_ex].mean() if neg_ex.any() else np.nan,
                        spec_cloud_exp_normals=1 - yc[neg_ex].mean() if neg_ex.any() else np.nan)
                    for t, g in mt.groupby("exp_fault_type"):
                        ix = g.index.to_numpy()
                        pos, neg = ix[y[ix] == 1], ix[y[ix] == 0]
                        by_type.append(dict(variant=v, design=design, pair=pair, split=int(j), held_out=held[0],
                                            family=t,
                                            subset="+".join(sorted(g.subset.unique())),
                                            n_pos=len(pos), n_neg=len(neg),
                                            recall_edge=ye[pos].mean() if len(pos) else np.nan,
                                            recall_cloud=yc[pos].mean() if len(pos) else np.nan,
                                            spec_edge=1 - ye[neg].mean() if len(neg) else np.nan,
                                            spec_cloud=1 - yc[neg].mean() if len(neg) else np.nan))
    out = pd.DataFrame(rows)
    out = attach_shortcut(out)
    out.to_csv(OUT / "holdout_diagnostics.csv", index=False)
    pd.DataFrame(by_type).to_csv(OUT / "holdout_skab_by_type.csv", index=False)
    m = pd.read_parquet(ROOT / "data" / "processed" / "skab" / "meta.parquet")
    comp = (m.groupby(["session", "exp_fault_type"])
            .agg(windows=("label", "size"), fault_windows=("label", "sum"), files=("file", "nunique"))
            .reset_index())
    comp.to_csv(OUT / "skab_session_composition.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["variant", "pair", "split", "held_out", "n_fit", "n_val", "n_test", "n_pos_groups_fit", "ba_edge",
               "ba_cloud", "ppr_edge", "ppr_cloud", "auc_edge", "auc_cloud", "C_R1"]].round(3).to_string())
    print(pd.DataFrame(by_type).round(3).to_string())
    print(comp.to_string())


if __name__ == "__main__":
    main()
