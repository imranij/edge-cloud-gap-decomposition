"""What the pre-specified H4 test can and cannot show.
Post-plan analysis, not pre-specified. No model is refitted.

    python src/extra/h4_nulls.py

Outputs (results/extra/):
  h4_null_slopes.csv      pooled MixedLM slopes on rho_pos (pairs A and B, h2_q = 1) of
                          Cap(R6)-Cap(R1) (the plan's test), Cap(R0)-Cap(R1), Cap(R9)-Cap(R1),
                          Cap(R1), Cap(R6), and Cap(R6)-Cap(R0) (= difference between the R6 slope
                          and the R0-null slope, a paired test on the same cells)
  h4_per_dataset.csv      per-dataset OLS slopes of each left-hand side
  h4_rho_variants.csv     the plan's model with rho computed on all test windows and within the
                          test negatives instead of within the test positives
  h4_validation_rho.csv   rho_pos computed on the VALIDATION fold (stored validation probabilities
                          and thresholds, validation labels from the deterministic split
                          reconstruction) and the H4 model of test dcap on it
  h4_two_stage.csv        per-dataset OLS slope with a Nadeau-Bengio-type variance inflation
                          (1 + J n_test/n_train, a heuristic), pooled by HKSJ over datasets
  h4_cells_extra.csv      the per-cell values used above
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X
from escalation.routers import r1_confidence
from escalation.metrics import escalation_gain


def cells() -> pd.DataFrame:
    df = X.long_results()
    d = df[df.dataset.isin(X.DATASETS) & df.pair.isin(["A", "B"]) & (df.h2_q == 1.0)]
    cap = d[d.quantity == "capture"].pivot_table(index=["dataset", "pair", "split"], columns="router",
                                                 values="value")
    cap.columns = [f"cap_{c}" for c in cap.columns]
    rho = d[d.quantity == "rho_pos"].set_index(["dataset", "pair", "split"])["value"].rename("rho_pos")
    out = pd.concat([cap[["cap_R0", "cap_R1", "cap_R6", "cap_R9"]], rho], axis=1).reset_index()
    rows = []
    for r in out.itertuples(index=False):
        z = X.arrays(r.dataset, r.pair, r.split)
        y, g, s1 = z["y_test"].astype(int), z["g_dep"], z["score_R1"]
        rho_all = X.spearman(s1, g)
        rho_neg = X.spearman(s1[y == 0], g[y == 0])
        # validation version: stored validation probabilities of the deployed pair + thresholds
        sp = X.splits(r.dataset)[int(r.split)]
        yv = X.dataset(r.dataset)["y"][sp["val"]].astype(int)
        pe, pc = z["pval_ef_ev"], z["pval_cf_cv"]
        if len(pe) != len(yv):
            raise SystemExit(f"validation reconstruction mismatch {r.dataset} {r.pair} {r.split}")
        te, tc = float(z["t_ef_ev"]), float(z["t_cf_cv"])
        gv = escalation_gain(yv, (pe >= te).astype(int), (pc >= tc).astype(int))
        s1v = r1_confidence(pe, te)
        rho_val = X.spearman(s1v[yv == 1], gv[yv == 1])
        rows.append(dict(rho_all=rho_all, rho_neg=rho_neg, rho_pos_val=rho_val,
                         n_val_pos=int(yv.sum()), n_val_pos_gain_nonzero=int((gv[yv == 1] != 0).sum())))
    out = pd.concat([out, pd.DataFrame(rows)], axis=1)
    out["dcap_R6"] = out.cap_R6 - out.cap_R1
    out["dcap_R0"] = out.cap_R0 - out.cap_R1
    out["dcap_R9"] = out.cap_R9 - out.cap_R1
    out["cap6_minus_cap0"] = out.cap_R6 - out.cap_R0
    return out


LHS = [("dcap_R6", "Cap(R6) - Cap(R1) [plan's H4 test]"),
       ("dcap_R0", "Cap(R0) - Cap(R1) [random router in place of R6]"),
       ("dcap_R9", "Cap(R9) - Cap(R1) [label-blind schedule in place of R6]"),
       ("cap_R1", "Cap(R1)"), ("cap_R6", "Cap(R6)"),
       ("cap6_minus_cap0", "Cap(R6) - Cap(R0) [R6 slope minus R0-null slope]")]


def main():
    c = cells()
    c.to_csv(X.OUT / "h4_cells_extra.csv", index=False)
    base = c.dropna(subset=["rho_pos"])
    rows = []
    for col, lab in LHS:
        res = X.mixed_slope(base[["dataset", col, "rho_pos"]].rename(columns={col: "y"}), "y ~ rho_pos", "rho_pos")
        from scipy.stats import norm
        res.update(lhs=col, label=lab, p_two_sided=float(2 * norm.sf(abs(res["z"]))))
        rows.append(res)
    pd.DataFrame(rows).to_csv(X.OUT / "h4_null_slopes.csv", index=False)

    pdx = []
    for ds in X.DATASETS:
        g = base[base.dataset == ds]
        for col, lab in LHS:
            r = X.ols_slope(g.rho_pos, g[col])
            pdx.append(dict(dataset=ds, lhs=col, **r, mean_rho_pos=g.rho_pos.mean(), mean_lhs=g[col].mean()))
    pd.DataFrame(pdx).to_csv(X.OUT / "h4_per_dataset.csv", index=False)

    var = []
    for rho in ("rho_pos", "rho_all", "rho_neg", "rho_pos_val"):
        for col in ("dcap_R6", "dcap_R0", "cap_R6"):
            d = c[["dataset", col, rho]].rename(columns={col: "y", rho: "x"})
            r = X.mixed_slope(d, "y ~ x", "x")
            var.append(dict(rho=rho, lhs=col, **r))
    v = pd.DataFrame(var)
    v[v.rho != "rho_pos_val"].to_csv(X.OUT / "h4_rho_variants.csv", index=False)
    val = v[v.rho == "rho_pos_val"].copy()
    ok = c.dropna(subset=["rho_pos", "rho_pos_val"])
    val["corr_val_test_rho_spearman"] = X.spearman(ok.rho_pos_val, ok.rho_pos)
    val["n_cells_val_rho_defined"] = int(c.rho_pos_val.notna().sum())
    val["median_val_pos_windows"] = float(c.n_val_pos.median())
    val.to_csv(X.OUT / "h4_validation_rho.csv", index=False)
    # per-dataset correlation of validation and test rho
    pc = []
    for ds in X.DATASETS:
        g = ok[ok.dataset == ds]
        pc.append(dict(dataset=ds, n=len(g), spearman_val_test_rho=X.spearman(g.rho_pos_val, g.rho_pos),
                       mean_rho_pos_val=g.rho_pos_val.mean(), mean_rho_pos_test=g.rho_pos.mean(),
                       ols_dcap_on_val_rho=X.ols_slope(g.rho_pos_val, g.dcap_R6)["slope"]))
    pd.DataFrame(pc).to_csv(X.OUT / "h4_validation_rho_by_dataset.csv", index=False)

    # two-stage (heuristic NB inflation of per-dataset OLS slope variance), pooled by HKSJ
    ts = []
    for col in ("dcap_R6", "dcap_R0", "cap6_minus_cap0"):
        means, ses = [], []
        for ds in X.DATASETS:
            g = base[base.dataset == ds]
            r = X.ols_slope(g.rho_pos, g[col])
            sz = X.sizes()
            s = sz[(sz.dataset == ds) & (sz.pair == "A")].iloc[0]
            J = g.split.nunique()
            infl = np.sqrt(1.0 + J * s.n_test / s.n_train)
            means.append(r["slope"]); ses.append(r["se"] * infl)
            ts.append(dict(lhs=col, dataset=ds, slope=r["slope"], se_ols=r["se"], inflation=infl,
                           se_inflated=r["se"] * infl))
        p = X.S.pool_hksj(means, ses)
        ts.append(dict(lhs=col, dataset="pooled_HKSJ", slope=p["mean"], se_inflated=p["se"], ci_lo=p["ci_lo"],
                       ci_hi=p["ci_hi"], p_two_sided=p["p"]))
    pd.DataFrame(ts).to_csv(X.OUT / "h4_two_stage.csv", index=False)
    print(pd.DataFrame(rows)[["lhs", "slope", "se", "ci_lo", "ci_hi", "n"]].to_string())
    print(pd.DataFrame(pdx).pivot(index="lhs", columns="dataset", values="slope").round(3))
    print(v.round(3).to_string())
    print(pd.DataFrame(pc).round(3).to_string())
    print(pd.DataFrame(ts).round(3).to_string())


if __name__ == "__main__":
    main()
