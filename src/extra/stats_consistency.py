"""Consistent statistical reporting.
Post-plan analysis, not pre-specified. No model is refitted; only the inferential
summaries of stored per-cell values are recomputed.

    python src/extra/stats_consistency.py

1. H5 permutation p-value. The pre-specified run (src/analyze.py, seed 9000, 10,000 within-dataset
   permutations) gave p = 0.0027 and the post-plan breakdown (src/extra/h5_audit_breakdown.py,
   seed 9100, same procedure) gave 0.0038. We report the Monte Carlo standard error of each,
   sqrt(p (1 - p) / N), and a longer run of 200,000 permutations (seed 9300) as a precise value.
   Output: h5_pvalue_mc.csv
2. H4 under both estimators. For every left-hand side of the H4 table: the plan's mixed model
   (dataset random intercepts, REML) and the two-stage estimate (per-dataset OLS slope, standard
   error inflated by sqrt(1 + J n_test / n_train), HKSJ pooling), each with a one-sided p-value
   for a negative slope (the direction of the plan's H4). Output: h4_both_estimators.csv
3. Convergence. Every mixed model fitted in src/analyze.py and src/extra/ is refitted on its
   stored cells (same data, same estimator) to record whether the optimiser converged, and is
   checked against an OLS with dataset fixed effects and standard errors clustered by
   dataset-by-split block. Output: mixed_model_convergence.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import common as X
from h5_audit_breakdown import cells as h5cells, perm_within


def h5_mc():
    pre = pd.read_csv(X.SUM / "h5.csv").iloc[0]
    br = pd.read_csv(X.OUT / "h5_breakdown.csv").set_index("subset").loc["all cells (pre-specified)"]
    c = h5cells()
    _, p_long = perm_within(c.call.to_numpy(int), c.outcome.to_numpy(int), c.dataset.to_numpy(), n=200000, seed=9300)
    rows = []
    for lab, p, n in (("pre-specified run (analyze.py, seed 9000)", float(pre.p_perm), 10000),
                      ("post-plan breakdown (seed 9100)", float(br.p_perm_within_dataset), 10000),
                      ("long run (seed 9300)", p_long, 200000)):
        rows.append(dict(run=lab, n_perm=n, p=p, mc_se=float(np.sqrt(p * (1 - p) / n))))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "h5_pvalue_mc.csv", index=False)
    return o


def two_stage(base, col, x="rho_pos"):
    means, ses, per = [], [], {}
    sz = X.sizes()
    for ds in X.DATASETS:
        g = base[base.dataset == ds].dropna(subset=[col, x])
        r = X.ols_slope(g[x], g[col])
        s = sz[(sz.dataset == ds) & (sz.pair == "A")].iloc[0]
        J = g.split.nunique()
        infl = np.sqrt(1.0 + J * s.n_test / s.n_train)
        means.append(r["slope"]); ses.append(r["se"] * infl); per[ds] = r["slope"]
    p = X.S.pool_hksj(means, ses)
    p["p_one_sided_neg"] = float(stats.t.cdf(p["mean"] / p["se"], p["df"]))
    return p, per


def h4_both():
    c = pd.read_csv(X.OUT / "h4_cells_extra.csv")
    base = c.dropna(subset=["rho_pos"])
    lhs = [("dcap_R6", "Cap(R6)-Cap(R1), plan's test"), ("dcap_R0", "Cap(R0)-Cap(R1), random router"),
           ("dcap_R9", "Cap(R9)-Cap(R1), label-blind schedule"), ("cap_R1", "Cap(R1)"), ("cap_R6", "Cap(R6)"),
           ("cap6_minus_cap0", "Cap(R6)-Cap(R0)")]
    rows = []
    for col, lab in lhs:
        m = X.mixed_slope(base[["dataset", col, "rho_pos"]].rename(columns={col: "y"}), "y ~ rho_pos", "rho_pos")
        t, per = two_stage(base, col)
        rows.append(dict(lhs=col, label=lab, mixed_slope=m["slope"], mixed_lo=m["ci_lo"], mixed_hi=m["ci_hi"],
                         mixed_p_one_sided_neg=float(stats.norm.cdf(m["z"])), mixed_note=m["note"],
                         two_stage_slope=t["mean"], two_stage_lo=t["ci_lo"], two_stage_hi=t["ci_hi"],
                         two_stage_p_one_sided_neg=t["p_one_sided_neg"],
                         **{f"ols_{d}": v for d, v in per.items()}))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "h4_both_estimators.csv", index=False)
    return o


def fe_ols(data, ycol, xcol):
    import statsmodels.formula.api as smf
    d = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[ycol, xcol]).copy()
    d["block"] = d.dataset.astype(str) + "|" + d.split.astype(str)
    d = d.rename(columns={ycol: "y", xcol: "x"})
    fit = smf.ols("y ~ x + C(dataset)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.block)[0]})
    lo, hi = fit.conf_int().loc["x"]
    return dict(fe_slope=float(fit.params["x"]), fe_se=float(fit.bse["x"]), fe_lo=float(lo), fe_hi=float(hi),
                fe_n_clusters=int(d.block.nunique()))


def converged(note: str) -> str:
    n = (note or "").lower()
    if "failed to converge" in n or "singular" in n or "optimization failed" in n:
        return "not converged"
    if "boundary" in n:
        return "converged, variance on boundary"
    return "converged"


def convergence():
    h2 = pd.read_csv(X.OUT / "h2_cells_extra.csv")
    h4 = pd.read_csv(X.OUT / "h4_cells_extra.csv")
    fits = [("H2 plan's test (analyze.py)", h2, "C_R6", "log2_npos", None),
            ("H2 C(R6)-C(R1)", h2, "dC", "log2_npos", None),
            ("H2 on log2 kept units", h2, "C_R6", "log2_units", None),
            ("H2 C(R1)", h2, "C_R1", "log2_npos", None)]
    for col, lab in (("dcap_R6", "H4 plan's test (analyze.py)"), ("dcap_R0", "H4 R0 null"), ("dcap_R9", "H4 R9 null"),
                     ("cap_R1", "H4 Cap(R1)"), ("cap_R6", "H4 Cap(R6)"), ("cap6_minus_cap0", "H4 Cap(R6)-Cap(R0)")):
        fits.append((lab, h4.dropna(subset=["rho_pos"]), col, "rho_pos", None))
    for rho in ("rho_all", "rho_neg", "rho_pos_val"):
        for col in ("dcap_R6", "dcap_R0", "cap_R6"):
            fits.append((f"H4 {col} on {rho}", h4, col, rho, None))
    for ex in X.DATASETS:
        keep = [d for d in X.DATASETS if d != ex]
        fits.append((f"LODO without {ex}: H2", h2, "C_R6", "log2_npos", keep))
        fits.append((f"LODO without {ex}: H4", h4.dropna(subset=["rho_pos"]), "dcap_R6", "rho_pos", keep))
    rows = []
    for lab, data, y, x, keep in fits:
        d = data if keep is None else data[data.dataset.isin(keep)]
        m = X.mixed_slope(d[["dataset", y, x]].rename(columns={y: "yy", x: "xx"}), "yy ~ xx", "xx")
        r = dict(fit=lab, n=m["n"], estimator=m["model"], mixed_slope=m["slope"], mixed_lo=m["ci_lo"],
                 mixed_hi=m["ci_hi"], status=converged(m["note"]), note=m["note"])
        r.update(fe_ols(d, y, x))
        rows.append(r)
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "mixed_model_convergence.csv", index=False)
    return o


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    print(h5_mc().to_string())
    print(h4_both().drop(columns=["mixed_note"]).round(4).to_string())
    print(convergence().drop(columns=["note"]).round(4).to_string())


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
