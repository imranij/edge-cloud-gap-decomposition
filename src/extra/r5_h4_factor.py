"""POST-PLAN, evaluation only. Refits no model.

The overlap-corrected two-stage H4 of src/extra/overlap_corrected_tests.py inflates each per-dataset OLS slope SE
over the J splits by sqrt(1 + J n_test/n_train), the Nadeau-Bengio ratio for the variance of a MEAN of overlapping
split estimates. That factor is not derived for a slope (a contrast over splits). This script reports the H4
two-stage estimates with and without the factor, so that the reader sees both bounds:
  with the factor     conservative (the pilot's two-stage H4 rejected in 0 of 105 S0 replicates);
  without the factor  treats the J splits of a dataset as independent (ignores the overlap).
Neither is a correction derived for a slope; the paper says so.

For each of (pre-specified H4, declared replication H4, corrected H4 contrast Cap(R6)-Cap(R0) on validation
rho_pos) it gives the pooled HKSJ estimate, CI, one-sided p, HKSJ q and DL tau^2, the minimum detectable effect at
80% power (same rule as src/extra/r4_power.py), and the Holm p over the five primary tests with the two-stage H2 and
H4 p-values in place of the mixed-model ones (H1, H3, H5 as run).
It also writes the pilot facts that bear on the factor (results_sens/simulation/rates.csv, snapshot of 13:22 local
time at 104/105 replicates): the empirical size of the pilot's two-stage corrected H4 in S0 with its exact one-sided
95% upper bound, the MixedLM version of the same contrast, and the plan's H2 MixedLM rate in S0.

Inputs : results/extra/h4_cells_extra.csv, results/summary/{primary,sizes}.csv, results/extra/overlap_corrected_tests.csv,
         results_sens/deconf/summary/{h4_data,sizes,primary}.csv, results_sens/simulation/rates.csv
Outputs: results/extra/r5_h4_factor.csv, results/extra/r5_h4_pilot_size.csv
    python src/extra/r5_h4_factor.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as X
import overlap_corrected_tests as OC

RS = X.ROOT / "results_sens"
ALPHA, POWER, DF = 0.05, 0.80, 3


def h4(cells, xcol, ycol, sizes, size_key, factor):
    """Same code path as OC.h4_two_stage (pairs averaged per split), with the variance factor switchable."""
    means, ses = [], []
    for ds in X.DATASETS:
        key = size_key(ds)
        g = cells[cells.dataset == key].dropna(subset=[xcol, ycol])
        g = g.groupby("split", as_index=False)[[xcol, ycol]].mean()
        o = X.ols_slope(g[xcol], g[ycol])
        s = sizes[(sizes.dataset == key) & (sizes.pair == "A")].iloc[0]
        J = g.split.nunique()
        f = np.sqrt(1.0 + J * s.n_test / s.n_train) if factor else 1.0
        means.append(o["slope"]); ses.append(o["se"] * f)
    p = X.S.pool_hksj(means, ses)
    p["p_one_sided"] = float(stats.t.cdf(p["mean"] / p["se"], p["df"]))
    p["mde_80"] = (stats.t.ppf(1 - ALPHA, DF) + stats.t.ppf(POWER, DF)) * p["se"]
    p["per_dataset_factor"] = ";".join(
        f"{np.sqrt(1.0 + 20 * r.n_test / r.n_train):.2f}" for ds in X.DATASETS
        for r in [sizes[(sizes.dataset == size_key(ds)) & (sizes.pair == 'A')].iloc[0]]) if factor else "1"
    return p


def main():
    cells = pd.read_csv(X.OUT / "h4_cells_extra.csv")
    sz = X.sizes()
    hd = pd.read_csv(RS / "deconf" / "summary" / "h4_data.csv").rename(columns={"dcap": "dcap_R6"})
    dsz = pd.read_csv(RS / "deconf" / "summary" / "sizes.csv")
    oc = pd.read_csv(X.OUT / "overlap_corrected_tests.csv")
    mp = pd.read_csv(X.SUM / "primary.csv").set_index("hypothesis")
    dp = pd.read_csv(RS / "deconf" / "summary" / "primary.csv").set_index("hypothesis")

    def h2ts(run):
        return float(oc[(oc.run == run) & (oc.hypothesis == "H2")
                        & (oc.estimator == "two-stage pooled (HKSJ)")].p_one_sided.iloc[0])

    specs = [
        ("pre-specified H4", "Cap(R6)-Cap(R1) on test rho_pos", cells, "rho_pos", "dcap_R6", sz, lambda d: d,
         mp, h2ts("pre-specified")),
        ("deconfounded replication H4", "Cap(R6)-Cap(R1) on test rho_pos", hd, "rho_pos", "dcap_R6", dsz,
         lambda d: OC.DECL[d][1], dp, h2ts("deconfounded replication")),
        ("corrected H4 (proposed)", "Cap(R6)-Cap(R0) on validation rho_pos", cells, "rho_pos_val",
         "cap6_minus_cap0", sz, lambda d: d, None, None),
    ]
    rows = []
    for name, lhs, c, xc, yc, s, key, prim, h2p in specs:
        for factor in (True, False):
            p = h4(c, xc, yc, s, key, factor)
            holm = np.nan
            if prim is not None:
                ps = [prim.loc["H1", "p_raw"], h2p, prim.loc["H3", "p_raw"], p["p_one_sided"], prim.loc["H5", "p_raw"]]
                holm = X.S.holm(ps)[3]
            rows.append(dict(test=name, lhs=lhs,
                             variance_factor="sqrt(1 + J n_test/n_train)" if factor else "none (splits independent)",
                             estimate=p["mean"], se=p["se"], ci_lo=p["ci_lo"], ci_hi=p["ci_hi"],
                             p_one_sided=p["p_one_sided"], p_holm_two_stage_set=holm, hksj_q=p["q"],
                             tau2=p["tau2"], mde_80=p["mde_80"], factor_per_dataset=p["per_dataset_factor"]))
    out = pd.DataFrame(rows)
    # the with-factor rows must reproduce the published values exactly
    pub = oc[(oc.run == "pre-specified") & (oc.hypothesis == "H4") & (oc.estimator == "two-stage pooled (HKSJ)")].iloc[0]
    w = out[(out.test == "pre-specified H4") & out.variance_factor.str.startswith("sqrt")].iloc[0]
    assert np.isclose(w.estimate, pub.estimate) and np.isclose(w.ci_lo, pub.ci_lo) and np.isclose(w.p_holm_two_stage_set, pub.p_holm)
    c4 = pd.read_csv(X.OUT / "corrected_h4_val.csv")
    c4p = c4[(c4.run == "corrected H4 (post-plan)") & (c4.estimator == "two-stage pooled (HKSJ)")].iloc[0]
    w = out[(out.test == "corrected H4 (proposed)") & out.variance_factor.str.startswith("sqrt")].iloc[0]
    assert np.isclose(w.estimate, c4p.estimate) and np.isclose(w.ci_hi, c4p.ci_hi)
    out.to_csv(X.OUT / "r5_h4_factor.csv", index=False)

    # pilot facts (snapshot as stored; later replicates are not used)
    rt = pd.read_csv(RS / "simulation" / "rates.csv")
    keep = {"h4_cor_2stage_supported": "corrected H4, two-stage with the factor: one-sided p < 0.05",
            "h4_cor_mixed_p_lt05": "corrected H4, same contrast, MixedLM: one-sided p < 0.05",
            "h4_orig_supported": "plan's H4 (MixedLM): SUPPORTED",
            "h4_origR0_supported": "plan's H4 with R0 in place of R6: SUPPORTED",
            "h2_orig_supported": "plan's H2 (MixedLM): SUPPORTED",
            "h2_orig_2stage_reject": "plan's H2 contrast, two-stage: one-sided p < 0.05"}
    pr = rt[rt.key.isin(keep)][["key", "scenario", "rate", "mcse", "n_rep"]].copy()
    pr["what"] = pr.key.map(keep)
    k = (pr.rate * pr.n_rep).round().astype(int)
    pr["n_reject"] = k
    # exact (Clopper-Pearson) one-sided 95% upper bound on the rate
    pr["upper95_one_sided"] = [float(stats.beta.ppf(0.95, kk + 1, n - kk)) if kk < n else 1.0
                               for kk, n in zip(k, pr.n_rep)]
    pr.to_csv(X.OUT / "r5_h4_pilot_size.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["lhs"]).round(4).to_string())
    print(pr.round(4).to_string())


if __name__ == "__main__":
    main()
