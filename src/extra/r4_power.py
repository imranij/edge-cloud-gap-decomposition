"""POST-PLAN, evaluation only. Refits no model.

1. Minimum detectable effect (MDE) at 80% power, one-sided alpha = 0.05, of the overlap-corrected and proposed
   corrected designs at the realised design (J splits, four datasets, the per-dataset spread of the real data):
     two-stage HKSJ estimates (3 df): MDE = (t_{0.95,3} + t_{0.80,3}) x SE, with SE recovered from the reported
     95% CI (SE = half-width / t_{0.975,3}); this is the effect that a test with the realised standard error
     detects with 80% power if the SE stays the same under the alternative.
       H2, two-stage (pre-specified run and replication)   results/extra/overlap_corrected_tests.csv
       H2, fixed tiers (proposed)                          results_sens/routers/p3_slopes.csv
       H4, two-stage, pairs aggregated per split           results/extra/overlap_corrected_tests.csv
       H4, corrected (Cap(R6)-Cap(R0) on validation rho)    results/extra/corrected_h4_val.csv
     corrected audit (within-class null, all 400 cells, undefined as NO-GO): the BA of the GO calls is tested by
     permuting calls within datasets (or within dataset x pair x level blocks), 10,000 permutations as in
     src/extra/audit_within_class.py. MDE = 95th percentile of the permutation null + z_{0.80} x its SD (normal
     approximation, SD unchanged under the alternative), for both permutation schemes.
2. Pilot rates of the simulation (results_sens/simulation/rates.csv; replicate counts and Monte Carlo SEs as
   stored; the H5 rows' SEs there come from a replicate bootstrap, src/sens/simulation/summarize.py boot_stats):
   the plan's H4 SUPPORTED rate, the pilot's corrected two-stage H4 rejection rate and its mean slope per
   scenario, and the plan's H2 SUPPORTED rate.
Outputs: results/extra/r4_mde.csv, results/extra/r4_pilot_rates.csv
    python src/extra/r4_power.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as X

ALPHA, POWER, DF = 0.05, 0.80, 3
N_PERM = 10_000


def mde_from_ci(est, lo, hi, df=DF):
    se = (hi - lo) / 2 / stats.t.ppf(0.975, df)
    return se, (stats.t.ppf(1 - ALPHA, df) + stats.t.ppf(POWER, df)) * se


def ba_stats(call, out):
    tp = ((call == 1) & (out == 1)).sum(-1); fp = ((call == 1) & (out == 0)).sum(-1)
    fn = ((call == 0) & (out == 1)).sum(-1); tn = ((call == 0) & (out == 0)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def perm_null(call, out, blocks, seed):
    rng = np.random.default_rng(seed)
    P = np.empty((N_PERM, len(call)), dtype=np.int8)
    for b in np.unique(blocks):
        ix = np.flatnonzero(blocks == b)
        P[:, ix] = call[ix][rng.permuted(np.tile(np.arange(len(ix)), (N_PERM, 1)), axis=1)]
    return ba_stats(P, out[None, :])


def main():
    rows = []
    oc = pd.read_csv(X.OUT / "overlap_corrected_tests.csv")
    for run in ("pre-specified", "deconfounded replication"):
        for h in ("H2", "H4"):
            r = oc[(oc.run == run) & (oc.hypothesis == h) & (oc.estimator == "two-stage pooled (HKSJ)")].iloc[0]
            se, mde = mde_from_ci(r.estimate, r.ci_lo, r.ci_hi)
            rows.append(dict(design=f"{h} two-stage ({run})", estimate=r.estimate, ci_lo=r.ci_lo, ci_hi=r.ci_hi, se=se,
                             mde_80=mde, direction="greater" if h == "H2" else "less"))
    p3 = pd.read_csv(X.ROOT / "results_sens" / "routers" / "p3_slopes.csv")
    ft = p3[(p3.dataset == "POOLED (HKSJ)") & p3.design.str.startswith("fixed tiers")].iloc[0]
    se, mde = mde_from_ci(ft.two_stage_mean, ft.ci_lo, ft.ci_hi)
    rows.append(dict(design="H2 fixed tiers, two-stage (proposed)", estimate=ft.two_stage_mean, ci_lo=ft.ci_lo,
                     ci_hi=ft.ci_hi, se=se, se_stored=ft.se, mde_80=mde, direction="greater"))
    c4 = pd.read_csv(X.OUT / "corrected_h4_val.csv")
    r = c4[(c4.run == "corrected H4 (post-plan)") & (c4.estimator == "two-stage pooled (HKSJ)")].iloc[0]
    se, mde = mde_from_ci(r.estimate, r.ci_lo, r.ci_hi)
    rows.append(dict(design="H4 corrected, Cap(R6)-Cap(R0) on validation rho (proposed)", estimate=r.estimate,
                     ci_lo=r.ci_lo, ci_hi=r.ci_hi, se=se, mde_80=mde, direction="less"))
    # corrected audit
    d = pd.read_csv(X.ROOT / "results_sens" / "routers_wc" / "p1w_cells.csv")
    call = d.go_cor.to_numpy().astype(int)       # undefined cells carry go_cor = 0 (NO-GO), as in Table 6
    out = d.out_capture.to_numpy().astype(int)
    obs = float(ba_stats(call, out))
    for scheme, blocks, seed in (("within datasets", d.dataset.to_numpy(), 9600),
                                 ("within dataset x pair x level", d.block.to_numpy(), 9601)):
        nb = perm_null(call, out, blocks, seed)
        q95, sd = float(np.nanquantile(nb, 0.95)), float(np.nanstd(nb, ddof=1))
        rows.append(dict(design=f"audit, within-class null, BA of calls (permutation {scheme})", estimate=obs,
                         null_mean=float(np.nanmean(nb)), null_q95=q95, null_sd=sd,
                         mde_80=q95 + stats.norm.ppf(POWER) * sd, direction="greater",
                         p_perm=float((1 + (nb >= obs - 1e-12).sum()) / (N_PERM + 1))))
    m = pd.DataFrame(rows)
    m.to_csv(X.OUT / "r4_mde.csv", index=False)

    rt = pd.read_csv(X.ROOT / "results_sens" / "simulation" / "rates.csv")
    keys = ["h4_orig_supported", "h4_origR0_supported", "h4_cor_2stage_supported", "h4_cor_2stage_slope",
            "h2_orig_supported", "h1_ds_cover"]
    p = rt[rt.key.isin(keys)][["key", "test", "design", "quantity", "scenario", "rate", "mcse", "n_rep"]]
    p.to_csv(X.OUT / "r4_pilot_rates.csv", index=False)
    pd.set_option("display.width", 250)
    print(m.round(4).to_string())
    print(p.pivot_table(index="key", columns="scenario", values="rate").round(3).to_string())
    print(p.groupby("key").n_rep.agg(["min", "max"]))


if __name__ == "__main__":
    main()
