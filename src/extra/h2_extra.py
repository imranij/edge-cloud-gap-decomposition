"""What the pre-specified H2 slope mixes, and inference that respects the resampling.
Post-plan analysis, not pre-specified. No model is refitted.

    python src/extra/h2_extra.py

Cells: pair A, every dataset x split x H2 level (q = 0.1, 0.25, 0.5 from the --h2 runs, q = 1.0
from the main run). Outputs (results/extra/):
  h2_extra.csv        pooled MixedLM slopes (dataset random intercepts) of
                        C(R6) on log2 positive training windows (the plan's test, reproduced),
                        C(R6) - C(R1) on the same regressor (confidence-router control),
                        C(R6) on log2 kept positive UNITS (fit + validation),
                      per-dataset OLS slopes, and a two-stage estimate: per split the OLS slope over
                      the four levels, Nadeau-Bengio corrected mean over the 20 splits of a dataset,
                      HKSJ pooling over datasets; one-sided p-values (slope > 0) for the
                      MixedLM, per-dataset two-stage and pooled two-stage estimates
  h2_tiers_by_q.csv   how the deployed tiers and the gains change with q: mean edge BA, cloud BA,
                      oracle gap sum(max(g, 0)), share of windows with g > 0 and g < 0, and C of R1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as X


def cells():
    df = X.long_results()
    d = df[df.dataset.isin(X.DATASETS) & (df.pair == "A")]
    k = ["dataset", "pair", "split", "h2_q"]
    c6 = X.per_split(d, "C", "R6", name="C_R6"); c1 = X.per_split(d, "C", "R1", name="C_R1")
    base = pd.concat([c6, c1], axis=1).reset_index()
    o = X.wide(d, ["ba_ef_ev", "ba_cf_cv", "n_pos_groups_fit_kept", "n_pos_groups_val_kept"])
    npos = d[d.quantity == "C"].drop_duplicates(k).set_index(k)["n_pos_train"].rename("n_pos_train").reset_index()
    c = base.merge(o, on=k, how="left").merge(npos, on=k, how="left")
    gaps = []
    for r in c.itertuples(index=False):
        z = X.arrays(r.dataset, "A", r.split, None if r.h2_q == 1.0 else r.h2_q)
        g = z["g_dep"]
        gaps.append(dict(oracle_gap=float(np.maximum(g, 0).sum()), share_g_pos=float((g > 0).mean()),
                         share_g_neg=float((g < 0).mean()), total_gain=float(g.sum())))
    c = pd.concat([c, pd.DataFrame(gaps)], axis=1)
    c["log2_npos"] = np.log2(c.n_pos_train.clip(lower=1))
    c["log2_units"] = np.log2((c.n_pos_groups_fit_kept + c.n_pos_groups_val_kept).clip(lower=1))
    c["dC"] = c.C_R6 - c.C_R1
    return c


def two_stage(c, col, x):
    per = []
    for ds in X.DATASETS:
        sl = []
        for s, g in c[c.dataset == ds].groupby("split"):
            g = g.dropna(subset=[col, x])
            if len(g) >= 3 and np.ptp(g[x]) > 0:
                sl.append(np.polyfit(g[x], g[col], 1)[0])
        t = X.nb(np.array(sl), ds, "A")
        per.append(dict(dataset=ds, J=t["J"], mean=t["mean"], se=t["se"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"],
                        p1=float(stats.t.sf(t["mean"] / t["se"], t["J"] - 1))))
    per = pd.DataFrame(per)
    p = X.S.pool_hksj(per["mean"], per["se"])
    return per, p


def main():
    c = cells()
    c.to_csv(X.OUT / "h2_cells_extra.csv", index=False)
    rows = []
    for col, x, lab in [("C_R6", "log2_npos", "C(R6) ~ log2 positive windows [plan's H2 test]"),
                        ("dC", "log2_npos", "C(R6) - C(R1) ~ log2 positive windows"),
                        ("C_R6", "log2_units", "C(R6) ~ log2 kept positive units"),
                        ("C_R1", "log2_npos", "C(R1) ~ log2 positive windows")]:
        r = X.mixed_slope(c[["dataset", col, x]].rename(columns={col: "y", x: "x"}), "y ~ x", "x")
        rows.append(dict(model=lab, estimator="MixedLM pooled", slope=r["slope"], se=r["se"], ci_lo=r["ci_lo"],
                         ci_hi=r["ci_hi"], p_one_sided=float(stats.norm.sf(r["z"])), n=r["n"], note=r["note"]))
        for ds in X.DATASETS:
            g = c[c.dataset == ds]
            o = X.ols_slope(g[x], g[col])
            rows.append(dict(model=lab, estimator=f"OLS {ds}", slope=o["slope"], se=o["se"], ci_lo=o["ci_lo"],
                             ci_hi=o["ci_hi"], n=o["n"]))
        per, p = two_stage(c, col, x)
        for r2 in per.itertuples(index=False):
            rows.append(dict(model=lab, estimator=f"two-stage {r2.dataset} (NB over splits)", slope=r2.mean, se=r2.se,
                             ci_lo=r2.ci_lo, ci_hi=r2.ci_hi, n=r2.J, p_one_sided=r2.p1))
        rows.append(dict(model=lab, estimator="two-stage pooled (HKSJ)", slope=p["mean"], se=p["se"], ci_lo=p["ci_lo"],
                         ci_hi=p["ci_hi"], p_one_sided=float(stats.t.sf(p["mean"] / p["se"], p["df"])), n=4))
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "h2_extra.csv", index=False)
    tiers = c.groupby(["dataset", "h2_q"]).agg(J=("split", "size"), edge_ba=("ba_ef_ev", "mean"),
                                               cloud_ba=("ba_cf_cv", "mean"), oracle_gap=("oracle_gap", "mean"),
                                               share_g_pos=("share_g_pos", "mean"), share_g_neg=("share_g_neg", "mean"),
                                               C_R1=("C_R1", "mean"), C_R6=("C_R6", "mean"),
                                               kept_units=("log2_units", lambda s: float(np.mean(2 ** s)))).reset_index()
    tiers.to_csv(X.OUT / "h2_tiers_by_q.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.drop(columns=["note"]).round(4).to_string())
    print(tiers.round(3).to_string())


if __name__ == "__main__":
    main()
