"""POST-PLAN, evaluation only. Refits no model.

The plan's H2 and H4 tests are mixed models with dataset random intercepts only, so their standard
errors treat cells from overlapping resampled training sets (and, for H2, the four levels inside one
split) as independent. This script puts the overlap-corrected two-stage estimates beside every
pre-specified and replication verdict and Holm-adjusts the five primary p-values with the two-stage
p-values of H2 and H4 in place of the mixed-model ones. The pre-specified verdicts are not changed.

Two-stage estimators (the same as src/extra/h2_extra.py and h4_nulls.py):
  H2  per split, the OLS slope of C(R6) on log2 positive training windows over the four levels;
      Nadeau-Bengio corrected mean over the splits of a dataset; HKSJ pooling (3 df).
  H4  per dataset, the OLS slope of the left-hand side on rho, with its variance inflated by
      1 + J n_test/n_train (a heuristic Nadeau-Bengio-type factor); HKSJ pooling (3 df).
      Pairs A and B share their test folds, so their cells are not independent. The
      left-hand side and rho are now averaged over the pairs within each split before the per-dataset regression
      (one point per split; AGGREGATE_PAIRS = True). The earlier outputs, which treated the (pair, split) cells as
      independent, are kept as overlap_corrected_tests_r3cells.csv and corrected_h4_val_r3cells.csv.
Inputs:
  results/extra/h2_extra.csv, h4_two_stage.csv, h4_cells_extra.csv     (pre-specified run)
  results/summary/primary.csv, sizes.csv                               (pre-specified run)
  results_sens/{hydskab,threewseu}/<declared variant>/A/split_*.parquet (declared replication, H2)
  results_sens/deconf/summary/{primary,h4_data,sizes}.csv              (declared replication, H4)
Outputs (results/extra/):
  overlap_corrected_tests.csv   one row per run x hypothesis: estimator, estimate, CI, one-sided p,
                                Holm p over the five tests (mixed-model set and two-stage set)
  corrected_h4_val.csv          the corrected H4 contrast Cap(R6)-Cap(R0) on validation rho_pos,
                                two-stage, per dataset and pooled
    python src/extra/overlap_corrected_tests.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as X

RS = X.ROOT / "results_sens"
DECL = {"hydraulic": ("hydskab", "hyd_blocks"), "skab": ("hydskab", "skab_filez"),
        "seu": ("threewseu", "seu_dm"), "threew": ("threewseu", "threew_dm")}


def one_sided(m, se, df, direction):
    t = m / se
    return float(stats.t.sf(t, df) if direction == "greater" else stats.t.cdf(t, df))


def h2_two_stage_deconf():
    sz = pd.read_csv(RS / "deconf" / "summary" / "sizes.csv")
    means, ses, per = [], [], []
    for ds, (area, v) in DECL.items():
        fs = sorted((RS / area / v / "A").glob("split_*.parquet"))
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        d["router"] = d["router"].fillna("")
        c = d[(d.quantity == "C") & (d.router == "R6")][["split", "h2_q", "value", "n_pos_train"]]
        sl = []
        for s, g in c.groupby("split"):
            if g.h2_q.nunique() >= 3:
                sl.append(np.polyfit(np.log2(g.n_pos_train.clip(lower=1)), g.value, 1)[0])
        r = sz[(sz.dataset == v) & (sz.pair == "A")].iloc[0]
        t = X.S.nb_test(np.array(sl), r.n_train, r.n_test)
        per.append(dict(run="deconfounded replication", hypothesis="H2", estimator=f"two-stage {ds} ({v})",
                        estimate=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], J=t["J"],
                        p_one_sided=one_sided(t["mean"], t["se"], t["J"] - 1, "greater")))
        means.append(t["mean"]); ses.append(t["se"])
    p = X.S.pool_hksj(means, ses)
    per.append(dict(run="deconfounded replication", hypothesis="H2", estimator="two-stage pooled (HKSJ)",
                    estimate=p["mean"], ci_lo=p["ci_lo"], ci_hi=p["ci_hi"], J=4,
                    p_one_sided=one_sided(p["mean"], p["se"], p["df"], "greater")))
    return per


AGGREGATE_PAIRS = True


def h4_two_stage(cells, xcol, ycol, sizes, label, run, size_key=lambda ds: ds):
    means, ses, per = [], [], []
    for ds in X.DATASETS:
        key = size_key(ds)
        g = cells[cells.dataset == key].dropna(subset=[xcol, ycol])
        if AGGREGATE_PAIRS:
            g = g.groupby("split", as_index=False)[[xcol, ycol]].mean()
        o = X.ols_slope(g[xcol], g[ycol])
        s = sizes[(sizes.dataset == key) & (sizes.pair == "A")].iloc[0]
        J = g.split.nunique()
        se = o["se"] * np.sqrt(1.0 + J * s.n_test / s.n_train)
        means.append(o["slope"]); ses.append(se)
        per.append(dict(run=run, hypothesis="H4", estimator=f"two-stage {ds}{'' if key == ds else ' (' + key + ')'}",
                        estimate=o["slope"], ci_lo=np.nan, ci_hi=np.nan, J=J,
                        p_one_sided=one_sided(o["slope"], se, max(J - 1, 1), "less"), lhs=label))
    p = X.S.pool_hksj(means, ses)
    per.append(dict(run=run, hypothesis="H4", estimator="two-stage pooled (HKSJ)", estimate=p["mean"],
                    ci_lo=p["ci_lo"], ci_hi=p["ci_hi"], J=4, p_one_sided=one_sided(p["mean"], p["se"], p["df"], "less"),
                    lhs=label))
    return per


def main():
    rows = []
    # ---------------- pre-specified run
    mp = pd.read_csv(X.SUM / "primary.csv").set_index("hypothesis")
    h2 = pd.read_csv(X.OUT / "h2_extra.csv")
    h2p = h2[(h2.model.str.startswith("C(R6) ~ log2 positive windows")) & (h2.estimator == "two-stage pooled (HKSJ)")].iloc[0]
    cells = pd.read_csv(X.OUT / "h4_cells_extra.csv")
    h4 = h4_two_stage(cells, "rho_pos", "dcap_R6", X.sizes(), "Cap(R6)-Cap(R1) on test rho_pos", "pre-specified")
    h4p = h4[-1]
    ts_p = {"H1": mp.loc["H1", "p_raw"], "H2": h2p.p_one_sided, "H3": mp.loc["H3", "p_raw"], "H4": h4p["p_one_sided"],
            "H5": mp.loc["H5", "p_raw"]}
    holm_mixed = dict(zip(mp.index, X.S.holm(mp.p_raw.values)))
    holm_ts = dict(zip(ts_p, X.S.holm(list(ts_p.values()))))
    for hyp in ["H1", "H2", "H3", "H4", "H5"]:
        rows.append(dict(run="pre-specified", hypothesis=hyp, estimator="plan (as run)", verdict=mp.loc[hyp, "verdict"],
                         p_one_sided=mp.loc[hyp, "p_raw"], p_holm=holm_mixed[hyp]))
    rows.append(dict(run="pre-specified", hypothesis="H2", estimator="two-stage pooled (HKSJ)", estimate=h2p.slope,
                     ci_lo=h2p.ci_lo, ci_hi=h2p.ci_hi, J=4, p_one_sided=h2p.p_one_sided, p_holm=holm_ts["H2"]))
    rows.append(dict(h4p, p_holm=holm_ts["H4"]))
    for hyp in ("H1", "H3", "H5"):
        rows.append(dict(run="pre-specified", hypothesis=hyp, estimator="plan, in the two-stage Holm set",
                         p_one_sided=ts_p[hyp], p_holm=holm_ts[hyp]))
    # ---------------- declared deconfounded replication
    dp = pd.read_csv(RS / "deconf" / "summary" / "primary.csv").set_index("hypothesis")
    h2d = h2_two_stage_deconf()
    hd = pd.read_csv(RS / "deconf" / "summary" / "h4_data.csv")
    hd = hd.rename(columns={"dcap": "dcap_R6"})
    dsz = pd.read_csv(RS / "deconf" / "summary" / "sizes.csv")
    h4d = h4_two_stage(hd, "rho_pos", "dcap_R6", dsz, "Cap(R6)-Cap(R1) on test rho_pos", "deconfounded replication",
                       size_key=lambda ds: DECL[ds][1])
    ts_d = {"H1": dp.loc["H1", "p_raw"], "H2": h2d[-1]["p_one_sided"], "H3": dp.loc["H3", "p_raw"],
            "H4": h4d[-1]["p_one_sided"], "H5": dp.loc["H5", "p_raw"]}
    holm_dm = dict(zip(dp.index, X.S.holm(dp.p_raw.values)))
    holm_dt = dict(zip(ts_d, X.S.holm(list(ts_d.values()))))
    for hyp in ["H1", "H2", "H3", "H4", "H5"]:
        rows.append(dict(run="deconfounded replication", hypothesis=hyp, estimator="plan (as run)",
                         verdict=dp.loc[hyp, "verdict"], p_one_sided=dp.loc[hyp, "p_raw"], p_holm=holm_dm[hyp]))
    for r in h2d[:-1] + h4d[:-1]:
        rows.append(r)
    rows.append(dict(h2d[-1], p_holm=holm_dt["H2"]))
    rows.append(dict(h4d[-1], p_holm=holm_dt["H4"]))
    for hyp in ("H1", "H3", "H5"):
        rows.append(dict(run="deconfounded replication", hypothesis=hyp, estimator="plan, in the two-stage Holm set",
                         p_one_sided=ts_d[hyp], p_holm=holm_dt[hyp]))
    for r in h4[:-1]:
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "overlap_corrected_tests.csv", index=False)
    # ---------------- corrected H4 contrast, R0-referenced, validation rho, two-stage
    c5 = h4_two_stage(cells, "rho_pos_val", "cap6_minus_cap0", X.sizes(), "Cap(R6)-Cap(R0) on validation rho_pos",
                      "corrected H4 (post-plan)")
    c5b = h4_two_stage(cells, "rho_pos_val", "dcap_R6", X.sizes(), "Cap(R6)-Cap(R1) on validation rho_pos",
                       "corrected H4, R1-referenced (for comparison)")
    c5 = pd.DataFrame(c5 + c5b)
    c5["n_cells"] = [int(cells[(cells.dataset == e.split()[1]) if e.split()[1] in X.DATASETS else slice(None)]
                         .dropna(subset=["rho_pos_val"]).shape[0]) if e.split()[1] in X.DATASETS
                     else int(cells.dropna(subset=["rho_pos_val"]).shape[0]) for e in c5.estimator]
    c5["p_two_sided"] = 2 * np.minimum(c5.p_one_sided, 1 - c5.p_one_sided)
    c5.to_csv(X.OUT / "corrected_h4_val.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=[c for c in ("verdict", "lhs") if c in out]).round(4).to_string())
    print(c5.round(4).to_string())


if __name__ == "__main__":
    main()
