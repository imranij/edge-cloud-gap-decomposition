"""POST-PLAN summaries for p2_crossfit.py and p3_fixedtier.py.

NB-corrected means and 95% CIs per dataset x pair use n_train, n_test from results/summary/sizes.csv
(as analyze.nb_sizes); pooled differences use escalation.stats.pool_hksj over the four datasets.
Writes p2_routers.csv, p2_diffs.csv, p2_counts_summary.csv, p2_stream_summary.csv, p2_comb_summary.csv,
p3_by_level.csv, p3_slopes.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from util import OUT, ROOT, CM
from escalation import stats as S

SIZES = pd.read_csv(ROOT / "results" / "summary" / "sizes.csv")


def sz(ds, pair):
    r = SIZES[(SIZES.dataset == ds) & (SIZES.pair == pair)].iloc[0]
    return r["n_train"], r["n_test"]


def nbrow(v, ds, pair):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    t = S.nb_test(v, *sz(ds, pair))
    return dict(J=t["J"], mean=t["mean"], se=t["se"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], p=t["p"])


def routers_table(rows):
    out = []
    rows = rows.copy(); rows["budget"] = rows["budget"].fillna(-1)
    for (ds, pair, q, r, b), g in rows.groupby(["dataset", "pair", "quantity", "router", "budget"]):
        out.append(dict(dataset=ds, pair=pair, quantity=q, router=r, budget=np.nan if b == -1 else b,
                        **nbrow(g.sort_values("split").value, ds, pair)))
    return pd.DataFrame(out)


def diffs(rows):
    C = rows[rows.quantity == "C"].pivot_table(index=["dataset", "pair", "split"], columns="router", values="value")
    specs = [("R6x", "R1"), ("R6x", "R6"), ("R10", "R1"), ("HyR", "R1"), ("HyS", "R1"), ("R4x", "R4"),
             ("R5x", "R5"), ("R10", "R6x"), ("R6x", "R8")]
    out = []
    for a, b in specs:
        if a not in C or b not in C:
            continue
        for pair in ("A", "B"):
            per = []
            for ds in CM.DATASETS:
                if (ds, pair) not in {(i[0], i[1]) for i in C.index}:
                    continue
                d = (C.loc[(ds, pair)][a] - C.loc[(ds, pair)][b]).dropna()
                if len(d) == 0:
                    continue
                t = nbrow(d, ds, pair)
                per.append(dict(contrast=f"C({a}) - C({b})", pair=pair, dataset=ds, **t))
            if not per:
                continue
            pdx = pd.DataFrame(per)
            pool = S.pool_hksj(pdx["mean"], pdx["se"])
            out += per + [dict(contrast=f"C({a}) - C({b})", pair=pair, dataset="POOLED (HKSJ)", J=pool["k"],
                               mean=pool["mean"], se=pool["se"], ci_lo=pool["ci_lo"], ci_hi=pool["ci_hi"], p=pool["p"])]
    return pd.DataFrame(out)


def counts_summary(cnt):
    cols = ["n_val", "pos_win_val", "pos_groups_val", "nonzero_g_val", "n_fit", "pos_win_fit", "pos_groups_fit",
            "n_T", "pos_win_T", "pos_groups_T", "n_folds", "r10_n_nonzero", "r10_n_gpos", "ba_E_oof_T",
            "ba_C_oof_T", "C_T_R1_pooled", "C_T_R6x_pooled", "C_T_R1_foldavg", "C_T_R6x_foldavg", "seconds_models"]
    s = cnt.groupby(["dataset", "pair"])[cols].mean().reset_index()
    s["hys_R6x_share"] = cnt.groupby(["dataset", "pair"]).hys_pick.apply(lambda x: (x == "R6x").mean()).values
    s["J"] = cnt.groupby(["dataset", "pair"]).size().values
    return s


def stream_summary(st):
    out = []
    for (ds, pair, r, f), g in st.groupby(["dataset", "pair", "router", "f"]):
        d = nbrow(g.ba_causal - g.ba_lookahead, ds, pair)
        out.append(dict(dataset=ds, pair=pair, router=r, f=f, J=len(g), frac_mean=g.realised_frac.mean(),
                        frac_sd=g.realised_frac.std(ddof=1), frac_min=g.realised_frac.min(),
                        frac_max=g.realised_frac.max(), ba_causal=g.ba_causal.mean(), ba_lookahead=g.ba_lookahead.mean(),
                        diff_mean=d["mean"], diff_ci_lo=d["ci_lo"], diff_ci_hi=d["ci_hi"]))
    return pd.DataFrame(out)


def comb_summary(cb):
    out = []
    for (ds, pair), g in cb.groupby(["dataset", "pair"]):
        d1 = nbrow(g.ba_comb - g.ba_stacker_dep, ds, pair)
        d2 = nbrow(g.ba_comb - g[["ba_ef_ev", "ba_cf_cv"]].max(axis=1), ds, pair)
        comb = nbrow(g.ba_comb, ds, pair)
        out.append(dict(dataset=ds, pair=pair, J=len(g), ba_edge=g.ba_ef_ev.mean(), ba_cloud=g.ba_cf_cv.mean(),
                        ba_stacker=g.ba_stacker_dep.mean(), ba_oracle=g.ba_oracle_dep.mean(),
                        ba_comb=comb["mean"], ba_comb_lo=comb["ci_lo"], ba_comb_hi=comb["ci_hi"],
                        esc_frac=g.esc_frac.mean(),
                        comb_minus_stacker=d1["mean"], cms_lo=d1["ci_lo"], cms_hi=d1["ci_hi"],
                        comb_minus_best_single=d2["mean"], cmb_lo=d2["ci_lo"], cmb_hi=d2["ci_hi"],
                        unresolved=g.unresolved_dep.mean(),
                        share_recovered=(g.ba_comb - g.ba_stacker_dep).mean() / g.unresolved_dep.mean()))
    return pd.DataFrame(out)


def p3(cells):
    by = cells.groupby(["dataset", "q"]).agg(J=("split", "size"), C_R6=("C_R6", "mean"), C_R1=("C_R1", "mean"),
                                             dC=("dC", "mean"), pos_win_val_kept=("pos_win_val_kept", "mean"),
                                             pos_groups_val_kept=("pos_groups_val_kept", "mean"),
                                             nonzero_g_kept=("nonzero_g_kept", "mean"),
                                             checks=("check", lambda x: ",".join(sorted(set(x))))).reset_index()
    for c in ("C_R6", "dC"):
        lo, hi = [], []
        for _, r in by.iterrows():
            t = nbrow(cells[(cells.dataset == r.dataset) & (cells.q == r.q)][c], r.dataset, "A")
            lo.append(t["ci_lo"]); hi.append(t["ci_hi"])
        by[f"{c}_lo"] = lo; by[f"{c}_hi"] = hi
    cells = cells.copy()
    cells["x"] = np.log2(cells.pos_win_val_kept.clip(lower=1))
    sl = []
    per = []
    for ds, g in cells.groupby("dataset"):
        g = g[np.isfinite(g.C_R6)]
        b_ols = np.polyfit(g.x, g.C_R6, 1)[0]
        ps = []
        for sid, h in g.groupby("split"):
            if h.x.nunique() >= 2:
                ps.append(np.polyfit(h.x, h.C_R6, 1)[0])
        t = nbrow(ps, ds, "A")
        per.append(dict(dataset=ds, ols_slope=b_ols, two_stage_mean=t["mean"], se=t["se"], ci_lo=t["ci_lo"],
                        ci_hi=t["ci_hi"], J=t["J"]))
    pdx = pd.DataFrame(per)
    pool = S.pool_hksj(pdx.two_stage_mean, pdx.se)
    sl = pd.concat([pdx, pd.DataFrame([dict(dataset="POOLED (HKSJ)", two_stage_mean=pool["mean"], se=pool["se"],
                                            ci_lo=pool["ci_lo"], ci_hi=pool["ci_hi"], J=pool["k"])])])
    sl.insert(0, "design", "fixed tiers (this analysis), x = log2 kept positive val windows")
    return by, sl


def confirmatory_two_stage():
    """Same two-stage slope on the STORED confirmatory H2 run (tiers retrained), for a like-for-like comparison,
    with x = log2 n_pos_train (as in the plan) and x = log2 n_pos_val."""
    out = []
    for xname in ("n_pos_train", "n_pos_val"):
        per = []
        for ds in CM.DATASETS:
            ps = []
            for sid in CM.stored_split_ids(ds, "A", None):
                xs, cs = [], []
                for q in (1.0, 0.1, 0.25, 0.5):
                    r = CM.load_rows(ds, "A", sid, None if q == 1.0 else q)
                    if r is None or "skipped" in set(r.quantity):
                        continue
                    c = r[(r.quantity == "C") & (r.router == "R6")].value
                    if xname == "n_pos_train":
                        x = r.n_pos_train.iloc[0]
                    else:
                        x = r[r.quantity == "n_pos_val"].value.iloc[0]
                    if len(c) and np.isfinite(c.iloc[0]):
                        xs.append(np.log2(max(x, 1))); cs.append(c.iloc[0])
                if len(set(xs)) >= 2:
                    ps.append(np.polyfit(xs, cs, 1)[0])
            t = nbrow(ps, ds, "A")
            per.append(dict(dataset=ds, two_stage_mean=t["mean"], se=t["se"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], J=t["J"]))
        pdx = pd.DataFrame(per)
        pool = S.pool_hksj(pdx.two_stage_mean, pdx.se)
        pdx = pd.concat([pdx, pd.DataFrame([dict(dataset="POOLED (HKSJ)", two_stage_mean=pool["mean"], se=pool["se"],
                                                 ci_lo=pool["ci_lo"], ci_hi=pool["ci_hi"], J=pool["k"])])])
        pdx.insert(0, "design", f"confirmatory H2 run (tiers retrained), x = log2 {xname}")
        out.append(pdx)
    return pd.concat(out)


def main():
    import sys
    which = sys.argv[1:] or ["p2", "p3"]
    pd.set_option("display.width", 250)
    if "p2" in which:
        rows = pd.read_parquet(OUT / "p2_rows.parquet")
        routers_table(rows).to_csv(OUT / "p2_routers.csv", index=False)
        dd = diffs(rows); dd.to_csv(OUT / "p2_diffs.csv", index=False); print(dd.round(4).to_string())
        cs = counts_summary(pd.read_parquet(OUT / "p2_counts.parquet")); cs.to_csv(OUT / "p2_counts_summary.csv", index=False)
        print(cs.round(3).to_string())
        ss = stream_summary(pd.read_parquet(OUT / "p2_stream.parquet")); ss.to_csv(OUT / "p2_stream_summary.csv", index=False)
        print(ss.round(4).to_string())
        cb = comb_summary(pd.read_parquet(OUT / "p2_comb.parquet")); cb.to_csv(OUT / "p2_comb_summary.csv", index=False)
        print(cb.round(4).to_string())
    if "p3" in which:
        by, sl = p3(pd.read_parquet(OUT / "p3_cells.parquet"))
        sl = pd.concat([sl, confirmatory_two_stage()])
        by.to_csv(OUT / "p3_by_level.csv", index=False); sl.to_csv(OUT / "p3_slopes.csv", index=False)
        print(by.round(4).to_string()); print(sl.round(4).to_string())


if __name__ == "__main__":
    main()
