"""POST-PLAN, evaluation only. Refits no model.

1. Architecture comparison at a nominal 10% budget, per dataset x pair (pre-specified splits) and for the
   declared deconfounded variants and the causal SKAB offset variant (skab_base, pairs A and B):
     edge only (B_eE); random escalation (expected value, exact: B_eE + (k/n) sum g with k = round(0.1 n));
     the label-blind periodic schedule R9; R1, R6 and the cross-fitted R6x with look-ahead ranking;
     R1 with the causal streaming threshold of the main text (the (1 - f) quantile of cross-fitted training
     scores; results_sens/routers/p2_stream.parquet, ba_causal); gateway (B_eC); cloud (B_cC);
     paired Nadeau-Bengio intervals for R1 - edge, R1 realised - edge, R1 - random, R9 - edge and
     gateway - R1.
   Sources: results/<ds>/<pair>/{split_*.parquet, arrays_*.npz}; results_sens/routers/p2_{rows,stream}.parquet;
   results_sens/{hydskab,threewseu}/<variant>/<pair>/{split_*.parquet, arrays_*.npz}.
2. System cost per hour with ONE gateway model (raw samples from the cloud-only nodes to the gateway, which
   extracts every feature), messages counted as trigger + one upload per cloud-only node + reply, one node
   per cloud-only channel (an upper bound; co-located channels could share a node), and the uplink range of
   escalation from the same causal R1 threshold rule as the main text (pair A, per-fold realised fractions).
   Also the alternative gateway model with features extracted at the sensor nodes. Both gateway
   rows assume the gateway is co-located with the edge node, so their local traffic is that of the cloud-only nodes
   (results/extra/r5_hop_loads.csv gives the separate-box case); the pull flow and the full cloud now carry their
   bytes on the local network as well. Sources:
   results/extra/{system_cost,uplink_per_hour}.csv, results_sens/routers/p2_stream.parquet,
   data/processed/<ds>/channels.json. Also average link rates (bit/s).
Outputs: results/extra/arch_r3.csv, results/extra/system_cost_r3.csv
    python src/extra/arch_r3.py
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

import common as X

F = 0.10
RSS = X.ROOT / "results_sens"
VARIANTS = [("hydraulic", "hydskab", "hyd_blocks", "AB"), ("skab", "hydskab", "skab_base", "AB"),
            ("skab", "hydskab", "skab_filez", "AB"), ("seu", "threewseu", "seu_dm", "AB"),
            ("threew", "threewseu", "threew_dm", "AB")]


def split_values(base, pair, dataset_label):
    fs = sorted(f for f in (base / pair).glob("split_*.parquet") if "_q" not in f.stem)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d.h2_q == 1.0]; d["router"] = d["router"].fillna("")
    rows = []
    for j, g in d.groupby("split"):
        def q(name, r="", b=None):
            m = (g.quantity == name) & (g.router == r)
            m &= g.budget.isna() if b is None else np.isclose(g.budget.fillna(-1), b)
            return float(g[m].value.iloc[0]) if m.any() else np.nan
        z = np.load(base / pair / f"arrays_{int(j)}.npz")
        gd = z["g_dep"]; n = len(gd); k = int(round(F * n))
        e = q("ba_ef_ev")
        rows.append(dict(split=int(j), edge=e, random=e + k / n * gd.sum(), R9=q("ba_at_budget", "R9", F),
                         R1=q("ba_at_budget", "R1", F), R6=q("ba_at_budget", "R6", F), gateway=q("ba_ef_cv"),
                         cloud=q("ba_cf_cv"), n_train=float(g.n_train.iloc[0]), n_test=float(g.n_test.iloc[0])))
    return pd.DataFrame(rows).set_index("split")


def nb(v, ntr, nte):
    return X.S.nb_test(np.asarray(v, float), ntr, nte)


def arch():
    p2 = pd.read_parquet(RSS / "routers" / "p2_rows.parquet")
    st = pd.read_parquet(RSS / "routers" / "p2_stream.parquet")
    out = []
    jobs = [(ds, ds, X.RES / ds, pair) for ds in X.DATASETS for pair in ("A", "B")]
    jobs += [(ds, v, RSS / area / v, pair) for ds, area, v, pairs in VARIANTS for pair in pairs]
    for ds, label, base, pair in jobs:
        w = split_values(base, pair, label)
        if label == ds:
            r6x = p2[(p2.dataset == ds) & (p2.pair == pair) & (p2.quantity == "ba_at_budget") & (p2.router == "R6x")
                     & np.isclose(p2.budget.fillna(-1), F)].set_index("split").value
            rr = st[(st.dataset == ds) & (st.pair == pair) & (st.router == "R1") & np.isclose(st.f, F)].set_index("split")
            w["R6x"] = r6x; w["R1_realised"] = rr.ba_causal; w["R1_realised_frac"] = rr.realised_frac
        ntr, nte = w.n_train.mean(), w.n_test.mean()
        rec = dict(dataset=ds, variant=label, pair=pair, J=len(w))
        for c in ("edge", "random", "R9", "R1", "R6", "R6x", "R1_realised", "R1_realised_frac", "gateway", "cloud"):
            rec[c] = w[c].mean() if c in w else np.nan
        for lab, a, b in (("R1 - edge", "R1", "edge"), ("R1 realised - edge", "R1_realised", "edge"),
                          ("R1 - random", "R1", "random"), ("R9 - edge", "R9", "edge"),
                          ("random - edge", "random", "edge"), ("gateway - R1", "gateway", "R1"),
                          ("gateway - R1 realised", "gateway", "R1_realised"), ("gateway - cloud", "gateway", "cloud")):
            if a in w and b in w and w[a].notna().all():
                t = nb(w[a] - w[b], ntr, nte)
                rec[f"{lab} mean"], rec[f"{lab} lo"], rec[f"{lab} hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
        out.append(rec)
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "arch_r3.csv", index=False)
    return o


def cost():
    sc = pd.read_csv(X.OUT / "system_cost.csv")
    u = pd.read_csv(X.OUT / "uplink_per_hour.csv")
    st = pd.read_parquet(RSS / "routers" / "p2_stream.parquet")
    rows = []
    for ds in X.DATASETS:
        ch = json.load(open(X.ROOT / "data" / "processed" / ds / "channels.json"))
        n_co = len(set(ch["cloud"]) - set(ch["edge"]))
        uu = u[(u.dataset == ds) & np.isclose(u.budget, F)].iloc[0]
        wph = float(uu.windows_per_hour)
        fr = st[(st.dataset == ds) & (st.pair == "A") & (st.router == "R1") & np.isclose(st.f, F)].realised_frac
        msg_esc = n_co + 2                                   # trigger + one upload per cloud-only node + reply
        s = sc[sc.dataset == ds].set_index("architecture")
        esc = s.loc["escalation 10%, pull, on-demand extraction"]
        gw = s.loc["gateway"]; fc = s.loc["full cloud"]; eo = s.loc["edge only"]
        feat_co = float(uu.features_cloud_B - uu.features_edge_B)
        base = dict(dataset=ds, windows_per_hour=wph, cloud_only_nodes=n_co)
        rows.append(dict(base, architecture="edge only", ops_per_window=eo.ops_total, uplink_B_per_h=0.0,
                         uplink_bit_s=0.0, messages_per_h=0.0, local_B_per_h=0.0))
        rows.append(dict(base, architecture="escalation 10%, pull, on-demand", ops_per_window=esc.ops_total,
                         uplink_B_per_h=float(uu.pull_features_B_per_h), uplink_bit_s=8 * float(uu.pull_features_B_per_h) / 3600,
                         uplink_raw_B_per_h=float(uu.pull_raw_B_per_h), uplink_raw_bit_s=8 * float(uu.pull_raw_B_per_h) / 3600,
                         burst_bit_s_features=8 * float(uu.features_cloud_B) / float(uu.stride_s),
                         burst_bit_s_raw=8 * float(uu.raw_cloud_window_B) / float(uu.stride_s),
                         messages_per_h=msg_esc * F * wph, messages_per_escalation=msg_esc,
                         realised_frac_min=fr.min(), realised_frac_max=fr.max(),
                         uplink_B_per_h_realised_min=fr.min() * wph * float(uu.features_cloud_B),
                         uplink_B_per_h_realised_max=fr.max() * wph * float(uu.features_cloud_B),
                         # in the pull flow every escalation's bytes cross the local network as well
                         # as the backhaul (Table 7 caption); the local column was 0 before
                         local_B_per_h=float(uu.pull_features_B_per_h), local_bit_s=8 * float(uu.pull_features_B_per_h) / 3600))
        rows.append(dict(base, architecture="gateway, raw to gateway", ops_per_window=gw.ops_total, uplink_B_per_h=0.0,
                         uplink_bit_s=0.0, messages_per_h=0.0, local_B_per_h=float(uu.raw_stream_cloud_only_B_per_h),
                         local_bit_s=8 * float(uu.raw_stream_cloud_only_B_per_h) / 3600, local_streams=n_co))
        rows.append(dict(base, architecture="gateway, features at sensor nodes", ops_per_window=gw.ops_total,
                         ops_at_gateway=eo.ops_total - 0 + (gw.ops_total - s.loc["escalation 10%, pull, continuous extraction"].ops_other_nodes - eo.ops_total),
                         uplink_B_per_h=0.0, uplink_bit_s=0.0, messages_per_h=0.0,
                         local_B_per_h=feat_co * wph, local_bit_s=8 * feat_co * wph / 3600, local_messages_per_h=n_co * wph))
        rows.append(dict(base, architecture="full cloud", ops_per_window=fc.ops_total,
                         uplink_B_per_h=float(uu.full_cloud_features_B_per_h),
                         uplink_bit_s=8 * float(uu.full_cloud_features_B_per_h) / 3600,
                         uplink_raw_B_per_h=float(uu.raw_stream_all_B_per_h), uplink_raw_bit_s=8 * float(uu.raw_stream_all_B_per_h) / 3600,
                         messages_per_h=(n_co + 2) * wph,
                         # the full stream crosses the local network too (was 0)
                         local_B_per_h=float(uu.full_cloud_features_B_per_h),
                         local_bit_s=8 * float(uu.full_cloud_features_B_per_h) / 3600))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "system_cost_r3.csv", index=False)
    return o


def main():
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
    a = arch()
    print(a[["dataset", "variant", "pair", "J", "edge", "random", "R9", "R1", "R1_realised", "R6", "R6x", "gateway", "cloud",
             "R1 - edge mean", "R1 - random mean", "R1 - random lo", "R9 - edge mean", "gateway - R1 mean", "gateway - R1 lo",
             "gateway - R1 hi"]].round(3).to_string())
    print(cost().round(1).to_string())


if __name__ == "__main__":
    main()
