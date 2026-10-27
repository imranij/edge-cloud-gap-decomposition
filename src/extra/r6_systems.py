"""POST-PLAN, evaluation only.
Refits no model. Reads stored per-split outputs and arrays only. Hydraulic rig unless stated.

1. Edge + FS1 wired and streamed. Variant results_sens/hydabl/hyd_blk_e_fs1 (cloud view = edge sensors + FS1;
   first ten block-grouped splits, same test folds as results_sens/hydskab/hyd_blocks, asserted). The small model on
   edge + FS1 (ba_ef_cv: the edge family on the wired view, no uplink) and the cloud family on that view (ba_cf_cv),
   with paired Nadeau-Bengio differences (J = 10, n_train / n_test of the variant) from
     look-ahead R1 at 10%   ba_at_budget(R1, 0.10) of hyd_blocks, the comparator of r4_informative_stream.csv;
     causal-threshold R1 at 10%   R1v of r4_arch_deployable_splits.csv (hyd_blocks has no cross-fitted scores, so the
                                  threshold is the (1 - f) quantile of R1's validation scores; Table 7 footnote v).
   The same for hyd_blk_e_flow (both flow sensors; must reproduce r4_informative_stream.csv) and hyd_blk_e_fs2.
   Byte model of r4_systems.py: 17 features x 4 B per channel and window, 60 windows per hour, edge features 136 B.
   Rows are APPENDED to results/extra/r4_informative_stream.csv (channels "FS1", "FS2"; the "FS1 FS2" rows unchanged)
   and to results/extra/r5_hop_loads.csv / r5_budget_band_per_hop.csv (architectures "edge and FS1 features
   streamed", "FS1 wired to edge node (local only)", "gateway, cloud model, co-located with edge node"; pre-existing
   rows unchanged).
2. Cloud model at the co-located gateway. Same traffic as the small-model gateway (node features over the local
   network, no backhaul, no downlink requests); accuracy B_cC. Paired NB differences cloud - R1r (pre-specified runs,
   causal threshold from cross-fitted training scores) and cloud - R1v (variant runs), for every run of Table 7, from
   stored arrays (B_cC = BA of p_cf_cv >= t_cf_cv) and r4_arch_deployable_splits.csv. Model size and operations from
   results/extra/edge_cost.csv. Cooler hold-out: cloud per fold from results/extra/r5_gateway_holdout.csv.
3. Realised traffic of escalation. Mean realised escalated fraction of R1r at a nominal 10% (pre-specified
   splits, results_sens/routers/p2_stream.parquet) and the hourly 95th percentile (r4_escalation_horizon.csv) turned
   into uplink and downlink messages per cloud-only node per hour and the average volume per hop, beside the
   provisioned capacity (r5_hop_loads.csv) and the gateway.
4. Direction of the block-grouped information step. Per-split Shapley information step (shapley_info) of
   results_sens/hydskab/hyd_blocks, 20 splits per pair: number positive, minimum, maximum, per-repeat means.
5. Information step by cooler level, block-grouped run: per split and cooler level, the four
   balanced accuracies from the stored arrays (thresholded at the stored thresholds) on that level's test windows,
   where both classes are present, and the Shapley information step 0.5[(B_eC - B_eE) + (B_cC - B_cE)]; mean over the
   splits where it is defined. Descriptive.
Outputs (results/extra/): r6_fs1_designs.csv, r6_cloud_gateway.csv, r6_realised_traffic.csv, r6_block_step_direction.csv,
  r6_step_by_cooler.csv; rows appended as stated in 1.
    python src/extra/r6_systems.py
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

import common as X
from escalation.metrics import balanced_accuracy

F = 0.10
RSS = X.ROOT / "results_sens"
WPH = 60.0
FE = 136.0                       # edge features per window, B (34 x 4)
FNODE = 68.0                     # features of one cloud-only channel per window, B (17 x 4)
N_CO = 12


def nb(v, ntr, nte):
    return X.S.nb_test(np.asarray(v, float), ntr, nte)


def q(d, name, router="", budget=None):
    d = d[(d.h2_q == 1.0) & (d.quantity == name) & (d.router.fillna("") == router)]
    d = d[d.budget.isna()] if budget is None else d[np.isclose(d.budget.fillna(-1), budget)]
    return float(d.value.iloc[0])


def sizes_of(d):
    d = d[(d.quantity == "ba_ef_ev") & (d.h2_q == 1.0)]
    return float(d.n_train.iloc[0]), float(d.n_test.iloc[0])


def fs_designs():
    dep = pd.read_csv(X.OUT / "r4_arch_deployable_splits.csv")
    dep = dep[dep.variant == "hyd_blocks"]
    rows, srows = [], []
    for var, chans in (("hyd_blk_e_flow", ["FS1", "FS2"]), ("hyd_blk_e_fs1", ["FS1"]), ("hyd_blk_e_fs2", ["FS2"])):
        for pair in ("A", "B"):
            vals = []
            for f in sorted(f for f in (RSS / "hydabl" / var / pair).glob("split_*.parquet") if "_q" not in f.stem):
                j = int(f.stem.split("_")[1])
                d = pd.read_parquet(f)
                r = pd.read_parquet(RSS / "hydskab" / "hyd_blocks" / pair / f"split_{j}.parquet")
                za = np.load(RSS / "hydabl" / var / pair / f"arrays_{j}.npz")
                zr = np.load(RSS / "hydskab" / "hyd_blocks" / pair / f"arrays_{j}.npz")
                assert np.array_equal(za["test_idx"], zr["test_idx"]), (var, pair, j)
                ntr, nte = sizes_of(d)
                dj = dep[(dep.pair == pair) & (dep.split == j)].iloc[0]
                assert abs(dj.edge - q(r, "ba_ef_ev")) < 1e-9
                vals.append(dict(split=j, edge=q(d, "ba_ef_ev"), wired_edge_node=q(d, "ba_ef_cv"),
                                 cloud_family=q(d, "ba_cf_cv"), cloud_family_edge_view=q(d, "ba_cf_ev"),
                                 ref_cloud=q(r, "ba_cf_cv"), R1_lookahead=q(r, "ba_at_budget", "R1", F), R1v=dj.R1v,
                                 n_train=ntr, n_test=nte))
            w = pd.DataFrame(vals)
            srows += [dict(v, variant=var, pair=pair) for v in vals]
            rec = dict(variant=var, channels=" ".join(chans), pair=pair, J=len(w))
            for c in ("edge", "wired_edge_node", "cloud_family", "cloud_family_edge_view", "ref_cloud", "R1_lookahead", "R1v"):
                rec[c] = w[c].mean()
            for lab, a, b in (("wired - R1 lookahead", "wired_edge_node", "R1_lookahead"),
                              ("wired - R1v causal", "wired_edge_node", "R1v"),
                              ("stream cloud - R1 lookahead", "cloud_family", "R1_lookahead"),
                              ("stream cloud - R1v causal", "cloud_family", "R1v"),
                              ("wired - edge", "wired_edge_node", "edge")):
                t = nb(w[a] - w[b], w.n_train.mean(), w.n_test.mean())
                rec[f"{lab} mean"], rec[f"{lab} lo"], rec[f"{lab} hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            feat = 17 * 4 * len(chans) * WPH
            rec["stream_inf_features_bit_s"] = feat * 8 / 3600
            rec["stream_inf_plus_edge_bit_s"] = (feat + FE * WPH) * 8 / 3600
            rows.append(rec)
    o = pd.DataFrame(rows)
    # check against the earlier informative-stream rows (both flow sensors)
    inf = pd.read_csv(X.OUT / "r4_informative_stream.csv", float_precision="round_trip")
    for pair in ("A", "B"):
        a = o[(o.variant == "hyd_blk_e_flow") & (o.pair == pair)].iloc[0]
        b = inf[(inf.dataset == "hydraulic") & (inf.pair == pair) & (inf.channels == "FS1 FS2")].iloc[0]
        assert abs(a.wired_edge_node - b.inf_edge_node) < 1e-9 and abs(a.R1_lookahead - b.ref_R1) < 1e-9
        assert abs(a["wired - R1 lookahead mean"] - b["inf_edge_node - R1 mean"]) < 1e-9
        assert abs(a.stream_inf_plus_edge_bit_s - b.stream_inf_plus_edge_bit_s) < 1e-9
    o.to_csv(X.OUT / "r6_fs1_designs.csv", index=False)
    pd.DataFrame(srows).to_csv(X.OUT / "r6_fs1_designs_splits.csv", index=False)
    # append single-channel rows to r4_informative_stream.csv (same columns; pre-existing rows unchanged)
    inf = inf[~((inf.dataset == "hydraulic") & inf.channels.isin(["FS1", "FS2"]))]
    ch = json.load(open(X.ROOT / "data" / "processed" / "hydraulic" / "channels.json"))
    add = []
    for _, r in o[o.variant != "hyd_blk_e_flow"].iterrows():
        chans = r.channels.split()
        raw = sum(4 * ch["rates_hz"][c] * 3600 for c in chans)
        base = inf[(inf.dataset == "hydraulic") & (inf.pair == r.pair)].iloc[0]
        add.append({"dataset": "hydraulic", "pair": r.pair, "channels": r.channels, "J": r.J, "edge": r.edge,
                    "inf_cloud": r.cloud_family, "inf_edge_node": r.wired_edge_node, "ref_cloud": r.ref_cloud,
                    "ref_R1": r.R1_lookahead,
                    "inf_cloud - R1 mean": r["stream cloud - R1 lookahead mean"],
                    "inf_cloud - R1 lo": r["stream cloud - R1 lookahead lo"],
                    "inf_cloud - R1 hi": r["stream cloud - R1 lookahead hi"],
                    "inf_edge_node - R1 mean": r["wired - R1 lookahead mean"],
                    "inf_edge_node - R1 lo": r["wired - R1 lookahead lo"],
                    "inf_edge_node - R1 hi": r["wired - R1 lookahead hi"],
                    "stream_inf_features_B_h": 17 * 4 * len(chans) * WPH,
                    "stream_inf_features_bit_s": r.stream_inf_features_bit_s,
                    "stream_inf_plus_edge_B_h": 17 * 4 * len(chans) * WPH + FE * WPH,
                    "stream_inf_plus_edge_bit_s": r.stream_inf_plus_edge_bit_s,
                    "raw_inf_B_h": raw, "raw_inf_bit_s": raw * 8 / 3600, "inf_messages_per_h": WPH * (len(chans) + 1),
                    "escalation10_backhaul_bit_s": base.escalation10_backhaul_bit_s,
                    "full_stream_bit_s": base.full_stream_bit_s})
    inf = pd.concat([inf, pd.DataFrame(add)], ignore_index=True)
    inf.to_csv(X.OUT / "r4_informative_stream.csv", index=False)
    return o


def cloud_gateway():
    dep = pd.read_csv(X.OUT / "r4_arch_deployable_splits.csv")
    runs = [(ds, ds, X.RES / ds) for ds in X.DATASETS] + [
        ("hydraulic", "hyd_blocks", RSS / "hydskab" / "hyd_blocks"), ("skab", "skab_base", RSS / "hydskab" / "skab_base"),
        ("skab", "skab_filez", RSS / "hydskab" / "skab_filez"), ("seu", "seu_dm", RSS / "threewseu" / "seu_dm"),
        ("threew", "threew_dm", RSS / "threewseu" / "threew_dm")]
    cost = pd.read_csv(X.OUT / "edge_cost.csv").set_index(["dataset", "pair"])
    out = []
    for ds, var, base in runs:
        for pair in ("A", "B"):
            dd = dep[(dep.variant == var) & (dep.pair == pair)].set_index("split")
            vals = []
            for j, r in dd.iterrows():
                z = np.load(base / pair / f"arrays_{j}.npz")
                y = z["y_test"].astype(int)
                cloud = balanced_accuracy(y, (z["p_cf_cv"] >= float(z["t_cf_cv"])).astype(int))
                vals.append(dict(split=j, cloud=cloud, gateway=r.gateway, R1r=r.R1r, R1v=r.R1v, n_train=r.n_train,
                                 n_test=r.n_test))
            w = pd.DataFrame(vals)
            causal = "R1r" if w.R1r.notna().all() else "R1v"
            rec = dict(dataset=ds, variant=var, pair=pair, J=len(w), causal_rule=causal, cloud=w.cloud.mean(),
                       gateway_small=w.gateway.mean(), causal_R1=w[causal].mean())
            for lab, a, b in (("cloud_at_gateway - causal R1", "cloud", causal), ("cloud_at_gateway - small gateway", "cloud", "gateway")):
                t = nb(w[a] - w[b], w.n_train.mean(), w.n_test.mean())
                rec[f"{lab} mean"], rec[f"{lab} lo"], rec[f"{lab} hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            c = cost.loc[(ds, pair)]
            rec.update(cloud_model_ops=c.cloud_model_ops, cloud_param_bytes=c.cloud_param_bytes_deploy,
                       small_model_ops=c.gateway_model_ops, small_param_bytes=c.gateway_param_bytes_deploy,
                       feature_extraction_ops_edge=c.feature_extraction_ops,
                       feature_extraction_ops_all=c.feature_extraction_ops_all_sensors)
            out.append(rec)
    o = pd.DataFrame(out)
    ar = pd.read_csv(X.OUT / "arch_r3.csv")
    for _, r in o.iterrows():                       # the cloud column of Table 7
        a = ar[(ar.variant == r.variant) & (ar.pair == r.pair)]
        if len(a):
            assert abs(a.cloud.iloc[0] - r.cloud) < 1e-9, (r.variant, r.pair)
    h = pd.read_csv(X.OUT / "r5_gateway_holdout.csv")
    for pair in ("A", "B"):
        m = h[(h.pair == pair) & (h.split == -1)].iloc[0]
        f = h[(h.pair == pair) & (h.split >= 0)]
        o = pd.concat([o, pd.DataFrame([dict(dataset="hydraulic", variant="hyd_cooler2 (cooler hold-out, 3 folds)", pair=pair,
                                             J=len(f), causal_rule="R1v", cloud=m.cloud, gateway_small=m.gateway,
                                             causal_R1=m.R1v, cloud_min_fold=f.cloud.min(),
                                             gateway_small_min_fold=f.gateway.min())])], ignore_index=True)
    o.to_csv(X.OUT / "r6_cloud_gateway.csv", index=False)
    return o


def hop_rows(fs):
    """Append the new architectures to r5_hop_loads.csv and r5_budget_band_per_hop.csv (old rows unchanged)."""
    L = pd.read_csv(X.OUT / "r5_hop_loads.csv", float_precision="round_trip")
    new_arch = ["edge and FS1 features streamed", "FS1 wired to edge node (local only)",
                "gateway, cloud model, co-located with edge node"]
    L = L[~L.architecture.isin(new_arch)]
    add = []
    s1 = float(fs[(fs.variant == "hyd_blk_e_fs1") & (fs.pair == "A")].stream_inf_plus_edge_bit_s.iloc[0])
    f1 = float(fs[(fs.variant == "hyd_blk_e_fs1") & (fs.pair == "A")].stream_inf_features_bit_s.iloc[0])
    for pair in ("A", "B"):
        add.append(dict(pair=pair, architecture=new_arch[0], backhaul_bit_s=s1, local_bit_s=s1, node_up_bit_s=f1,
                        node_up_msg_h=WPH, node_down_msg_h=0.0, note="FS1 node only; the other cloud-only nodes send nothing"))
        add.append(dict(pair=pair, architecture=new_arch[1], backhaul_bit_s=0.0, local_bit_s=f1, node_up_bit_s=f1,
                        node_up_msg_h=WPH, node_down_msg_h=0.0,
                        note="FS1 on its own node sending its features to the edge node; 0 if FS1 is cabled to the edge node"))
        g = L[(L.pair == pair) & (L.architecture == "gateway, node features, co-located with edge node")].iloc[0]
        add.append(dict(g.to_dict(), architecture=new_arch[2], note="same traffic as the small-model gateway; accuracy B_cC"))
    L = pd.concat([L, pd.DataFrame(add)], ignore_index=True)
    L.to_csv(X.OUT / "r5_hop_loads.csv", index=False)
    B = pd.read_csv(X.OUT / "r5_budget_band_per_hop.csv", float_precision="round_trip")
    B = B[~B.alternative.isin(new_arch)]
    bands = []
    for pair in ("A", "B"):
        l = L[L.pair == pair].set_index("architecture")
        for prov in ("escalation, provisioned at hourly q95", "escalation, provisioned for q95 delay <= 10 min",
                     "escalation, one-window deadline (burst rate)"):
            for alt in new_arch:
                for hop, col in (("backhaul", "backhaul_bit_s"), ("local aggregate", "local_bit_s"),
                                 ("per cloud-only node", "node_up_bit_s")):
                    if hop == "per cloud-only node" and alt != new_arch[2]:
                        continue
                    lo, hi = float(l.loc[prov, col]), float(l.loc[alt, col])
                    bands.append(dict(pair=pair, escalation=prov, alternative=alt, hop=hop, escalation_bit_s=lo,
                                      alternative_bit_s=hi, band=("empty" if lo >= hi else f"{lo:.1f} to {hi:.1f} bit/s"),
                                      escalation_down_msg_h_per_node=float(l.loc[prov, "node_down_msg_h"])))
    B = pd.concat([B, pd.DataFrame(bands)], ignore_index=True)
    B.to_csv(X.OUT / "r5_budget_band_per_hop.csv", index=False)
    return L, B


def realised_traffic():
    st = pd.read_parquet(RSS / "routers" / "p2_stream.parquet")
    st = st[(st.dataset == "hydraulic") & (st.router == "R1") & np.isclose(st.f, F)]
    hz = pd.read_csv(X.OUT / "r4_escalation_horizon.csv")
    L = pd.read_csv(X.OUT / "r5_hop_loads.csv", float_precision="round_trip")
    full = WPH * (FE + N_CO * FNODE) * 8 / 3600
    rows = []
    for pair in ("A", "B"):
        f = float(st[st.pair == pair].realised_frac.mean())
        q95 = float(hz[(hz.pair == pair) & (hz.router == "R1r")].q95.iloc[0])
        prov = L[(L.pair == pair) & (L.architecture == "escalation, provisioned for q95 delay <= 10 min")].iloc[0]
        gw = L[(L.pair == pair) & (L.architecture == "gateway, node features, co-located with edge node")].iloc[0]
        rows.append(dict(pair=pair, realised_fraction_mean=f, hourly_q95_fraction=q95,
                         escalation_avg_backhaul_bit_s=f * full, escalation_avg_local_bit_s=f * full,
                         escalation_up_msg_h_per_node_mean=f * WPH, escalation_down_msg_h_per_node_mean=f * WPH,
                         escalation_down_msg_h_per_node_q95_hour=q95 * WPH,
                         escalation_provisioned_10min_bit_s=prov.local_bit_s,
                         escalation_provisioned_down_msg_h=prov.node_down_msg_h,
                         gateway_local_bit_s=gw.local_bit_s, gateway_up_msg_h_per_node=gw.node_up_msg_h,
                         escalation10_nominal_msgs_h_excl_requests=F * WPH * (N_CO + 2),
                         escalation10_nominal_downlink_requests_h=F * WPH * N_CO))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "r6_realised_traffic.csv", index=False)
    return o


def block_direction():
    rows = []
    for pair in ("A", "B"):
        vals = []
        for f in sorted(f for f in (RSS / "hydskab" / "hyd_blocks" / pair).glob("split_*.parquet") if "_q" not in f.stem):
            d = pd.read_parquet(f)
            d = d[(d.h2_q == 1.0) & (d.quantity == "shapley_info")]
            vals.append(dict(split=int(d.split.iloc[0]), repeat=int(d.repeat.iloc[0]), step=float(d.value.iloc[0])))
        w = pd.DataFrame(vals)
        rep = w.groupby("repeat").step.mean()
        rows.append(dict(pair=pair, J=len(w), n_positive=int((w.step > 0).sum()), min=w.step.min(), max=w.step.max(),
                         mean=w.step.mean(), repeat_mean_min=rep.min(), repeat_mean_max=rep.max(), n_repeats=len(rep)))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "r6_block_step_direction.csv", index=False)
    return o


def step_by_cooler():
    meta = pd.read_parquet(X.ROOT / "data" / "processed_sens" / "hyd_blocks" / "meta.parquet").reset_index(drop=True)
    rows = []
    for pair in ("A", "B"):
        for f in sorted(f for f in (RSS / "hydskab" / "hyd_blocks" / pair).glob("arrays_*.npz") if "_q" not in f.stem):
            j = int(f.stem.split("_")[1]); z = np.load(f)
            mt = meta.iloc[z["test_idx"]].reset_index(drop=True)
            y = z["y_test"].astype(int)
            assert np.array_equal(mt.label.to_numpy().astype(int), y)
            pr = {k: (z[f"p_{k}"] >= float(z[f"t_{k}"])).astype(int) for k in ("ef_ev", "ef_cv", "cf_ev", "cf_cv")}
            for lev, g in mt.groupby("cooler"):
                ix = g.index.to_numpy()
                if len(np.unique(y[ix])) < 2:
                    rows.append(dict(pair=pair, split=j, cooler=lev, n=len(ix), defined=False))
                    continue
                b = {k: balanced_accuracy(y[ix], v[ix]) for k, v in pr.items()}
                rows.append(dict(pair=pair, split=j, cooler=lev, n=len(ix), defined=True, B_eE=b["ef_ev"], B_eC=b["ef_cv"],
                                 B_cE=b["cf_ev"], B_cC=b["cf_cv"],
                                 info_shapley=0.5 * ((b["ef_cv"] - b["ef_ev"]) + (b["cf_cv"] - b["cf_ev"]))))
    w = pd.DataFrame(rows)
    w.to_csv(X.OUT / "r6_step_by_cooler_splits.csv", index=False)
    s = w[w.defined].groupby(["pair", "cooler"]).agg(n_splits=("info_shapley", "size"), info_mean=("info_shapley", "mean"),
                                                     info_min=("info_shapley", "min"), info_max=("info_shapley", "max"),
                                                     B_eE=("B_eE", "mean"), B_cC=("B_cC", "mean")).reset_index()
    s["n_splits_with_level"] = w.groupby(["pair", "cooler"]).size().values
    s.to_csv(X.OUT / "r6_step_by_cooler.csv", index=False)
    return s


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    fs = fs_designs(); print(fs.round(4).T.to_string())
    cg = cloud_gateway(); print(cg.round(4).to_string())
    L, B = hop_rows(fs); print(L.round(2).to_string())
    print(realised_traffic().round(3).T.to_string())
    print(block_direction().round(4).to_string())
    print(step_by_cooler().round(3).to_string())


if __name__ == "__main__":
    main()
