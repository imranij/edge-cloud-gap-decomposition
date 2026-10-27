"""POST-PLAN, evaluation only. Refits no model.

1. Deployable escalation against the gateway. For every run in Table 7 (pre-specified, pairs A and B; the
   declared variants and the causal SKAB offset, skab_base):
     R1r  R1 with the causal threshold of the main text, the (1 - f) quantile of cross-fitted training scores
          (results_sens/routers/p2_stream.parquet; exists only for the pre-specified runs);
     R1v  R1 with a threshold that is also fixed before the test fold: the (1 - f) quantile of R1's scores on the
          stored validation fold (r1_confidence(pval_ef_ev, t_ef_ev)), test windows strictly above it escalated;
          this is the validation-fixed rule of src/extra/realised_budget.py (checked against
          results/extra/realised_budget_cells.csv for the pre-specified runs); computable for every run.
   At f = 0.10: realised escalated fraction, balanced accuracy, random escalation at the same realised fraction
   (exact expectation), and paired Nadeau-Bengio CIs of gateway - R1r, gateway - R1v, R1v - edge and
   R1v - random at the realised fraction. Checks: stored score_R1 equals r1_confidence(p_ef_ev, t_ef_ev); the BA
   with escalation equals B_eE plus the sum of g over escalated windows.
   Output: results/extra/r4_arch_deployable.csv
2. Realised escalation over a common time horizon. Hydraulic rig, R1r, f = 0.10, pre-specified splits: in
   each repeat the five test folds cover every cycle once, so the per-cycle decisions of the five folds form one
   series over the 2,205 consecutive 60-s cycles. Rolling 60-cycle (1-h) escalated fractions over that series
   (all 1-h windows of consecutive cycles), pooled over the four repeats: median, 90th and 95th percentile,
   maximum. Also per 111-cycle test block under the block-grouped splits (hyd_blocks), with R1v. The other
   datasets have no contiguous 1-h horizon (SKAB files <= 22 min, SEU recordings 80 s, 3W windows subsampled).
   Output: results/extra/r4_escalation_horizon.csv
3. Per-node and per-hop traffic and the always-on informative-channel design. Topology: sensor nodes
   (the edge node and one node per cloud-only channel) -> local network -> gateway -> backhaul -> cloud. Byte model
   of results/extra/uplink_per_hour.csv: 17 features of 4 B per channel and window; raw samples 4 B. Per
   architecture: bytes and messages per hour per cloud-only node (uplink, local network), downlink requests per
   cloud-only node (the pull flow needs one per escalation; not costed before), the edge node's uplink, and the
   backhaul aggregate. "Edge + informative channels, always on": the channels that carried the step in the
   post-plan ablations (hydraulic FS1 and FS2; SKAB flow rate) either stream their features with the edge's
   features to the cloud, or are wired to the edge node, which then runs the edge family on edge + those channels
   (no uplink). Accuracy from results_sens/hydabl/hyd_blk_e_flow and results_sens/round3/skab_e_flow (B_cC and
   B_eC per split, first 10 splits) beside escalation by R1 at 10% on the same splits of the reference run.
   Output: results/extra/r4_per_node_cost.csv, results/extra/r4_informative_stream.csv
4. Budget band: escalation needed (link below the always-on alternative) and deployable after
   provisioning (link at or above the 1-h 95th-percentile escalation rate). Output: results/extra/r4_budget_band.csv
    python src/extra/r4_systems.py
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

import common as X
from escalation.routers import r1_confidence
from escalation.metrics import balanced_accuracy

F = 0.10
RSS = X.ROOT / "results_sens"
P2 = RSS / "routers" / "p2_arrays"
RUNS = [(ds, ds, X.RES / ds) for ds in X.DATASETS] + [
    ("hydraulic", "hyd_blocks", RSS / "hydskab" / "hyd_blocks"), ("skab", "skab_base", RSS / "hydskab" / "skab_base"),
    ("skab", "skab_filez", RSS / "hydskab" / "skab_filez"), ("seu", "seu_dm", RSS / "threewseu" / "seu_dm"),
    ("threew", "threew_dm", RSS / "threewseu" / "threew_dm")]


def nb(v, ntr, nte):
    return X.S.nb_test(np.asarray(v, float), ntr, nte)


def sizes_of(base, pair, j):
    d = pd.read_parquet(base / pair / f"split_{j}.parquet")
    d = d[(d.quantity == "ba_ef_ev") & (d.h2_q == 1.0)]
    return float(d.n_train.iloc[0]), float(d.n_test.iloc[0])


def deployable():
    rb = pd.read_csv(X.OUT / "realised_budget_cells.csv")
    rb = rb[(rb.router == "R1") & np.isclose(rb.budget, F)].set_index(["dataset", "pair", "split"])
    st = pd.read_parquet(RSS / "routers" / "p2_stream.parquet")
    st = st[(st.router == "R1") & np.isclose(st.f, F)]
    out, per = [], []
    for ds, label, base in RUNS:
        for pair in ("A", "B"):
            fs = sorted(f for f in (base / pair).glob("arrays_*.npz") if "_q" not in f.stem)
            rows = []
            for f in fs:
                j = int(f.stem.split("_")[1])
                z = np.load(f)
                y = z["y_test"].astype(int); g = z["g_dep"]
                te, tc = float(z["t_ef_ev"]), float(z["t_cf_cv"])
                pe = (z["p_ef_ev"] >= te).astype(int); pc = (z["p_cf_cv"] >= tc).astype(int)
                s = z["score_R1"]
                assert np.allclose(s, r1_confidence(z["p_ef_ev"], te))
                edge = balanced_accuracy(y, pe)
                gate = balanced_accuracy(y, (z["p_ef_cv"] >= float(z["t_ef_cv"])).astype(int))
                thr = float(np.quantile(r1_confidence(z["pval_ef_ev"], te), 1 - F))
                esc = s > thr          # strictly above: the validation-fixed rule (src/extra/realised_budget.py)
                r1v = balanced_accuracy(y, np.where(esc, pc, pe))
                assert abs(r1v - (edge + g[esc].sum())) < 1e-9
                ntr, nte = sizes_of(base, pair, j)
                rec = dict(split=j, edge=edge, gateway=gate, R1v=r1v, R1v_frac=float(esc.mean()),
                           random_at_R1v_frac=edge + esc.mean() * g.sum(), n_train=ntr, n_test=nte)
                if label == ds:
                    ref = rb.loc[(ds, pair, j)]
                    assert abs(ref.realised_fraction - esc.mean()) < 1e-12 and abs(ref.ba_realised - r1v) < 1e-9
                    r = st[(st.dataset == ds) & (st.pair == pair) & (st.split == j)]
                    rec["R1r"] = float(r.ba_causal.iloc[0]); rec["R1r_frac"] = float(r.realised_frac.iloc[0])
                    rec["random_at_R1r_frac"] = edge + rec["R1r_frac"] * g.sum()
                rows.append(rec)
                per.append(dict(rec, dataset=ds, variant=label, pair=pair))
            w = pd.DataFrame(rows).set_index("split")
            ntr, nte = w.n_train.mean(), w.n_test.mean()
            rec = dict(dataset=ds, variant=label, pair=pair, J=len(w))
            for c in ("edge", "gateway", "R1v", "R1v_frac", "R1r", "R1r_frac"):
                rec[c] = w[c].mean() if c in w else np.nan
                if c.endswith("frac") and c in w:
                    rec[c + "_min"], rec[c + "_max"] = w[c].min(), w[c].max()
            for lab, a, b in (("gateway - R1r", "gateway", "R1r"), ("gateway - R1v", "gateway", "R1v"),
                              ("R1v - edge", "R1v", "edge"), ("R1v - random", "R1v", "random_at_R1v_frac"),
                              ("R1r - random", "R1r", "random_at_R1r_frac"), ("R1r - edge", "R1r", "edge")):
                if a in w and b in w:
                    t = nb(w[a] - w[b], ntr, nte)
                    rec[f"{lab} mean"], rec[f"{lab} lo"], rec[f"{lab} hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            out.append(rec)
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "r4_arch_deployable.csv", index=False)
    pd.DataFrame(per).to_csv(X.OUT / "r4_arch_deployable_splits.csv", index=False)
    return o


def horizon():
    ds = "hydraulic"
    meta = X.dataset(ds)["meta"].reset_index(drop=True)
    cyc = meta.cycle.to_numpy()
    order = np.argsort(cyc)
    assert np.all(np.diff(cyc[order]) == 1), "hydraulic cycles are not consecutive"
    sp = X.splits(ds)
    st = pd.read_parquet(RSS / "routers" / "p2_stream.parquet")
    rows = []
    for pair in ("A", "B"):
        rates = []
        for rep in sorted({s["repeat"] for s in sp.values()}):
            dec = np.full(len(cyc), np.nan)
            for j in [k for k, s in sp.items() if s["repeat"] == rep]:
                z = X.arrays(ds, pair, j); p = np.load(P2 / f"{ds}_{pair}_{j}.npz")
                thr = float(np.quantile(r1_confidence(p["pE"], float(z["t_ef_ev"])), 1 - F))
                esc = p["score_R1"] >= thr
                ref = st[(st.dataset == ds) & (st.pair == pair) & (st.split == j) & (st.router == "R1")
                         & np.isclose(st.f, F)].realised_frac.iloc[0]
                assert abs(esc.mean() - ref) < 1e-12
                assert np.all(np.isnan(dec[z["test_idx"]]))
                dec[z["test_idx"]] = esc
            assert not np.isnan(dec).any()
            series = dec[order]
            roll = pd.Series(series).rolling(60).mean().dropna().to_numpy()
            rates.append(roll)
        r = np.concatenate(rates)
        rows.append(dict(dataset=ds, pair=pair, router="R1r", horizon="rolling 1 h (60 consecutive cycles)",
                         n_windows=len(r), median=np.median(r), q90=np.quantile(r, 0.90), q95=np.quantile(r, 0.95),
                         max=r.max(), share_above_2x=float((r > 2 * F).mean())))
    # per 111-cycle test block under hyd_blocks, R1v
    base = RSS / "hydskab" / "hyd_blocks"
    for pair in ("A", "B"):
        vals = []
        for f in sorted(f for f in (base / pair).glob("arrays_*.npz") if "_q" not in f.stem):
            z = np.load(f)
            te = float(z["t_ef_ev"])
            thr = float(np.quantile(r1_confidence(z["pval_ef_ev"], te), 1 - F))
            esc = z["score_R1"] > thr
            for u in np.unique(z["groups_test"]):
                ix = z["groups_test"] == u
                vals.append((ix.sum(), esc[ix].mean()))
        v = np.array(vals)
        rows.append(dict(dataset=ds, pair=pair, router="R1v", horizon="per 111-cycle test block (hyd_blocks)",
                         n_windows=len(v), block_size_median=np.median(v[:, 0]), median=np.median(v[:, 1]),
                         q90=np.quantile(v[:, 1], 0.90), q95=np.quantile(v[:, 1], 0.95), max=v[:, 1].max(),
                         share_above_2x=float((v[:, 1] > 2 * F).mean())))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "r4_escalation_horizon.csv", index=False)
    return o


def per_node_and_informative(hz):
    u = pd.read_csv(X.OUT / "uplink_per_hour.csv")
    rows = []
    for ds in X.DATASETS:
        ch = json.load(open(X.ROOT / "data" / "processed" / ds / "channels.json"))
        n_co = len(set(ch["cloud"]) - set(ch["edge"]))
        uu = u[(u.dataset == ds) & np.isclose(u.budget, F)].iloc[0]
        wph = float(uu.windows_per_hour)
        fe, fc = float(uu.features_edge_B), float(uu.features_cloud_B)
        f_node = (fc - fe) / n_co                      # 17 features x 4 B per cloud-only channel
        esc_ph = F * wph
        base = dict(dataset=ds, windows_per_hour=wph, cloud_only_nodes=n_co, node_features_B=f_node)
        rows.append(dict(base, architecture="escalation 10%, pull", node_up_B_h=esc_ph * f_node, node_up_msg_h=esc_ph,
                         node_down_msg_h=esc_ph, edge_up_B_h=esc_ph * fe, edge_up_msg_h=esc_ph, edge_down_msg_h=esc_ph,
                         backhaul_B_h=esc_ph * fc, backhaul_msg_h=esc_ph * (n_co + 2)))
        rows.append(dict(base, architecture="full cloud stream", node_up_B_h=wph * f_node, node_up_msg_h=wph,
                         node_down_msg_h=0.0, edge_up_B_h=wph * fe, edge_up_msg_h=wph, edge_down_msg_h=wph,
                         backhaul_B_h=wph * fc, backhaul_msg_h=wph * (n_co + 2)))
        rows.append(dict(base, architecture="gateway, raw to gateway",
                         node_up_B_h=float(uu.raw_stream_cloud_only_B_per_h) / n_co, node_up_msg_h=np.nan,
                         node_down_msg_h=0.0, edge_up_B_h=0.0, edge_up_msg_h=0.0, edge_down_msg_h=0.0,
                         backhaul_B_h=0.0, backhaul_msg_h=0.0))
    pn = pd.DataFrame(rows)
    for c in ("node_up_B_h", "edge_up_B_h", "backhaul_B_h"):
        pn[c.replace("_B_h", "_bit_s")] = pn[c] * 8 / 3600
    pn.to_csv(X.OUT / "r4_per_node_cost.csv", index=False)

    # always-on informative channels
    inf = {"hydraulic": (RSS / "hydabl" / "hyd_blk_e_flow", RSS / "hydskab" / "hyd_blocks", ["FS1", "FS2"]),
           "skab": (RSS / "round3" / "skab_e_flow", X.RES / "skab", ["Volume Flow RateRMS"])}
    out = []
    for ds, (vd, rd, chans) in inf.items():
        uu = u[(u.dataset == ds) & np.isclose(u.budget, F)].iloc[0]
        wph = float(uu.windows_per_hour); fe = float(uu.features_edge_B)
        ch = json.load(open(X.ROOT / "data" / "processed" / ds / "channels.json"))
        n_co = len(set(ch["cloud"]) - set(ch["edge"]))
        f_node = (float(uu.features_cloud_B) - fe) / n_co
        rate = ch["rates_hz"]
        for pair in ("A", "B"):
            vals = []
            for f in sorted(f for f in (vd / pair).glob("split_*.parquet") if "_q" not in f.stem):
                j = int(f.stem.split("_")[1])
                d = pd.read_parquet(f); d = d[(d.h2_q == 1.0) & d.budget.isna() & (d.router.fillna("") == "")]
                r = pd.read_parquet(rd / pair / f"split_{j}.parquet")
                r = r[r.h2_q == 1.0]; r["router"] = r.router.fillna("")
                q = lambda x, n: float(x[x.quantity == n].value.iloc[0])
                r1 = float(r[(r.quantity == "ba_at_budget") & (r.router == "R1") & np.isclose(r.budget.fillna(-1), F)].value.iloc[0])
                rb = r[r.budget.isna() & (r.router == "")]
                vals.append(dict(split=j, inf_cloud=q(d, "ba_cf_cv"), inf_edge_node=q(d, "ba_ef_cv"), edge=q(d, "ba_ef_ev"),
                                 ref_cloud=q(rb, "ba_cf_cv"), ref_R1=r1, n_train=float(d.n_train.iloc[0]), n_test=float(d.n_test.iloc[0])))
            w = pd.DataFrame(vals)
            rec = dict(dataset=ds, pair=pair, channels=" ".join(chans), J=len(w))
            for c in ("edge", "inf_cloud", "inf_edge_node", "ref_cloud", "ref_R1"):
                rec[c] = w[c].mean()
            for lab, a, b in (("inf_cloud - R1", "inf_cloud", "ref_R1"), ("inf_edge_node - R1", "inf_edge_node", "ref_R1")):
                t = nb(w[a] - w[b], w.n_train.mean(), w.n_test.mean())
                rec[f"{lab} mean"], rec[f"{lab} lo"], rec[f"{lab} hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            feat_inf = 17 * 4 * len(chans) * wph
            rec["stream_inf_features_B_h"] = feat_inf
            rec["stream_inf_features_bit_s"] = feat_inf * 8 / 3600
            rec["stream_inf_plus_edge_B_h"] = feat_inf + fe * wph
            rec["stream_inf_plus_edge_bit_s"] = (feat_inf + fe * wph) * 8 / 3600
            raw = sum(4 * rate[c] * 3600 for c in chans)
            rec["raw_inf_B_h"] = raw; rec["raw_inf_bit_s"] = raw * 8 / 3600
            rec["inf_messages_per_h"] = wph * (len(chans) + 1)
            rec["escalation10_backhaul_bit_s"] = F * wph * float(uu.features_cloud_B) * 8 / 3600
            rec["full_stream_bit_s"] = wph * float(uu.features_cloud_B) * 8 / 3600
            out.append(rec)
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "r4_informative_stream.csv", index=False)

    # budget band, hydraulic (only dataset with a 1-h horizon)
    band = []
    for pair in ("A", "B"):
        h = hz[(hz.pair == pair) & (hz.router == "R1r")].iloc[0]
        ir = o[(o.dataset == "hydraulic") & (o.pair == pair)].iloc[0]
        full = ir.full_stream_bit_s
        band.append(dict(dataset="hydraulic", pair=pair, nominal_escalation_bit_s=F * full,
                         provisioned_q95_1h_rate=h.q95, provisioned_escalation_bit_s=h.q95 * full,
                         informative_stream_bit_s=ir.stream_inf_plus_edge_bit_s,
                         informative_stream_only_bit_s=ir.stream_inf_features_bit_s, full_stream_bit_s=full,
                         band_vs_full_stream=f"{h.q95 * full:.1f} to {full:.1f} bit/s",
                         band_vs_informative_stream=("empty" if h.q95 * full >= ir.stream_inf_plus_edge_bit_s else
                                                     f"{h.q95 * full:.1f} to {ir.stream_inf_plus_edge_bit_s:.1f} bit/s")))
    b = pd.DataFrame(band)
    b.to_csv(X.OUT / "r4_budget_band.csv", index=False)
    return pn, o, b


def main():
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
    a = deployable()
    print(a.round(3).to_string())
    hz = horizon()
    print(hz.round(3).to_string())
    pn, o, b = per_node_and_informative(hz)
    print(pn.round(1).to_string()); print(o.round(3).T.to_string()); print(b.round(2).T.to_string())


if __name__ == "__main__":
    main()
