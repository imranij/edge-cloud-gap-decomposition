"""POST-PLAN, evaluation only.
Refits no model. Hydraulic rig unless stated.

1. Escalation delay. The R1r decision series of src/extra/r4_systems.py::horizon() is rebuilt with the same
   code path and checks (causal threshold = (1 - f) quantile of R1's cross-fitted training scores, f = 0.10; in each
   repeat the five test folds cover the 2,205 consecutive 60-s cycles once). Every escalated window sends its
   238-feature payload (952 B: 34 edge features with the trigger, 17 features from each of the 12 cloud-only nodes)
   at the end of its window; a FIFO link of rate r bit/s serves 952 B in 952*8/r s. Delay = time from window end
   until the cloud holds the payload (transmission only; trigger, request, protocol overhead and inference ignored).
   Each repeat is its own timeline; delays are pooled over the four repeats. Outputs per pair and link rate: median,
   95th percentile and maximum delay, and the share of escalations delayed beyond one window. Also the smallest rate
   at which the 95th percentile (and the maximum) of the delay stays within a bound D, by bisection.
2. Deadline with dropping. With rate r and deadline D, an escalation is sent only if it would reach the cloud
   within D of its window end; otherwise it is dropped and the edge decision stands. Per test fold the balanced
   accuracy is B_eE + sum of g_dep over delivered escalations (checked: with nothing dropped it equals the stored
   ba_causal of results_sens/routers/p2_stream.parquet); mean over the 20 splits.
3. Per-hop band. Loads per hop (backhaul aggregate, local-network aggregate, per cloud-only node uplink and
   downlink) for: escalation at 10% nominal; escalation provisioned at the hourly 95th percentile (r4) and at the
   delay bounds of item 1; the gateway running the small model on node-extracted features (co-located with the edge
   node, or a separate box that also receives the edge node's features); the full feature stream; the edge and flow
   features streamed. In the pull flow every escalation's bytes cross the local network and the backhaul. A band
   against alternative a on hop h is [escalation's provisioned load on h, a's load on h), empty if the first is not
   below the second (escalation is then never needed on h, or never provisionable below a).
4. Look-ahead versus causal margins and gap shares. From results/extra/arch_r3.csv and r4_arch_deployable.csv:
   R1 (look-ahead) - random and R1r (causal) - random at 10% with their CIs and the random comparator used; the
   share of the edge-to-cloud gap recovered at 10% by R1, R1r and random escalation, (BA - edge)/(cloud - edge).
5. Gateway under the cooler hold-out. results_sens/holdout2/hyd_cooler2 (one fold per held-out cooler level):
   edge, gateway (B_eC), cloud, and R1 at 10% with the validation-fixed threshold (R1v, the rule of r4_systems.py)
   and with look-ahead, per fold and mean over folds.
Outputs (results/extra/): r5_escalation_delay.csv, r5_delay_bound_rates.csv, r5_deadline_drop.csv,
  r5_hop_loads.csv, r5_budget_band_per_hop.csv, r5_margins_and_gap_share.csv, r5_gateway_holdout.csv;
  r4_budget_band.csv is rewritten with added per-hop gateway and delay-bound columns (its old columns unchanged).
    python src/extra/r5_systems.py
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
WIN = 60.0                 # s, hydraulic window = stride
RATES = [20, 25, 30, 952 * 8 * 60 / 3600 * 0.2857, 40, 42.31, 44.43, 50, 64, 80, 100, 952 * 8 * 60 / 3600]
BOUNDS = [60, 300, 600, 1800, 3600]          # s


def series(pair):
    """Per repeat: escalation indicator per cycle (ordered by cycle), and per fold the arrays needed for BA."""
    ds = "hydraulic"
    meta = X.dataset(ds)["meta"].reset_index(drop=True)
    cyc = meta.cycle.to_numpy(); order = np.argsort(cyc)
    assert np.all(np.diff(cyc[order]) == 1)
    pos = np.empty(len(cyc), int); pos[order] = np.arange(len(cyc))      # time index of each window
    sp = X.splits(ds)
    st = pd.read_parquet(RSS / "routers" / "p2_stream.parquet")
    reps, folds = [], []
    for rep in sorted({s["repeat"] for s in sp.values()}):
        dec = np.full(len(cyc), np.nan)
        for j in [k for k, s in sp.items() if s["repeat"] == rep]:
            z = X.arrays(ds, pair, j); p = np.load(P2 / f"{ds}_{pair}_{j}.npz")
            te = float(z["t_ef_ev"])
            thr = float(np.quantile(r1_confidence(p["pE"], te), 1 - F))
            esc = p["score_R1"] >= thr
            ref = st[(st.dataset == ds) & (st.pair == pair) & (st.split == j) & (st.router == "R1") & np.isclose(st.f, F)].iloc[0]
            assert abs(esc.mean() - ref.realised_frac) < 1e-12
            y = z["y_test"].astype(int)
            edge = balanced_accuracy(y, (z["p_ef_ev"] >= te).astype(int))
            assert abs(edge + z["g_dep"][esc].sum() - ref.ba_causal) < 1e-9
            assert np.all(np.isnan(dec[z["test_idx"]]))
            dec[z["test_idx"]] = esc
            folds.append(dict(rep=rep, split=j, t=pos[z["test_idx"]], esc=esc, g=z["g_dep"], edge=edge,
                              ba_causal=float(ref.ba_causal)))
        assert not np.isnan(dec).any()
        reps.append((rep, dec[order].astype(bool)))
    return reps, folds


def fifo(arrivals, service):
    """Departure times of a FIFO single server with deterministic service time."""
    d = np.empty(len(arrivals)); last = -np.inf
    for i, a in enumerate(arrivals):
        last = max(a, last) + service
        d[i] = last
    return d


def delays(reps, rate, payload_B):
    s = payload_B * 8 / rate
    out = []
    for _, esc in reps:
        arr = (np.flatnonzero(esc) + 1) * WIN            # window end
        out.append(fifo(arr, s) - arr)
    return np.concatenate(out)


def min_rate(reps, payload_B, bound, stat):
    lo, hi = 1.0, 2000.0
    f = (lambda d: np.quantile(d, 0.95)) if stat == "q95" else np.max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(delays(reps, mid, payload_B)) <= bound:
            hi = mid
        else:
            lo = mid
    return hi


def deadline_drop(folds, rate, bound, payload_B):
    """Admission: send only if delivered within `bound`; the queue holds only admitted escalations."""
    s = payload_B * 8 / rate
    by_rep = {}
    for f in folds:
        by_rep.setdefault(f["rep"], []).append(f)
    ba, sent, tot = [], 0, 0
    for rep, fs in by_rep.items():
        ev = []                                           # (arrival, fold index, window index in fold)
        for k, f in enumerate(fs):
            for i in np.flatnonzero(f["esc"]):
                ev.append(((f["t"][i] + 1) * WIN, k, i))
        ev.sort()
        deliver = [np.zeros(len(f["esc"]), bool) for f in fs]
        last = -np.inf
        for a, k, i in ev:
            done = max(a, last) + s
            if done - a <= bound + 1e-9:
                deliver[k][i] = True; last = done
        for k, f in enumerate(fs):
            ba.append(f["edge"] + f["g"][deliver[k]].sum())
            sent += deliver[k].sum(); tot += f["esc"].sum()
    return float(np.mean(ba)), 1 - sent / tot


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    u = pd.read_csv(X.OUT / "uplink_per_hour.csv")
    uu = u[(u.dataset == "hydraulic") & np.isclose(u.budget, F)].iloc[0]
    wph = float(uu.windows_per_hour); fe = float(uu.features_edge_B); fc = float(uu.features_cloud_B)
    ch = json.load(open(X.ROOT / "data" / "processed" / "hydraulic" / "channels.json"))
    n_co = len(set(ch["cloud"]) - set(ch["edge"]))
    f_node = (fc - fe) / n_co
    assert fc == 952 and n_co == 12 and f_node == 68
    full = wph * fc * 8 / 3600

    # ---- 1 and 2: delay
    drows, brows, xrows = [], [], []
    S = {}
    for pair in ("A", "B"):
        reps, folds = series(pair)
        S[pair] = (reps, folds)
        for r in RATES:
            d = delays(reps, r, fc)
            drows.append(dict(pair=pair, rate_bit_s=r, n_escalations=len(d), median_s=np.median(d),
                              q95_s=np.quantile(d, 0.95), max_s=d.max(), share_beyond_one_window=float((d > WIN + 1e-6).mean()),
                              median_min=np.median(d) / 60, q95_min=np.quantile(d, 0.95) / 60, max_min=d.max() / 60))
        for b in BOUNDS:
            brows.append(dict(pair=pair, delay_bound_s=b, min_rate_q95_bit_s=min_rate(reps, fc, b, "q95"),
                              min_rate_max_bit_s=min_rate(reps, fc, b, "max")))
        base = float(np.mean([f["ba_causal"] for f in folds]))
        for r in (42.31, 64, 952 * 8 * 60 / 3600):
            for b in (60, 600, 1800):
                ba, drop = deadline_drop(folds, r, b, fc)
                xrows.append(dict(pair=pair, rate_bit_s=r, deadline_s=b, share_dropped=drop, ba_mean=ba,
                                  ba_R1r_no_drop=base, ba_change=ba - base))
    dl = pd.DataFrame(drows); bd = pd.DataFrame(brows); dd = pd.DataFrame(xrows)
    dl.to_csv(X.OUT / "r5_escalation_delay.csv", index=False)
    bd.to_csv(X.OUT / "r5_delay_bound_rates.csv", index=False)
    dd.to_csv(X.OUT / "r5_deadline_drop.csv", index=False)
    print(dl.round(2).to_string()); print(bd.round(1).to_string()); print(dd.round(4).to_string())

    # ---- 3: per-hop loads and bands
    hz = pd.read_csv(X.OUT / "r4_escalation_horizon.csv")
    inf = pd.read_csv(X.OUT / "r4_informative_stream.csv")
    flow_bit_s = float(inf[(inf.dataset == "hydraulic") & (inf.pair == "A")].stream_inf_plus_edge_bit_s.iloc[0])
    loads = []
    for pair in ("A", "B"):
        q95 = float(hz[(hz.pair == pair) & (hz.router == "R1r")].q95.iloc[0])
        b10 = float(bd[(bd.pair == pair) & (bd.delay_bound_s == 600)].min_rate_q95_bit_s.iloc[0])
        esc = lambda frac, name: dict(pair=pair, architecture=name, backhaul_bit_s=frac * wph * fc * 8 / 3600,
                                      local_bit_s=frac * wph * fc * 8 / 3600,
                                      node_up_bit_s=frac * wph * f_node * 8 / 3600, node_up_msg_h=frac * wph,
                                      node_down_msg_h=frac * wph, escalation_fraction=frac)
        loads.append(esc(F, "escalation 10%, nominal"))
        loads.append(esc(q95, "escalation, provisioned at hourly q95"))
        loads.append(esc(b10 / full, "escalation, provisioned for q95 delay <= 10 min"))
        loads.append(esc(1.0, "escalation, one-window deadline (burst rate)"))
        loads.append(dict(pair=pair, architecture="gateway, node features, co-located with edge node", backhaul_bit_s=0.0,
                          local_bit_s=n_co * f_node * wph * 8 / 3600, node_up_bit_s=f_node * wph * 8 / 3600,
                          node_up_msg_h=wph, node_down_msg_h=0.0))
        loads.append(dict(pair=pair, architecture="gateway, node features, separate from edge node", backhaul_bit_s=0.0,
                          local_bit_s=fc * wph * 8 / 3600, node_up_bit_s=f_node * wph * 8 / 3600,
                          node_up_msg_h=wph, node_down_msg_h=0.0))
        loads.append(dict(pair=pair, architecture="full feature stream", backhaul_bit_s=full, local_bit_s=full,
                          node_up_bit_s=f_node * wph * 8 / 3600, node_up_msg_h=wph, node_down_msg_h=0.0))
        loads.append(dict(pair=pair, architecture="edge and flow features streamed", backhaul_bit_s=flow_bit_s,
                          local_bit_s=flow_bit_s, node_up_bit_s=f_node * wph * 8 / 3600, node_up_msg_h=wph,
                          node_down_msg_h=0.0, note="flow nodes only; the other cloud-only nodes send nothing"))
    L = pd.DataFrame(loads)
    L.to_csv(X.OUT / "r5_hop_loads.csv", index=False)
    print(L.round(2).to_string())
    bands = []
    for pair in ("A", "B"):
        l = L[L.pair == pair].set_index("architecture")
        for prov in ("escalation, provisioned at hourly q95", "escalation, provisioned for q95 delay <= 10 min",
                     "escalation, one-window deadline (burst rate)"):
            for alt in ("gateway, node features, co-located with edge node", "gateway, node features, separate from edge node",
                        "full feature stream", "edge and flow features streamed"):
                for hop, col in (("backhaul", "backhaul_bit_s"), ("local aggregate", "local_bit_s"),
                                 ("per cloud-only node", "node_up_bit_s")):
                    if hop == "per cloud-only node" and alt == "edge and flow features streamed":
                        continue
                    lo, hi = float(l.loc[prov, col]), float(l.loc[alt, col])
                    bands.append(dict(pair=pair, escalation=prov, alternative=alt, hop=hop, escalation_bit_s=lo,
                                      alternative_bit_s=hi,
                                      band=("empty" if lo >= hi else f"{lo:.1f} to {hi:.1f} bit/s"),
                                      escalation_down_msg_h_per_node=float(l.loc[prov, "node_down_msg_h"])))
    B = pd.DataFrame(bands)
    B.to_csv(X.OUT / "r5_budget_band_per_hop.csv", index=False)
    print(B.to_string())
    # extend r4_budget_band.csv (old columns kept)
    rb = pd.read_csv(X.OUT / "r4_budget_band.csv")
    rb = rb[[c for c in rb.columns if not c.startswith("r5_")]]
    for i, r in rb.iterrows():
        g = B[(B.pair == r.pair) & (B.escalation == "escalation, provisioned at hourly q95")].set_index(["alternative", "hop"]).band
        rb.loc[i, "r5_band_vs_gateway_backhaul"] = g[("gateway, node features, co-located with edge node", "backhaul")]
        rb.loc[i, "r5_band_vs_gateway_colocated_local"] = g[("gateway, node features, co-located with edge node", "local aggregate")]
        rb.loc[i, "r5_band_vs_gateway_separate_local"] = g[("gateway, node features, separate from edge node", "local aggregate")]
        rb.loc[i, "r5_band_vs_gateway_per_node"] = g[("gateway, node features, co-located with edge node", "per cloud-only node")]
        rb.loc[i, "r5_min_rate_q95_delay_10min_bit_s"] = float(bd[(bd.pair == r.pair) & (bd.delay_bound_s == 600)].min_rate_q95_bit_s.iloc[0])
        rb.loc[i, "r5_min_rate_one_window_bit_s"] = float(bd[(bd.pair == r.pair) & (bd.delay_bound_s == 60)].min_rate_max_bit_s.iloc[0])
    rb.to_csv(X.OUT / "r4_budget_band.csv", index=False)

    # ---- 4: margins and gap share
    a3 = pd.read_csv(X.OUT / "arch_r3.csv"); ad = pd.read_csv(X.OUT / "r4_arch_deployable.csv")
    m = []
    for _, r in a3.iterrows():
        d = ad[(ad.variant == r.variant) & (ad.pair == r.pair)].iloc[0]
        gap = r.cloud - r.edge
        causal = "R1r" if pd.notna(d.R1r) else "R1v"
        cb = d[f"{causal}"]
        cm = d[f"{causal} - random mean"]; clo = d[f"{causal} - random lo"]; chi = d[f"{causal} - random hi"]
        m.append(dict(dataset=r.dataset, variant=r.variant, pair=r.pair, edge=r.edge, cloud=r.cloud, gap=gap,
                      R1_lookahead=r.R1, random_exact=r.random, causal_rule=causal, R1_causal=cb,
                      lookahead_minus_random=r["R1 - random mean"], lookahead_lo=r["R1 - random lo"], lookahead_hi=r["R1 - random hi"],
                      lookahead_comparator="exact expectation of random escalation of round(0.1 n) windows",
                      causal_minus_random=cm, causal_lo=clo, causal_hi=chi,
                      causal_comparator="exact expectation of random escalation at the realised fraction",
                      share_gap_R1=(r.R1 - r.edge) / gap if gap > 0 else np.nan,
                      share_gap_causal=(cb - r.edge) / gap if gap > 0 else np.nan,
                      share_gap_random=(r.random - r.edge) / gap if gap > 0 else np.nan))
    M = pd.DataFrame(m)
    M.to_csv(X.OUT / "r5_margins_and_gap_share.csv", index=False)
    print(M.round(3).to_string())

    # ---- 5: gateway and R1 on the cooler hold-out
    hrows = []
    base = RSS / "holdout2" / "hyd_cooler2"
    hd = pd.read_csv(X.OUT / "holdout_diagnostics.csv")
    for pair in ("A", "B"):
        for f in sorted((base / pair).glob("arrays_*.npz")):
            j = int(f.stem.split("_")[1]); z = np.load(f)
            y = z["y_test"].astype(int); g = z["g_dep"]; te = float(z["t_ef_ev"])
            pe = (z["p_ef_ev"] >= te).astype(int)
            edge = balanced_accuracy(y, pe)
            gate = balanced_accuracy(y, (z["p_ef_cv"] >= float(z["t_ef_cv"])).astype(int))
            cloud = balanced_accuracy(y, (z["p_cf_cv"] >= float(z["t_cf_cv"])).astype(int))
            s = z["score_R1"]; assert np.allclose(s, r1_confidence(z["p_ef_ev"], te))
            thr = float(np.quantile(r1_confidence(z["pval_ef_ev"], te), 1 - F))
            esc = s > thr
            k = int(round(F * len(s))); top = np.argsort(-s, kind="stable")[:k]
            ref = hd[(hd.variant == "hyd_cooler2") & (hd.pair == pair) & (hd.split == j)].iloc[0]
            assert abs(ref.ba_edge - edge) < 1e-9 and abs(ref.ba_gateway - gate) < 1e-9 and abs(ref.ba_cloud - cloud) < 1e-9
            hrows.append(dict(pair=pair, split=j, held_out_cooler=ref.held_out, edge=edge, gateway=gate, cloud=cloud,
                              R1v=edge + g[esc].sum(), R1v_frac=float(esc.mean()), R1_lookahead=edge + g[top].sum()))
    H = pd.DataFrame(hrows)
    mean = H.groupby("pair")[["edge", "gateway", "cloud", "R1v", "R1v_frac", "R1_lookahead"]].mean().reset_index()
    mean["split"] = -1; mean["held_out_cooler"] = "mean over folds"
    H = pd.concat([H, mean], ignore_index=True)
    H.to_csv(X.OUT / "r5_gateway_holdout.csv", index=False)
    print(H.round(3).to_string())


if __name__ == "__main__":
    main()
