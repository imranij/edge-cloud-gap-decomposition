"""Analytic cost model (uplink bytes per hour, cloud inferences per hour, edge
operations and memory) and the four-architecture comparison. Post-plan analysis, not
pre-specified. Nothing is measured on hardware and no model is refitted.

    python src/extra/cost_model.py

Inputs: data/processed/<ds>/channels.json and bytes.json (rates, payload), results/summary/
footprint.csv (parameters, MACs, nodes visited of the split-0 models), payload.csv, per_dataset.csv
and routers.csv (balanced accuracies).

System model (stated in the paper, Section III-A): the edge node holds the edge sensors and runs
feature extraction, the edge model and the router on every window. The cloud-only sensors sit at
other nodes. Two data flows are costed:
  pull    cloud-only nodes buffer their latest window and send it only when the edge escalates
          that window; per escalated window the uplink carries the full cloud view
          (edge part + cloud-only part), as features or as raw samples;
  stream  cloud-only nodes stream continuously, so their bytes are spent whatever the router
          does, and escalation adds only the edge's part of the window.
Window rate = decisions per hour of continuous operation: hydraulic one 60-s cycle per minute,
SKAB one 60-s window every 10 s (stride; consecutive windows share 50 of 60 samples), SEU
non-overlapping 1,024-sample windows at the inferred 5,120 Hz (0.2 s), 3W non-overlapping 300-s
windows. Raw bytes are float32 at native rates; raw per-window sums for SKAB count shared samples
again, so the continuous stream is also given.

Edge operation counts are rough analytic estimates (stated constants): per channel of L samples,
15 L operations for the time-domain statistics and spectral moments plus 2.5 L log2 L for the real
FFT; LR 2 d MACs; trees one comparison per node visited; R6 50 trees of depth 3 (150 comparisons +
50 additions); R3 a binary search over the n_val stored validation scores; R7 = R6 + 14 extra inputs
(attributions d MACs, five perturbed attribution passes 5 x 3 d, drift 3 d); R8 3 d MACs.
Parameter bytes: float32 for LR (3 d + 1 values incl. scaler) and CNN weights; 12 B per tree node
(2 B feature index, 4 B threshold or leaf value, 2 x 2 B child indices, 2 B padding).
Outputs: results/extra/uplink_per_hour.csv, edge_cost.csv, architectures.csv
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import common as X

WIN = {  # window length (s), decisions per hour
    "hydraulic": dict(window_s=60.0, stride_s=60.0),
    "skab": dict(window_s=60.0, stride_s=10.0),
    "seu": dict(window_s=1024 / 5120.0, stride_s=1024 / 5120.0),
    "threew": dict(window_s=300.0, stride_s=300.0),
}
BUDGETS = [0.05, 0.10, 0.20, 0.30]
NODE_B = 12


def channels(ds):
    return json.loads((X.ROOT / "data" / "processed" / ds / "channels.json").read_text())


def payload(ds):
    return json.loads((X.ROOT / "data" / "processed" / ds / "bytes.json").read_text())


def stream_bytes_per_hour(ch, names):
    return float(sum(ch["rates_hz"][n] * 4 * 3600 for n in names))


def feat_ops(L):
    return 15 * L + 2.5 * L * np.log2(max(L, 2))


def main():
    pdt = pd.read_csv(X.SUM / "per_dataset.csv")
    rt = pd.read_csv(X.SUM / "routers.csv")
    fp = pd.read_csv(X.SUM / "footprint.csv")
    df = X.long_results()
    nv = df[(df.quantity == "n_val") & (df.h2_q == 1.0) & (df.pair == "A")].groupby("dataset").value.mean()
    N_VAL = {d: int(round(nv[d])) for d in X.DATASETS}          # mean validation size (R3 stores n_val scores)
    upl, cost, arch = [], [], []
    for ds in X.DATASETS:
        ch, pl, w = channels(ds), payload(ds), WIN[ds]
        wph = 3600.0 / w["stride_s"]
        edge, cloud = ch["edge"], ch["cloud"]
        conly = [c for c in cloud if c not in edge]
        raw_edge_h = stream_bytes_per_hour(ch, edge)
        raw_conly_h = stream_bytes_per_hour(ch, conly)
        raw_all_h = raw_edge_h + raw_conly_h
        fe, fc = pl["features_edge"], pl["features_cloud"]
        base = dict(dataset=ds, window_s=w["window_s"], stride_s=w["stride_s"], windows_per_hour=wph,
                    features_edge_B=fe, features_cloud_B=fc, raw_edge_window_B=pl["raw_edge"],
                    raw_cloud_window_B=pl["raw_cloud"], raw_stream_all_B_per_h=raw_all_h,
                    raw_stream_cloud_only_B_per_h=raw_conly_h, raw_stream_edge_B_per_h=raw_edge_h)
        for f in [0.0] + BUDGETS + [1.0]:
            upl.append(dict(base, budget=f, cloud_inferences_per_h=f * wph,
                            pull_features_B_per_h=f * wph * fc,
                            pull_raw_B_per_h=min(f * wph * pl["raw_cloud"], raw_all_h) if ds != "skab" else f * wph * pl["raw_cloud"],
                            pull_raw_dedup_upper_B_per_h=min(f * wph * pl["raw_cloud"], raw_all_h),
                            stream_features_B_per_h=wph * (fc - fe) + f * wph * fe,
                            stream_raw_B_per_h=raw_conly_h + min(f * wph * pl["raw_edge"], raw_edge_h),
                            gateway_local_features_B_per_h=wph * fc, full_cloud_features_B_per_h=wph * fc,
                            full_cloud_raw_B_per_h=raw_all_h))
        # edge cost per window
        L = {c: int(round(ch["rates_hz"][c] * w["window_s"])) for c in edge}
        fx = float(sum(feat_ops(l) for l in L.values()))
        Lc = {c: int(round(ch["rates_hz"][c] * w["window_s"])) for c in cloud}
        fx_all = float(sum(feat_ops(l) for l in Lc.values()))
        buf = 4 * sum(L.values()); fftw = 4 * max(L.values()) + 8 * (max(L.values()) // 2 + 1)
        dE = len(edge) * 17
        routers_ops = {"R1": 5, "R3": 5 + 2 * np.ceil(np.log2(N_VAL[ds])), "R6": 200, "R7": 200 + dE + 15 * dE + 3 * dE,
                       "R8": 3 * dE + 4}
        router_mem = {"R1": 8, "R3": 4 * N_VAL[ds], "R6": 50 * 15 * NODE_B, "R7": 50 * 15 * NODE_B + 8 * dE,
                      "R8": 4 * (3 * dE + 3) + 8 * dE}
        for pair in ("A", "B", "C"):
            e = fp[(fp.dataset == ds) & (fp.pair == pair) & (fp.model == "ef_ev")]
            c = fp[(fp.dataset == ds) & (fp.pair == pair) & (fp.model == "cf_cv")]
            g = fp[(fp.dataset == ds) & (fp.pair == pair) & (fp.model == "ef_cv")]
            if e.empty:
                continue
            e, c, g = e.iloc[0], c.iloc[0], g.iloc[0]

            def pbytes(r):
                return 4 * r.params if r.unit == "MAC" else NODE_B * r.params
            row = dict(dataset=ds, pair=pair, edge_family=e.family, feature_extraction_ops=fx,
                       feature_extraction_ops_all_sensors=fx_all,
                       raw_channel_samples=sum(L.values()), window_buffer_B=buf, fft_workspace_B=fftw,
                       edge_model_ops=float(e.macs), edge_model_unit=e.unit, edge_params=int(e.params),
                       edge_param_bytes_deploy=pbytes(e), edge_pickle_bytes=int(e.bytes),
                       gateway_model_ops=float(g.macs), gateway_param_bytes_deploy=pbytes(g),
                       cloud_model_ops=float(c.macs), cloud_param_bytes_deploy=pbytes(c), cloud_pickle_bytes=int(c.bytes))
            for k, v in routers_ops.items():
                row[f"router_{k}_ops"] = float(v); row[f"router_{k}_mem_B"] = float(router_mem[k])
            # pair C routers read tabular edge features too: extraction must also run
            row["edge_total_ops_R1"] = (fx + e.macs + routers_ops["R1"])
            row["edge_total_ops_R6"] = (fx + e.macs + routers_ops["R6"]) if pair != "C" else (fx + e.macs + routers_ops["R6"])
            cost.append(row)
            # four architectures
            def ba(q):
                r = pdt[(pdt.dataset == ds) & (pdt.pair == pair) & (pdt.quantity == q) & pdt.router.isna()]
                return r.iloc[0] if len(r) else None
            for lab, q, f in (("edge only", "ba_ef_ev", 0.0), ("gateway, small model on all sensors", "ba_ef_cv", None),
                              ("full cloud", "ba_cf_cv", 1.0)):
                r = ba(q)
                if r is None:
                    continue
                arch.append(dict(dataset=ds, pair=pair, architecture=lab, budget=f, router="", mean=r["mean"],
                                 ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                                 uplink_features_B_per_h=0.0 if f is None else (f or 0.0) * wph * fc,
                                 uplink_raw_B_per_h=0.0 if f is None else (raw_all_h if f == 1.0 else 0.0),
                                 local_features_B_per_h=wph * fc if f is None else 0.0,
                                 cloud_inferences_per_h=0.0 if f is None else (f or 0.0) * wph,
                                 edge_ops=fx + e.macs if lab == "edge only" else (fx_all + g.macs if f is None else np.nan)))
            for f in BUDGETS:
                sub = rt[(rt.dataset == ds) & (rt.pair == pair) & (rt.quantity == "ba_at_budget") & np.isclose(rt.budget, f)
                         & ~rt.router.isin(["oracle"])]
                for router in ("R1", "R6"):
                    r = sub[sub.router == router]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    arch.append(dict(dataset=ds, pair=pair, architecture="edge + escalation", budget=f, router=router,
                                     mean=r["mean"], ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                                     uplink_features_B_per_h=f * wph * fc,
                                     uplink_raw_B_per_h=min(f * wph * pl["raw_cloud"], raw_all_h),
                                     local_features_B_per_h=0.0, cloud_inferences_per_h=f * wph,
                                     edge_ops=fx + e.macs + routers_ops[router]))
                best = sub[~sub.router.isin(["R0", "R9"])].sort_values("mean").iloc[-1]
                arch.append(dict(dataset=ds, pair=pair, architecture="edge + escalation (best router, chosen on test)",
                                 budget=f, router=best.router, mean=best["mean"], ci_lo=best.ci_lo, ci_hi=best.ci_hi,
                                 uplink_features_B_per_h=f * wph * fc,
                                 uplink_raw_B_per_h=min(f * wph * pl["raw_cloud"], raw_all_h),
                                 local_features_B_per_h=0.0, cloud_inferences_per_h=f * wph, edge_ops=np.nan))
    pd.DataFrame(upl).to_csv(X.OUT / "uplink_per_hour.csv", index=False)
    pd.DataFrame(cost).to_csv(X.OUT / "edge_cost.csv", index=False)
    a = pd.DataFrame(arch)
    a.to_csv(X.OUT / "architectures.csv", index=False)
    pd.set_option("display.width", 250)
    u = pd.DataFrame(upl)
    print(u[u.budget.isin([0.1, 1.0])][["dataset", "budget", "windows_per_hour", "pull_features_B_per_h", "pull_raw_B_per_h",
                                         "stream_features_B_per_h", "stream_raw_B_per_h", "full_cloud_raw_B_per_h",
                                         "cloud_inferences_per_h"]].round(1).to_string())
    print(pd.DataFrame(cost)[["dataset", "pair", "feature_extraction_ops", "edge_model_ops", "edge_param_bytes_deploy",
                              "edge_pickle_bytes", "window_buffer_B", "router_R6_ops", "router_R3_mem_B"]].to_string())
    print(a[(a.pair != "C")].round(3).to_string())


if __name__ == "__main__":
    main()
