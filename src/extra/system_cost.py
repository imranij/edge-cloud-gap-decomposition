"""Whole-system cost per window for each architecture, paired intervals for the
architecture comparison, and the capture range translated into uplink per hour.
Post-plan analysis, not pre-specified. Analytic only: nothing is measured on hardware and
no model is refitted. Inputs: results/extra/edge_cost.csv, uplink_per_hour.csv, realised_budget.csv
(src/extra/cost_model.py and realised_budget.py) and the stored per-split balanced accuracies.

    python src/extra/system_cost.py

Architectures (pair A models unless stated; per window; 'ops' are the operation counts of
cost_model.py, i.e. rough analytic estimates):
  edge only        edge node: extraction on the edge channels + edge model.
  escalation 10%   edge only + router R6 at the edge node, plus, in the pull flow,
                   (a) on-demand: each cloud-only node double-buffers its raw window and extracts
                       features only for escalated windows (0.10 x cloud-only extraction), or
                   (b) continuous: each cloud-only node extracts features from every window and
                       keeps the latest feature vector;
                   plus 0.10 x one cloud inference and 0.10 trigger messages from the edge.
  gateway          extraction on every channel + the edge family on all sensors (B_eC) at a node
                   where all sensors converge; nothing is sent upstream, and the local network
                   carries every window's features.
  full cloud       extraction on every channel + the cloud model on every window; every window's
                   cloud features go upstream.
Memory: edge-node working memory = raw window buffer + FFT workspace + model parameters (+ R6);
cloud-only nodes in the pull flow hold two raw windows (on-demand) or one raw window and one FFT
workspace sized like the edge's (continuous; an approximation).
Outputs: results/extra/system_cost.csv, results/extra/arch_paired.csv,
results/extra/capture_range_uplink.csv, results/extra/latency_budget.csv (analytic latency terms)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

F = 0.10


def system_cost():
    e = pd.read_csv(X.OUT / "edge_cost.csv")
    u = pd.read_csv(X.OUT / "uplink_per_hour.csv")
    rows = []
    for ds in X.DATASETS:
        c = e[(e.dataset == ds) & (e.pair == "A")].iloc[0]
        uu = u[(u.dataset == ds) & np.isclose(u.budget, F)].iloc[0]
        ext_e, ext_all = float(c.feature_extraction_ops), float(c.feature_extraction_ops_all_sensors)
        ext_co = ext_all - ext_e
        raw_co = float(uu.raw_cloud_window_B - uu.raw_edge_window_B)
        edge_mem = float(c.window_buffer_B + c.fft_workspace_B + c.edge_param_bytes_deploy)
        wph = float(uu.windows_per_hour)
        base = dict(dataset=ds, windows_per_hour=wph)
        rows.append(dict(base, architecture="edge only", ops_edge_node=ext_e + c.edge_model_ops, ops_other_nodes=0.0,
                         ops_cloud=0.0, mem_edge_node_B=edge_mem, mem_other_nodes_B=0.0,
                         uplink_B_per_h=0.0, local_B_per_h=0.0, messages_per_h=0.0))
        esc_edge_ops = ext_e + c.edge_model_ops + c.router_R6_ops
        esc_edge_mem = edge_mem + c.router_R6_mem_B
        rows.append(dict(base, architecture="escalation 10%, pull, on-demand extraction",
                         ops_edge_node=esc_edge_ops, ops_other_nodes=F * ext_co, ops_cloud=F * c.cloud_model_ops,
                         mem_edge_node_B=esc_edge_mem, mem_other_nodes_B=2 * raw_co,
                         uplink_B_per_h=float(uu.pull_features_B_per_h), local_B_per_h=0.0,
                         messages_per_h=F * wph))
        rows.append(dict(base, architecture="escalation 10%, pull, continuous extraction",
                         ops_edge_node=esc_edge_ops, ops_other_nodes=ext_co, ops_cloud=F * c.cloud_model_ops,
                         mem_edge_node_B=esc_edge_mem, mem_other_nodes_B=raw_co + float(c.fft_workspace_B),
                         uplink_B_per_h=float(uu.pull_features_B_per_h), local_B_per_h=0.0,
                         messages_per_h=F * wph))
        rows.append(dict(base, architecture="gateway", ops_edge_node=ext_all + c.gateway_model_ops, ops_other_nodes=0.0,
                         ops_cloud=0.0, mem_edge_node_B=float(uu.raw_cloud_window_B + c.fft_workspace_B
                                                              + c.gateway_param_bytes_deploy),
                         mem_other_nodes_B=0.0, uplink_B_per_h=0.0,
                         local_B_per_h=float(uu.gateway_local_features_B_per_h), messages_per_h=0.0))
        rows.append(dict(base, architecture="full cloud", ops_edge_node=0.0, ops_other_nodes=ext_all,
                         ops_cloud=float(c.cloud_model_ops), mem_edge_node_B=0.0,
                         mem_other_nodes_B=float(uu.raw_cloud_window_B), uplink_B_per_h=float(uu.full_cloud_features_B_per_h),
                         local_B_per_h=0.0, messages_per_h=wph))
    o = pd.DataFrame(rows)
    o["ops_total"] = o.ops_edge_node + o.ops_other_nodes + o.ops_cloud
    o.to_csv(X.OUT / "system_cost.csv", index=False)
    return o


def arch_paired():
    df = X.long_results()
    d = df[df.dataset.isin(X.DATASETS) & df.pair.isin(["A", "B"]) & (df.h2_q == 1.0)]
    w = X.wide(d, ["ba_ef_ev", "ba_ef_cv", "ba_cf_cv"])
    for r in ("R1", "R6"):
        for f in (0.10, 0.30):
            s = X.per_split(d, "ba_at_budget", r, budget=f, name=f"esc_{r}_{int(f * 100)}").reset_index()
            w = w.merge(s, on=["dataset", "pair", "split", "h2_q"], how="left")
    rb = pd.read_csv(X.OUT / "realised_budget_cells.csv")
    rb = rb[(rb.router == "R1") & np.isclose(rb.budget, 0.10)][["dataset", "pair", "split", "ba_realised"]]
    w = w.merge(rb.rename(columns={"ba_realised": "esc_R1_10_realised"}), on=["dataset", "pair", "split"], how="left")
    rows = []
    comps = [("gateway - edge", "ba_ef_cv", "ba_ef_ev"), ("gateway - escalation R1 10%", "ba_ef_cv", "esc_R1_10"),
             ("gateway - escalation R6 10%", "ba_ef_cv", "esc_R6_10"), ("gateway - escalation R6 30%", "ba_ef_cv", "esc_R6_30"),
             ("gateway - cloud", "ba_ef_cv", "ba_cf_cv"), ("escalation R1 10% - edge", "esc_R1_10", "ba_ef_ev"),
             ("escalation R6 10% - edge", "esc_R6_10", "ba_ef_ev"),
             ("escalation R1 10% realised - edge", "esc_R1_10_realised", "ba_ef_ev")]
    for (ds, pair), g in w.groupby(["dataset", "pair"]):
        for lab, a, b in comps:
            t = X.nb((g[a] - g[b]).to_numpy(float), ds, pair)
            rows.append(dict(dataset=ds, pair=pair, comparison=lab, J=t["J"], mean=t["mean"], ci_lo=t["ci_lo"],
                             ci_hi=t["ci_hi"], p_two_sided=t["p"]))
        t = X.nb(g["esc_R1_10_realised"].to_numpy(float), ds, pair)
        rows.append(dict(dataset=ds, pair=pair, comparison="escalation R1 10% realised (level)", J=t["J"],
                         mean=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], p_two_sided=np.nan))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "arch_paired.csv", index=False)
    return o


def capture_range():
    u = pd.read_csv(X.OUT / "uplink_per_hour.csv")
    rows = []
    for ds in X.DATASETS:
        a = u[(u.dataset == ds) & np.isclose(u.budget, 0.05)].iloc[0]
        b = u[(u.dataset == ds) & np.isclose(u.budget, 0.30)].iloc[0]
        rows.append(dict(dataset=ds, pull_features_B_per_h_at_5=a.pull_features_B_per_h,
                         pull_features_B_per_h_at_30=b.pull_features_B_per_h,
                         pull_raw_B_per_h_at_5=a.pull_raw_B_per_h, pull_raw_B_per_h_at_30=b.pull_raw_B_per_h,
                         cloud_calls_per_h_at_5=a.cloud_inferences_per_h, cloud_calls_per_h_at_30=b.cloud_inferences_per_h,
                         full_cloud_features_B_per_h=a.full_cloud_features_B_per_h,
                         raw_stream_all_B_per_h=a.raw_stream_all_B_per_h,
                         features_expand_raw=bool(a.full_cloud_features_B_per_h > a.raw_stream_all_B_per_h)))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "capture_range_uplink.csv", index=False)
    return o


def latency_budget():
    """Analytic latency terms per dataset. Only the window term and the payload-derived link
    rates are numeric; processing, trigger, round-trip and reply times depend on hardware and links
    that we did not measure and are left symbolic in the paper. The link rate below is the rate at
    which the escalated payload (cloud-only part plus the edge part, as in the pull flow) can be
    uploaded within one window stride, so that escalations do not queue up."""
    u = pd.read_csv(X.OUT / "uplink_per_hour.csv")
    rows = []
    for ds in X.DATASETS:
        x = u[(u.dataset == ds) & np.isclose(u.budget, F)].iloc[0]
        stride = float(x.stride_s)
        rows.append(dict(dataset=ds, window_s=float(x.window_s), stride_s=stride,
                         escalations_per_h_at_10=float(x.cloud_inferences_per_h),
                         mean_s_between_escalations_at_10=3600.0 / float(x.cloud_inferences_per_h),
                         features_cloud_B=float(x.features_cloud_B), raw_cloud_window_B=float(x.raw_cloud_window_B),
                         link_Bps_features_within_stride=float(x.features_cloud_B) / stride,
                         link_bitps_features_within_stride=8 * float(x.features_cloud_B) / stride,
                         link_Bps_raw_within_stride=float(x.raw_cloud_window_B) / stride,
                         link_bitps_raw_within_stride=8 * float(x.raw_cloud_window_B) / stride))
    o = pd.DataFrame(rows)
    o.to_csv(X.OUT / "latency_budget.csv", index=False)
    return o


def main():
    pd.set_option("display.width", 250)
    print(latency_budget().round(1).to_string())
    print(system_cost().round(1).to_string())
    print(arch_paired().round(3).to_string())
    print(capture_range().to_string())


if __name__ == "__main__":
    main()
