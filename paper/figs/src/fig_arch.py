"""Supplementary Fig. arch: hydraulic rig, pair A,
balanced accuracy against the per-hop rate for each architecture, on the first ten block-grouped splits.

Accuracy (means over the ten splits; results/extra/r6_fs1_designs.csv and r4_arch_deployable_splits.csv, both written
by evaluation-only scripts in src/extra/):
  edge only            B_eE;
  escalation (R1v)     causal-threshold R1 at a nominal 10% with the validation-fixed threshold;
  gateway, small       B_eC (small model on every sensor, co-located gateway);
  gateway, cloud       B_cC of the block-grouped reference run (cloud model on every sensor, co-located gateway);
  FS1 wired            small model on the edge sensors and FS1 (post hoc);
  edge + FS1 stream    cloud family on the edge sensors and FS1 (post hoc);
  full cloud stream    B_cC, every cloud feature streamed.
Rates (bit/s, analytic payload; results/extra/r5_hop_loads.csv, rows appended by r6_systems.py): backhaul
and local-network aggregate. Escalation is drawn at its rate provisioned for a 95th-percentile delay of 10 min and
at the one-window burst rate, both from the pre-specified R1r series (Table S-delay); the band between them is
shaded on the backhaul panel. Nothing here was measured on hardware.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import common as C

EXTRA = C.ROOT / "results" / "extra"


def main():
    C.setup_style()
    fs = pd.read_csv(EXTRA / "r6_fs1_designs.csv")
    f1 = fs[(fs.variant == "hyd_blk_e_fs1") & (fs.pair == "A")].iloc[0]
    dep = pd.read_csv(EXTRA / "r4_arch_deployable_splits.csv")
    gw = dep[(dep.variant == "hyd_blocks") & (dep.pair == "A") & (dep.split < 10)]
    assert len(gw) == 10 and abs(gw.edge.mean() - f1.edge) < 1e-9 and abs(gw.R1v.mean() - f1.R1v) < 1e-9
    L = pd.read_csv(EXTRA / "r5_hop_loads.csv")
    L = L[L.pair == "A"].set_index("architecture")
    pts = [  # label, BA, backhaul, local, marker, colour
        ("edge only", f1.edge, 0.0, 0.0, "o", C.GREY_MID),
        ("gateway, small model", gw.gateway.mean(), L.loc["gateway, node features, co-located with edge node", "backhaul_bit_s"],
         L.loc["gateway, node features, co-located with edge node", "local_bit_s"], "s", C.GREY_DARK),
        ("gateway, cloud model", f1.ref_cloud, L.loc["gateway, cloud model, co-located with edge node", "backhaul_bit_s"],
         L.loc["gateway, cloud model, co-located with edge node", "local_bit_s"], "s", C.ACCENT),
        ("FS1 wired to edge (post hoc)", f1.wired_edge_node, 0.0, L.loc["FS1 wired to edge node (local only)", "local_bit_s"],
         "D", C.ACCENT),
        ("edge + FS1 stream (post hoc)", f1.cloud_family, L.loc["edge and FS1 features streamed", "backhaul_bit_s"],
         L.loc["edge and FS1 features streamed", "local_bit_s"], "^", C.ACCENT),
        ("full feature stream", f1.ref_cloud, L.loc["full feature stream", "backhaul_bit_s"],
         L.loc["full feature stream", "local_bit_s"], "v", C.INK),
    ]
    esc = [(L.loc["escalation, provisioned for q95 delay <= 10 min", "backhaul_bit_s"], "10-min"),
           (L.loc["escalation, one-window deadline (burst rate)", "backhaul_bit_s"], "one window")]
    fig, axes = C.plt.subplots(1, 2, figsize=(C.W_MAIN, 3.3), sharey=True,
                               gridspec_kw=dict(left=0.10, right=0.985, top=0.72, bottom=0.15, wspace=0.08))
    for k, (ax, hop) in enumerate(zip(axes, ("backhaul", "local"))):
        if hop == "backhaul":
            ax.axvspan(esc[0][0], esc[1][0], color=C.GREY_LIGHT, alpha=0.45, lw=0, zorder=0)
        for lab, ba, bh, lo, mk, col in pts:
            x = bh if hop == "backhaul" else lo
            ax.plot([x], [ba], marker=mk, color=col, ms=5, ls="none", zorder=3)
        ax.plot([e[0] for e in esc], [f1.R1v] * 2, color=C.INK2, lw=1.0, marker="o", mfc="white", ms=4.5, zorder=3)
        ax.set_xlim(-6, 140)
        ax.set_xlabel(f"{'Backhaul' if hop == 'backhaul' else 'Local network'} rate (bit/s)")
        C.panel_tag(ax, f"({'ab'[k]}) {hop}", y=1.02)
    axes[0].set_ylabel("Balanced accuracy")
    handles = [C.matplotlib.lines.Line2D([0], [0], marker=mk, color=col, ls="none", ms=5, label=lab)
               for lab, _, _, _, mk, col in pts]
    handles.append(C.matplotlib.lines.Line2D([0], [0], color=C.INK2, marker="o", mfc="white", ms=4.5,
                                             label="escalation, 10-min / one-window provisioning"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.54, 1.0), fontsize=C.FS - 0.5,
               columnspacing=0.8, handletextpad=0.3)
    C.save(fig, "fig_arch")


if __name__ == "__main__":
    main()
