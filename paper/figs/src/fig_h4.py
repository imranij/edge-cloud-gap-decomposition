"""Fig. h4: capture advantage of the learned router over confidence, dcap = capture(R6) - capture(R1),
against rho_pos (results/summary/h4_data.csv: one point per dataset x pair x split). Pair A filled, pair B/C
hollow (the H4 model pools pairs A and B). Lines: the pooled mixed-model slope (h4_model.csv) anchored at
each dataset's mean (dataset random intercepts). Reference: dcap = 0 (R6 captures as much as R1).
"""
from __future__ import annotations
import numpy as np
from matplotlib.lines import Line2D
import common as C

PAIR_MARK_HOLLOW = {"B": "o", "C": "s"}


def main():
    C.setup_style()
    d = C.summary("h4_data")
    mod = C.summary("h4_model").iloc[0]
    fig, ax = C.plt.subplots(figsize=(C.W1, 2.9), gridspec_kw=dict(left=0.15, right=0.985, top=0.78, bottom=0.135))
    ax.axhline(0.0, color=C.INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ymin = d.dcap.min()
    for ds in C.DATASETS:
        col, mk = C.DS_COLOR[ds], C.DS_MARKER[ds]
        g = d[d.dataset == ds]
        a = g[g.pair == "A"]
        ax.plot(a.rho_pos, a.dcap, mk, ms=3.6, mfc=col, mec=C.SURF, mew=0.5, alpha=0.9, zorder=3)
        for p in g.pair.unique():
            if p == "A":
                continue
            b = g[g.pair == p]
            ax.plot(b.rho_pos, b.dcap, mk, ms=3.8, mfc=C.SURF, mec=col, mew=0.8, zorder=3)
        x0, x1 = g.rho_pos.min(), g.rho_pos.max()
        xm, ym = g.rho_pos.mean(), g.dcap.mean()
        ax.plot([x0, x1], [ym + mod.slope * (x0 - xm), ym + mod.slope * (x1 - xm)], color=col, lw=1.1, zorder=4)
    ax.set_xlabel(r"Spearman $\rho_{\mathrm{pos}}$, R1 score vs. gain (positives)")
    ax.set_ylabel(r"$\Delta$capture = capture(R6) $-$ capture(R1)")
    ax.xaxis.set_major_locator(C.matplotlib.ticker.MaxNLocator(6))
    ax.set_ylim(top=1.0 if d.dcap.max() <= 1.0 else d.dcap.max() * 1.05)
    handles = [Line2D([0], [0], color=C.DS_COLOR[x], marker=C.DS_MARKER[x], lw=1.1, ms=4, label=C.DS_LABEL[x]) for x in C.DATASETS]
    handles.append(Line2D([0], [0], color=C.INK2, lw=0.8, ls=(0, (4, 2)), label="R6 = R1"))
    handles.append(Line2D([0], [0], color=C.INK2, lw=0, marker="o", ms=3.8, mfc=C.SURF, mew=0.8, label="pair B (hollow)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.57, 1.0), fontsize=C.FS,
               columnspacing=0.9, handlelength=1.5)
    nul = C.pd.read_csv(C.ROOT / "results" / "extra" / "h4_null_slopes.csv").set_index("lhs")   #
    r0, r9 = nul.loc["dcap_R0"], nul.loc["dcap_R9"]
    lab = (f"pooled slope {mod.slope:.2f}\n95% CI [{mod.ci_lo:.2f}, {mod.ci_hi:.2f}]\n$n$ = {int(mod.n)} splits\n"
           f"R0 for R6: {r0.slope:.2f}\nR9 for R6: {r9.slope:.2f}").replace("-", "\u2212")
    ax.text(0.98, 0.98, lab, transform=ax.transAxes, ha="right", va="top", fontsize=C.FS, linespacing=1.15)
    C.save(fig, "fig_h4")


if __name__ == "__main__":
    main()
