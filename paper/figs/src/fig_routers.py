"""Fig. routers: dot plot of identifiability C (corrected 95% CI) for routers R0-R9, one panel per dataset.

Source: results/summary/routers.csv (quantity C). Rows are sorted by the mean of C over the four datasets
(pair A), the same order in every panel. Reference lines: C = 0.5 (random ranking) and C = 1 (oracle
ranking). Pair B/C (if present with >= MIN_SPLITS splits) are drawn as hollow markers just below the
pair A marker; routers R7, R8 exist for pair A only. CIs are cut at the C = 1 ceiling.
"""
from __future__ import annotations
import numpy as np
from matplotlib.lines import Line2D
import common as C

PAIR_MARK = {"A": "o", "B": "D", "C": "s"}
PAIR_STEP = 0.34          # vertical offset between pairs in the same router row


def main():
    C.setup_style()
    r = C.summary("routers")
    r = r[(r.quantity == "C") & r.router.str.match(r"^R\d$", na=False)]
    a = r[r.pair == "A"]
    order = a.groupby("router")["mean"].mean().sort_values(ascending=False).index.tolist()   # best first
    ypos = {k: i for i, k in enumerate(order)}
    fig, axes = C.plt.subplots(1, 4, figsize=(C.W_MAIN, 3.3), sharey=True,
                               gridspec_kw=dict(left=0.205, right=0.995, top=0.80, bottom=0.135, wspace=0.10))
    tags = "abcd"
    for i, (ax, ds) in enumerate(zip(axes, C.DATASETS)):
        ax.axvline(0.5, color=C.INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.axvline(1.0, color=C.INK, lw=0.8, ls=(0, (1, 1.6)), zorder=1)
        pairs = ["A"] + C.optional_pairs(ds)
        for pi, p in enumerate(pairs):
            dy = (pi - (len(pairs) - 1) / 2) * PAIR_STEP
            d = r[(r.dataset == ds) & (r.pair == p)]
            for _, row in d.iterrows():
                y = ypos[row.router] + dy
                col = C.ACCENT if row.router == "R6" else C.INK2
                lo, hi = max(row.ci_lo, 0.0), min(row.ci_hi, 1.0)
                if ds not in C.NO_WHISKER:
                    ax.plot([lo, hi], [y, y], color=col, lw=0.9, zorder=3, solid_capstyle="butt")
                if p == "A":
                    ax.plot(row["mean"], y, "o", ms=4.2, mfc=col, mec=C.SURF, mew=0.8, zorder=4)
                else:
                    ax.plot(row["mean"], y, PAIR_MARK[p], ms=3.6, mfc=C.SURF, mec=col, mew=0.9, zorder=4)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(len(order) - 0.45 + 0.3, -0.6)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(False); ax.grid(True, axis="x")
        ax.set_yticks(range(len(order)))
        if i == 0:
            ax.set_yticklabels([f"{k}  {C.ROUTER_NAME[k]}" for k in order], fontsize=C.FS)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(i == 0)
        C.panel_tag(ax, f"({tags[i]}) {C.DS_LABEL[ds]}" + (" (no CIs)" if ds in C.NO_WHISKER else ""), y=1.02)
        ax.set_xlabel("Identifiability $C$")
    handles = [Line2D([0], [0], color=C.INK2, lw=0.8, ls=(0, (4, 2)), label="random ranking ($C=0.5$)"),
               Line2D([0], [0], color=C.INK, lw=0.9, ls=(0, (1, 1.6)), label="oracle ranking ($C=1$)"),
               Line2D([0], [0], color=C.ACCENT, lw=0.9, marker="o", ms=4.2, label="R6, headline router"),
               Line2D([0], [0], color=C.INK2, lw=0.9, marker="o", ms=4.2, label="other routers, pair A")]
    for p in sorted({q for d in C.DATASETS for q in C.optional_pairs(d)}):
        handles.append(Line2D([0], [0], color=C.INK2, lw=0.9, marker=PAIR_MARK[p], ms=3.6, mfc=C.SURF,
                              label=f"{C.PAIR_NAME[p]} (hollow)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.0),
               fontsize=C.FS, columnspacing=0.9, handlelength=1.6)
    C.save(fig, "fig_routers")


if __name__ == "__main__":
    main()
