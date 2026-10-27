"""Fig. h2: identifiability of R6 against labelled fault units (log2 of positive training windows).

Per dataset and H2 level q in {0.1, 0.25, 0.5, 1}: the mean C(R6) over splits and its corrected 95% CI
(results/summary/per_dataset_h2.csv for q < 1, per_dataset.csv for q = 1), plotted at the mean log2 number
of positive training windows (h2_by_level.csv). Light dots are the individual split cells (split_*.parquet),
the points that enter the mixed model. The grey lines carry the pooled mixed-model slope (h2_model.csv),
anchored at each dataset's mean (dataset random intercepts). Reference line: C = 0.5 (random ranking).
Pair A only (H2 is a pair A test).
"""
from __future__ import annotations
import numpy as np
from matplotlib.lines import Line2D
import common as C


def main():
    C.setup_style()
    by = C.summary("h2_by_level")
    mod = C.summary("h2_model").iloc[0]
    pd_h2, pd_main = C.summary("per_dataset_h2"), C.summary("per_dataset")
    long = C.long_results()
    cells = long[(long.pair == "A") & (long.quantity == "C") & (long.router == "R6")].copy()
    cells["x"] = np.log2(cells.n_pos_train.clip(lower=1))
    fig, axes = C.plt.subplots(1, 4, figsize=(C.W_MAIN, 2.6), sharey=True,
                               gridspec_kw=dict(left=0.085, right=0.995, top=0.775, bottom=0.17, wspace=0.08))
    rng = np.random.default_rng(0)
    tags = "abcd"
    for i, (ax, ds) in enumerate(zip(axes, C.DATASETS)):
        col, mk = C.DS_COLOR[ds], C.DS_MARKER[ds]
        ax.axhline(0.5, color=C.INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
        c = cells[cells.dataset == ds]
        ax.plot(c.x + rng.normal(0, 0.02, len(c)), c.value, mk, ms=2.0, mfc=col, mec="none", alpha=0.25, zorder=2)
        b = by[by.dataset == ds].sort_values("h2_q")
        xs, ys, los, his = [], [], [], []
        for _, r in b.iterrows():
            if np.isclose(r.h2_q, 1.0):
                t = pd_main[(pd_main.dataset == ds) & (pd_main.pair == "A") & (pd_main.quantity == "C") & (pd_main.router == "R6")].iloc[0]
            else:
                t = pd_h2[(pd_h2.dataset == ds) & (pd_h2.pair == "A") & np.isclose(pd_h2.h2_q, r.h2_q) & (pd_h2.quantity == "C") & (pd_h2.router == "R6")].iloc[0]
            xs.append(c[np.isclose(c.h2_q, r.h2_q)].x.mean()); ys.append(t["mean"]); los.append(t.ci_lo); his.append(t.ci_hi)
        xs, ys = np.array(xs), np.array(ys)
        x0, x1 = xs.min() - 0.15, xs.max() + 0.15
        ax.plot([x0, x1], [ys.mean() + mod.slope * (x0 - xs.mean()), ys.mean() + mod.slope * (x1 - xs.mean())],
                color=C.INK, lw=1.0, zorder=3)
        # SEU is drawn without intervals, as in the other figures (four healthy recordings)
        for x_, lo, hi in zip(xs, los, his):
            if ds == "seu":
                continue
            ax.plot([x_, x_], [max(lo, 0), min(hi, 1)], color=col, lw=0.9, zorder=4, solid_capstyle="butt")
        ax.plot(xs, ys, mk, ms=4.4, mfc=col, mec=C.SURF, mew=0.7, zorder=5)
        ax.set_xlim(xs.min() - 0.7, xs.max() + 0.7); ax.set_ylim(-0.03, 1.03)
        ax.xaxis.set_major_locator(C.matplotlib.ticker.MaxNLocator(integer=True, nbins=4))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(False); ax.grid(True, axis="y")
        C.panel_tag(ax, f"({tags[i]}) {C.DS_LABEL[ds]}", y=1.0)
        if i == 0:
            ax.set_ylabel("$C$ of R6")
    fig.supxlabel("$\\log_2$ positive training windows", fontsize=C.FS, y=0.0)
    handles = [Line2D([0], [0], color=C.INK2, lw=0.8, ls=(0, (4, 2)), label="random ranking ($C=0.5$)"),
               Line2D([0], [0], color=C.INK, lw=1.0,
                      label=f"pooled slope {mod.slope:.3f} per doubling, 95% CI [{mod.ci_lo:.3f}, {mod.ci_hi:.3f}]"),
               Line2D([0], [0], color=C.INK2, marker="o", lw=0.9, ms=4.4, label="mean over splits, corrected 95% CI"),
               Line2D([0], [0], color=C.INK2, marker="o", lw=0, ms=2.2, alpha=0.4, label="single split")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=C.FS,
               columnspacing=1.2, handlelength=1.6)
    C.save(fig, "fig_h2")


if __name__ == "__main__":
    main()
