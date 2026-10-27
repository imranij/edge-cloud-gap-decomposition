"""Fig. budget: deferral curves, 2x2 by dataset. Balanced accuracy against escalation fraction (0-0.5).

For every split, BA(f) = BA(deployed edge) + G(round(f n)), where G is the expected cumulative gain from
escalating the top-ranked windows (escalation.metrics.gain_curve; ties in expectation) and g_dep, the
router scores and the oracle ranking come from results/<dataset>/<pair>/arrays_<split>.npz. BA(edge) and
BA(cloud) per split come from split_<split>.parquet. Curves are means over splits; the band (R6 only) is the
Nadeau-Bengio corrected 95% CI of R6 gain over random, paired across splits. The curve values at the budgets of
routers.csv (0.05 ... 0.5) are checked against the stored ba_at_budget.
Pair A is drawn solid. Pair B/C (when present with >= MIN_SPLITS splits) add R6 as a thin dashed/dotted line.
drawn at the elsarticle text width (common.W_MAIN) so the 8-pt fonts print at 8 pt;
the oracle ranking is dotted and black and R1 solid dark grey with markers, so the two differ in line style and not
only in grey level; the band is the corrected 95% CI of R1's gain over random (R1 is the router the guidance
recommends); the pair B and C curves, which start from their own edges, moved to the supplementary figure
fig_budget_b.pdf (pair B, every router of the main figure; written by main(pair="B")).
"""
from __future__ import annotations
import numpy as np
from matplotlib.lines import Line2D
import common as C
from escalation.metrics import gain_curve, escalation_gain  # noqa: F401  (used through g_dep)

FGRID = np.linspace(0.0, 0.5, 101)
SERIES = [  # (label, router key, colour, linewidth, band, linestyle, marker)
    ("random", "R0", "#a9a8a0", 1.0, False, "-", None),
    ("confidence (R1)", "R1", C.GREY_DARK, 1.1, True, "-", "o"),
    ("gain regressor (R6)", "R6", C.ACCENT, 1.3, False, "-", None),
    ("oracle ranking", "oracle", C.INK, 1.2, False, (0, (1, 1.3)), None),
]
MARK_EVERY = 10          # one marker every 0.05 of escalation fraction


def curves(ds, pair):
    """Returns dict router -> (J x len(FGRID)) BA matrix, plus edge/cloud BA vectors and n_train, n_test."""
    ba_e = C.split_values(ds, pair, "ba_ef_ev"); ba_c = C.split_values(ds, pair, "ba_cf_cv")
    splits = [s for s in ba_e.index if (C.RES / ds / pair / f"arrays_{s}.npz").exists()]
    out = {k: [] for _, k, *_ in SERIES}
    for s in splits:
        z = C.load_arrays(ds, pair, s)
        g = z["g_dep"]; n = len(g)
        ks = np.round(FGRID * n).astype(int)
        for _, k, *_ in SERIES:
            if k == "R0":                                  # expected random: all windows tied
                G = FGRID * g.sum()
            else:
                G = gain_curve(g, z[f"score_{k}"])[ks]
            out[k].append(ba_e[s] + G)
    return {k: np.array(v) for k, v in out.items()}, ba_e.loc[splits].to_numpy(), ba_c.loc[splits].to_numpy()


def check_against_stored(ds, pair, cur):
    """Mean curve at the stored budgets equals the mean of ba_at_budget in the parquet files."""
    worst = 0.0
    for k in ("R1", "R6", "oracle"):
        for b in (0.05, 0.1, 0.2, 0.3, 0.5):
            st = C.split_values(ds, pair, "ba_at_budget", router=k, budget=b)
            if st.empty:
                continue
            mine = cur[k][:, np.argmin(np.abs(FGRID - b))].mean()
            worst = max(worst, abs(mine - st.mean()))
    return worst


def main(pair="A"):
    C.setup_style()
    sz = C.sizes()
    fig, axes = C.plt.subplots(2, 2, figsize=(C.W_MAIN, 4.6),
                               gridspec_kw=dict(left=0.095, right=0.985, top=0.885, bottom=0.085,
                                                wspace=0.24, hspace=0.32))
    tags = "abcd"
    for i, (ax, ds) in enumerate(zip(axes.flat, C.DATASETS)):
        cur, ba_e, ba_c = curves(ds, pair)
        dev = check_against_stored(ds, pair, cur)
        print(f"  {ds}: max |curve - stored ba_at_budget| = {dev:.2e}")
        if dev > 1e-6:
            print(f"  WARNING {ds}: deferral curve disagrees with stored ba_at_budget")
        r = sz[(sz.dataset == ds) & (sz.pair == pair)].iloc[0]
        J = cur["R6"].shape[0]
        # cloud reference
        ax.axhline(ba_c.mean(), color=C.INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
        for lab, k, col, lw, band, ls, mk in SERIES:
            m = cur[k].mean(0)
            if band and ds not in C.NO_WHISKER:            # CI of the gain over random (paired across splits)
                hw = C.nb_half_width((cur[k] - cur["R0"]).std(0, ddof=1), J, r.n_train, r.n_test)
                ax.fill_between(FGRID, m - hw, m + hw, color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(FGRID, m, color=col, lw=lw, ls=ls, marker=mk, markevery=MARK_EVERY, markersize=2.6,
                    zorder=5 if k == "oracle" else (3 if k != "R6" else 4), solid_capstyle="round")
        means = [cur[k].mean(0) for _, k, *_ in SERIES] + [np.full_like(FGRID, ba_c.mean())]
        ylo, yhi = min(m.min() for m in means), max(m.max() for m in means)
        pad = 0.04 * (yhi - ylo)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(ylo - pad, min(1.0, yhi + pad))
        ax.xaxis.set_major_locator(C.matplotlib.ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(C.matplotlib.ticker.MaxNLocator(5))
        C.panel_tag(ax, f"({tags[i]}) {C.DS_LABEL[ds]}" + (" (no CI band)" if ds in C.NO_WHISKER else ""), y=1.02)
        ax.set_xlabel("Escalation fraction" if i >= 2 else "")
    fig.supylabel("Balanced accuracy", fontsize=C.FS, x=0.012)
    handles = [Line2D([0], [0], color=c, lw=lw, ls=ls, marker=mk, markersize=2.6, label=l)
               for l, _, c, lw, _, ls, mk in SERIES]
    handles.append(Line2D([0], [0], color=C.INK2, lw=0.8, ls=(0, (4, 2)), label="cloud model"))
    handles.append(C.matplotlib.patches.Patch(color=C.GREY_DARK, alpha=0.16, label="95% CI, R1 gain over random"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.0),
               fontsize=C.FS, columnspacing=1.0, handlelength=2.0)
    C.save(fig, "fig_budget" if pair == "A" else f"fig_budget_{pair.lower()}")


if __name__ == "__main__":
    main()
    main("B")
