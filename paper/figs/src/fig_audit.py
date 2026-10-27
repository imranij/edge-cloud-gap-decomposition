"""Fig. audit: pre-specified routing audit. One point per audit cell (dataset x pair x split x H2 level).

x: validation lower bound of C for R6 (audit_C_lo; GO iff > 0.52). y: test capture of R6 over budgets
0.05-0.30 (quantity `capture`, router R6; outcome iff > 0.10). Cells come from results/summary/h5_cells.csv
(the cells analysed in H5); audit_C_lo and capture are read from the split_*.parquet files and the stored
call/outcome are re-derived and checked. Cells whose validation bound is undefined (too few positive groups
to cross-fit; forced NO-GO) sit in the left strip. Capture values below the axis floor are pinned to it
(downward triangles). Pair B/C cells appear automatically as hollow markers when present in h5_cells.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import common as C

GO, OUT = 0.52, 0.10
YLO, YHI = -1.1, 1.05


def build():
    cells = C.summary("h5_cells").copy()
    L = C.long_results()
    L = L[(L.router == "R6") & L.quantity.isin(["capture", "audit_C_lo"])]
    w = L.pivot_table(index=["dataset", "pair", "split", "h2_q"], columns="quantity", values="value", dropna=False, observed=True)
    w = w.reset_index()
    m = cells.merge(w, on=["dataset", "pair", "split", "h2_q"], how="left", validate="1:1")
    m["go_chk"] = (m.audit_C_lo > GO).astype(float)
    m["out_chk"] = (m.capture > OUT).astype(float)
    bad_go = int((m.go_chk != m.call).sum()); bad_out = int((m.out_chk != m.outcome).sum())
    if bad_go or bad_out:
        print(f"  WARNING: stored call/outcome disagree with re-derivation: go {bad_go}, outcome {bad_out}")
    return m


def main():
    C.setup_style()
    m = build()
    undefined = m.audit_C_lo.isna()
    fig = C.plt.figure(figsize=(C.W1, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 6.2], left=0.145, right=0.985, top=0.76, bottom=0.14, wspace=0.05)
    axu, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)
    rng = np.random.default_rng(1)

    def draw(a, x, y, ds, pair):
        yc = np.clip(y, YLO, YHI)
        below = y < YLO
        col, mk = C.DS_COLOR[ds], C.DS_MARKER[ds]
        kw = dict(ms=3.4, zorder=3)
        ok = ~below
        if pair == "A":
            a.plot(x[ok], yc[ok], mk, mfc=col, mec=C.SURF, mew=0.4, alpha=0.85, **kw)
        else:
            a.plot(x[ok], yc[ok], mk, mfc=C.SURF, mec=col, mew=0.8, **kw)
        if below.any():
            a.plot(x[below], yc[below], "v", mfc=col if pair == "A" else C.SURF, mec=col, mew=0.6, ms=3.8, zorder=3, clip_on=False)

    for ds in C.DATASETS:
        for pair in sorted(m.pair.unique()):
            g = m[(m.dataset == ds) & (m.pair == pair)]
            gd, gu = g[~g.audit_C_lo.isna()], g[g.audit_C_lo.isna()]
            if len(gd):
                draw(ax, gd.audit_C_lo.to_numpy(), gd.capture.to_numpy(), ds, pair)
            if len(gu):
                draw(axu, rng.uniform(-0.38, 0.38, len(gu)), gu.capture.to_numpy(), ds, pair)
    for a in (ax, axu):
        a.axhline(OUT, color=C.INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
        a.set_ylim(YLO, YHI)
    ax.axvline(GO, color=C.INK, lw=0.8, ls=(0, (1, 1.6)), zorder=1)
    ax.set_xlim(-0.03, 0.9)
    ax.tick_params(axis="y", labelleft=False)
    axu.set_xlim(-0.5, 0.5); axu.set_xticks([0]); axu.set_xticklabels(["n/a"])
    axu.set_ylabel("Test capture of R6 (budgets 0.05–0.30)")
    ax.set_xlabel("Validation lower bound of $C$ (R6)")
    axu.spines["left"].set_visible(True)
    ax.spines["left"].set_visible(False)
    axu.grid(False); axu.grid(True, axis="y")
    ax.grid(False); ax.grid(True, axis="y")
    axu.tick_params(axis="x", length=0)
    # quadrant labels with counts
    go = m.audit_C_lo > GO
    pays = m.capture > OUT
    cnt = {"gp": int((go & pays).sum()), "gn": int((go & ~pays).sum()), "np": int((~go & pays).sum()), "nn": int((~go & ~pays).sum())}
    kw = dict(fontsize=C.FS, color=C.INK, transform=ax.transAxes, linespacing=1.1,
              bbox=dict(boxstyle="square,pad=0.15", fc=C.SURF, ec="none", alpha=0.85))
    ax.text(0.985, 0.985, f"GO, pays\n$n$ = {cnt['gp']}", ha="right", va="top", **kw)
    ax.text(0.985, 0.015, f"GO, no gain\n$n$ = {cnt['gn']}", ha="right", va="bottom", **kw)
    ax.text(0.02, 0.985, f"NO-GO, pays\n(missed), $n$ = {cnt['np']}", ha="left", va="top", **kw)
    ax.text(0.02, 0.015, f"NO-GO, no gain\n$n$ = {cnt['nn']}", ha="left", va="bottom", **kw)
    handles = [Line2D([0], [0], color=C.DS_COLOR[d], marker=C.DS_MARKER[d], lw=0, ms=4, label=C.DS_LABEL[d]) for d in C.DATASETS]
    handles += [Line2D([0], [0], color=C.INK, lw=0.8, ls=(0, (1, 1.6)), label="GO line, 0.52"),
                Line2D([0], [0], color=C.INK2, lw=0.8, ls=(0, (4, 2)), label="outcome line, 0.10")]
    if (m.pair != "A").any():
        handles.append(Line2D([0], [0], color=C.INK2, lw=0, marker="o", ms=3.8, mfc=C.SURF, mew=0.8, label="pair B (hollow)"))
    nb = int((m.capture < YLO).sum())
    if nb:
        handles.append(Line2D([0], [0], color=C.INK2, lw=0, marker="v", ms=3.8, label=f"below axis ($n$={nb})"))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.54, 1.0), fontsize=C.FS,
               columnspacing=0.7, handlelength=1.2, handletextpad=0.3)
    print(f"  cells {len(m)}; undefined bound {int(undefined.sum())}; quadrants {cnt}")
    C.save(fig, "fig_audit")


if __name__ == "__main__":
    main()
