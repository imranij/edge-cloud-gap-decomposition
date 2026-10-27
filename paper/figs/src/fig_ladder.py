"""Fig. ladder: waterfall from the deployed edge BA to the per-window oracle, 2x2 by dataset (pair A).

Steps (all from results/summary/per_dataset.csv, corrected Nadeau-Bengio 95% CIs):
  model step        shapley_model     (Shapley-averaged over the two information views)
  information step  shapley_info      (Shapley-averaged over the two model families)
  combination step  comb_step_dep     (stacker minus cloud, deployed edge)
  unresolved        unresolved_dep    (per-window oracle minus stacker)
Levels: ba_ef_ev (deployed edge), ba_cf_cv (cloud), ba_oracle_dep (oracle). Reference: ba_oracle_self
(oracle over the cloud model and its retrained twin = self-oracle headroom).
shapley_model + shapley_info = ba_cf_cv - ba_ef_ev exactly, so the ladder closes at the cloud level.
"""
from __future__ import annotations
import numpy as np
from matplotlib.lines import Line2D
import common as C

PAIR = "A"
COLS = [  # (x label, kind, quantity)
    ("Edge", "level", "ba_ef_ev"),
    ("Model", "step", "shapley_model"),
    ("Information", "step", "shapley_info"),
    ("Cloud", "level", "ba_cf_cv"),
    ("Combination", "step", "comb_step_dep"),
    ("Unresolved", "step", "unresolved_dep"),
    ("Oracle", "level", "ba_oracle_dep"),
]


def get(pdt, ds, q):
    r = pdt[(pdt.dataset == ds) & (pdt.pair == PAIR) & (pdt.quantity == q) & (pdt.router.isna())]
    assert len(r) == 1, (ds, q, len(r))
    return r.iloc[0]


def ladder_data(ds):
    pdt = C.summary("per_dataset")
    rows, cum = [], None
    for lab, kind, q in COLS:
        r = get(pdt, ds, q)
        if kind == "level":
            if cum is not None and abs(cum - r["mean"]) > 1e-6:
                print(f"  WARNING {ds}: ladder does not close at {q}: cumulative {cum:.6f} vs {r['mean']:.6f}")
            cum = r["mean"]
            rows.append(dict(lab=lab, kind=kind, lo=r["mean"], hi=r["mean"], mean=r["mean"],
                             ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], end=r["mean"], delta=np.nan))
        else:
            start = cum; cum = cum + r["mean"]
            rows.append(dict(lab=lab, kind=kind, lo=min(start, cum), hi=max(start, cum), mean=r["mean"],
                             ci_lo=start + r["ci_lo"], ci_hi=start + r["ci_hi"], end=cum, delta=r["mean"]))
    self_or = get(pdt, ds, "ba_oracle_self")["mean"]
    return rows, self_or


def fmt(v):
    return f"{v:+.3f}".replace("-", "\u2212")


def main():
    C.setup_style()
    # drawn at the elsarticle text width so that the 8-pt fonts print at 8 pt
    fig, axes = C.plt.subplots(2, 2, figsize=(C.W_MAIN, 5.3),
                               gridspec_kw=dict(left=0.095, right=0.995, top=0.885, bottom=0.11,
                                                wspace=0.20, hspace=0.45))
    tags = "abcd"
    x = np.arange(len(COLS))
    bw = 0.62
    CEIL = 1.0
    for i, (ax, ds) in enumerate(zip(axes.flat, C.DATASETS)):
        rows, self_or = ladder_data(ds)
        vals = [r["lo"] for r in rows] + ([] if ds in C.NO_WHISKER else [r["ci_lo"] for r in rows])
        lo_v = min(vals)
        ylo = np.floor((lo_v - 0.005) * 20) / 20 if lo_v < 0.9 else np.floor((lo_v - 0.003) * 100) / 100
        span = CEIL - ylo
        ax.set_ylim(ylo, CEIL + 0.035 * span)
        ax.set_xlim(-0.6, len(COLS) - 0.4)
        cloud = [r for r in rows if r["lab"] == "Cloud"][0]["mean"]
        ax.axhline(CEIL, color=C.AXIS, lw=0.6, zorder=1)                       # BA ceiling
        ax.axhline(cloud, color=C.GREY_MID, lw=0.8, ls=(0, (4, 2)), zorder=1)  # cloud level
        ax.plot([3.5, len(COLS) - 0.4], [self_or, self_or], color=C.INK, lw=0.9, ls=(0, (1, 1.6)), zorder=1)
        for j, r in enumerate(rows):
            if r["kind"] == "level":
                ax.plot([j - bw / 2, j + bw / 2], [r["mean"]] * 2, color=C.INK, lw=2.4, solid_capstyle="butt", zorder=3)
            else:
                col = C.ACCENT if r["lab"].startswith("Infor") else C.GREY_MID
                ax.bar(j, max(r["hi"] - r["lo"], 1e-9), bottom=r["lo"], width=bw, color=col, zorder=3, linewidth=0)
            if j > 0:
                prev = rows[j - 1]["end"]
                start = r["mean"] if r["kind"] == "level" else (r["end"] - r["delta"])
                ax.plot([j - 1 + bw / 2, j - bw / 2], [prev, start], color=C.GREY_MID, lw=0.5, zorder=2)
            if ds in C.NO_WHISKER:                                    # SEU: indicative only, no whiskers
                continue
            lo, hi = max(r["ci_lo"], ylo), min(r["ci_hi"], CEIL)      # corrected 95% CI, cut at BA = 1
            wcol = C.INK
            ax.plot([j, j], [lo, hi], color=wcol, lw=0.8, zorder=4, solid_capstyle="butt")
            for yy in (lo, hi):
                ax.plot([j - 0.1, j + 0.1], [yy, yy], color=wcol, lw=0.8, zorder=4, solid_capstyle="butt")
        # full step names, rotated, so they do not collide at the 5.4-in width; the step means are
        # in Table 2 (pair A, same source) and are no longer printed under the labels
        labs = [c[0] for c in COLS]
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=C.FS, rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0)
        ax.grid(False); ax.grid(True, axis="y")
        ax.yaxis.set_major_locator(C.matplotlib.ticker.MaxNLocator(5))
        C.panel_tag(ax, f"({tags[i]}) {C.DS_LABEL[ds]}" + (" (means only)" if ds in C.NO_WHISKER else ""), x=0.0, y=1.02)
    fig.supylabel("Balanced accuracy", fontsize=C.FS, x=0.012)
    handles = [
        C.matplotlib.patches.Patch(color=C.ACCENT, label="information step"),
        C.matplotlib.patches.Patch(color=C.GREY_MID, label="other steps"),
        Line2D([0], [0], color=C.INK, lw=2.4, label="level (edge, cloud, oracle)"),
        Line2D([0], [0], color=C.INK, lw=0.8, marker="_", markersize=5, label="corrected 95% CI (cut at 1)"),
        Line2D([0], [0], color=C.GREY_MID, lw=0.8, ls=(0, (4, 2)), label="cloud level"),
        Line2D([0], [0], color=C.INK, lw=0.9, ls=(0, (1, 1.6)), label="self-oracle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.54, 1.0), fontsize=C.FS,
               columnspacing=0.9, handlelength=1.5)
    C.save(fig, "fig_ladder")


if __name__ == "__main__":
    main()
