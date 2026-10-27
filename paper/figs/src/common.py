"""Shared style, palette and data loaders for the results figures of the paper.

Every number drawn comes from results/ (summary CSVs, split_*.parquet, arrays_*.npz). Nothing is
retyped here. Pair B and pair C are included automatically when their files exist and carry at
least MIN_SPLITS splits (J in results/summary/sizes.csv); otherwise they are skipped.
"""
from __future__ import annotations
import functools
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                         # repository root
RES = Path(os.environ.get("PAPERA_RESULTS", ROOT / "results"))
SUM = RES / "summary"
OUT = Path(os.environ.get("PAPERA_FIGS", HERE.parent))   # paper/figs (override only for tests)
sys.path.insert(0, str(ROOT / "src"))          # for escalation.metrics.gain_curve (single source of truth)

W1, W2 = 3.5, 7.16                             # IEEE single / double column width, inches
W_MAIN = 390.0 / 72.27                         # elsarticle preprint \textwidth (390 pt = 5.40 in);
                                               # figures included at \textwidth are drawn at this width, so the
                                               # 8-pt fonts are 8 pt on the page (no LaTeX scaling)
FS = 8.0
NO_WHISKER = {"seu"}                      # SEU intervals rest on four healthy recordings, not drawn
MIN_SPLITS = 10                                # an optional pair needs this many splits to be drawn

DATASETS = ["hydraulic", "seu", "skab", "threew"]
DS_LABEL = {"hydraulic": "Hydraulic", "seu": "SEU", "skab": "SKAB", "threew": "3W"}
ROUTER_NAME = {"R0": "random", "R1": "confidence", "R2": "entropy", "R3": "conformal",
               "R4": "Diff-01", "R5": "Diff-Prob", "R6": "metric-weighted", "R7": "R6 + explanation",
               "R8": "learning to defer", "R9": "periodic"}

# ---- palette (dataviz skill reference palette, light mode) ------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GREY_LIGHT, GREY_MID, GREY_DARK = "#c3c2b7", "#898781", "#52514e"
ACCENT = "#2a78d6"                              # one accent: the learned / information-carrying quantity
# dataset identity: blue, aqua, yellow, red. This 4-set clears the validator on --pairs all (light);
# colour is always paired with a marker shape, so identity never rests on hue alone.
DS_COLOR = {"hydraulic": "#2a78d6", "seu": "#e34948", "skab": "#1baf7a", "threew": "#eda100"}
DS_MARKER = {"hydraulic": "o", "seu": "^", "skab": "s", "threew": "D"}
# pair styling: A is primary (filled / solid); B and C are extra series (hollow / dashed / dotted)
PAIR_LS = {"A": "-", "B": (0, (3, 1.6)), "C": (0, (1, 1.4))}
PAIR_NAME = {"A": "pair A", "B": "pair B", "C": "pair C"}


def setup_style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS,
        "ytick.labelsize": FS, "legend.fontsize": FS, "figure.titlesize": FS,
        "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.labelpad": 2,
        "lines.linewidth": 1.0, "lines.markersize": 4, "patch.linewidth": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
    })


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.pdf"
    fig.savefig(p)                              # no bbox_inches='tight': figure size is the final size
    plt.close(fig)
    print("wrote", p)
    return p


# ---- loaders ------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def summary(name: str) -> pd.DataFrame:
    return pd.read_csv(SUM / f"{name}.csv")


def sizes() -> pd.DataFrame:
    return summary("sizes")


def pair_J(dataset: str, pair: str) -> int:
    s = sizes()
    r = s[(s.dataset == dataset) & (s.pair == pair)]
    return int(r.J.iloc[0]) if len(r) else 0


def optional_pairs(dataset: str, min_splits: int = MIN_SPLITS) -> list[str]:
    """Pairs B/C that have a results directory and >= min_splits analysed splits for this dataset."""
    out = []
    for p in ("B", "C"):
        if (RES / dataset / p).is_dir() and pair_J(dataset, p) >= min_splits:
            out.append(p)
    return out


def pair_note() -> str:
    """Short human-readable status of the optional pairs, used in FIGURES.md and printed at build."""
    parts = []
    for p in ("B", "C"):
        have = [f"{d}({pair_J(d, p)})" for d in DATASETS if (RES / d / p).is_dir()]
        parts.append(f"pair {p}: " + (", ".join(have) if have else "absent"))
    return "; ".join(parts)


@functools.lru_cache(maxsize=None)
def long_results() -> pd.DataFrame:
    """All split_*.parquet files (long format), with the file's h2_q level and pair."""
    fs = sorted(RES.glob("*/*/split_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["router"] = d["router"].fillna("")
    return d


def split_values(dataset, pair, quantity, router="", h2_q=1.0, budget=None) -> pd.Series:
    d = long_results()
    m = (d.dataset == dataset) & (d.pair == pair) & (d.quantity == quantity) & (d.router == router) \
        & np.isclose(d.h2_q, h2_q)
    if budget is not None:
        m &= np.isclose(d.budget, budget)
    return d[m].set_index("split")["value"].sort_index()


def load_arrays(dataset, pair, split):
    return np.load(RES / dataset / pair / f"arrays_{split}.npz", allow_pickle=False)


def nb_half_width(sd, J, n_train, n_test, alpha=0.05):
    """Nadeau-Bengio corrected 95% half-width of a mean over J resplits (same rule as stats.nb_test)."""
    return stats.t.ppf(1 - alpha / 2, J - 1) * sd * np.sqrt(1.0 / J + n_test / n_train)


# ---- small drawing helpers ----------------------------------------------------------------------
def panel_tag(ax, text, x=0.0, y=1.0, **kw):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS, color=INK, **kw)


def clean_axes(ax, grid_axis="y"):
    ax.grid(False)
    if grid_axis in ("x", "both"):
        ax.grid(True, axis="x")
    if grid_axis in ("y", "both"):
        ax.grid(True, axis="y")
