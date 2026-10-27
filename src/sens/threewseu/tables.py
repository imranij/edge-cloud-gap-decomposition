"""Markdown tables for results_sens/threewseu/REPORT.md from summary/key_quantities.csv (POST-PLAN).

    python src/sens/threewseu/tables.py [variant ...]  > results_sens/threewseu/summary/tables.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[3] / "results_sens" / "threewseu" / "summary"


def f(m, lo, hi):
    if not np.isfinite(m):
        return "NA"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else f"{m:.3f}"


def label(r):
    s = r.quantity + (f" {r.router}" if isinstance(r.router, str) and r.router else "")
    return s + (f" @{r.budget:g}" if np.isfinite(r.budget) else "")


def main():
    k = pd.read_csv(S / "key_quantities.csv")
    k["router"] = k.router.fillna("")
    vs = sys.argv[1:] or list(dict.fromkeys(k.variant))
    for v in vs:
        for pair in ("A", "B"):
            d = k[(k.variant == v) & (k.pair == pair)]
            if d.empty:
                continue
            print(f"\n#### {v}, pair {pair} (J = {int(d.J.max())}; parent {d.parent.iloc[0]})\n")
            print("| quantity | variant (NB 95% CI) | main, all splits | main, same splits | variant - main, same splits |")
            print("|---|---|---|---|---|")
            for _, r in d.iterrows():
                mm = f(r.get("main_mean", np.nan), r.get("main_ci_lo", np.nan), r.get("main_ci_hi", np.nan)) \
                    + (f" (J={int(r.main_J)})" if np.isfinite(r.get("main_J", np.nan)) else "")
                ms = f(r.get("main_same_splits_mean", np.nan), r.get("main_same_ci_lo", np.nan), r.get("main_same_ci_hi", np.nan))
                dd = f(r.get("diff_mean", np.nan), r.get("diff_ci_lo", np.nan), r.get("diff_ci_hi", np.nan))
                print(f"| {label(r)} | {f(r['mean'], r.ci_lo, r.ci_hi)} | {mm} | {ms} | {dd} |")


if __name__ == "__main__":
    main()
