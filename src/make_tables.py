"""Regenerate the paper's tables from results/summary/*.csv (Section 10 of the plan).

    python src/make_tables.py [--results results] [--out results/summary/tables]

Writes LaTeX tabular bodies (booktabs) for: primary tests, H1 per dataset, the escalation
decomposition per dataset and pair, router C and capture, payload and footprint. No number is
typed by hand: every cell comes from the CSVs written by src/analyze.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def f(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "--" if not np.isfinite(x) else f"{x:.{d}f}"


def tex(df: pd.DataFrame, cols, header, d=3) -> str:
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(f(r[c], d) if c not in ("dataset", "pair", "router", "quantity", "hypothesis",
                                                           "verdict", "model", "family", "view", "unit")
                                else str(r[c]).replace("_", r"\_") for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def with_ci(df):
    df = df.copy()
    df["est"] = [f"{f(m)} [{f(a)}, {f(b)}]" for m, a, b in zip(df["mean"], df["ci_lo"], df["ci_hi"])]
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    s = Path(a.results) / "summary"
    out = Path(a.out) if a.out else s / "tables"
    out.mkdir(parents=True, exist_ok=True)
    prim = pd.read_csv(s / "primary.csv")
    (out / "primary.tex").write_text(tex(prim, ["hypothesis", "p_raw", "p_holm", "verdict"],
                                         ["Hypothesis", "$p$", "$p_{\\mathrm{Holm}}$", "Verdict"], 4))
    h1 = pd.read_csv(s / "h1_per_dataset.csv")
    if len(h1):
        (out / "h1.tex").write_text(tex(with_ci(h1), ["dataset", "J", "est", "p_one_sided"],
                                        ["Dataset", "$J$", "Shapley info. step [95\\% CI]", "$p_1$"], 4))
    pdt = pd.read_csv(s / "per_dataset.csv")
    keep = ["shapley_model", "shapley_info", "interaction", "comb_step_dep", "unresolved_dep", "comb_step_upg",
            "unresolved_upg", "self_oracle_headroom"]
    dec = with_ci(pdt[pdt.quantity.isin(keep) & pdt.router.isna()])
    if len(dec):
        wide = dec.pivot_table(index=["dataset", "pair"], columns="quantity", values="est", aggfunc="first").reset_index()
        cols = ["dataset", "pair"] + [c for c in keep if c in wide.columns]
        (out / "decomposition.tex").write_text(tex(wide.fillna("--"), cols, [c.replace("_", r"\_") for c in cols]))
    r = pdt[pdt.quantity.isin(["C", "capture"]) & pdt.router.notna()]
    if len(r):
        r = with_ci(r)
        wide = r.pivot_table(index=["dataset", "pair", "router"], columns="quantity", values="est",
                             aggfunc="first").reset_index()
        (out / "routers.tex").write_text(tex(wide, ["dataset", "pair", "router", "C", "capture"],
                                             ["Dataset", "Pair", "Router", "$C$", "Capture (0.05--0.30)"]))
    for nm in ("payload", "footprint"):
        p = s / f"{nm}.csv"
        if p.exists() and p.stat().st_size > 1:
            d = pd.read_csv(p)
            cols = [c for c in d.columns if c not in ("role", "input_shape", "note")]
            (out / f"{nm}.tex").write_text(tex(d, cols, [c.replace("_", r"\_") for c in cols], 0))
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
