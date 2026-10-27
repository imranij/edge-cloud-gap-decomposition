"""Write the LaTeX tables of the results section from the analysis CSVs.

    python paper/tables/src/make_results_tables.py

Every cell is read from results/summary/*.csv (final pairs A and B) or from
results/extra/*.csv (post-plan analyses).
No number is typed by hand. Output: paper/tables/*.tex (tabular environments only; captions and
labels live in sections/results.tex).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
S = ROOT / "results" / "summary"
SC = ROOT / "results" / "summary_pairC_hydraulic"
OUT = ROOT / "paper" / "tables"
DS = ["hydraulic", "skab", "seu", "threew"]
DSN = {"hydraulic": "Hydraulic", "skab": "SKAB", "seu": "SEU", "threew": "3W"}


def n(x, d=3):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = s[1:]
        return "$-$" + s if float(s) != 0 else s
    return s


def mci(row, d=3):
    return f"{n(row['mean'], d)} [{n(row['ci_lo'], d)}, {n(row['ci_hi'], d)}]"


def get(df, ds, pair, q, router=None):
    m = (df.dataset == ds) & (df.pair == pair) & (df.quantity == q)
    m &= df.router.isna() if router is None else (df.router == router)
    if "budget" in df:
        m &= df.budget.isna()
    r = df[m]
    return None if r.empty else r.iloc[0]


ROWS = [
    ("ba_ef_ev", r"Deployed edge $B_{eE}$"),
    ("ba_cf_ev", r"Upgraded edge $B_{cE}$"),
    ("ba_ef_cv", r"Edge family, all sensors $B_{eC}$"),
    ("ba_cf_cv", r"Cloud $B_{cC}$"),
    ("ba_oracle_dep", r"Per-window oracle"),
    ("shapley_model", r"Model step $\bar\Delta_{\mathrm{mod}}$"),
    ("shapley_info", r"Information step $\bar\Delta_{\mathrm{inf}}$"),
    ("interaction", r"Interaction $\Delta_{\times}$"),
    ("comb_step_dep", r"Combination step $\Delta_{\mathrm{comb}}$"),
    ("unresolved_dep", r"Unresolved disagreement $\Delta_{\mathrm{unr}}$"),
    ("unresolved_upg", r"Unresolved, upgraded edge $\Delta'_{\mathrm{unr}}$"),
    ("self_oracle_headroom", r"Self-oracle headroom $H_{\mathrm{self}}$"),
]


def overview():
    """Two files for the single-column layout: overview.tex holds Hydraulic and SKAB,
    overview_b.tex holds SEU and 3W (printed as the continuation of the same table)."""
    pdt = pd.read_csv(S / "per_dataset.csv")
    for fname, dss in (("overview.tex", DS[:2]), ("overview_b.tex", DS[2:])):
        ncol = 1 + len(dss)
        L = [r"\begin{tabular}{@{}l " + "l" * len(dss) + r"@{}}", r"\toprule",
             " & ".join(["Quantity"] + [DSN[d] for d in dss]) + r" \\"]
        for pair in ("A", "B", "C"):
            head = r", $J=10$ splits (12 on SEU)" if pair == "C" else r", $J=20$ splits per dataset"
            L += [r"\midrule", r"\multicolumn{" + str(ncol) + r"}{@{}l}{\textit{Pair " + pair + head + r"}} \\"]
            for q, lab in ROWS:
                if pair in ("B", "C") and q == "self_oracle_headroom":
                    continue
                cells = []
                for d in dss:
                    r = get(pdt, d, pair, q)
                    cells.append("--" if r is None else mci(r))
                L.append(" & ".join([lab] + cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (OUT / fname).write_text("\n".join(L) + "\n")


def routers():
    r = pd.read_csv(S / "routers.csv")
    names = {"R0": "R0 random", "R1": "R1 confidence", "R2": "R2 entropy", "R3": "R3 conformal",
             "R4": "R4 Diff-01", "R5": "R5 Diff-Prob", "R6": "R6 gain", "R7": "R7 R6+expl.",
             "R8": "R8 L2D", "R9": "R9 periodic"}
    L = [r"\begin{tabular}{@{}l " + "rr" * 4 + "@{}}", r"\toprule",
         " & ".join([""] + [r"\multicolumn{2}{c}{" + DSN[d] + "}" for d in DS]) + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         " & ".join(["Router"] + [r"$C$", r"$\Capt$"] * 4) + r" \\"]
    for pair in ("A", "B", "C"):
        L += [r"\midrule", r"\multicolumn{9}{@{}l}{\textit{Pair " + pair + (r", $J=10$ (12 on SEU)" if pair == "C" else "") + r"}} \\"]
        rows = [f"R{i}" for i in range(10)]
        for rt in rows:
            cells = []
            any_ = False
            for d in DS:
                for q in ("C", "capture"):
                    x = r[(r.dataset == d) & (r.pair == pair) & (r.router == rt) & (r.quantity == q)
                          & r.budget.isna()]
                    if x.empty:
                        cells.append("--")
                    else:
                        any_ = True
                        cells.append(n(x.iloc[0]["mean"]))
            if any_:
                L.append(" & ".join([names[rt]] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "routers.tex").write_text("\n".join(L) + "\n")


def tests():
    prim = pd.read_csv(S / "primary.csv").set_index("hypothesis")
    h1 = pd.read_csv(S / "h1_pooled.csv").iloc[0]
    h2 = pd.read_csv(S / "h2_model.csv").iloc[0]
    h3 = pd.read_csv(S / "h3_pooled.csv").iloc[0]
    h4 = pd.read_csv(S / "h4_model.csv").iloc[0]
    h5 = pd.read_csv(S / "h5.csv").iloc[0]
    est = {
        "H1": ("Pooled $\\bar\\Delta_{\\mathrm{inf}}$", h1["mean"], h1.ci_lo, h1.ci_hi),
        "H2": ("Slope of $C$ per doubling", h2.slope, h2.ci_lo, h2.ci_hi),
        "H3": ("Pooled $dC$", h3["mean"], h3.ci_lo, h3.ci_hi),
        "H4": ("Slope of $d\\mathrm{cap}$ on $\\rho_{\\mathrm{pos}}$", h4.slope, h4.ci_lo, h4.ci_hi),
        "H5": ("Balanced accuracy of calls", h5.ba, h5.ci_lo, h5.ci_hi),
    }
    L = [r"\begin{tabular}{@{}l l l l l l@{}}", r"\toprule",
         r"& Estimate & Value [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & Verdict \\", r"\midrule"]
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        lab, m, lo, hi = est[h]
        v = prim.loc[h, "verdict"].split(" (")[0]
        def pf(p):
            return f"{p:.4f}" if p >= 0.0001 else r"\num{" + f"{p:.1e}" + "}"
        L.append(f"{h} & {lab} & {n(m)} [{n(lo)}, {n(hi)}] & {pf(prim.loc[h, 'p_raw'])} & "
                 f"{pf(prim.loc[h, 'p_holm'])} & {v}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tests.tex").write_text("\n".join(L) + "\n")


def h1h3():
    a = pd.read_csv(S / "h1_per_dataset.csv").set_index("dataset")
    b = pd.read_csv(S / "h3_per_dataset.csv").set_index("dataset")
    fz = pd.read_csv(ROOT / "results" / "extra" / "h3_feasibility.csv").set_index("test")   #
    L = [r"\begin{tabular}{@{}l l l l l >{\raggedright\arraybackslash}p{1.9cm}@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{H1, $\bar\Delta_{\mathrm{inf}}$ (pair A)} & \multicolumn{3}{c}{H3, $dC=C(\mathrm{R7})-C(\mathrm{R6})$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-6}",
         r"Dataset & Mean [95\% CI] & $p_1$ & Mean [95\% CI] & SE at $J=20$ & Equivalence attainable \\", r"\midrule"]

    def att(r):
        m = r.max_abs_dC_for_equivalence
        return "no" if m <= 0 else r"only if $|dC|<" + f"{m:.3f}" + "$"
    for d in DS:
        r1 = a.loc[d]
        r3 = b.loc[d]
        f = fz.loc[d]
        p1 = r1.p_one_sided
        p1s = f"{p1:.3f}" if p1 >= 0.001 else r"\num{" + f"{p1:.1e}" + "}"
        L.append(f"{DSN[d]} & {n(r1['mean'])} [{n(r1.ci_lo)}, {n(r1.ci_hi)}] & {p1s} & "
                 f"{n(r3['mean'])} [{n(r3.ci_lo)}, {n(r3.ci_hi)}] & {n(f.se_at_J, 4)} & {att(f)}" + r" \\")
    h1p = pd.read_csv(S / "h1_pooled.csv").iloc[0]
    h3p = pd.read_csv(S / "h3_pooled.csv").iloc[0]
    f = fz.loc["pooled"]
    L.append(r"\midrule")
    L.append(f"Pooled & {n(h1p['mean'])} [{n(h1p.ci_lo)}, {n(h1p.ci_hi)}] & {h1p.p_one_sided:.3f} & "
             f"{n(h3p['mean'])} [{n(h3p.ci_lo)}, {n(h3p.ci_hi)}] & {n(f.se_at_J, 4)} & {att(f)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "h1h3.tex").write_text("\n".join(L) + "\n")


def by_well():
    pdt = pd.read_csv(S / "per_dataset.csv")
    r = pd.read_csv(S / "routers.csv")
    rows = [("ba_ef_ev", r"Edge $B_{eE}$"), ("ba_cf_ev", r"$B_{cE}$"),
            ("ba_ef_cv", r"$B_{eC}$"), ("ba_cf_cv", r"Cloud $B_{cC}$"),
            ("ba_oracle_dep", "Oracle"),
            ("shapley_model", r"$\bar\Delta_{\mathrm{mod}}$"), ("shapley_info", r"$\bar\Delta_{\mathrm{inf}}$"),
            ("unresolved_dep", r"$\Delta_{\mathrm{unr}}$")]
    L = [r"\begin{tabular}{@{}l l l@{}}", r"\toprule",
         r"Pair A & By episode & By well \\", r"\midrule"]
    for q, lab in rows:
        a = get(pdt, "threew", "A", q)
        b = get(pdt, "threew__by_well", "A", q)
        L.append(f"{lab} & {mci(a)} & {mci(b)}" + r" \\")
    for rt, q, lab in [("R1", "C", r"$C$, R1"), ("R6", "C", r"$C$, R6"),
                       ("R6", "capture", r"$\Capt$, R6")]:
        a = get(r, "threew", "A", q, rt)
        b = get(r, "threew__by_well", "A", q, rt)
        L.append(f"{lab} & {mci(a)} & {mci(b)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "bywell.tex").write_text("\n".join(L) + "\n")


def payload_footprint():
    """Uplink per hour of operation and edge cost per window (analytic, results/extra/)."""
    up = pd.read_csv(ROOT / "results" / "extra" / "uplink_per_hour.csv")
    ec = pd.read_csv(ROOT / "results" / "extra" / "edge_cost.csv")

    def big(v):
        if v >= 1e6:
            return f"{v / 1e6:.1f}\\,M"
        if v >= 1e4:
            return f"{v / 1e3:.0f}\\,k"
        if v >= 1e3:
            return f"{v / 1e3:.1f}\\,k"
        return f"{v:.0f}"
    L = [r"\begin{tabular}{@{}l rr rrrr rrr@{}}", r"\toprule",
         r"& & & \multicolumn{4}{c}{Uplink (B/h)} & \multicolumn{3}{c}{Edge, per window} \\",
         r"\cmidrule(lr){4-7}\cmidrule(l){8-10}",
         r"Dataset & Win.\ (s) & Dec./h & Esc.\ 10\% & Cloud, feat. & Stream, raw & Cloud-only, raw & Feat.\ ops & Model B & R6 ops \\",
         r"\midrule"]
    for d in DS:
        u10 = up[(up.dataset == d) & np.isclose(up.budget, 0.10)].iloc[0]
        e = ec[(ec.dataset == d) & (ec.pair == "A")].iloc[0]
        ws = u10.window_s
        wtxt = f"{ws:.1f}" if ws < 1 else f"{ws:.0f}"
        L.append(f"{DSN[d]} & {wtxt} & {big(u10.windows_per_hour)} & {big(u10.pull_features_B_per_h)} & "
                 f"{big(u10.full_cloud_features_B_per_h)} & {big(u10.raw_stream_all_B_per_h)} & "
                 f"{big(u10.raw_stream_cloud_only_B_per_h)} & {big(e.feature_extraction_ops)} & "
                 f"{int(e.edge_param_bytes_deploy)} & {int(e.router_R6_ops)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "payload.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    overview(); routers(); tests(); h1h3(); by_well(); payload_footprint()
    print("tables written to", OUT)
