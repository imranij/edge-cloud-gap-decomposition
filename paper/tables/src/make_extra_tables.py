"""Write the LaTeX tables of the post-plan analyses. NOT pre-specified.

    python paper/tables/src/make_extra_tables.py

Every cell is read from results/extra/*.csv (written by src/extra/*.py) or results/summary/*.csv.
No number is typed by hand. Output: paper/tables/rev_*.tex (tabular environments only).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
E = ROOT / "results" / "extra"
S = ROOT / "results" / "summary"
OUT = ROOT / "paper" / "tables"
DS = ["hydraulic", "skab", "seu", "threew"]
DSN = {"hydraulic": "Hydraulic", "skab": "SKAB", "seu": "SEU", "threew": "3W", "threew__by_well": "3W by well"}


def n(x, d=3):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = s[1:]
        return "$-$" + s if float(s) != 0 else s
    return s


def ci(m, lo, hi, d=3):
    return f"{n(m, d)} [{n(lo, d)}, {n(hi, d)}]"


def pv(p):
    if not np.isfinite(p):
        return "--"
    return f"{p:.3f}" if p >= 0.001 else r"\num{" + f"{p:.1e}" + "}"


def write(name, L):
    (OUT / f"{name}.tex").write_text("\n".join(L) + "\n")


def h4():
    nul = pd.read_csv(E / "h4_null_slopes.csv").set_index("lhs")
    pdx = pd.read_csv(E / "h4_per_dataset.csv")
    var = pd.read_csv(E / "h4_rho_variants.csv")
    val = pd.read_csv(E / "h4_validation_rho.csv")
    rows = [("dcap_R6", r"$\Capt(\mathrm{R6})-\Capt(\mathrm{R1})$, plan's test"),
            ("dcap_R0", r"$\Capt(\mathrm{R0})-\Capt(\mathrm{R1})$, random router"),
            ("dcap_R9", r"$\Capt(\mathrm{R9})-\Capt(\mathrm{R1})$, label-blind schedule"),
            ("cap_R1", r"$\Capt(\mathrm{R1})$"), ("cap_R6", r"$\Capt(\mathrm{R6})$"),
            ("cap6_minus_cap0", r"$\Capt(\mathrm{R6})-\Capt(\mathrm{R0})$")]
    L = [r"\begin{tabular}{@{}l l rrrr@{}}", r"\toprule",
         r"Left-hand side & Pooled slope [95\% CI] & Hyd. & SKAB & SEU & 3W \\", r"\midrule",
         r"\multicolumn{6}{@{}l}{\textit{Regressor $\rho_{\mathrm{pos}}$ on the test fold, as in the plan}} \\"]
    for k, lab in rows:
        r = nul.loc[k]
        per = [pdx[(pdx.dataset == d) & (pdx.lhs == k)].slope.iloc[0] for d in DS]
        L.append(f"{lab} & {ci(r.slope, r.ci_lo, r.ci_hi)} & " + " & ".join(n(x, 2) for x in per) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{@{}l}{\textit{Plan's left-hand side, other versions of the regressor}} \\")
    labs = {"rho_all": r"$\rho$ on all test windows", "rho_neg": r"$\rho$ within test negatives"}
    for rho in ("rho_all", "rho_neg"):
        r = var[(var.rho == rho) & (var.lhs == "dcap_R6")].iloc[0]
        L.append(f"{labs[rho]} & {ci(r.slope, r.ci_lo, r.ci_hi)} & \\multicolumn{{4}}{{c}}{{}}" + r" \\")
    r = val[val.lhs == "dcap_R6"].iloc[0]
    L.append(r"$\rho_{\mathrm{pos}}$ on the validation fold & " + ci(r.slope, r.ci_lo, r.ci_hi) + r" & \multicolumn{4}{c}{} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_h4", L)


def audit():
    br = pd.read_csv(E / "h5_breakdown.csv").set_index("subset")
    bp = pd.read_csv(E / "h5_block_perm.csv").set_index("subset")
    L = [r"\begin{tabular}{@{}l rrrrrl@{}}", r"\toprule",
         r"Cells & $n$ & GO & Prec. & Base & BA & $p$ \\", r"\midrule"]

    def row(key, lab, p=None):
        r = br.loc[key]
        pp = r.get("p_perm_within_dataset", np.nan) if p is None else p
        L.append(f"{lab} & {int(r.n)} & {int(r.n_go)} & {n(r.go_precision, 2)} & {n(r.base_rate, 2)} & {n(r.ba)} & {pv(pp)}" + r" \\")
    row("all cells (pre-specified)", "All, within-dataset perm.")
    L.append(f"All, split blocks kept & & & & & & {pv(bp.loc['all cells', 'p_block_perm'])}" + r" \\")
    row("q = 1 only", r"$q=1$ only, within-dataset perm.")
    L.append(f"$q=1$ only, split blocks kept & & & & & & {pv(bp.loc['q = 1 only', 'p_block_perm'])}" + r" \\")
    L.append(r"\midrule")
    for d in DS:
        row(f"dataset={d}", DSN[d], p=np.nan)
    L.append(r"\midrule")
    for q in ("0.1", "0.25", "0.5", "1.0"):
        row(f"level q={q}", f"$q={q}$")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_audit", L)


def oracle():
    o = pd.read_csv(E / "oracle_reference.csv")

    def g(ds, pair, q):
        r = o[(o.dataset == ds) & (o.pair == pair) & (o.quantity == q)].iloc[0]
        return r
    L = [r"\begin{tabular}{@{}l l l l l@{}}", r"\toprule",
         r"Dataset, pair & Oracle & Independent errors & Oracle $-$ independent & Stacker $-$ selected tier \\", r"\midrule"]
    for ds in DS + ["threew__by_well"]:
        for pair in (("A", "B") if ds != "threew__by_well" else ("A",)):
            a, b = g(ds, pair, "oracle"), g(ds, pair, "oracle_indep")
            c, d = g(ds, pair, "excess_over_indep"), g(ds, pair, "stacker_minus_val_selected_tier")
            L.append(f"{DSN[ds]}, {pair} & {n(a['mean'])} & {n(b['mean'])} & {ci(c['mean'], c.ci_lo, c.ci_hi)} & "
                     f"{ci(d['mean'], d.ci_lo, d.ci_hi)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_oracle", L)


def arch():
    a = pd.read_csv(E / "architectures.csv")
    L = [r"\begin{tabular}{@{}l l rrrrrr@{}}", r"\toprule",
         r"& & Edge & \multicolumn{2}{c}{Escalation 10\%} & Esc.\ 30\% & Gateway & Cloud \\",
         r"\cmidrule(lr){4-5}",
         r"Dataset & Pair & only & R1 & R6 & R6 & $B_{eC}$ & $B_{cC}$ \\", r"\midrule"]
    for d in DS:
        for p in ("A", "B"):
            g = a[(a.dataset == d) & (a.pair == p)]

            def v(arch, f=None, router=""):
                m = g[(g.architecture == arch) & (g.router.fillna("") == router)]
                if f is not None:
                    m = m[np.isclose(m.budget, f)]
                return n(m["mean"].iloc[0])
            L.append(f"{DSN[d] if p == 'A' else ''} & {p} & {v('edge only')} & {v('edge + escalation', 0.1, 'R1')} & "
                     f"{v('edge + escalation', 0.1, 'R6')} & {v('edge + escalation', 0.3, 'R6')} & "
                     f"{v('gateway, small model on all sensors')} & {v('full cloud')}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_arch", L)


def lodo():
    d = pd.read_csv(E / "lodo.csv")
    tests = [("H1 pooled Shapley information step", "H1 pooled information step"),
             ("H2 slope of C(R6) per doubling", "H2 slope per doubling"),
             ("H4 slope of dcap on rho_pos", r"H4 slope on $\rho_{\mathrm{pos}}$"),
             ("H5 BA of audit calls", "H5 BA of calls")]
    ex = ["none (all four)", "hydraulic", "skab", "seu", "threew"]
    exn = {"none (all four)": "None", "hydraulic": "Hydraulic", "skab": "SKAB", "seu": "SEU", "threew": "3W"}
    L = [r"\begin{tabular}{@{}l " + "l" * len(tests) + r"@{}}", r"\toprule",
         "Excluded & " + " & ".join(t[1] for t in tests) + r" \\", r"\midrule"]
    for e in ex:
        cells = []
        for t, _ in tests:
            r = d[(d.excluded == e) & (d.test == t)].iloc[0]
            if t.startswith("H5"):
                cells.append(f"{n(r.estimate)} ($p={pv(r.p_one_sided)}$)")
            else:
                cells.append(f"{ci(r.estimate, r.ci_lo, r.ci_hi)}")
        L.append(f"{exn[e]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_lodo", L)


def h2():
    d = pd.read_csv(E / "h2_extra.csv")
    models = [("C(R6) ~ log2 positive windows [plan's H2 test]", r"$C(\mathrm{R6})$ on $\log_2$ windows (plan)"),
              ("C(R6) - C(R1) ~ log2 positive windows", r"$C(\mathrm{R6})-C(\mathrm{R1})$ on $\log_2$ windows"),
              ("C(R6) ~ log2 kept positive units", r"$C(\mathrm{R6})$ on $\log_2$ units")]
    L = [r"\begin{tabular}{@{}l l l rrrr@{}}", r"\toprule",
         r"& Mixed model & Two-stage & \multicolumn{4}{c}{Two-stage, per dataset} \\", r"\cmidrule(l){4-7}",
         r"Model & [95\% CI] & [95\% CI] & Hyd. & SKAB & SEU & 3W \\", r"\midrule"]
    for m, lab in models:
        g = d[d.model == m]
        a = g[g.estimator == "MixedLM pooled"].iloc[0]
        b = g[g.estimator == "two-stage pooled (HKSJ)"].iloc[0]
        per = [g[g.estimator == f"two-stage {x} (NB over splits)"].slope.iloc[0] for x in DS]
        L.append(f"{lab} & {ci(a.slope, a.ci_lo, a.ci_hi)} & {ci(b.slope, b.ci_lo, b.ci_hi)} & " +
                 " & ".join(n(x) for x in per) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_h2", L)


def tiers():
    t = pd.read_csv(E / "h2_tiers_by_q.csv")
    L = [r"\begin{tabular}{@{}l l rrrrrr@{}}", r"\toprule",
         r"Dataset & $q$ & Units & Edge BA & Cloud BA & Oracle gap & $C$(R1) & $C$(R6) \\", r"\midrule"]
    for d in DS:
        for _, r in t[t.dataset == d].iterrows():
            L.append(f"{DSN[d] if r.h2_q == 0.1 else ''} & {r.h2_q:g} & {r.kept_units:.1f} & {n(r.edge_ba)} & "
                     f"{n(r.cloud_ba)} & {n(r.oracle_gap)} & {n(r.C_R1)} & {n(r.C_R6)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_h2tiers", L)


def faults():
    f = pd.read_csv(E / "info_by_fault_type.csv")
    hv = pd.read_csv(E / "info_step_halves.csv")
    reach = {"edge": "edge", "mainly loop": "loop", "loop": "loop", "partly edge": "partly", "both": "both",
             "mainly topside": "topside"}

    def short(s):
        for k, v in reach.items():
            if s.startswith(k):
                return v
        return "--"
    names = {"valve1_inlet_closure": "Valve 1 (inlet) closure", "valve2_outlet_closure": "Valve 2 (outlet) closure",
             "rotor_imbalance": "Rotor imbalance", "cavitation": "Cavitation", "fluid_leak_or_addition": "Fluid leak or addition",
             "water_increase": "Water increase", "warm_water_supply": "Warm water supply",
             "pump_leak_weak": "Weak pump leakage", "pump_leak_severe": "Severe pump leakage",
             "flow_instability": "Flow instability", "scaling_in_pck": "Scaling in production choke",
             "severe_slugging": "Severe slugging", "rapid_productivity_loss": "Rapid productivity loss",
             "hydrate_in_service_line": "Hydrate in service line", "hydrate_in_production_line": "Hydrate in production line",
             "abrupt_increase_of_bsw": "Abrupt increase of BSW", "spurious_closure_of_dhsv": "Spurious DHSV closure"}
    L = [r"\begin{tabular}{@{}l l r l l@{}}", r"\toprule",
         r"Fault type & Reaches & $J_t$ & Pair A $\bar\Delta_{\mathrm{inf},t}$ & Pair B $\bar\Delta_{\mathrm{inf},t}$ \\", r"\midrule"]
    for d in ("hydraulic", "skab", "threew"):
        L.append(r"\multicolumn{5}{@{}l}{\textit{" + DSN[d] + r"}} \\")
        g = f[(f.dataset == d) & (f.pair == "A")].sort_values("mean_n_pos", ascending=False)
        for _, r in g.iterrows():
            b = f[(f.dataset == d) & (f.pair == "B") & (f.fault_type == r.fault_type)].iloc[0]
            L.append(f"{names.get(r.fault_type, r.fault_type)} & {short(r.edge_reach)} & {int(r.J_present)} & "
                     f"{ci(r.info_step, r.info_step_lo, r.info_step_hi)} & {ci(b.info_step, b.info_step_lo, b.info_step_hi)}" + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{5}{@{}l}{\textit{Whole information step split into its two halves}} \\")
    for d in DS:
        for half, lab in (("info_spec_half", "negatives (specificity)"), ("info_recall_half", "positives (recall)")):
            a = hv[(hv.dataset == d) & (hv.pair == "A") & (hv.quantity == half)].iloc[0]
            b = hv[(hv.dataset == d) & (hv.pair == "B") & (hv.quantity == half)].iloc[0]
            L.append(f"{DSN[d]}, {lab} & & & {ci(a['mean'], a.ci_lo, a.ci_hi)} & {ci(b['mean'], b.ci_lo, b.ci_hi)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_faults", L)


def seu():
    r = pd.read_csv(E / "seu_by_healthy_recording.csv")
    sp = pd.read_csv(E / "seu_specificity_by_recording.csv")
    r = r[(r.pair == "A") & ~r.healthy.str.startswith("SUMMARY")]
    L = [r"\begin{tabular}{@{}l r rrrr rr@{}}", r"\toprule",
         r"Healthy recording & $J_h$ & Sp.\ $e$ & Sp.\ $c$ & $B_{eE}$ & $B_{cC}$ & $\bar\Delta_{\mathrm{mod}}$ & $C$(R6) \\",
         r"\midrule"]
    for _, x in r.iterrows():
        se = sp[(sp.pair == "A") & (sp.model == "ef_ev") & (sp.healthy == x.healthy)].mean_spec.iloc[0]
        sc = sp[(sp.pair == "A") & (sp.model == "cf_cv") & (sp.healthy == x.healthy)].mean_spec.iloc[0]
        sub, rest = x.healthy.split("/")
        name = sub.capitalize() + ", " + rest.split("_", 1)[1].replace("_", "-")
        L.append(f"{name} & {int(x.n_splits)} & {n(se, 2)} & {n(sc, 2)} & {n(x.ba_ef_ev)} & {n(x.ba_cf_cv)} & "
                 f"{n(x.shapley_model)} & {n(x.C_R6)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_seu", L)


if __name__ == "__main__":
    for f in (h4, audit, oracle, arch, lodo, h2, tiers, faults, seu):
        f()
    print("post-plan tables written to", OUT)
