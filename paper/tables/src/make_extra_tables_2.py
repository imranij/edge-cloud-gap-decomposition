"""Write the LaTeX tables of the post-plan analyses, second batch. NOT pre-specified.

    python paper/tables/src/make_extra_tables_2.py

Every cell is read from results/extra/*.csv (written by src/extra/*.py) or results/summary/*.csv.
No number is typed by hand. Output: paper/tables/*.tex (tabular environments only). Run after
make_extra_tables.py; the tables rev_faults, rev_oracle, rev_arch, rev_audit, rev_lodo and
rev_h4 written here replace the earlier versions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from make_extra_tables import E, S, OUT, DS, DSN, n, ci, pv, write

NAMES = {"valve1_inlet_closure": "Valve 1 (inlet) closure", "valve2_outlet_closure": "Valve 2 (outlet) closure",
         "rotor_imbalance": "Rotor imbalance", "cavitation": "Cavitation", "fluid_leak_or_addition": "Fluid leak or addition",
         "water_increase": "Water increase", "warm_water_supply": "Warm water supply",
         "pump_leak_weak": "Weak pump leakage", "pump_leak_severe": "Severe pump leakage",
         "flow_instability": "Flow instability", "scaling_in_pck": "Scaling in production choke",
         "severe_slugging": "Severe slugging", "rapid_productivity_loss": "Rapid productivity loss",
         "hydrate_in_service_line": "Hydrate in service line", "hydrate_in_production_line": "Hydrate in production line",
         "abrupt_increase_of_bsw": "Abrupt increase of BSW", "spurious_closure_of_dhsv": "Spurious DHSV closure",
         "normal": "Normal (no event)"}


def kb(x):
    """Bytes (or B/h) in compact form."""
    if not np.isfinite(x):
        return "--"
    if x >= 1e6:
        return f"{x / 1e6:.1f}\\,M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}\\,k" if x < 1e4 else f"{x / 1e3:.0f}\\,k"
    return f"{x:.0f}"


def skab():
    """Two stacked panels (single-column layout): tier means, then the information step by pair."""
    s = pd.read_csv(E / "skab_breakdown.csv")

    def g(pair, q):
        r = s[(s.pair == pair) & (s.quantity == q)].iloc[0]
        return r
    rows = [("Specificity, anomaly-free recording", "spec_free", "info_free"),
            ("Specificity, normals before the anomaly", "spec_pre", "info_pre"),
            ("Specificity, normals after the anomaly", "spec_post", "info_post"),
            ("Recall, all fault windows", "recall", "info_fault"),
            ("BA within experiment files", "ba_file", "file_info"),
            ("AUC within experiment files", "auc_file", None),
            ("BA at matched elapsed time", "ba_timestrat", "timestrat_info")]
    L = [r"\begin{tabular}{@{}l rrrr@{}}", r"\toprule",
         r"& \multicolumn{4}{c}{Pair A, mean over splits} \\",
         r"\cmidrule(l){2-5}",
         r"Subset and metric & $B_{eE}$ & $B_{cE}$ & $B_{eC}$ & $B_{cC}$ \\", r"\midrule"]
    for lab, pre, inf in rows:
        vals = " & ".join(n(g("A", f"{pre}_{m}")["mean"]) for m in ("eE", "cE", "eC", "cC"))
        L.append(f"{lab} & {vals}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
          r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.3cm} lll@{}}", r"\toprule",
          r"& \multicolumn{3}{c}{Information step on the subset [95\% CI]} \\",
          r"\cmidrule(l){2-4}",
          r"Subset and metric & Pair A & Pair B & Pair C \\", r"\midrule"]
    for lab, pre, inf in rows:
        if inf is None:
            steps = r"\multicolumn{3}{c}{--}"
        else:
            steps = " & ".join(ci(g(p, inf)["mean"], g(p, inf).ci_lo, g(p, inf).ci_hi) for p in ("A", "B", "C"))
        L.append(f"{lab} & {steps}" + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{@{}l}{\textit{Specificity half of the information step, by subset}} \\")
    for lab, q in (("Anomaly-free recording", "contrib_free"), ("Normals before the anomaly", "contrib_pre"),
                   ("Normals after the anomaly", "contrib_post"), ("Total (specificity half)", "info_spec_half")):
        L.append(f"{lab} & " + " & ".join(ci(g(p, q)["mean"], g(p, q).ci_lo, g(p, q).ci_hi) for p in ("A", "B", "C")) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{@{}l}{\textit{Decomposition restricted to experiment files}} \\")
    for lab, q in (("Model step", "file_model"), ("Information step", "file_info"), ("Interaction", "file_inter")):
        L.append(f"{lab} & " + " & ".join(ci(g(p, q)["mean"], g(p, q).ci_lo, g(p, q).ci_hi) for p in ("A", "B", "C")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_skab", L)


def faults():
    f = pd.read_csv(E / "info_by_fault_type.csv")
    hv = pd.read_csv(E / "info_step_halves.csv")
    L = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.8cm} r l l rrrr@{}}", r"\toprule",
         r"& & \multicolumn{2}{c}{Recall part of the information step} & \multicolumn{4}{c}{Recall on the type, pair A} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-8}",
         r"Fault type & $J_t$ & Pair A [95\% CI] & Pair B [95\% CI] & $e,E$ & $c,E$ & $e,C$ & $c,C$ \\", r"\midrule"]
    for d in ("hydraulic", "skab", "threew"):
        a = hv[(hv.dataset == d) & (hv.pair == "A") & (hv.quantity == "info_spec_half")].iloc[0]
        b = hv[(hv.dataset == d) & (hv.pair == "B") & (hv.quantity == "info_spec_half")].iloc[0]
        L.append(r"\multicolumn{8}{@{}l}{\textit{" + DSN[d] + r"}} \\")
        L.append(f"Specificity half, shared by all types & & {ci(a['mean'], a.ci_lo, a.ci_hi)} & {ci(b['mean'], b.ci_lo, b.ci_hi)} & & & & " + r" \\")
        g = f[(f.dataset == d) & (f.pair == "A")].sort_values("mean_n_pos", ascending=False)
        for _, r in g.iterrows():
            bb = f[(f.dataset == d) & (f.pair == "B") & (f.fault_type == r.fault_type)].iloc[0]
            L.append(f"{NAMES.get(r.fault_type, r.fault_type)} & {int(r.J_present)} & "
                     f"{ci(r.info_recall_part, r.info_recall_part_lo, r.info_recall_part_hi)} & "
                     f"{ci(bb.info_recall_part, bb.info_recall_part_lo, bb.info_recall_part_hi)} & "
                     + " & ".join(n(r[f"recall_{k}"], 2) for k in ("ef_ev", "cf_ev", "ef_cv", "cf_cv")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_faults", L)


def oracle():
    """Two stacked panels (single-column layout): oracle headroom, then the stacker."""
    o = pd.read_csv(E / "oracle_reference.csv")

    def g(ds, pair, q):
        return o[(o.dataset == ds) & (o.pair == pair) & (o.quantity == q)].iloc[0]
    panels = [(("headroom_over_best_tier", "indep_headroom_over_best_tier", "excess_over_indep"),
               r"& Oracle over the better & Same, independent & Oracle $-$ \\",
               r"Dataset, pair & tier in each split & errors & independent \\"),
              (("stacker_minus_val_selected_tier", "stacker_minus_best_test_tier"),
               r"& Stacker $-$ tier & Stacker $-$ better \\",
               r"Dataset, pair & selected on val. & test tier \\")]
    L = []
    for k, (qs, h1, h2) in enumerate(panels):
        if k:
            L.append(r"\par\medskip")
        L += [r"\begin{tabular}{@{}l " + "l" * len(qs) + r"@{}}", r"\toprule", h1, h2, r"\midrule"]
        for ds in DS + ["threew__by_well"]:
            for pair in (("A", "B") if ds != "threew__by_well" else ("A",)):
                cells = [g(ds, pair, q) for q in qs]
                L.append(f"{DSN[ds]}, {pair} & " + " & ".join(ci(c['mean'], c.ci_lo, c.ci_hi) for c in cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_oracle", L)


def arch():
    a = pd.read_csv(E / "architectures.csv")
    p = pd.read_csv(E / "arch_paired.csv")
    L = [r"\begin{tabular}{@{}l rrrrrr ll@{}}", r"\toprule",
         r"& Edge & \multicolumn{3}{c}{Escalation 10\%} & Gate- & & \multicolumn{2}{c}{Gateway minus [95\% CI]} \\",
         r"\cmidrule(lr){3-5}\cmidrule(l){8-9}",
         r"Pair & only & R1 & R1 real. & R6 & way & Cloud & escalation R1 10\% & cloud \\", r"\midrule"]
    mark = {"seu": r"$^{\dagger}$", "threew": r"$^{\ddagger}$"}
    for d in DS:
        for pr in ("A", "B"):
            g = a[(a.dataset == d) & (a.pair == pr)]

            def v(arch, f=None, router=""):
                m = g[(g.architecture == arch) & (g.router.fillna("") == router)]
                if f is not None:
                    m = m[np.isclose(m.budget, f)]
                return n(m["mean"].iloc[0])

            def pp(comp):
                r = p[(p.dataset == d) & (p.pair == pr) & (p.comparison == comp)].iloc[0]
                return r
            real = pp("escalation R1 10% realised (level)")
            c1, c2 = pp("gateway - escalation R1 10%"), pp("gateway - cloud")
            if pr == "A":
                L.append(r"\multicolumn{9}{@{}l}{\textit{" + DSN[d] + r"}" + mark.get(d, "") + r"} \\")
            L.append(f"{pr} & {v('edge only')} & {v('edge + escalation', 0.1, 'R1')} & {n(real['mean'])} & "
                     f"{v('edge + escalation', 0.1, 'R6')} & {v('gateway, small model on all sensors')} & {v('full cloud')} & "
                     f"{ci(c1['mean'], c1.ci_lo, c1.ci_hi)} & {ci(c2['mean'], c2.ci_lo, c2.ci_hi)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_arch", L)


def syscost():
    s = pd.read_csv(E / "system_cost.csv")
    rb = pd.read_csv(E / "realised_budget.csv")
    arch = [("edge only", "Edge only"), ("escalation 10%, pull, on-demand extraction", "Esc.\\ 10\\%, on-demand"),
            ("escalation 10%, pull, continuous extraction", "Esc.\\ 10\\%, continuous"),
            ("gateway", "Gateway"), ("full cloud", "Full cloud")]
    L = [r"\begin{tabular}{@{}l rrrr rr@{}}", r"\toprule",
         r"& \multicolumn{4}{c}{Per window} & \multicolumn{2}{c}{Per hour} \\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
         r"Architecture & Ops, edge & Ops, other & Ops, total & Memory (B) & Uplink (B) & Messages \\",
         r"\midrule"]
    for d in DS:
        g = s[s.dataset == d]
        r1 = rb[(rb.dataset == d) & (rb.pair == "A") & (rb.router == "R1") & np.isclose(rb.budget, 0.10)].iloc[0]
        L.append(r"\multicolumn{7}{@{}l}{\textit{" + DSN[d] + r"}} \\")
        for i, (k, lab) in enumerate(arch):
            r = g[g.architecture == k].iloc[0]
            up = kb(r.uplink_B_per_h)
            if k.startswith("escalation"):
                up += f" ({kb(r1.uplink_min_B_per_h)}--{kb(r1.uplink_max_B_per_h)})"
            if k == "gateway":
                up = f"0 (local {kb(r.local_B_per_h)})"
            L.append(f"{lab} & {kb(r.ops_edge_node)} & {kb(r.ops_other_nodes + r.ops_cloud)} & "
                     f"{kb(r.ops_total)} & {kb(r.mem_edge_node_B + r.mem_other_nodes_B)} & {up} & {kb(r.messages_per_h)}" + r" \\")
        if d != DS[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_syscost", L)


def realised():
    r = pd.read_csv(E / "realised_budget.csv")
    L = [r"\begin{tabular}{@{}l l lll lll@{}}", r"\toprule",
         r"& & \multicolumn{3}{c}{Nominal 10\%} & \multicolumn{3}{c}{Nominal 30\%} \\",
         r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"Dataset & Pair & Fraction [min, max] & BA real. & BA look. & Fraction [min, max] & BA real. & BA look. \\",
         r"\midrule"]
    for d in DS:
        for p in ("A", "B"):
            cells = []
            for f in (0.10, 0.30):
                x = r[(r.dataset == d) & (r.pair == p) & (r.router == "R1") & np.isclose(r.budget, f)].iloc[0]
                cells += [f"{n(x.frac_mean, 2)} [{n(x.frac_min, 2)}, {n(x.frac_max, 2)}]", n(x.ba_realised), n(x.ba_lookahead)]
            L.append(f"{DSN[d] if p == 'A' else ''} & {p} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_realised", L)


def h4():
    b = pd.read_csv(E / "h4_both_estimators.csv")
    lab = {"dcap_R6": r"$\Capt(\mathrm{R6})-\Capt(\mathrm{R1})$, plan's test",
           "dcap_R0": r"$\Capt(\mathrm{R0})-\Capt(\mathrm{R1})$, random router",
           "dcap_R9": r"$\Capt(\mathrm{R9})-\Capt(\mathrm{R1})$, label-blind schedule",
           "cap_R1": r"$\Capt(\mathrm{R1})$", "cap_R6": r"$\Capt(\mathrm{R6})$",
           "cap6_minus_cap0": r"$\Capt(\mathrm{R6})-\Capt(\mathrm{R0})$"}
    # Two stacked panels (single-column layout): pooled estimators, then the OLS slope per dataset.
    L = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.4cm} l r l r@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{Mixed model (plan)} & \multicolumn{2}{c}{Two-stage} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Left-hand side & Slope [95\% CI] & $p_1$ & Slope [95\% CI] & $p_1$ \\", r"\midrule"]
    for _, r in b.iterrows():
        L.append(f"{lab[r.lhs]} & {ci(r.mixed_slope, r.mixed_lo, r.mixed_hi)} & {pv(r.mixed_p_one_sided_neg)} & "
                 f"{ci(r.two_stage_slope, r.two_stage_lo, r.two_stage_hi)} & {pv(r.two_stage_p_one_sided_neg)}" + r" \\")
    var = pd.read_csv(E / "h4_rho_variants.csv")
    val = pd.read_csv(E / "h4_validation_rho.csv")
    L.append(r"\midrule")
    L.append(r"\multicolumn{5}{@{}l}{\textit{Plan's left-hand side, other versions of the regressor (mixed model)}} \\")
    for rho, l2 in (("rho_all", r"$\rho$ on all test windows"), ("rho_neg", r"$\rho$ within test negatives$^{\ast}$")):
        r = var[(var.rho == rho) & (var.lhs == "dcap_R6")].iloc[0]
        L.append(f"{l2} & {ci(r.slope, r.ci_lo, r.ci_hi)} & & & " + r" \\")
    r = val[val.lhs == "dcap_R6"].iloc[0]
    L.append(r"$\rho_{\mathrm{pos}}$ on the validation fold$^{\ast}$ & " + ci(r.slope, r.ci_lo, r.ci_hi) + r" & & & \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
          r"\begin{tabular}{@{}l rrrr@{}}", r"\toprule",
          r"& \multicolumn{4}{c}{OLS slope per dataset} \\", r"\cmidrule(l){2-5}",
          r"Left-hand side & Hyd. & SKAB & SEU & 3W \\", r"\midrule"]
    for _, r in b.iterrows():
        L.append(f"{lab[r.lhs]} & " + " & ".join(n(r[f"ols_{d}"], 2) for d in DS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_h4", L)


def h2():
    d = pd.read_csv(E / "h2_extra.csv")
    models = [("C(R6) ~ log2 positive windows [plan's H2 test]", r"$C(\mathrm{R6})$ (plan)"),
              ("C(R6) - C(R1) ~ log2 positive windows", r"$C(\mathrm{R6})-C(\mathrm{R1})$"),
              ("C(R1) ~ log2 positive windows", r"$C(\mathrm{R1})$")]
    # Two stacked panels (single-column layout): pooled estimators, then the two-stage slope per dataset.
    L = [r"\begin{tabular}{@{}l l r l r@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{Mixed model} & \multicolumn{2}{c}{Two-stage, pooled} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Slope of & [95\% CI] & $p_1$ & [95\% CI] & $p_1$ \\", r"\midrule"]
    P2 = []
    for m, lab in models:
        g = d[d.model == m]
        a = g[g.estimator == "MixedLM pooled"].iloc[0]
        b = g[g.estimator == "two-stage pooled (HKSJ)"].iloc[0]
        per = [g[g.estimator == f"two-stage {x} (NB over splits)"].iloc[0] for x in DS]
        L.append(f"{lab} & {ci(a.slope, a.ci_lo, a.ci_hi)} & {pv(a.p_one_sided)} & {ci(b.slope, b.ci_lo, b.ci_hi)} & "
                 f"{pv(b.p_one_sided)}" + r" \\")
        P2.append(f"{lab} & " + " & ".join(f"{n(x.slope)} ({n(x.p_one_sided, 2)})" for x in per) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
          r"\begin{tabular}{@{}l llll@{}}", r"\toprule",
          r"& \multicolumn{4}{c}{Two-stage per dataset, slope ($p_1$)} \\", r"\cmidrule(l){2-5}",
          r"Slope of & Hyd. & SKAB & SEU & 3W \\", r"\midrule"] + P2
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_h2", L)


def audit():
    br = pd.read_csv(E / "h5_breakdown.csv").set_index("subset")
    bp = pd.read_csv(E / "h5_block_perm.csv").set_index("subset")
    pre = pd.read_csv(S / "h5.csv").iloc[0]
    L = [r"\begin{tabular}{@{}l rrrrrl@{}}", r"\toprule",
         r"Cells & $n$ & GO & Prec. & Base & BA & $p$ \\", r"\midrule"]

    def row(key, lab, p=None):
        r = br.loc[key]
        pp = r.get("p_perm_within_dataset", np.nan) if p is None else p
        ps = f"{pp:.4f}" if key == "all cells (pre-specified)" else pv(pp)
        L.append(f"{lab} & {int(r.n)} & {int(r.n_go)} & {n(r.go_precision, 2)} & {n(r.base_rate, 2)} & {n(r.ba)} & {ps}" + r" \\")
    row("all cells (pre-specified)", "All, within-dataset perm.", p=float(pre.p_perm))
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


def lodo():
    d = pd.read_csv(E / "lodo.csv")
    conv = pd.read_csv(E / "mixed_model_convergence.csv").set_index("fit")
    pre = pd.read_csv(S / "h5.csv").iloc[0]
    tests = [("H1 pooled Shapley information step", "H1 pooled information step"),
             ("H2 slope of C(R6) per doubling", "H2 slope per doubling"),
             ("H4 slope of dcap on rho_pos", r"H4 slope on $\rho_{\mathrm{pos}}$"),
             ("H5 BA of audit calls", "H5 BA of calls")]
    ex = ["none (all four)", "hydraulic", "skab", "seu", "threew"]
    exn = {"none (all four)": "None", "hydraulic": "Hydraulic", "skab": "SKAB", "seu": "SEU", "threew": "3W"}
    rows = []
    for e in ex:
        cells = []
        for t, _ in tests:
            r = d[(d.excluded == e) & (d.test == t)].iloc[0]
            if t.startswith("H5"):
                p = float(pre.p_perm) if e == "none (all four)" else r.p_one_sided
                cells.append(f"{n(r.estimate)} ($p={p:.4f}$)" if e == "none (all four)" else f"{n(r.estimate)} ($p={pv(p)}$)")
            else:
                mk = ""
                if t.startswith("H4") and e != "none (all four)":
                    st = conv.loc[f"LODO without {e}: H4", "status"]
                    mk = r"$^{\ast}$" if st == "not converged" else ""
                cells.append(f"{ci(r.estimate, r.ci_lo, r.ci_hi)}{mk}")
        rows.append((exn[e], cells))
    # Two stacked panels (single-column layout): H1 and H2, then H4 and H5.
    L = []
    for k, cols in enumerate(((0, 1), (2, 3))):
        if k:
            L.append(r"\par\medskip")
        L += [r"\begin{tabular}{@{}l ll@{}}", r"\toprule",
              "Excluded & " + " & ".join(tests[c][1] for c in cols) + r" \\", r"\midrule"]
        for name, cells in rows:
            L.append(f"{name} & " + " & ".join(cells[c] for c in cols) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_lodo", L)


def convergence():
    c = pd.read_csv(E / "mixed_model_convergence.csv")
    L = [r"\begin{tabular}{@{}l r >{\raggedright\arraybackslash}p{1.7cm} l l@{}}", r"\toprule",
         r"Mixed model & $n$ & Status & Mixed slope [95\% CI] & Fixed-effects OLS [95\% CI] \\", r"\midrule"]
    for _, r in c.iterrows():
        lab = r.fit.replace("_", r"\_").replace("rho", r"$\rho$").replace("dcap", "dcap")
        L.append(f"{lab} & {int(r.n)} & {r.status} & {ci(r.mixed_slope, r.mixed_lo, r.mixed_hi)} & "
                 f"{ci(r.fe_slope, r.fe_lo, r.fe_hi)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("sup_convergence", L)


def threew_events():
    t = pd.read_csv(E / "threew_event_table.csv")
    L = [r"\begin{tabular}{@{}l rrrrrrr@{}}", r"\toprule",
         r"Instance event type & Inst. & Episodes & Wells & Windows & Fault win. & No fault win. & Both classes \\",
         r"\midrule"]
    for _, r in t.iterrows():
        L.append(f"{NAMES.get(r.instance_event_name, r.instance_event_name)} & {int(r.instances)} & {int(r.episodes)} & "
                 f"{int(r.wells)} & {int(r.windows)} & {int(r.fault_windows)} & {int(r.instances_without_fault)} & "
                 f"{int(r.instances_with_both_classes)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rev_threew_events", L)


def payload_detail():
    u = pd.read_csv(E / "uplink_per_hour.csv")
    e = pd.read_csv(E / "edge_cost.csv")
    # Two stacked panels (single-column layout): uplink per hour, then the edge node per window.
    L = [r"\begin{tabular}{@{}l rr rrrrr@{}}", r"\toprule",
         r"& & & \multicolumn{5}{c}{Uplink (B/h)} \\",
         r"\cmidrule(l){4-8}",
         r"Dataset & Win.\ (s) & Dec./h & Feat.\ 10\% & Raw 10\% & Cloud, feat. & Stream, raw & Cloud-only, raw \\",
         r"\midrule"]
    P2 = [r"\begin{tabular}{@{}l rrrr@{}}", r"\toprule",
          r"& \multicolumn{4}{c}{Edge node, per window} \\", r"\cmidrule(l){2-5}",
          r"Dataset & Feat.\ ops & Param.\ B & FFT B & R6 ops \\", r"\midrule"]
    for d in DS:
        x = u[(u.dataset == d) & np.isclose(u.budget, 0.10)].iloc[0]
        c = e[(e.dataset == d) & (e.pair == "A")].iloc[0]
        L.append(f"{DSN[d]} & {x.window_s:g} & {kb(x.windows_per_hour)} & {kb(x.pull_features_B_per_h)} & {kb(x.pull_raw_B_per_h)} & "
                 f"{kb(x.full_cloud_features_B_per_h)} & {kb(x.raw_stream_all_B_per_h)} & {kb(x.raw_stream_cloud_only_B_per_h)}" + r" \\")
        P2.append(f"{DSN[d]} & {kb(c.feature_extraction_ops)} & {int(c.edge_param_bytes_deploy)} & {kb(c.fft_workspace_B)} & "
                  f"{int(c.router_R6_ops)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\medskip"] + P2 + [r"\bottomrule", r"\end{tabular}"]
    write("sup_payload", L)


def budgets():
    r = pd.read_csv(S / "routers.csv")
    r = r[(r.pair == "A") & (r.quantity == "ba_at_budget")] if "quantity" in r.columns else r
    write_ok = True
    rows = sorted(r.router.unique(), key=lambda s: (len(s), s))
    L = [r"\begin{tabular}{@{}l " + "rr" * 4 + r"@{}}", r"\toprule",
         "& " + " & ".join(r"\multicolumn{2}{c}{" + DSN[d] + "}" for d in DS) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)),
         "Router & " + " & ".join(["10\\%", "30\\%"] * 4) + r" \\", r"\midrule"]
    for rt in rows:
        cells = []
        for d in DS:
            for f in (0.10, 0.30):
                m = r[(r.dataset == d) & (r.router == rt) & np.isclose(r.budget, f)]
                cells.append(n(m["mean"].iloc[0]) if len(m) else "--")
        L.append(f"{rt} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if write_ok:
        write("sup_budgets", L)


def realised_all():
    """Two files for the single-column layout: sup_realised.tex (Hydraulic, SKAB) and
    sup_realised_b.tex (SEU, 3W), printed as one table and its continuation."""
    r = pd.read_csv(E / "realised_budget.csv")
    for fname, dss in (("sup_realised", DS[:2]), ("sup_realised_b", DS[2:])):
        realised_part(r, fname, dss)


def realised_part(r, fname, dss):
    L = [r"\begin{tabular}{@{}l l l lll@{}}", r"\toprule",
         r"Dataset & Router & Budget & Fraction, mean (SD) [min, max] & BA real. & BA look. \\", r"\midrule"]
    for d in dss:
        first = True
        for rt in ("R0", "R1", "R2", "R3", "R9"):
            for f in (0.05, 0.10, 0.30):
                x = r[(r.dataset == d) & (r.pair == "A") & (r.router == rt) & np.isclose(r.budget, f)].iloc[0]
                L.append(f"{DSN[d] if first else ''} & {rt if np.isclose(f, 0.05) else ''} & {int(round(f * 100))}\\% & "
                         f"{n(x.frac_mean, 3)} ({n(x.frac_sd, 3)}) [{n(x.frac_min, 3)}, {n(x.frac_max, 3)}] & "
                         f"{n(x.ba_realised)} & {n(x.ba_lookahead)}" + r" \\")
                first = False
        if d != dss[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(fname, L)


def threew_wells():
    w = pd.read_csv(E / "threew_event_by_well.csv")
    L = [r"\begin{tabular}{@{}l l rrrrr@{}}", r"\toprule",
         r"Event type & Well & Inst. & Episodes & Windows & Fault win. & No fault win. \\", r"\midrule"]
    for ev, g in w.groupby("ev", sort=False):
        for i, (_, r) in enumerate(g.iterrows()):
            L.append(f"{NAMES.get(ev, ev) if i == 0 else ''} & {r.well} & {int(r.instances)} & {int(r.episodes)} & "
                     f"{int(r.windows)} & {int(r.fault_windows)} & {int(r.instances_without_fault)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("sup_threew_wells", L)


if __name__ == "__main__":
    for f in (skab, faults, oracle, arch, syscost, realised, h4, h2, audit, lodo, convergence, threew_events, payload_detail,
              budgets, realised_all, threew_wells):
        f()
    print("second-batch tables written to", OUT)
