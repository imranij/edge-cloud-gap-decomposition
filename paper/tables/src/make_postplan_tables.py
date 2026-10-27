#!/usr/bin/env python
"""Tables of the paper that use the POST-PLAN analyses.  Reads only; fits no model.

    python paper/tables/src/postplan_summary.py      # per-variant summaries -> paper/tables/data/pp_variants.csv
    python paper/tables/src/make_postplan_tables.py  # this script

Main-text tables (paper/tables/):
  pp_overview.tex   decomposition of the pre-specified run (pair A), pair B information step, shortcut baselines
  pp_controls.tex   information step, cloud and edge accuracy and router C under every confound control
  pp_routers.tex    router concordance: pre-specified routers and the cross-fitted post-plan routers
  pp_tests.tex      the five pre-specified tests, the deconfounded replication and the corrected designs
  pp_arch.tex       architecture comparison, pre-specified run and declared deconfounded variants
Supplement tables (paper/tables/sup_pp_*.tex): every shortcut predictor, the full decomposition of every
variant, the corrected audit, the fixed-tier H2, streaming thresholds, the feature-aware combiner and the
simulation.

Sources (every number is read from one of these files):
  results/summary/{per_dataset,primary,h1_pooled,h2_model,h3_pooled,h4_model,h5}.csv  (pre-specified run)
  results/extra/{arch_paired,architectures,h2_extra,h4_two_stage}.csv                 (evaluation-only)
  results_sens/shortcuts/shortcuts.csv
  paper/tables/data/pp_variants.csv  <- results_sens/{hydskab,threewseu}/<variant>/<pair>/split_*.parquet
  results_sens/deconf/summary/{primary,h1_pooled,h2_model,h3_pooled,h4_model,h5}.csv     (declared replication)
  results_sens/routers/{p1_ba,p1_C_summary,p2_routers,p2_diffs,p2_stream_summary,p2_comb_summary,p3_slopes}.csv
  results_sens/simulation/rates.csv
Anything not yet complete is printed as \\PENDING{...} naming its source, never as a number.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "paper" / "tables"
RS = ROOT / "results_sens"
MAIN = pd.read_csv(ROOT / "results" / "summary" / "per_dataset.csv")
MAIN["router"] = MAIN["router"].fillna("")
PV = pd.read_csv(TAB / "data" / "pp_variants.csv")
PV["router"] = PV["router"].fillna("")
DS = ["hydraulic", "skab", "seu", "threew"]
DSNAME = {"hydraulic": "Hydraulic", "skab": "SKAB", "seu": "SEU", "threew": "3W"}
SIM_TARGET_REPS = 500   # replicates planned for the full validation (1,000 for null scenarios)
PILOT_TARGET_REPS = 100  # the pilot of the corrected designs is reported at R = 100 (results_sens/simulation/)


# ------------------------------------------------------------------ formatting
def num(x, d=3):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        if float(s) == 0:
            s = s[1:]
        else:
            s = "$-$" + s[1:]
    return s


def ci(m, lo, hi, d=3):
    if not np.isfinite(lo):
        return num(m, d)
    return f"{num(m, d)} [{num(lo, d)}, {num(hi, d)}]"


def pval(p):
    if not np.isfinite(p):
        return "--"
    if p < 1e-3:
        e = f"{p:.1e}"
        man, ex = e.split("e")
        return f"${man}\\times10^{{{int(ex)}}}$"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"


def pend(what):
    return f"\\PENDING{{{what}}}"


def tex(s):
    return s.replace("_", "\\_")


def write(name, body):
    (TAB / name).write_text(body)
    print("wrote", name)


# ------------------------------------------------------------------ lookups
def main_row(ds, pair, q, router=""):
    m = (MAIN.dataset == ds) & (MAIN.pair == pair) & (MAIN.quantity == q) & (MAIN.router == router) & MAIN.budget.isna()
    g = MAIN[m]
    return None if g.empty else g.iloc[0]


def main_budget(ds, pair, router, b):
    m = (MAIN.dataset == ds) & (MAIN.pair == pair) & (MAIN.quantity == "ba_at_budget") & (MAIN.router == router) \
        & np.isclose(MAIN.budget.fillna(-1), b)
    g = MAIN[m]
    return None if g.empty else g.iloc[0]


def pv_row(v, pair, q, router=""):
    m = (PV.dataset == v) & (PV.pair == pair) & (PV.quantity == q) & (PV.router == router) & PV.budget.isna()
    g = PV[m]
    return None if g.empty else g.iloc[0]


def pv_complete(v, pair):
    g = PV[(PV.dataset == v) & (PV.pair == pair)]
    return bool(len(g)) and bool(g.complete.max())


def pv_J(v, pair):
    g = PV[(PV.dataset == v) & (PV.pair == pair)]
    j = g.J_pair.max() if "J_pair" in g and len(g) else 0
    return 0 if not np.isfinite(j) else int(j)


def cell(r, with_ci=True, d=3):
    if r is None:
        return "--"
    return ci(r["mean"], r["ci_lo"], r["ci_hi"], d) if with_ci else num(r["mean"], d)


# ------------------------------------------------------------------ table 1: overview of the pre-specified run
def overview():
    sc = pd.read_csv(RS / "shortcuts" / "shortcuts.csv")
    # Best shortcut per dataset (highest mean BA), as in Table tab:survival and the caption of tab:overview (audit
    # SEU and 3W previously showed the two-edge-means and well-fault-rate predictors, not the best).
    best = {d: sc.loc[sc[sc.dataset == d].ba_mean.idxmax(), "shortcut"] for d in DS}
    assert best == {"hydraulic": "a_cycle_1nn", "skab": "a_t_start_tree3", "seu": "b_cloud8_means_logreg",
                    "threew": "b_well_onehot_logreg"}, best
    rows = [
        ("Edge $B_{eE}$", "ba_ef_ev", True), ("Upgraded edge $B_{cE}$", "ba_cf_ev", False),
        ("Gateway $B_{eC}$", "ba_ef_cv", False), ("Cloud $B_{cC}$", "ba_cf_cv", True),
        ("Oracle", "ba_oracle_dep", False),
        ("Model step", "shapley_model", True),
        ("Information step", "shapley_info", True),
        ("Interaction", "interaction", False),
        ("Combination step", "comb_step_dep", False),
        ("Unresolved", "unresolved_dep", False),
    ]
    L = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.2cm} llll@{}}", "\\toprule",
         "Pair A & " + " & ".join(DSNAME[d] for d in DS) + " \\\\", "\\midrule"]
    for lab, q, wci in rows:
        L.append(lab + " & " + " & ".join(cell(main_row(d, "A", q), wci) for d in DS) + " \\\\")
    hv = pd.read_csv(ROOT / "results" / "extra" / "info_step_halves.csv")
    for lab, qq in (("\\quad specificity half$^{b}$", "info_spec_half"), ("\\quad recall half$^{b}$", "info_recall_half")):
        L.append(lab + " & " + " & ".join(
            ci(*hv[(hv.dataset == d) & (hv.pair == "A") & (hv.quantity == qq)][["mean", "ci_lo", "ci_hi"]].iloc[0].tolist())
            for d in DS) + " \\\\")
    L.append("\\addlinespace")
    L.append("Pair B, information step & " + " & ".join(cell(main_row(d, "B", "shapley_info")) for d in DS)
             + " \\\\")
    L.append("Shortcut baseline$^{a}$ & " + " & ".join(
        ci(*sc[(sc.dataset == d) & (sc.shortcut == best[d])][["ba_mean", "ci_lo", "ci_hi"]].iloc[0].tolist())
        for d in DS) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("pp_overview.tex", "\n".join(L) + "\n")


# ------------------------------------------------------------------ table 2: confound controls
# Labels: d = declared deconfounded variant; e = non-causal (uses later samples of the recording or instance);
# z = z-scored over the whole recording, so the normalisation depends on the file's label composition;
# f = one split per held-out level or session; g = sessions coincide with disturbance families (SKAB).
# The level and session hold-outs are the REDESIGNED ones
# (results_sens/holdout2/; fit + validation = every other level or session). The plan's design (one fit level or
# session) is kept only in the supplement's per-fold diagnostics (sup_pp_holdout.tex).
CONTROLS = [
    ("hydraulic", [("Pre-specified splits", None, "AB"), ("Block-grouped in time\\,$^{d}$", "hyd_blocks", "AB"),
                   ("Weak leakage only", "hyd_weak", "AB"), ("Severe leakage only", "hyd_severe", "AB"),
                   ("Cooler level held out$^{f}$", "hyd_cooler2", "AB"),
                   ("Accumulator level held out$^{f}$", "hyd_accum2", "AB")]),
    ("skab", [("Pre-specified splits", None, "AB"), ("Z-scored within file$^{d,e,z}$", "skab_filez", "AB"),
              ("Offset from first 100 s (causal)", "skab_base", "AB"),
              ("Session and disturbance family held out$^{f,g}$", "skab_lso2", "AB"),
              # the same hold-out on the causal-offset features
              ("Session held out, causal offset$^{f,g}$", "skab_lso2_base", "AB"),
              ("Cloud without flow", "skab_cm_flow", "A")]),
    # the remaining SKAB channel controls move to the supplement table (pp_controls_b)
    ("skab_more", [("No temperature channels", "skab_notemp", "AB"), ("Pressure moved to edge", "skab_edgeP", "AB"),
                   ("Cloud without pressure", "skab_cm_pressure", "A"),
                   ("Cloud without fluid temperature", "skab_cm_thermo", "A")]),
    # two hydraulic controls move to the supplement table (pp_controls_b)
    ("hyd_more", [("Unchunked runs", "hyd_runs", "AB"), ("Transitional cycles removed", "hyd_stable", "AB")]),
    ("seu", [("Pre-specified splits", None, "AB"), ("De-meaned windows$^{d}$", "seu_dm", "AB"),
             ("Z-normalised windows", "seu_zn", "AB")]),
    ("threew", [("Pre-specified splits (episodes)", None, "AB"), ("Wells held out", "threew__by_well", "A"),
                ("De-meaned windows$^{d}$", "threew_dm", "AB"), ("De-meaned, wells held out", "threew_dm__by_well", "AB"),
                ("Z-normalised windows", "threew_zn", "AB"), ("Z-normalised, wells held out", "threew_zn__by_well", "AB"),
                ("Instance-normalised$^{e}$", "threew_inorm", "AB"),
                ("Instance-normalised, wells held out$^{e}$", "threew_inorm__by_well", "AB"),
                ("Cloud without annulus pressure", "threew_noann", "AB")]),
]
CTRL_Q = [("ba_ef_ev", False), ("ba_cf_cv", False), ("shapley_info", True)]
# hold-outs whose tiers predict (almost) one class (>= 95% of test windows) in every fold carry no
# routing information; C is printed as '--' there (results/extra/holdout_diagnostics.csv, redesign rows).
ONE_CLASS_SHARE = 0.95


def one_class_holdout(v, pair):
    d = pd.read_csv(ROOT / "results" / "extra" / "holdout_diagnostics.csv")
    d = d[(d.variant == v) & (d.pair == pair) & d.design.str.contains("redesign")]
    if d.empty:
        return False
    oc = lambda p: np.maximum(p, 1 - p) >= ONE_CLASS_SHARE
    return bool((oc(d.ppr_edge) | oc(d.ppr_cloud)).all())


UNDEFINED_A: set = set()   # the plan design's one-fit-group hold-outs are no longer in Table 3 (redesign has >= 10)
SC_MAIN = pd.read_csv(RS / "shortcuts" / "shortcuts.csv")
SC_CTRL = pd.read_csv(RS / "shortcuts_controls" / "summary.csv")
SC_DS = {"hydraulic": "hydraulic", "skab": "skab", "seu": "seu", "threew": "threew"}


def area_of(v):
    if v in ("skab_lso2", "hyd_cooler2", "hyd_accum2", "skab_lso2_base"):
        return "holdout2"
    return "hydskab" if v.startswith(("hyd", "skab")) else "threewseu"


def same_folds(v, ref_dir):
    """True if every stored test fold of variant v equals the reference run's (only features changed)."""
    for pair in ("A", "B"):
        d = RS / area_of(v) / v / pair
        fs = sorted(f for f in d.glob("arrays_*.npz") if "_q" not in f.stem) if d.exists() else []
        if not fs:
            continue
        for f in fs:
            ref = ref_dir / pair / f.name
            if not ref.exists():
                ref = ref_dir / "A" / f.name
            if not np.array_equal(np.load(f)["test_idx"], np.load(ref)["test_idx"]):
                return False
        return True
    return False


def shortcut_cell(ds, v):
    """Best metadata shortcut (max mean BA over the dataset's predictors) on the condition's own splits."""
    if v is None:
        g = SC_MAIN[SC_MAIN.dataset == SC_DS[ds]]
        return num(g.ba_mean.max())
    if v == "threew__by_well":
        return num(SC_MAIN[SC_MAIN.dataset == "threew__by_well"].ba_mean.max())
    g = SC_CTRL[SC_CTRL.variant == v]
    if len(g) and int(g.J.min()) >= expected_J(v):
        return num(g.ba_mean.max())
    if v.endswith("__by_well"):
        ref = ROOT / "results" / "threew__by_well"
        return (num(SC_MAIN[SC_MAIN.dataset == "threew__by_well"].ba_mean.max()) + "$^{s}$") if same_folds(v, ref) else "--"
    if v in ("hyd_cooler", "hyd_accum", "skab_lso", "hyd_stable", "skab_lso2", "hyd_cooler2", "hyd_accum2",
             "skab_lso2_base"):
        return "--"
    if same_folds(v, ROOT / "results" / ds):
        return num(SC_MAIN[SC_MAIN.dataset == SC_DS[ds]].ba_mean.max()) + "$^{s}$"
    return "--"


def expected_J(v):
    g = PV[PV.dataset == v]
    return int(g.J_expected.max()) if len(g) else 20


def controls():
    head = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.9cm} c r lll ll l@{}}", "\\toprule",
            "Condition & Pair & $J$ & $B_{eE}$ & $B_{cC}$ & $\\bar\\Delta_{\\mathrm{inf}}$ [95\\% CI] & $C$(R1) & $C$(R6)"
            " & Shortcut$^{h}$ \\\\",
            "\\midrule"]
    L = list(head)
    pending = []
    for ds, conds in CONTROLS:
        if ds == "skab_more":
            L += ["\\bottomrule", "\\end{tabular}"]
            write("pp_controls.tex", "\n".join(L) + "\n")
            L = list(head)
        title = {"skab_more": "SKAB, channel controls", "hyd_more": "Hydraulic, further controls"}.get(ds, DSNAME.get(ds))
        ds = {"skab_more": "skab", "hyd_more": "hydraulic"}.get(ds, ds)
        L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for lab, v, pairs in conds:
            sc = shortcut_cell(ds, v)
            for k, pair in enumerate(pairs):
                name = lab if k == 0 else ""
                scc = sc if k == 0 else ""
                if v is None or v == "threew__by_well":
                    src = ds if v is None else v
                    r = [main_row(src, pair, q) for q, _ in CTRL_Q]
                    if r[0] is None:
                        continue
                    J = int(r[0]["J"])
                    cells = [cell(x, w) for x, (_, w) in zip(r, CTRL_Q)]
                    cells += [cell(main_row(src, pair, "C", rt), False) for rt in ("R1", "R6")]
                    L.append(f"{name} & {pair} & {J} & " + " & ".join(cells) + f" & {scc} \\\\")
                    continue
                if pair == "A" and v in UNDEFINED_A:
                    L.append(f"{name} & {pair} & -- & \\multicolumn{{5}}{{l}}{{undefined (one fit group; no R8 cross-fit)}} & {scc} \\\\")
                    continue
                if not pv_complete(v, pair):
                    j = pv_J(v, pair)
                    pending.append(f"{v}/{pair}")
                    L.append(f"{name} & {pair} & {j} & \\multicolumn{{5}}{{l}}{{"
                             + pend(f"{tex(v)}/{pair}, {j} of {expected_J(v)}; R3")
                             + f"}} & {scc} \\\\")
                    continue
                r = [pv_row(v, pair, q) for q, _ in CTRL_Q]
                cells = [cell(x, w) for x, (_, w) in zip(r, CTRL_Q)]
                if one_class_holdout(v, pair):
                    cells += ["--", "--"]
                else:
                    cells += [cell(pv_row(v, pair, "C", rt), False) for rt in ("R1", "R6")]
                L.append(f"{name} & {pair} & {pv_J(v, pair)} & " + " & ".join(cells) + f" & {scc} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("pp_controls_b.tex", "\n".join(L) + "\n")
    return pending


# ------------------------------------------------------------------ survival criterion, one row per dataset
def survival():
    """Table 4 (stronger criterion (ii+) against the shortcut-augmented edge where the
    runs exist). Source: results/extra/survival_summary.csv (src/extra/survival_summary.py). Also
    verdicts per pair (verdict_pairs); (ii+) redefined as the cloud minus the better, per
    split, of the augmented edge and the shortcut alone; SKAB labelled benchmark-confounded; extra causal-offset
    SKAB row (skab_base, not the declared variant)."""
    s = pd.read_csv(ROOT / "results" / "extra" / "survival_summary.csv")
    vl = {"hyd_blocks": "block-grouped in time", "skab_filez": "z-scored within file$^{e}$",
          "seu_dm": "de-meaned windows", "threew_dm": "de-meaned windows",
          "skab_base": "causal offset, not declared$^{d}$"}
    scl = {"c_settings_tree3": "component settings, tree", "a_cycle_1nn": "cycle index, 1-NN",
           "a_t_start_tree3": "time in file, tree", "b_cloud8_means_logreg": "eight channel means",
           "b_well_onehot_logreg": "well identity", "a_well_rate": "well fault rate"}
    noaug = {"seu": "none (no shortcut)", "threew": "not run$^{c}$", "skab_offset": "not run"}
    dsname = dict(DSNAME, skab_offset="SKAB")
    P = ">{\\raggedright\\arraybackslash}p"
    C = f"{P}{{2.1cm}}"

    def ci(m, lo, hi, mark=""):                       # mean on one line, interval on the next
        return f"{num(m)}{mark}\\newline [{num(lo)}, {num(hi)}]"
    L = [f"\\begin{{tabular}}{{@{{}}{P}{{2.15cm}} c {C} {P}{{1.55cm}} {C} {C} {C}@{{}}}}", "\\toprule",
         "Dataset, variant, verdict$^{b}$ & Pair & $\\bar\\Delta_{\\mathrm{inf}}$ [95\\% CI] & "
         "Best shortcut (BA) & $B_{cC}$ minus shortcut & $B_{cC}$ minus better of augmented edge and shortcut$^{a}$ & "
         "$\\bar\\Delta_{\\mathrm{inf}}$, shortcut in both views$^{a}$ \\\\", "\\midrule"]
    pending = []
    order = ["hydraulic", "skab", "skab_offset", "seu", "threew"]
    s = s.set_index("dataset").loc[[d for d in order if d in set(s.dataset)]].reset_index()
    for _, r in s.iterrows():
        for k, p in enumerate("AB"):
            if r.dataset in noaug:
                aug = ("\\multicolumn{2}{l}{" + noaug[r.dataset] + "}") if k == 0 else " & "
            elif not bool(r.get(f"aug_complete_{p}", False)):
                pending.append(f"{r.aug_variant}/{p}")
                aug = "\\multicolumn{2}{l}{" + pend(f"{tex(r.aug_variant)}/{p}; R4") + "}"
            else:
                mark = "$^{f}$" if r.dataset == "skab" else ""
                aug = (f"{ci(r[f'cloud_minus_best_{p}'], r[f'cmb_{p}_lo'], r[f'cmb_{p}_hi'], mark)} & "
                       f"{ci(r[f'info_aug_{p}'], r[f'info_aug_{p}_lo'], r[f'info_aug_{p}_hi'], mark)}")
            verdict = r.verdict_pairs
            first = f"{dsname[r.dataset]}, {vl[r.variant]}" if k == 0 else f"\\textit{{{verdict}}}"
            L.append(f"{first} & {p} & {ci(r[f'info_{p}'], r[f'info_{p}_lo'], r[f'info_{p}_hi'])} & "
                     + (f"{scl.get(r.best_shortcut, r.best_shortcut)} ({num(r.shortcut_ba)})" if k == 0 else "")
                     + f" & {ci(r[f'cloud_minus_shortcut_{p}'], r[f'cms_{p}_lo'], r[f'cms_{p}_hi'])} & " + aug
                     + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("pp_survival.tex", "\n".join(L) + "\n")
    return pending


# ------------------------------------------------------------------ table 3: routers
def routers():
    p2 = pd.read_csv(RS / "routers" / "p2_routers.csv")
    p2d = pd.read_csv(RS / "routers" / "p2_diffs.csv")
    cols = [(d, p) for d in DS for p in ("A", "B")]

    def p2_ok(d, p):
        g = p2[(p2.dataset == d) & (p2.pair == p) & (p2.quantity == "C") & (p2.router == "R1")]
        return len(g) and int(g.J.iloc[0]) >= 20

    def p2c(d, p, rt):
        if not p2_ok(d, p):
            return None
        g = p2[(p2.dataset == d) & (p2.pair == p) & (p2.quantity == "C") & (p2.router == rt)]
        return None if g.empty else g.iloc[0]["mean"]

    L = ["\\begin{tabular}{@{}l " + "rr" * 4 + "@{}}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DSNAME[d]}}}" for d in DS) + " \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
         "Router & " + " & ".join(p for _, p in cols) + " \\\\", "\\midrule",
         f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{Pre-specified (learned routers trained on the validation fold)}}}} \\\\"]
    for rt, lab in [("R0", "R0 random"), ("R9", "R9 periodic"), ("R1", "R1 confidence"), ("R2", "R2 entropy"),
                    ("R4", "R4 Diff-01"), ("R5", "R5 Diff-Prob"), ("R6", "R6 gain regressor"), ("R8", "R8 L2D")]:
        L.append(lab + " & " + " & ".join(cell(main_row(d, p, "C", rt), False) for d, p in cols) + " \\\\")
    L.append("\\multicolumn{9}{@{}l}{\\textit{Post-plan (trained on cross-fitted fit and validation predictions)}} \\\\")
    pend_cols = [f"{d}/{p}" for d, p in cols if not p2_ok(d, p)]

    def p2cells(d, fn):
        if not p2_ok(d, "A") or not p2_ok(d, "B"):
            if not p2_ok(d, "A") and not p2_ok(d, "B"):
                return "\\multicolumn{2}{c}{" + pend(DSNAME[d]) + "}"
            return " & ".join(fn(d, p) if p2_ok(d, p) else pend(DSNAME[d]) for p in ("A", "B"))
        return " & ".join(fn(d, p) for p in ("A", "B"))
    for rt, lab in [("R4x", "R4x"), ("R5x", "R5x"), ("R6x", "R6x"), ("HyR", "Rank average R1, R6x"),
                    ("HyS", "Selected R1 or R6x"), ("R10", "R10 two-stage surrogate")]:
        L.append(lab + " & " + " & ".join(p2cells(d, lambda d, p: num(p2c(d, p, rt))) for d in DS) + " \\\\")
    # the paired-difference block moved to Table S-routers-well (sup_routers_well)
    L += ["\\bottomrule", "\\end{tabular}"]
    write("pp_routers.tex", "\n".join(L) + "\n")
    return pend_cols


# ------------------------------------------------------------------ table 4: tests
def short_verdict(v):
    return v.split(" (")[0]


def tests():
    """Table 5: verdicts as the plan's rules give them, beside the overlap-corrected two-stage estimates
, the declared replication, and the proposed corrected
    designs. The simulation that would validate the corrected designs is not complete, so it has
    no column; the caption says so."""
    mp = pd.read_csv(ROOT / "results" / "summary" / "primary.csv").set_index("hypothesis")
    dp = pd.read_csv(RS / "deconf" / "summary" / "primary.csv").set_index("hypothesis")
    oc = pd.read_csv(ROOT / "results" / "extra" / "overlap_corrected_tests.csv")
    p3 = pd.read_csv(RS / "routers" / "p3_slopes.csv")
    aw = pd.read_csv(ROOT / "results" / "extra" / "audit_within_class.csv")
    c4 = pd.read_csv(ROOT / "results" / "extra" / "corrected_h4_val.csv")
    h3f = pd.read_csv(ROOT / "results" / "extra" / "h3_feasibility.csv").set_index("test").loc["pooled"]
    pooled_ft = p3[(p3.dataset == "POOLED (HKSJ)") & p3.design.str.startswith("fixed tiers")].iloc[0]
    ft_p = float(stats.t.sf(pooled_ft.two_stage_mean / pooled_ft.se, 3))

    hf = pd.read_csv(ROOT / "results" / "extra" / "r5_h4_factor.csv")   # src/extra/r5_h4_factor.py

    def ts(run, h):
        g = oc[(oc.run == run) & (oc.hypothesis == h) & (oc.estimator == "two-stage pooled (HKSJ)")]
        if g.empty:
            return "--"
        r = g.iloc[0]
        out = f"{ci(r.estimate, r.ci_lo, r.ci_hi)}; $p_1$ {num(r.p_one_sided)}, $p_{{\\mathrm{{Holm}}}}$ {num(r.p_holm)}"
        if h == "H4" and run == "pre-specified":     # the replication without the factor is in r5_h4_factor.csv
            n = hf[(hf.test == "pre-specified H4") & hf.variance_factor.str.startswith("none")].iloc[0]
            out += (f"; no factor$^{{b}}$ {ci(n.estimate, n.ci_lo, n.ci_hi)}; $p_1$ {num(n.p_one_sided)}, "
                    f"$p_{{\\mathrm{{Holm}}}}$ {num(n.p_holm_two_stage_set)}")
        return out

    def aud(oc_name):
        r = aw[(aw.null == "within-class null") & (aw.subset == "all cells") & (aw.cells == "undefined as NO-GO")
               & (aw.outcome == oc_name)].iloc[0]
        return num(r.ba)
    c4p = c4[(c4.run == "corrected H4 (post-plan)") & (c4.estimator == "two-stage pooled (HKSJ)")].iloc[0]
    c4n = hf[(hf.test == "corrected H4 (proposed)") & hf.variance_factor.str.startswith("none")].iloc[0]
    # minimum detectable effect at 80% power (results/extra/r4_mde.csv, src/extra/r4_power.py)
    mde = pd.read_csv(ROOT / "results" / "extra" / "r4_mde.csv").set_index("design").mde_80
    m_h2 = mde["H2 fixed tiers, two-stage (proposed)"]
    m_h4 = mde["H4 corrected, Cap(R6)-Cap(R0) on validation rho (proposed)"]
    m_a1 = mde["audit, within-class null, BA of calls (permutation within datasets)"]
    m_a2 = mde["audit, within-class null, BA of calls (permutation within dataset x pair x level)"]
    S = _simf()                         # full validation (post-plan), results_sens/simulation_full/summary

    def sv(key, sc, d=2):
        v = S[(key, sc)][0]
        return num(v, d)
    sim = {
        "H1": (f"Sim.$^{{d}}$: NB coverage {sv('h1_ds_cover', 'S1b')}--{sv('h1_ds_cover', 'S0')}, "
               f"{sv('h1_ds_cover_truth', 'T1_0.6')} or more under adjacent-block dependence"),
        "H2": (f"; sim.$^{{d}}$ size {sv('h2_cor_reject', 'S1b', 3)}, power {sv('h2_cor_reject', 'S2')} and "
               f"{sv('h2_cor_reject', 'S3')}"),
        "H3": (f"; sim.$^{{d}}$ declared in at most {sv('h3_pool_equiv', 'S3')} at $dC=0$, "
               f"never at $dC=0.02$"),
        "H4": (f"; sim.$^{{d}}$ size {sv('h4_2s_factor_reject', 'S0', 0)} with the factor, "
               f"{sv('h4_2s_repeat_naive_reject', 'S0')} with slopes per repeat"),
        "H5": (f"; sim.$^{{d}}$ GO in {sv('go_ds|go_rate|out_C', 'S0')} of cells without information "
               f"(nominal 0.05), BA of calls $\\le${sv('go_ds|ba|out_C', 'S3')}"),
    }
    corrected = {
        "H1": sim["H1"],
        "H2": (f"Fixed tiers, two-stage: {ci(pooled_ft.two_stage_mean, pooled_ft.ci_lo, pooled_ft.ci_hi)}; "
               f"$p_1$ {num(ft_p)}; MDE {num(m_h2)}" + sim["H2"]),
        "H3": (f"Feasibility at the realised $J$: infeasible ($p_{{\\mathrm{{eq}}}}$ = {num(h3f.p_eq_if_dC_zero)} "
               f"at $dC=0$; about {int(h3f.J_needed)} splits needed)" + sim["H3"]),
        "H4": (f"$\\Capt(\\mathrm{{R6}})-\\Capt(\\mathrm{{R0}})$ on validation $\\rho_{{\\mathrm{{pos}}}}$, two-stage: "
               f"{ci(c4p.estimate, c4p.ci_lo, c4p.ci_hi)}; $p_1$ {num(c4p.p_one_sided)}; MDE $-${num(m_h4, 2)} "
               f"($-${num(c4n.mde_80, 2)} with no factor$^{{b}}$)" + sim["H4"]),
        "H5": (f"Within-class null: BA {aud('routing beats random')} (beats random), "
               f"{aud('escalation beats the edge')} (beats edge); MDE {num(m_a1)} or {num(m_a2)}" + sim["H5"]),
    }
    W = ">{\\raggedright\\arraybackslash}p"
    L = [f"\\begin{{tabular}}{{@{{}}l {W}{{2.1cm}} {W}{{2.65cm}} {W}{{2.1cm}} {W}{{2.65cm}} {W}{{2.7cm}}@{{}}}}", "\\toprule",
         "& Pre-specified verdict ($p_{\\mathrm{Holm}}$) & Two-stage$^{a}$ "
         "& Deconfounded replication$^{c}$ ($p_{\\mathrm{Holm}}$) & Replication, two-stage$^{a}$ "
         "& Proposed corrected design \\\\", "\\midrule"]
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        L.append(f"{h} & {short_verdict(mp.loc[h, 'verdict'])}\\newline ({pval(mp.loc[h, 'p_holm'])}) & "
                 f"{ts('pre-specified', h) if h in ('H2', 'H4') else '--'} & "
                 f"{short_verdict(dp.loc[h, 'verdict'])}\\newline ({pval(dp.loc[h, 'p_holm'])}) & "
                 f"{ts('deconfounded replication', h) if h in ('H2', 'H4') else '--'} & {corrected[h]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("pp_tests.tex", "\n".join(L) + "\n")
    return []   # the full simulation (R2) was not run; Table 6 says the corrected designs are not validated


# ------------------------------------------------------------------ table 5: architectures
DECL = {"hydraulic": ("hydskab", "hyd_blocks"), "skab": ("hydskab", "skab_filez"),
        "seu": ("threewseu", "seu_dm"), "threew": ("threewseu", "threew_dm")}


VLAB = {"hyd_blocks": "blocked in time", "skab_filez": "z-scored in file", "seu_dm": "de-meaned",
        "threew_dm": "de-meaned"}


def variant_split_values(area, v, pair):
    fs = sorted((RS / area / v / pair).glob("split_*.parquet"))
    fs = [f for f in fs if "_q" not in f.stem]
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d.h2_q == 1.0]
    d["router"] = d["router"].fillna("")

    def per_split(q, r="", b=None):
        m = (d.quantity == q) & (d.router == r)
        m &= d.budget.isna() if b is None else np.isclose(d.budget.fillna(-1), b)
        return d[m].set_index("split")["value"].sort_index()
    sz = d.drop_duplicates("split")[["n_train", "n_test"]].mean()
    return per_split, float(sz.n_train), float(sz.n_test)


def nb(x, ntr, nte):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    J = len(x); m = x.mean(); sd = x.std(ddof=1)
    se = sd * np.sqrt(1 / J + nte / ntr)
    t = stats.t.ppf(0.975, J - 1)
    return m, m - t * se, m + t * se


def arch():
    """Table 6: accuracy at a nominal 10% budget with random, periodic, R1 (look-ahead and
    causal streaming threshold), R6 and R6x escalation beside the gateway and the cloud, with paired intervals;
    and a cost panel (results/extra/system_cost_r3.csv). Rows: pre-specified splits, the declared deconfounded
    variants and the causal SKAB offset variant. Source: results/extra/arch_r3.csv (src/extra/arch_r3.py)."""
    a = pd.read_csv(ROOT / "results" / "extra" / "arch_r3.csv")
    # deployable R1 and the paired gateway comparison against it (src/extra/r4_systems.py).
    # Pre-specified rows: R1r, the causal threshold from cross-fitted training scores. Variant rows (no cross-fitted
    # scores exist): R1v, the threshold set on the stored validation fold, marked v.
    dep = pd.read_csv(ROOT / "results" / "extra" / "r4_arch_deployable.csv")
    vlab = {"hyd_blocks": "blocks", "skab_base": "offset", "skab_filez": "z-sc.$^{e}$",
            "seu_dm": "de-m.", "threew_dm": "de-m."}
    mark = {"hydraulic": "$^{\\S}$", "skab": "$^{\\P}$", "seu": "$^{\\dagger}$", "threew": "$^{\\ddagger}$"}
    MAIN_VARIANTS = {"hyd_blocks", "skab_base"}   # other variant rows in sup_pp_arch_full.tex
    Lfull = None
    L = ["\\begin{tabular}{@{}l rrrrrrrrrr ll@{}}", "\\toprule",
         "& & \\multicolumn{7}{c}{Escalation of 10\\% of windows} & & & \\multicolumn{2}{c}{Paired differences [95\\% CI]} \\\\",
         "\\cmidrule(lr){3-9}\\cmidrule(l){12-13}",
         "Pair & Edge & Rand. & R9 & R1 & R1r & $f$ & R6 & R6x & Gate. & Cloud & R1 $-$ random & Gateway $-$ R1r \\\\",
         "\\midrule"]
    FULL = list(L)
    # the main table drops the R6x column too (both R6 and R6x stay in sup_pp_arch_full.tex)
    # the main table drops R9 (kept in sup_pp_arch_full.tex) and adds the cloud model run at the
    # co-located gateway, cloud - R1r paired over splits (results/extra/r6_cloud_gateway.csv, src/extra/r6_systems.py)
    cg = pd.read_csv(ROOT / "results" / "extra" / "r6_cloud_gateway.csv")
    # R1 - random moved to sup_pp_arch_full.tex (quoted in the text for the hydraulic rig)
    L = ["\\begin{tabular}{@{}l rrrrrrr ll@{}}", "\\toprule",
         "& & \\multicolumn{4}{c}{Escalation of 10\\% of windows} & & & \\multicolumn{2}{c}{Paired differences [95\\% CI]} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(l){9-10}",
         "Pair & Edge & Rand. & R1 & R1r & $f$ & Gate. & Cloud$^{w}$ & Gateway $-$ R1r & Cloud $-$ R1r \\\\",
         "\\midrule"]
    for ds in DS:
        FULL.append(f"\\multicolumn{{13}}{{@{{}}l}}{{\\textit{{{DSNAME[ds]}}}{mark[ds]}}} \\\\")
        L.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{DSNAME[ds]}}}{mark[ds]}}} \\\\")
        g = a[a.dataset == ds]
        order = [r for _, r in g[g.variant == ds].iterrows()] + [r for _, r in g[g.variant != ds].iterrows()]
        for r in order:
            lab = r.pair if r.variant == ds else f"{r.pair}, {vlab[r.variant]}"
            d = dep[(dep.variant == r.variant) & (dep.pair == r.pair)].iloc[0]
            assert abs(d.edge - r.edge) < 1e-9 and abs(d.gateway - r.gateway) < 1e-9
            if r.variant == ds:
                assert abs(d.R1r - r.R1_realised) < 1e-9
                r1d, fr, gd = num(d.R1r), num(d.R1r_frac, 2), ci(d["gateway - R1r mean"], d["gateway - R1r lo"], d["gateway - R1r hi"])
            else:
                r1d, fr = num(d.R1v) + "$^{v}$", num(d.R1v_frac, 2)
                gd = ci(d["gateway - R1v mean"], d["gateway - R1v lo"], d["gateway - R1v hi"]) + "$^{v}$"
            vals = [num(r[c]) for c in ("edge", "random", "R9", "R1")] + [r1d, fr] + \
                   [num(r[c]) for c in ("R6", "R6x", "gateway", "cloud")]
            dd = [ci(r["R1 - random mean"], r["R1 - random lo"], r["R1 - random hi"]), gd]
            FULL.append(f"{lab} & " + " & ".join(vals + dd) + " \\\\")
            if r.variant == ds or r.variant in MAIN_VARIANTS:     # main table: no R9, R6, R6x columns (width)
                k = cg[(cg.variant == r.variant) & (cg.pair == r.pair)].iloc[0]
                assert abs(k.cloud - r.cloud) < 1e-9
                cd = ci(k["cloud_at_gateway - causal R1 mean"], k["cloud_at_gateway - causal R1 lo"],
                        k["cloud_at_gateway - causal R1 hi"]) + ("" if r.variant == ds else "$^{v}$")
                L.append(f"{lab} & " + " & ".join(vals[:2] + vals[3:6] + vals[8:] + dd[1:] + [cd]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_arch_full.tex", "\n".join(FULL + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    # cost panel
    c = pd.read_csv(ROOT / "results" / "extra" / "system_cost_r3.csv")

    def si(x):
        if not np.isfinite(x):
            return "--"
        for f, u in ((1e6, "\\,M"), (1e3, "\\,k")):
            if abs(x) >= f:
                v = x / f
                return (f"{v:.1f}" if v < 10 else f"{v:.0f}") + u
        return f"{x:.0f}" if x >= 10 or x == 0 else f"{x:.1f}"
    # model bytes of the gateway's small and cloud model (results/extra/edge_cost.csv, pair A), and the
    # escalation message count including the downlink requests (one per cloud-only node and escalation)
    ec = pd.read_csv(ROOT / "results" / "extra" / "edge_cost.csv").set_index(["dataset", "pair"])
    # final trim (length): the cost panel moves from the main table to the supplement (sup_pp_arch_cost.tex)
    C = ["\\begin{tabular}{@{}l r rrr rrrr rr@{}}", "\\toprule",
         "& Edge only & \\multicolumn{3}{c}{Escalation 10\\%, pull} & \\multicolumn{4}{c}{Gateway, node features}"
         " & \\multicolumn{2}{c}{Full cloud} \\\\",
         "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\\cmidrule(l){10-11}",
         "Dataset & Ops/win. & Ops/win. & Backh.\\ B/h & Msg./h$^{r}$ & Ops/win. & Local B/h & Small B & Cloud B"
         " & Backh.\\ B/h & Msg./h \\\\",
         "\\midrule"]
    for ds in DS:
        g = c[c.dataset == ds].set_index("architecture")
        e, x, gw, fc = (g.loc[k] for k in ("edge only", "escalation 10%, pull, on-demand",
                                            "gateway, features at sensor nodes", "full cloud"))
        req = float(x.cloud_only_nodes) * 0.10 * float(x.windows_per_hour)     # downlink requests per hour
        m = ec.loc[(ds, "A")]
        C.append(f"{DSNAME[ds]} & {si(e.ops_per_window)} & {si(x.ops_per_window)} & {si(x.uplink_B_per_h)} & "
                 f"{si(x.messages_per_h + req)} & {si(gw.ops_per_window)} & {si(gw.local_B_per_h)} & "
                 f"{si(m.gateway_param_bytes_deploy)} & {si(m.cloud_param_bytes_deploy)} & "
                 f"{si(fc.uplink_B_per_h)} & {si(fc.messages_per_h)} \\\\")
    C += ["\\bottomrule", "\\end{tabular}"]
    write("pp_arch.tex", "\n".join(L) + "\n")
    write("sup_pp_arch_cost.tex", "\n".join(C) + "\n")


# ------------------------------------------------------------------ supplement tables
def sup_shortcuts():
    sc = pd.read_csv(RS / "shortcuts" / "shortcuts.csv")
    lab = {"a_cycle_1nn": "Cycle index, 1-NN", "b_cycle_tree3": "Cycle index, depth-3 tree",
           "c_settings_tree3": "Other settings, depth-3 tree",
           "a_t_start_tree3": "Time in file, depth-3 tree", "b_order_tree3": "Window order, depth-3 tree",
           "c_session_logreg": "Session, logistic", "a_edge2_means_logreg": "Two edge means, logistic",
           "b_cloud8_means_logreg": "Eight means, logistic", "a_well_rate": "Well fault rate",
           "b_well_onehot_logreg": "Well one-hot, logistic"}
    L = ["\\begin{tabular}{@{}l l l ll@{}}", "\\toprule",
         "Dataset & Shortcut predictor & BA [95\\% CI] & Edge, pair A & Cloud, pair A \\\\", "\\midrule"]
    for _, r in sc.iterrows():
        dsn = {"threew__by_well": "3W by well"}.get(r.dataset, DSNAME.get(r.dataset, r.dataset))
        L.append(f"{dsn} & {lab[r.shortcut]} & {ci(r.ba_mean, r.ci_lo, r.ci_hi)} & "
                 f"{ci(r.edge_A, r.edge_A_lo, r.edge_A_hi)} & {ci(r.cloud_A, r.cloud_A_lo, r.cloud_A_hi)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_shortcuts.tex", "\n".join(L) + "\n")


def sup_variants():
    qs = [("ba_ef_ev", "$B_{eE}$"), ("ba_cf_ev", "$B_{cE}$"), ("ba_ef_cv", "$B_{eC}$"), ("ba_cf_cv", "$B_{cC}$"),
          ("shapley_model", "$\\bar\\Delta_{\\mathrm{mod}}$"), ("shapley_info", "$\\bar\\Delta_{\\mathrm{inf}}$"),
          ("interaction", "$\\Delta_{\\times}$"), ("comb_step_dep", "$\\Delta_{\\mathrm{comb}}$"),
          ("unresolved_dep", "$\\Delta_{\\mathrm{unr}}$"), ("ba_oracle_dep", "Oracle")]
    rq = [("C", r, f"$C$({r})") for r in ("R0", "R1", "R2", "R6", "R9")] + \
         [("capture", r, f"Cap({r})") for r in ("R1", "R6")]
    for name, cols in (("sup_pp_variants.tex", [(q, "", l) for q, l in qs]), ("sup_pp_variants_r.tex", rq)):
        L = ["\\begin{tabular}{@{}l c r " + "r" * len(cols) + "@{}}", "\\toprule",
             "Variant & Pair & $J$ & " + " & ".join(c[2] for c in cols) + " \\\\", "\\midrule"]
        for ds, conds in CONTROLS:
            for lab, v, pairs in conds:
                if v is None or v == "threew__by_well":
                    continue
                for pair in pairs:
                    if not pv_complete(v, pair):
                        L.append(f"{tex(v)} & {pair} & {pv_J(v, pair)} & \\multicolumn{{{len(cols)}}}{{l}}"
                                 f"{{{pend(tex(v) + ' ' + pair)}}} \\\\")
                        continue
                    cells = []
                    for q, r, _ in cols:
                        x = pv_row(v, pair, q, r)
                        cells.append(num(x["mean"]) if x is not None else "--")
                    L.append(f"{tex(v)} & {pair} & {pv_J(v, pair)} & " + " & ".join(cells) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        write(name, "\n".join(L) + "\n")


def sup_audit():
    p1 = pd.read_csv(RS / "routers" / "p1_ba.csv")
    cs = pd.read_csv(RS / "routers" / "p1_C_summary.csv").set_index("subset")
    out = {"out_capture": "Routing beats random", "out_edge": "Escalation beats the edge"}
    subsets = ["pooled", "q=1 (A+B)", "dataset=hydraulic", "dataset=skab", "dataset=seu", "dataset=threew"]
    sl = {"pooled": "All 400 cells", "q=1 (A+B)": "$q=1$, pairs A and B", "dataset=hydraulic": "Hydraulic",
          "dataset=skab": "SKAB", "dataset=seu": "SEU", "dataset=threew": "3W"}
    L = ["\\begin{tabular}{@{}l l rrrr rrrr@{}}", "\\toprule",
         "& & \\multicolumn{4}{c}{Pre-specified audit} & \\multicolumn{4}{c}{Corrected audit} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(l){7-10}",
         "Outcome & Cells & GO & BA & $p_{\\mathrm{ds}}$ & $p_{\\mathrm{blk}}$ & GO & BA & $p_{\\mathrm{ds}}$ & $p_{\\mathrm{blk}}$ \\\\",
         "\\midrule"]
    for oc, ol in out.items():
        L.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{ol}}}}} \\\\")
        for s in subsets:
            o = p1[(p1.audit == "orig") & (p1.outcome == oc) & (p1.subset == s)].iloc[0]
            c = p1[(p1.audit == "cor") & (p1.outcome == oc) & (p1.subset == s)].iloc[0]
            L.append(f"{sl[s]} & {int(o.n)} & {int(o.n_go)} & {num(o.ba)} & {num(o.p_perm_within_dataset)} & "
                     f"{num(o.p_perm)} & {int(c.n_go)} & {num(c.ba)} & {num(c.p_perm_within_dataset)} & {num(c.p_perm)} \\\\")
    L.append("\\addlinespace")
    L.append(f"\\multicolumn{{10}}{{@{{}}l}}{{Mean validation $C$: pre-specified pooled out-of-fold "
             f"{num(cs.loc['pooled', 'C_orig_pooled_oof'])}, corrected fold average {num(cs.loc['pooled', 'C_cor'])} "
             f"(permutation null {num(cs.loc['pooled', 'null_mean'])}).}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_audit.tex", "\n".join(L) + "\n")


def sup_fixedtier():
    p3 = pd.read_csv(RS / "routers" / "p3_slopes.csv")
    L = ["\\begin{tabular}{@{}l lll@{}}", "\\toprule",
         "Dataset & Fixed tiers (post-plan) & Joint retraining, $x$ = training positives & one-sided $p$, fixed tiers \\\\",
         "\\midrule"]
    for d, lab in [("hydraulic", "Hydraulic"), ("skab", "SKAB"), ("seu", "SEU"), ("threew", "3W"),
                   ("POOLED (HKSJ)", "Pooled (HKSJ)")]:
        f = p3[(p3.dataset == d) & p3.design.str.startswith("fixed tiers")].iloc[0]
        j = p3[(p3.dataset == d) & p3.design.str.contains("n_pos_train")].iloc[0]
        df = (3 if d.startswith("POOLED") else int(f.J) - 1)
        p = stats.t.sf(f.two_stage_mean / f.se, df)
        L.append(f"{lab} & {ci(f.two_stage_mean, f.ci_lo, f.ci_hi)} & {ci(j.two_stage_mean, j.ci_lo, j.ci_hi)} & {num(p)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    # per-level C(R6) - C(R1) on the hydraulic rig with the tiers fixed (cited in the main text).
    lv = pd.read_csv(RS / "routers" / "p3_by_level.csv").drop_duplicates()
    lv = lv[lv.dataset == "hydraulic"].sort_values("q")
    L += ["\\par\\medskip", "\\begin{tabular}{@{}l rrr l@{}}", "\\toprule",
          "Hydraulic, fixed tiers & $q$ & $C$(R6) & $C$(R1) & $C$(R6) $-$ $C$(R1) [95\\% CI] \\\\", "\\midrule"]
    for k, r in enumerate(lv.itertuples()):
        L.append(f"{'Pair A' if k == 0 else ''} & {r.q:g} & {num(r.C_R6)} & {num(r.C_R1)} & {ci(r.dC, r.dC_lo, r.dC_hi)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_fixedtier.tex", "\n".join(L) + "\n")


def sup_stream():
    s = pd.read_csv(RS / "routers" / "p2_stream_summary.csv")
    L = ["\\begin{tabular}{@{}l c l r r r r l@{}}", "\\toprule",
         "Dataset & Pair & Router & $J$ & Mean & SD & Range & BA, causal minus look-ahead \\\\", "\\midrule"]
    for _, r in s[np.isclose(s.f, 0.10)].iterrows():
        if int(r.J) < 20:
            L.append(f"{DSNAME[r.dataset]} & {r.pair} & {r.router} & {int(r.J)} & \\multicolumn{{4}}{{l}}"
                     f"{{{pend('routers/p2\\_stream\\_summary.csv, ' + str(int(r.J)) + ' of 20 splits')}}} \\\\")
            continue
        L.append(f"{DSNAME[r.dataset]} & {r.pair} & {r.router} & {int(r.J)} & {num(r.frac_mean)} & {num(r.frac_sd)} & "
                 f"{num(r.frac_min)}--{num(r.frac_max)} & {ci(r.diff_mean, r.diff_ci_lo, r.diff_ci_hi)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_stream.tex", "\n".join(L) + "\n")


def sup_comb():
    c = pd.read_csv(RS / "routers" / "p2_comb_summary.csv")
    L = ["\\begin{tabular}{@{}l c r rrrr l l@{}}", "\\toprule",
         "Dataset & Pair & $J$ & Edge & Cloud & Stacker & Combiner & Combiner minus stacker & Escalated share \\\\",
         "\\midrule"]
    for _, r in c.iterrows():
        if int(r.J) < 20:
            L.append(f"{DSNAME[r.dataset]} & {r.pair} & {int(r.J)} & \\multicolumn{{6}}{{l}}"
                     f"{{{pend('routers/p2\\_comb\\_summary.csv, ' + str(int(r.J)) + ' of 20 splits')}}} \\\\")
            continue
        L.append(f"{DSNAME[r.dataset]} & {r.pair} & {int(r.J)} & {num(r.ba_edge)} & {num(r.ba_cloud)} & "
                 f"{num(r.ba_stacker)} & {num(r.ba_comb)} & {ci(r.comb_minus_stacker, r.cms_lo, r.cms_hi)} & "
                 f"{num(r.esc_frac)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_comb.tex", "\n".join(L) + "\n")


def sup_sim():
    """The pilot table. The full simulation (J = 20, >= 500 replicates per scenario) is reported separately
    (sup_simfull_*). This table is the PILOT of the corrected designs at 10 splits (results_sens/simulation/),
    reported only as a pilot with its replicate count and Monte Carlo SEs once it has
    PILOT_TARGET_REPS replicates per scenario; until then a PENDING marker. It validates no proposed design."""
    rates = pd.read_csv(RS / "simulation" / "rates.csv")
    # replicates per scenario = the largest n_rep of that scenario (conditional rates, e.g. 'declared | feasible',
    # are computed on the subset of replicates where the condition holds and have smaller n_rep by design)
    per_scen = rates.groupby("scenario").n_rep.max()
    nmin, nmax = int(per_scen.min()), int(per_scen.max())
    if nmin < PILOT_TARGET_REPS:
        body = ("\\begin{tabular}{@{}p{0.95\\textwidth}@{}}\\toprule\n"
                + pend(f"pilot of the corrected designs at 10 splits, rates with Monte Carlo SEs; "
                       f"results\\_sens/simulation/rates.csv holds {nmin}--{nmax} of the "
                       f"{PILOT_TARGET_REPS} replicates per scenario (pilot still running)")
                + " \\\\\n\\bottomrule\n\\end{tabular}\n")
        write("sup_pp_sim.tex", body)
        return ["simulation pilot"]
    sel = [("H2", "original (joint retraining, MixedLM)", "SUPPORTED rate", "H2 plan, SUPPORTED"),
           ("H2", "corrected (fixed tiers, C(R6) - C(R1), two-stage)", "one-sided p < 0.05", "H2 fixed tiers, $p_1<0.05$"),
           ("H3 equivalence", "pooled TOST +-0.02 (true dC = 0)", "equivalence declared", "H3 TOST, declared at $dC=0$"),
           ("H3 equivalence", "original feasibility rule (asymptotic floor)", "predicted feasible", "H3 plan rule, feasible"),
           ("H3 equivalence", "corrected feasibility rule (realised J, HKSJ)", "predicted feasible", "H3 corrected rule, feasible"),
           ("H4", "original (test rho, R6 - R1, MixedLM)", "SUPPORTED rate", "H4 plan, SUPPORTED"),
           ("H4", "original with R0 in place of R6", "SUPPORTED rate", "H4 plan with R0, SUPPORTED"),
           ("H4", "corrected (val rho, R6 - R0, two-stage HKSJ)", "one-sided p < 0.05", "H4 corrected, $p_1<0.05$"),
           ("H5 audit", "original (GroupKFold, pooled OOF C, bootstrap LB > 0.52)", "go_rate", "H5 plan, GO rate"),
           ("H5 audit", "original (GroupKFold, pooled OOF C, bootstrap LB > 0.52)", "ba vs (iii) Cap(R6) > 0.10",
            "H5 plan, BA of calls"),
           ("H5 audit", "corrected (SGKF, fold-mean C > pooled within-class null 95th pct)", "go_rate",
            "H5 corrected, GO rate"),
           ("H5 audit", "corrected (SGKF, fold-mean C > pooled within-class null 95th pct)", "ba vs (iii) Cap(R6) > 0.10",
            "H5 corrected, BA of calls")]
    scen = [c for c in ["S0", "S1", "S1b", "S2", "S3"] if c in set(rates.scenario)]
    L = ["\\begin{tabular}{@{}l " + "l" * len(scen) + "@{}}", "\\toprule",
         "Design and rate & " + " & ".join(scen) + " \\\\", "\\midrule",
         "Replicates & " + " & ".join(str(int(rates[rates.scenario == c].n_rep.max())) for c in scen) + " \\\\"]
    for test, design, qty, lab in sel:
        cells = []
        for c in scen:
            g = rates[(rates.test == test) & (rates.design == design) & (rates.quantity == qty) & (rates.scenario == c)]
            cells.append("--" if g.empty else f"{num(g.rate.iloc[0], 2)} ({num(g.mcse.iloc[0], 2)})")
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_sim.tex", "\n".join(L) + "\n")
    return []



# ------------------------------------------------------------------ full validation of the proposed designs
SIMF = RS / "simulation_full" / "summary"


def _simf():
    r = pd.read_csv(SIMF / "rates.csv")
    return {(k, sc): (rt, se, ub) for k, sc, rt, se, ub in r[["key", "scenario", "rate", "mcse", "upper95_if_zero"]]
            .itertuples(index=False)}


def _cell(R, key, sc, d=3, se=True):
    v = R.get((key, sc))
    if v is None or not np.isfinite(v[0]):
        return "--"
    rt, m, ub = v
    if se and np.isfinite(m) and m > 0:
        return f"{num(rt, d)} ({num(m, d)})"
    if se and rt == 0 and np.isfinite(ub):
        return f"0 ($\\le${num(ub, d)})"
    return num(rt, d)


def sup_simfull_base():
    """Base family of the full validation, J = 20, R fixed and hashed
    (src/sens/simulation_full/, results_sens/simulation_full/summary/rates.csv). Rates with Monte Carlo SEs; for a
    zero count the exact one-sided 95% upper bound."""
    R = _simf()
    n = pd.read_csv(SIMF / "n_rep.csv").set_index("scenario").n_rep
    scen = ["S0", "S1", "S1b", "S2", "S3"]
    rows = [("H1, per-dataset NB 95\\% CI, coverage of the true step", "h1_ds_cover"),
            ("H1, naive $t$ 95\\% CI, coverage", "h1_ds_cover_naive"),
            ("H1, plan verdict, SUPPORTED", "h1_verdict_supported"),
            ("H2, plan (joint retraining, mixed model), SUPPORTED", "h2_orig_supported"),
            ("H2, fixed tiers, two-stage, $p_1<0.05$", "h2_cor_reject"),
            ("H3, pooled TOST, declared at true $dC=0$", "h3_pool_equiv"),
            ("H3, pooled TOST, declared at true $dC=0.02$", "h3_margin_pool_equiv"),
            ("H3, plan feasibility rule, feasible", "h3_feas_orig"),
            ("H3, corrected feasibility rule, feasible", "h3_feas_cor"),
            ("H4, plan, SUPPORTED", "h4_orig_supported"),
            ("H4, plan with R0 in place of R6, SUPPORTED", "h4_origR0_supported"),
            ("H4 corrected, mixed model, $p_1<0.05$", "h4_cor_mixed_p_lt05"),
            ("H4 corrected, two-stage, with the mean factor, $p_1<0.05$", "h4_2s_factor_reject"),
            ("H4 corrected, two-stage, no factor, $p_1<0.05$", "h4_2s_nofactor_reject"),
            ("H4 corrected, slopes per repeat, $p_1<0.05$", "h4_2s_repeat_naive_reject"),
            ("H4 corrected, slopes per repeat with NB factor, $p_1<0.05$", "h4_2s_repeat_nb_reject"),
            ("H5, plan audit, GO rate (cells)", "go_orig|go_rate|out_C"),
            ("H5, plan audit, BA of calls vs $\\Capt(\\mathrm{R6})>0.10$", "go_orig|ba|out_cap"),
            ("H5, within-class null per dataset, GO rate (cells)", "go_ds|go_rate|out_C"),
            ("H5, within-class null per dataset, BA of calls vs $\\Capt(\\mathrm{R6})>0.10$", "go_ds|ba|out_cap")]
    L = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.7cm} " + "l" * len(scen) + "@{}}", "\\toprule",
         "Design and rate & " + " & ".join(scen) + " \\\\", "\\midrule",
         "Replicates & " + " & ".join(str(int(n[c])) for c in scen) + " \\\\"]
    for lab, key in rows:
        L.append(f"{lab} & " + " & ".join(_cell(R, key, c) for c in scen) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_simfull_base.tex", "\n".join(L) + "\n")


def sup_simfull_tc():
    """Temporal-dependence (T) and confounded (C) families of the full validation."""
    R = _simf()
    tr = pd.read_csv(SIMF / "truth.csv")
    T = {(q, s): m for q, s, m in tr[["quantity", "scenario", "mean"]].itertuples(index=False)}
    n = pd.read_csv(SIMF / "n_rep.csv").set_index("scenario").n_rep
    tlab = {"T0_0.0": "None, $\\phi=0$", "T0_0.3": "None, $\\phi=0.3$", "T0_0.6": "None, $\\phi=0.6$",
            "T0_0.9": "None, $\\phi=0.9$", "T0s_0.9": "None, $\\phi=0.9$, $\\tau=1$", "T1_0.0": "0.08, $\\phi=0$",
            "T1_0.3": "0.08, $\\phi=0.3$", "T1_0.6": "0.08, $\\phi=0.6$", "T1_0.9": "0.08, $\\phi=0.9$"}
    L = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.6cm} r *{7}{>{\\raggedleft\\arraybackslash}p{1.1cm}}@{}}",
         "\\toprule",
         "\\multicolumn{9}{@{}l}{(a) Temporal dependence: hydraulic labels and 20 blocks, AR(1) block effects} \\\\",
         "\\midrule",
         "Information, $\\phi$ & $R$ & Step, new series & Step, CV & NB coverage & Naive coverage & Rejected "
         "& Positive splits & Survives \\\\", "\\midrule"]
    for sc, lab in tlab.items():
        L.append(f"{lab} & {int(n[sc])} & {num(T.get(('info_step', sc), np.nan))} & "
                 f"{num(T.get(('cv_expectation_info_step', sc), np.nan))} & "
                 + " & ".join(_cell(R, k, sc, 3, se=False) for k in
                              ("h1_ds_cover_truth", "h1_ds_cover_naive_truth", "h1_ds_reject", "h1_share_pos", "survive"))
                 + " \\\\")
    L += ["\\midrule",
          "\\multicolumn{9}{@{}l}{(b) Confounded files: cloud-only clock or offset without fault information} \\\\",
          "\\midrule",
          "Scenario & $R$ & Step, plain & H1 rule, plain & Beats shortcut & H1 rule, z & H1 rule, aug "
          "& (iii) met & Survives \\\\", "\\midrule"]
    clab = {"C_ref": "Real step, no confound", "C_clock": "Clock, no information", "C_offset": "Offset, no information"}
    for sc, lab in clab.items():
        L.append(f"{lab} & {int(n[sc])} & {num(T.get(('cv_expectation_info_step', sc), np.nan))} & "
                 + " & ".join(_cell(R, k, sc, 3, se=False) for k in
                              ("h1_verdict_supported", "sc_beats_plain", "h1_z_verdict_supported",
                               "h1_aug_verdict_supported", "sc_aug_beats", "survive")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_simfull_tc.tex", "\n".join(L) + "\n")

# ------------------------------------------------------------------ supplement tables (evaluation only)
EXTRA = ROOT / "results" / "extra"


def sup_holdout():
    """Per-fold diagnostics of the level and session hold-outs (src/extra/holdout_diagnostics.py).
    Upper panel, the redesigned hold-outs (fit + validation = every other level or session);
    middle panel, the plan's design (one fit level or session), kept with its caveat; lower panel,
    SKAB recall and specificity by disturbance family."""
    d = pd.read_csv(EXTRA / "holdout_diagnostics.csv")
    sc = pd.read_csv(RS / "shortcuts_controls" / "summary.csv")
    vl = {"skab_lso": "SKAB, session held out", "hyd_cooler": "Hydraulic, cooler level held out",
          "hyd_accum": "Hydraulic, accumulator level held out",
          "skab_lso2": "SKAB, session and disturbance family held out",
          "hyd_cooler2": "Hydraulic, cooler level held out", "hyd_accum2": "Hydraulic, accumulator level held out",
          "skab_lso2_base": "SKAB, session held out, causal-offset features"}
    pre = {"hyd_cooler2": "cooler ", "hyd_accum2": "acc. "}

    def short(v, h):
        h = str(h)
        return pre.get(v, "") + h.replace("accumulator", "acc. ").replace("cooler", "cooler ").replace("2020-", "")

    def sc_folds(v):
        g = sc[sc.variant == v]
        if g.empty:
            return None
        best = g.loc[g.ba_mean.idxmax()]
        return [float(x) for x in str(best.per_fold).split(";")]
    L = ["\\begin{tabular}{@{}l rrrr rrrr rr rr rrr r@{}}", "\\toprule",
         "& & & & & \\multicolumn{4}{c}{BA} & \\multicolumn{2}{c}{Pred.\\ pos.} & \\multicolumn{2}{c}{AUC} & & & & \\\\",
         "\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}",
         "Held out & $n_{\\mathrm{fit}}$ & $n_{\\mathrm{val}}$ & $n_{\\mathrm{test}}$ & $G^{+}_{\\mathrm{fit}}$ & "
         "$B_{eE}$ & $B_{cE}$ & $B_{eC}$ & $B_{cC}$ & Edge & Cloud & Edge & Cloud & $C$(R1) & $C$(R6) & "
         "$\\Delta_{\\mathrm{inf}}$ & Short.$^{a}$ \\\\", "\\midrule"]
    red = d[d.design != "plan"]
    order = ["skab_lso2", "skab_lso2_base", "hyd_cooler2", "hyd_accum2"]   # the two SKAB designs side by side
    red = red.assign(_o=red.variant.map({v: k for k, v in enumerate(order)})).sort_values(["_o", "pair", "split"],
                                                                                         kind="stable")
    for (v, pair), g in red.groupby(["variant", "pair"], sort=False):
        L.append(f"\\multicolumn{{17}}{{@{{}}l}}{{\\textit{{Redesign: {vl[v]}, pair {pair}}}}} \\\\")
        scf = sc_folds(v)
        for _, r in g.iterrows():
            scv = num(scf[int(r.split)]) if (scf is not None and pair == "A" and int(r.split) < len(scf)) else ""
            L.append(f"{short(v, r.held_out)} & {int(r.n_fit)} & {int(r.n_val)} & {int(r.n_test)} & "
                     f"{int(r.n_pos_groups_fit)} & {num(r.ba_edge)} & {num(r.ba_upgraded_edge)} & {num(r.ba_gateway)} & "
                     f"{num(r.ba_cloud)} & {num(r.ppr_edge)} & {num(r.ppr_cloud)} & {num(r.auc_edge)} & "
                     f"{num(r.auc_cloud)} & {num(r.C_R1)} & {num(r.C_R6)} & {num(r.shapley_info)} & {scv} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_holdout.tex", "\n".join(L) + "\n")          # redesign (one float)
    L = ["\\begin{tabular}{@{}l rrrr rr rr rr r@{}}", "\\toprule",
          "& & & & & \\multicolumn{2}{c}{BA} & \\multicolumn{2}{c}{Pred.\\ pos.\\ rate} & \\multicolumn{2}{c}{AUC} & \\\\",
          "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
          "Held out & $n_{\\mathrm{fit}}$ & $n_{\\mathrm{val}}$ & $n_{\\mathrm{test}}$ & $G^{+}_{\\mathrm{fit}}$ & "
          "Edge & Cloud & Edge & Cloud & Edge & Cloud & $C$(R1) \\\\", "\\midrule"]
    for (v, pair), g in d[d.design == "plan"].groupby(["variant", "pair"], sort=False):
        L.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{Plan's design: {vl[v]}, pair {pair}}}}} \\\\")
        for _, r in g.iterrows():
            L.append(f"{short(v, r.held_out)} & {int(r.n_fit)} & {int(r.n_val)} & "
                     f"{int(r.n_test)} & {int(r.n_pos_groups_fit)} & {num(r.ba_edge)} & {num(r.ba_cloud)} & {num(r.ppr_edge)} & "
                     f"{num(r.ppr_cloud)} & {num(r.auc_edge)} & {num(r.auc_cloud)} & {num(r.C_R1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_holdout_b.tex", "\n".join(L) + "\n")       # plan's design (second float)
    # the by-family panel is its own float (sup_pp_holdout_fam.tex), and carries the recall and
    # specificity of the time-in-file tree and of the fixed 560-990 s schedule rule on the same windows
    # (results/extra/r6_skab_schedule_rule_by_family.csv, src/extra/r6_skab_offschedule.py; the tree's predictions
    # are reconstructed and checked against its stored test BA). The raw-feature redesign has the same windows, folds
    # and start times as the offset redesign, so the tree's values are the same; the plan's design fits on one session.
    rf = pd.read_csv(EXTRA / "r6_skab_schedule_rule_by_family.csv")
    t = pd.read_csv(EXTRA / "holdout_skab_by_type.csv")
    fam = {"cavitation": "cavitation", "normal": "anomaly-free recording", "rotor_imbalance": "rotor imbalance",
           "warm_water_supply": "warm water supply", "water_increase": "water increase",
           "valve1_inlet_closure": "valve 1 (inlet) closure", "valve2_outlet_closure": "valve 2 (outlet) closure",
           "fluid_leak_or_addition": "fluid leak or addition"}
    L = ["\\begin{tabular}{@{}l l l rr rrr rrr@{}}", "\\toprule",
         "& & & & & \\multicolumn{3}{c}{Recall} & \\multicolumn{3}{c}{Specificity} \\\\",
         "\\cmidrule(lr){6-8}\\cmidrule(l){9-11}",
         "Design, pair & Session & Disturbance family & Faults & Normals & Edge & Cloud & Time$^{a}$ & Edge & Cloud & Time$^{a}$ \\\\",
         "\\midrule"]
    sel = [("redesign (R1)", "A", "Redesign, A"), ("redesign (R1)", "B", "Redesign, B"),
           ("redesign, causal offset (R1 r4)", "A", "Offset, A"), ("redesign, causal offset (R1 r4)", "B", "Offset, B"),
           ("plan", "B", "Plan, B")]
    for des, pair, lab in sel:
        g = t[(t.design == des) & (t.pair == pair)]
        for k, (_, r) in enumerate(g.iterrows()):
            tr, ts = "--", "--"
            if des != "plan":
                x = rf[(rf.pair == "A") & (rf.held_out == r.held_out) & (rf.family == r.family)]
                assert len(x) == 1 and int(x.n_pos.iloc[0]) == int(r.n_pos) and int(x.n_neg.iloc[0]) == int(r.n_neg)
                tr = num(x.recall_tree.iloc[0]) if r.n_pos > 0 else "--"
                ts = num(x.spec_tree.iloc[0])
            L.append(f"{lab if k == 0 else ''} & {r.held_out.replace('2020-', '')} & {fam.get(r.family, r.family)} & "
                     f"{int(r.n_pos)} & {int(r.n_neg)} & {num(r.recall_edge)} & {num(r.recall_cloud)} & {tr} & "
                     f"{num(r.spec_edge)} & {num(r.spec_cloud)} & {ts} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_holdout_fam.tex", "\n".join(L) + "\n")     # SKAB by family (third float)


def sup_offsched():
    """The four SKAB experiment files whose injection does not follow the 560-990 s schedule
    (results/extra/r6_skab_offschedule.csv, src/extra/r6_skab_offschedule.py; post-plan, descriptive). Recall,
    specificity and the first alarm (median over the file's test appearances) of the edge, the cloud and the
    time-in-file tree, for the pre-specified run, the causal-offset run and the offset session hold-out."""
    o = pd.read_csv(EXTRA / "r6_skab_offschedule.csv")
    runs = {"pre-specified": "Pre-specified", "causal offset": "Causal offset", "offset session hold-out": "Offset, session out"}
    fa = lambda x: "--" if not np.isfinite(x) else f"{x:.0f}"
    L = ["\\begin{tabular}{@{}l l r rrr rrr rrr@{}}", "\\toprule",
         "& & & \\multicolumn{3}{c}{Edge} & \\multicolumn{3}{c}{Cloud} & \\multicolumn{3}{c}{Time in file$^{a}$} \\\\",
         "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(l){10-12}",
         "Run, pair & File & Onset (s) & Rec. & Spec. & First & Rec. & Spec. & First & Rec. & Spec. & First \\\\", "\\midrule"]
    for run, lab in runs.items():
        for pair in "AB":
            g = o[(o.run == run) & (o.pair == pair)]
            for k, (_, r) in enumerate(g.iterrows()):
                L.append(f"{lab + ', ' + pair if k == 0 else ''} & {r.file.replace('.csv', '')}$^{{{'' if r.in_schedule_window else 'b'}}}$ & "
                         f"{int(r.onset_s)} & {num(r.recall_edge, 2)} & {num(r.spec_edge, 2)} & {fa(r.first_alarm_edge_median_s)} & "
                         f"{num(r.recall_cloud, 2)} & {num(r.spec_cloud, 2)} & {fa(r.first_alarm_cloud_median_s)} & "
                         f"{num(r.recall_time_tree, 2)} & {num(r.spec_time_tree, 2)} & {fa(r.first_alarm_time_tree_median_s)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_offsched.tex", "\n".join(L).replace("$^{}$", "") + "\n")


def sup_audit_wc():
    """The corrected audit with the within-class null, all cells and defined cells only."""
    a = pd.read_csv(EXTRA / "audit_within_class.csv")
    sub = {"all cells": "All levels", "q = 1, pairs A and B": "$q=1$"}
    cel = {"undefined as NO-GO": "undefined as NO-GO", "defined cells only": "defined only"}
    out = {"routing beats random": "beats random", "escalation beats the edge": "beats edge"}
    L = ["\\begin{tabular}{@{}l l l rrr rrr@{}}", "\\toprule",
         "Levels & Cells & Outcome & $n$ & Undef. & GO & BA & $p_{\\mathrm{ds}}$ & $p_{\\mathrm{blk}}$ \\\\", "\\midrule"]
    for null, g in a.groupby("null", sort=False):
        L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{null[0].upper() + null[1:]}}}}} \\\\")
        for _, r in g.iterrows():
            L.append(f"{sub[r.subset]} & {cel[r.cells]} & {out[r.outcome]} & {int(r.n)} & {int(r.n_undefined)} & "
                     f"{int(r.n_go)} & {num(r.ba)} & {num(r.p_within_dataset)} & {num(r.p_within_block)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_audit_wc.tex", "\n".join(L) + "\n")


def sup_routers_well():
    """Holm-adjusted router contrasts and the 3W well-stratified concordance."""
    h = pd.read_csv(EXTRA / "router_holm.csv")
    w = pd.read_csv(EXTRA / "router_by_well.csv")
    L = ["\\begin{tabular}{@{}l c l l rr@{}}", "\\toprule",
         "Contrast & Pair & Dataset & Mean [95\\% CI] & $p$ & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
    for con, lab in (("C(R6x) - C(R1)", "$C$(R6x) $-$ $C$(R1)"), ("C(HyR) - C(R1)", "$C$(rank avg.) $-$ $C$(R1)"),
                     ("C(R6x) - C(R6)", "$C$(R6x) $-$ $C$(R6)")):
        k = 0
        for pair in "AB":
            for d in DS:
                r = h[(h.contrast == con) & (h.pair == pair) & (h.dataset == d)].iloc[0]
                L.append((lab if k == 0 else "") + f" & {pair} & {DSNAME[d]} & {ci(r['mean'], r.ci_lo, r.ci_hi)} & "
                         f"{num(r.p)} & {num(r.p_holm_within_contrast)} \\\\")
                k += 1
    # R4x and R5x against R1, same correction and Holm rule (results/extra/r5_router_r4x_r5x.csv)
    r5 = pd.read_csv(EXTRA / "r5_router_r4x_r5x.csv")
    for con, lab in (("C(R4x) - C(R1)", "$C$(R4x) $-$ $C$(R1)"), ("C(R5x) - C(R1)", "$C$(R5x) $-$ $C$(R1)")):
        k = 0
        for pair in "AB":
            for d in DS:
                r = r5[(r5.contrast == con) & (r5.pair == pair) & (r5.dataset == d)].iloc[0]
                L.append((lab if k == 0 else "") + f" & {pair} & {DSNAME[d]} & {ci(r['mean'], r.ci_lo, r.ci_hi)} & "
                         f"{num(r.p)} & {num(r.p_holm_within_contrast)} \\\\")
                k += 1
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_routers_well.tex", "\n".join(L) + "\n")   # the 3W within-well panel is its own float
    L = ["\\begin{tabular}{@{}l ll ll@{}}", "\\toprule",
          "& \\multicolumn{2}{c}{Pair A} & \\multicolumn{2}{c}{Pair B} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
          "3W quantity & All pairs of windows & Within wells & All pairs of windows & Within wells \\\\", "\\midrule"]
    for r_, lab in (("R1", "$C$(R1)"), ("R6x", "$C$(R6x)"), ("HyR", "$C$(rank avg.)"), ("R4x", "$C$(R4x)")):
        cells = []
        for pair in "AB":
            for kind in ("pooled over wells", "within wells"):
                g = w[(w.pair == pair) & (w.quantity == f"C({r_}) {kind}")].iloc[0]
                cells.append(ci(g["mean"], g.ci_lo, g.ci_hi))
        L.append(lab + " & " + " & ".join(cells) + " \\\\")
    for r_, lab in (("R6x", "$C$(R6x) $-$ $C$(R1)"), ("HyR", "$C$(rank avg.) $-$ $C$(R1)"), ("R4x", "$C$(R4x) $-$ $C$(R1)")):
        cells = []
        for pair in "AB":
            for kind in ("pooled over wells", "within wells"):
                g = w[(w.pair == pair) & (w.quantity == f"C({r_}) - C(R1) {kind}")].iloc[0]
                cells.append(ci(g["mean"], g.ci_lo, g.ci_hi))
        L.append(lab + " & " + " & ".join(cells) + " \\\\")
    g = w[w.quantity == "share of oracle weight D within wells"].set_index("pair")
    L.append("Share of $D$ within wells & \\multicolumn{2}{l}{" + num(g.loc["A", "mean"]) + "} & \\multicolumn{2}{l}{"
             + num(g.loc["B", "mean"]) + "} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_routers_bywell.tex", "\n".join(L) + "\n")


def sup_unit():
    """Realised escalation per fold and per unit at a nominal 10% (causal streaming thresholds)."""
    u = pd.read_csv(EXTRA / "per_unit_escalation.csv")
    unit = {"hydraulic": "run", "skab": "file", "seu": "recording", "threew": "episode"}
    L = ["\\begin{tabular}{@{}l c l l r rrrr@{}}", "\\toprule",
         "& & & Per fold & Zero & \\multicolumn{4}{c}{Per unit (all splits)} \\\\", "\\cmidrule(l){6-9}",
         "Dataset (unit) & Pair & Router & min--mean--max & folds & median & 90\\% & 95\\% & max \\\\", "\\midrule"]
    for _, r in u.iterrows():
        L.append(f"{DSNAME[r.dataset]} ({unit[r.dataset]}) & {r.pair} & {r.router} & "
                 f"{num(r.fold_min)}--{num(r.fold_mean)}--{num(r.fold_max)} & {int(r.n_folds_zero)} & {num(r.unit_median)} & "
                 f"{num(r.unit_q90)} & {num(r.unit_q95)} & {num(r.unit_max)} \\\\")
    # a common time horizon (results/extra/r4_escalation_horizon.csv, src/extra/r4_systems.py)
    hz = pd.read_csv(EXTRA / "r4_escalation_horizon.csv")
    L.append("\\midrule")
    for _, r in hz.iterrows():
        lab = "Hydraulic (1 h)" if r.horizon.startswith("rolling") else "Hydraulic, blocks (111 cycles)"
        rt = "R1" if r.router == "R1r" else "R1$^{v}$"
        L.append(f"{lab} & {r.pair} & {rt} & -- & -- & {num(r['median'])} & {num(r.q90)} & {num(r.q95)} & {num(r['max'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    # mean per-unit rate of R1 by leakage and cooler level (results/extra/r5_escalation_by_setting.csv)
    es = pd.read_csv(EXTRA / "r5_escalation_by_setting.csv")
    L += ["\\par\\medskip", "\\begin{tabular}{@{}l rrr rrr@{}}", "\\toprule",
          "& \\multicolumn{3}{c}{Internal pump leakage} & \\multicolumn{3}{c}{Cooler level (\\%)} \\\\",
          "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
          "Hydraulic, R1 & None & Weak & Severe & 3 & 20 & 100 \\\\", "\\midrule"]
    for pair in "AB":
        g = es[es.pair == pair]
        v = [g[(g.setting == "pump") & (g.level == k)].mean_unit_rate.iloc[0] for k in (0, 1, 2)]
        v += [g[(g.setting == "cooler") & (g.level == k)].mean_unit_rate.iloc[0] for k in (3, 20, 100)]
        L.append(f"Pair {pair}, mean per-unit rate & " + " & ".join(num(x) for x in v) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_unit.tex", "\n".join(L) + "\n")


def sup_syscost():
    """Whole-system cost with one gateway model and full message counts."""
    c = pd.read_csv(EXTRA / "system_cost_r3.csv")

    def si(x, d=None):
        if x is None or not np.isfinite(x):
            return "--"
        for f, u_ in ((1e6, "\\,M"), (1e3, "\\,k")):
            if abs(x) >= f:
                v = x / f
                return (f"{v:.1f}" if v < 10 else f"{v:.0f}") + u_
        return f"{x:.0f}" if x >= 10 or x == 0 else f"{x:.1f}"
    L = ["\\begin{tabular}{@{}l rrrr rr@{}}", "\\toprule",
         "Architecture & Ops/window & Uplink B/h & Uplink bit/s & Msg./h & Local B/h & Local bit/s \\\\", "\\midrule"]
    for ds in DS:
        g = c[c.dataset == ds]
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{DSNAME[ds]}}} ({int(g.cloud_only_nodes.iloc[0])} cloud-only nodes, "
                 f"{si(g.windows_per_hour.iloc[0])} windows per hour)}} \\\\")
        for _, r in g.iterrows():
            lab = {"edge only": "Edge only", "escalation 10%, pull, on-demand": "Escalation 10\\%, pull",
                   "gateway, raw to gateway": "Gateway, raw to gateway",
                   "gateway, features at sensor nodes": "Gateway, features at nodes", "full cloud": "Full cloud"}[r.architecture]
            up = si(r.uplink_B_per_h)
            if r.architecture.startswith("escalation"):
                up += f" ({si(r.uplink_B_per_h_realised_min)}--{si(r.uplink_B_per_h_realised_max)})"
            L.append(f"{lab} & {si(r.ops_per_window)} & {up} & {si(r.uplink_bit_s)} & {si(r.messages_per_h)} & "
                     f"{si(r.local_B_per_h)} & {si(r.get('local_bit_s', np.nan))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_syscost.tex", "\n".join(L) + "\n")


def sup_capture():
    c = pd.read_csv(EXTRA / "capture_ratio_of_means.csv")
    L = ["\\begin{tabular}{@{}l c rrr rrr@{}}", "\\toprule",
         "& & \\multicolumn{3}{c}{R1} & \\multicolumn{3}{c}{R6} \\\\", "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
         "Dataset & Pair & Mean of ratios & Median & Ratio of means & Mean of ratios & Median & Ratio of means \\\\", "\\midrule"]
    for ds in DS:
        for pair in "AB":
            r1 = c[(c.dataset == ds) & (c.pair == pair) & (c.router == "R1")].iloc[0]
            r6 = c[(c.dataset == ds) & (c.pair == pair) & (c.router == "R6")].iloc[0]
            L.append(f"{DSNAME[ds]} & {pair} & {num(r1.mean_of_ratios)} & {num(r1.median_of_ratios)} & {num(r1.ratio_of_means)} & "
                     f"{num(r6.mean_of_ratios)} & {num(r6.median_of_ratios)} & {num(r6.ratio_of_means)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_capture.tex", "\n".join(L) + "\n")


def sup_timestrat():
    t = pd.read_csv(EXTRA / "skab_timestrat_variants.csv")
    vl = {"pre-specified": "Pre-specified features", "skab_filez": "Z-scored within file (non-causal)",
          "skab_base": "Offset from first 100 s (causal)", "skab_notemp": "No temperature channels",
          "skab_edgeP": "Pressure moved to edge", "skab_cm_flow": "Cloud without flow",
          "skab_cm_thermo": "Cloud without fluid temperature", "skab_cm_pressure": "Cloud without pressure",
          "pre-specified, splits 0-9": "Pre-specified features, splits 0--9",
          "skab_e_flow": "Cloud = drive + flow (splits 0--9)", "skab_e_press": "Cloud = drive + pressure (splits 0--9)",
          "skab_e_thermo": "Cloud = drive + fluid temp.\\ (splits 0--9)",
          "skab_sc": "Time in file in both views (splits 0--9)"}
    # the time-in-file tree itself under the same bin rule, on the same splits
    # (results/extra/skab_clock_timestrat.csv, src/extra/r4_skab_clock.py)
    clk = pd.read_csv(EXTRA / "skab_clock_timestrat.csv").set_index("split").clock_timestrat
    L = ["\\begin{tabular}{@{}l c r rrr l@{}}", "\\toprule",
         "SKAB features & Pair & $J$ & Clock & Edge & Cloud & Information step [95\\% CI] \\\\", "\\midrule"]
    for _, r in t.iterrows():
        assert bool(r.same_test_folds_as_main)
        L.append(f"{vl.get(r.variant, tex(r.variant))} & {r.pair} & {int(r.J)} & {num(clk.iloc[:int(r.J)].mean())} & "
                 f"{num(r.edge_timestrat)} & "
                 f"{num(r.cloud_timestrat)} & {ci(r.timestrat_info, r.ci_lo, r.ci_hi)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_timestrat.tex", "\n".join(L) + "\n")


def sup_round3():
    """The shortcut-augmented, single-channel and sensor-family runs, each beside the run it
    modifies on the same splits. Source: results/extra/round3_variants.csv (src/extra/round3_variants.py)."""
    r3 = pd.read_csv(EXTRA / "round3_variants.csv")
    groups = [
        ("Hydraulic, block-grouped splits 0--9, one sensor family added or removed", [
            ("hyd_blk_e_flow", "Edge + flow (FS1, FS2)"),
            # one flow sensor
            ("hyd_blk_e_fs1", "Edge + FS1 only$^{b}$"), ("hyd_blk_e_fs2", "Edge + FS2 only$^{b}$"),
            ("hyd_blk_e_press", "Edge + pressure (PS1--PS6)"),
            ("hyd_blk_e_temp", "Edge + temperature (TS1--TS4)"), ("hyd_blk_cm_flow", "All minus flow"),
            ("hyd_blk_cm_press", "All minus pressure"), ("hyd_blk_cm_temp", "All minus temperature")]),
        ("Hydraulic, block-grouped splits; shortcut added to both views", [
            ("hyd_blocks_sc", "Cycle index"), ("hyd_blocks_sc2", "Cycle index and settings")]),
        ("SKAB, pre-specified splits 0--9", [
            ("skab_e_flow", "Cloud = drive + flow"), ("skab_e_press", "Cloud = drive + pressure"),
            ("skab_e_thermo", "Cloud = drive + fluid temp."), ("skab_sc", "Time in file in both views")]),
    ]
    L = ["\\begin{tabular}{@{}l c r rr l r l r@{}}", "\\toprule",
         "Variant & Pair & $J$ & $B_{eE}$ & $B_{cC}$ & $\\bar\\Delta_{\\mathrm{inf}}$ [95\\% CI] & "
         "Ref. & Difference [95\\% CI] & $C$(R1) \\\\", "\\midrule"]
    pending = []
    for title, rows in groups:
        L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for v, lab in rows:
            for k, pair in enumerate("AB"):
                g = r3[(r3.variant == v) & (r3.pair == pair)]
                name = lab if k == 0 else ""
                if g.empty or not bool(g.complete.iloc[0]):
                    j = 0 if g.empty else int(g.J.iloc[0])
                    jx = 0 if g.empty else int(g.J_expected.iloc[0])
                    pending.append(f"{v}/{pair}")
                    L.append(f"{name} & {pair} & {j} & \\multicolumn{{6}}{{l}}{{" + pend(f"{tex(v)}/{pair}, {j} of {jx}")
                             + "} \\\\")
                    continue
                r = g.iloc[0]
                L.append(f"{name} & {pair} & {int(r.J)} & {num(r.edge)} & {num(r.cloud)} & "
                         f"{ci(r['info'], r.info_lo, r.info_hi)} & {num(r.ref_info)} & "
                         f"{ci(r.info_minus_ref, r.imr_lo, r.imr_hi)} & {num(r.C_R1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_round3.tex", "\n".join(L) + "\n")
    return pending


def sup_pernode():
    """Traffic per hop and per node, the always-on informative-channel designs and the
    provisioned escalation rate. Sources: results/extra/r4_per_node_cost.csv, r4_informative_stream.csv,
    r4_escalation_horizon.csv, r4_budget_band.csv (src/extra/r4_systems.py)."""
    pn = pd.read_csv(EXTRA / "r4_per_node_cost.csv")
    inf = pd.read_csv(EXTRA / "r4_informative_stream.csv")
    band = pd.read_csv(EXTRA / "r4_budget_band.csv")

    def si(x):
        if x is None or not np.isfinite(x):
            return "--"
        for f, u_ in ((1e6, "\\,M"), (1e3, "\\,k")):
            if abs(x) >= f:
                v = x / f
                return (f"{v:.1f}" if v < 10 else f"{v:.0f}") + u_
        return f"{x:.0f}" if x >= 10 or x == 0 else f"{x:.1f}"
    # a local-network column (aggregate, bit/s), the gateway on node-extracted features (co-located
    # with the edge node) and escalation provisioned for a 95th-percentile delay of 10 min (results/extra/r5_hop_loads.csv)
    hl = pd.read_csv(EXTRA / "r5_hop_loads.csv")
    L = ["\\begin{tabular}{@{}l rrr rr rr@{}}", "\\toprule",
         "& \\multicolumn{3}{c}{Per cloud-only node} & \\multicolumn{2}{c}{Edge node} & Local & Backhaul \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}\\cmidrule(l){8-8}",
         "Architecture & Up B/h & Up msg./h & Down msg./h & Up B/h & Down msg./h & bit/s & bit/s \\\\", "\\midrule"]
    lab = {"escalation 10%, pull": "Escalation 10\\%, pull", "full cloud stream": "Full cloud stream",
           "gateway, raw to gateway": "Gateway, raw to gateway$^{d}$"}
    for ds in DS:
        g = pn[pn.dataset == ds]
        n_co = int(g.cloud_only_nodes.iloc[0]); wph = g.windows_per_hour.iloc[0]
        L.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{DSNAME[ds]}}} ({n_co} cloud-only nodes, "
                 f"{si(wph)} windows per hour)}} \\\\")
        for _, r in g.iterrows():
            if r.architecture.startswith("gateway"):
                loc = n_co * r.node_up_B_h * 8 / 3600
            else:
                loc = r.backhaul_bit_s                       # pull flow and full stream cross the local network too
            L.append(f"{lab[r.architecture]} & {si(r.node_up_B_h)} & {si(r.node_up_msg_h)} & {si(r.node_down_msg_h)} & "
                     f"{si(r.edge_up_B_h)} & {si(r.edge_down_msg_h)} & {si(loc)} & {si(r.backhaul_bit_s)} \\\\")
            if ds == "hydraulic" and r.architecture.startswith("escalation"):
                for key, tag in (("escalation, provisioned at hourly q95", "a"),
                                 ("escalation, provisioned for q95 delay <= 10 min", "e")):
                    h5 = hl[(hl.pair == "A") & (hl.architecture == key)].iloc[0]
                    k = h5.escalation_fraction / 0.10
                    L.append(f"Escalation, provisioned at {num(h5.escalation_fraction, 2)}$^{{{tag}}}$ & "
                             f"{si(r.node_up_B_h * k)} & {si(r.node_up_msg_h * k)} & {si(r.node_down_msg_h * k)} & "
                             f"{si(r.edge_up_B_h * k)} & {si(r.edge_down_msg_h * k)} & {si(h5.local_bit_s)} & "
                             f"{si(h5.backhaul_bit_s)} \\\\")
            if r.architecture == "full cloud stream":
                gl = n_co * r.node_up_B_h * 8 / 3600
                if ds == "hydraulic":
                    ref = hl[(hl.pair == "A") & (hl.architecture == "gateway, node features, co-located with edge node")].iloc[0]
                    assert abs(ref.local_bit_s - gl) < 1e-9
                L.append(f"Gateway, features at nodes$^{{d}}$ & {si(r.node_up_B_h)} & {si(r.node_up_msg_h)} & 0 & 0 & 0 & "
                         f"{si(gl)} & 0 \\\\")
        # on the hydraulic rig the informative channel is FS1 alone (rows appended to
        # r4_informative_stream.csv by src/extra/r6_systems.py); SKAB keeps its flow meter
        chan = {"hydraulic": "FS1", "skab": "Volume Flow RateRMS"}
        h = inf[(inf.dataset == ds) & (inf.pair == "A") & (inf.channels == chan.get(ds, ""))]
        if len(h):
            h = h.iloc[0]
            per_node = 17 * 4 * wph
            fe = h.stream_inf_plus_edge_B_h - h.stream_inf_features_B_h
            L.append(f"Edge + informative features streamed$^{{b}}$ & {si(per_node)} & {si(wph)} & 0 & "
                     f"{si(fe)} & {si(wph)} & {si(h.stream_inf_plus_edge_bit_s)} & {si(h.stream_inf_plus_edge_bit_s)} \\\\")
            L.append(f"Informative channel wired to edge$^{{c}}$ & {si(per_node)} & {si(wph)} & 0 & 0 & 0 & "
                     f"{si(h.stream_inf_features_bit_s)} & 0 \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_pernode.tex", "\n".join(L) + "\n")


def sup_delay():
    """FIFO delay of escalated decisions against link rate, the smallest rate for a delay bound,
    the per-hop band against each always-on alternative, and dropping at a deadline (hydraulic, R1 with the causal
    threshold, f = 0.10). Sources: results/extra/r5_escalation_delay.csv, r5_delay_bound_rates.csv,
    r5_budget_band_per_hop.csv, r5_deadline_drop.csv (src/extra/r5_systems.py)."""
    dl = pd.read_csv(EXTRA / "r5_escalation_delay.csv")
    bd = pd.read_csv(EXTRA / "r5_delay_bound_rates.csv")
    bb = pd.read_csv(EXTRA / "r5_budget_band_per_hop.csv")
    dd = pd.read_csv(EXTRA / "r5_deadline_drop.csv")
    m1 = lambda x: f"{x / 60:.1f}"
    L = ["\\begin{tabular}{@{}l rrr rrr@{}}", "\\toprule",
         "& \\multicolumn{3}{c}{Pair A, delay (min)} & \\multicolumn{3}{c}{Pair B, delay (min)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
         "Link rate (bit/s) & median & 95\\% & max & median & 95\\% & max \\\\", "\\midrule"]
    for r in sorted(dl.rate_bit_s.unique()):
        if r < 29:
            continue
        a = dl[(dl.pair == "A") & np.isclose(dl.rate_bit_s, r)].iloc[0]
        b = dl[(dl.pair == "B") & np.isclose(dl.rate_bit_s, r)].iloc[0]
        L.append(f"{r:.0f} & {m1(a.median_s)} & {m1(a.q95_s)} & {m1(a.max_s)} & {m1(b.median_s)} & {m1(b.q95_s)} & "
                 f"{m1(b.max_s)} \\\\")
    L += ["\\midrule", "Delay bound & \\multicolumn{3}{l}{Smallest rate, 95\\% / max (A)} & "
          "\\multicolumn{3}{l}{Smallest rate, 95\\% / max (B)} \\\\", "\\midrule"]
    for b_ in sorted(bd.delay_bound_s.unique()):
        a = bd[(bd.pair == "A") & (bd.delay_bound_s == b_)].iloc[0]
        b = bd[(bd.pair == "B") & (bd.delay_bound_s == b_)].iloc[0]
        lab = "one window (1 min)" if b_ == 60 else f"{b_ // 60:.0f} min"
        L.append(f"{lab} & \\multicolumn{{3}}{{l}}{{{a.min_rate_q95_bit_s:.0f} / {a.min_rate_max_bit_s:.0f} bit/s}} & "
                 f"\\multicolumn{{3}}{{l}}{{{b.min_rate_q95_bit_s:.0f} / {b.min_rate_max_bit_s:.0f} bit/s}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_delay.tex", "\n".join(L) + "\n")          # delay and band are separate floats
    # band per hop, escalation provisioned for a 95th-percentile delay of 10 min, and at the hourly 95th percentile
    # the informative stream is edge + FS1 (the both-flow rows stay in the csv); FS1 wired to the edge
    # node; the cloud model at the gateway has the small-model gateway's traffic, so its band rows are the same
    alt = {"gateway, node features, co-located with edge node": "Gateway, either model$^{a}$",
           "gateway, node features, separate from edge node": "Gateway, separate box",
           "full feature stream": "Full feature stream", "edge and FS1 features streamed": "Edge and FS1 features",
           "FS1 wired to edge node (local only)": "FS1 to the edge node$^{c}$"}
    hop = {"backhaul": "backhaul", "local aggregate": "local", "per cloud-only node": "per node"}
    L = ["\\begin{tabular}{@{}l l ll ll@{}}", "\\toprule",
          "& & \\multicolumn{2}{c}{95\\% delay $\\le$ 10 min} & \\multicolumn{2}{c}{Hourly 95th percentile} \\\\",
          "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
          "Alternative & Hop & Pair A & Pair B & Pair A & Pair B \\\\", "\\midrule"]
    keys = ("escalation, provisioned for q95 delay <= 10 min", "escalation, provisioned at hourly q95")
    for a_, la in alt.items():
        for h_, lh in hop.items():
            g = bb[(bb.alternative == a_) & (bb.hop == h_)]
            if g.empty:
                continue
            cells = []
            for k in keys:
                for pair in "AB":
                    x = g[(g.escalation == k) & (g.pair == pair)].iloc[0]
                    cells.append("empty" if x.band == "empty" else x.band.replace(" bit/s", "").replace(" to ", "--"))
            L.append(f"{la if h_ == 'backhaul' else ''} & {lh} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule"]
    for r in (42.31, 64.0):
        cells = []
        for pair in "AB":
            x = dd[(dd.pair == pair) & np.isclose(dd.rate_bit_s, r) & (dd.deadline_s == 600)].iloc[0]
            cells.append(f"{x.share_dropped:.3f}, {x.ba_change:+.4f}".replace("+", "$+$").replace("-", "$-$"))
        L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Dropped beyond 10 min at {r:.0f} bit/s$^{{b}}$}} & {cells[0]} & {cells[1]} & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("sup_pp_band.tex", "\n".join(L) + "\n")


def main():
    pend_all = {}
    overview()
    pend_all["controls"] = controls()
    pend_all["routers"] = routers()
    pend_all["tests"] = tests()
    arch()
    sup_shortcuts()
    sup_variants()
    sup_audit()
    sup_fixedtier()
    sup_stream()
    sup_comb()
    sup_offsched()
    pend_all["sim"] = sup_sim()
    sup_simfull_base()
    sup_simfull_tc()
    pend_all["survival"] = survival()
    sup_holdout()
    sup_audit_wc()
    sup_routers_well()
    sup_unit()
    sup_syscost()
    sup_pernode()
    sup_delay()
    sup_capture()
    sup_timestrat()
    pend_all["round3"] = sup_round3()
    print("PENDING:", {k: v for k, v in pend_all.items() if v})


if __name__ == "__main__":
    main()
