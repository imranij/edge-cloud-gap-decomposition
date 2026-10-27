"""POST-PLAN: markdown tables for results_sens/routers/REPORT.md (reads the CSVs written by the summaries)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from util import OUT


def fmt(x, d=3):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{d}f}"


def md(df, cols, d=3, names=None):
    names = names or cols
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c], d) for c in cols) + " |")
    return "\n".join(out) + "\n"


def ci(m, lo, hi, d=3):
    return f"{fmt(m, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def p1():
    s = []
    cs = pd.read_csv(OUT / "p1_C_summary.csv")
    s.append("### Audit T1. Validation C of the original and corrected audits (means over cells)\n")
    s.append(md(cs, ["subset", "n", "n_C_orig_finite", "C_orig_pooled_oof", "C_orig_foldavg", "n_C_cor_finite", "C_cor",
                     "null_mean", "C_test_R6", "corr_Corig_Ctest", "corr_Ccor_Ctest"],
                names=["subset", "cells", "orig C defined", "orig C (pooled OOF, stored)", "orig C (within-fold)",
                       "corrected C defined", "corrected C (within-fold)", "null mean", "test C(R6)",
                       "r(orig C, test C)", "r(corr C, test C)"]))
    th = pd.read_csv(OUT / "p1_null_thresholds.csv")
    s.append("\n### Audit T2. Permutation-null GO thresholds per dataset x pair x level\n")
    s.append(md(th, ["block", "n_null", "null_mean", "null_sd", "thr95", "mean_C_cor", "mean_C_orig"],
                names=["block", "null draws", "null mean", "null sd", "GO threshold (95th pct)", "mean corrected C",
                       "mean original C"]))
    ba = pd.read_csv(OUT / "p1_ba.csv")
    lab = {"out_capture": "(i) routing beats random: capture(R6) > 0.10",
           "out_edge": "(ii) routing beats the edge alone: BA(R6 @ 10%) > BA(edge)",
           "out_C": "(iii) test C(R6) > 0.5"}
    for k, (oc, t) in enumerate(lab.items()):
        x = ba[ba.outcome == oc].copy()
        x["audit"] = x.audit.map({"orig": "original", "cor": "corrected"})
        s.append(f"\n### Audit T{3 + k}. GO calls against outcome {t}\n")
        s.append(md(x, ["subset", "audit", "n", "n_go", "n_outcome", "base_rate", "precision", "recall", "ba", "p_perm",
                        "p_perm_within_dataset"],
                    names=["subset", "audit", "cells", "GO", "outcome = 1", "base rate", "precision", "recall", "BA",
                           "p (perm. within ds x pair x level)", "p (perm. within dataset)"]))
    return "\n".join(s)


def p2():
    s = []
    r = pd.read_csv(OUT / "p2_routers.csv")
    order = ["R1", "R6", "R8", "R4x", "R5x", "R6x", "R10", "HyR", "HyS"]
    for pair in ("A", "B"):
        rows = []
        for ds in ("hydraulic", "skab", "seu", "threew"):
            for ro in order:
                g = r[(r.dataset == ds) & (r.pair == pair) & (r.router == ro)]
                if g.empty:
                    continue
                def get(q, b=None):
                    h = g[g.quantity == q]
                    if b is not None:
                        h = h[np.isclose(h.budget, b)]
                    return h.iloc[0] if len(h) else None
                c, cap = get("C"), get("capture")
                row = dict(dataset=ds, router=ro, J=int(c.J), C=ci(c["mean"], c.ci_lo, c.ci_hi),
                           capture=ci(cap["mean"], cap.ci_lo, cap.ci_hi))
                for b in (0.05, 0.10, 0.20, 0.30):
                    v = get("ba_at_budget", b)
                    row[f"BA@{b:.2f}"] = fmt(v["mean"])
                rows.append(row)
        df = pd.DataFrame(rows)
        s.append(f"\n### Routers T{'1' if pair == 'A' else '2'}. Pair {pair}: routers on the test folds (NB mean [95% CI]; BA at budget = mean)\n")
        s.append(md(df, list(df.columns)))
    d = pd.read_csv(OUT / "p2_diffs.csv")
    d["est"] = [ci(a, b, c) for a, b, c in zip(d["mean"], d.ci_lo, d.ci_hi)]
    s.append("\n### Routers T3. Differences in test C (per dataset: NB mean [95% CI]; pooled: HKSJ over datasets)\n")
    w = d.pivot_table(index=["contrast", "pair"], columns="dataset", values="est", aggfunc="first").reset_index()
    cols = ["contrast", "pair", "hydraulic", "skab", "seu", "threew", "POOLED (HKSJ)"]
    w = w[[c for c in cols if c in w.columns]]
    pp = d[d.dataset == "POOLED (HKSJ)"].set_index(["contrast", "pair"])["p"]
    w["pooled p"] = [fmt(pp.get((a, b), np.nan)) for a, b in zip(w.contrast, w.pair)]
    s.append(md(w, list(w.columns)))
    c = pd.read_csv(OUT / "p2_counts_summary.csv")
    s.append("\n### Routers T4. What each router design is trained on (means over splits)\n")
    s.append(md(c, ["dataset", "pair", "J", "n_val", "pos_win_val", "pos_groups_val", "nonzero_g_val", "n_fit",
                    "pos_groups_fit", "n_T", "pos_win_T", "pos_groups_T", "n_folds", "r10_n_nonzero", "r10_n_gpos"],
                names=["dataset", "pair", "J", "R3-R7/audit: windows (val)", "val pos windows", "val pos groups",
                       "val windows g != 0", "R8: windows (fit)", "fit pos groups", "R4x-R6x: windows (T)",
                       "T pos windows", "T pos groups", "cross-fit folds", "R10: windows (g != 0 in T)", "of which g > 0"], d=1))
    s.append("\n### Routers T5. Cross-fitted tiers and C on the training set (means over splits)\n")
    s.append(md(c, ["dataset", "pair", "ba_E_oof_T", "ba_C_oof_T", "C_T_R1_pooled", "C_T_R6x_pooled", "C_T_R1_foldavg",
                    "C_T_R6x_foldavg", "hys_R6x_share", "seconds_models"],
                names=["dataset", "pair", "BA edge (OOF on T)", "BA cloud (OOF on T)", "C_T R1 pooled", "C_T R6x pooled",
                       "C_T R1 within-fold", "C_T R6x within-fold", "HyS picks R6x (share)", "s per split (tiers)"]))
    st = pd.read_csv(OUT / "p2_stream_summary.csv")
    s.append("\n### Streaming T1. Causal streaming thresholds (quantile of out-of-fold training scores)\n")
    st["frac (mean, sd, min-max)"] = [f"{a:.3f}, {b:.3f}, {c:.3f}-{e:.3f}" for a, b, c, e in
                                      zip(st.frac_mean, st.frac_sd, st.frac_min, st.frac_max)]
    st["causal - look-ahead BA"] = [ci(a, b, c) for a, b, c in zip(st.diff_mean, st.diff_ci_lo, st.diff_ci_hi)]
    s.append(md(st, ["dataset", "pair", "router", "f", "J", "frac (mean, sd, min-max)", "ba_causal", "ba_lookahead",
                     "causal - look-ahead BA"],
                names=["dataset", "pair", "router", "target f", "J", "realised fraction (mean, sd, min-max)",
                       "BA causal", "BA look-ahead", "difference NB [95% CI]"]))
    cb = pd.read_csv(OUT / "p2_comb_summary.csv")
    cb["comb"] = [ci(a, b, c) for a, b, c in zip(cb.ba_comb, cb.ba_comb_lo, cb.ba_comb_hi)]
    cb["comb - stacker"] = [ci(a, b, c) for a, b, c in zip(cb.comb_minus_stacker, cb.cms_lo, cb.cms_hi)]
    cb["comb - best single"] = [ci(a, b, c) for a, b, c in zip(cb.comb_minus_best_single, cb.cmb_lo, cb.cmb_hi)]
    s.append("\n### Combiner T1. Feature-aware combiner (BA on test; NB mean [95% CI])\n")
    s.append(md(cb, ["dataset", "pair", "J", "ba_edge", "ba_cloud", "ba_stacker", "ba_oracle", "comb", "esc_frac",
                     "comb - stacker", "comb - best single", "unresolved", "share_recovered"],
                names=["dataset", "pair", "J", "BA edge", "BA cloud", "BA stacker (stored)", "BA oracle (stored)",
                       "BA combiner", "cloud share", "combiner - stacker", "combiner - max(edge, cloud)",
                       "unresolved (oracle - stacker)", "share of unresolved recovered"]))
    return "\n".join(s)


def p3():
    s = []
    by = pd.read_csv(OUT / "p3_by_level.csv")
    by["C(R6)"] = [ci(a, b, c) for a, b, c in zip(by.C_R6, by.C_R6_lo, by.C_R6_hi)]
    by["C(R6) - C(R1)"] = [ci(a, b, c) for a, b, c in zip(by.dC, by.dC_lo, by.dC_hi)]
    s.append("### Fixed tiers T1. Fixed tiers: R6 retrained on the H2 validation subsample, scored on the q = 1 test fold\n")
    s.append(md(by, ["dataset", "q", "J", "pos_win_val_kept", "pos_groups_val_kept", "nonzero_g_kept", "C(R6)", "C_R1",
                     "C(R6) - C(R1)", "checks"],
                names=["dataset", "q", "J", "kept positive val windows", "kept positive val groups", "val windows g != 0",
                       "C(R6) NB [95% CI]", "C(R1) (fixed)", "C(R6) - C(R1)", "subsample check"]))
    sl = pd.read_csv(OUT / "p3_slopes.csv")
    sl["two-stage slope"] = [ci(a, b, c, 4) for a, b, c in zip(sl.two_stage_mean, sl.ci_lo, sl.ci_hi)]
    s.append("\n### Fixed tiers T2. Slope of C(R6) on log2(positive windows): fixed tiers vs the confirmatory H2 run\n")
    s.append(md(sl, ["design", "dataset", "ols_slope", "two-stage slope", "J"],
                names=["design", "dataset", "pooled-OLS slope (within dataset)", "two-stage slope [95% CI]", "J / k"], d=4))
    return "\n".join(s)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["p1", "p2", "p3"]
    for w in which:
        (OUT / f"tables_{w}.md").write_text(globals()[w]())
        print(f"wrote tables_{w}.md")
