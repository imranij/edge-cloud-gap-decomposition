"""POST-PLAN: markdown tables for REPORT.md from rates.csv, truth.csv and audit_thresholds.csv.

    python src/sens/simulation/tables.py > results_sens/simulation/tables.md
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[3] / "results_sens" / "simulation"
SC = ["S0", "S1", "S1b", "S2", "S3"]


def cell(r, se, d=3):
    if not np.isfinite(r):
        return "NA"
    return f"{r:.{d}f} ({se:.{d}f})" if np.isfinite(se) else f"{r:.{d}f}"


def table(df, keys, title, d=3):
    print(f"\n#### {title}\n")
    print("| design | quantity | " + " | ".join(SC) + " |")
    print("|---|---|" + "---|" * len(SC))
    for k in keys:
        g = df[df.key == k].set_index("scenario")
        if g.empty:
            continue
        row = g.iloc[0]
        vals = [cell(g.loc[s, "rate"], g.loc[s, "mcse"], d) if s in g.index else "" for s in SC]
        print(f"| {row.design} | {row.quantity} | " + " | ".join(vals) + " |")


def main():
    r = pd.read_csv(OUT / "rates.csv")
    t = pd.read_csv(OUT / "truth.csv")
    n = r.groupby("scenario").n_rep.max()
    print("Replicates per scenario: " + ", ".join(f"{s} {int(n.get(s, 0))}" for s in SC))
    print("\n#### Ground truth (Monte Carlo mean over draws, MC SE in parentheses)\n")
    qs = ["info_step", "info_step_fit800", "ba_edge", "ba_cloud", "share_g_nonzero", "C_R0", "C_R1", "C_R6",
          "dC_R6_R1", "cap_R1", "cap_R6", "dcap_R6_R1", "ba10_R1_minus_edge", "ba10_R6_minus_edge", "C_R6_gt_half",
          "dC_R6a_R6b", "rho_pos_test", "h2f_slope_C_R6", "h2j_slope_C_R6", "h2j_slope_dC_R6_R1"]
    print("| quantity | " + " | ".join(SC) + " |")
    print("|---|" + "---|" * len(SC))
    for q in qs:
        g = t[t.quantity == q].set_index("scenario")
        print(f"| {q} | " + " | ".join(cell(g.loc[s, "mean"], g.loc[s, "mcse"], 4) if s in g.index else "" for s in SC)
              + " |")
    nd = t.groupby("scenario").n_draws.max()
    print("\nDraws per scenario: " + ", ".join(f"{s} {int(nd.get(s, 0))}" for s in SC))
    table(r, ["h1_ds_reject", "h1_pool_reject", "h1_verdict_supported", "h1_ds_cover", "h1_ds_cover_fit800",
              "h1_ds_cover_naive", "h1_pool_cover", "h1_ds_size_at_truth_upper", "h1_ds_size_at_truth_lower",
              "h1_pool_size_at_truth_upper", "h1_pool_size_at_truth_lower"], "H1: information-step tests")
    table(r, ["h3_pool_equiv", "h3_feas_orig", "h3_feas_cor", "h3_equiv_and_feas_orig", "h3_equiv_and_feas_cor",
              "h3_equiv_given_h3_feas_orig", "h3_equiv_given_h3_feas_cor", "h3_orig_not_cor",
              "h3_ds_equiv", "h3_ds_feas_orig", "h3_ds_feas_cor", "h3_margin_pool_equiv", "h3_margin_ds_equiv",
              "h3_pool_cover0", "h3_sd_dC", "h3_pool_se", "h3_pool_floor"], "H3: TOST at +-0.02 (dC = C(R6a) - C(R6b))", 4)
    table(r, ["h4_orig_p_lt05", "h4_orig_supported", "h4_origR0_supported", "h4_valrho_only_supported",
              "h4_R0ref_testrho_supported", "h4_cor_mixed_p_lt05", "h4_cor_mixed_supported", "h4_cor_2stage_supported",
              "h4_ols_fallback", "h4_orig_slope", "h4_origR0_slope", "h4_cor_mixed_slope", "h4_cor_2stage_slope",
              "h4_rho_val_test_spearman"], "H4")
    table(r, ["h2_orig_p_lt05", "h2_orig_supported", "h2_orig_2stage_reject", "h2_cor_reject",
              "h2_level_R6_gt_R1_reject", "h2_orig_slope", "h2_orig_mean_C_q01", "h2_cor_slope", "h2_level_dC"], "H2")
    keys = []
    for call in ("go_orig", "go_cor", "go_cor_w"):
        keys.append(f"{call}|go_rate|out_C")
        if call == "go_orig":
            keys += [f"go_orig|outcome_rate|{o}" for o in ("out_C", "out_ba10", "out_cap")]
        for o in ("out_C", "out_ba10", "out_cap"):
            keys += [f"{call}|{m}|{o}" for m in ("precision", "recall", "ba")]
    table(r, keys, "H5: audit calls (cells = dataset x split; MC SE by replicate bootstrap)")
    th = pd.read_csv(OUT / "audit_thresholds.csv")
    print("\n#### H5: corrected-audit null calibration (pooled within-class permutation draws)\n")
    print(th.round(4).to_markdown(index=False))


if __name__ == "__main__":
    main()
