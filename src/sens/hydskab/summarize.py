"""POST-PLAN summaries for area hydskab (not confirmatory).

    python src/sens/hydskab/summarize.py

Reads results_sens/hydskab/<variant>/<pair>/split_*.parquet, computes NB-corrected means and 95% CIs exactly as
src/analyze.py does (load_results, nb_sizes, per_dataset_table), puts them next to the parent dataset's main result
(results/summary/per_dataset.csv), lists per-fold values for the leave-one-level-out designs, and recomputes the
per-dataset parts of H1-H5 for the declared deconfounded variants with analyze.py's own functions. Writes CSVs and
tables.md under results_sens/hydskab/summary/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import analyze as A                                   # noqa: E402

RES = ROOT / "results_sens" / "hydskab"
OUT = RES / "summary"
PARENT = lambda v: "hydraulic" if v.startswith("hyd") else "skab"   # noqa: E731
FOLD_ONLY = {"hyd_cooler", "hyd_accum", "skab_lso"}
DECLARED = ["hyd_blocks", "skab_filez"]
ROUTERS = [f"R{i}" for i in range(10)]
STEP_Q = ["ba_ef_ev", "ba_cf_cv", "ba_cf_ev", "ba_ef_cv", "shapley_model", "shapley_info", "interaction",
          "ba_stacker_dep", "ba_oracle_dep", "unresolved_dep", "self_oracle_headroom", "rho_pos", "audit_go", "outcome"]
ORDER = ["hyd_blocks", "hyd_runs", "hyd_weak", "hyd_severe", "hyd_stable", "hyd_cooler", "hyd_accum",
         "skab_filez", "skab_base", "skab_lso", "skab_notemp", "skab_edgeP",
         "skab_cm_pressure", "skab_cm_thermo", "skab_cm_flow"]


def cell(r, d=3):
    if r is None or not np.isfinite(r["mean"]):
        return "NA"
    if not np.isfinite(r.get("ci_lo", np.nan)):
        return A.fmt(r["mean"], d)
    return f"{A.fmt(r['mean'], d)} [{A.fmt(r['ci_lo'], d)}, {A.fmt(r['ci_hi'], d)}]"


def key_rows(pdt):
    """Rows of the key-quantity list (router '' for steps, routers for C/capture/ba_at_budget)."""
    out = []
    for q in STEP_Q:
        rt = "R6" if q in ("audit_go", "outcome") else ("R1" if q == "rho_pos" else "")
        out.append((q, rt, np.nan))
    for r in ROUTERS:
        out.append(("C", r, np.nan))
    for r in ("R1", "R6"):
        out.append(("capture", r, np.nan))
    for r in ("R1", "R6"):
        for b in (0.10, 0.30):
            out.append(("ba_at_budget", r, b))
    return out


def lookup(pdt, ds, pair, q, r, b):
    m = (pdt.dataset == ds) & (pdt.pair == pair) & (pdt.quantity == q) & (pdt.router.fillna("") == r)
    m &= pdt.budget.isna() if not np.isfinite(b) else np.isclose(pdt.budget, b)
    g = pdt[m]
    return None if g.empty else g.iloc[0]


def compare_table(v, pdt, main):
    pairs = [p for p in ("A", "B") if ((pdt.dataset == v) & (pdt.pair == p)).any()]
    par = PARENT(v)
    head = ["quantity", "router", "budget"] + sum([[f"{par} {p} (main)", f"{v} {p}"] for p in pairs], [])
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for q, r, b in key_rows(pdt):
        cells = []
        for p in pairs:
            cells += [cell(lookup(main, par, p, q, r, b)), cell(lookup(pdt, v, p, q, r, b))]
        if all(c == "NA" for c in cells[1::2]):
            continue
        L.append("| " + " | ".join([q, r, "" if not np.isfinite(b) else f"{b:.2f}"] + cells) + " |")
    return "\n".join(L) + "\n"


def router_summary(pdt, ds, pair):
    """Routers ordered by mean C (excl. oracle), and those whose 95% CI lies above 0.5."""
    t = pdt[(pdt.dataset == ds) & (pdt.pair == pair) & (pdt.quantity == "C") & pdt.router.isin(ROUTERS)]
    if t.empty:
        return "", ""
    t = t.sort_values("mean", ascending=False)
    order = " > ".join(f"{r} {m:.3f}" for r, m in zip(t.router, t["mean"]))
    above = ", ".join(t[t.ci_lo > 0.5].router) or "none"
    return order, above


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def auc_table(sizes, main_sizes, variants):
    """Test ROC AUC of the four factorial models per split (threshold-free; descriptive), NB mean and CI."""
    rows = []
    for v in ["hydraulic", "skab"] + variants:
        base = (ROOT / "results" / v) if v in ("hydraulic", "skab") else (RES / v)
        sz_all = main_sizes if v in ("hydraulic", "skab") else sizes
        for pair in ("A", "B"):
            files = sorted((base / pair).glob("arrays_*.npz"))
            files = [f for f in files if "_q" not in f.stem]
            if not files:
                continue
            vals = {k: [] for k in ("ef_ev", "cf_ev", "ef_cv", "cf_cv")}
            for f in files:
                with np.load(f) as z:
                    for k in vals:
                        if f"p_{k}" in z.files:
                            vals[k].append(auc(z["y_test"], z[f"p_{k}"]))
            sz = sz_all[(sz_all.dataset == v) & (sz_all.pair == pair)]
            rec = dict(dataset=v, pair=pair, J=len(files))
            for k, xs in vals.items():
                if v in FOLD_ONLY or sz.empty:
                    rec[f"auc_{k}"] = A.fmt(float(np.nanmean(xs)), 3) if xs else "NA"
                else:
                    rec[f"auc_{k}"] = cell(A.nb(xs, sz.iloc[0]))
            rows.append(rec)
    return pd.DataFrame(rows)


def folds_table(df, v):
    d = df[(df.dataset == v) & (df.h2_q == 1.0)]
    rows = []
    for (pair, sid), g in d.groupby(["pair", "split"]):
        gg = g[g.budget.isna()].drop_duplicates(["quantity", "router"])
        s = {(q, r): x for q, r, x in zip(gg.quantity, gg.router.fillna(""), gg.value)}
        arr = np.load(RES / v / pair / f"arrays_{sid}.npz")
        test_groups = ",".join(sorted(np.unique(arr["groups_test"])))
        rec = dict(variant=v, pair=pair, split=sid, test_group=test_groups,
                   n_train=int(g.n_train.iloc[0]), n_pos_train=int(g.n_pos_train.iloc[0]),
                   n_test=int(g.n_test.iloc[0]), n_pos_test=int(s[("n_pos_test", "")]))
        for k in ("ef_ev", "cf_cv"):
            rec[f"auc_{k}"] = auc(arr["y_test"], arr[f"p_{k}"])
            rec[f"pp_{k}"] = float(np.mean(arr[f"p_{k}"] >= arr[f"t_{k}"]))
        for q in STEP_Q:
            rt = "R6" if q in ("audit_go", "outcome") else ("R1" if q == "rho_pos" else "")
            rec[q] = s.get((q, rt), np.nan)
        for r in ROUTERS:
            rec[f"C_{r}"] = s.get(("C", r), np.nan)
        for r in ("R1", "R6"):
            rec[f"capture_{r}"] = s.get(("capture", r), np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def declared_h(df, sizes, v):
    """Per-dataset parts of H1-H5 for one variant, with analyze.py's functions (single dataset: no pooling)."""
    out = {}
    main_rows = df[df.h2_q == 1.0]
    pdx, _, _ = A.h1(main_rows, sizes, [v])
    out["H1"] = pdx
    h2res, h2by, m01 = A.h2(df, [v])
    out["H2"] = (h2res, h2by, m01)
    pdx3, _, _ = A.h3(main_rows, sizes, [v])
    out["H3"] = pdx3
    out["H4"] = A.h4(df, [v])[0]
    out["H5"] = A.h5(df, [v], 10000, 10000, 9000)[0]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, skipped = A.load_results(RES)
    variants = [v for v in ORDER if v in set(df.dataset)]
    roles = {v: "sensitivity" for v in variants}
    main_rows = df[df.h2_q == 1.0]
    sizes = A.nb_sizes(main_rows)
    pdt = A.per_dataset_table(main_rows, sizes, roles)
    pdt.to_csv(OUT / "per_variant.csv", index=False)
    sizes.to_csv(OUT / "sizes.csv", index=False)
    skipped.to_csv(OUT / "skipped.csv", index=False)
    par_sizes_main = pd.read_csv(ROOT / "results" / "summary" / "sizes.csv")
    main = pd.read_csv(ROOT / "results" / "summary" / "per_dataset.csv")
    main = main[main.dataset.isin(["hydraulic", "skab"])]
    main["router"] = main["router"].fillna("")
    pdt["router"] = pdt["router"].fillna("")

    L = ["# hydskab tables (POST-PLAN, generated by src/sens/hydskab/summarize.py)", "",
         f"Skipped splits: {len(skipped)}.", "", "## Splits and sizes", "",
         A.md_table(sizes, ["dataset", "pair", "J", "n_train", "n_test"], 1)]
    # router ordering overview
    rs = []
    for v in ["hydraulic", "skab"] + variants:
        src = main if v in ("hydraulic", "skab") else pdt
        if v in FOLD_ONLY:
            continue
        for p in ("A", "B"):
            o, ab = router_summary(src, v, p)
            if o:
                rs.append(dict(dataset=v, pair=p, order_by_mean_C=o, CI_above_0_5=ab))
    rs = pd.DataFrame(rs)
    rs.to_csv(OUT / "router_order.csv", index=False)
    L += ["## Router ordering (mean C, NB 95% CI above 0.5)", "", A.md_table(rs, list(rs.columns))]
    # step overview
    ov = []
    for v in ["hydraulic", "skab"] + [x for x in variants if x not in FOLD_ONLY]:
        src = main if v in ("hydraulic", "skab") else pdt
        for p in ("A", "B"):
            rec = dict(dataset=v, pair=p)
            for q in ("ba_ef_ev", "ba_cf_cv", "shapley_model", "shapley_info", "interaction"):
                rec[q] = cell(lookup(src, v, p, q, "", np.nan))
            rec["C_R1"] = cell(lookup(src, v, p, "C", "R1", np.nan))
            rec["C_R6"] = cell(lookup(src, v, p, "C", "R6", np.nan))
            if rec["ba_ef_ev"] != "NA":
                ov.append(rec)
    ov = pd.DataFrame(ov)
    L += ["## Overview: decomposition and router C (mean [NB 95% CI])", "", A.md_table(ov, list(ov.columns))]
    for v in variants:
        if v in FOLD_ONLY:
            continue
        L += [f"## {v} (parent {PARENT(v)}): key quantities, mean [NB 95% CI]", "", compare_table(v, pdt, main)]
    at = auc_table(sizes, par_sizes_main, variants)
    at.to_csv(OUT / "auc.csv", index=False)
    L += ["## Test ROC AUC of the factorial models (threshold-free, descriptive; NB 95% CI where J = 20)", "",
          A.md_table(at, list(at.columns))]
    fo = [folds_table(df, v) for v in variants if v in FOLD_ONLY]
    if fo:
        fo = pd.concat(fo, ignore_index=True)
        fo.to_csv(OUT / "folds.csv", index=False)
        cols1 = ["variant", "pair", "split", "test_group", "n_train", "n_pos_train", "n_test", "n_pos_test",
                 "auc_ef_ev", "auc_cf_cv", "pp_ef_ev", "pp_cf_cv", "ba_ef_ev", "ba_cf_cv", "ba_cf_ev", "ba_ef_cv", "shapley_model", "shapley_info", "interaction",
                 "unresolved_dep"]
        cols2 = ["variant", "pair", "split", "test_group"] + [f"C_{r}" for r in ROUTERS] + \
                ["capture_R1", "capture_R6", "rho_pos", "audit_go", "outcome"]
        L += ["## Leave-one-level-out designs, per fold (first repeat only, no NB intervals)", "",
              A.md_table(fo, cols1), "", A.md_table(fo, cols2)]
        mean = fo.groupby(["variant", "pair"])[[c for c in cols1[8:]] + [f"C_{r}" for r in ROUTERS]].mean().reset_index()
        L += ["Unweighted mean over folds (descriptive):", "", A.md_table(mean, list(mean.columns))]
    # declared variants: per-dataset H1-H5 pieces
    hrows = []
    par_df, _ = A.load_results(ROOT / "results")
    par_df = par_df[par_df.dataset.isin(["hydraulic", "skab"])]
    par_sizes = A.nb_sizes(par_df[par_df.h2_q == 1.0])
    for v in [x for x in DECLARED if x in variants]:
        for nm, d_, sz, ds in ((v, df, sizes, v), (PARENT(v) + " (main)", par_df, par_sizes, PARENT(v))):
            h = declared_h(d_, sz, ds)
            h1 = h["H1"].iloc[0] if len(h["H1"]) else None
            h2res, h2by, m01 = h["H2"]
            h3 = h["H3"].iloc[0] if len(h["H3"]) else None
            h4 = h["H4"]; h5 = h["H5"]
            hrows.append(dict(
                dataset=nm,
                H1_shapley_info=cell(h1, 4) if h1 is not None else "NA",
                H1_p_one_sided=A.fmt(h1["p_one_sided"], 4) if h1 is not None else "NA",
                H2_slope_C_R6=f"{A.fmt(h2res['slope'], 4)} [{A.fmt(h2res['ci_lo'], 4)}, {A.fmt(h2res['ci_hi'], 4)}] ({h2res['model']})",
                H2_mean_C_q01=A.fmt(m01, 3),
                H3_dC_R7_R6=cell(h3, 4) if h3 is not None else "NA",
                H3_p_eq=A.fmt(h3["p_eq"], 4) if h3 is not None else "NA",
                H4_slope=f"{A.fmt(h4['slope'], 4)} [{A.fmt(h4['ci_lo'], 4)}, {A.fmt(h4['ci_hi'], 4)}] ({h4['model']})",
                H5_cells=h5["n_cells"], H5_go=h5["n_go"], H5_outcome_pos=h5["n_outcome_pos"],
                H5_ba=A.fmt(h5["ba"], 3), H5_p_perm=A.fmt(h5["p_perm"], 4)))
            if nm == v:
                h2by.to_csv(OUT / f"h2_by_level_{v}.csv", index=False)
                L += [f"### H2 by level, {v}", "", A.md_table(h2by, ["dataset", "h2_q", "J", "mean_npos", "mean_C", "sd_C"])]
    if hrows:
        hr = pd.DataFrame(hrows)
        hr.to_csv(OUT / "declared_h1_h5.csv", index=False)
        L += ["## Declared deconfounded variants: per-dataset H1-H5 quantities (analyze.py functions, single dataset, "
              "no pooling, no Holm; H2/H4 fall back to OLS with one dataset)", "", A.md_table(hr, list(hr.columns))]
    (OUT / "tables.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
