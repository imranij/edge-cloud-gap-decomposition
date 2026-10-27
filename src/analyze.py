"""Hypothesis tests of Section 7 and the descriptive tables of Section 8.

    python src/analyze.py [--results results] [--confirmatory hydraulic,skab,seu,threew]
                          [--n-perm 10000] [--n-boot 10000] [--seed 9000] [--allow-tiny]

Reads results/<dataset>/<pair>/split_*.parquet (and footprint.json, log.jsonl) and writes
results/summary/*.csv and results/summary/REPORT.md. Pilot datasets (SECOM, MetroPT-3) and
sensitivity runs (dataset names containing "__", e.g. threew__by_well) are reported separately
and never pooled. The confirmatory set defaults to the plan's four datasets (config.CONFIRMATORY);
any other dataset found is role 'other', excluded from H1-H5 and listed as excluded in REPORT.md.
Results written with test-only --tiny settings (a `tiny_settings` row = 1, or "tiny": true in a
log.jsonl) make the analysis abort; --allow-tiny exists only for the test suite.
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escalation import stats as S                         # noqa: E402
from escalation.config import PILOTS, BUDGETS, CONFIRMATORY  # noqa: E402

ALPHA = 0.05
MARGIN = 0.02
GO_BA_NULL = 0.5


# ------------------------------------------------------------------ loading
def role_of(name: str, confirmatory: list[str] | None = None) -> str:
    if confirmatory is None:
        confirmatory = list(CONFIRMATORY)
    low = name.lower()
    if low in PILOTS or any(low.startswith(p) for p in PILOTS):
        return "pilot"
    if "__" in name:
        return "sensitivity"
    if name not in confirmatory:
        return "other"
    return "confirmatory"


def tiny_sources(df: pd.DataFrame, root: Path) -> list[str]:
    """Where test-only --tiny settings were used: dataset/pair of tiny_settings rows, and log files."""
    t = df[(df.quantity == "tiny_settings") & (df.value == 1)]
    found = sorted({f"{d}/{p}" for d, p in t[["dataset", "pair"]].itertuples(index=False)})
    for lg in sorted(root.glob("*/*/log.jsonl")):
        for line in lg.read_text().splitlines():
            if line.strip() and json.loads(line).get("tiny") is True:
                found.append(str(lg.relative_to(root)))
                break
    return found


def load_results(root: Path, allow_tiny: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(root.glob("*/*/split_*.parquet"))
    files = [f for f in files if f.parts[-3] != "summary"]
    if not files:
        raise SystemExit(f"no results under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tiny = tiny_sources(df, root)
    if tiny and not allow_tiny:
        raise SystemExit("results written with test-only --tiny settings found (" + ", ".join(tiny[:10]) +
                         "); remove them and rerun at plan settings")
    df = df[df.quantity != "tiny_settings"]
    skipped = df[df.quantity == "skipped"][["dataset", "pair", "split", "h2_q"]].copy()
    reasons = {}
    for lg in root.glob("*/*/log.jsonl"):
        for line in lg.read_text().splitlines():
            r = json.loads(line)
            if "skipped" in r:
                reasons[(lg.parts[-3], lg.parts[-2], str(r["split"]))] = r["skipped"]
    skipped["reason"] = [reasons.get((d, p, f"{s}" + ("" if q == 1.0 else f"_q{q}")), "")
                         for d, p, s, q in skipped[["dataset", "pair", "split", "h2_q"]].itertuples(index=False)]
    return df[df.quantity != "skipped"].copy(), skipped


def nb_sizes(df_main: pd.DataFrame) -> pd.DataFrame:
    """n_train (fit + val) and n_test averaged over splits, per dataset x pair (main run)."""
    per = df_main.drop_duplicates(["dataset", "pair", "split"])
    return per.groupby(["dataset", "pair"]).agg(n_train=("n_train", "mean"), n_test=("n_test", "mean"),
                                                J=("split", "nunique")).reset_index()


def nb(values, sizes_row, margin=None) -> dict:
    return S.nb_test(np.asarray(values, float), sizes_row["n_train"], sizes_row["n_test"], margin=margin)


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if abs(x) < 10 ** (-d) and x != 0:
        return f"{x:.1e}"
    return f"{x:.{d}f}"


def ci(lo, hi, d=3):
    return f"[{fmt(lo, d)}, {fmt(hi, d)}]"


# ------------------------------------------------------------------ per-dataset tables
def per_dataset_table(df_main, sizes, roles) -> pd.DataFrame:
    out = []
    keys = ["dataset", "pair", "quantity", "router", "budget"]
    d = df_main.copy()
    d["budget"] = d["budget"].fillna(-1.0)
    for (ds, pair, q, r, b), g in d.groupby(keys, sort=True):
        if q in ("seconds",):
            continue
        sz = sizes[(sizes.dataset == ds) & (sizes.pair == pair)].iloc[0]
        t = nb(g["value"], sz)
        out.append(dict(dataset=ds, role=roles[ds], pair=pair, quantity=q, router=r,
                        budget=np.nan if b == -1.0 else b, J=t["J"], mean=t["mean"], sd=t.get("sd", np.nan),
                        se_nb=t["se"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], p_two_sided=t["p"],
                        se_floor=t["se_floor"], n_train=sz["n_train"], n_test=sz["n_test"]))
    return pd.DataFrame(out)


def series(df, pair, quantity, router="", h2_q=1.0, budget=None):
    m = (df.pair == pair) & (df.quantity == quantity) & (df.router == router) & (df.h2_q == h2_q)
    if budget is not None:
        m &= np.isclose(df.budget, budget)
    return df[m]


# ------------------------------------------------------------------ hypotheses
def per_dataset_effect(df, sizes, datasets, pair, quantity, router="", margin=None, direction="greater"):
    rows = []
    for ds in datasets:
        g = series(df[df.dataset == ds], pair, quantity, router)
        if g.empty:
            continue
        sz = sizes[(sizes.dataset == ds) & (sizes.pair == pair)].iloc[0]
        t = nb(g.sort_values("split")["value"], sz, margin)
        t["p_one_sided"] = S.one_sided_p(t, direction) if t["J"] >= 2 and t["se"] > 0 else np.nan
        t.update(dataset=ds, n_train=sz["n_train"], n_test=sz["n_test"])
        rows.append(t)
    return pd.DataFrame(rows)


def h1(df, sizes, datasets):
    pdx = per_dataset_effect(df, sizes, datasets, "A", "shapley_info")
    pool = S.pool_hksj(pdx["mean"], pdx["se"]) if len(pdx) else dict(k=0, mean=np.nan, se=np.nan, ci_lo=np.nan,
                                                                        ci_hi=np.nan, df=None)
    p1 = S.one_sided_p(pool, "greater") if pool.get("k", 0) >= 2 else np.nan
    return pdx, pool, p1


def h1_composite(df, sizes, datasets):
    pdx = per_dataset_effect(df, sizes, datasets, "A", "comp_shapley_info")
    pool = S.pool_hksj(pdx["mean"], pdx["se"]) if len(pdx) else {}
    return pdx, pool


def mixed_slope(data: pd.DataFrame, formula: str, term: str) -> dict:
    """Linear mixed model with dataset random intercepts (statsmodels MixedLM, REML). Falls back
    to OLS when fewer than two datasets are present (reported)."""
    import statsmodels.formula.api as smf
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    out = dict(n=len(data), n_groups=data["dataset"].nunique() if len(data) else 0, model="MixedLM")
    if len(data) < 3:
        return dict(out, slope=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan, z=np.nan, note="too few rows")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            if out["n_groups"] < 2:
                raise ValueError("one group")
            fit = smf.mixedlm(formula, data, groups=data["dataset"]).fit(reml=True)
            note = "; ".join(sorted({str(x.message)[:80] for x in w})) or ""
        except Exception as e:                               # noqa: BLE001
            fit = smf.ols(formula, data).fit()
            out["model"] = "OLS (fallback)"
            note = f"MixedLM not estimable ({str(e)[:60]}); OLS used"
    b = float(fit.params[term]); se = float(fit.bse[term])
    lo, hi = (float(v) for v in fit.conf_int().loc[term])
    return dict(out, slope=b, se=se, ci_lo=lo, ci_hi=hi, z=b / se if se > 0 else np.nan, note=note)


def h2(df, datasets):
    d = df[(df.dataset.isin(datasets)) & (df.pair == "A") & (df.quantity == "C") & (df.router == "R6")].copy()
    d["log2_npos"] = np.log2(d["n_pos_train"].clip(lower=1))
    res = mixed_slope(d[["dataset", "value", "log2_npos"]].rename(columns={"value": "C"}), "C ~ log2_npos",
                      "log2_npos")
    res["p_one_sided"] = float(sps.norm.sf(res["z"])) if np.isfinite(res.get("z", np.nan)) else np.nan
    by = d.groupby(["dataset", "h2_q"]).agg(mean_C=("value", "mean"), sd_C=("value", "std"), J=("value", "size"),
                                            mean_npos=("n_pos_train", "mean")).reset_index()
    mean_q01 = float(d[np.isclose(d.h2_q, 0.1)]["value"].mean()) if (np.isclose(d.h2_q, 0.1)).any() else np.nan
    return res, by, mean_q01


def h2_kept(df, datasets) -> pd.DataFrame:
    """Realised H2 subsampling (applied per part, plan/DEVIATIONS.md): positive groups kept / available
    in the fit and validation parts, averaged over splits, per dataset and level (q = 1.0 for reference)."""
    names = ["n_pos_groups_fit", "n_pos_groups_fit_kept", "n_pos_groups_val", "n_pos_groups_val_kept"]
    d = df[df.dataset.isin(datasets) & (df.pair == "A") & df.quantity.isin(names)]
    if d.empty:
        return pd.DataFrame(columns=["dataset", "h2_q", "J"] + names + ["frac_fit_kept", "frac_val_kept"])
    w = d.pivot_table(index=["dataset", "h2_q", "split"], columns="quantity", values="value").reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        w["frac_fit_kept"] = w["n_pos_groups_fit_kept"] / w["n_pos_groups_fit"]
        w["frac_val_kept"] = w["n_pos_groups_val_kept"] / w["n_pos_groups_val"]
    agg = {c: (c, "mean") for c in names + ["frac_fit_kept", "frac_val_kept"]}
    return w.groupby(["dataset", "h2_q"]).agg(J=("split", "size"), **agg).reset_index()


def h3(df, sizes, datasets):
    rows = []
    for ds in datasets:
        a = series(df[df.dataset == ds], "A", "C", "R7").set_index("split")["value"]
        b = series(df[df.dataset == ds], "A", "C", "R6").set_index("split")["value"]
        dC = (a - b).dropna()
        if dC.empty:
            continue
        sz = sizes[(sizes.dataset == ds) & (sizes.pair == "A")].iloc[0]
        t = S.nb_test(dC.to_numpy(), sz["n_train"], sz["n_test"], margin=MARGIN)
        t["p_one_sided_greater"] = S.one_sided_p(t, "greater") if t["J"] >= 2 and t["se"] > 0 else np.nan
        pf = S.power_floor(t.get("sd", np.nan), sz["n_train"], sz["n_test"], MARGIN, t["J"]) \
            if t["J"] >= 2 else dict(se_floor=np.nan, feasible=False, repeats_needed=None)
        t.update(dataset=ds, n_train=sz["n_train"], n_test=sz["n_test"], tost_feasible=pf["feasible"],
                 repeats_needed=pf["repeats_needed"])
        rows.append(t)
    pdx = pd.DataFrame(rows)
    if pdx.empty:
        return pdx, {}, {}
    pdx["p_holm_within_h3"] = S.holm(pdx["p_one_sided_greater"].fillna(1.0))
    pool = S.pool_hksj(pdx["mean"], pdx["se"], margin=MARGIN)
    # pooled floor: HKSJ se >= sqrt(1 / sum(1 / floor_i^2)) (tau^2 >= 0, q* >= 1)
    fl = pdx["se_floor"].to_numpy(float)
    fl = fl[np.isfinite(fl) & (fl > 0)]
    k = len(fl)
    pooled_floor = float(np.sqrt(1.0 / (1.0 / fl ** 2).sum())) if k else np.nan
    dfp = max(k - 1, 1)
    feasible = bool(np.isfinite(pooled_floor) and pooled_floor * sps.t.ppf(1 - ALPHA, dfp) < MARGIN)
    return pdx, pool, dict(pooled_se_floor=pooled_floor, pooled_tost_feasible=feasible)


def h4(df, datasets):
    d = df[df.dataset.isin(datasets) & df.pair.isin(["A", "B"]) & (df.h2_q == 1.0)]
    rho = d[d.quantity == "rho_pos"].set_index(["dataset", "pair", "split"])["value"].rename("rho_pos")
    dcap = d[d.quantity == "dcap_R6_R1"].set_index(["dataset", "pair", "split"])["value"].rename("dcap")
    m = pd.concat([rho, dcap], axis=1).reset_index()
    res = mixed_slope(m[["dataset", "dcap", "rho_pos"]], "dcap ~ rho_pos", "rho_pos")
    res["p_one_sided"] = float(sps.norm.cdf(res["z"])) if np.isfinite(res.get("z", np.nan)) else np.nan
    return res, m


def _ba_from_counts(tp, fn, tn, fp):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def h5(df, datasets, n_perm, n_boot, seed):
    d = df[df.dataset.isin(datasets) & df.pair.isin(["A", "B"])]
    keys = ["dataset", "pair", "split", "h2_q"]
    call = d[d.quantity == "audit_go"].set_index(keys)["value"].rename("call")
    outc = d[d.quantity == "outcome"].set_index(keys)["value"].rename("outcome")
    cells = pd.concat([call, outc], axis=1).dropna().reset_index()
    res = dict(n_cells=len(cells), n_go=int(cells["call"].sum()) if len(cells) else 0,
               n_outcome_pos=int(cells["outcome"].sum()) if len(cells) else 0)
    if len(cells) == 0 or cells["outcome"].nunique() < 2:
        return dict(res, ba=np.nan, p_perm=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    note="outcomes do not contain both classes; BA undefined"), cells
    c = cells["call"].to_numpy().astype(int); o = cells["outcome"].to_numpy().astype(int)

    def ba(cc, oo):
        tp = ((cc == 1) & (oo == 1)).sum(-1); fn = ((cc == 0) & (oo == 1)).sum(-1)
        tn = ((cc == 0) & (oo == 0)).sum(-1); fp = ((cc == 1) & (oo == 0)).sum(-1)
        return _ba_from_counts(tp, fn, tn, fp)
    obs = float(ba(c, o))
    rng = np.random.default_rng(seed)
    perm_c = np.empty((n_perm, len(c)), dtype=np.int8)
    for ds, idx in cells.groupby("dataset").indices.items():
        idx = np.asarray(idx)
        perm_c[:, idx] = c[idx][rng.permuted(np.tile(np.arange(len(idx)), (n_perm, 1)), axis=1)]
    bperm = ba(perm_c, o[None, :])
    p_perm = float((1 + (bperm >= obs - 1e-12).sum()) / (n_perm + 1))
    # dataset x split block bootstrap
    cells["block"] = cells["dataset"].astype(str) + "|" + cells["split"].astype(str)
    blk = cells.groupby("block").apply(lambda g: pd.Series(dict(
        tp=((g.call == 1) & (g.outcome == 1)).sum(), fn=((g.call == 0) & (g.outcome == 1)).sum(),
        tn=((g.call == 0) & (g.outcome == 0)).sum(), fp=((g.call == 1) & (g.outcome == 0)).sum())),
        include_groups=False)
    M = blk[["tp", "fn", "tn", "fp"]].to_numpy(float)
    rng2 = np.random.default_rng(seed + 1)
    pick = rng2.integers(0, len(M), size=(n_boot, len(M)))
    tot = M[pick].sum(axis=1)
    bb = _ba_from_counts(*tot.T)
    lo, hi = np.nanpercentile(bb, [2.5, 97.5])
    return dict(res, ba=obs, p_perm=p_perm, ci_lo=float(lo), ci_hi=float(hi), n_blocks=len(M), note=""), cells


# ------------------------------------------------------------------ descriptive extras
def footprints(root: Path, roles) -> tuple[pd.DataFrame, pd.DataFrame]:
    fr, pr = [], []
    for f in sorted(root.glob("*/*/footprint.json")):
        j = json.loads(f.read_text())
        for k, v in j["footprint"].items():
            fr.append(dict(dataset=j["dataset"], role=roles.get(j["dataset"], "?"), pair=j["pair"], model=k, **v))
        pr.append(dict(dataset=j["dataset"], role=roles.get(j["dataset"], "?"), pair=j["pair"],
                       **{k: v for k, v in j["payload_bytes"].items() if not isinstance(v, str)}))
    fp = pd.DataFrame(fr)
    if len(fp):
        fp["input_shape"] = fp["input_shape"].astype(str)
    pay = pd.DataFrame(pr).drop_duplicates("dataset").drop(columns=["pair"], errors="ignore") if pr else pd.DataFrame()
    return fp, pay


def router_table(pdt: pd.DataFrame) -> pd.DataFrame:
    r = pdt[pdt.quantity.isin(["C", "capture", "ba_at_budget", "recall_at_budget"]) & (pdt.router != "")]
    return r.sort_values(["role", "dataset", "pair", "quantity", "budget", "router"])


# ------------------------------------------------------------------ verdicts and report
def verdicts(h1r, h2r, h3r, h4r, h5r, holm_p):
    v = {}
    pdx, pool, _ = h1r
    n_pos = int((pdx["p_one_sided"] < ALPHA).sum()) if len(pdx) else 0
    ci_has0 = not (pool.get("ci_lo", np.nan) > 0 or pool.get("ci_hi", np.nan) < 0) if np.isfinite(pool.get("ci_lo", np.nan)) else True
    if holm_p["H1"] < ALPHA and n_pos >= 2:
        v["H1"] = "SUPPORTED"
    elif ci_has0 and n_pos == 0:
        v["H1"] = "FALSIFIED"
    else:
        v["H1"] = "INCONCLUSIVE (neither the support nor the falsification rule is met)"
    res, by, mean01 = h2r
    ci_has0 = not (res["ci_lo"] > 0 or res["ci_hi"] < 0) if np.isfinite(res["ci_lo"]) else True
    if ci_has0 or (np.isfinite(mean01) and mean01 > 0.6):
        v["H2"] = "FALSIFIED" + (" (slope CI includes 0)" if ci_has0 else "") + \
                  (" (mean C at q = 0.1 > 0.6)" if np.isfinite(mean01) and mean01 > 0.6 else "")
    elif holm_p["H2"] < ALPHA:
        v["H2"] = "SUPPORTED (slope > 0 after Holm; falsification rule not met)"
    else:
        v["H2"] = "INCONCLUSIVE"
    pdx3, pool3, feas = h3r
    n_up = int(((pdx3["mean"] > 0) & (pdx3["p_holm_within_h3"] < ALPHA)).sum()) if len(pdx3) else 0
    if n_up >= 3:
        v["H3"] = "FALSIFIED (at least three datasets show dC > 0, Holm within H3)"
    elif not feas.get("pooled_tost_feasible", False):
        v["H3"] = "NOT TESTABLE AS EQUIVALENCE: the corrected-SE floor makes the pooled TOST infeasible; no equivalence claimed"
    elif holm_p["H3"] < ALPHA:
        v["H3"] = "SUPPORTED (pooled TOST p_eq < 0.05 after Holm)"
    else:
        v["H3"] = "INCONCLUSIVE (equivalence not shown; falsification rule not met)"
    r4, _ = h4r
    ci_has0 = not (r4["ci_lo"] > 0 or r4["ci_hi"] < 0) if np.isfinite(r4["ci_lo"]) else True
    if not np.isfinite(r4["slope"]) or r4["slope"] >= 0 or ci_has0:
        v["H4"] = "FALSIFIED" + (" (slope >= 0)" if np.isfinite(r4["slope"]) and r4["slope"] >= 0 else "") + \
                  (" (slope CI includes 0)" if ci_has0 else "")
    elif holm_p["H4"] < ALPHA:
        v["H4"] = "SUPPORTED (slope < 0 after Holm)"
    else:
        v["H4"] = "INCONCLUSIVE"
    r5, _ = h5r
    if not np.isfinite(r5["ba"]):
        v["H5"] = "NOT EVALUABLE (" + r5.get("note", "") + ")"
    elif holm_p["H5"] < ALPHA and r5["ci_lo"] > GO_BA_NULL:
        v["H5"] = "SUPPORTED"
    else:
        v["H5"] = "NOT SUPPORTED (support rule not met)"
    return v


def md_table(df: pd.DataFrame, cols, d=3) -> str:
    if df is None or len(df) == 0:
        return "_(none)_\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(fmt(v, d) if isinstance(v, (float, np.floating, int, np.integer)) and not isinstance(v, bool)
                         else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--confirmatory", default=None)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--allow-tiny", action="store_true", help="test suite only")
    a = ap.parse_args(argv)
    root = Path(a.results)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    conf = a.confirmatory.split(",") if a.confirmatory else list(CONFIRMATORY)
    df, skipped = load_results(root, a.allow_tiny)
    roles = {ds: role_of(ds, conf) for ds in df.dataset.unique()}
    confirm = sorted(ds for ds, r in roles.items() if r == "confirmatory")
    pilots = sorted(ds for ds, r in roles.items() if r == "pilot")
    sens = sorted(ds for ds, r in roles.items() if r == "sensitivity")
    other = sorted(ds for ds, r in roles.items() if r == "other")
    main_rows = df[df.h2_q == 1.0]
    sizes = nb_sizes(main_rows)
    sizes.assign(role=sizes.dataset.map(roles)).to_csv(out / "sizes.csv", index=False)
    skipped.to_csv(out / "skipped.csv", index=False)

    pdt = per_dataset_table(main_rows, sizes, roles)
    pdt.to_csv(out / "per_dataset.csv", index=False)
    router_table(pdt).to_csv(out / "routers.csv", index=False)
    if (df.h2_q != 1.0).any():
        hq = df[df.h2_q != 1.0]
        hs = nb_sizes(hq.assign(pair=hq.pair + "_q" + hq.h2_q.astype(str)))
        rows = []
        for (ds, pair, q), g in hq.groupby(["dataset", "pair", "h2_q"]):
            for (qq, r), gg in g.groupby(["quantity", "router"]):
                sz = hs[(hs.dataset == ds) & (hs.pair == f"{pair}_q{q}")].iloc[0]
                t = nb(gg["value"], sz)
                rows.append(dict(dataset=ds, role=roles[ds], pair=pair, h2_q=q, quantity=qq, router=r, J=t["J"],
                                 mean=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"]))
        pd.DataFrame(rows).to_csv(out / "per_dataset_h2.csv", index=False)

    # ---------------- hypotheses (confirmatory only)
    h1r = h1(main_rows, sizes, confirm)
    h1c = h1_composite(main_rows, sizes, confirm)
    h2r = h2(df, confirm)
    h2k = h2_kept(df, confirm)
    h3r = h3(main_rows, sizes, confirm)
    h4r = h4(df, confirm)
    h5r = h5(df, confirm, a.n_perm, a.n_boot, a.seed)
    prim = dict(H1=h1r[2], H2=h2r[0]["p_one_sided"], H3=h3r[1].get("p_eq", np.nan) if h3r[1] else np.nan,
                H4=h4r[0]["p_one_sided"], H5=h5r[0]["p_perm"])
    raw = np.array([prim[k] for k in ("H1", "H2", "H3", "H4", "H5")], float)
    adj = S.holm(np.where(np.isfinite(raw), raw, 1.0))
    holm_p = dict(zip(("H1", "H2", "H3", "H4", "H5"), adj))
    v = verdicts(h1r, h2r, h3r, h4r, h5r, holm_p)
    primary = pd.DataFrame(dict(hypothesis=list(prim), p_raw=raw, p_holm=adj, verdict=[v[k] for k in prim],
                                p_raw_missing_set_to_1=~np.isfinite(raw)))
    primary.to_csv(out / "primary.csv", index=False)
    h1r[0].to_csv(out / "h1_per_dataset.csv", index=False)
    pd.DataFrame([dict(h1r[1], p_one_sided=h1r[2])]).to_csv(out / "h1_pooled.csv", index=False)
    pd.DataFrame([h2r[0]]).to_csv(out / "h2_model.csv", index=False)
    h2r[1].to_csv(out / "h2_by_level.csv", index=False)
    h2k.to_csv(out / "h2_kept_fraction.csv", index=False)
    h3r[0].to_csv(out / "h3_per_dataset.csv", index=False)
    pd.DataFrame([dict(h3r[1], **h3r[2])]).to_csv(out / "h3_pooled.csv", index=False)
    pd.DataFrame([h4r[0]]).to_csv(out / "h4_model.csv", index=False)
    h4r[1].to_csv(out / "h4_data.csv", index=False)
    pd.DataFrame([h5r[0]]).to_csv(out / "h5.csv", index=False)
    h5r[1].to_csv(out / "h5_cells.csv", index=False)
    # power floors for every +-0.02 equivalence test (H3) and, for information, H1
    pf = []
    for nm, t in (("H3 dC", h3r[0]), ("H1 shapley_info", h1r[0])):
        for _, r in t.iterrows():
            p = S.power_floor(r.get("sd", np.nan), r["n_train"], r["n_test"], MARGIN, int(r["J"]))
            pf.append(dict(test=nm, dataset=r["dataset"], J=r["J"], sd=r.get("sd", np.nan), se_floor=p["se_floor"],
                           tost_feasible_pm002=p["feasible"], repeats_needed=p["repeats_needed"]))
    pf = pd.DataFrame(pf)
    pf.to_csv(out / "power_floors.csv", index=False)
    fp, pay = footprints(root, roles)
    fp.to_csv(out / "footprint.csv", index=False)
    pay.to_csv(out / "payload.csv", index=False)

    # ---------------- report
    L = ["# Paper A: confirmatory analysis report", ""]
    if a.allow_tiny:
        L += ["**WARNING: --allow-tiny was set; these results may use test-only reduced settings and are not "
              "confirmatory.**", ""]
    L += [
         "Generated by `src/analyze.py` from the per-split results. Decision rules are those of "
         "`plan/ANALYSIS_PLAN.md`, Section 7. Per-dataset tests use the Nadeau-Bengio corrected t-test "
         "(n_train = fit + validation, n_test = test, both averaged over splits); pooled tests use "
         "random effects with the HKSJ correction; the five primary p-values are Holm-adjusted together.", "",
         f"- Confirmatory datasets pooled: {', '.join(confirm) or 'none'}",
         f"- Pilot datasets (never pooled): {', '.join(pilots) or 'none'}",
         f"- Sensitivity runs (never pooled): {', '.join(sens) or 'none'}",
         f"- Excluded from H1-H5 (not in the confirmatory list {', '.join(conf)}): "
         f"{', '.join(other) or 'none'}",
         f"- Skipped splits: {len(skipped)} (see skipped.csv)", ""]
    if len(confirm) != 4:
        L += [f"**Note.** The plan specifies four confirmatory datasets; {len(confirm)} are present, so the "
              f"pooled t has {max(len(confirm) - 1, 0)} df instead of 3.", ""]
    L += ["## Splits and sizes", "", md_table(sizes.assign(role=sizes.dataset.map(roles)),
                                              ["dataset", "role", "pair", "J", "n_train", "n_test"], 1)]
    L += ["## Primary tests (Holm across the five primary p-values)", "",
          md_table(primary, ["hypothesis", "p_raw", "p_holm", "verdict"], 4)]
    # H1
    pdx, pool, p1 = h1r
    L += ["## H1. The information step is positive (pair A, Shapley information step)", "",
          md_table(pdx.assign(ci=[ci(a_, b_) for a_, b_ in zip(pdx.get("ci_lo", []), pdx.get("ci_hi", []))]),
                   ["dataset", "J", "mean", "ci", "se", "se_floor", "p_one_sided"], 4) if len(pdx) else "_(none)_\n",
          f"Pooled (HKSJ, df = {pool.get('df')}): mean {fmt(pool.get('mean', np.nan), 4)}, 95% CI "
          f"{ci(pool.get('ci_lo', np.nan), pool.get('ci_hi', np.nan), 4)}, one-sided p {fmt(p1, 4)}, Holm p "
          f"{fmt(holm_p['H1'], 4)}. Datasets individually positive (one-sided p < 0.05): "
          f"{int((pdx['p_one_sided'] < ALPHA).sum()) if len(pdx) else 0}.", "",
          f"**Verdict: {v['H1']}.**", ""]
    cpdx, cpool = h1c
    if len(cpdx):
        L += [f"Sensitivity (composite thresholds): pooled Shapley information step {fmt(cpool.get('mean', np.nan), 4)}, "
              f"95% CI {ci(cpool.get('ci_lo', np.nan), cpool.get('ci_hi', np.nan), 4)}.", ""]
    # H2
    res, by, mean01 = h2r
    L += ["## H2. Identifiability rises with labelled fault units (pair A, R6, test C)", "",
          f"{res['model']} with dataset random intercepts, C ~ log2(positive training windows): slope "
          f"{fmt(res['slope'], 4)}, 95% CI {ci(res['ci_lo'], res['ci_hi'], 4)}, one-sided p {fmt(res['p_one_sided'], 4)}, "
          f"Holm p {fmt(holm_p['H2'], 4)} (n = {res['n']} cells, {res['n_groups']} datasets). Mean C at q = 0.1: "
          f"{fmt(mean01, 4)}. {('Model notes: ' + res['note']) if res.get('note') else ''}", "",
          md_table(by, ["dataset", "h2_q", "J", "mean_npos", "mean_C", "sd_C"], 3),
          "Realised subsampling. q is applied separately to the fit and validation parts, each keeping "
          "max(1, round(q x positive groups in that part)) groups (plan/DEVIATIONS.md), so the kept "
          "fraction can exceed q when a part has few positive groups. Means over splits:", "",
          md_table(h2k, ["dataset", "h2_q", "J", "n_pos_groups_fit", "n_pos_groups_fit_kept", "frac_fit_kept",
                         "n_pos_groups_val", "n_pos_groups_val_kept", "frac_val_kept"], 2),
          f"**Verdict: {v['H2']}.**", ""]
    # H3
    pdx3, pool3, feas = h3r
    L += ["## H3. Explanation and drift inputs do not raise C (dC = C(R7) - C(R6), pair A, margin 0.02)", ""]
    if len(pdx3):
        L += [md_table(pdx3.assign(ci=[ci(x, y) for x, y in zip(pdx3.ci_lo, pdx3.ci_hi)]),
                       ["dataset", "J", "mean", "ci", "se", "se_floor", "p_eq", "tost_feasible",
                        "p_one_sided_greater", "p_holm_within_h3"], 4),
              f"Pooled (HKSJ): mean {fmt(pool3.get('mean', np.nan), 4)}, 95% CI "
              f"{ci(pool3.get('ci_lo', np.nan), pool3.get('ci_hi', np.nan), 4)}, TOST p_eq "
              f"{fmt(pool3.get('p_eq', np.nan), 4)}, Holm p {fmt(holm_p['H3'], 4)}. Pooled SE floor "
              f"{fmt(feas.get('pooled_se_floor', np.nan), 4)}; pooled TOST feasible: {feas.get('pooled_tost_feasible')}.", ""]
    L += [f"**Verdict: {v['H3']}.**", ""]
    # H4
    r4, m4 = h4r
    L += ["## H4. Learned routers beat confidence only where confidence fails to order gains", "",
          f"{r4['model']}: dcap = capture(R6) - capture(R1) ~ rho_pos + (1 | dataset), pairs A and B: slope "
          f"{fmt(r4['slope'], 4)}, 95% CI {ci(r4['ci_lo'], r4['ci_hi'], 4)}, one-sided p (slope < 0) "
          f"{fmt(r4['p_one_sided'], 4)}, Holm p {fmt(holm_p['H4'], 4)} (n = {r4['n']}). "
          f"{('Model notes: ' + r4['note']) if r4.get('note') else ''}", "",
          f"**Verdict: {v['H4']}.**", ""]
    # H5
    r5, cells = h5r
    L += ["## H5. A pre-specified audit predicts whether routing pays", "",
          f"Cells: {r5['n_cells']} (dataset x pair A/B x split x H2 level); GO calls {r5['n_go']}; realised "
          f"outcomes (capture(R6, 0.05-0.30) > 0.10) {r5['n_outcome_pos']}. Balanced accuracy of the calls "
          f"{fmt(r5['ba'], 4)}, dataset-stratified permutation p {fmt(r5['p_perm'], 4)} ({a.n_perm} permutations), "
          f"Holm p {fmt(holm_p['H5'], 4)}, dataset x split block-bootstrap 95% CI {ci(r5['ci_lo'], r5['ci_hi'], 4)} "
          f"({a.n_boot} resamples). {r5.get('note', '')}", "",
          f"**Verdict: {v['H5']}.**", ""]
    # power
    L += ["## Power and feasibility (Section 9)", "",
          "Corrected per-dataset SEs cannot fall below sd * sqrt(n_test / n_train). For every +-0.02 equivalence "
          "test (H3) the floor and the TOST feasibility are listed; H1 floors are given for information.", "",
          md_table(pf, ["test", "dataset", "J", "sd", "se_floor", "tost_feasible_pm002", "repeats_needed"], 4)]
    # secondary
    key_q = ["ba_ef_ev", "ba_cf_ev", "ba_ef_cv", "ba_cf_cv", "shapley_model", "shapley_info", "interaction",
             "comb_step_dep", "unresolved_dep", "comb_step_upg", "unresolved_upg", "self_oracle_headroom"]
    for role, dsl in (("Confirmatory", confirm), ("Pilot (not pooled)", pilots), ("Sensitivity (not pooled)", sens)):
        if not dsl:
            continue
        t = pdt[pdt.dataset.isin(dsl) & pdt.quantity.isin(key_q) & (pdt.router == "")].copy()
        t["ci"] = [ci(x, y) for x, y in zip(t.ci_lo, t.ci_hi)]
        L += [f"## Secondary: escalation quantities, {role.lower()} datasets (mean, corrected 95% CI)", "",
              md_table(t, ["dataset", "pair", "quantity", "J", "mean", "ci"], 4)]
        r = pdt[pdt.dataset.isin(dsl) & pdt.quantity.isin(["C", "capture"]) & (pdt.router != "")].copy()
        r["ci"] = [ci(x, y) for x, y in zip(r.ci_lo, r.ci_hi)]
        L += [f"### Routers, {role.lower()} datasets (C and capture over 0.05-0.30)", "",
              md_table(r, ["dataset", "pair", "router", "quantity", "mean", "ci"], 3),
              "BA and recall at every budget (" + ", ".join(str(b) for b in BUDGETS) + ") are in routers.csv.", ""]
    if len(pay):
        L += ["## Payload per escalated window (bytes)", "", md_table(pay, list(pay.columns), 0)]
    if len(fp):
        L += ["## Model footprint (descriptive; no on-board timing claimed)", "",
              md_table(fp, ["dataset", "pair", "model", "family", "view", "params", "bytes", "macs", "unit"], 0)]
    (out / "REPORT.md").write_text("\n".join(L))
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
