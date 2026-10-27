"""POST-PLAN: tests per replicate, rejection rates with Monte Carlo SEs, ground-truth table.
Post-plan, not pre-specified.

    python src/sens/simulation/summarize.py

Reads results_sens/simulation/reps/*.csv (cells: dataset x split) and truth_draws_*.csv; writes
truth.csv, tests_per_replicate.csv, audit_cells.csv, audit_thresholds.csv and rates.csv.

Tests (alpha = 0.05; no Holm adjustment across tests, since each design is studied on its own):
 H1  per dataset: escalation.stats.nb_test on the split-wise information steps with n_train = mean
     (fit + val), n_test = mean test, one-sided p by stats.one_sided_p (as analyze.h1 /
     per_dataset_effect); pooled: stats.pool_hksj over the 4 datasets, one-sided p (as analyze.h1).
     Verdict SUPPORTED as analyze.verdicts (pooled p < 0.05 and >= 2 datasets with p < 0.05).
     Coverage of the NB 95% CI and of the pooled HKSJ 95% CI for the Monte Carlo true expected step;
     "size at the truth" = one-sided rejection of H0: step <= true step (upper) or >= true step (lower).
 H3  dC = C(R6a) - C(R6b) per split (true value 0 by exchangeability). nb_test with margin 0.02 per
     dataset, pool_hksj(margin = 0.02) -> p_eq; equivalence declared iff p_eq < 0.05. Original
     feasibility rule = analyze.h3 (pooled asymptotic floor (sum 1/(sd_i sqrt(n_test/n_train))^2)^-1/2
     times t_{0.95,3} < 0.02). Corrected rule = src/extra/h3_feasibility.py (per-dataset SE at the
     realised J, sd_i sqrt(1/J + n_test/n_train); smallest attainable HKSJ SE with tau^2 = 0 and
     max(1, q) = 1; feasible iff t_{0.95,3} * that < 0.02). Margin case: dC + 0.02 (true value
     exactly +0.02; TOST size at the boundary).
 H4  original = analyze.h4: MixedLM dcap ~ rho_pos + (1 | dataset), dcap = Cap(R6) - Cap(R1),
     rho_pos on the test fold, one-sided p = Phi(z); SUPPORTED as analyze.verdicts (slope < 0, CI
     excludes 0, p < 0.05). Corrected (results.tex Sec. "H4 is passed by a random router"): rho_pos
     on the validation fold, outcome Cap(R6) - Cap(R0) (= R6's slope minus the R0-null slope on the
     same cells), (a) same MixedLM and verdict rule, (b) two-stage: per-dataset OLS slope with SE
     inflated by sqrt(1 + J n_test/n_train) (src/extra/h4_nulls.py), HKSJ pooling, one-sided p from
     t with 3 df. Diagnostic: the original model with R0 in place of R6.
 H5  original = escalation.audit.run_audit GO. Corrected: fold-averaged validation C above the 95th
     percentile of the pooled null (within-class permutations of the validation gains, same folds;
     one draw per cell, pooled over all cells of a scenario). Outcomes per cell: (i) test C(R6) > 0.5,
     (ii) BA with 10% escalation by R6 > BA(edge), (iii) Cap(R6) > 0.10 (the plan's outcome).
 H2  original = analyze.h2: MixedLM C(R6) ~ log2(positive training windows) + (1 | dataset) over the
     joint-retraining cells (q = 0.1, 0.25, 0.5, 1), one-sided p = 1 - Phi(z); SUPPORTED as
     analyze.verdicts (CI excludes 0, mean C at q = 0.1 <= 0.6, p < 0.05). Corrected: fixed tiers,
     y = C(R6) - C(R1) on x = log2(positive validation windows seen by R6); per split OLS slope over the
     four levels, nb_test per dataset, pool_hksj, one-sided p from t with 3 df (src/extra/h2_extra.py
     two_stage). Supplementary: the level test of C(R6) - C(R1) at q = 1 (same two-stage inference).
MC SEs: sd / sqrt(R) over replicates for per-replicate quantities; replicate bootstrap (2,000
resamples) for ratios (precision, recall, BA of the audit calls).
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from escalation import stats as S                                   # noqa: E402
from escalation.config import H2_LEVELS                              # noqa: E402
import analyze as AN                                                  # noqa: E402

OUT = HERE.parents[2] / "results_sens" / "simulation"
SCEN_ORDER = ["S0", "S1", "S1b", "S2", "S3"]
ALPHA, MARGIN = 0.05, 0.02
LEVELS = sorted(list(H2_LEVELS) + [1.0])


# ------------------------------------------------------------------ truth
def truth_table() -> pd.DataFrame:
    rows = []
    for sc in SCEN_ORDER:
        fn = OUT / f"truth_draws_{sc}.csv"
        if not fn.exists():
            continue
        d = pd.read_csv(fn)
        d["dC_R6_R1"] = d.C_R6 - d.C_R1
        d["dcap_R6_R1"] = d.cap_R6 - d.cap_R1
        d["dC_R6a_R6b"] = d.C_R6a - d.C_R6b
        d["ba10_R6_minus_edge"] = d.ba10_R6 - d.ba_edge
        d["ba10_R1_minus_edge"] = d.ba10_R1 - d.ba_edge
        d["C_R6_gt_half"] = (d.C_R6 > 0.5).astype(float)
        for c in d.columns:
            if c in ("scenario", "draw"):
                continue
            v = d[c].astype(float).dropna()
            rows.append(dict(scenario=sc, quantity=c, mean=v.mean(), sd=v.std(ddof=1), mcse=v.std(ddof=1) / np.sqrt(len(v)),
                             n_draws=len(v)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ per-replicate tests
def nb_ds(v, ntr, nte, margin=None):
    return S.nb_test(np.asarray(v, float), ntr, nte, margin=margin)


def t_onesided(mean, se, df, direction="greater"):
    if not (np.isfinite(mean) and np.isfinite(se) and se > 0):
        return np.nan
    return float(stats.t.sf(mean / se, df) if direction == "greater" else stats.t.cdf(mean / se, df))


def mixed(data, formula, term):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AN.mixed_slope(data, formula, term)


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = (res ** 2).sum() / (len(x) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(b[1]), float(np.sqrt(cov[1, 1]))


def polyslope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 and np.ptp(x[ok]) > 0 else np.nan


def replicate_tests(c: pd.DataFrame, truth: dict) -> dict:
    r = {}
    c = c.sort_values(["dataset", "split"])
    dsets = sorted(c.dataset.unique())
    sizes = {k: (g.n_train.mean(), g.n_test.mean(), g.split.nunique()) for k, g in c.groupby("dataset")}
    tcrit3 = stats.t.ppf(1 - ALPHA, len(dsets) - 1)

    # ---------------- H1
    ti = truth.get("info_step", np.nan); ti8 = truth.get("info_step_fit800", np.nan)
    per = []
    for k in dsets:
        g = c[c.dataset == k]; ntr, nte, J = sizes[k]
        t = nb_ds(g.info_step, ntr, nte)
        p1 = S.one_sided_p(t, "greater")
        tc = stats.t.ppf(0.975, t["J"] - 1)
        per.append(dict(mean=t["mean"], se=t["se"], p1=p1, lo=t["ci_lo"], hi=t["ci_hi"],
                        nlo=t["mean"] - tc * t["se_naive"], nhi=t["mean"] + tc * t["se_naive"],
                        p_up=t_onesided(t["mean"] - ti, t["se"], t["J"] - 1, "greater"),
                        p_dn=t_onesided(t["mean"] - ti, t["se"], t["J"] - 1, "less")))
    per = pd.DataFrame(per)
    pool = S.pool_hksj(per["mean"], per["se"])
    pp = S.one_sided_p(pool, "greater")
    r.update(h1_ds_reject=float((per.p1 < ALPHA).mean()), h1_pool_reject=float(pp < ALPHA),
             h1_verdict_supported=float(pp < ALPHA and (per.p1 < ALPHA).sum() >= 2),
             h1_ds_cover=float(((per.lo <= ti) & (ti <= per.hi)).mean()),
             h1_ds_cover_fit800=float(((per.lo <= ti8) & (ti8 <= per.hi)).mean()),
             h1_ds_cover_naive=float(((per.nlo <= ti) & (ti <= per.nhi)).mean()),
             h1_ds_size_at_truth_upper=float((per.p_up < ALPHA).mean()),
             h1_ds_size_at_truth_lower=float((per.p_dn < ALPHA).mean()),
             h1_pool_cover=float(pool["ci_lo"] <= ti <= pool["ci_hi"]),
             h1_pool_size_at_truth_upper=float(t_onesided(pool["mean"] - ti, pool["se"], pool["df"]) < ALPHA),
             h1_pool_size_at_truth_lower=float(t_onesided(pool["mean"] - ti, pool["se"], pool["df"], "less") < ALPHA),
             h1_pool_mean=pool["mean"], h1_ds_mean=float(per["mean"].mean()))

    # ---------------- H3
    per, perm = [], []
    for k in dsets:
        g = c[c.dataset == k]; ntr, nte, J = sizes[k]
        dC = (g.C_R6a - g.C_R6b).to_numpy()
        t = nb_ds(dC, ntr, nte, MARGIN); tm = nb_ds(dC + MARGIN, ntr, nte, MARGIN)
        pf = S.power_floor(t["sd"], ntr, nte, MARGIN, t["J"])
        per.append(dict(mean=t["mean"], se=t["se"], sd=t["sd"], floor=t["se_floor"], p_eq=t["p_eq"], J=t["J"],
                        feas_orig_ds=pf["feasible"],
                        feas_cor_ds=bool(stats.t.ppf(1 - ALPHA, t["J"] - 1) * t["se"] < MARGIN)))
        perm.append(dict(mean=tm["mean"], se=tm["se"], p_eq=tm["p_eq"]))
    per, perm = pd.DataFrame(per), pd.DataFrame(perm)
    pool = S.pool_hksj(per["mean"], per["se"], margin=MARGIN)
    poolm = S.pool_hksj(perm["mean"], perm["se"], margin=MARGIN)
    floor_p = float(np.sqrt(1.0 / (1.0 / per.floor ** 2).sum()))
    se_min = float(np.sqrt(1.0 / (1.0 / per.se ** 2).sum()))
    feas_o = bool(floor_p * tcrit3 < MARGIN); feas_c = bool(se_min * tcrit3 < MARGIN)
    rej = float(pool["p_eq"] < ALPHA)
    r.update(h3_pool_equiv=rej, h3_feas_orig=float(feas_o), h3_feas_cor=float(feas_c),
             h3_equiv_and_feas_orig=float(rej and feas_o), h3_equiv_and_feas_cor=float(rej and feas_c),
             h3_ds_equiv=float((per.p_eq < ALPHA).mean()), h3_ds_feas_orig=float(per.feas_orig_ds.mean()),
             h3_ds_feas_cor=float(per.feas_cor_ds.mean()),
             h3_margin_pool_equiv=float(poolm["p_eq"] < ALPHA), h3_margin_ds_equiv=float((perm.p_eq < ALPHA).mean()),
             h3_sd_dC=float(per.sd.mean()), h3_pool_se=pool["se"], h3_pool_se_min=se_min, h3_pool_floor=floor_p,
             h3_pool_cover0=float(pool["ci_lo"] <= 0 <= pool["ci_hi"]))

    # ---------------- H4
    c = c.assign(dataset_s=c.dataset.astype(str))
    m = pd.DataFrame(dict(dataset=c.dataset_s, dcap=c.cap_R6 - c.cap_R1, dcap0=c.cap_R0 - c.cap_R1,
                          d60=c.cap_R6 - c.cap_R0, rho_pos=c.rho_pos_test, rho_val=c.rho_pos_val, split=c.split))

    def h4fit(y, x):
        res = mixed(m[["dataset", y, x]].rename(columns={y: "yy", x: "xx"}), "yy ~ xx", "xx")
        p1 = float(stats.norm.cdf(res["z"])) if np.isfinite(res.get("z", np.nan)) else np.nan
        sup = bool(np.isfinite(res["slope"]) and res["slope"] < 0 and res["ci_hi"] < 0 and p1 < ALPHA)
        return res, p1, sup
    o, p1, sup = h4fit("dcap", "rho_pos")
    r.update(h4_orig_p_lt05=float(p1 < ALPHA), h4_orig_supported=float(sup), h4_orig_slope=o["slope"],
             h4_orig_model=o["model"])
    o, p1, sup = h4fit("dcap0", "rho_pos")
    r.update(h4_origR0_p_lt05=float(p1 < ALPHA), h4_origR0_supported=float(sup), h4_origR0_slope=o["slope"])
    o, p1, sup = h4fit("dcap", "rho_val")
    r.update(h4_valrho_only_supported=float(sup))
    o, p1, sup = h4fit("d60", "rho_pos")
    r.update(h4_R0ref_testrho_supported=float(sup))
    o, p1, sup = h4fit("d60", "rho_val")
    r.update(h4_cor_mixed_p_lt05=float(p1 < ALPHA), h4_cor_mixed_supported=float(sup), h4_cor_mixed_slope=o["slope"],
             h4_cor_model=o["model"])
    means, ses = [], []
    for k in dsets:
        g = m[m.dataset == str(k)]; ntr, nte, J = sizes[k]
        b, se = ols(g.rho_val, g.d60)
        means.append(b); ses.append(se * np.sqrt(1.0 + J * nte / ntr))
    pool = S.pool_hksj(means, ses)
    p1 = t_onesided(pool["mean"], pool["se"], pool.get("df") or 3, "less")
    r.update(h4_cor_2stage_supported=float(np.isfinite(p1) and p1 < ALPHA), h4_cor_2stage_slope=pool["mean"],
             h4_rho_val_test_spearman=float(stats.spearmanr(m.rho_val, m.rho_pos, nan_policy="omit").statistic))

    # ---------------- H2 original (joint retraining)
    long = []
    for row in c.itertuples(index=False):
        for q in LEVELS:
            long.append(dict(dataset=str(row.dataset), split=row.split, q=q, C=getattr(row, f"h2j_C_R6_q{q}".replace(".", "_")),
                             npos=getattr(row, f"h2j_npos_q{q}".replace(".", "_"))))
    L = pd.DataFrame(long)
    L["log2_npos"] = np.log2(L.npos.clip(lower=1))
    res = mixed(L[["dataset", "C", "log2_npos"]], "C ~ log2_npos", "log2_npos")
    p1 = float(stats.norm.sf(res["z"])) if np.isfinite(res.get("z", np.nan)) else np.nan
    mean01 = float(L[np.isclose(L.q, 0.1)].C.mean())
    sup = bool(np.isfinite(res["ci_lo"]) and res["ci_lo"] > 0 and mean01 <= 0.6 and p1 < ALPHA)
    r.update(h2_orig_p_lt05=float(p1 < ALPHA), h2_orig_supported=float(sup), h2_orig_slope=res["slope"],
             h2_orig_mean_C_q01=mean01)
    # two-stage on the same joint cells (diagnostic)
    means, ses = [], []
    for k in dsets:
        ntr, nte, J = sizes[k]
        sl = [polyslope(g.log2_npos, g.C) for _, g in L[L.dataset == str(k)].groupby("split")]
        t = nb_ds(sl, ntr, nte); means.append(t["mean"]); ses.append(t["se"])
    pool = S.pool_hksj(means, ses)
    r.update(h2_orig_2stage_reject=float(t_onesided(pool["mean"], pool["se"], pool["df"]) < ALPHA))
    # ---------------- H2 corrected (fixed tiers)
    means, ses, lm, ls = [], [], [], []
    for k in dsets:
        g = c[c.dataset == k]; ntr, nte, J = sizes[k]
        sl = []
        for row in g.itertuples(index=False):
            x = [np.log2(max(getattr(row, f"h2f_npos_val_q{q}".replace(".", "_")), 1)) for q in LEVELS]
            y = [getattr(row, f"h2f_C_R6_q{q}".replace(".", "_")) - row.C_R1 for q in LEVELS]
            sl.append(polyslope(x, y))
        t = nb_ds(sl, ntr, nte); means.append(t["mean"]); ses.append(t["se"])
        t2 = nb_ds(g.C_R6 - g.C_R1, ntr, nte); lm.append(t2["mean"]); ls.append(t2["se"])
    pool = S.pool_hksj(means, ses)
    r.update(h2_cor_reject=float(t_onesided(pool["mean"], pool["se"], pool["df"]) < ALPHA), h2_cor_slope=pool["mean"])
    pool = S.pool_hksj(lm, ls)
    r.update(h2_level_R6_gt_R1_reject=float(t_onesided(pool["mean"], pool["se"], pool["df"]) < ALPHA),
             h2_level_dC=pool["mean"])
    return r


# ------------------------------------------------------------------ audit
def audit_cells(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thr = []
    out = []
    for sc, g in cells.groupby("scenario"):
        g = g.copy()
        null = g["audit_null_C_0"].dropna().to_numpy()
        nullw = g["audit_null_Cw_0"].dropna().to_numpy()
        t95, t95w = np.percentile(null, 95), np.percentile(nullw, 95)
        thr.append(dict(scenario=sc, n_null=len(null), thr95_mean=t95, thr95_wmean=t95w,
                        null_mean=null.mean(), null_sd=null.std(ddof=1),
                        obs_mean=g.audit_cor_C.mean(), orig_C_val_mean=g.audit_C_val.mean()))
        g["go_orig"] = g.audit_go_orig.astype(float)
        g["go_cor"] = (g.audit_cor_C > t95).astype(float)
        g["go_cor_w"] = (g.audit_cor_Cw > t95w).astype(float)
        g["out_C"] = (g.C_R6 > 0.5).astype(float)
        g["out_ba10"] = (g.ba10_R6 > g.ba_edge).astype(float)
        g["out_cap"] = (g.cap_R6 > 0.10).astype(float)
        out.append(g[["scenario", "rep", "dataset", "split", "go_orig", "go_cor", "go_cor_w", "out_C", "out_ba10",
                      "out_cap", "audit_C_val", "audit_C_lo", "audit_cor_C", "audit_null_C_0"]])
    return pd.concat(out, ignore_index=True), pd.DataFrame(thr)


def call_stats(a: pd.DataFrame, call: str, outc: str) -> dict:
    c, o = a[call].to_numpy(), a[outc].to_numpy()
    tp = ((c == 1) & (o == 1)).sum(); fp = ((c == 1) & (o == 0)).sum()
    fn = ((c == 0) & (o == 1)).sum(); tn = ((c == 0) & (o == 0)).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(go_rate=c.mean(), outcome_rate=o.mean(), precision=prec, recall=rec,
                ba=0.5 * (rec + spec) if np.isfinite(rec) and np.isfinite(spec) else np.nan)


def boot_stats(a: pd.DataFrame, call: str, outc: str, B=2000, seed=11):
    """Estimates and replicate-bootstrap SEs (replicates resampled with replacement; the per-replicate
    counts of GO x outcome are summed, which is the same as concatenating the resampled cells)."""
    est = call_stats(a, call, outc)
    c, o = a[call].to_numpy(), a[outc].to_numpy()
    cnt = pd.DataFrame(dict(rep=a.rep.to_numpy(), tp=(c == 1) & (o == 1), fp=(c == 1) & (o == 0),
                            fn=(c == 0) & (o == 1), tn=(c == 0) & (o == 0), go=c, out=o, n=1)).groupby("rep").sum()
    Mx = cnt[["tp", "fp", "fn", "tn", "go", "out", "n"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    tot = Mx[rng.integers(0, len(Mx), size=(B, len(Mx)))].sum(axis=1)
    tp, fp, fn, tn, go, out, n = tot.T
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan); rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
    bs = dict(go_rate=go / n, outcome_rate=out / n, precision=prec, recall=rec, ba=0.5 * (rec + spec))
    return {k: (v, float(np.nanstd(bs[k], ddof=1))) for k, v in est.items()}


# ------------------------------------------------------------------ main
LABELS = {
    "h1_ds_reject": ("H1 information step", "per-dataset NB one-sided", "rejection rate (step > 0)"),
    "h1_pool_reject": ("H1 information step", "pooled HKSJ one-sided", "rejection rate (step > 0)"),
    "h1_verdict_supported": ("H1 information step", "plan verdict rule", "SUPPORTED rate"),
    "h1_ds_cover": ("H1 information step", "per-dataset NB 95% CI", "coverage of true step (fit 600)"),
    "h1_ds_cover_fit800": ("H1 information step", "per-dataset NB 95% CI", "coverage of true step (fit 800)"),
    "h1_ds_cover_naive": ("H1 information step", "per-dataset naive t 95% CI", "coverage of true step (fit 600)"),
    "h1_ds_size_at_truth_upper": ("H1 information step", "per-dataset NB one-sided", "size at truth, upper tail"),
    "h1_ds_size_at_truth_lower": ("H1 information step", "per-dataset NB one-sided", "size at truth, lower tail"),
    "h1_pool_cover": ("H1 information step", "pooled HKSJ 95% CI", "coverage of true step (fit 600)"),
    "h1_pool_size_at_truth_upper": ("H1 information step", "pooled HKSJ one-sided", "size at truth, upper tail"),
    "h1_pool_size_at_truth_lower": ("H1 information step", "pooled HKSJ one-sided", "size at truth, lower tail"),
    "h3_pool_equiv": ("H3 equivalence", "pooled TOST +-0.02 (true dC = 0)", "equivalence declared"),
    "h3_feas_orig": ("H3 equivalence", "original feasibility rule (asymptotic floor)", "predicted feasible"),
    "h3_feas_cor": ("H3 equivalence", "corrected feasibility rule (realised J, HKSJ)", "predicted feasible"),
    "h3_equiv_and_feas_orig": ("H3 equivalence", "pooled TOST", "declared AND original rule feasible"),
    "h3_equiv_and_feas_cor": ("H3 equivalence", "pooled TOST", "declared AND corrected rule feasible"),
    "h3_ds_equiv": ("H3 equivalence", "per-dataset TOST +-0.02 (true dC = 0)", "equivalence declared"),
    "h3_ds_feas_orig": ("H3 equivalence", "per-dataset original rule", "predicted feasible"),
    "h3_ds_feas_cor": ("H3 equivalence", "per-dataset corrected rule", "predicted feasible"),
    "h3_margin_pool_equiv": ("H3 equivalence", "pooled TOST, true dC = +0.02", "equivalence declared (size)"),
    "h3_margin_ds_equiv": ("H3 equivalence", "per-dataset TOST, true dC = +0.02", "equivalence declared (size)"),
    "h3_pool_cover0": ("H3 equivalence", "pooled HKSJ 95% CI", "coverage of true dC = 0"),
    "h4_orig_p_lt05": ("H4", "original (test rho, R6 - R1, MixedLM)", "one-sided p < 0.05"),
    "h4_orig_supported": ("H4", "original (test rho, R6 - R1, MixedLM)", "SUPPORTED rate"),
    "h4_origR0_supported": ("H4", "original with R0 in place of R6", "SUPPORTED rate"),
    "h4_valrho_only_supported": ("H4", "validation rho only (R6 - R1)", "SUPPORTED rate"),
    "h4_R0ref_testrho_supported": ("H4", "R0 reference only (test rho)", "SUPPORTED rate"),
    "h4_cor_mixed_p_lt05": ("H4", "corrected (val rho, R6 - R0, MixedLM)", "one-sided p < 0.05"),
    "h4_cor_mixed_supported": ("H4", "corrected (val rho, R6 - R0, MixedLM)", "SUPPORTED rate"),
    "h4_cor_2stage_supported": ("H4", "corrected (val rho, R6 - R0, two-stage HKSJ)", "one-sided p < 0.05"),
    "h2_orig_p_lt05": ("H2", "original (joint retraining, MixedLM)", "one-sided p < 0.05"),
    "h2_orig_supported": ("H2", "original (joint retraining, MixedLM)", "SUPPORTED rate"),
    "h2_orig_2stage_reject": ("H2", "joint retraining, two-stage HKSJ (diagnostic)", "one-sided p < 0.05"),
    "h2_cor_reject": ("H2", "corrected (fixed tiers, C(R6) - C(R1), two-stage)", "one-sided p < 0.05"),
    "h2_level_R6_gt_R1_reject": ("H2", "level C(R6) - C(R1) at q = 1, two-stage (supplementary)", "one-sided p < 0.05"),
}
MEANS = ["h1_pool_mean", "h1_ds_mean", "h3_sd_dC", "h3_pool_se", "h3_pool_se_min", "h3_pool_floor", "h4_orig_slope",
         "h4_origR0_slope", "h4_cor_mixed_slope", "h4_cor_2stage_slope", "h4_rho_val_test_spearman", "h2_orig_slope",
         "h2_orig_mean_C_q01", "h2_cor_slope", "h2_level_dC"]


def main():
    tt = truth_table()
    tt.to_csv(OUT / "truth.csv", index=False)
    truth = {sc: dict(zip(g.quantity, g["mean"])) for sc, g in tt.groupby("scenario")} if len(tt) else {}
    files = sorted((OUT / "reps").glob("*.csv"))
    cells = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cells.columns = [col.replace(".", "_") if col.startswith("h2") else col for col in cells.columns]
    # per-replicate tests are cached (keyed by scenario, replicate and the truth values they use)
    cache_fn = OUT / "tests_per_replicate.csv"
    cache = pd.read_csv(cache_fn) if cache_fn.exists() else pd.DataFrame(columns=["scenario", "rep", "truth_tag"])
    cached = {(r.scenario, int(r.rep), r.truth_tag): r._asdict() for r in cache.itertuples(index=False)} \
        if "truth_tag" in cache.columns and len(cache) else {}
    rows = []
    for (sc, rep), c in cells.groupby(["scenario", "rep"]):
        tr = truth.get(sc, {})
        tag = f"{tr.get('info_step', np.nan):.10f}|{tr.get('info_step_fit800', np.nan):.10f}"
        hit = cached.get((sc, int(rep), tag))
        rows.append(hit if hit is not None else dict(scenario=sc, rep=rep, truth_tag=tag, **replicate_tests(c, tr)))
    T = pd.DataFrame(rows)
    T.to_csv(cache_fn.with_suffix(".tmp"), index=False); cache_fn.with_suffix(".tmp").replace(cache_fn)
    rates = []
    for sc, g in T.groupby("scenario"):
        R = len(g)
        for k, (test, design, qty) in LABELS.items():
            v = g[k].astype(float).dropna()
            rates.append(dict(test=test, design=design, quantity=qty, scenario=sc, rate=v.mean(),
                              mcse=v.std(ddof=1) / np.sqrt(len(v)), n_rep=len(v), key=k))
        for k in MEANS:
            v = pd.to_numeric(g[k], errors="coerce").dropna()
            rates.append(dict(test="descriptive", design=k, quantity="mean over replicates", scenario=sc,
                              rate=v.mean(), mcse=v.std(ddof=1) / np.sqrt(len(v)), n_rep=len(v), key=k))
        for cond, lab in (("h3_feas_orig", "original rule feasible"), ("h3_feas_cor", "corrected rule feasible")):
            v = g.loc[g[cond] == 1, "h3_pool_equiv"].astype(float)
            rates.append(dict(test="H3 equivalence", design="pooled TOST +-0.02 (true dC = 0)",
                              quantity=f"equivalence declared | {lab}", scenario=sc, rate=v.mean() if len(v) else np.nan,
                              mcse=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan, n_rep=len(v),
                              key=f"h3_equiv_given_{cond}"))
        v = ((g.h3_feas_orig == 1) & (g.h3_feas_cor == 0)).astype(float)
        rates.append(dict(test="H3 equivalence", design="feasibility rules", quantity="original feasible, corrected not",
                          scenario=sc, rate=v.mean(), mcse=v.std(ddof=1) / np.sqrt(len(v)), n_rep=len(v),
                          key="h3_orig_not_cor"))
        rates.append(dict(test="H4", design="MixedLM fallback to OLS", quantity="share of replicates", scenario=sc,
                          rate=float((g.h4_orig_model != "MixedLM").mean()), mcse=np.nan, n_rep=R, key="h4_ols_fallback"))
    A, thr = audit_cells(cells)
    A.to_csv(OUT / "audit_cells.csv", index=False)
    thr.to_csv(OUT / "audit_thresholds.csv", index=False)
    for sc, a in A.groupby("scenario"):
        R = a.rep.nunique()
        for call, design in (("go_orig", "original (GroupKFold, pooled OOF C, bootstrap LB > 0.52)"),
                             ("go_cor", "corrected (SGKF, fold-mean C > pooled within-class null 95th pct)"),
                             ("go_cor_w", "corrected, D-weighted fold mean (variant)")):
            for outc, lab in (("out_C", "(i) test C(R6) > 0.5"), ("out_ba10", "(ii) BA at 10% by R6 > edge"),
                              ("out_cap", "(iii) Cap(R6) > 0.10")):
                bs = boot_stats(a, call, outc)
                for qn, (v, se) in bs.items():
                    if qn == "go_rate" and outc != "out_C":
                        continue
                    nm = qn if qn in ("go_rate",) else f"{qn} vs {lab}"
                    if qn == "outcome_rate":
                        if call != "go_orig":
                            continue
                        nm = f"base rate of {lab}"
                    rates.append(dict(test="H5 audit", design=design, quantity=nm, scenario=sc, rate=v, mcse=se,
                                      n_rep=R, key=f"{call}|{qn}|{outc}"))
    out = pd.DataFrame(rates)
    out["scenario"] = pd.Categorical(out.scenario, SCEN_ORDER, ordered=True)
    out = out.sort_values(["test", "design", "quantity", "scenario"])
    out.to_csv(OUT / "rates.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 500)
    wide = out.pivot_table(index=["test", "design", "quantity"], columns="scenario", values="rate", observed=True)
    print(wide.round(3).to_string())
    print(thr.round(4).to_string())


if __name__ == "__main__":
    main()
