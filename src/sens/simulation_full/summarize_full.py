"""POST-PLAN: tests per replicate, rejection rates with Monte Carlo SEs, audit calls and ground truth
of the full validation. Post-plan, not pre-specified.

    OMP_NUM_THREADS=1 python src/sens/simulation_full/summarize_full.py --out results_sens/simulation_full --procs N [--boot 2000]

Reads <out>/reps/*.csv (cells: dataset x split, written by worker.py) and <out>/truth/*.csv (truth draws, chunks
concatenated per scenario); writes into <out>/summary/: truth.csv, n_rep.csv, tests_per_replicate.csv (cache keyed
by scenario, replicate and the truth values used), audit_cells.csv, audit_thresholds.csv, rates.csv, tables.md.
Works on partial results (any subset of scenarios and replicates); n_rep is reported per scenario. Replicates with
fewer than 4 datasets or missing columns are skipped and counted (n_rep.csv). Split rows flagged `skipped` by
make_splits are dropped before testing. alpha = 0.05 throughout; MCSE = sd / sqrt(R) over replicates for
per-replicate quantities, replicate bootstrap (--boot resamples) for the audit ratios. Every rate whose count is 0
gets the exact one-sided 95% Clopper-Pearson upper bound 1 - 0.05^(1/n) (column upper95_if_zero), with n = number of
replicates (pooled tests), of datasets (per-dataset tests: datasets are independent within a replicate) or, for audit
GO rates, of (replicate, dataset) units (zero GO cells implies zero datasets with a GO, so the dataset-level bound is a
valid bound for the cell-level rate despite dependence between the splits of a dataset).

Base family (S0, S1, S1b, S2, S3; J = 20 splits, 4 repeats of K = 5):
 H1-H4  src/sens/simulation/summarize.replicate_tests UNCHANGED (imported as SM), plus
 H1     share of splits with info_step > 0.
 H4     corrected two-stage test of the per-dataset slope of d60 = Cap(R6) - Cap(R0) on rho_val = rho_pos_val
        (H0 slope >= 0, one-sided "less"; HKSJ pooling over the 4 datasets, t with k - 1 df), four variance options:
          factor        OLS slope, SE x sqrt(1 + J n_test/n_train)  (= SM's h4_cor_2stage; recomputed so all sit together)
          nofactor      OLS slope, plain OLS SE
          repeat_naive  per repeat the OLS slope over its K splits; estimate = mean of the R repeat slopes,
                        SE = sd(repeat slopes) / sqrt(R)
          repeat_nb     same estimate, SE = sd(repeat slopes) sqrt(1/R + n_test/n_train)  (slope-specific NB correction)
        each: pooled rejection, pooled slope and SE, per-dataset one-sided rejection (t with n - 2 df for the OLS options,
        R - 1 df for the repeat options).
 H5     audit, cell level (cell = dataset x split). original: audit_go_orig. corrected, per-dataset null (the design
        under validation, as src/sens/designs/p1w_within_class.py): threshold = 95th percentile of the finite
        values audit_null_C_0..9 of the same (replicate, dataset), pooled over its 20 splits; GO = audit_cor_C >
        threshold. D-weighted variant with audit_cor_Cw / audit_null_Cw_*. Pilot (oracle) variant: threshold pooled over
        all null values of the scenario. Undefined cells (audit_cor_C NaN or no finite null): counted; every corrected
        variant is scored with undefined = NO-GO (the real-data rule) and over defined cells only. Outcomes and call
        statistics: SM.audit_cells outcomes (C_R6 > 0.5, ba10_R6 > ba_edge, cap_R6 > 0.10), SM.call_stats/boot_stats.
 Truth  SM.truth_table's derived quantities on the concatenated truth chunks.

T and C families (4 datasets per replicate; n_train, n_test = per-dataset means as SM), per design tag ("" plain;
C also "_z" within-file z-scored = the declared variant, "_aug" shortcut variables appended):
 H1     per dataset escalation.stats.nb_test on info_step{tag}: one-sided p (greater) < 0.05, NB 95% CI, naive t CI
        (mean +- t_{0.975,J-1} se_naive); pooled stats.pool_hksj, one-sided rejection, plan verdict (pooled p < 0.05 and
        >= 2 datasets with p < 0.05). Coverage of the NB, naive and pooled CIs for two targets: (A) deployment truth =
        scenario mean of the truth draws' info_step{tag} (truth_full.py), (B) cross-validated expectation = mean over all
        replicates and datasets of the scenario of the per-dataset mean of info_step{tag} (computed before the
        per-replicate pass; in truth.csv as cv_expectation_info_step{tag}). One-sided size at the target (upper tail,
        H0 step <= target, as SM's h1_ds_size_at_truth_upper), per dataset and pooled. Share of splits with step > 0.
 Shortcuts  per dataset the best shortcut = the ba_sc_* column with the highest mean over the dataset's splits;
        (ii) d2 = ba_cloud{tag} - ba_best per split, nb_test, "cloud beats the best shortcut" iff ci_lo > 0 (plain; C also
        z), "exposed" = not beating it; C (iii) d3 = ba_cloud_aug - max(ba_edge_aug, ba_best) per split, beats iff ci_lo > 0.
 Survival   the paper's Sec. 5.3 rule per dataset. T: plain info_step CI lo > 0 AND (ii) plain. C: info_step_z CI lo > 0
        AND (ii) on z AND (iii). Each condition also on its own; rates = share of the 4 datasets, averaged over replicates.
NaN handling as SM: an undefined test, CI or condition counts as not rejected / not covered / not met. Departures:
a per-dataset one-sided p is NaN (not 0) when the NB se is undefined (J < 2; escalation.stats.one_sided_p would return
0 there), and a pooled test over fewer than 2 datasets is undefined (not a z-test on one dataset); neither occurs
with J = 20 and 4 datasets.
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                     # src/
sys.path.insert(0, str(HERE.parent / "simulation"))          # src/sens/simulation
sys.path.insert(0, str(HERE))
import summarize as SM                                        # noqa: E402  (pilot summarizer; reused)
from escalation import stats as S                             # noqa: E402
from gen_full import family                                   # noqa: E402

DEFAULT_OUT = HERE.parents[2] / "results_sens" / "simulation_full"
SCEN_ORDER = ["S0", "S1", "S1b", "S2", "S3", "T0_0.0", "T0_0.3", "T0_0.6", "T0_0.9", "T0s_0.9",
              "T1_0.0", "T1_0.3", "T1_0.6", "T1_0.9", "C_ref", "C_clock", "C_offset"]
ALPHA = SM.ALPHA
N_DATASETS = 4
TAGS = {"T": [""], "C": ["", "_z", "_aug"]}
TAG_NAME = {"": "plain", "_z": "z (within-file z-scored, declared)", "_aug": "aug (shortcut variables appended)"}
H4_OPTS = {"factor": "OLS SE x sqrt(1 + J n_test/n_train) (published factor)",
           "nofactor": "plain OLS SE (no factor)",
           "repeat_naive": "repeat slopes, SE = sd / sqrt(R)",
           "repeat_nb": "repeat slopes, SE = sd sqrt(1/R + n_test/n_train) (slope-specific NB)"}
BASE_REQ = ["dataset", "split", "repeat", "n_train", "n_test", "info_step", "C_R6a", "C_R6b", "cap_R0", "cap_R1",
            "cap_R6", "C_R1", "C_R6", "ba_edge", "ba10_R6", "rho_pos_test", "rho_pos_val", "audit_go_orig",
            "audit_C_val", "audit_C_lo", "audit_cor_C", "audit_cor_Cw", "audit_null_C_0", "audit_null_Cw_0"]
TC_REQ = ["dataset", "split", "n_train", "n_test", "ba_edge", "ba_cloud", "info_step"]
C_REQ = TC_REQ + ["ba_edge_z", "ba_cloud_z", "info_step_z", "ba_edge_aug", "ba_cloud_aug", "info_step_aug"]


def ind(x) -> float:
    """1.0 if x is True, 0.0 if False or undefined (SM's convention: undefined = not rejected / not covered)."""
    return float(bool(x)) if x is not None else 0.0


def p_greater(t: dict) -> float:
    """stats.one_sided_p, but NaN when the NB se is undefined (J < 2)."""
    return S.one_sided_p(t, "greater") if np.isfinite(t.get("se", np.nan)) else np.nan


def pool_p(pool: dict, direction: str = "greater") -> float:
    """One-sided p of a pool_hksj result (t, k - 1 df); undefined for k < 2."""
    if pool.get("df") is None or not np.isfinite(pool.get("se", np.nan)):
        return np.nan
    return SM.t_onesided(pool["mean"], pool["se"], pool["df"], direction)


def pool_t(pool: dict, target: float) -> float:
    if pool.get("df") is None:
        return np.nan
    return SM.t_onesided(pool["mean"] - target, pool["se"], pool["df"], "greater")


def lt(p) -> float:
    return float(np.isfinite(p) and p < ALPHA)


# ------------------------------------------------------------------ truth
def load_truth(out: Path) -> dict[str, pd.DataFrame]:
    parts: dict[str, list] = {}
    for fn in sorted((out / "truth").glob("*_c*.csv")):
        sc = re.sub(r"_c\d+$", "", fn.stem)
        parts.setdefault(sc, []).append(pd.read_csv(fn))
    return {sc: pd.concat(v, ignore_index=True).drop_duplicates("draw", keep="last").sort_values("draw")
            for sc, v in parts.items()}


def base_truth_derived(d: pd.DataFrame) -> pd.DataFrame:
    """The derived quantities of SM.truth_table."""
    d = d.copy()
    d["dC_R6_R1"] = d.C_R6 - d.C_R1
    d["dcap_R6_R1"] = d.cap_R6 - d.cap_R1
    d["dC_R6a_R6b"] = d.C_R6a - d.C_R6b
    d["ba10_R6_minus_edge"] = d.ba10_R6 - d.ba_edge
    d["ba10_R1_minus_edge"] = d.ba10_R1 - d.ba_edge
    d["C_R6_gt_half"] = (d.C_R6 > 0.5).astype(float)
    return d


def summ(sc, qty, v) -> dict:
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna().astype(float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else np.nan
    return dict(scenario=sc, quantity=qty, mean=v.mean() if n else np.nan, sd=sd, mcse=sd / np.sqrt(n) if n > 1 else np.nan,
                n_draws=n)


def truth_rows(draws: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for sc in [s for s in SCEN_ORDER if s in draws] + sorted(set(draws) - set(SCEN_ORDER)):
        d = draws[sc]
        if family(sc) == "base":
            d = base_truth_derived(d)
        for c in d.columns:
            if c in ("scenario", "draw", "split") or not pd.api.types.is_numeric_dtype(d[c]):
                continue
            rows.append(summ(sc, c, d[c]))
    return rows


# ------------------------------------------------------------------ reading replicates
def read_rep(fn: str):
    try:
        d = pd.read_csv(fn)
        d.columns = [c.replace(".", "_") if c.startswith("h2") else c for c in d.columns]
        return fn, d, None
    except Exception as e:                                   # noqa: BLE001
        return fn, None, repr(e)


def check_rep(sc: str, c: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Drop skipped splits; return (cells, None) or (None, reason)."""
    if "skipped" in c.columns:
        c = c[c["skipped"].isna()].drop(columns="skipped")
    fam = family(sc)
    req = BASE_REQ if fam == "base" else (C_REQ if fam == "C" else TC_REQ)
    miss = [k for k in req if k not in c.columns]
    if miss:
        return None, "missing columns: " + ",".join(miss[:5])
    if fam != "base" and not any(k.startswith("ba_sc_") for k in c.columns):
        return None, "missing shortcut columns"
    if c.dataset.nunique() < N_DATASETS:
        return None, f"{c.dataset.nunique()} datasets"
    return c, None


def cv_expectations(reps: dict) -> tuple[dict, list[dict]]:
    """(B) target: mean over all replicates and datasets of a scenario of the per-dataset mean step."""
    by = {}
    for (sc, rep), c in reps.items():
        fam = family(sc)
        if fam == "base":
            continue
        for tag in TAGS[fam]:
            by.setdefault((sc, tag), []).extend(c.groupby("dataset")[f"info_step{tag}"].mean().tolist())
    cv, rows = {}, []
    for (sc, tag), v in by.items():
        rr = summ(sc, f"cv_expectation_info_step{tag}", v)
        cv.setdefault(sc, {})[tag] = rr["mean"]
        rows.append(rr)
    return cv, rows


# ------------------------------------------------------------------ base family: extra tests
def h4_dataset(g: pd.DataFrame, ntr: float, nte: float, J: int) -> dict:
    """Per-dataset slope of d60 on rho_val under the four variance options: {opt: (estimate, se, df)}."""
    x = g.rho_pos_val.to_numpy(float); y = (g.cap_R6 - g.cap_R0).to_numpy(float)
    n = int((np.isfinite(x) & np.isfinite(y)).sum())
    b, se = SM.ols(x, y)
    out = {"factor": (b, se * np.sqrt(1.0 + J * nte / ntr), n - 2), "nofactor": (b, se, n - 2)}
    sl = []
    for _, gr in g.groupby("repeat"):
        s, _ = SM.ols(gr.rho_pos_val, gr.cap_R6 - gr.cap_R0)
        if np.isfinite(s):
            sl.append(s)
    R = len(sl)
    if R >= 2:
        m, sd = float(np.mean(sl)), float(np.std(sl, ddof=1))
        out["repeat_naive"] = (m, sd / np.sqrt(R), R - 1)
        out["repeat_nb"] = (m, sd * np.sqrt(1.0 / R + nte / ntr), R - 1)
    else:
        out["repeat_naive"] = out["repeat_nb"] = (np.nan, np.nan, 0)
    return out


def h4_options(c: pd.DataFrame) -> dict:
    r = {}
    per = {o: [] for o in H4_OPTS}
    for k, g in c.groupby("dataset"):
        res = h4_dataset(g, g.n_train.mean(), g.n_test.mean(), g.split.nunique())
        for o, v in res.items():
            per[o].append(v)
    for o, v in per.items():
        b = np.array([e[0] for e in v], float); se = np.array([e[1] for e in v], float)
        p_ds = [SM.t_onesided(e[0], e[1], e[2], "less") if e[2] > 0 else np.nan for e in v]
        pool = S.pool_hksj(b, se)
        r[f"h4_2s_{o}_reject"] = lt(pool_p(pool, "less"))
        r[f"h4_2s_{o}_ds_reject"] = float(np.mean([lt(p) for p in p_ds]))
        r[f"h4_2s_{o}_slope"] = pool["mean"]
        r[f"h4_2s_{o}_se"] = pool["se"]
        r[f"h4_2s_{o}_ds_se"] = float(np.nanmean(se)) if np.isfinite(se).any() else np.nan
    return r


def base_tests(c: pd.DataFrame, truth: dict) -> dict:
    r = SM.replicate_tests(c, truth)
    for tq, keys in (("info_step", ("h1_ds_cover", "h1_ds_cover_naive", "h1_pool_cover", "size_at_truth")),
                     ("info_step_fit800", ("h1_ds_cover_fit800",))):
        if not np.isfinite(truth.get(tq, np.nan)):           # no truth yet: undefined, not 0
            for k in r:
                if any(k == s or (s == "size_at_truth" and s in k) for s in keys):
                    r[k] = np.nan
    r.update(h4_options(c))
    v = c.info_step.to_numpy(float); v = v[np.isfinite(v)]
    r["h1_share_pos"] = float((v > 0).mean()) if len(v) else np.nan
    r["n_ds"] = c.dataset.nunique()
    return r


# ------------------------------------------------------------------ T and C families
def tc_tests(c: pd.DataFrame, fam: str, truth: dict, cv: dict) -> dict:
    r = {}
    c = c.sort_values(["dataset", "split"])
    dsets = sorted(c.dataset.unique())
    G = {k: c[c.dataset == k] for k in dsets}
    sizes = {k: (g.n_train.mean(), g.n_test.mean()) for k, g in G.items()}
    lo_by_tag = {}
    for tag in TAGS[fam]:
        p = "h1" + tag
        col = f"info_step{tag}"
        tA, tB = truth.get(col, np.nan), cv.get(tag, np.nan)
        per = []
        for k in dsets:
            ntr, nte = sizes[k]
            t = SM.nb_ds(G[k][col], ntr, nte)
            df = t["J"] - 1
            tc = stats.t.ppf(0.975, df) if df > 0 else np.nan
            se = t["se"]
            per.append(dict(mean=t["mean"], se=se, sd=t.get("sd", np.nan), se_naive=t["se_naive"], p1=p_greater(t),
                            lo=t["ci_lo"], hi=t["ci_hi"], nlo=t["mean"] - tc * t["se_naive"],
                            nhi=t["mean"] + tc * t["se_naive"],
                            pA=SM.t_onesided(t["mean"] - tA, se, df) if df > 0 else np.nan,
                            pB=SM.t_onesided(t["mean"] - tB, se, df) if df > 0 else np.nan))
        per = pd.DataFrame(per)
        lo_by_tag[tag] = per.lo.to_numpy()
        pool = S.pool_hksj(per["mean"], per["se"])
        pp = pool_p(pool)
        cov = lambda lo, hi, x: float(np.mean([(a <= x <= b) for a, b in zip(lo, hi)]))  # noqa: E731
        plo, phi = pool.get("ci_lo", np.nan), pool.get("ci_hi", np.nan)
        v = c[col].to_numpy(float); v = v[np.isfinite(v)]
        r.update({f"{p}_ds_reject": float(np.mean([lt(x) for x in per.p1])), f"{p}_pool_reject": lt(pp),
                  f"{p}_verdict_supported": float(lt(pp) == 1 and sum(lt(x) for x in per.p1) >= 2),
                  f"{p}_ds_cover_truth": cov(per.lo, per.hi, tA), f"{p}_ds_cover_cv": cov(per.lo, per.hi, tB),
                  f"{p}_ds_cover_naive_truth": cov(per.nlo, per.nhi, tA),
                  f"{p}_ds_cover_naive_cv": cov(per.nlo, per.nhi, tB),
                  f"{p}_pool_cover_truth": float(plo <= tA <= phi), f"{p}_pool_cover_cv": float(plo <= tB <= phi),
                  f"{p}_ds_size_truth_upper": float(np.mean([lt(x) for x in per.pA])),
                  f"{p}_ds_size_cv_upper": float(np.mean([lt(x) for x in per.pB])),
                  f"{p}_pool_size_truth_upper": lt(pool_t(pool, tA)), f"{p}_pool_size_cv_upper": lt(pool_t(pool, tB)),
                  f"{p}_share_pos": float((v > 0).mean()) if len(v) else np.nan,
                  f"{p}_pool_mean": pool["mean"], f"{p}_pool_se": pool["se"], f"{p}_ds_mean": float(per["mean"].mean()),
                  f"{p}_ds_se": float(per.se.mean()), f"{p}_ds_se_naive": float(per.se_naive.mean()),
                  f"{p}_ds_sd": float(per.sd.mean()),
                  f"ba_edge{tag}": float(c[f"ba_edge{tag}"].mean()), f"ba_cloud{tag}": float(c[f"ba_cloud{tag}"].mean())})
        for nm, target in (("truth", tA), ("cv", tB)):         # no target yet (e.g. no truth draws): undefined, not 0
            if not np.isfinite(target):
                for k in [k for k in r if k.startswith(p + "_") and f"_{nm}" in k]:
                    r[k] = np.nan
    # ---------------- shortcut baselines and the survival rule
    sc_cols = sorted(k for k in c.columns if k.startswith("ba_sc_"))
    rows = []
    for i, k in enumerate(dsets):
        g = G[k]; ntr, nte = sizes[k]
        means = g[sc_cols].mean()
        if means.notna().sum() == 0:
            rows.append(dict(best=None)); continue
        best = means.idxmax(); bb = g[best].to_numpy(float)

        def beat(diff):
            lo = SM.nb_ds(diff, ntr, nte)["ci_lo"]
            return ind(np.isfinite(lo) and lo > 0)
        d = dict(best=best, ba_best=float(np.nanmean(bb)), beats_plain=beat(g.ba_cloud.to_numpy(float) - bb))
        if fam == "C":
            d["beats_z"] = beat(g.ba_cloud_z.to_numpy(float) - bb)
            d["beats_aug"] = beat(g.ba_cloud_aug.to_numpy(float) - np.maximum(g.ba_edge_aug.to_numpy(float), bb))
            d["cond_step"] = ind(np.isfinite(lo_by_tag["_z"][i]) and lo_by_tag["_z"][i] > 0)
            d["cond_sc"] = d["beats_z"]; d["cond_aug"] = d["beats_aug"]
            d["survive"] = float(d["cond_step"] and d["cond_sc"] and d["cond_aug"])
        else:
            d["cond_step"] = ind(np.isfinite(lo_by_tag[""][i]) and lo_by_tag[""][i] > 0)
            d["cond_sc"] = d["beats_plain"]
            d["survive"] = float(d["cond_step"] and d["cond_sc"])
        rows.append(d)
    D = pd.DataFrame(rows)
    nd = len(dsets)
    share = lambda col: float(D[col].fillna(0).sum() / nd) if col in D else np.nan  # noqa: E731
    r.update(sc_beats_plain=share("beats_plain"), sc_exposed_plain=1.0 - share("beats_plain"),
             surv_cond_step=share("cond_step"), surv_cond_sc=share("cond_sc"), survive=share("survive"),
             sc_ba_best=float(D.ba_best.mean()) if "ba_best" in D else np.nan)
    if fam == "C":
        r.update(sc_beats_z=share("beats_z"), sc_exposed_z=1.0 - share("beats_z"), sc_aug_beats=share("beats_aug"),
                 surv_cond_aug=share("cond_aug"))
    for col in sc_cols:
        r[col] = float(c[col].mean())
        r[f"sc_best_is_{col[6:]}"] = float((D.get("best") == col).sum() / nd) if "best" in D else np.nan
    r["n_ds"] = nd
    return r


# ------------------------------------------------------------------ per-replicate pass (parallel, cached)
_G: dict = {}


def truth_tag(sc: str, truth: dict, cv: dict) -> str:
    if family(sc) == "base":
        return f"{truth.get('info_step', np.nan):.10f}|{truth.get('info_step_fit800', np.nan):.10f}"
    tags = TAGS[family(sc)]
    return "|".join([f"{truth.get('info_step' + t, np.nan):.10f}" for t in tags]
                    + [f"{cv.get(t, np.nan):.10f}" for t in tags])


def run_one(key):
    sc, rep = key
    c = _G["reps"][key]
    fam = family(sc)
    t0 = time.process_time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                if fam == "base":
                    r = base_tests(c, _G["truth"].get(sc, {}))
                else:
                    r = tc_tests(c, fam, _G["truth"].get(sc, {}), _G["cv"].get(sc, {}))
        return key, r, None, time.process_time() - t0
    except Exception as e:                                   # noqa: BLE001
        return key, None, repr(e)[:300], time.process_time() - t0


def per_replicate(reps: dict, truth: dict, cv: dict, cache_fn: Path, procs: int) -> tuple[pd.DataFrame, dict]:
    cached = {}
    if cache_fn.exists():
        cache = pd.read_csv(cache_fn, dtype={"truth_tag": str}, low_memory=False)
        cached = {(r["scenario"], int(r["rep"]), r["truth_tag"]): r for r in cache.to_dict("records")}
    rows, todo, tags = [], [], {}
    for key in reps:
        sc, rep = key
        tags[key] = truth_tag(sc, truth.get(sc, {}), cv.get(sc, {}))
        hit = cached.get((sc, rep, tags[key]))
        if hit is not None:
            rows.append(hit)
        else:
            todo.append(key)
    todo.sort(key=lambda k: (family(k[0]) != "base", SCEN_ORDER.index(k[0]) if k[0] in SCEN_ORDER else 99, k[1]))
    print(f"per-replicate tests: {len(rows)} cached, {len(todo)} to compute", flush=True)
    errors = {}

    def flush():
        out = pd.DataFrame(rows)
        tmp = cache_fn.with_name(cache_fn.name + ".tmp")
        out.to_csv(tmp, index=False); os.replace(tmp, cache_fn)
        return out

    _G.update(reps=reps, truth=truth, cv=cv)
    if todo:
        pool = mp.get_context("fork").Pool(procs) if procs > 1 else None
        it = pool.imap_unordered(run_one, todo, chunksize=1) if pool else map(run_one, todo)
        t0 = last = time.time()
        for n, (key, r, err, cpu) in enumerate(it, 1):
            if err is not None:
                errors[key] = err
            else:
                rows.append(dict(scenario=key[0], rep=key[1], truth_tag=tags[key], cpu_s=round(cpu, 3), **r))
            if time.time() - last > 60 or n == len(todo):
                flush(); last = time.time()
                print(f"  [{n}/{len(todo)} {time.time() - t0:.0f}s]", flush=True)
        if pool:
            pool.close(); pool.join()
    T = flush() if rows else pd.DataFrame(columns=["scenario", "rep", "truth_tag"])
    return T, errors


# ------------------------------------------------------------------ audit (base family)
def null_cols(cols, w=False) -> list[str]:
    pat = re.compile(r"audit_null_Cw_(\d+)$" if w else r"audit_null_C_(\d+)$")
    return sorted([c for c in cols if pat.match(c)], key=lambda c: int(pat.match(c).group(1)))


def pct95(v) -> float:
    v = np.asarray(v, float).ravel(); v = v[np.isfinite(v)]
    return float(np.percentile(v, 95)) if len(v) else np.nan


def audit_full(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, thr = [], []
    nc, ncw = null_cols(cells.columns), null_cols(cells.columns, w=True)
    for sc, g in cells.groupby("scenario", sort=False):
        g = g.reset_index(drop=True).copy()
        N, Nw = g[nc].to_numpy(float), g[ncw].to_numpy(float)
        t_sc, t_scw = pct95(N), pct95(Nw)
        t_ds, t_dsw = np.full(len(g), np.nan), np.full(len(g), np.nan)
        for _, ix in g.groupby(["rep", "dataset"]).indices.items():
            t_ds[ix], t_dsw[ix] = pct95(N[ix]), pct95(Nw[ix])
        obs, obsw = g.audit_cor_C.to_numpy(float), g.audit_cor_Cw.to_numpy(float)
        g["thr_ds"], g["thr_dsw"] = t_ds, t_dsw
        g["def_ds"] = (np.isfinite(obs) & np.isfinite(t_ds)).astype(float)
        g["def_dsw"] = (np.isfinite(obsw) & np.isfinite(t_dsw)).astype(float)
        g["def_sc"] = (np.isfinite(obs) & np.isfinite(t_sc)).astype(float)
        g["def_scw"] = (np.isfinite(obsw) & np.isfinite(t_scw)).astype(float)
        with np.errstate(invalid="ignore"):
            g["go_orig"] = g.audit_go_orig.fillna(0).astype(float)
            g["go_ds"] = ((g.def_ds == 1) & (obs > t_ds)).astype(float)
            g["go_dsw"] = ((g.def_dsw == 1) & (obsw > t_dsw)).astype(float)
            g["go_sc"] = ((g.def_sc == 1) & (obs > t_sc)).astype(float)
            g["go_scw"] = ((g.def_scw == 1) & (obsw > t_scw)).astype(float)
            g["out_C"] = (g.C_R6 > 0.5).astype(float)
            g["out_ba10"] = (g.ba10_R6 > g.ba_edge).astype(float)
            g["out_cap"] = (g.cap_R6 > 0.10).astype(float)
        fin = N[np.isfinite(N)]
        ud = g.groupby(["rep", "dataset"]).thr_ds.first()
        thr.append(dict(scenario=sc, n_rep=g.rep.nunique(), n_cells=len(g), n_null=len(fin),
                        null_mean=fin.mean() if len(fin) else np.nan,
                        null_sd=fin.std(ddof=1) if len(fin) > 1 else np.nan,
                        thr95_scenario=t_sc, thr95w_scenario=t_scw,
                        thr95_ds_mean=float(np.nanmean(ud)) if ud.notna().any() else np.nan,
                        thr95_ds_sd=float(np.nanstd(ud, ddof=1)) if ud.notna().sum() > 1 else np.nan,
                        thr95_ds_p05=float(np.nanpercentile(ud, 5)) if ud.notna().any() else np.nan,
                        thr95_ds_p95=float(np.nanpercentile(ud, 95)) if ud.notna().any() else np.nan,
                        obs_mean=float(np.nanmean(obs)) if np.isfinite(obs).any() else np.nan,
                        obs_w_mean=float(np.nanmean(obsw)) if np.isfinite(obsw).any() else np.nan,
                        orig_C_val_mean=g.audit_C_val.mean(),
                        n_undef_ds=int((g.def_ds == 0).sum()), share_undef_ds=float((g.def_ds == 0).mean()),
                        n_undef_dsw=int((g.def_dsw == 0).sum()), share_undef_dsw=float((g.def_dsw == 0).mean()),
                        n_undef_sc=int((g.def_sc == 0).sum()), share_undef_sc=float((g.def_sc == 0).mean())))
        out.append(g[["scenario", "rep", "dataset", "split", "go_orig", "go_ds", "go_dsw", "go_sc", "go_scw", "def_ds",
                      "def_dsw", "def_sc", "def_scw", "out_C", "out_ba10", "out_cap", "audit_C_val", "audit_C_lo",
                      "audit_cor_C", "audit_cor_Cw", "thr_ds", "thr_dsw"]])
    return pd.concat(out, ignore_index=True), pd.DataFrame(thr)


AUDIT_CALLS = [  # (call, defined-cells column or None, design label)
    ("go_orig", None, "original (GroupKFold, pooled OOF C, bootstrap LB > 0.52)"),
    ("go_ds", None, "corrected, per-dataset null 95th pct (undefined = NO-GO) [design under validation]"),
    ("go_ds", "def_ds", "corrected, per-dataset null 95th pct (defined cells only)"),
    ("go_dsw", None, "corrected D-weighted, per-dataset null (undefined = NO-GO)"),
    ("go_dsw", "def_dsw", "corrected D-weighted, per-dataset null (defined cells only)"),
    ("go_sc", None, "corrected, null pooled per scenario (pilot oracle; undefined = NO-GO)"),
    ("go_sc", "def_sc", "corrected, null pooled per scenario (pilot oracle; defined cells only)"),
    ("go_scw", None, "corrected D-weighted, null pooled per scenario (undefined = NO-GO)"),
    ("go_scw", "def_scw", "corrected D-weighted, null pooled per scenario (defined cells only)"),
]
OUTCOMES = [("out_C", "(i) test C(R6) > 0.5"), ("out_ba10", "(ii) BA at 10% by R6 > edge"),
            ("out_cap", "(iii) Cap(R6) > 0.10")]


def audit_rates(A: pd.DataFrame, B: int) -> list[dict]:
    rows = []
    for sc, a0 in A.groupby("scenario", sort=False):
        n_units = a0.groupby(["rep", "dataset"]).ngroups
        for call, dcol, design in AUDIT_CALLS:
            a = a0 if dcol is None else a0[a0[dcol] == 1]
            if dcol is not None:
                v = a0.groupby("rep")[dcol].mean()
                v = 1.0 - v
                rows.append(dict(test="H5 audit", design=design, quantity="share of undefined cells", scenario=sc,
                                 rate=v.mean(), mcse=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                                 n_rep=len(v), key=f"{call}|undef_share", n_trials=np.nan))
            if not len(a):
                continue
            R = a.rep.nunique()
            for outc, lab in OUTCOMES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bs = SM.boot_stats(a, call, outc, B=B)
                for qn, (v, se) in bs.items():
                    if qn == "go_rate" and outc != "out_C":
                        continue
                    if qn == "outcome_rate":
                        if call != "go_orig" and dcol is None:
                            continue
                        nm = f"base rate of {lab}"
                    else:
                        nm = qn if qn == "go_rate" else f"{qn} vs {lab}"
                    nt = (a.groupby(["rep", "dataset"]).ngroups if dcol else n_units) if qn == "go_rate" else np.nan
                    rows.append(dict(test="H5 audit", design=design, quantity=nm, scenario=sc, rate=v, mcse=se,
                                     n_rep=R, key=f"{call}{'|' + dcol if dcol else ''}|{qn}|{outc}", n_trials=nt))
    return rows


# ------------------------------------------------------------------ labels and rates
def base_labels() -> tuple[dict, list]:
    lab = {k: (*v, "ds" if "_ds_" in k else "rep") for k, v in SM.LABELS.items()}
    lab["h1_share_pos"] = ("H1 information step", "split level", "share of splits with step > 0", None)
    for o, d in H4_OPTS.items():
        lab[f"h4_2s_{o}_reject"] = ("H4", f"corrected two-stage HKSJ, {d}", "pooled one-sided p < 0.05", "rep")
        lab[f"h4_2s_{o}_ds_reject"] = ("H4", f"corrected two-stage HKSJ, {d}", "per-dataset one-sided p < 0.05", "ds")
    means = list(SM.MEANS) + [f"h4_2s_{o}_{q}" for o in H4_OPTS for q in ("slope", "se", "ds_se")]
    return lab, means


def tc_labels(fam: str, cols) -> tuple[dict, list]:
    lab, means = {}, []
    H = "H1 information step"
    for tag in TAGS[fam]:
        p, dn = "h1" + tag, TAG_NAME[tag]
        lab.update({
            f"{p}_ds_reject": (H, f"{dn}: per-dataset NB one-sided", "rejection rate (step > 0)", "ds"),
            f"{p}_pool_reject": (H, f"{dn}: pooled HKSJ one-sided", "rejection rate (step > 0)", "rep"),
            f"{p}_verdict_supported": (H, f"{dn}: plan verdict rule", "SUPPORTED rate", "rep"),
            f"{p}_ds_cover_truth": (H, f"{dn}: per-dataset NB 95% CI", "coverage of (A) deployment truth", "ds"),
            f"{p}_ds_cover_cv": (H, f"{dn}: per-dataset NB 95% CI", "coverage of (B) CV expectation", "ds"),
            f"{p}_ds_cover_naive_truth": (H, f"{dn}: per-dataset naive t 95% CI", "coverage of (A) deployment truth", "ds"),
            f"{p}_ds_cover_naive_cv": (H, f"{dn}: per-dataset naive t 95% CI", "coverage of (B) CV expectation", "ds"),
            f"{p}_pool_cover_truth": (H, f"{dn}: pooled HKSJ 95% CI", "coverage of (A) deployment truth", "rep"),
            f"{p}_pool_cover_cv": (H, f"{dn}: pooled HKSJ 95% CI", "coverage of (B) CV expectation", "rep"),
            f"{p}_ds_size_truth_upper": (H, f"{dn}: per-dataset NB one-sided", "size at (A) truth, upper tail", "ds"),
            f"{p}_ds_size_cv_upper": (H, f"{dn}: per-dataset NB one-sided", "size at (B) CV expectation, upper tail", "ds"),
            f"{p}_pool_size_truth_upper": (H, f"{dn}: pooled HKSJ one-sided", "size at (A) truth, upper tail", "rep"),
            f"{p}_pool_size_cv_upper": (H, f"{dn}: pooled HKSJ one-sided", "size at (B) CV expectation, upper tail", "rep"),
            f"{p}_share_pos": (H, f"{dn}: split level", "share of splits with step > 0", None)})
        means += [f"{p}_pool_mean", f"{p}_pool_se", f"{p}_ds_mean", f"{p}_ds_se", f"{p}_ds_se_naive", f"{p}_ds_sd",
                  f"ba_edge{tag}", f"ba_cloud{tag}"]
    SB, SV = "Shortcut baselines", "Survival (Sec. 5.3 rule)"
    lab["sc_beats_plain"] = (SB, "(ii) plain: BA(cloud) - BA(best shortcut), NB CI lo > 0", "cloud beats best shortcut", "ds")
    lab["sc_exposed_plain"] = (SB, "(ii) plain: BA(cloud) - BA(best shortcut), NB CI lo > 0", "exposed (not beaten)", "ds")
    if fam == "C":
        lab["sc_beats_z"] = (SB, "(ii) z: BA(cloud_z) - BA(best shortcut), NB CI lo > 0", "cloud beats best shortcut", "ds")
        lab["sc_exposed_z"] = (SB, "(ii) z: BA(cloud_z) - BA(best shortcut), NB CI lo > 0", "exposed (not beaten)", "ds")
        lab["sc_aug_beats"] = (SB, "(iii) aug: BA(cloud_aug) - max(BA(edge_aug), BA(best)), NB CI lo > 0",
                               "cloud beats edge-given-shortcut", "ds")
        lab["surv_cond_step"] = (SV, "condition: info_step_z NB CI lo > 0", "share of datasets", "ds")
        lab["surv_cond_sc"] = (SV, "condition: (ii) on z", "share of datasets", "ds")
        lab["surv_cond_aug"] = (SV, "condition: (iii) on aug", "share of datasets", "ds")
        lab["survive"] = (SV, "all three (z step, (ii) z, (iii) aug)", "survival rate", "ds")
    else:
        lab["surv_cond_step"] = (SV, "condition: info_step NB CI lo > 0", "share of datasets", "ds")
        lab["surv_cond_sc"] = (SV, "condition: (ii) plain", "share of datasets", "ds")
        lab["survive"] = (SV, "both (step, (ii) plain)", "survival rate", "ds")
    means += ["sc_ba_best"] + sorted(k for k in cols if k.startswith("ba_sc_") or k.startswith("sc_best_is_"))
    return lab, means


def rate_row(test, design, qty, sc, v, key, n_trials=np.nan) -> dict:
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna().astype(float)
    n = len(v)
    return dict(test=test, design=design, quantity=qty, scenario=sc, rate=v.mean() if n else np.nan,
                mcse=v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan, n_rep=n, key=key, n_trials=n_trials)


def replicate_rates(T: pd.DataFrame) -> list[dict]:
    rows = []
    for sc, g in T.groupby("scenario", sort=False):
        fam = family(sc)
        lab, means = base_labels() if fam == "base" else tc_labels(fam, g.columns)
        nds = pd.to_numeric(g.get("n_ds", pd.Series(N_DATASETS, index=g.index)), errors="coerce").fillna(N_DATASETS)
        for k, (test, design, qty, unit) in lab.items():
            if k not in g or g[k].isna().all():
                continue
            v = pd.to_numeric(g[k], errors="coerce")
            nt = {"rep": v.notna().sum(), "ds": nds[v.notna()].sum()}.get(unit, np.nan)
            rows.append(rate_row(test, design, qty, sc, v, k, nt))
        for k in means:
            if k in g and g[k].notna().any():
                rows.append(rate_row("descriptive", k, "mean over replicates", sc, g[k], k))
        if fam == "base":                                    # the conditional rows of SM.main
            for cond, cl in (("h3_feas_orig", "original rule feasible"), ("h3_feas_cor", "corrected rule feasible")):
                v = g.loc[g[cond] == 1, "h3_pool_equiv"]
                rows.append(rate_row("H3 equivalence", "pooled TOST +-0.02 (true dC = 0)",
                                     f"equivalence declared | {cl}", sc, v, f"h3_equiv_given_{cond}", len(v)))
            v = ((g.h3_feas_orig == 1) & (g.h3_feas_cor == 0)).astype(float)
            rows.append(rate_row("H3 equivalence", "feasibility rules", "original feasible, corrected not", sc, v,
                                 "h3_orig_not_cor"))
            v = (g.h4_orig_model.astype(str) != "MixedLM").astype(float)
            rows.append(dict(rate_row("H4", "MixedLM fallback to OLS", "share of replicates", sc, v, "h4_ols_fallback"),
                             mcse=np.nan))
        if "cpu_s" in g:
            rows.append(rate_row("descriptive", "cpu_s", "summary CPU seconds per replicate", sc, g.cpu_s, "cpu_s"))
    return rows


def cp_upper_if_zero(rate, n) -> float:
    """Exact one-sided 95% Clopper-Pearson upper bound for 0 successes in n trials."""
    if not (np.isfinite(rate) and rate == 0 and np.isfinite(n) and n > 0):
        return np.nan
    return float(1.0 - 0.05 ** (1.0 / n))


# ------------------------------------------------------------------ tables
def fmt(r) -> str:
    if not np.isfinite(r.rate):
        return ""
    s = f"{r.rate:.3f}"
    if np.isfinite(r.mcse):
        s += f" [{r.mcse:.3f}]"
    if np.isfinite(r.upper95_if_zero):
        s += f" (ub {r.upper95_if_zero:.4f})"
    return s


def write_tables(out: pd.DataFrame, nrep: pd.DataFrame, tt: pd.DataFrame, fn: Path):
    o = out.copy()
    o["cell"] = [fmt(r) for r in o.itertuples(index=False)]
    o["row_order"] = o.groupby(["test", "design", "quantity"], sort=False).ngroup()
    lines = ["# Full validation (R6-2): rates per scenario", "",
             "Rates to 3 decimals, Monte Carlo SE in brackets; (ub x) = exact one-sided 95% upper bound when the count "
             "is 0. Generated by src/sens/simulation_full/summarize_full.py.", "", "## Replicates", "",
             nrep.to_markdown(index=False), ""]
    for fam, title in (("base", "Base family (S0-S3)"), ("T", "Temporal dependence (T)"), ("C", "Confounded (C)")):
        scs = [s for s in SCEN_ORDER if s in set(o.scenario) and family(s) == fam]
        if not scs:
            continue
        sub = o[o.scenario.isin(scs)]
        w = sub.pivot_table(index=["row_order", "test", "design", "quantity"], columns="scenario", values="cell",
                            aggfunc="first", observed=True).reindex(columns=scs).fillna("")
        w = w.sort_index(level=0).reset_index().drop(columns="row_order")
        lines += [f"## {title}", "", w.to_markdown(index=False), ""]
        tq = tt[tt.scenario.isin(scs) & tt.quantity.str.contains(r"^(?:cv_expectation_)?info_step", regex=True)]
        if len(tq):
            tw = tq.assign(cell=[f"{m:.4f} [{e:.4f}]" if np.isfinite(e) else f"{m:.4f}" for m, e in zip(tq["mean"], tq.mcse)]) \
                .pivot_table(index="quantity", columns="scenario", values="cell", aggfunc="first").reindex(columns=scs)
            lines += [f"Targets (truth mean and cross-validated expectation, MCSE in brackets), {title}:", "",
                      tw.fillna("").reset_index().to_markdown(index=False), ""]
    fn.write_text("\n".join(lines))


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out); summ_dir = out / "summary"; summ_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- replicates
    files = sorted(str(f) for f in (out / "reps").glob("*_r[0-9][0-9][0-9][0-9].csv"))
    if a.procs > 1 and len(files) > 50:
        with mp.get_context("fork").Pool(a.procs) as pool:
            loaded = pool.map(read_rep, files, chunksize=64)
    else:
        loaded = [read_rep(f) for f in files]
    reps, skipped = {}, []
    for fn, d, err in loaded:
        m = re.match(r"(.+)_r(\d{4})$", Path(fn).stem)
        sc, rep = m.group(1), int(m.group(2))
        try:
            family(sc)
        except KeyError:
            skipped.append(dict(scenario=sc, rep=rep, reason="unknown scenario")); continue
        if d is None:
            skipped.append(dict(scenario=sc, rep=rep, reason=f"unreadable: {err}")); continue
        c, why = check_rep(sc, d)
        if c is None:
            skipped.append(dict(scenario=sc, rep=rep, reason=why)); continue
        reps[(sc, rep)] = c
    print(f"read {len(files)} replicate files, {len(reps)} usable, {len(skipped)} skipped "
          f"({time.time() - t_start:.0f}s)", flush=True)

    # ---- truth and the cross-validated expectations
    draws = load_truth(out)
    cv, cv_rows = cv_expectations(reps)
    tt = pd.DataFrame(truth_rows(draws) + cv_rows, columns=["scenario", "quantity", "mean", "sd", "mcse", "n_draws"])
    tt["_o"] = [SCEN_ORDER.index(s) if s in SCEN_ORDER else 99 for s in tt.scenario]
    tt = tt.sort_values(["_o"], kind="stable").drop(columns="_o")
    tt.to_csv(summ_dir / "truth.csv", index=False)
    truth = {sc: dict(zip(g.quantity, g["mean"])) for sc, g in tt.groupby("scenario")}

    # ---- per-replicate tests
    T, errors = per_replicate(reps, truth, cv, summ_dir / "tests_per_replicate.csv", a.procs)
    skipped += [dict(scenario=k[0], rep=k[1], reason=f"test error: {e}") for k, e in errors.items()]
    if errors:
        print(f"WARNING: {len(errors)} replicates raised in the tests; first: {next(iter(errors.items()))}", flush=True)
    rates = replicate_rates(T) if len(T) else []

    # ---- audit (base family)
    base_keys = [k for k in reps if family(k[0]) == "base"]
    if base_keys:
        cells = pd.concat([reps[k] for k in base_keys], ignore_index=True)
        A, thr = audit_full(cells)
        A.to_csv(summ_dir / "audit_cells.csv", index=False)
        thr.to_csv(summ_dir / "audit_thresholds.csv", index=False)
        rates += audit_rates(A, a.boot)

    # ---- rates, n_rep, tables
    R = pd.DataFrame(rates)
    R["upper95_if_zero"] = [cp_upper_if_zero(r, n) for r, n in zip(R.rate, R.n_trials)]
    R["_o"] = [SCEN_ORDER.index(s) if s in SCEN_ORDER else 99 for s in R.scenario]
    R["_g"] = R.groupby(["test", "design", "quantity"], sort=False).ngroup()
    R = R.sort_values(["_g", "_o"], kind="stable").drop(columns=["_o", "_g"])
    R = R[["test", "design", "quantity", "scenario", "rate", "mcse", "n_rep", "key", "upper95_if_zero", "n_trials"]]
    R.to_csv(summ_dir / "rates.csv", index=False)
    sk = pd.DataFrame(skipped, columns=["scenario", "rep", "reason"])
    sk.to_csv(summ_dir / "skipped.csv", index=False)
    scs = [s for s in SCEN_ORDER if s in {k[0] for k in reps} | set(sk.scenario) | set(draws)]
    nrep = pd.DataFrame([dict(scenario=s, family=family(s), n_rep=int((T.scenario == s).sum()) if len(T) else 0,
                              n_skipped=int((sk.scenario == s).sum()),
                              n_truth_draws=len(draws[s]) if s in draws else 0) for s in scs])
    nrep.to_csv(summ_dir / "n_rep.csv", index=False)
    write_tables(R, nrep, tt, summ_dir / "tables.md")
    print(nrep.to_string(index=False))
    if len(sk):
        print(f"WARNING: {len(sk)} replicates skipped (see summary/skipped.csv):",
              sk.reason.str.slice(0, 40).value_counts().to_dict())
    print(f"done in {time.time() - t_start:.0f}s -> {summ_dir}")


if __name__ == "__main__":
    main()
