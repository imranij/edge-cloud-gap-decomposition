"""POST-PLAN: the confirmatory pipeline applied to one synthetic dataset, per split.
Post-plan, not pre-specified. Nothing here re-implements the pipeline: splits, models,
thresholds, gains, routers, metrics and the original audit are the frozen escalation modules.

Per split (first `n_splits` of escalation.splits.make_splits with PLAN, i.e. two repeats of K = 5):
  tiers   pair A edge family (LRModel via make_model("A", "edge")) on the edge view and on the cloud
          view, seed 3000 + split; thresholds threshold_max_ba on validation. The model step is zero
          by construction and the information step is BA(LR, cloud view) - BA(LR, edge view).
  routers R0 (seed 5000 + split), R1, and R6 = fit_regressor_router(router_inputs(X_edge[val], p_E),
          g_val, BA weights) exactly as run_dataset.py's R6.
  H3      R6a / R6b: the R6 regressor with N_NOISE = 1 extra input column of independent N(0, 1)
          noise, two independent noise draws. (A first version used 14 columns, as many inputs as R7
          adds; over 13 scenario x replicate units it gave split-level sd(dC) of 0.06-0.11, larger than
          the real R7 - R6 spreads of about 0.02-0.06, and made every TOST trivially infeasible. One
          column gives sd(dC) of about 0.03-0.04 in a 3-dataset check, the real-data range. Those 13
          units are kept in results_sens/simulation/reps_superseded_noise14/ and are not used.) HGB with early_stopping=False and no
          subsampling ignores random_state (verified: identical predictions), so "R6 with another
          random_state" would give dC = 0 identically; the noise-augmented pair is exchangeable, so
          E[C(R6a) - C(R6b)] = 0 exactly.
  H4      rho_pos on the test fold (as run_dataset.py) and on the validation fold (R1 score of the
          validation edge probabilities, validation gains); captures over [0.05, 0.30].
  H5      original: escalation.audit.run_audit (crossfit_r6 and the GO rule reused; the 1,000
          group-bootstrap C values computed by the vectorised group_bootstrap_C_fast, which consumes
          the same random draws and is checked equal to the frozen loop by verify_audit.py). corrected: StratifiedGroupKFold(n_splits =
          min(5, G+, G-), shuffle, random_state 8000 + split) on validation; R6 refitted on the other
          folds; C within each held-out fold, averaged (plain mean of finite folds; the D-weighted
          mean is stored too); null draws permute the validation gains within class (so the BA
          weights stay fixed) through the same folds; one null draw in splits 0 and 5 of each dataset
          (NULL_SPLITS), pooled per scenario to calibrate the GO threshold.
  H2      joint (original): h2_subsample on fit and validation, tiers, thresholds and R6 refitted;
          fixed-tier (corrected): tiers and thresholds of q = 1, only the validation part seen by R6
          subsampled (the validation output of the same h2_subsample call, so levels are nested and
          identical to the joint design's validation subsample).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from escalation.config import PLAN, CAPTURE_RANGE, H2_LEVELS          # noqa: E402
from escalation import metrics as M                                   # noqa: E402
from escalation import routers as RT                                  # noqa: E402
from escalation.models import make_model                              # noqa: E402
from escalation.splits import make_splits, h2_subsample, check_split, groups_per_class  # noqa: E402
from escalation.audit import run_audit, crossfit_r6, group_bootstrap_C  # noqa: E402
from scipy.stats import rankdata                                      # noqa: E402
from run_dataset import spearman                                      # noqa: E402

N_NOISE = 1           # one noise input per router of the H3 pair (see the H3 note in the docstring)
AUDIT_COR_SEED = 8000


def concordance_denominator(g) -> float:
    """D of escalation.metrics.weighted_concordance (sum over pairs of |g_i - g_j|)."""
    gs = np.sort(np.asarray(g, float)); n = len(gs)
    return float((gs * (2 * np.arange(1, n + 1) - n - 1)).sum())


def fit_tiers(ds, fit, val, test, seed, cfg=PLAN):
    y = ds["y"]
    out = {}
    for key, view in (("e", "X_edge"), ("c", "X_cloud")):
        m = make_model("A", "edge", seed, cfg).fit(ds[view][fit], y[fit])
        pv, pt = m.predict_proba(ds[view][val]), m.predict_proba(ds[view][test])
        t = M.threshold_max_ba(pv, y[val])
        out[key] = (pv, pt, t)
    return out


def r6_scores(Z_val, tg, Z_test, seed, cfg=PLAN):
    return RT.fit_regressor_router(Z_val, tg["g"], seed, cfg, sample_weight=tg["w"]).predict(Z_test)


def group_bootstrap_C_fast(g, s, groups, n_boot: int, seed: int) -> np.ndarray:
    """Same draws and same statistic as escalation.audit.group_bootstrap_C (the rng is consumed by
    the identical sequence of rng.integers calls), with the C of all resamples computed in one
    vectorised pass (average ranks along axis 1, the closed form of metrics.weighted_concordance).
    Equality with the frozen function is checked by verify_fast_audit() before the study runs."""
    g = np.asarray(g, float); s = np.asarray(s, float); groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(ug), len(ug)) for _ in range(n_boot)]
    lens = {len(np.concatenate([members[k] for k in p])) for p in picks}
    if len(lens) != 1:                                    # unequal group sizes: use the frozen loop
        return group_bootstrap_C(g, s, groups, n_boot, seed)
    IX = np.stack([np.concatenate([members[k] for k in p]) for p in picks])
    G, Sc = g[IX], s[IX]
    n = IX.shape[1]
    gs = np.sort(G, axis=1)
    D = (gs * (2 * np.arange(1, n + 1) - n - 1)).sum(axis=1)
    R = rankdata(Sc, method="average", axis=1)
    Ssum = (G * (R - (n + 1) / 2.0)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(D > 0, 0.5 + Ssum / np.where(D > 0, D, 1.0), np.nan)
    return out


def run_audit_fast(Z_val, g_val, w_val, groups_val, split_id: int, cfg=PLAN) -> dict:
    """escalation.audit.run_audit with group_bootstrap_C_fast in place of the loop (same result)."""
    seed = cfg.model_seed + split_id
    oof = crossfit_r6(Z_val, g_val, w_val, groups_val, seed, cfg)
    if not np.isfinite(oof).all():
        return dict(audit_C_val=np.nan, audit_C_lo=np.nan, audit_C_hi=np.nan, audit_go=0.0, oof=oof)
    C = M.weighted_concordance(g_val, oof)
    boot = group_bootstrap_C_fast(g_val, oof, groups_val, cfg.audit_boot, cfg.audit_seed + split_id)
    if np.isfinite(boot).sum() == 0:
        lo = hi = np.nan
    else:
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    go = float(np.isfinite(lo) and lo > cfg.audit_go)
    return dict(audit_C_val=C, audit_C_lo=float(lo), audit_C_hi=float(hi), audit_go=go, oof=oof)


def corrected_audit_stat(Z, g, w, folds, seed, cfg=PLAN):
    Cs, Ds = [], []
    for tr, te in folds:
        m = RT.fit_regressor_router(Z[tr], g[tr], seed, cfg, sample_weight=w[tr])
        Cs.append(M.weighted_concordance(g[te], m.predict(Z[te])))
        Ds.append(concordance_denominator(g[te]))
    Cs, Ds = np.asarray(Cs, float), np.asarray(Ds, float)
    fin = np.isfinite(Cs)
    if not fin.any():
        return np.nan, np.nan
    return float(Cs[fin].mean()), float((Cs[fin] * Ds[fin]).sum() / Ds[fin].sum())


NULL_SPLITS = (0, 5)  # corrected-audit null draws: splits 0 and 5 of every dataset (8 per replicate)


def run_sim_dataset(ds: dict, ds_seed: int, n_splits: int = 10, null_draws: int = 1, cfg=PLAN,
                    audit_fn=None) -> list[dict]:
    audit_fn = audit_fn or run_audit_fast
    y, groups, Xe = ds["y"], ds["groups"], ds["X_edge"]
    splits = make_splits(y, groups, "tabular", cfg)[:n_splits]
    nrng = np.random.default_rng([ds_seed, 7])
    noise_a = nrng.normal(size=(len(y), N_NOISE)); noise_b = nrng.normal(size=(len(y), N_NOISE))
    rows = []
    for sp in splits:
        if sp["skip"]:
            rows.append(dict(split=sp["split_id"], skipped=sp["skip"])); continue
        sid = sp["split_id"]; seed = cfg.model_seed + sid
        fit, val, test = sp["fit"], sp["val"], sp["test"]
        yv, yt, gv = y[val], y[test], groups[val]
        T = fit_tiers(ds, fit, val, test, seed, cfg)
        (pe_v, pe_t, tE), (pc_v, pc_t, tC) = T["e"], T["c"]
        pe_b, pc_b = (pe_t >= tE).astype(int), (pc_t >= tC).astype(int)
        ba_e, ba_c = M.balanced_accuracy(yt, pe_b), M.balanced_accuracy(yt, pc_b)
        g = M.escalation_gain(yt, pe_b, pc_b)
        Z_val, Z_test = RT.router_inputs(Xe[val], pe_v), RT.router_inputs(Xe[test], pe_t)
        tg = RT.router_targets(yv, pe_v, pc_v, tE, tC)
        sc = {"R0": RT.r0_random(len(test), cfg.router_seed + sid), "R1": RT.r1_confidence(pe_t, tE),
              "R6": r6_scores(Z_val, tg, Z_test, seed, cfg),
              "R6a": r6_scores(np.column_stack([Z_val, noise_a[val]]), tg,
                               np.column_stack([Z_test, noise_a[test]]), seed, cfg),
              "R6b": r6_scores(np.column_stack([Z_val, noise_b[val]]), tg,
                               np.column_stack([Z_test, noise_b[test]]), seed, cfg)}
        r = dict(split=sid, repeat=sp["repeat"], fold=sp["fold"], n_fit=len(fit), n_val=len(val), n_test=len(test),
                 n_train=len(fit) + len(val), n_pos_val=int(yv.sum()), n_pos_train=int(y[fit].sum() + yv.sum()),
                 ba_edge=ba_e, ba_cloud=ba_c, info_step=ba_c - ba_e, g_sum=float(g.sum()),
                 share_g_nonzero=float((g != 0).mean()), share_gval_nonzero=float((tg["g"] != 0).mean()))
        for k, s in sc.items():
            r[f"C_{k}"] = M.weighted_concordance(g, s)
            r[f"cap_{k}"] = M.capture(g, s, *CAPTURE_RANGE)
        for k in ("R1", "R6"):
            r[f"ba10_{k}"] = M.ba_at_budget(yt, pe_b, pc_b, sc[k], 0.10)
        r["rho_pos_test"] = spearman(sc["R1"][yt == 1], g[yt == 1])
        r["rho_pos_val"] = spearman(RT.r1_confidence(pe_v, tE)[yv == 1], tg["g"][yv == 1])
        # ---- H5 original
        au = audit_fn(Z_val, tg["g"], tg["w"], gv, sid, cfg)
        r.update(audit_go_orig=au["audit_go"], audit_C_val=au["audit_C_val"], audit_C_lo=au["audit_C_lo"])
        # ---- H5 corrected
        gp, gn = groups_per_class(yv, gv)
        nf = min(5, gp, gn)
        r["audit_cor_folds"] = nf
        if nf >= 2:
            folds = list(StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=AUDIT_COR_SEED + sid)
                         .split(np.zeros(len(yv)), yv, gv))
            r["audit_cor_C"], r["audit_cor_Cw"] = corrected_audit_stat(Z_val, tg["g"], tg["w"], folds, seed, cfg)
            prng = np.random.default_rng([ds_seed, sid, 77])
            for b in range(null_draws if sid in NULL_SPLITS else 0):
                gperm = tg["g"].copy()
                for c in (0, 1):
                    ix = np.flatnonzero(yv == c)
                    gperm[ix] = tg["g"][prng.permutation(ix)]
                r[f"audit_null_C_{b}"], r[f"audit_null_Cw_{b}"] = corrected_audit_stat(Z_val, gperm, tg["w"], folds,
                                                                                      seed, cfg)
        # ---- H2: q = 1 values, then the subsampled levels
        r["h2j_C_R6_q1.0"], r["h2j_C_R1_q1.0"], r["h2j_npos_q1.0"] = r["C_R6"], r["C_R1"], r["n_pos_train"]
        r["h2f_C_R6_q1.0"], r["h2f_npos_val_q1.0"] = r["C_R6"], r["n_pos_val"]
        for q in H2_LEVELS:
            fq, vq = h2_subsample(y, groups, fit, val, q, sid, cfg)
            # fixed tiers: only the validation part seen by R6 is subsampled
            pos = np.searchsorted(val, vq)
            assert np.array_equal(val[pos], vq)
            tgq = RT.router_targets(y[vq], pe_v[pos], pc_v[pos], tE, tC)
            r[f"h2f_C_R6_q{q}"] = M.weighted_concordance(g, r6_scores(Z_val[pos], tgq, Z_test, seed, cfg))
            r[f"h2f_npos_val_q{q}"] = int(y[vq].sum())
            # joint (pre-specified): tiers, thresholds and R6 retrained on the subsampled parts
            if check_split(y, groups, fq, vq, test):
                r[f"h2j_C_R6_q{q}"] = r[f"h2j_C_R1_q{q}"] = np.nan; r[f"h2j_npos_q{q}"] = np.nan; continue
            Tq = fit_tiers(ds, fq, vq, test, seed, cfg)
            (qe_v, qe_t, qtE), (qc_v, qc_t, qtC) = Tq["e"], Tq["c"]
            gq = M.escalation_gain(yt, (qe_t >= qtE).astype(int), (qc_t >= qtC).astype(int))
            tgj = RT.router_targets(y[vq], qe_v, qc_v, qtE, qtC)
            s6 = r6_scores(RT.router_inputs(Xe[vq], qe_v), tgj, RT.router_inputs(Xe[test], qe_t), seed, cfg)
            r[f"h2j_C_R6_q{q}"] = M.weighted_concordance(gq, s6)
            r[f"h2j_C_R1_q{q}"] = M.weighted_concordance(gq, RT.r1_confidence(qe_t, qtE))
            r[f"h2j_npos_q{q}"] = int(y[fq].sum() + y[vq].sum())
        rows.append(r)
    return rows
