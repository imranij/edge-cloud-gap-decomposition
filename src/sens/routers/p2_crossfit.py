"""POST-PLAN: cross-fitted routers and hybrid baselines, the fixed-predictor two-stage deferral
surrogate R10, causal streaming thresholds and the feature-aware combiner.

For every stored main split (q = 1) of pairs A and B:
  T = fit U val. Deployed-edge model (family 'edge', edge view) and cloud model (family 'cloud',
  cloud view) are cross-fitted on T with StratifiedGroupKFold(shuffle, 6000 + split,
  n_splits = min(4, positive groups, negative groups in T)), make_model(pair, family,
  PLAN.model_seed + split), giving out-of-fold probabilities p_E_oof, p_C_oof on every training
  window, binarised at the STORED deployed thresholds t_ef_ev, t_cf_cv.
  R4x/R5x/R6x: routers.learned_routers on Z_T = router_inputs(X_edge[T], p_E_oof); test inputs
  router_inputs(X_edge[test], stored p_ef_ev).
  R10: StandardScaler + LogisticRegression(C = 1) on the windows of T with g != 0, label 1[g > 0],
  sample weight |g| / mean|g|; score = P(label 1).
  HyR: mean of the test ranks of R1 and R6x. HyS: R1 or R6x, whichever has the higher fold-averaged
  C on T (R6x out of fold from a second cross-fit over the same folds, R1 from p_E_oof).
  Streaming: thresholds at the (1 - f) quantile of the out-of-fold training scores of R1 / R6x.
  Combiner: HGB regressor (PLAN.router_gb, seed) of g_T on [X_cloud[T], p_E_oof, p_C_oof]; on test
  the cloud decides where the prediction is > 0, the edge elsewhere.
Writes results_sens/routers/p2_rows.parquet (long), p2_counts.parquet, p2_stream.parquet,
p2_comb.parquet and per-split out-of-fold arrays in p2_arrays/.

    OMP_NUM_THREADS=1 nice -n 10 python src/sens/routers/p2_crossfit.py
"""
from __future__ import annotations
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from util import CM, PLAN, OUT, DATASETS, load_cell, stored_value
from escalation import metrics as M
from escalation import routers as RT
from escalation.models import make_model
from escalation.splits import groups_per_class

BUDG = (0.05, 0.10, 0.20, 0.30)
AD = OUT / "p2_arrays"
AD.mkdir(exist_ok=True)


def crossfit(pair, fam, X, y, folds, seed):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            p[te] = float(y[tr][0])
        else:
            p[te] = make_model(pair, fam, seed, PLAN).fit(X[tr], y[tr]).predict_proba(X[te])
    return p


def fold_avg_C(g, s, folds):
    c = np.array([M.weighted_concordance(g[te], s[te]) for _, te in folds], float)
    return float(np.nanmean(c)) if np.isfinite(c).any() else float("nan")


def r10_fit_score(Z_T, g, Z_test):
    nz = g != 0
    lab = (g[nz] > 0).astype(int)
    if nz.sum() < 2 or len(np.unique(lab)) < 2:
        return np.zeros(len(Z_test)), int(nz.sum()), int((g > 0).sum())
    sw = np.abs(g[nz]); sw = sw / sw.mean()
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    m.fit(Z_T[nz], lab, logisticregression__sample_weight=sw)
    return m.predict_proba(Z_test)[:, 1], int(nz.sum()), int((g > 0).sum())


def router_rows(base, y, pe, pc, g, scores):
    out = []
    for r, s in scores.items():
        out.append(dict(base, quantity="C", router=r, budget=np.nan, value=M.weighted_concordance(g, s)))
        out.append(dict(base, quantity="capture", router=r, budget=np.nan, value=M.capture(g, s, 0.05, 0.30)))
        for f in BUDG:
            out.append(dict(base, quantity="ba_at_budget", router=r, budget=f, value=M.ba_at_budget(y, pe, pc, s, f)))
    return out


def run_split(ds, pair, sid):
    sp, arr, rows = load_cell(ds, pair, 1.0, sid)
    D = CM.dataset(ds)
    y, groups, Xe, Xc = D["y"], D["groups"], D["X_edge"], D["X_cloud"]
    fit, val, test = sp["fit"], sp["val"], sp["test"]
    T = np.sort(np.concatenate([fit, val]))
    yT, gT_ = y[T], groups[T]
    seed = PLAN.model_seed + sid
    te_, tc_ = float(arr["t_ef_ev"]), float(arr["t_cf_cv"])
    gp, gn = groups_per_class(yT, gT_)
    nf = min(4, gp, gn)
    folds = list(StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=6000 + sid)
                 .split(np.zeros(len(yT)), yT, gT_))
    t0 = time.time()
    pE = crossfit(pair, "edge", np.asarray(Xe[T]), yT, folds, seed)
    pC = crossfit(pair, "cloud", np.asarray(Xc[T]), yT, folds, seed)
    t_models = time.time() - t0
    tg = RT.router_targets(yT, pE, pC, te_, tc_)
    gT, wT = tg["g"], tg["w"]
    Z_T = RT.router_inputs(Xe[T], pE)
    Z_te = RT.router_inputs(Xe[test], arr["p_ef_ev"])
    yt = arr["y_test"]; g = arr["g_dep"]
    pe = (arr["p_ef_ev"] >= te_).astype(int); pc = (arr["p_cf_cv"] >= tc_).astype(int)
    assert np.allclose(M.escalation_gain(yt, pe, pc), g)
    lr = RT.learned_routers(Z_T, Z_te, tg, seed, PLAN)
    r10, n_nz, n_gpos = r10_fit_score(Z_T, gT, Z_te)
    # second cross-fit of R6x over the same folds (out-of-fold scores on T)
    r6_oof = np.full(len(yT), np.nan)
    for tr, tst in folds:
        r6_oof[tst] = RT.fit_regressor_router(Z_T[tr], gT[tr], seed, PLAN, sample_weight=wT[tr]).predict(Z_T[tst])
    r1_T = RT.r1_confidence(pE, te_)
    C_T = dict(R1_pooled=M.weighted_concordance(gT, r1_T), R6x_pooled=M.weighted_concordance(gT, r6_oof),
               R1_foldavg=fold_avg_C(gT, r1_T, folds), R6x_foldavg=fold_avg_C(gT, r6_oof, folds))
    R1 = arr["score_R1"]
    assert np.allclose(R1, RT.r1_confidence(arr["p_ef_ev"], te_))
    scores = {"R1": R1, "R6": arr["score_R6"], "R4": arr["score_R4"], "R5": arr["score_R5"],
              "R4x": lr["R4"], "R5x": lr["R5"], "R6x": lr["R6"], "R10": r10,
              "HyR": (rankdata(R1) + rankdata(lr["R6"])) / 2.0}
    if "score_R8" in arr:
        scores["R8"] = arr["score_R8"]
    hys_pick = "R6x" if C_T["R6x_foldavg"] > C_T["R1_foldavg"] else "R1"
    scores["HyS"] = scores[hys_pick]
    base = dict(dataset=ds, pair=pair, split=sid, n_train=len(T), n_test=len(test))
    out_rows = router_rows(base, yt, pe, pc, g, scores)
    # sanity: stored C of R1/R6 reproduced
    for r in ("R1", "R6"):
        st = stored_value(rows, "C", r)
        mine = [x["value"] for x in out_rows if x["quantity"] == "C" and x["router"] == r][0]
        assert abs(st - mine) < 1e-9, (ds, pair, sid, r, st, mine)
    # counts
    yv = y[val]; yf = y[fit]
    cnt = dict(base, n_val=len(val), pos_win_val=int(yv.sum()), pos_groups_val=groups_per_class(yv, groups[val])[0],
               n_fit=len(fit), pos_win_fit=int(yf.sum()), pos_groups_fit=groups_per_class(yf, groups[fit])[0],
               n_T=len(T), pos_win_T=int(yT.sum()), pos_groups_T=gp, neg_groups_T=gn, n_folds=nf,
               r10_n_nonzero=n_nz, r10_n_gpos=n_gpos, nonzero_g_val=int((M.escalation_gain(
                   yv, (arr["pval_ef_ev"] >= te_).astype(int), (arr["pval_cf_cv"] >= tc_).astype(int)) != 0).sum()),
               hys_pick=hys_pick, seconds_models=t_models,
               ba_E_oof_T=M.balanced_accuracy(yT, (pE >= te_).astype(int)),
               ba_C_oof_T=M.balanced_accuracy(yT, (pC >= tc_).astype(int)), **{f"C_T_{k}": v for k, v in C_T.items()})
    # streaming (causal thresholds from out-of-fold training scores)
    stream = []
    for r, s_tr, s_te in (("R1", r1_T, R1), ("R6x", r6_oof, lr["R6"])):
        for f in BUDG:
            thr = float(np.quantile(s_tr, 1 - f))
            esc = s_te >= thr
            stream.append(dict(base, router=r, f=f, realised_frac=float(esc.mean()),
                               ba_causal=M.ba_with_escalation(yt, pe, pc, esc),
                               ba_lookahead=M.ba_at_budget(yt, pe, pc, s_te, f)))
    # feature-aware combiner
    gbr = HistGradientBoostingRegressor(**PLAN.router_gb, early_stopping=False, random_state=seed)
    gbr.fit(np.column_stack([Xc[T], pE, pC]), gT)
    pred = gbr.predict(np.column_stack([Xc[test], arr["p_ef_ev"], arr["p_cf_cv"]]))
    ba_comb = M.ba_with_escalation(yt, pe, pc, pred > 0)
    comb = dict(base, ba_comb=ba_comb, esc_frac=float((pred > 0).mean()),
                ba_ef_ev=stored_value(rows, "ba_ef_ev"), ba_cf_cv=stored_value(rows, "ba_cf_cv"),
                ba_stacker_dep=stored_value(rows, "ba_stacker_dep"), ba_oracle_dep=stored_value(rows, "ba_oracle_dep"),
                unresolved_dep=stored_value(rows, "unresolved_dep"))
    np.savez_compressed(AD / f"{ds}_{pair}_{sid}.npz", T=T, pE=pE, pC=pC, r6_oof=r6_oof,
                        fold=np.concatenate([np.full(len(te2), i) for i, (_, te2) in enumerate(folds)])[
                            np.argsort(np.concatenate([te2 for _, te2 in folds]))],
                        **{f"score_{k}": v for k, v in scores.items()})
    return out_rows, cnt, stream, comb


def main():
    import sys
    only = sys.argv[1:] or list(DATASETS)
    fr = OUT / "p2_rows.parquet"
    parts = {n: (pd.read_parquet(OUT / f"p2_{n}.parquet") if (OUT / f"p2_{n}.parquet").exists() else pd.DataFrame())
             for n in ("rows", "counts", "stream", "comb")}
    done = set(map(tuple, parts["counts"][["dataset", "pair", "split"]].values.tolist())) if len(parts["counts"]) else set()
    t0 = time.time()
    for ds in only:
        for pair in ("A", "B"):
            for sid in CM.stored_split_ids(ds, pair, None):
                if (ds, pair, sid) in done:
                    continue
                t1 = time.time()
                r, c, s, cb = run_split(ds, pair, sid)
                parts["rows"] = pd.concat([parts["rows"], pd.DataFrame(r)], ignore_index=True)
                parts["counts"] = pd.concat([parts["counts"], pd.DataFrame([c])], ignore_index=True)
                parts["stream"] = pd.concat([parts["stream"], pd.DataFrame(s)], ignore_index=True)
                parts["comb"] = pd.concat([parts["comb"], pd.DataFrame([cb])], ignore_index=True)
                for n, df in parts.items():
                    df.to_parquet(OUT / f"p2_{n}.parquet", index=False)
                C = {x["router"]: x["value"] for x in r if x["quantity"] == "C"}
                print(f"{ds} {pair} {sid}: C R1 {C['R1']:.3f} R6 {C['R6']:.3f} R6x {C['R6x']:.3f} R10 {C['R10']:.3f} "
                      f"HyR {C['HyR']:.3f} ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
    del fr


if __name__ == "__main__":
    main()
