"""POST-PLAN: corrected routing audit.

Phase 1 (verify): for every stored audit cell rebuild Z_val and the router targets from the stored
validation probabilities and thresholds and rerun escalation.audit.run_audit; it must reproduce the
stored audit_C_val, audit_C_lo and audit_go (|diff| <= 1e-9). The script stops if any cell differs.
Phase 2 (correct): StratifiedGroupKFold(shuffle, 8100 + split) on validation labels/groups,
n_splits = min(5, positive groups, negative groups) (>= 2, else undefined); R6 fitted on the other
folds, C computed within each held-out fold and averaged over finite folds. Null: the same pipeline
(same folds) with the (g, w) pairs permuted jointly across validation windows, B = 10 per cell,
rng = default_rng([8300, dataset index, pair index, level index, split]).

    nice -n 10 python src/sens/routers/p1_audit.py
"""
from __future__ import annotations
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from util import (CM, PLAN, OUT, DATASETS, cells, load_cell, stored_value, JsonlStore, key)  # noqa: F401
from escalation import routers as RT
from escalation.audit import run_audit
from escalation.metrics import weighted_concordance
from escalation.splits import groups_per_class

B_PERM = 10
LEVEL_IX = {1.0: 0, 0.1: 1, 0.25: 2, 0.5: 3}


def fold_avg_C(Z, g, w, folds, seed):
    Cs = []
    for tr, te in folds:
        m = RT.fit_regressor_router(Z[tr], g[tr], seed, PLAN, sample_weight=w[tr])
        Cs.append(weighted_concordance(g[te], m.predict(Z[te])))
    Cs = np.asarray(Cs, float)
    fin = np.isfinite(Cs)
    return (float(Cs[fin].mean()) if fin.any() else float("nan")), int(fin.sum()), Cs.tolist()


def original_fold_avg(g, oof, groups, n_folds):
    """Within-fold C of the ORIGINAL audit's out-of-fold scores (same GroupKFold folds), diagnostic."""
    ng = len(np.unique(groups))
    if ng < 2 or not np.isfinite(oof).all():
        return float("nan")
    Cs = [weighted_concordance(g[te], oof[te])
          for _, te in GroupKFold(n_splits=min(n_folds, ng)).split(oof, g, groups)]
    Cs = np.asarray(Cs, float)
    return float(np.nanmean(Cs)) if np.isfinite(Cs).any() else float("nan")


def build(ds, pair, q, sid):
    sp, arr, rows = load_cell(ds, pair, q, sid)
    D = CM.dataset(ds)
    val = sp["val"]
    yv = D["y"][val]; gv = D["groups"][val]
    Z = RT.router_inputs(D["X_edge"][val], arr["pval_ef_ev"])
    tg = RT.router_targets(yv, arr["pval_ef_ev"], arr["pval_cf_cv"], float(arr["t_ef_ev"]), float(arr["t_cf_cv"]))
    return sp, arr, rows, Z, tg, yv, gv


def main():
    allc = cells()
    print(f"{len(allc)} cells", flush=True)
    ver = JsonlStore(OUT / "p1_verify.jsonl")
    t0 = time.time()
    for c in allc:
        k = key(*c)
        if k in ver.done:
            continue
        ds, pair, q, sid = c
        sp, arr, rows, Z, tg, yv, gv = build(*c)
        au = run_audit(Z, tg["g"], tg["w"], gv, sid, PLAN)
        st = {n: stored_value(rows, n, "R6") for n in ("audit_C_val", "audit_C_lo", "audit_go")}

        def same(a, b):
            return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9
        match = all(same(float(au[n]), st[n]) for n in st)
        ver.add(dict(key=k, dataset=ds, pair=pair, q=q, split=sid, match=bool(match),
                     **{f"stored_{n}": st[n] for n in st}, **{f"re_{n}": float(au[n]) for n in st},
                     orig_foldavg_C=original_fold_avg(tg["g"], au["oof"], gv, PLAN.audit_folds)))
    n_ok = sum(r["match"] for r in ver.done.values())
    print(f"verify: {n_ok} / {len(ver.done)} exact matches ({time.time() - t0:.0f}s)", flush=True)
    if n_ok != len(allc):
        print("STOP: reconstruction does not reproduce the stored audit", flush=True)
        sys.exit(2)

    cor = JsonlStore(OUT / "p1_corrected.jsonl")
    t0 = time.time()
    for i, c in enumerate(allc):
        k = key(*c)
        if k in cor.done:
            continue
        ds, pair, q, sid = c
        sp, arr, rows, Z, tg, yv, gv = build(*c)
        g, w = tg["g"], tg["w"]
        gp, gn = groups_per_class(yv, gv)
        nf = min(5, gp, gn)
        rec = dict(key=k, dataset=ds, pair=pair, q=q, split=sid, n_val=len(yv), pos_groups_val=gp,
                   neg_groups_val=gn, n_folds=nf)
        seed = PLAN.model_seed + sid
        if nf < 2:
            rec.update(C_cor=float("nan"), n_folds_finite=0, fold_C=[], null=[float("nan")] * B_PERM)
        else:
            folds = list(StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=8100 + sid)
                         .split(np.zeros(len(yv)), yv, gv))
            C, nfin, fC = fold_avg_C(Z, g, w, folds, seed)
            rng = np.random.default_rng([8300, DATASETS.index(ds), "AB".index(pair), LEVEL_IX[q], sid])
            null = []
            for _ in range(B_PERM):
                p = rng.permutation(len(g))
                null.append(fold_avg_C(Z, g[p], w[p], folds, seed)[0])
            rec.update(C_cor=C, n_folds_finite=nfin, fold_C=fC, null=null)
        cor.add(rec)
        if i % 20 == 0:
            print(f"[{i}/{len(allc)}] {k} C_cor={rec['C_cor']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"corrected done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
