"""POST-PLAN: corrected audit with the WITHIN-CLASS permutation null.

The corrected audit specified in the paper calibrates its GO threshold against permutations of the
validation gains WITHIN CLASS (g permuted among windows of the same label, so the balanced-accuracy
weights w stay attached to their labels). The first run (src/sens/routers/p1_audit.py) used
a joint permutation of the (g, w) pairs across all windows. This script recomputes only the null, with
the same cells, folds (StratifiedGroupKFold, shuffle, 8100 + split), R6 settings and fold-averaged C
as p1_audit.py, B = 10 within-class permutations per cell, pooled within dataset x pair x level, and
reruns the p1_summary analysis with it. The observed fold-averaged C (C_cor) is taken unchanged from
results_sens/routers/p1_corrected.jsonl.

    nice -n 10 python src/sens/designs/p1w_within_class.py [--summary-only]
Writes results_sens/routers_wc/{p1w_null.jsonl, p1w_cells.csv, p1w_null_thresholds.csv, p1w_ba.csv}.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "sens" / "routers"))
from util import PLAN, DATASETS, cells, JsonlStore, key       # noqa: E402
import p1_audit as P1                                           # noqa: E402
import p1_summary as S1                                         # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold        # noqa: E402
from escalation.splits import groups_per_class                  # noqa: E402

OUTW = ROOT / "results_sens" / "routers_wc"
OUTW.mkdir(parents=True, exist_ok=True)
B_PERM = 10


def within_class_perm(g, y, rng):
    gp = g.copy()
    for c in (0, 1):
        ix = np.flatnonzero(y == c)
        gp[ix] = g[ix][rng.permutation(len(ix))]
    return gp


def compute():
    store = JsonlStore(OUTW / "p1w_null.jsonl")
    allc = cells()
    t0 = time.time()
    for i, c in enumerate(allc):
        k = key(*c)
        if k in store.done:
            continue
        ds, pair, q, sid = c
        sp, arr, rows, Z, tg, yv, gv = P1.build(*c)
        g, w = tg["g"], tg["w"]
        gp_, gn_ = groups_per_class(yv, gv)
        nf = min(5, gp_, gn_)
        rec = dict(key=k, dataset=ds, pair=pair, q=q, split=sid)
        if nf < 2:
            rec["null_wc"] = [float("nan")] * B_PERM
        else:
            folds = list(StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=8100 + sid)
                         .split(np.zeros(len(yv)), yv, gv))
            rng = np.random.default_rng([8400, DATASETS.index(ds), "AB".index(pair), P1.LEVEL_IX[q], sid])
            seed = PLAN.model_seed + sid
            rec["null_wc"] = [P1.fold_avg_C(Z, within_class_perm(g, yv, rng), w, folds, seed)[0] for _ in range(B_PERM)]
        store.add(rec)
        if i % 20 == 0:
            print(f"[{i}/{len(allc)}] {k} ({time.time() - t0:.0f}s)", flush=True)


def summarise():
    d = S1.load()
    wc = pd.DataFrame([json.loads(x) for x in (OUTW / "p1w_null.jsonl").read_text().splitlines() if x.strip()])
    wc = wc.drop_duplicates("key", keep="last")[["key", "null_wc"]]
    d = d.merge(wc, on="key", how="inner")
    d["null_joint"] = d["null"]
    d["null"] = d["null_wc"]                       # S1.thresholds pools the 'null' column
    th = S1.thresholds(d)
    d = d.merge(th[["block", "thr95"]], on="block")
    d["go_orig"] = d.stored_audit_go.astype(int)
    d["go_cor"] = (np.isfinite(d.C_cor) & (d.C_cor > d.thr95)).astype(int)
    d.drop(columns=["null", "null_joint", "null_wc", "fold_C"]).to_csv(OUTW / "p1w_cells.csv", index=False)
    th.to_csv(OUTW / "p1w_null_thresholds.csv", index=False)
    subsets = [("pooled", np.ones(len(d), bool)), ("q=1 (A+B)", (d.q == 1.0).to_numpy())]
    subsets += [(f"dataset={x}", (d.dataset == x).to_numpy()) for x in sorted(d.dataset.unique())]
    subsets += [(f"level=A q={x}", ((d.pair == "A") & (d.q == x)).to_numpy()) for x in (1.0, 0.5, 0.25, 0.1)]
    subsets += [("level=B q=1.0", (d.pair == "B").to_numpy())]
    res, seed = [], 9100
    for oc in ("out_capture", "out_edge", "out_C"):
        for nm, m in subsets:
            sub = d[m]
            c, o = sub["go_cor"].to_numpy(), sub[oc].to_numpy().astype(int)
            ba, tp, fp, fn, tn = S1.ba_stats(c, o)
            obs, p = S1.perm_p(c, o, sub.block.to_numpy(), seed)
            _, p_ds = S1.perm_p(c, o, sub.dataset.to_numpy(), seed + 5000)
            seed += 1
            res.append(dict(audit="cor_within_class", outcome=oc, subset=nm, n=len(sub), n_go=int(c.sum()),
                            n_outcome=int(o.sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                            precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / (tp + fn) if tp + fn else np.nan,
                            ba=float(ba), p_perm=p, base_rate=o.mean(), p_perm_within_dataset=p_ds))
    out = pd.DataFrame(res)
    out.to_csv(OUTW / "p1w_ba.csv", index=False)
    print(th.round(3).to_string())
    print(out[out.subset.isin(["pooled", "q=1 (A+B)"])].round(3).to_string())


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        compute()
    summarise()
