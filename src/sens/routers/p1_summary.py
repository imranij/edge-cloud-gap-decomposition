"""POST-PLAN summary: original vs corrected audit against three outcomes.

Reads results_sens/routers/p1_verify.jsonl and p1_corrected.jsonl plus the stored main-run rows.
GO (corrected) iff fold-averaged C > 95th percentile of the pooled permutation null of its
dataset x pair x level block (cells with undefined C are NO-GO, as in the original audit).
Outcomes: (i) stored outcome (capture(R6) > 0.10); (ii) stored ba_at_budget(R6, 0.10) > stored
ba_ef_ev; (iii) stored test C(R6) > 0.5. Permutation test: GO calls permuted within
dataset x pair x level blocks, 10,000 permutations, rng seed 9100.
Writes p1_cells.csv, p1_null_thresholds.csv, p1_ba.csv.
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from util import OUT, load_cell, stored_value

N_PERM = 10_000


def load():
    ver = pd.DataFrame([json.loads(x) for x in (OUT / "p1_verify.jsonl").read_text().splitlines() if x.strip()])
    cor = pd.DataFrame([json.loads(x) for x in (OUT / "p1_corrected.jsonl").read_text().splitlines() if x.strip()])
    ver = ver.drop_duplicates("key", keep="last"); cor = cor.drop_duplicates("key", keep="last")
    d = ver.merge(cor.drop(columns=["dataset", "pair", "q", "split"]), on="key")
    outs = []
    for _, r in d.iterrows():
        _, _, rows = load_cell(r.dataset, r.pair, r.q, int(r.split))
        outs.append(dict(key=r.key, out_capture=stored_value(rows, "outcome", "R6"),
                         out_edge=float(stored_value(rows, "ba_at_budget", "R6", 0.10) > stored_value(rows, "ba_ef_ev")),
                         out_C=float(stored_value(rows, "C", "R6") > 0.5), C_test_R6=stored_value(rows, "C", "R6"),
                         capture_R6=stored_value(rows, "capture", "R6"),
                         dba_edge=stored_value(rows, "ba_at_budget", "R6", 0.10) - stored_value(rows, "ba_ef_ev")))
    d = d.merge(pd.DataFrame(outs), on="key")
    d["block"] = d.dataset + "|" + d.pair + "|" + d.q.astype(str)
    return d


def thresholds(d):
    rows = []
    for b, g in d.groupby("block"):
        null = np.concatenate([np.asarray(v, float) for v in g["null"]])
        null = null[np.isfinite(null)]
        rows.append(dict(block=b, n_null=len(null), null_mean=null.mean() if len(null) else np.nan,
                         null_sd=null.std(ddof=1) if len(null) > 1 else np.nan,
                         thr95=np.percentile(null, 95) if len(null) else np.nan,
                         mean_C_cor=np.nanmean(g.C_cor), mean_C_orig=np.nanmean(g.re_audit_C_val)))
    return pd.DataFrame(rows)


def ba_stats(call, out):
    call = np.asarray(call, int); out = np.asarray(out, int)
    tp = ((call == 1) & (out == 1)).sum(-1); fp = ((call == 1) & (out == 0)).sum(-1)
    fn = ((call == 0) & (out == 1)).sum(-1); tn = ((call == 0) & (out == 0)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp)), tp, fp, fn, tn


def perm_p(call, out, blocks, seed):
    call = np.asarray(call, int); out = np.asarray(out, int)
    obs = float(ba_stats(call, out)[0])
    if not np.isfinite(obs):
        return obs, np.nan
    rng = np.random.default_rng(seed)
    P = np.empty((N_PERM, len(call)), dtype=np.int8)
    for b in np.unique(blocks):
        ix = np.flatnonzero(blocks == b)
        P[:, ix] = call[ix][rng.permuted(np.tile(np.arange(len(ix)), (N_PERM, 1)), axis=1)]
    bp = ba_stats(P, out[None, :])[0]
    return obs, float((1 + (bp >= obs - 1e-12).sum()) / (N_PERM + 1))


def main():
    d = load()
    th = thresholds(d)
    d = d.merge(th[["block", "thr95"]], on="block")
    d["go_orig"] = d.stored_audit_go.astype(int)
    d["go_cor"] = (np.isfinite(d.C_cor) & (d.C_cor > d.thr95)).astype(int)
    d.drop(columns=["null", "fold_C"]).to_csv(OUT / "p1_cells.csv", index=False)
    th.to_csv(OUT / "p1_null_thresholds.csv", index=False)
    subsets = [("pooled", np.ones(len(d), bool)), ("q=1 (A+B)", (d.q == 1.0).to_numpy())]
    subsets += [(f"dataset={x}", (d.dataset == x).to_numpy()) for x in sorted(d.dataset.unique())]
    subsets += [(f"level=A q={x}", ((d.pair == "A") & (d.q == x)).to_numpy()) for x in (1.0, 0.5, 0.25, 0.1)]
    subsets += [("level=B q=1.0", (d.pair == "B").to_numpy())]
    res = []
    seed = 9100
    for aud in ("orig", "cor"):
        for oc in ("out_capture", "out_edge", "out_C"):
            for nm, m in subsets:
                sub = d[m]
                c, o = sub[f"go_{aud}"].to_numpy(), sub[oc].to_numpy().astype(int)
                ba, tp, fp, fn, tn = ba_stats(c, o)
                obs, p = perm_p(c, o, sub.block.to_numpy(), seed)
                _, p_ds = perm_p(c, o, sub.dataset.to_numpy(), seed + 5000)
                seed += 1
                res.append(dict(audit=aud, outcome=oc, subset=nm, n=len(sub), n_go=int(c.sum()), n_outcome=int(o.sum()),
                                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                                precision=tp / (tp + fp) if tp + fp else np.nan,
                                recall=tp / (tp + fn) if tp + fn else np.nan, ba=float(ba), p_perm=p,
                                base_rate=o.mean(), p_perm_within_dataset=p_ds))
    pd.DataFrame(res).to_csv(OUT / "p1_ba.csv", index=False)
    cs = []
    for nm, m in subsets:
        sub = d[m]
        cs.append(dict(subset=nm, n=len(sub), n_C_orig_finite=int(np.isfinite(sub.re_audit_C_val).sum()),
                       C_orig_pooled_oof=np.nanmean(sub.re_audit_C_val), C_orig_foldavg=np.nanmean(sub.orig_foldavg_C),
                       n_C_cor_finite=int(np.isfinite(sub.C_cor).sum()), C_cor=np.nanmean(sub.C_cor),
                       null_mean=np.nanmean(np.concatenate([np.asarray(v, float) for v in sub.null])),
                       C_test_R6=np.nanmean(sub.C_test_R6),
                       corr_Ccor_Ctest=sub[["C_cor", "C_test_R6"]].corr().iloc[0, 1],
                       corr_Corig_Ctest=sub[["re_audit_C_val", "C_test_R6"]].corr().iloc[0, 1]))
    pd.DataFrame(cs).to_csv(OUT / "p1_C_summary.csv", index=False)
    print(pd.DataFrame(cs).round(3).to_string())
    print(pd.DataFrame(res).to_string())


if __name__ == "__main__":
    main()
