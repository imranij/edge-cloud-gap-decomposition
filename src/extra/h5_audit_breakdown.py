"""What the pre-specified H5 audit verdict rests on.
Post-plan analysis, not pre-specified. No model is refitted.

    python src/extra/h5_audit_breakdown.py

Reads the stored audit calls, validation C (out-of-fold, audit_C_val), lower bounds, test C and
capture of R6, ba_at_budget(R6) and ba_ef_ev of every audit cell (dataset x pair A/B x split x H2
level). Outputs (results/extra/):
  h5_breakdown.csv      BA of the calls, GO count, GO precision and base rate: all cells, q = 1 only,
                        per dataset, per H2 level, per dataset x level; within-dataset permutation
                        p-values (10,000 permutations, seed 9100)
  h5_block_perm.csv     permutation that keeps dataset x split blocks intact (the vector of calls of
                        one split is moved as a whole to another split of the same dataset)
  h5_oof_vs_test.csv    mean out-of-fold validation C against mean test C of R6, per dataset x pair x
                        level, and the AUC of audit_C_val for the outcome per dataset
  h5_deploy_outcome.csv secondary outcome 'escalation beats the edge alone at a 10% budget'
                        (ba_at_budget(R6, 0.10) > ba_ef_ev), its overlap with the plan's outcome
                        ('routing beats random', capture > 0.10) and the audit's BA against it
  h5_go_failures.csv    GO calls whose test capture was negative
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

N_PERM = 10000
SEED = 9100


def ba(c, o):
    c = np.asarray(c, int); o = np.asarray(o, int)
    tp = ((c == 1) & (o == 1)).sum(-1); fn = ((c == 0) & (o == 1)).sum(-1)
    tn = ((c == 0) & (o == 0)).sum(-1); fp = ((c == 1) & (o == 0)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def perm_within(c, o, strata, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    c = np.asarray(c, int); o = np.asarray(o, int); strata = np.asarray(strata)
    obs = float(ba(c, o))
    P = np.empty((n, len(c)), dtype=np.int8)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        P[:, idx] = c[idx][rng.permuted(np.tile(np.arange(len(idx)), (n, 1)), axis=1)]
    b = ba(P, o[None, :])
    return obs, float((1 + (b >= obs - 1e-12).sum()) / (n + 1))


def summary_row(g, label, outcome="outcome", perm=True):
    c, o = g.call.to_numpy(int), g[outcome].to_numpy(int)
    row = dict(subset=label, n=len(g), n_go=int(c.sum()), n_outcome=int(o.sum()),
               base_rate=float(o.mean()), go_precision=float(o[c == 1].mean()) if c.sum() else np.nan,
               recall=float(c[o == 1].mean()) if o.sum() else np.nan, ba=float(ba(c, o)) if 0 < o.sum() < len(o) else np.nan)
    if perm and 0 < o.sum() < len(o) and g.dataset.nunique() >= 1:
        _, p = perm_within(c, o, g.dataset.to_numpy())
        row["p_perm_within_dataset"] = p
    return row


def cells() -> pd.DataFrame:
    df = X.long_results()
    d = df[df.dataset.isin(X.DATASETS) & df.pair.isin(["A", "B"])]
    a = X.wide(d, ["audit_go", "audit_C_val", "audit_C_lo", "outcome"], router="R6")
    b = X.wide(d, ["C", "capture"], router="R6")
    e = X.wide(d, ["ba_ef_ev", "ba_cf_cv"])
    bb = X.wide(d, ["ba_at_budget"], router="R6", budget=0.10).rename(columns={"ba_at_budget": "ba_R6_10"})
    k = ["dataset", "pair", "split", "h2_q"]
    c = a.merge(b, on=k, how="left").merge(e, on=k, how="left").merge(bb, on=k, how="left")
    c = c.rename(columns={"audit_go": "call", "C": "C_test", "capture": "cap_test"})
    c["deploy_gain_10"] = c.ba_R6_10 - c.ba_ef_ev
    c["deploy"] = (c.deploy_gain_10 > 1e-12).astype(int)
    c["call"] = c.call.astype(int); c["outcome"] = c.outcome.fillna(0).astype(int)  # NaN capture -> outcome 0, as in analyze.py
    return c


def main():
    c = cells()
    assert len(c) == 400, len(c)
    rows = [summary_row(c, "all cells (pre-specified)"), summary_row(c[c.h2_q == 1.0], "q = 1 only")]
    for ds in X.DATASETS:
        rows.append(summary_row(c[c.dataset == ds], f"dataset={ds}", perm=False))
        rows.append(summary_row(c[(c.dataset == ds) & (c.h2_q == 1.0)], f"dataset={ds}, q=1", perm=False))
    for q in sorted(c.h2_q.unique()):
        rows.append(summary_row(c[c.h2_q == q], f"level q={q}"))
        for ds in X.DATASETS:
            rows.append(summary_row(c[(c.h2_q == q) & (c.dataset == ds)], f"dataset={ds}, q={q}", perm=False))
    for ex in ("seu", "threew", "hydraulic", "skab"):
        rows.append(summary_row(c[c.dataset != ex], f"excluding {ex}"))
    br = pd.DataFrame(rows)
    br.to_csv(X.OUT / "h5_breakdown.csv", index=False)

    # block permutation: keep dataset x split blocks intact
    rng = np.random.default_rng(SEED + 1)
    c["cellkey"] = c.pair + "|" + c.h2_q.astype(str)
    piv_c = c.pivot_table(index=["dataset", "split"], columns="cellkey", values="call")
    piv_o = c.pivot_table(index=["dataset", "split"], columns="cellkey", values="outcome")
    assert piv_c.notna().all().all()
    Cm, Om = piv_c.to_numpy(int), piv_o.to_numpy(int)
    dsv = piv_c.index.get_level_values(0).to_numpy()
    obs = float(ba(Cm.ravel(), Om.ravel()))
    cnt = 0
    for _ in range(N_PERM):
        P = Cm.copy()
        for ds in np.unique(dsv):
            idx = np.flatnonzero(dsv == ds)
            P[idx] = Cm[rng.permutation(idx)]
        cnt += ba(P.ravel(), Om.ravel()) >= obs - 1e-12
    p_block = (1 + cnt) / (N_PERM + 1)
    # same at q = 1 only (pairs A and B, two cells per block)
    cols1 = [k for k in piv_c.columns if k.endswith("|1.0")]
    C1, O1 = piv_c[cols1].to_numpy(int), piv_o[cols1].to_numpy(int)
    obs1 = float(ba(C1.ravel(), O1.ravel())); cnt1 = 0
    for _ in range(N_PERM):
        P = C1.copy()
        for ds in np.unique(dsv):
            idx = np.flatnonzero(dsv == ds)
            P[idx] = C1[rng.permutation(idx)]
        cnt1 += ba(P.ravel(), O1.ravel()) >= obs1 - 1e-12
    pd.DataFrame([dict(subset="all cells", ba=obs, p_block_perm=p_block, n_blocks=len(Cm), n_perm=N_PERM),
                  dict(subset="q = 1 only", ba=obs1, p_block_perm=(1 + cnt1) / (N_PERM + 1), n_blocks=len(C1),
                       n_perm=N_PERM)]).to_csv(X.OUT / "h5_block_perm.csv", index=False)

    # OOF validation C against test C
    from sklearn.metrics import roc_auc_score
    oo = c.groupby(["dataset", "pair", "h2_q"]).agg(n=("call", "size"), n_val_C_defined=("audit_C_val", "count"),
                                                     mean_C_val_oof=("audit_C_val", "mean"),
                                                     mean_C_test=("C_test", "mean"),
                                                     share_C_val_below_half=("audit_C_val", lambda s: float((s < 0.5).mean()))).reset_index()
    tot = dict(dataset="ALL", pair="", h2_q=np.nan, n=len(c), n_val_C_defined=int(c.audit_C_val.notna().sum()),
               mean_C_val_oof=float(c.audit_C_val.mean()), mean_C_test=float(c.C_test.mean()),
               share_C_val_below_half=float((c.audit_C_val.dropna() < 0.5).mean()))
    aucs = []
    for ds in X.DATASETS:
        g = c[(c.dataset == ds) & c.audit_C_val.notna()]
        aucs.append(dict(dataset=ds, pair="AUC", h2_q=np.nan, n=len(g),
                         auc_C_val_for_outcome=roc_auc_score(g.outcome, g.audit_C_val) if g.outcome.nunique() == 2 else np.nan,
                         mean_C_val_oof=float(g.audit_C_val.mean()), mean_C_test=float(g.C_test.mean())))
    pd.concat([oo, pd.DataFrame([tot]), pd.DataFrame(aucs)], ignore_index=True).to_csv(X.OUT / "h5_oof_vs_test.csv", index=False)

    # deployment-relevant secondary outcome
    dep = [summary_row(c, "all cells: audit vs 'escalation beats edge at 10%'", outcome="deploy"),
           summary_row(c[c.h2_q == 1.0], "q = 1: audit vs 'escalation beats edge at 10%'", outcome="deploy")]
    for ds in X.DATASETS:
        g = c[c.dataset == ds]
        pay = g[g.outcome == 1]
        dep.append(dict(subset=f"dataset={ds}", n=len(g), n_outcome=int(g.outcome.sum()),
                        n_deploy=int(g.deploy.sum()), n_paying_without_gain=int((pay.deploy_gain_10 <= 1e-12).sum()),
                        ba=float(ba(g.call, g.deploy)) if 0 < g.deploy.sum() < len(g) else np.nan,
                        go_precision=float(g.deploy[g.call == 1].mean()) if g.call.sum() else np.nan,
                        base_rate=float(g.deploy.mean())))
    pay = c[c.outcome == 1]
    dep.append(dict(subset="paying cells (capture > 0.10) without BA gain over edge at 10%",
                    n=int(len(pay)), n_paying_without_gain=int((pay.deploy_gain_10 <= 1e-12).sum())))
    pd.DataFrame(dep).to_csv(X.OUT / "h5_deploy_outcome.csv", index=False)

    fails = c[(c.call == 1) & (c.cap_test < 0)][["dataset", "pair", "split", "h2_q", "audit_C_lo", "C_test",
                                                 "cap_test", "deploy_gain_10"]]
    fails.to_csv(X.OUT / "h5_go_failures.csv", index=False)
    c.drop(columns=["cellkey"]).to_csv(X.OUT / "h5_cells_extra.csv", index=False)
    pd.set_option("display.width", 250)
    print(br.round(3).to_string())
    print(pd.read_csv(X.OUT / "h5_block_perm.csv"))
    print(pd.read_csv(X.OUT / "h5_oof_vs_test.csv").round(3).to_string())
    print(pd.DataFrame(dep).round(3).to_string())
    print(fails.round(3).to_string())


if __name__ == "__main__":
    main()
