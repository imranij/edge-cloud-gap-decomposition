"""SEU at recording level. Post-plan analysis, not pre-specified. No model refitted.

    python src/extra/seu_recordings.py

Every SEU test fold holds exactly one of the four healthy recordings, so the negative half of
balanced accuracy, g, the oracle and every router result rest on one recording per split.
Outputs (results/extra/):
  seu_specificity.csv        per pair x model: specificity on the healthy test recording in every
                             split; share of (split, recording) cases with specificity >= 0.5;
                             share with specificity <= 0.1 or >= 0.9; per healthy recording the mean
                             specificity and the share of its test appearances classified correctly
  seu_by_healthy_recording.csv  leave-one-healthy-recording enumeration: for each of the four healthy
                             recordings, the mean over the splits in which it is the test negative of
                             B_eE, B_cE, B_eC, B_cC, the oracle, the Shapley model and information
                             steps, C and capture of R1 and R6; plus a recording-level summary
                             (mean of the four recording means, t interval with 3 df, min and max)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as X

MODELS = ["ef_ev", "cf_ev", "ef_cv", "cf_cv"]


def main():
    df = X.long_results()
    rows, rec_rows = [], []
    for pair in ("A", "B", "C"):
        sz = X.sizes()
        J = int(sz[(sz.dataset == "seu") & (sz.pair == pair)].J.iloc[0])
        for s in range(J):
            z = X.arrays("seu", pair, s)
            y, grp = z["y_test"].astype(int), z["groups_test"]
            neg = np.unique(grp[y == 0])
            assert len(neg) == 1, (pair, s, neg)
            for k in MODELS:
                pred = (z[f"p_{k}"] >= float(z[f"t_{k}"])).astype(int)
                rows.append(dict(pair=pair, split=s, healthy=neg[0], model=k,
                                 specificity=float((pred[y == 0] == 0).mean()),
                                 recall=float((pred[y == 1] == 1).mean())))
    sp = pd.DataFrame(rows)
    sp.to_csv(X.OUT / "seu_specificity_per_split.csv", index=False)
    summ = sp.groupby(["pair", "model"]).agg(n_cases=("specificity", "size"), n_distinct_healthy=("healthy", "nunique"),
                                             share_correct=("specificity", lambda v: float((v >= 0.5).mean())),
                                             share_extreme=("specificity", lambda v: float(((v <= 0.1) | (v >= 0.9)).mean())),
                                             mean_specificity=("specificity", "mean")).reset_index()
    per_rec = sp.groupby(["pair", "model", "healthy"]).agg(n=("specificity", "size"),
                                                          mean_spec=("specificity", "mean"),
                                                          share_correct=("specificity", lambda v: float((v >= 0.5).mean()))).reset_index()
    rec_correct = per_rec.groupby(["pair", "model"]).agg(
        recordings_mostly_correct=("share_correct", lambda v: int((v > 0.5).sum()))).reset_index()
    summ = summ.merge(rec_correct, on=["pair", "model"])
    summ.to_csv(X.OUT / "seu_specificity.csv", index=False)
    per_rec.to_csv(X.OUT / "seu_specificity_by_recording.csv", index=False)

    # leave-one-healthy-recording enumeration from the stored per-split quantities
    seu = df[(df.dataset == "seu") & (df.h2_q == 1.0)]
    hmap = sp.drop_duplicates(["pair", "split"]).set_index(["pair", "split"])["healthy"]
    q = ["ba_ef_ev", "ba_cf_ev", "ba_ef_cv", "ba_cf_cv", "ba_oracle_dep", "shapley_model", "shapley_info"]
    out = []
    for pair in ("A", "B", "C"):
        d = seu[seu.pair == pair]
        w = X.wide(d, q)
        for rt in ("R1", "R6"):
            for qq in ("C", "capture"):
                v = X.per_split(d, qq, rt, name=f"{qq}_{rt}").reset_index()
                w = w.merge(v[["split", f"{qq}_{rt}"]], on="split", how="left")
        w["healthy"] = [hmap[(pair, s)] for s in w.split]
        cols = [c for c in w.columns if c not in ("dataset", "pair", "split", "h2_q", "healthy")]
        m = w.groupby("healthy")[cols].mean()
        n = w.groupby("healthy").size()
        for h, r in m.iterrows():
            out.append(dict(pair=pair, healthy=h, n_splits=int(n[h]), **r.to_dict()))
        for c in cols:
            vals = m[c].to_numpy()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            tc = stats.t.ppf(0.975, len(vals) - 1)
            out.append(dict(pair=pair, healthy=f"SUMMARY {c}", n_splits=len(vals), mean_of_recordings=vals.mean(),
                            ci_lo=vals.mean() - tc * se, ci_hi=vals.mean() + tc * se, min=vals.min(), max=vals.max()))
    pd.DataFrame(out).to_csv(X.OUT / "seu_by_healthy_recording.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    print(per_rec[per_rec.pair == "A"].round(3).to_string())
    o = pd.DataFrame(out)
    print(o[o.pair == "A"].round(3).to_string())


if __name__ == "__main__":
    main()
