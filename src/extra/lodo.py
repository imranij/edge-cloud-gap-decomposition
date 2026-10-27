"""Leave-one-dataset-out versions of the pooled H1 estimate and of the H2, H4 and H5 models.
Post-plan analysis, not pre-specified. No model is refitted; the estimators are those of
src/analyze.py (HKSJ pooling of Nadeau-Bengio per-dataset estimates, MixedLM with dataset random
intercepts, BA of audit calls with a within-dataset permutation p-value).

    python src/extra/lodo.py      -> results/extra/lodo.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as X
from h5_audit_breakdown import cells as h5cells, perm_within


def main():
    df = X.long_results()
    h1 = pd.read_csv(X.SUM / "h1_per_dataset.csv").set_index("dataset")
    h2d = df[df.dataset.isin(X.DATASETS) & (df.pair == "A") & (df.quantity == "C") & (df.router == "R6")].copy()
    h2d["log2_npos"] = np.log2(h2d.n_pos_train.clip(lower=1))
    h4d = pd.read_csv(X.SUM / "h4_data.csv")
    h5c = h5cells()
    rows = []
    for ex in [None] + X.DATASETS:
        keep = [d for d in X.DATASETS if d != ex]
        tag = "none (all four)" if ex is None else ex
        p = X.S.pool_hksj(h1.loc[keep, "mean"], h1.loc[keep, "se"])
        rows.append(dict(excluded=tag, test="H1 pooled Shapley information step", estimate=p["mean"],
                         ci_lo=p["ci_lo"], ci_hi=p["ci_hi"], df=p["df"],
                         p_one_sided=float(stats.t.sf(p["mean"] / p["se"], p["df"]))))
        r = X.mixed_slope(h2d[h2d.dataset.isin(keep)][["dataset", "value", "log2_npos"]].rename(columns={"value": "C"}),
                          "C ~ log2_npos", "log2_npos")
        rows.append(dict(excluded=tag, test="H2 slope of C(R6) per doubling", estimate=r["slope"], ci_lo=r["ci_lo"],
                         ci_hi=r["ci_hi"], p_one_sided=float(stats.norm.sf(r["z"])), n=r["n"], note=r["note"]))
        r = X.mixed_slope(h4d[h4d.dataset.isin(keep)][["dataset", "dcap", "rho_pos"]], "dcap ~ rho_pos", "rho_pos")
        rows.append(dict(excluded=tag, test="H4 slope of dcap on rho_pos", estimate=r["slope"], ci_lo=r["ci_lo"],
                         ci_hi=r["ci_hi"], p_one_sided=float(stats.norm.cdf(r["z"])), n=r["n"], note=r["note"]))
        g = h5c[h5c.dataset.isin(keep)]
        b, pp = perm_within(g.call.to_numpy(int), g.outcome.to_numpy(int), g.dataset.to_numpy())
        rows.append(dict(excluded=tag, test="H5 BA of audit calls", estimate=b, p_one_sided=pp, n=len(g),
                         note="within-dataset permutation p (10,000)"))
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "lodo.csv", index=False)
    # R1 question: 95% prediction interval for a new dataset's information step (HKSJ pooled
    # mean, DL tau^2, t with k - 2 df; Higgins-Thompson-Spiegelhalter form)
    p = pd.read_csv(X.SUM / "h1_pooled.csv").iloc[0]
    half = stats.t.ppf(0.975, int(p.k) - 2) * np.sqrt(p.tau2 + p.se ** 2)
    pd.DataFrame([dict(mean=p["mean"], tau2=p.tau2, se=p.se, df=int(p.k) - 2, pi_lo=p["mean"] - half,
                       pi_hi=p["mean"] + half)]).to_csv(X.OUT / "h1_prediction_interval.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["note"]).round(4).to_string())


if __name__ == "__main__":
    main()
