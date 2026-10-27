"""Where on SKAB the information step lives.
Post-plan analysis, not pre-specified. No model is refitted: the stored test probabilities
and validation thresholds of the four factorial models (results/skab/<pair>/arrays_<j>.npz) are
re-scored on subsets of each test fold defined by data/processed/skab/meta.parquet.

    python src/extra/skab_breakdown.py

Subsets of the test negatives:
  free   windows of the anomaly-free recording (groups free_0..free_7)
  pre    normal windows of an experiment file that start before the file's first anomalous window
  post   normal windows of an experiment file that start after the file's first anomalous window
Positives are all fault windows (every one sits in an experiment file).

Per pair (A, B, C), split and factorial model (eE = ef_ev, cE = cf_ev, eC = ef_cv, cC = cf_cv):
  spec_free, spec_pre, spec_post, recall,
  ba_file, auc_file    BA and AUC on the experiment-file windows only (both classes present),
  ba_timestrat         BA within 120-s bins of elapsed time since file start (t_start_s),
                       averaged over the bins that hold at least 5 normal and 5 fault windows
                       of the test fold (experiment files only); a comparison at matched
                       elapsed time, so a pure clock cannot separate the classes inside a bin.
Derived per split:
  info_free/info_pre/info_post/info_fault   Shapley information step on the accuracy of each
                       subset, 0.5 * ((eC - eE) + (cC - cE)),
  contrib_free/pre/post  contribution of each negative subset to the specificity half of the
                       information step, (n_s / n_neg) * 0.25 * ((eC - eE) + (cC - cE)); the three
                       sum to info_spec_half of results/extra/info_step_halves.csv,
  file_model, file_info, file_inter     the decomposition restricted to experiment files,
  timestrat_info       Shapley information step on ba_timestrat.
Outputs: results/extra/skab_breakdown_cells.csv (per split), results/extra/skab_breakdown.csv
(mean and Nadeau-Bengio 95% CI over splits).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as X

MODELS = {"eE": "ef_ev", "cE": "cf_ev", "eC": "ef_cv", "cC": "cf_cv"}
BIN_S = 120
MIN_PER_CLASS = 5


def subset_labels(meta: pd.DataFrame) -> np.ndarray:
    first_anom = meta[meta.label == 1].groupby("group_id").t_start_s.min()
    fa = meta.group_id.map(first_anom)
    lab = np.where(meta.subset == "anomaly-free", "free",
                   np.where(meta.label == 1, "fault",
                            np.where(meta.t_start_s < fa, "pre", "post")))
    return lab


def ba(y, pred):
    y = np.asarray(y).astype(int)
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def shapley_info(d):
    return 0.5 * ((d["eC"] - d["eE"]) + (d["cC"] - d["cE"]))


def main():
    ds = X.dataset("skab")
    meta = ds["meta"]
    sub_all = subset_labels(meta)
    tstart = meta.t_start_s.to_numpy()
    sz = X.sizes()
    rows = []
    for pair in ("A", "B", "C"):
        J = int(sz[(sz.dataset == "skab") & (sz.pair == pair)].J.iloc[0])
        for s in range(J):
            z = X.arrays("skab", pair, s)
            idx = z["test_idx"]
            y = z["y_test"].astype(int)
            sub = sub_all[idx]
            ts = tstart[idx]
            infile = sub != "free"
            n_neg = int((y == 0).sum())
            n_s = {k: int((sub == k).sum()) for k in ("free", "pre", "post", "fault")}
            r = dict(pair=pair, split=s, **{f"n_{k}": v for k, v in n_s.items()})
            bins = (ts // BIN_S).astype(int)
            ok_bins = [b for b in np.unique(bins[infile])
                       if ((bins == b) & infile & (y == 0)).sum() >= MIN_PER_CLASS
                       and ((bins == b) & infile & (y == 1)).sum() >= MIN_PER_CLASS]
            r["n_timestrat_bins"] = len(ok_bins)
            for short, key in MODELS.items():
                p = z[f"p_{key}"].astype(float)
                pred = (p >= float(z[f"t_{key}"])).astype(int)
                for k in ("free", "pre", "post"):
                    m = sub == k
                    r[f"spec_{k}_{short}"] = float((pred[m] == 0).mean()) if m.any() else np.nan
                r[f"recall_{short}"] = float((pred[y == 1] == 1).mean())
                r[f"acc_fault_{short}"] = r[f"recall_{short}"]
                r[f"ba_file_{short}"] = float(ba(y[infile], pred[infile]))
                r[f"auc_file_{short}"] = float(roc_auc_score(y[infile], p[infile]))
                r[f"ba_all_{short}"] = float(ba(y, pred))
                if ok_bins:
                    r[f"ba_timestrat_{short}"] = float(np.mean([ba(y[(bins == b) & infile], pred[(bins == b) & infile])
                                                                for b in ok_bins]))
                else:
                    r[f"ba_timestrat_{short}"] = np.nan
            for k in ("free", "pre", "post"):
                d = {m: r[f"spec_{k}_{m}"] for m in MODELS}
                r[f"info_{k}"] = shapley_info(d)
                r[f"contrib_{k}"] = (n_s[k] / n_neg) * 0.5 * shapley_info(d) if n_s[k] else 0.0
            r["info_fault"] = shapley_info({m: r[f"recall_{m}"] for m in MODELS})
            r["info_spec_half"] = r["contrib_free"] + r["contrib_pre"] + r["contrib_post"]
            r["info_innegs_file"] = shapley_info({m: ((n_s["pre"] * r[f"spec_pre_{m}"] if n_s["pre"] else 0)
                                                      + (n_s["post"] * r[f"spec_post_{m}"] if n_s["post"] else 0))
                                                     / max(1, n_s["pre"] + n_s["post"]) for m in MODELS})
            f = {m: r[f"ba_file_{m}"] for m in MODELS}
            r["file_model"] = 0.5 * ((f["cE"] - f["eE"]) + (f["cC"] - f["eC"]))
            r["file_info"] = shapley_info(f)
            r["file_inter"] = f["cC"] - f["cE"] - f["eC"] + f["eE"]
            a = {m: r[f"ba_all_{m}"] for m in MODELS}
            r["all_info"] = shapley_info(a)
            t = {m: r[f"ba_timestrat_{m}"] for m in MODELS}
            r["timestrat_info"] = shapley_info(t)
            rows.append(r)
    c = pd.DataFrame(rows)
    c.to_csv(X.OUT / "skab_breakdown_cells.csv", index=False)
    out = []
    qcols = [k for k in c.columns if k not in ("pair", "split")]
    for pair, g in c.groupby("pair"):
        for q in qcols:
            v = g[q].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) >= 3:
                t = X.nb(v, "skab", pair)
                out.append(dict(pair=pair, quantity=q, J=t["J"], mean=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"],
                                p_two_sided=t["p"]))
            else:
                out.append(dict(pair=pair, quantity=q, J=len(v), mean=float(np.mean(v)) if len(v) else np.nan,
                                ci_lo=np.nan, ci_hi=np.nan, p_two_sided=np.nan))
    o = pd.DataFrame(out)
    o.to_csv(X.OUT / "skab_breakdown.csv", index=False)
    pd.set_option("display.width", 200)
    keep = [q for q in o.quantity.unique() if not q.startswith("acc_fault")]
    print(o[o.quantity.isin(keep)].pivot(index="quantity", columns="pair", values="mean").round(3).to_string())
    # check: contributions add up to the stored specificity half
    h = pd.read_csv(X.OUT / "info_step_halves.csv")
    h = h[(h.dataset == "skab") & (h.quantity == "info_spec_half")].set_index("pair")["mean"]
    chk = c.groupby("pair").info_spec_half.mean()
    print("check spec half (stored vs recomputed):", h.round(4).to_dict(), chk.round(4).to_dict())


if __name__ == "__main__":
    main()
