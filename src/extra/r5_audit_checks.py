"""POST-PLAN, evaluation only. Refits no model.

1. R1 minus the random router R0 in balanced accuracy at a 10% budget (h2_q = 1) for every stored hydraulic
   control (results_sens/hydskab/hyd_* and results_sens/holdout2/hyd_*), per pair, as a mean over the stored
   splits or hold-out folds. For runs with at least 10 splits the Nadeau-Bengio interval is added with the
   n_train/n_test of each split; hold-outs (3 or 4 folds) get no interval, as in the paper.
   R0 is the seeded random router of the run, not its exact expectation, so the pre-specified and block-grouped
   rows differ from Table 6 (exact expectation) in the third decimal.
2. Raw SKAB anomaly timing: per experiment file the first and last second with anomaly == 1, measured from the
   first timestamp of the file (data/raw/skab/repo/data/{other,valve1,valve2}/*.csv).

Outputs: results/extra/r5_r1_random_hyd_controls.csv, results/extra/r5_skab_raw_anomaly_timing.csv
    python src/extra/r5_audit_checks.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]


def nb(d, n_train, n_test):
    d = np.asarray(d, float)
    J = len(d)
    m = float(d.mean())
    if J < 10:
        return J, m, np.nan, np.nan
    se = float(d.std(ddof=1)) * np.sqrt(1.0 / J + n_test / n_train)
    t = stats.t.ppf(0.975, J - 1)
    return J, m, m - t * se, m + t * se


def r1_random():
    rows = []
    runs = sorted(glob.glob(str(ROOT / "results_sens" / "hydskab" / "hyd_*"))) + \
        sorted(glob.glob(str(ROOT / "results_sens" / "holdout2" / "hyd_*")))
    for run in runs:
        for pair in "AB":
            fs = sorted(glob.glob(f"{run}/{pair}/split_*.parquet"))
            if not fs:
                continue
            d = pd.concat([pd.read_parquet(f) for f in fs])
            b = d[(d.quantity == "ba_at_budget") & (d.h2_q == 1.0) & (np.abs(d.budget - 0.1) < 1e-9)]
            t = b.pivot_table(index="split", columns="router", values="value")
            if "R1" not in t or "R0" not in t:
                continue
            x = (t.R1 - t.R0).dropna()
            sz = d.drop_duplicates("split").set_index("split")
            J, m, lo, hi = nb(x.values, float(sz.n_train.mean()), float(sz.n_test.mean()))
            rows.append(dict(run=Path(run).parent.name + "/" + Path(run).name, pair=pair, J=J,
                             R1_minus_R0_mean=m, ci_lo=lo, ci_hi=hi,
                             per_split=";".join(f"{v:.4f}" for v in x.values)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "extra" / "r5_r1_random_hyd_controls.csv", index=False)
    print(out.round(4).to_string())


def skab_timing():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "skab" / "repo" / "data" / "*" / "*.csv"))):
        if "anomaly-free" in f:
            continue
        d = pd.read_csv(f, sep=";", parse_dates=["datetime"])
        t = (d.datetime - d.datetime.iloc[0]).dt.total_seconds()
        a = t[d.anomaly == 1]
        rows.append(dict(file=str(Path(f).relative_to(ROOT / "data" / "raw" / "skab" / "repo" / "data")),
                         first_anomalous_s=float(a.min()), last_anomalous_s=float(a.max())))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "extra" / "r5_skab_raw_anomaly_timing.csv", index=False)
    on, en = out.first_anomalous_s, out.last_anomalous_s
    print(len(out), "files; onset 597-601 s:", int(((on >= 597) & (on <= 601)).sum()),
          "; end 1018-1023 s:", int(((en >= 1018) & (en <= 1023)).sum()))


if __name__ == "__main__":
    r1_random()
    skab_timing()
