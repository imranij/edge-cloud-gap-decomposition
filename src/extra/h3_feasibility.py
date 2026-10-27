"""Feasibility of the H3 equivalence test at the realised design (J = 20 splits).
Post-plan analysis, not pre-specified. No model is refitted.

    python src/extra/h3_feasibility.py

The plan judged TOST feasibility against the asymptotic floor sd*sqrt(n_test/n_train), which
omits the 1/J term of the Nadeau-Bengio SE and, for the pooled test, the HKSJ max(1, q) factor.
Here feasibility is judged at the J actually run:
  per dataset  se_J = sd*sqrt(1/J + n_test/n_train); feasible iff t_{0.95, J-1}*se_J < margin;
               the largest |dC| that could still give p_eq < 0.05 is margin - t_{0.95,J-1}*se_J.
  pooled       the smallest attainable HKSJ SE (tau^2 = 0 and max(1, q) = 1) is
               (sum_d 1/se_J,d^2)^(-1/2); feasible iff t_{0.95,3}*se_min < margin.
Also reported: p_eq at an observed dC of exactly 0, the J that would make each test feasible
(sd and n_test/n_train held at their realised values), and the margin that would have been
feasible at J = 20. Output: results/extra/h3_feasibility.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as X

MARGIN = 0.02


def main():
    pf = pd.read_csv(X.SUM / "power_floors.csv")
    pf = pf[pf.test == "H3 dC"].set_index("dataset")
    h3 = pd.read_csv(X.SUM / "h3_per_dataset.csv").set_index("dataset")
    pooled = pd.read_csv(X.SUM / "h3_pooled.csv").iloc[0]
    sz = X.sizes()
    rows, se_J = [], {}
    for ds in X.DATASETS:
        s = sz[(sz.dataset == ds) & (sz.pair == "A")].iloc[0]
        r = s.n_test / s.n_train
        sd = float(pf.loc[ds, "sd"]); J = int(pf.loc[ds, "J"])
        se = sd * np.sqrt(1.0 / J + r)
        tc = stats.t.ppf(0.95, J - 1)
        se_J[ds] = (sd, r)
        need = next((JJ for JJ in range(2, 100001) if stats.t.ppf(0.95, JJ - 1) * sd * np.sqrt(1 / JJ + r) < MARGIN),
                    None)
        rows.append(dict(test=ds, J=J, sd=sd, ratio_test_train=r, se_floor_asymptotic=sd * np.sqrt(r),
                         se_at_J=se, se_reported=float(h3.loc[ds, "se"]), t_crit=tc,
                         feasible_at_J=bool(tc * se < MARGIN), max_abs_dC_for_equivalence=max(MARGIN - tc * se, 0.0),
                         p_eq_if_dC_zero=float(stats.t.sf(MARGIN / se, J - 1)),
                         J_needed=need, margin_feasible_at_J=tc * se,
                         observed_dC=float(h3.loc[ds, "mean"]), observed_p_eq=float(h3.loc[ds, "p_eq"])))

    def pooled_min_se(J):
        w = sum(1.0 / (sd * np.sqrt(1.0 / J + r)) ** 2 for sd, r in se_J.values())
        return 1.0 / np.sqrt(w)
    tc3 = stats.t.ppf(0.95, 3)
    se_min = pooled_min_se(20)
    needJ = next((JJ for JJ in range(2, 100001) if tc3 * pooled_min_se(JJ) < MARGIN), None)
    rows.append(dict(test="pooled", J=20, se_floor_asymptotic=float(pooled.pooled_se_floor), se_at_J=se_min,
                     se_reported=float(pooled.se), t_crit=tc3, feasible_at_J=bool(tc3 * se_min < MARGIN),
                     max_abs_dC_for_equivalence=max(MARGIN - tc3 * se_min, 0.0),
                     p_eq_if_dC_zero=float(stats.t.sf(MARGIN / se_min, 3)), J_needed=needJ,
                     margin_feasible_at_J=tc3 * se_min, observed_dC=float(pooled["mean"]),
                     observed_p_eq=float(pooled.p_eq),
                     asymptotic_floor_times_t3=tc3 * float(pooled.pooled_se_floor)))
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "h3_feasibility.csv", index=False)
    print(out.round(5).to_string())


if __name__ == "__main__":
    main()
