"""POST-PLAN, evaluation only. Refits no model.

Capture over the operating range [0.05, 0.30] is a per-split ratio whose denominator (the oracle
ranking's gain over random) can be small, so means of the ratios can be unstable. For every dataset x pair
x router (R1, R2, R6) of the pre-specified run this gives, from the stored test gains and router scores
(results/<ds>/<pair>/arrays_<j>.npz), the mean of the per-split ratios (as reported), the median of the
ratios, and the ratio of the mean numerator to the mean denominator.
Output: results/extra/capture_ratio_of_means.csv
    python src/extra/capture_ratio_of_means.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

LO, HI = 0.05, 0.30


def num_den(g, s):
    n = len(g); k = np.arange(n + 1)
    sel = (k / n >= LO) & (k / n <= HI) & (k > 0)
    rand = k / n * g.sum()
    return float((X.M.gain_curve(g, s) - rand)[sel].sum()), float((X.M.gain_curve(g, g) - rand)[sel].sum())


def main():
    rows = []
    for ds in X.DATASETS:
        for pair in ("A", "B"):
            for r in ("R1", "R2", "R6"):
                nums, dens = [], []
                for j in range(20):
                    z = X.arrays(ds, pair, j)
                    a, b = num_den(z["g_dep"], z[f"score_{r}"])
                    nums.append(a); dens.append(b)
                nums, dens = np.array(nums), np.array(dens)
                ok = dens > 0
                ratio = nums[ok] / dens[ok]
                rows.append(dict(dataset=ds, pair=pair, router=r, J=int(ok.sum()), mean_of_ratios=ratio.mean(),
                                 median_of_ratios=float(np.median(ratio)), ratio_of_means=nums[ok].mean() / dens[ok].mean(),
                                 min_denominator=dens[ok].min(), median_denominator=float(np.median(dens[ok]))))
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "capture_ratio_of_means.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
