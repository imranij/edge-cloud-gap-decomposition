"""POST-PLAN: pilot to set mu_c so that the information steps span about 0, 0.03, 0.08 BA.

For each (lam, mu_c) on a grid: draws of the CV-sized design (fit 24 groups, validation 8 groups,
half positive), LR (pair A edge family) on the edge and on the cloud view, thresholds by
threshold_max_ba on validation, BA on an independent 20,000-window test sample.

    OMP_NUM_THREADS=1 nice -n 19 python src/sens/simulation/pilot.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from generator import Params, sample                          # noqa: E402
from escalation.config import PLAN                             # noqa: E402
from escalation.models import make_model                       # noqa: E402
from escalation import metrics as M                            # noqa: E402

OUT = HERE.parents[2] / "results_sens" / "simulation"


def info_step(p: Params, rng, m=25, n_test_groups=800):
    tr = sample(p, rng, 16, 16, m)
    te = sample(p, rng, n_test_groups // 2, n_test_groups // 2, m)
    gid = np.array([int(g[1:]) for g in tr["groups"]])
    val = np.isin(gid, [12, 13, 14, 15, 28, 29, 30, 31])          # 4 positive + 4 negative groups
    fit = ~val
    out = {}
    for view in ("X_edge", "X_cloud"):
        mdl = make_model("A", "edge", PLAN.model_seed, PLAN).fit(tr[view][fit], tr["y"][fit])
        t = M.threshold_max_ba(mdl.predict_proba(tr[view][val]), tr["y"][val])
        out[view] = M.balanced_accuracy(te["y"], (mdl.predict_proba(te[view]) >= t).astype(int))
    return out["X_cloud"] - out["X_edge"], out["X_edge"]


GRID1 = [(lam, mu) for lam in (0.0, 1.0) for mu in (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0) if not (lam > 0 and mu == 0)]
GRID2 = [(1.0, 2.25), (1.0, 2.5), (1.0, 3.0)]          # second pilot round: lam = 1 saturates


def main(n_draws=30, grid=None, fn="pilot.csv"):
    rows = []
    for lam, mu in (grid or GRID1):
        if True:
            rng = np.random.default_rng([7001, int(lam * 10), int(mu * 100)])
            v = np.array([info_step(Params(mu_c=mu, lam=lam), rng) for _ in range(n_draws)])
            rows.append(dict(lam=lam, mu_c=mu, n_draws=n_draws, info_step=v[:, 0].mean(),
                             info_step_mcse=v[:, 0].std(ddof=1) / np.sqrt(n_draws), ba_edge=v[:, 1].mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / fn, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "round2":
        main(grid=GRID2, fn="pilot_round2.csv")
    else:
        main()
