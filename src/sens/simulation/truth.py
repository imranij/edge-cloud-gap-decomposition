"""POST-PLAN: Monte Carlo ground truth per scenario. Post-plan, not pre-specified.

    OMP_NUM_THREADS=1 nohup nice -n 19 python src/sens/simulation/truth.py --draws 200 > logs/sens/sim_truth.log 2>&1 &

One draw: 40 fresh groups (20 positive, 20 negative, 25 windows each) for training and an independent
test sample of 2,000 groups (50,000 windows). Primary sizes match the cross-validated design the tests
see (K = 5 outer, K_in = 4 inner): fit = 24 groups (600 windows: positive groups 0-11, negative
20-31), validation = 8 groups (200 windows: positive 16-19, negative 36-39). Secondary sizes as in the
task text: fit = 32 groups (800 windows: positive 0-15, negative 20-35), same validation.
Models, thresholds, routers and metrics are the same calls as pipeline.py (seed 3000, R0 seed 5000).
Per draw it records the information step, BAs, C and capture of R0, R1, R6 and of the
noise-augmented pair R6a/R6b, BA at a 10% budget, rho_pos, and the H2 curves (fixed-tier and joint)
with each draw's OLS slope over the four levels. The expectation over draws is the ground truth;
results_sens/simulation/truth_draws_<scenario>.csv holds the draws (resumable per scenario).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from generator import SCENARIOS, sample                               # noqa: E402
from pipeline import N_NOISE                                          # noqa: E402
from escalation.config import PLAN, CAPTURE_RANGE, H2_LEVELS          # noqa: E402
from escalation import metrics as M                                   # noqa: E402
from escalation import routers as RT                                  # noqa: E402
from escalation.models import make_model                              # noqa: E402
from escalation.splits import h2_subsample                            # noqa: E402
from run_dataset import spearman                                      # noqa: E402

OUT = HERE.parents[2] / "results_sens" / "simulation"
SCEN_ORDER = ["S0", "S1", "S1b", "S2", "S3"]
SEED = PLAN.model_seed


def tiers(tr, te, fit, val):
    o = {}
    for k, view in (("e", "X_edge"), ("c", "X_cloud")):
        m = make_model("A", "edge", SEED, PLAN).fit(tr[view][fit], tr["y"][fit])
        pv = m.predict_proba(tr[view][val])
        o[k] = (pv, m.predict_proba(te[view]), M.threshold_max_ba(pv, tr["y"][val]))
    return o


def r6(Zv, tg, Zt):
    return RT.fit_regressor_router(Zv, tg["g"], SEED, PLAN, sample_weight=tg["w"]).predict(Zt)


def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 and np.ptp(x[ok]) > 0 else np.nan


def one_draw(scen: str, d: int) -> dict:
    p = SCENARIOS[scen]
    rng = np.random.default_rng([20261012, 99, SCEN_ORDER.index(scen), d])
    tr = sample(p, rng, 20, 20, 25)
    te = sample(p, rng, 1000, 1000, 25, group_prefix="t")
    gid = np.repeat(np.arange(40), 25)
    val = np.flatnonzero(np.isin(gid, [16, 17, 18, 19, 36, 37, 38, 39]))
    fit = np.flatnonzero(np.isin(gid, list(range(0, 12)) + list(range(20, 32))))
    fit800 = np.flatnonzero(np.isin(gid, list(range(0, 16)) + list(range(20, 36))))
    y, yt, Xe = tr["y"], te["y"], tr["X_edge"]
    yv = y[val]
    T = tiers(tr, te, fit, val)
    (pe_v, pe_t, tE), (pc_v, pc_t, tC) = T["e"], T["c"]
    pe_b, pc_b = (pe_t >= tE).astype(int), (pc_t >= tC).astype(int)
    r = dict(scenario=scen, draw=d, ba_edge=M.balanced_accuracy(yt, pe_b), ba_cloud=M.balanced_accuracy(yt, pc_b))
    r["info_step"] = r["ba_cloud"] - r["ba_edge"]
    g = M.escalation_gain(yt, pe_b, pc_b)
    Zv, Zt = RT.router_inputs(Xe[val], pe_v), RT.router_inputs(te["X_edge"], pe_t)
    tg = RT.router_targets(yv, pe_v, pc_v, tE, tC)
    na_v, na_t = rng.normal(size=(len(val), N_NOISE)), rng.normal(size=(len(yt), N_NOISE))
    nb_v, nb_t = rng.normal(size=(len(val), N_NOISE)), rng.normal(size=(len(yt), N_NOISE))
    sc = {"R0": RT.r0_random(len(yt), PLAN.router_seed), "R1": RT.r1_confidence(pe_t, tE), "R6": r6(Zv, tg, Zt),
          "R6a": r6(np.column_stack([Zv, na_v]), tg, np.column_stack([Zt, na_t])),
          "R6b": r6(np.column_stack([Zv, nb_v]), tg, np.column_stack([Zt, nb_t]))}
    for k, s in sc.items():
        r[f"C_{k}"] = M.weighted_concordance(g, s)
        r[f"cap_{k}"] = M.capture(g, s, *CAPTURE_RANGE)
    for k in ("R1", "R6"):
        r[f"ba10_{k}"] = M.ba_at_budget(yt, pe_b, pc_b, sc[k], 0.10)
    r["rho_pos_test"] = spearman(sc["R1"][yt == 1], g[yt == 1])
    r["share_g_nonzero"] = float((g != 0).mean())
    # H2 curves
    xs_f, ys_f, xs_j, ys_j, ys_j1 = [np.log2(yv.sum())], [r["C_R6"]], [np.log2(y[fit].sum() + yv.sum())], \
        [r["C_R6"]], [r["C_R1"]]
    for q in H2_LEVELS:
        fq, vq = h2_subsample(y, tr["groups"], fit, val, q, d, PLAN)
        pos = np.searchsorted(val, vq)
        tgq = RT.router_targets(y[vq], pe_v[pos], pc_v[pos], tE, tC)
        r[f"h2f_C_R6_q{q}"] = M.weighted_concordance(g, r6(Zv[pos], tgq, Zt))
        xs_f.append(np.log2(y[vq].sum())); ys_f.append(r[f"h2f_C_R6_q{q}"])
        Tq = tiers(tr, te, fq, vq)
        (qe_v, qe_t, qtE), (qc_v, qc_t, qtC) = Tq["e"], Tq["c"]
        gq = M.escalation_gain(yt, (qe_t >= qtE).astype(int), (qc_t >= qtC).astype(int))
        tgj = RT.router_targets(y[vq], qe_v, qc_v, qtE, qtC)
        r[f"h2j_C_R6_q{q}"] = M.weighted_concordance(gq, r6(RT.router_inputs(Xe[vq], qe_v), tgj,
                                                            RT.router_inputs(te["X_edge"], qe_t)))
        r[f"h2j_C_R1_q{q}"] = M.weighted_concordance(gq, RT.r1_confidence(qe_t, qtE))
        xs_j.append(np.log2(y[fq].sum() + y[vq].sum())); ys_j.append(r[f"h2j_C_R6_q{q}"])
        ys_j1.append(r[f"h2j_C_R1_q{q}"])
    r["h2f_slope_C_R6"] = slope(xs_f, ys_f)                     # = slope of C(R6) - C(R1): R1 fixed
    r["h2j_slope_C_R6"] = slope(xs_j, ys_j)
    r["h2j_slope_dC_R6_R1"] = slope(xs_j, np.array(ys_j) - np.array(ys_j1))
    # secondary sizes: fit 800, same validation
    T8 = tiers(tr, te, fit800, val)
    r["info_step_fit800"] = (M.balanced_accuracy(yt, (T8["c"][1] >= T8["c"][2]).astype(int))
                             - M.balanced_accuracy(yt, (T8["e"][1] >= T8["e"][2]).astype(int)))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--scenarios", default=",".join(SCEN_ORDER))
    a = ap.parse_args(argv)
    for sc in a.scenarios.split(","):
        fn = OUT / f"truth_draws_{sc}.csv"
        done = pd.read_csv(fn) if fn.exists() else pd.DataFrame()
        have = set(done["draw"]) if len(done) else set()
        rows = done.to_dict("records")
        t0 = time.time()
        for d in range(a.draws):
            if d in have:
                continue
            rows.append(one_draw(sc, d))
            if (d + 1) % 20 == 0 or d == a.draws - 1:
                pd.DataFrame(rows).to_csv(fn.with_suffix(".tmp"), index=False)
                fn.with_suffix(".tmp").replace(fn)
                print(f"{sc} {d + 1}/{a.draws} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(fn.with_suffix(".tmp"), index=False)
        fn.with_suffix(".tmp").replace(fn)


if __name__ == "__main__":
    main()
