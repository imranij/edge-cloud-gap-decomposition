"""POST-PLAN: Monte Carlo ground truth of the full validation. Post-plan, not pre-specified.

Base family: src/sens/simulation/truth.one_draw UNCHANGED (training sample of 40 groups, fit 24 / validation 8
groups, independent test sample of 2,000 groups).

T and C families: one draw = one training dataset from the scenario's generator; the split used is split
(draw mod 20) of make_splits on it (the next valid one if that split is skipped), so that the truth averages over
the same fit/validation group sets as the cross-validated estimate; tiers are fitted exactly as in
pipe_full.tiers_ba; the test sample is TEST_COPIES independent datasets of the same scenario (fresh block or
file effects, independent of the training series), so time-neighbour leakage cannot reach it. For C the truth is
computed for every design (plain, z, aug) on the same draw. This is the step that would hold on new data of the
same process ("deployment truth"); the cross-validated expectation is estimated in the summary from the replicates.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "simulation")); sys.path.insert(0, str(HERE.parents[1]))
import truth as BT                                                     # noqa: E402
from gen_full import family, T_SCENARIOS, C_SCENARIOS, sample_T, sample_C  # noqa: E402
from pipe_full import control_views, J_FULL                            # noqa: E402
from escalation.config import PLAN                                     # noqa: E402
from escalation import metrics as M                                    # noqa: E402
from escalation.splits import make_splits                              # noqa: E402
from escalation.models import make_model                              # noqa: E402

TRUTH_SEED = 20261013
TEST_COPIES = 10
NEW_ORDER = list(T_SCENARIOS) + list(C_SCENARIOS)


def _concat(parts: list[dict], fam: str) -> dict:
    out = dict(family=fam, y=np.concatenate([p["y"] for p in parts]),
               X_edge=np.concatenate([p["X_edge"] for p in parts]), X_cloud=np.concatenate([p["X_cloud"] for p in parts]),
               groups=np.concatenate([np.char.add(f"c{k}_", p["groups"].astype(str)) for k, p in enumerate(parts)]))
    for key in ("t", "ftype", "cycle"):
        if key in parts[0]:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def one_draw(scen: str, d: int) -> dict:
    fam = family(scen)
    if fam == "base":
        return BT.one_draw(scen, d)
    rng = np.random.default_rng([TRUTH_SEED, 98, NEW_ORDER.index(scen), d])
    if fam == "T":
        p = T_SCENARIOS[scen]; tr = sample_T(p, rng); tr["family"] = "T"
        te = _concat([sample_T(p, rng) for _ in range(TEST_COPIES)], "T")
    else:
        p = C_SCENARIOS[scen]; tr = sample_C(p, rng); tr["family"] = "C"
        te = _concat([sample_C(p, rng) for _ in range(TEST_COPIES)], "C")
    splits = make_splits(tr["y"], tr["groups"], "tabular", PLAN)[:J_FULL]
    k = d % J_FULL
    while splits[k]["skip"]:
        k = (k + 1) % J_FULL
    sp = splits[k]
    seed = PLAN.model_seed + sp["split_id"]
    vtr, vte = control_views(tr), control_views(te)
    r = dict(scenario=scen, draw=d, split=sp["split_id"], n_fit=len(sp["fit"]), n_val=len(sp["val"]))
    for tag in vtr:
        (Xe, Xc), (Te, Tc) = vtr[tag], vte[tag]
        fit, val, y = sp["fit"], sp["val"], tr["y"]
        ba = []
        for Xtr, Xte in ((Xe, Te), (Xc, Tc)):          # as pipeline.fit_tiers: LR, threshold_max_ba on validation
            m = make_model("A", "edge", seed, PLAN).fit(Xtr[fit], y[fit])
            thr = M.threshold_max_ba(m.predict_proba(Xtr[val]), y[val])
            ba.append(M.balanced_accuracy(te["y"], (m.predict_proba(Xte) >= thr).astype(int)))
        be, bc = ba
        r[f"ba_edge{tag}"], r[f"ba_cloud{tag}"], r[f"info_step{tag}"] = be, bc, bc - be
    return r
