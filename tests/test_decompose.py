import numpy as np
from synthetic import make_synthetic
from escalation.splits import make_splits
from escalation.models import make_model
from escalation.config import tiny
from escalation import metrics as M
from escalation.decompose import (factorial_ba, factorial_steps, combination, threshold_composite,
                                  self_oracle_headroom, KEYS)

FV = {"ef_ev": ("edge", "X_edge"), "cf_ev": ("cloud", "X_edge"), "ef_cv": ("edge", "X_cloud"),
      "cf_cv": ("cloud", "X_cloud")}


def _split_models(s, sp, pair="A", cfg=None):
    cfg = cfg or tiny()
    y = s["y"]
    pv, pt, thr = {}, {}, {}
    for k, (fam, view) in FV.items():
        X = s[view]
        m = make_model(pair, fam, 3000 + sp["split_id"], cfg).fit(X[sp["fit"]], y[sp["fit"]])
        pv[k], pt[k] = m.predict_proba(X[sp["val"]]), m.predict_proba(X[sp["test"]])
        thr[k] = M.threshold_max_ba(pv[k], y[sp["val"]])
    return pv, pt, thr


def test_shapley_identity_exact_per_split_and_oracle_bound():
    s = make_synthetic(n_groups=20, windows_per_group=25, seed=11, raw=False)
    y, g = s["y"], s["groups"]
    for sp in make_splits(y, g, "tabular")[:4]:
        for pair in ("A", "B"):
            pv, pt, thr = _split_models(s, sp, pair)
            yt = y[sp["test"]]
            ba = factorial_ba(yt, pt, thr)
            st = factorial_steps(ba)
            assert abs(st["shapley_model"] + st["shapley_info"] - (ba["cf_cv"] - ba["ef_ev"])) < 1e-12
            assert abs(st["interaction"] - (st["model_step_cv"] - st["model_step_ev"])) < 1e-12
            for ek in ("ef_ev", "cf_ev"):
                res, _ = combination(yt, pt[ek], pt["cf_cv"], thr[ek], thr["cf_cv"], pv[ek], pv["cf_cv"],
                                     y[sp["val"]], g[sp["val"]], 1, tiny())
                pe = (pt[ek] >= thr[ek]).astype(int); pc = (pt["cf_cv"] >= thr["cf_cv"]).astype(int)
                assert res["ba_oracle"] >= max(M.balanced_accuracy(yt, pe), M.balanced_accuracy(yt, pc)) - 1e-12
                assert abs(res["unresolved"] - (res["ba_oracle"] - res["ba_stacker"])) < 1e-12
            so = self_oracle_headroom(yt, pt["cf_cv"], pt["cf_cv"][::-1].copy(), thr["cf_cv"], 0.5)
            assert so["self_oracle_headroom"] >= -1e-12


def test_steps_identity_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ba = dict(zip(KEYS, rng.random(4)))
        st = factorial_steps(ba)
        assert abs(st["shapley_model"] + st["shapley_info"] - (ba["cf_cv"] - ba["ef_ev"])) < 1e-15


def test_composite_threshold_in_grid():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.3).astype(int)
    p = np.clip(0.3 * y + rng.random(300) * 0.7, 0, 1)
    t = threshold_composite(p, y)
    assert 0 <= t <= 1
