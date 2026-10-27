import numpy as np
import torch
from synthetic import make_synthetic
from escalation import metrics as M
from escalation import routers as RT
from escalation.config import PLAN, tiny
from escalation.models import make_model
from escalation.splits import make_splits


def _planted(n_groups=60, per=50, seed=0):
    """g depends on edge feature x0: the edge is wrong (cloud right) mostly when x0 > 0."""
    rng = np.random.default_rng(seed)
    n = n_groups * per
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per)
    X = rng.normal(size=(n, 6))
    y = (rng.random(n) < 0.3).astype(int)
    edge_wrong = (X[:, 0] > 0) & (rng.random(n) < 0.8) | (rng.random(n) < 0.05)
    pe_hat = np.where(edge_wrong, 1 - y, y)
    p_e = np.clip(np.where(pe_hat == 1, 0.7, 0.3) + rng.normal(0, 0.1, n), 0.01, 0.99)
    p_c = np.where(y == 1, 0.9, 0.1)
    return X, y, groups, p_e, p_c


def test_r6_learns_planted_gain():
    X, y, groups, p_e, p_c = _planted()
    val = np.isin(groups, [f"g{i}" for i in range(30)]); test = ~val
    tg = RT.router_targets(y[val], p_e[val], p_c[val], 0.5, 0.5)
    sc = RT.learned_routers(RT.router_inputs(X[val], p_e[val]), RT.router_inputs(X[test], p_e[test]), tg, 1, PLAN)
    g = M.escalation_gain(y[test], (p_e[test] >= 0.5).astype(int), (p_c[test] >= 0.5).astype(int))
    C6 = M.weighted_concordance(g, sc["R6"])
    assert C6 > 0.6, C6
    assert M.weighted_concordance(g, RT.r0_random(len(g), 0)) < 0.6


def test_capture_identity_on_router_outputs():
    s = make_synthetic(n_groups=20, windows_per_group=30, seed=4, raw=False)
    y, g = s["y"], s["groups"]
    sp = make_splits(y, g, "tabular")[0]
    cfg = tiny()
    f, v, t = sp["fit"], sp["val"], sp["test"]
    e = make_model("A", "edge", 1, cfg).fit(s["X_edge"][f], y[f])
    c = make_model("A", "cloud", 1, cfg).fit(s["X_cloud"][f], y[f])
    pev, pet = e.predict_proba(s["X_edge"][v]), e.predict_proba(s["X_edge"][t])
    pcv, pct = c.predict_proba(s["X_cloud"][v]), c.predict_proba(s["X_cloud"][t])
    te, tc = M.threshold_max_ba(pev, y[v]), M.threshold_max_ba(pcv, y[v])
    gain = M.escalation_gain(y[t], (pet >= te).astype(int), (pct >= tc).astype(int))
    tg = RT.router_targets(y[v], pev, pcv, te, tc)
    scores = dict(R1=RT.r1_confidence(pet, te), R2=RT.r2_entropy(pet), R3=RT.r3_conformal(pev, y[v], te, pet),
                  R9=RT.r9_periodic(s["order"][t]),
                  **RT.learned_routers(RT.router_inputs(s["X_edge"][v], pev), RT.router_inputs(s["X_edge"][t], pet),
                                       tg, 1, cfg))
    for name, sc in scores.items():
        C = M.weighted_concordance(gain, sc)
        assert abs(M.capture(gain, sc, 0, 1) - (2 * C - 1)) < 1e-10, name
    assert ((scores["R3"] > 0) & (scores["R3"] <= 1)).all()
    assert ((scores["R9"] <= 0) & (scores["R9"] > -1)).all()
    E = RT.explanation_inputs(s["X_edge"][t], e, s["X_edge"][f], 7, cfg)
    assert E.shape == (len(t), cfg.r7_top + 6) and np.isfinite(E).all()
    assert (np.abs(E[:, cfg.r7_top + 4]) <= 1 + 1e-9).all()                    # stability is a cosine
    assert np.allclose(E, RT.explanation_inputs(s["X_edge"][t], e, s["X_edge"][f], 7, cfg))


def test_l2d_training_and_shape():
    rng = np.random.default_rng(0)
    n, d = 600, 5
    Z = rng.normal(size=(n, d)).astype(np.float32)
    y = (Z[:, 0] + 0.5 * rng.normal(size=n) > 0.5).astype(int)
    cloud_right = (Z[:, 1] > 0).astype(float)
    cfg = PLAN
    net0 = RT.L2D(d); torch.manual_seed(3); net0 = RT.L2D(d)
    w = torch.as_tensor(M.ba_weights(y) * n, dtype=torch.float32)
    with torch.no_grad():
        loss0 = float(RT.l2d_loss(net0(torch.as_tensor(Z)), torch.as_tensor(y), torch.as_tensor(cloud_right, dtype=torch.float32), w))
    net = RT.train_l2d(Z, y, cloud_right, 3, cfg)
    assert net.final_loss < loss0
    sc = RT.l2d_score(net, Z)
    assert sc.shape == (n,) and np.isfinite(sc).all()
    # deferral should be preferred where the cloud is right
    assert sc[cloud_right == 1].mean() > sc[cloud_right == 0].mean()
    # cross-fitted cloud correctness
    groups = np.repeat(np.arange(30), 20).astype(str)
    cr = RT.crossfit_cloud_correct(lambda: make_model("A", "cloud", 1, tiny()), Z, y, groups, 0.5, 3)
    assert cr.shape == (n,) and set(np.unique(cr)) <= {0.0, 1.0}
