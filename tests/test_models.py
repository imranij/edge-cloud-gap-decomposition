import time
import numpy as np
from escalation.models import make_model, footprint, CNNModel, _fast_find_binning_thresholds
from escalation.config import PLAN, tiny
from escalation.metrics import balanced_accuracy
from sklearn.ensemble._hist_gradient_boosting import binning as B


def _raw(n=240, c=3, L=64, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    t = np.arange(L) / L
    X = rng.normal(0, 1, (n, c, L)).astype(np.float32)
    X[:, 0] += (1 + 2 * y)[:, None] * np.sin(2 * np.pi * 5 * t)[None, :]
    return X, y


def test_cnns_train_within_seconds():
    X, y = _raw()
    Xv, yv = _raw(120, seed=1)
    for fam in ("edge", "cloud"):
        t0 = time.time()
        m = make_model("C", fam, 3000, PLAN).fit(X, y, Xv, yv)
        dt = time.time() - t0
        p = m.predict_proba(Xv)
        assert p.shape == (len(yv),) and ((p >= 0) & (p <= 1)).all()
        assert dt < 30, dt
        assert balanced_accuracy(yv, (p >= 0.5).astype(int)) > 0.7
        assert 1 <= m.epochs_run <= PLAN.cnn_epochs
        fp = footprint(m, X.shape[1:])
        assert fp["params"] > 0 and fp["bytes"] > 0 and fp["macs"] > 0
    # determinism
    a = make_model("C", "edge", 5, tiny()).fit(X, y, Xv, yv).predict_proba(Xv)
    b = make_model("C", "edge", 5, tiny()).fit(X, y, Xv, yv).predict_proba(Xv)
    assert np.allclose(a, b)


def test_tabular_families_and_footprint():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0.8).astype(int)
    for pair in ("A", "B"):
        for fam in ("edge", "cloud"):
            m = make_model(pair, fam, 1, tiny()).fit(X, y)
            p = m.predict_proba(X)
            assert p.shape == (300,)
            assert balanced_accuracy(y, (p >= 0.5).astype(int)) > 0.7
            fp = footprint(m, 12)
            assert fp["params"] > 0 and fp["bytes"] > 0 and fp["macs"] > 0


def test_fast_weighted_binning_matches_sklearn():
    rng = np.random.default_rng(0)
    for t in range(60):
        n = int(rng.integers(50, 2000))
        x = [rng.normal(size=n), np.round(rng.normal(size=n), 1), rng.integers(0, 400, n).astype(float)][t % 3]
        yy = rng.random(n) < 0.2
        w = np.where(yy, 0.5 / max(yy.sum(), 1), 0.5 / (~yy).sum()) * n
        a = B._orig_find_binning_thresholds(x.copy(), 255, w.copy())
        b = _fast_find_binning_thresholds(x.copy(), 255, w.copy())
        assert np.array_equal(a, b)
    assert B._find_binning_thresholds is _fast_find_binning_thresholds
