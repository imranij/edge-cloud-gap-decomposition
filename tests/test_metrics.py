import sys, pathlib, itertools
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from escalation import metrics as M
from escalation import stats as S


def _brute_C(g, s):
    num = den = 0.0
    for i, j in itertools.combinations(range(len(g)), 2):
        dg = abs(g[i] - g[j])
        if dg == 0:
            continue
        hi, lo = (i, j) if g[i] > g[j] else (j, i)
        k = 1.0 if s[hi] > s[lo] else (0.5 if s[hi] == s[lo] else 0.0)
        num += dg * k; den += dg
    return num / den


def _setup(rng, n=60, p=0.2):
    y = (rng.random(n) < p).astype(int); y[0] = 1; y[1] = 0
    pe = (rng.random(n) < 0.7).astype(int) * y + (rng.random(n) < 0.2).astype(int) * (1 - y)
    pc = (rng.random(n) < 0.8).astype(int) * y + (rng.random(n) < 0.15).astype(int) * (1 - y)
    return y, pe, pc


def test_ba_identity():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, pe, pc = _setup(rng)
        E = rng.random(len(y)) < 0.4
        g = M.escalation_gain(y, pe, pc)
        assert abs(M.ba_with_escalation(y, pe, pc, E) - (M.balanced_accuracy(y, pe) + g[E].sum())) < 1e-12


def test_concordance_matches_bruteforce_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(40):
        y, pe, pc = _setup(rng)
        g = M.escalation_gain(y, pe, pc)
        s = np.round(rng.normal(size=len(y)), 1)        # many ties
        assert abs(M.weighted_concordance(g, s) - _brute_C(g, s)) < 1e-10


def test_identity_budget_average_equals_2C_minus_1():
    rng = np.random.default_rng(2)
    for _ in range(40):
        y, pe, pc = _setup(rng, n=80)
        g = M.escalation_gain(y, pe, pc)
        s = np.round(rng.normal(size=len(y)), 1)
        n = len(g); k = np.arange(1, n + 1)
        G = M.gain_curve(g, s)[1:]; Go = M.gain_curve(g, g)[1:]
        rand = k / n * g.sum()
        ratio = (G - rand).sum() / (Go - rand).sum()
        assert abs(ratio - (2 * M.weighted_concordance(g, s) - 1)) < 1e-10
        assert abs(M.capture(g, s, 0, 1) - ratio) < 1e-10


def test_oracle_and_random_extremes():
    rng = np.random.default_rng(3)
    y, pe, pc = _setup(rng, n=200)
    g = M.escalation_gain(y, pe, pc)
    assert abs(M.weighted_concordance(g, g) - 1.0) < 1e-12
    assert abs(M.weighted_concordance(g, -g) - 0.0) < 1e-12
    assert abs(M.weighted_concordance(g, np.zeros_like(g)) - 0.5) < 1e-12
    assert abs(M.ba_with_escalation(y, pe, pc, np.ones(len(y), bool)) - M.balanced_accuracy(y, pc)) < 1e-12
    assert M.balanced_accuracy(y, M.oracle_predictions(y, pe, pc)) >= max(M.balanced_accuracy(y, pe), M.balanced_accuracy(y, pc))


def test_nb_correction_and_floor():
    rng = np.random.default_rng(4)
    d = rng.normal(0.01, 0.03, size=20)
    r = S.nb_test(d, n_train=800, n_test=200, margin=0.02)
    assert r["se"] > r["se_naive"]
    assert abs(r["se_floor"] - d.std(ddof=1) * 0.5) < 1e-12
    pf = S.power_floor(sd=0.05, n_train=800, n_test=200, margin=0.02, J=20)
    assert pf["feasible"] is False


def test_holm():
    adj = S.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_hksj_wider_than_dl_and_symmetric():
    r1 = S.pool_random_effects([0.01, 0.03, -0.01, 0.02], [0.01, 0.012, 0.009, 0.011])
    r2 = S.pool_hksj([0.01, 0.03, -0.01, 0.02], [0.01, 0.012, 0.009, 0.011], margin=0.05)
    assert r2["df"] == 3 and (r2["ci_hi"] - r2["ci_lo"]) >= (r1["ci_hi"] - r1["ci_lo"]) - 1e-12
    assert abs((r2["ci_hi"] + r2["ci_lo"]) / 2 - r2["mean"]) < 1e-12
