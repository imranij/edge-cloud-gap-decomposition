import numpy as np
import pytest
from synthetic import make_synthetic
from escalation.splits import make_splits, check_split, h2_subsample, outer_k, n_repeats
from escalation.config import PLAN


def _check_integrity(y, groups, sp):
    gf, gv, gt = (set(groups[sp[k]]) for k in ("fit", "val", "test"))
    assert not (gf & gv) and not (gf & gt) and not (gv & gt)
    for k in ("val", "test", "fit"):
        assert set(np.unique(y[sp[k]])) == {0, 1}


def test_split_integrity_and_counts():
    s = make_synthetic(n_groups=24, windows_per_group=20, seed=3, raw=False)
    y, g = s["y"], s["groups"]
    K = outer_k(y, g)
    sps = make_splits(y, g, "tabular")
    assert K == 5 and len(sps) == K * n_repeats(K, "tabular") >= 20
    for sp in sps:
        assert sp["skip"] is None
        _check_integrity(y, g, sp)
        assert len(sp["fit"]) + len(sp["val"]) + len(sp["test"]) == len(y)
    # each repeat's test folds partition the windows
    for r in range(n_repeats(K, "tabular")):
        te = np.concatenate([sp["test"] for sp in sps if sp["repeat"] == r])
        assert np.array_equal(np.sort(te), np.arange(len(y)))
    # determinism, and CNN splits are the first ceil(10/K) repeats of the same scheme
    again = make_splits(y, g, "tabular")
    cnn = make_splits(y, g, "cnn")
    assert len(cnn) == K * n_repeats(K, "cnn")
    for a, b, c in zip(sps, again, cnn):
        for k in ("fit", "val", "test"):
            assert np.array_equal(a[k], b[k]) and np.array_equal(a[k], c[k])


def test_k_rule_and_skips():
    # 3 positive groups -> K = 3; R = ceil(20/3) = 7
    y = np.r_[np.ones(30), np.zeros(100)].astype(int)
    g = np.r_[np.repeat(["p0", "p1", "p2"], 10), np.repeat([f"n{i}" for i in range(10)], 10)]
    assert outer_k(y, g) == 3 and n_repeats(3, "tabular") == 7 and n_repeats(3, "cnn") == 4
    with pytest.raises(ValueError):
        outer_k(y[10:], g[10:])                      # only 2 positive groups -> K < 3
    # violations are reported, not hidden
    assert check_split(y, g, np.arange(0, 20), np.arange(15, 40), np.arange(40, 60)) is not None
    assert "both classes" in check_split(y, g, np.arange(30, 60), np.arange(60, 80), np.arange(0, 30))


def test_h2_subsample_rules():
    s = make_synthetic(n_groups=40, windows_per_group=10, seed=5, raw=False, fault_group_share=0.5)
    y, g = s["y"], s["groups"]
    sp = make_splits(y, g, "tabular")[0]
    prev = None
    for q in (0.1, 0.25, 0.5):
        f, v = h2_subsample(y, g, sp["fit"], sp["val"], q, sp["split_id"])
        for part, full in ((f, sp["fit"]), (v, sp["val"])):
            assert np.array_equal(np.sort(part[y[part] == 0]), np.sort(full[y[full] == 0]))   # negatives untouched
            assert len(np.unique(g[part][y[part] == 1])) >= 1
            assert set(part) <= set(full)
        kept = set(np.r_[f, v][y[np.r_[f, v]] == 1])
        if prev is not None:
            assert prev <= kept                                                               # nested levels
        prev = kept
