import numpy as np
from escalation.audit import run_audit, group_bootstrap_C
from escalation.config import tiny, PLAN
from escalation import metrics as M


def _val(seed=0, n_groups=12, per=40):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per)
    Z = rng.normal(size=(n, 4))
    y = (rng.random(n) < 0.3).astype(int)
    edge_wrong = (Z[:, 0] > 0) & (rng.random(n) < 0.7)
    pe = np.where(edge_wrong, 1 - y, y); pc = y.copy()
    return Z, M.escalation_gain(y, pe, pc), M.ba_weights(y), groups


def test_audit_bootstrap_seeded_and_reproducible():
    Z, g, w, groups = _val()
    a = run_audit(Z, g, w, groups, split_id=3, cfg=tiny())
    b = run_audit(Z, g, w, groups, split_id=3, cfg=tiny())
    for k in ("audit_C_val", "audit_C_lo", "audit_C_hi", "audit_go"):
        assert a[k] == b[k]
    assert np.array_equal(a["oof"], b["oof"])
    assert a["audit_C_lo"] <= a["audit_C_val"] <= a["audit_C_hi"] or not np.isfinite(a["audit_C_lo"])
    assert a["audit_go"] == float(a["audit_C_lo"] > 0.52)
    b1 = group_bootstrap_C(g, a["oof"], groups, 200, 8003)
    b2 = group_bootstrap_C(g, a["oof"], groups, 200, 8003)
    b3 = group_bootstrap_C(g, a["oof"], groups, 200, 8004)
    assert np.array_equal(b1, b2, equal_nan=True) and not np.array_equal(b1, b3, equal_nan=True)


def test_audit_full_settings_go_on_planted_signal():
    Z, g, w, groups = _val(seed=1, n_groups=20, per=60)
    a = run_audit(Z, g, w, groups, split_id=0, cfg=PLAN)
    assert a["audit_C_val"] > 0.55 and a["audit_go"] == 1.0
