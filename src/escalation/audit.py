"""Section 7, H5: the pre-specified routing audit, computed from the validation set only.

R6 is cross-fitted by 5-fold grouped CV inside validation (GroupKFold, so it is deterministic),
giving out-of-fold R6 scores on every validation window. The validation C is
weighted_concordance(g_val, oof scores). Its 95% CI comes from a group bootstrap (groups resampled
with replacement, g kept at its full-validation values, C recomputed). GO iff lower bound > 0.52.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupKFold

from .config import PLAN, Config
from . import models as _models  # noqa: F401  (installs the exact fast HGB weighted binning)
from .metrics import weighted_concordance
from .routers import fit_regressor_router


def crossfit_r6(Z_val, g_val, w_val, groups_val, seed: int, cfg: Config = PLAN) -> np.ndarray:
    groups_val = np.asarray(groups_val)
    ng = len(np.unique(groups_val))
    oof = np.full(len(g_val), np.nan)
    if ng < 2:
        return oof
    for tr, te in GroupKFold(n_splits=min(cfg.audit_folds, ng)).split(Z_val, g_val, groups_val):
        m = fit_regressor_router(Z_val[tr], g_val[tr], seed, cfg, sample_weight=w_val[tr])
        oof[te] = m.predict(Z_val[te])
    return oof


def group_bootstrap_C(g, s, groups, n_boot: int, seed: int) -> np.ndarray:
    """C recomputed on n_boot group-bootstrap resamples (seeded, reproducible)."""
    g = np.asarray(g, float); s = np.asarray(s, float); groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        ix = np.concatenate([members[k] for k in pick])
        out[b] = weighted_concordance(g[ix], s[ix])
    return out


def run_audit(Z_val, g_val, w_val, groups_val, split_id: int, cfg: Config = PLAN) -> dict:
    seed = cfg.model_seed + split_id
    oof = crossfit_r6(Z_val, g_val, w_val, groups_val, seed, cfg)
    if not np.isfinite(oof).all():
        return dict(audit_C_val=np.nan, audit_C_lo=np.nan, audit_C_hi=np.nan, audit_go=0.0, oof=oof)
    C = weighted_concordance(g_val, oof)
    boot = group_bootstrap_C(g_val, oof, groups_val, cfg.audit_boot, cfg.audit_seed + split_id)
    if np.isfinite(boot).sum() == 0:
        lo = hi = np.nan
    else:
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    go = float(np.isfinite(lo) and lo > cfg.audit_go)
    return dict(audit_C_val=C, audit_C_lo=float(lo), audit_C_hi=float(hi), audit_go=go, oof=oof)
