"""Section 3 of the frozen plan: repeated stratified grouped K-fold with an inner validation fold.

Every split is a dict with keys split_id, repeat, fold, K, K_in, fit, val, test (index arrays)
and skip (None or a reason string). Splits that violate the integrity rules are returned with
`skip` set (and empty index arrays are NOT substituted), so callers record the reason.
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .config import PLAN, Config


def groups_per_class(y, groups) -> tuple[int, int]:
    """(number of groups containing positives, number of groups containing negatives)."""
    y = np.asarray(y).astype(int); groups = np.asarray(groups)
    return len(np.unique(groups[y == 1])), len(np.unique(groups[y == 0]))


def outer_k(y, groups, cfg: Config = PLAN) -> int:
    gp, gn = groups_per_class(y, groups)
    K = min(cfg.k_max, gp, gn)
    if K < cfg.k_min:
        raise ValueError(f"K = min({cfg.k_max}, {gp} positive groups, {gn} negative groups) = {K} < {cfg.k_min}")
    return K


def n_repeats(K: int, kind: str, cfg: Config = PLAN) -> int:
    J = cfg.j_cnn if kind == "cnn" else cfg.j_tabular
    return int(math.ceil(J / K))


def check_split(y, groups, fit, val, test) -> str | None:
    """Return None if the split is valid, else the reason."""
    y = np.asarray(y).astype(int); groups = np.asarray(groups)
    gf, gv, gt = (set(np.unique(groups[ix])) for ix in (fit, val, test))
    if gf & gv or gf & gt or gv & gt:
        return "fit/val/test share a group"
    if len(set(fit) & set(val)) or len(set(fit) & set(test)) or len(set(val) & set(test)):
        return "fit/val/test share a window"
    for nm, ix in (("val", val), ("test", test), ("fit", fit)):
        if len(ix) == 0:
            return f"{nm} is empty"
        if len(np.unique(y[ix])) < 2:
            return f"{nm} does not contain both classes"
    return None


def inner_split(y, groups, train_idx, split_id: int, cfg: Config = PLAN):
    """Inner StratifiedGroupKFold(K_in, shuffle, 2000+split) on the training windows; the FIRST
    inner fold is the validation set. Returns fit, val, K_in, reason."""
    yt, gt = y[train_idx], groups[train_idx]
    gp, gn = groups_per_class(yt, gt)
    K_in = min(cfg.k_in_max, gp, gn)
    if K_in < 2:
        return np.array([], int), np.array([], int), K_in, f"K_in = {K_in} < 2 (training groups per class {gp}/{gn})"
    sgkf = StratifiedGroupKFold(n_splits=K_in, shuffle=True, random_state=cfg.inner_seed + split_id)
    fit_rel, val_rel = next(iter(sgkf.split(np.zeros(len(yt)), yt, gt)))
    return np.sort(train_idx[fit_rel]), np.sort(train_idx[val_rel]), K_in, None


def make_splits(y, groups, kind: str = "tabular", cfg: Config = PLAN) -> list[dict]:
    """All outer x inner splits for a pair kind ('tabular' or 'cnn'). The CNN splits are the
    first ceil(10/K) repeats of the same scheme, so their split ids coincide with tabular ones."""
    y = np.asarray(y).astype(int); groups = np.asarray(groups).astype(str)
    K = outer_k(y, groups, cfg)
    R = n_repeats(K, kind, cfg)
    out = []
    X0 = np.zeros(len(y))
    for r in range(R):
        sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=cfg.outer_seed + r)
        for f, (tr, te) in enumerate(sgkf.split(X0, y, groups)):
            sid = r * K + f
            fit, val, K_in, reason = inner_split(y, groups, np.sort(tr), sid, cfg)
            te = np.sort(te)
            if reason is None:
                reason = check_split(y, groups, fit, val, te)
            out.append(dict(split_id=sid, repeat=r, fold=f, K=K, K_in=K_in, fit=fit, val=val, test=te,
                            skip=reason))
    return out


def h2_subsample(y, groups, fit, val, q: float, split_id: int, cfg: Config = PLAN):
    """H2: within the training split only, keep positive windows in a random fraction q of the
    groups that contain positives (at least one group) and drop positives elsewhere; negatives
    are untouched. Applied separately to the fit and the validation part (at least one positive
    group in each), with one permutation per part from seed 4000+split, so levels are nested.
    The per-part reading is logged in plan/DEVIATIONS.md; run_split saves the realised counts."""
    y = np.asarray(y).astype(int); groups = np.asarray(groups).astype(str)
    rng = np.random.default_rng(cfg.h2_seed + split_id)
    keep = []
    for part in (fit, val):
        pos_groups = np.unique(groups[part][y[part] == 1])
        perm = rng.permutation(pos_groups)
        m = max(1, int(round(q * len(pos_groups))))
        kept = set(perm[:m])
        mask = (y[part] == 0) | np.isin(groups[part], list(kept))
        keep.append(part[mask])
    return keep[0], keep[1]
