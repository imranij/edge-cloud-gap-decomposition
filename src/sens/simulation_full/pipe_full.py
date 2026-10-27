"""POST-PLAN: per-dataset pipelines of the full validation. Post-plan, not pre-specified.

Base family: src/sens/simulation/pipeline.run_sim_dataset UNCHANGED, called with all J = 20 splits of the plan
(two repeats were the pilot; the plan's tabular design is 4 repeats of K = 5) and the corrected audit's
within-class null drawn B = 10 times in EVERY split (NULL_SPLITS set to all split ids), so that the GO threshold
can be the 95th percentile of the 200 null values of the same dataset, as on the real data
(src/sens/designs/p1w_within_class.py: 10 permutations per cell pooled over the 20 splits of a block).

T and C families (H1 and the confound controls only): per split of make_splits (J = 20), the pair-A edge family
(LR) on the edge and the cloud view (pipeline.fit_tiers, seed 3000 + split, threshold_max_ba on validation),
the information step BA(cloud) - BA(edge) on the test fold, and the metadata shortcut baselines of the real data
(src/sens/shortcuts/run_shortcuts.py: trivial predictor fitted on the fit part, thresholded on validation, scored on
the test fold):
  T: cycle index, 1-nearest neighbour and depth-3 balanced tree (hydraulic a_cycle_1nn, b_cycle_tree3);
  C: time in file, depth-3 balanced tree (SKAB a_t_start_tree3); file type, balanced logistic regression
     (SKAB c_session_logreg); both, depth-3 balanced tree.
C also runs the two declared controls:
  z    features z-scored within each file (the SKAB declared deconfounded variant), both views;
  aug  the shortcut variables (t, file type) appended to both views (the "edge given the shortcut" control).
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation"))
sys.path.insert(0, str(HERE.parents[1]))
import pipeline as P                                                   # noqa: E402
from escalation.config import PLAN                                     # noqa: E402
from escalation import metrics as M                                    # noqa: E402
from escalation.splits import make_splits                              # noqa: E402
from sklearn.neighbors import KNeighborsClassifier                     # noqa: E402
from sklearn.tree import DecisionTreeClassifier                        # noqa: E402
from sklearn.linear_model import LogisticRegression                    # noqa: E402

J_FULL = 20
B_NULL = 10


def run_base(ds: dict, ds_seed: int) -> list[dict]:
    old = P.NULL_SPLITS
    P.NULL_SPLITS = tuple(range(J_FULL))
    try:
        return P.run_sim_dataset(ds, ds_seed, n_splits=J_FULL, null_draws=B_NULL)
    finally:
        P.NULL_SPLITS = old


# ------------------------------------------------------------------ shortcuts
def _tree():
    return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)


def _knn():
    return KNeighborsClassifier(n_neighbors=1)


def _logreg():
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def shortcut_specs(ds: dict) -> list[tuple[str, np.ndarray, callable]]:
    if ds["family"] == "T":
        c = ds["cycle"].reshape(-1, 1)
        return [("sc_cycle_1nn", c, _knn), ("sc_cycle_tree3", c, _tree)]
    t = ds["t"].reshape(-1, 1); f = ds["ftype"].reshape(-1, 1)
    return [("sc_t_tree3", t, _tree), ("sc_ftype_logreg", f, _logreg), ("sc_tftype_tree3", np.column_stack([t, f]), _tree)]


def score_shortcut(X, mk, y, fit, val, test) -> float:
    m = mk()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X[fit], y[fit])
        k = list(m.classes_).index(1)
        pv, pt = m.predict_proba(X[val])[:, k], m.predict_proba(X[test])[:, k]
    t = M.threshold_max_ba(pv, y[val])
    return M.balanced_accuracy(y[test], (pt >= t).astype(int))


# ------------------------------------------------------------------ controls
def zscore_within(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    for gname in np.unique(groups):
        ix = groups == gname
        mu = X[ix].mean(axis=0); sd = X[ix].std(axis=0)
        X[ix] = (X[ix] - mu) / np.where(sd > 0, sd, 1.0)
    return X.astype(np.float32)


def control_views(ds: dict) -> dict:
    """Views of each design: plain, and for C also z and aug."""
    views = {"": (ds["X_edge"], ds["X_cloud"])}
    if ds["family"] == "C":
        views["_z"] = (zscore_within(ds["X_edge"], ds["groups"]), zscore_within(ds["X_cloud"], ds["groups"]))
        meta = np.column_stack([ds["t"], ds["ftype"]]).astype(np.float32)
        views["_aug"] = (np.column_stack([ds["X_edge"], meta]), np.column_stack([ds["X_cloud"], meta]))
    return views


def tiers_ba(Xe, Xc, y, fit, val, test, seed, cfg=PLAN) -> tuple[float, float]:
    T = P.fit_tiers(dict(X_edge=Xe, X_cloud=Xc, y=y), fit, val, test, seed, cfg)
    (_, pe_t, tE), (_, pc_t, tC) = T["e"], T["c"]
    return (M.balanced_accuracy(y[test], (pe_t >= tE).astype(int)),
            M.balanced_accuracy(y[test], (pc_t >= tC).astype(int)))


def run_h1(ds: dict, ds_seed: int, cfg=PLAN) -> list[dict]:
    y, groups = ds["y"], ds["groups"]
    splits = make_splits(y, groups, "tabular", cfg)[:J_FULL]
    views = control_views(ds)
    specs = shortcut_specs(ds)
    rows = []
    for sp in splits:
        if sp["skip"]:
            rows.append(dict(split=sp["split_id"], skipped=sp["skip"])); continue
        sid = sp["split_id"]; seed = cfg.model_seed + sid
        fit, val, test = sp["fit"], sp["val"], sp["test"]
        r = dict(split=sid, repeat=sp["repeat"], fold=sp["fold"], n_fit=len(fit), n_val=len(val), n_test=len(test),
                 n_train=len(fit) + len(val), n_pos_test=int(y[test].sum()), n_groups_test=len(np.unique(groups[test])))
        for tag, (Xe, Xc) in views.items():
            be, bc = tiers_ba(Xe, Xc, y, fit, val, test, seed, cfg)
            r[f"ba_edge{tag}"], r[f"ba_cloud{tag}"], r[f"info_step{tag}"] = be, bc, bc - be
        for name, X, mk in specs:
            r[f"ba_{name}"] = score_shortcut(X, mk, y, fit, val, test)
        rows.append(r)
    return rows


def run_dataset(ds: dict, ds_seed: int) -> list[dict]:
    return run_base(ds, ds_seed) if ds["family"] == "base" else run_h1(ds, ds_seed)
