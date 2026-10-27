"""POST-PLAN wrapper for pair A on leave-one-level-out designs with 3 groups (hyd_cooler, skab_lso).

    python src/sens/hydskab/run_fold_A.py <variant> A --splits 0:3 --root data/processed_sens --out results_sens/hydskab

Why: with 3 groups, K = 3 and K_in = 2, so the fit part of every split holds ONE group. Router R8 (learning to
defer) needs 'cloud right' labels cross-fitted over at least two fit groups (routers.crossfit_cloud_correct uses
GroupKFold(min(3, groups))), and run_dataset.py stops with a ValueError. This wrapper runs run_dataset.main
unchanged except that, when the fit part has fewer than two groups, R8 is declared not estimable: its rows (and
l2d_cloud_right_fit) are set to NaN, score_R8 is not stored, and a row r8_not_estimable = 1 is added. Every other
model, threshold, decomposition quantity and router is computed by the unchanged pipeline code.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import run_dataset as RD                      # noqa: E402

_orig_cf = RD.RT.crossfit_cloud_correct
_orig_rs = RD.run_split
_state = {"na": False}


def _cf(make_cloud, X, y, groups_fit, t_c, n_folds=3):
    if len(np.unique(groups_fit)) < 2:
        _state["na"] = True
        return np.zeros(len(y))
    return _orig_cf(make_cloud, X, y, groups_fit, t_c, n_folds)


def _rs(ds, pair, sp, cfg, q=None):
    _state["na"] = False
    rows, arrays, info = _orig_rs(ds, pair, sp, cfg, q)
    if rows is not None and _state["na"]:
        for r in rows:
            if r["router"] == "R8" or r["quantity"] == "l2d_cloud_right_fit":
                r["value"] = float("nan")
        rows.append(dict(rows[0], quantity="r8_not_estimable", router="R8", budget=float("nan"), value=1.0))
        arrays.pop("score_R8", None)
    return rows, arrays, info


RD.RT.crossfit_cloud_correct = _cf
RD.run_split = _rs

if __name__ == "__main__":
    RD.main(sys.argv[1:])
