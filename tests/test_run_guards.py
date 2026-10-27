"""Regression tests for the audit fixes: --tiny isolation, atomic writes / resume, confirmatory
roles, and weight-scale invariance of the BA-weighted HGB routers."""
import json

import numpy as np
import pandas as pd
import pytest
from synthetic import write_synthetic

import analyze
import run_dataset
from escalation import audit as AU
from escalation import metrics as M
from escalation import routers as RT
from escalation.config import PLAN
from sklearn.ensemble import HistGradientBoostingRegressor


# ------------------------------------------------------------------ --tiny isolation
def test_tiny_may_not_write_to_repo_results(tmp_path):
    res = run_dataset.ROOT / "results"
    for out in (res, res / "sub", str(res) + "/./"):
        with pytest.raises(SystemExit, match="tiny"):
            run_dataset.check_tiny_out(True, out)
    run_dataset.check_tiny_out(True, tmp_path / "results")          # scratch is fine
    run_dataset.check_tiny_out(False, res)                          # plan settings are fine
    before = sorted(p.name for p in res.iterdir()) if res.exists() else []
    with pytest.raises(SystemExit, match="tiny"):                   # refused before any I/O
        run_dataset.main(["no_such_dataset", "A", "--tiny", "--out", str(res)])
    assert (sorted(p.name for p in res.iterdir()) if res.exists() else []) == before


def _fake_results(root, tiny_row: float, log_tiny: bool):
    d = root / "hydraulic" / "A"
    d.mkdir(parents=True)
    base = dict(dataset="hydraulic", pair="A", split=0, repeat=0, fold=0, h2_q=1.0, n_train=10, n_test=5,
                n_pos_train=3, router="", budget=np.nan)
    pd.DataFrame([dict(base, quantity="tiny_settings", value=tiny_row),
                  dict(base, quantity="shapley_info", value=0.1)]).to_parquet(d / "split_0.parquet", index=False)
    (d / "log.jsonl").write_text(json.dumps(dict(split="0", seconds=1.0, tiny=log_tiny)) + "\n")


@pytest.mark.parametrize("tiny_row,log_tiny", [(1.0, False), (0.0, True), (1.0, True)])
def test_analysis_refuses_tiny_results(tmp_path, tiny_row, log_tiny):
    _fake_results(tmp_path, tiny_row, log_tiny)
    with pytest.raises(SystemExit, match="tiny"):
        analyze.load_results(tmp_path)
    df, _ = analyze.load_results(tmp_path, allow_tiny=True)
    assert "tiny_settings" not in set(df.quantity)


def test_analysis_accepts_plan_results(tmp_path):
    _fake_results(tmp_path, 0.0, False)
    df, _ = analyze.load_results(tmp_path)
    assert set(df.quantity) == {"shapley_info"}


# ------------------------------------------------------------------ atomic writes and resume
def test_interrupted_write_leaves_no_completion_marker(tmp_path, monkeypatch):
    data, res = tmp_path / "data", tmp_path / "res"
    write_synthetic(data, "synW", n_groups=14, windows_per_group=15, seed=3, L=32)
    args = ["synW", "B", "--splits", "0:1", "--root", str(data), "--out", str(res), "--tiny"]
    orig = pd.DataFrame.to_parquet

    def crash(self, path, *a, **k):                  # partial file, then the process "dies"
        with open(path, "wb") as fh:
            fh.write(b"PAR1 truncated")
        raise KeyboardInterrupt
    monkeypatch.setattr(pd.DataFrame, "to_parquet", crash)
    with pytest.raises(KeyboardInterrupt):
        run_dataset.main(args)
    d = res / "synW" / "B"
    assert not list(d.glob("split_*.parquet")), "a truncated split file would be treated as finished"
    monkeypatch.setattr(pd.DataFrame, "to_parquet", orig)
    run_dataset.main(args)                           # resume recomputes the split
    df = pd.read_parquet(d / "split_0.parquet")
    assert "shapley_info" in set(df.quantity)
    assert (d / "arrays_0.npz").exists() and (d / "footprint.json").exists()
    np.load(d / "arrays_0.npz")["y_test"]
    assert not [p for p in d.iterdir() if p.name.endswith(".tmp") or ".tmp." in p.name]


# ------------------------------------------------------------------ confirmatory roles
def test_default_confirmatory_set_is_the_plans():
    assert {n: analyze.role_of(n) for n in ("hydraulic", "skab", "seu", "threew")} == \
        dict.fromkeys(("hydraulic", "skab", "seu", "threew"), "confirmatory")
    assert analyze.role_of("hydraulic_v2") == "other"
    assert analyze.role_of("synthetic_debug") == "other"
    assert analyze.role_of("threew__by_well") == "sensitivity"
    assert analyze.role_of("secom") == "pilot"
    assert analyze.role_of("synA", ["synA"]) == "confirmatory"


# ------------------------------------------------------------------ BA-weight scale (HGB min_hessian_to_split)
def _large_val(n=60000, prev=0.03, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (rng.random(n) < prev).astype(int)
    wrong = (X[:, 0] > 0) & (rng.random(n) < 0.8) | (rng.random(n) < 0.05)
    p_e = np.clip(np.where(np.where(wrong, 1 - y, y) == 1, 0.7, 0.3) + rng.normal(0, 0.1, n), 0.01, 0.99)
    p_c = np.where(y == 1, 0.9, 0.1)
    tg = RT.router_targets(y, p_e, p_c, 0.5, 0.5)
    return RT.router_inputs(X, p_e), tg, np.repeat(np.arange(n // 100), 100).astype(str)


def _reference(Z, g, w, seed):
    """Same fit with weights scaled far above sklearn's fixed min_hessian_to_split = 1e-3."""
    m = HistGradientBoostingRegressor(**PLAN.router_gb, early_stopping=False, random_state=seed)
    return m.fit(Z, g, sample_weight=np.asarray(w) / np.mean(w) * 1e3)


def test_ba_weighted_routers_do_not_depend_on_weight_scale():
    Z, tg, groups = _large_val()
    assert abs(tg["w"].sum() - 1.0) < 1e-9                          # BA weights as produced
    ref = _reference(Z, tg["g"], tg["w"], 7).predict(Z)
    r6 = RT.learned_routers(Z, Z, tg, 7, PLAN)["R6"]
    assert np.allclose(r6, ref, atol=1e-9, rtol=0), np.abs(r6 - ref).max()
    E = np.column_stack([Z[:, :2] ** 2])
    r7 = RT.r7_router(Z, Z, E, E, tg, 7, PLAN)
    ref7 = _reference(np.column_stack([Z, E]), tg["g"], tg["w"], 7).predict(np.column_stack([Z, E]))
    assert np.allclose(r7, ref7, atol=1e-9, rtol=0)
    # audit cross-fit: each fold's weights (a subset summing to < 1) are rescaled as well
    oof = AU.crossfit_r6(Z, tg["g"], tg["w"], groups, 7, PLAN)
    oof_scaled = AU.crossfit_r6(Z, tg["g"], tg["w"] * 1e4, groups, 7, PLAN)
    assert np.allclose(oof, oof_scaled, atol=1e-9, rtol=0)
    assert np.isfinite(M.weighted_concordance(tg["g"], oof))
