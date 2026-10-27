"""Full smoke run of run_dataset.py and analyze.py on two synthetic datasets (tiny settings)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from synthetic import write_synthetic

ROOT = Path(__file__).resolve().parents[1]
COLS = ["dataset", "pair", "split", "repeat", "fold", "h2_q", "n_train", "n_test", "n_pos_train",
        "quantity", "router", "budget", "value"]


def _run(*args):
    r = subprocess.run([sys.executable, str(ROOT / "src" / args[0]), *args[1:]], capture_output=True, text=True,
                       cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_smoke_two_datasets(tmp_path):
    data, res = tmp_path / "data", tmp_path / "results"
    write_synthetic(data, "synA", n_groups=16, windows_per_group=20, seed=1, L=32)
    write_synthetic(data, "synB", n_groups=14, windows_per_group=20, seed=2, prevalence=0.3, L=32)
    common = ["--root", str(data), "--out", str(res), "--tiny"]
    for ds in ("synA", "synB"):
        _run("run_dataset.py", ds, "A", "--splits", "0:3", *common)
        _run("run_dataset.py", ds, "B", "--splits", "0:2", *common)
        _run("run_dataset.py", ds, "C", "--splits", "0:1", *common)
        _run("run_dataset.py", ds, "A", "--h2", "--splits", "0:3", *common)
    _run("run_dataset.py", "synA", "A", "--group-col", "well", "--splits", "0:1", *common)
    # resume: nothing is recomputed
    d = res / "synA" / "A"
    before = {p.name: p.stat().st_mtime_ns for p in d.glob("split_*.parquet")}
    out = _run("run_dataset.py", "synA", "A", "--splits", "0:3", *common)
    assert out.strip() == ""
    assert before == {p.name: p.stat().st_mtime_ns for p in d.glob("split_*.parquet")}
    # long-format schema and contents
    df = pd.read_parquet(d / "split_0.parquet")
    assert list(df.columns) == COLS
    q = set(df.quantity)
    for need in ("ba_ef_ev", "ba_cf_ev", "ba_ef_cv", "ba_cf_cv", "shapley_info", "shapley_model", "interaction",
                 "comb_step_dep", "unresolved_dep", "comb_step_upg", "unresolved_upg", "self_oracle_headroom",
                 "C", "capture", "ba_at_budget", "recall_at_budget", "rho_pos", "audit_go", "audit_C_lo",
                 "audit_C_hi", "outcome", "comp_ba_ef_ev", "comp_shapley_info", "seconds"):
        assert need in q, need
    assert set(df[df.quantity == "C"].router) == {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "oracle"}
    assert set(df[df.quantity == "ba_at_budget"].budget) == {0.05, 0.1, 0.2, 0.3, 0.5}
    h2 = pd.read_parquet(d / "split_0_q0.1.parquet")
    assert set(h2.h2_q) == {0.1} and {"audit_go", "outcome"} <= set(h2.quantity)
    assert set(h2[h2.quantity == "C"].router) == {"R1", "R6", "oracle"}
    assert h2.n_pos_train.iloc[0] < df.n_pos_train.iloc[0]
    arr = np.load(d / "arrays_0.npz")
    for k in ("y_test", "groups_test", "g_dep", "g_upg", "p_ef_ev", "p_cf_cv", "p_twin", "t_ef_ev", "score_R6",
              "score_R8", "score_oracle"):
        assert k in arr.files, k
    assert np.allclose(arr["score_oracle"], arr["g_dep"])
    fp = json.loads((d / "footprint.json").read_text())
    assert set(fp["footprint"]) == {"ef_ev", "cf_ev", "ef_cv", "cf_cv", "twin"}
    assert (res / "ENVIRONMENT.txt").exists()
    # analysis
    assert not list(res.rglob("*.tmp*")), "temporary files left behind"
    for nm in ("tiny_settings", "n_pos_groups_fit", "n_pos_groups_fit_kept", "n_pos_groups_val",
               "n_pos_groups_val_kept"):
        assert nm in q, nm
    assert (df[df.quantity == "n_pos_groups_fit_kept"].value.to_numpy()
            == df[df.quantity == "n_pos_groups_fit"].value.to_numpy()).all()             # main run: nothing dropped
    assert (h2[h2.quantity == "n_pos_groups_fit_kept"].value.to_numpy()
            <= h2[h2.quantity == "n_pos_groups_fit"].value.to_numpy()).all()
    # tiny results are refused by the analysis unless --allow-tiny (test suite only)
    r = subprocess.run([sys.executable, str(ROOT / "src" / "analyze.py"), "--results", str(res),
                        "--confirmatory", "synA,synB", "--n-perm", "50", "--n-boot", "50"],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode != 0 and "tiny" in r.stderr
    _run("analyze.py", "--results", str(res), "--confirmatory", "synA,synB", "--allow-tiny",
         "--n-perm", "500", "--n-boot", "500")
    s = res / "summary"
    rep = (s / "REPORT.md").read_text()
    for h in ("H1", "H2", "H3", "H4", "H5", "Verdict", "Power and feasibility", "synA__by_well",
              "Realised subsampling", "frac_fit_kept", "WARNING"):
        assert h in rep, h
    prim = pd.read_csv(s / "primary.csv")
    assert list(prim.hypothesis) == ["H1", "H2", "H3", "H4", "H5"]
    assert (prim.p_holm >= prim.p_raw - 1e-12).all()
    pdt = pd.read_csv(s / "per_dataset.csv")
    assert set(pdt.dataset) == {"synA", "synB", "synA__by_well"}
    assert set(pdt[pdt.dataset == "synA__by_well"].role) == {"sensitivity"}
    hk = pd.read_csv(s / "h2_kept_fraction.csv")
    assert set(hk.h2_q) == {0.1, 0.25, 0.5, 1.0}
    assert np.allclose(hk[hk.h2_q == 1.0][["frac_fit_kept", "frac_val_kept"]], 1.0)
    assert "tiny_settings" not in set(pdt.quantity)
    _run("make_tables.py", "--results", str(res))
    for t in ("primary.tex", "h1.tex", "decomposition.tex", "routers.tex", "footprint.tex", "payload.tex"):
        assert (s / "tables" / t).exists(), t
