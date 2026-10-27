"""Shared loaders for the post-plan analyses. NOT pre-specified.

Every script in src/extra/ reads only files already in results/ (per-split parquet and npz
files, results/summary/*.csv) and data/processed/ (labels, groups, meta). No model is refitted:
the only recomputation is of metrics from stored predictions, scores and thresholds, and the
deterministic reconstruction of the split indices (src/escalation/splits.py with the plan's
seeds). Outputs go to results/extra/.
"""
from __future__ import annotations
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation import stats as S                     # noqa: E402
from escalation import metrics as M                   # noqa: E402

RES = ROOT / "results"
SUM = RES / "summary"
OUT = RES / "extra"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["hydraulic", "skab", "seu", "threew"]
DSN = {"hydraulic": "Hydraulic", "skab": "SKAB", "seu": "SEU", "threew": "3W", "threew__by_well": "3W by well"}


@lru_cache(maxsize=None)
def long_results() -> pd.DataFrame:
    """All per-split rows of the confirmatory and sensitivity runs (same files as analyze.py)."""
    files = [f for f in sorted(RES.glob("*/*/split_*.parquet")) if f.parts[-3] not in ("summary", "extra")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[~df.quantity.isin(["tiny_settings", "skipped"])].copy()


def wide(df: pd.DataFrame, quantities, router="", budget=None) -> pd.DataFrame:
    """One row per (dataset, pair, split, h2_q) with the requested quantities as columns."""
    m = df.quantity.isin(quantities) & (df.router == router)
    if budget is not None:
        m &= np.isclose(df.budget, budget)
    d = df[m]
    w = d.set_index(["dataset", "pair", "split", "h2_q", "quantity"])["value"].unstack("quantity")
    w.columns.name = None
    return w.reset_index()


def per_split(df, quantity, router="", budget=None, name=None):
    m = (df.quantity == quantity) & (df.router == router)
    if budget is not None:
        m &= np.isclose(df.budget, budget)
    return df[m].set_index(["dataset", "pair", "split", "h2_q"])["value"].rename(name or quantity)


@lru_cache(maxsize=None)
def sizes() -> pd.DataFrame:
    return pd.read_csv(SUM / "sizes.csv")


def nb(values, dataset, pair="A", margin=None) -> dict:
    sz = sizes()
    r = sz[(sz.dataset == dataset) & (sz.pair == pair)].iloc[0]
    return S.nb_test(np.asarray(values, float), r.n_train, r.n_test, margin=margin)


def arrays(dataset, pair, split, q=None):
    tag = f"{split}" + ("" if q is None or q == 1.0 else f"_q{q}")
    return np.load(RES / dataset / pair / f"arrays_{tag}.npz")


def mixed_slope(data: pd.DataFrame, formula: str, term: str) -> dict:
    """Same model class as src/analyze.py (MixedLM, dataset random intercepts, REML)."""
    import statsmodels.formula.api as smf
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    out = dict(n=len(data), n_groups=int(data["dataset"].nunique()), model="MixedLM", note="")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            if out["n_groups"] < 2:
                raise ValueError("one group")
            fit = smf.mixedlm(formula, data, groups=data["dataset"]).fit(reml=True)
            out["note"] = "; ".join(sorted({str(x.message)[:60] for x in w}))
        except Exception as e:                       # noqa: BLE001
            fit = smf.ols(formula, data).fit()
            out["model"] = "OLS"
            out["note"] = f"OLS ({str(e)[:40]})"
    b = float(fit.params[term]); se = float(fit.bse[term])
    lo, hi = (float(v) for v in fit.conf_int().loc[term])
    return dict(out, slope=b, se=se, ci_lo=lo, ci_hi=hi, z=b / se if se > 0 else np.nan)


def ols_slope(x, y) -> dict:
    import statsmodels.api as sm
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(n=len(x), slope=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    lo, hi = fit.conf_int()[1]
    return dict(n=len(x), slope=float(fit.params[1]), se=float(fit.bse[1]), ci_lo=float(lo), ci_hi=float(hi))


@lru_cache(maxsize=None)
def dataset(name: str):
    from escalation.data import load_dataset
    return load_dataset(name, ROOT / "data" / "processed")


@lru_cache(maxsize=None)
def splits(name: str, kind: str = "tabular"):
    """Deterministic reconstruction of the plan's splits (same code and seeds as the runs)."""
    from escalation.splits import make_splits
    ds = dataset(name)
    return {s["split_id"]: s for s in make_splits(ds["y"], ds["groups"], kind)}


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def fmt_ci(r, d=3):
    return f"{r['mean']:.{d}f} [{r['ci_lo']:.{d}f}, {r['ci_hi']:.{d}f}]"
