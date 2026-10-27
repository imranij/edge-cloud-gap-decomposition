"""Shared helpers for the post-plan sensitivity experiments.

Nothing here changes the confirmatory pipeline. `reconstruct_split` rebuilds the fit, validation
and test index arrays of a stored main-run split (and the H2-subsampled fit/val for q < 1) from the
deterministic seeds in escalation.config.PLAN, and checks them against what the main run stored
(test_idx in arrays_<id>.npz, n_fit/n_val in split_<id>.parquet), so every post-plan analysis
works on exactly the windows the confirmatory run used.
"""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.config import PLAN                              # noqa: E402
from escalation.data import load_dataset                        # noqa: E402
from escalation.splits import make_splits, h2_subsample        # noqa: E402
from escalation.models import PAIR_KIND                         # noqa: E402

RESULTS = ROOT / "results"
DATASETS = ("hydraulic", "skab", "seu", "threew")


@lru_cache(maxsize=None)
def dataset(name: str, root: str = "data/processed", group_col: str | None = None) -> dict:
    return load_dataset(name, ROOT / root, group_col)


@lru_cache(maxsize=None)
def splits(name: str, pair: str, root: str = "data/processed", group_col: str | None = None) -> tuple:
    ds = dataset(name, root, group_col)
    return tuple(make_splits(ds["y"], ds["groups"], PAIR_KIND[pair], PLAN))


def result_dir(name: str, pair: str, group_col: str | None = None) -> Path:
    d = f"{name}__by_{group_col}" if group_col else name
    return RESULTS / d / pair


def tag(split_id: int, q: float | None) -> str:
    return f"{split_id}" + (f"_q{q}" if q is not None and q != 1.0 else "")


def load_arrays(name: str, pair: str, split_id: int, q: float | None = None, group_col=None) -> dict | None:
    fn = result_dir(name, pair, group_col) / f"arrays_{tag(split_id, q)}.npz"
    if not fn.exists():
        return None
    with np.load(fn, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_rows(name: str, pair: str, split_id: int, q: float | None = None, group_col=None) -> pd.DataFrame | None:
    fn = result_dir(name, pair, group_col) / f"split_{tag(split_id, q)}.parquet"
    return pd.read_parquet(fn) if fn.exists() else None


def reconstruct_split(name: str, pair: str, split_id: int, q: float | None = None, group_col=None,
                      check: bool = True) -> dict | None:
    """fit, val, test index arrays of a stored split; q in (0.1, 0.25, 0.5) applies the H2
    subsampling exactly as run_split did. Returns None if the split was skipped or not stored.
    With check=True, asserts equality with the stored test_idx and the stored n_fit / n_val."""
    sp = splits(name, pair, "data/processed", group_col)[split_id]
    if sp["skip"]:
        return None
    ds = dataset(name, "data/processed", group_col)
    fit, val, test = sp["fit"], sp["val"], sp["test"]
    if q is not None and q != 1.0:
        fit, val = h2_subsample(ds["y"], ds["groups"], fit, val, q, split_id, PLAN)
    if check:
        arr = load_arrays(name, pair, split_id, q, group_col)
        rows = load_rows(name, pair, split_id, q, group_col)
        if arr is None or rows is None:
            return None
        if "skipped" in set(rows["quantity"]):
            return None
        assert np.array_equal(np.asarray(arr["test_idx"]), np.asarray(test)), (name, pair, split_id, q, "test_idx")
        r = rows.set_index("quantity")["value"]
        r = r[~r.index.duplicated()]
        assert int(r["n_fit"]) == len(fit) and int(r["n_val"]) == len(val), (name, pair, split_id, q, "sizes")
        if "pval_ef_ev" in arr:
            assert len(arr["pval_ef_ev"]) == len(val), (name, pair, split_id, q, "pval length")
    return dict(split_id=split_id, repeat=sp["repeat"], fold=sp["fold"], K=sp["K"], K_in=sp["K_in"],
                fit=np.asarray(fit), val=np.asarray(val), test=np.asarray(test))


def stored_split_ids(name: str, pair: str, q: float | None = None, group_col=None) -> list[int]:
    d = result_dir(name, pair, group_col)
    out = []
    for p in sorted(d.glob("split_*.parquet")):
        stem = p.stem[len("split_"):]
        sid, _, qq = stem.partition("_q")
        if (q is None or q == 1.0) and qq == "":
            out.append(int(sid))
        elif q is not None and q != 1.0 and qq != "" and float(qq) == q:
            out.append(int(sid))
    return sorted(out)


if __name__ == "__main__":
    # self-check: every stored split of every dataset/pair (and H2 level) reconstructs exactly
    import itertools
    n_ok = n_none = 0
    for name, pair in itertools.product(DATASETS, ("A", "B", "C")):
        levels = [None] + ([0.1, 0.25, 0.5] if pair == "A" else [])
        for q in levels:
            for sid in stored_split_ids(name, pair, q):
                r = reconstruct_split(name, pair, sid, q)
                if r is None:
                    n_none += 1
                else:
                    n_ok += 1
    for sid in stored_split_ids("threew", "A", None, "well"):
        r = reconstruct_split("threew", "A", sid, None, "well")
        n_ok += r is not None
    print(f"reconstructed and verified {n_ok} stored splits ({n_none} skipped or absent)")
