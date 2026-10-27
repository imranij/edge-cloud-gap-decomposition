"""POST-PLAN helpers for area `routers` (audit, cross-fitted routers, fixed tiers, streaming thresholds).

Nothing here changes the confirmatory pipeline; everything reads the stored main run (results/)
and reuses src/sens/common.reconstruct_split and the escalation package.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sens"))
import common as CM                                         # noqa: E402
from escalation.config import PLAN                          # noqa: E402

OUT = ROOT / "results_sens" / "routers"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = CM.DATASETS
LEVELS_A = (1.0, 0.1, 0.25, 0.5)


def stored_value(rows: pd.DataFrame, quantity: str, router: str = "", budget=None) -> float:
    m = (rows.quantity == quantity) & (rows.router == router)
    if budget is not None:
        m &= np.isclose(rows.budget, budget)
    v = rows.loc[m, "value"]
    return float(v.iloc[0]) if len(v) else float("nan")


def cells(levels=True):
    """(dataset, pair, q, split_id) of every stored audit cell: pairs A and B at q = 1, pair A at the H2 levels."""
    out = []
    for ds in DATASETS:
        for pair in ("A", "B"):
            qs = LEVELS_A if (pair == "A" and levels) else (1.0,)
            for q in qs:
                for sid in CM.stored_split_ids(ds, pair, None if q == 1.0 else q):
                    out.append((ds, pair, q, sid))
    return out


def load_cell(ds: str, pair: str, q: float, sid: int):
    qq = None if q == 1.0 else q
    sp = CM.reconstruct_split(ds, pair, sid, qq)
    if sp is None:
        return None
    return sp, CM.load_arrays(ds, pair, sid, qq), CM.load_rows(ds, pair, sid, qq)


class JsonlStore:
    """Append-only per-cell checkpoint so long runs resume."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.done = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.done[r["key"]] = r

    def add(self, rec: dict):
        self.done[rec["key"]] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


def key(ds, pair, q, sid) -> str:
    return f"{ds}|{pair}|{q}|{sid}"
