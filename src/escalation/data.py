"""Loading of processed datasets (schema in data/README.md)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_dataset(name: str, root: str | Path = "data/processed", group_col: str | None = None) -> dict:
    """Return a dict with meta, labels, groups, order, feature matrices, names, raw arrays
    (memory-mapped, or None if absent), channels and bytes.

    group_col: optional meta column that replaces group_id (sensitivity analyses such as
    3W grouped by well); the original group_id is kept in meta['group_id_orig']."""
    d = Path(root) / name
    meta = pd.read_parquet(d / "meta.parquet")
    if group_col is not None:
        meta["group_id_orig"] = meta["group_id"]
        meta["group_id"] = meta[group_col].astype(str)
    meta = meta.reset_index(drop=True)
    if "window_id" in meta and not (meta["window_id"].to_numpy() == np.arange(len(meta))).all():
        raise ValueError("window_id must be 0..n-1 in row order")
    Xe = np.load(d / "X_edge.npy", mmap_mode="r")
    Xc = np.load(d / "X_cloud.npy", mmap_mode="r")
    ne = json.loads((d / "names_edge.json").read_text())
    nc = json.loads((d / "names_cloud.json").read_text())
    if Xe.shape[0] != len(meta) or Xc.shape[0] != len(meta):
        raise ValueError("feature rows must match meta")
    if nc[:len(ne)] != ne:
        raise ValueError("edge features must be the first columns of cloud features")

    def _opt(fn, how="npy"):
        p = d / fn
        if not p.exists():
            return None
        return np.load(p, mmap_mode="r") if how == "npy" else json.loads(p.read_text())

    out = dict(name=name, root=str(root), meta=meta,
               y=meta["label"].to_numpy().astype(int),
               groups=meta["group_id"].astype(str).to_numpy(),
               order=meta["order"].to_numpy() if "order" in meta else np.arange(len(meta)),
               X_edge=np.ascontiguousarray(Xe, dtype=np.float32),
               X_cloud=np.ascontiguousarray(Xc, dtype=np.float32),
               names_edge=ne, names_cloud=nc,
               raw_edge=_opt("raw_edge.npy"), raw_cloud=_opt("raw_cloud.npy"),
               channels=_opt("channels.json", "json") or {}, bytes=_opt("bytes.json", "json") or {})
    if (out["raw_edge"] is None) != (out["raw_cloud"] is None):
        raise ValueError("raw_edge and raw_cloud must both exist or both be absent")
    return out


def cap_windows(ds: dict, max_windows: int, seed: int = 2026) -> dict:
    """Pair-C compute cap (plan/DEVIATIONS.md): keep about max_windows windows by sampling the
    same fraction inside every (group, label) cell, at least one window per non-empty cell,
    seeded. Raw arrays are materialised for the kept rows; window order inside groups is kept."""
    n = len(ds["y"])
    if max_windows is None or n <= max_windows:
        return ds
    frac = max_windows / n
    rng = np.random.default_rng(seed)
    keep = []
    cells = pd.DataFrame({"g": ds["groups"], "y": ds["y"], "i": np.arange(n)}).groupby(["g", "y"])["i"]
    for _, idx in cells:
        idx = idx.to_numpy()
        k = max(1, int(round(frac * len(idx))))
        keep.append(np.sort(rng.choice(idx, size=min(k, len(idx)), replace=False)))
    keep = np.sort(np.concatenate(keep))
    out = dict(ds)
    out["meta"] = ds["meta"].iloc[keep].reset_index(drop=True)
    out["meta"]["window_id"] = np.arange(len(keep))
    for k in ("y", "groups", "order"):
        out[k] = np.asarray(ds[k])[keep]
    for k in ("X_edge", "X_cloud"):
        out[k] = np.ascontiguousarray(np.asarray(ds[k])[keep])
    for k in ("raw_edge", "raw_cloud"):
        if ds[k] is not None:
            out[k] = np.ascontiguousarray(np.asarray(ds[k][keep]), dtype=np.float32)
    out["cap"] = dict(max_windows=int(max_windows), kept=int(len(keep)), of=int(n), seed=int(seed))
    return out
