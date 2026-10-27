"""Synthetic data that mimics the processed-data schema (data/README.md).

Structure:
- groups (independent units): a share of "fault" groups, each normal first and then faulty for a
  contiguous run of windows (as in run-to-fault recordings), and all-normal groups;
- each positive window has a fault type: type 0 is visible at the edge node, type 1 only in the
  extra cloud channels (the planted extra information), so the cloud has an information gain;
- group random offsets in every feature (between-group heterogeneity);
- adjustable prevalence (share of positive windows) and signal strengths.

`make_synthetic` returns the same dict as escalation.data.load_dataset; `write_synthetic` writes the
processed files (meta.parquet, X_*.npy, names_*.json, raw_*.npy, channels.json, bytes.json).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic(n_groups: int = 30, windows_per_group: int = 40, prevalence: float = 0.25,
                   n_edge: int = 20, n_extra: int = 20, fault_group_share: float = 0.5,
                   edge_signal: float = 1.2, cloud_signal: float = 1.5, type1_share: float = 0.5,
                   raw: bool = True, ch_edge: int = 2, ch_extra: int = 2, L: int = 64,
                   group_sd: float = 0.3, seed: int = 0, name: str = "synth") -> dict:
    rng = np.random.default_rng(seed)
    n_fault = max(2, int(round(fault_group_share * n_groups)))
    n_fault = min(n_fault, n_groups - 2)
    n = n_groups * windows_per_group
    # positives per fault group so that the overall prevalence matches
    pos_per = int(np.clip(round(prevalence * n / n_fault), 1, windows_per_group - 1))
    gid, label, order, ftype = [], [], [], []
    for g in range(n_groups):
        is_fault = g < n_fault
        t = rng.random() < type1_share if is_fault else False
        for o in range(windows_per_group):
            lab = int(is_fault and o >= windows_per_group - pos_per)
            gid.append(f"g{g:03d}"); label.append(lab); order.append(o)
            ftype.append(("cloud_only" if t else "edge_visible") if lab else "normal")
    y = np.array(label); ftype = np.array(ftype)
    ginv = np.unique(np.array(gid), return_inverse=True)[1]
    t0 = (ftype == "edge_visible").astype(float)
    t1 = (ftype == "cloud_only").astype(float)
    # window-level noise in fault visibility: some edge-visible faults are faint
    vis = rng.uniform(0.3, 1.5, size=n)
    off_e = rng.normal(0, group_sd, size=(n_groups, n_edge))[ginv]
    off_x = rng.normal(0, group_sd, size=(n_groups, n_extra))[ginv]
    Xe = rng.normal(size=(n, n_edge)) + off_e
    Xx = rng.normal(size=(n, n_extra)) + off_x
    k_e = max(1, n_edge // 4); k_x = max(1, n_extra // 4)
    Xe[:, :k_e] += (edge_signal * vis * t0 + 0.25 * edge_signal * t1)[:, None]
    Xx[:, :k_x] += (cloud_signal * t1 + 0.5 * cloud_signal * t0)[:, None]
    Xc = np.concatenate([Xe, Xx], axis=1).astype(np.float32)
    Xe = Xe.astype(np.float32)
    names_e = [f"e{j}__f" for j in range(n_edge)]
    names_c = names_e + [f"x{j}__f" for j in range(n_extra)]
    meta = pd.DataFrame(dict(window_id=np.arange(n), group_id=np.array(gid), label=y,
                             label_frac=y.astype(float), fault_type=ftype, condition="",
                             order=np.array(order), well=[f"w{g % 5}" for g in ginv]))
    out = dict(name=name, root="", meta=meta, y=y, groups=meta.group_id.to_numpy(), order=meta.order.to_numpy(),
               X_edge=Xe, X_cloud=Xc, names_edge=names_e, names_cloud=names_c, raw_edge=None, raw_cloud=None)
    chans_e = [f"E{j}" for j in range(ch_edge)]; chans_x = [f"X{j}" for j in range(ch_extra)]
    if raw:
        tt = np.arange(L) / L
        R = rng.normal(0, 1.0, size=(n, ch_edge + ch_extra, L)).astype(np.float32)
        freq = 4 + 2 * np.arange(ch_edge + ch_extra)
        base = np.sin(2 * np.pi * freq[None, :, None] * tt[None, None, :])
        amp = np.ones((n, ch_edge + ch_extra))
        amp[:, :ch_edge] += (1.5 * edge_signal * vis * t0)[:, None]
        amp[:, ch_edge:] += (1.5 * cloud_signal * t1)[:, None]
        R += (amp[:, :, None] * base).astype(np.float32)
        out["raw_cloud"] = R
        out["raw_edge"] = np.ascontiguousarray(R[:, :ch_edge])
    out["channels"] = dict(edge=chans_e, cloud=chans_e + chans_x, rates_hz={c: 100.0 for c in chans_e + chans_x},
                           edge_node="synthetic node")
    out["bytes"] = dict(raw_edge=ch_edge * L * 4, raw_cloud=(ch_edge + ch_extra) * L * 4,
                        features_edge=n_edge * 4, features_cloud=(n_edge + n_extra) * 4)
    return out


def write_synthetic(root, name: str, **kw) -> Path:
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    s = make_synthetic(name=name, **kw)
    s["meta"].to_parquet(d / "meta.parquet", index=False)
    np.save(d / "X_edge.npy", s["X_edge"]); np.save(d / "X_cloud.npy", s["X_cloud"])
    (d / "names_edge.json").write_text(json.dumps(s["names_edge"]))
    (d / "names_cloud.json").write_text(json.dumps(s["names_cloud"]))
    if s["raw_edge"] is not None:
        np.save(d / "raw_edge.npy", s["raw_edge"]); np.save(d / "raw_cloud.npy", s["raw_cloud"])
    (d / "channels.json").write_text(json.dumps(s["channels"]))
    (d / "bytes.json").write_text(json.dumps(s["bytes"]))
    return d
