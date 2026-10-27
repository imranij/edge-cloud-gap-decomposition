"""Consistency checks for data/processed/skab (no model is fitted)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import window_features, N_FEAT

D = ROOT / "data" / "processed" / "skab"
meta = pd.read_parquet(D / "meta.parquet")
Xe, Xc = np.load(D / "X_edge.npy"), np.load(D / "X_cloud.npy")
Re, Rc = np.load(D / "raw_edge.npy"), np.load(D / "raw_cloud.npy")
ne, nc = json.load(open(D / "names_edge.json")), json.load(open(D / "names_cloud.json"))
ch = json.load(open(D / "channels.json")); by = json.load(open(D / "bytes.json"))
n = len(meta)

assert (meta.window_id.to_numpy() == np.arange(n)).all()
assert Xe.shape == (n, len(ch["edge"]) * N_FEAT) and Xc.shape == (n, len(ch["cloud"]) * N_FEAT)
assert Re.shape == (n, len(ch["edge"]), 60) and Rc.shape == (n, len(ch["cloud"]), 60)
assert all(a.dtype == np.float32 for a in (Xe, Xc, Re, Rc))
for a in (Xe, Xc, Re, Rc):
    assert np.isfinite(a).all(), "NaN/inf"
# nested views: edge channels first, edge strict subset of cloud
assert ch["cloud"][:len(ch["edge"])] == ch["edge"] and len(ch["edge"]) < len(ch["cloud"])
assert nc[:len(ne)] == ne and np.array_equal(Xc[:, :Xe.shape[1]], Xe)
assert np.array_equal(Rc[:, :len(ch["edge"])], Re)
assert [c.split("__")[0] for c in ne[::N_FEAT]] == ch["edge"] and [c.split("__")[0] for c in nc[::N_FEAT]] == ch["cloud"]
# recompute features from raw arrays (shared code)
X2, _ = window_features(Rc, ch["cloud"]); assert np.allclose(X2, Xc, atol=1e-4, rtol=1e-4)
# labels
assert set(meta.label.unique()) <= {0, 1}
assert ((meta.label == 1) == (meta.label_frac >= 0.5)).all()
assert ((meta.label == 0) <= (meta.label_frac == 0)).all(), "label 0 windows must have no anomalous second"
assert (meta.loc[meta.label == 0, "fault_type"] == "normal").all() and (meta.loc[meta.label == 1, "fault_type"] != "normal").all()
# groups: one file or one chunk per group; contiguous time order, no group split across files
assert meta.groupby("group_id").file.nunique().le(1).all() or meta[meta.subset == "anomaly-free"].group_id.nunique() == 8
for gid, g in meta.groupby("group_id"):
    assert (np.diff(g.order.to_numpy()) == 1).all() and g.order.iloc[0] == 0
    assert (np.diff(g.t_start_s.to_numpy()) > 0).all()
assert not set(meta[meta.label == 1].group_id) & set(meta[meta.subset == "anomaly-free"].group_id)
assert by == dict(raw_edge=len(ch["edge"]) * 60 * 4, raw_cloud=len(ch["cloud"]) * 60 * 4,
                  features_edge=Xe.shape[1] * 4, features_cloud=Xc.shape[1] * 4, note=by["note"])



# counts
g = meta.groupby("group_id").agg(n=("label", "size"), pos=("label", "sum"))
print(f"windows {n}  groups {len(g)}  positives {int(meta.label.sum())}  prevalence {meta.label.mean():.4f}")
print("groups with positives", int((g.pos > 0).sum()), " anomaly-free", int((g.pos == 0).sum()))
print("positives per group: min %d median %d max %d" % (g.pos[g.pos > 0].min(), g.pos[g.pos > 0].median(), g.pos.max()))
print(meta.groupby("subset").agg(windows=("label", "size"), positives=("label", "sum"), groups=("group_id", "nunique")))
print("edge", ch["edge"]); print("cloud-only", ch["cloud"][len(ch["edge"]):])
print("features edge/cloud", Xe.shape[1], Xc.shape[1], " bytes", by)
print("imputed_frac mean %.4f max %.3f" % (meta.imputed_frac.mean(), meta.imputed_frac.max()))
print("zero-variance raw channel-windows (share):", dict(zip(ch["cloud"], np.round((Rc.std(axis=2) == 0).mean(axis=0), 4))))
print("OK")
