"""Consistency checks for data/processed/threew (no model is fitted)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import window_features, N_FEAT  # noqa: E402

D = ROOT / "data" / "processed" / "threew"
m = pd.read_parquet(D / "meta.parquet")
Xe, Xc = np.load(D / "X_edge.npy"), np.load(D / "X_cloud.npy")
Re, Rc = np.load(D / "raw_edge.npy", mmap_mode="r"), np.load(D / "raw_cloud.npy", mmap_mode="r")
ne, nc = json.load(open(D / "names_edge.json")), json.load(open(D / "names_cloud.json"))
ch = json.load(open(D / "channels.json"))
by = json.load(open(D / "bytes.json"))
n, E, C, L = len(m), ch["edge"], ch["cloud"], 300

# shapes, dtypes, finiteness
assert (m.window_id.to_numpy() == np.arange(n)).all()
assert Xe.shape == (n, len(E) * N_FEAT) and Xc.shape == (n, len(C) * N_FEAT)
assert Re.shape == (n, len(E), L) and Rc.shape == (n, len(C), L)
assert all(a.dtype == np.float32 for a in (Xe, Xc, Re, Rc))
for a in (Xe, Xc):
    assert np.isfinite(a).all()
for i in range(0, n, 5000):
    assert np.isfinite(Rc[i:i + 5000]).all() and np.isfinite(Re[i:i + 5000]).all()
# nested views
assert E == ["P-TPT", "T-TPT"] and C[:len(E)] == E and set(E) < set(C)
assert not set(ch["excluded"]) & set(C)
assert nc[:len(ne)] == ne and np.array_equal(Xc[:, :Xe.shape[1]], Xe)
assert [c.split("__")[0] for c in ne[::N_FEAT]] == E and [c.split("__")[0] for c in nc[::N_FEAT]] == C
for i in range(0, n, 10000):
    assert np.array_equal(Rc[i:i + 10000, :len(E)], Re[i:i + 10000])
# features reproducible from raw arrays (sample)
s = np.random.default_rng(0).choice(n, 2000, replace=False)
s.sort()
X2, _ = window_features(np.asarray(Rc[s]), C)
assert np.allclose(X2, Xc[s], rtol=1e-4, atol=1e-3)
# labels
assert set(m.label.unique()) <= {0, 1}
assert ((m.label == 1) == (m.label_frac >= 0.5)).all()
assert ((m.label == 0) == (m.fault_type == "normal")).all()
assert (m.n_labelled > 0).all() and m.label_frac.between(0, 1).all()
assert (m[m.instance_event == 0].label == 0).all() and (m[m.instance_event == 0].label_frac == 0).all()
# groups: one instance file per group, one well per group, strictly increasing order
g = m.groupby("group_id")
assert (g.episode_id.nunique() == 1).all() and (m.groupby("episode_id").well.nunique() == 1).all()
assert (g.well.nunique() == 1).all() and (g.instance_event.nunique() == 1).all()
assert (m.group_id.str.split("_").str[0] == m.well).all()
assert g.order.apply(lambda o: (np.diff(o.to_numpy()) > 0).all()).all()
assert g.size().max() <= 150
assert g.window_id.apply(lambda w: (np.diff(w.to_numpy()) == 1).all()).all()   # groups contiguous
assert by == dict(raw_edge=len(E) * L * 4, raw_cloud=len(C) * L * 4, features_edge=Xe.shape[1] * 4,
                  features_cloud=Xc.shape[1] * 4, note=by["note"])

pos = g.label.sum()
print(f"windows={n} groups={g.ngroups} wells={m.well.nunique()} positives={int(m.label.sum())} "
      f"prevalence={m.label.mean():.4f}")
print(f"edge={E} cloud={C} features edge/cloud={Xe.shape[1]}/{Xc.shape[1]}")
print("groups with positives:", int((pos > 0).sum()), " all-negative groups:", int((pos == 0).sum()),
      " mixed groups:", int(((pos > 0) & (pos < g.size())).sum()))
print("positives per positive group: min/median/max", pos[pos > 0].min(), pos[pos > 0].median(), pos[pos > 0].max())
print("windows per group: min/median/max", g.size().min(), g.size().median(), g.size().max())
print("fault_type:", m.fault_type.value_counts().to_dict())
print("label by instance event:\n", pd.crosstab(m.instance_event_name, m.label).to_string())
print("episodes:", m.episode_id.nunique(), " positive windows by well (top 10):",
      m[m.label == 1].well.value_counts().head(10).to_dict())
print("windows per well (top 10):", m.well.value_counts().head(10).to_dict())
print("label_frac strictly in (0,1):", int(((m.label_frac > 0) & (m.label_frac < 1)).sum()))
print("bytes:", by)
print("OK")
