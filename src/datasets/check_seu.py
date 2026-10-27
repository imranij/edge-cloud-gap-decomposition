import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import N_FEAT
D = ROOT / "data/processed/seu"
m = pd.read_parquet(D / "meta.parquet")
Xe, Xc = np.load(D / "X_edge.npy"), np.load(D / "X_cloud.npy")
Re, Rc = np.load(D / "raw_edge.npy"), np.load(D / "raw_cloud.npy")
ne, nc = json.load(open(D / "names_edge.json")), json.load(open(D / "names_cloud.json"))
ch = json.load(open(D / "channels.json")); by = json.load(open(D / "bytes.json"))
n = len(m)
assert list(m.window_id) == list(range(n))
assert Xe.shape == (n, 2 * N_FEAT) and Xc.shape == (n, 8 * N_FEAT) and Xe.dtype == Xc.dtype == np.float32
assert Re.shape == (n, 2, 1024) and Rc.shape == (n, 8, 1024) and Re.dtype == Rc.dtype == np.float32
for a in (Xe, Xc, Re, Rc): assert np.isfinite(a).all()
assert ne == nc[:len(ne)] and np.array_equal(Xe, Xc[:, :Xe.shape[1]])      # nested views
assert np.array_equal(Re, Rc[:, :2, :]) and ch["cloud"][:2] == ch["edge"]
assert set(m.label) <= {0, 1} and ((m.fault_type == "normal") == (m.label == 0)).all()
assert (m.label_frac == m.label).all()
g = m.groupby("group_id")
assert (g.label.nunique() == 1).all() and (g.condition.nunique() == 1).all() and (g.fault_type.nunique() == 1).all()
for _, x in g: assert x.order.tolist() == list(range(len(x))) and x.start_sample.is_monotonic_increasing
assert by == {**by, "raw_edge": 2 * 1024 * 4, "raw_cloud": 8 * 1024 * 4, "features_edge": Xe.shape[1] * 4, "features_cloud": Xc.shape[1] * 4}
print("windows", n, "groups", g.ngroups, "positives", int(m.label.sum()), "prevalence", round(m.label.mean(), 3))
print(g.agg(label=("label", "first"), fault=("fault_type", "first"), cond=("condition", "first"), n=("label", "size")).to_string())
print("OK")
