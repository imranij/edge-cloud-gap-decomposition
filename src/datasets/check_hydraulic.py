import json, pathlib
import numpy as np, pandas as pd
D = pathlib.Path(__file__).resolve().parents[2] / "data/processed/hydraulic"
m = pd.read_parquet(D / "meta.parquet")
Xe, Xc = np.load(D / "X_edge.npy"), np.load(D / "X_cloud.npy")
Re, Rc = np.load(D / "raw_edge.npy"), np.load(D / "raw_cloud.npy")
ne, nc = json.load(open(D / "names_edge.json")), json.load(open(D / "names_cloud.json"))
ch = json.load(open(D / "channels.json")); by = json.load(open(D / "bytes.json"))
n = len(m)
assert n == 2205 and (m.window_id.values == np.arange(n)).all()
assert Xe.shape == (n, len(ne)) and Xc.shape == (n, len(nc)) and Xe.dtype == Xc.dtype == np.float32
assert nc[:len(ne)] == ne and np.array_equal(Xc[:, :len(ne)], Xe)
assert Re.shape == (n, len(ch["edge"]), 600) and Rc.shape == (n, len(ch["cloud"]), 600)
assert ch["cloud"][:len(ch["edge"])] == ch["edge"] and np.array_equal(Rc[:, :len(ch["edge"])], Re)
assert set(ch["edge"]) < set(ch["cloud"]) and not set(ch["excluded"]) & set(ch["cloud"])
for a in (Xe, Xc, Re, Rc):
    assert np.isfinite(a).all()
assert set(m.label) <= {0, 1} and (m.label == (m.pump >= 1).astype(int)).all()
assert ((m.label == 0) == (m.fault_type == "normal")).all() and (m.label_frac == m.label).all()
g = m.groupby("group_id")
assert g.size().max() <= 60
assert (g[["cooler", "valve", "pump", "accumulator"]].nunique() == 1).all().all()
assert (g["label"].nunique() == 1).all()                      # label constant within group
assert (g["window_id"].apply(lambda s: (np.diff(s.values) == 1).all())).all()  # contiguous
assert (m.groupby("group_id")["order"].apply(lambda s: (s.values == np.arange(len(s))).all())).all()
assert by["raw_edge"] == 24240 and by["raw_cloud"] == 174000
pos = g["label"].sum()
print(f"windows={n} groups={g.ngroups} positives={int(m.label.sum())} prevalence={m.label.mean():.4f}")
print("group size: min/median/max", g.size().min(), g.size().median(), g.size().max())
print("positive groups:", int((pos > 0).sum()), "negative groups:", int((pos == 0).sum()))
print("positives per positive group: min/median/max", pos[pos > 0].min(), pos[pos > 0].median(), pos[pos > 0].max())
print("fault_type counts:", m.fault_type.value_counts().to_dict())
print("stable==1 windows:", int(m.stable.sum()), " features:", Xe.shape[1], Xc.shape[1], " bytes:", by)
print("OK")
