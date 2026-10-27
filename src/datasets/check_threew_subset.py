import json, pathlib, numpy as np, pandas as pd
D = pathlib.Path(__file__).resolve().parents[2] / "data/processed/threew"
m = pd.read_parquet(D / "meta.parquet")
Xe, Xc = np.load(D / "X_edge.npy", mmap_mode="r"), np.load(D / "X_cloud.npy", mmap_mode="r")
Re, Rc = np.load(D / "raw_edge.npy", mmap_mode="r"), np.load(D / "raw_cloud.npy", mmap_mode="r")
n = len(m)
assert Xe.shape[0] == Xc.shape[0] == Re.shape[0] == Rc.shape[0] == n
assert np.array_equal(np.asarray(Xe), np.asarray(Xc[:, :Xe.shape[1]])), "edge features must be the first cloud columns"
assert np.array_equal(np.asarray(Re), np.asarray(Rc[:, :Re.shape[1]])), "edge raw must be the first cloud channels"
assert np.isfinite(Xc).all() and np.isfinite(Rc).all()
assert set(m.label.unique()) <= {0, 1} and (m.window_id.to_numpy() == np.arange(n)).all()
g = m.groupby("group_id")
assert (g.well.nunique() == 1).all(), "an episode must belong to one well"
assert (g.order.apply(lambda s: (s.to_numpy() == np.arange(len(s))).all())).all()
print(f"ok: {n} windows, {m.group_id.nunique()} episodes, {m.instance.nunique()} instances, "
      f"{m.well.nunique()} wells, prevalence {m.label.mean():.3f}, edge {Xe.shape[1]} / cloud {Xc.shape[1]} features")
