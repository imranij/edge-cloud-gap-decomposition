import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from escalation.data import cap_windows


def test_cap_keeps_every_cell_and_fraction():
    rng = np.random.default_rng(0)
    n = 5000
    g = np.repeat(np.arange(50), 100).astype(str)
    y = (rng.random(n) < 0.3).astype(int)
    ds = dict(y=y, groups=g, order=np.tile(np.arange(100), 50),
              meta=pd.DataFrame({"window_id": np.arange(n), "group_id": g, "label": y}),
              X_edge=rng.normal(size=(n, 3)).astype(np.float32), X_cloud=rng.normal(size=(n, 5)).astype(np.float32),
              raw_edge=rng.normal(size=(n, 1, 8)).astype(np.float32), raw_cloud=rng.normal(size=(n, 2, 8)).astype(np.float32))
    out = cap_windows(ds, 1000)
    assert abs(len(out["y"]) - 1000) < 120
    cells_in = set(zip(g, y)); cells_out = set(zip(out["groups"], out["y"]))
    assert cells_in == cells_out
    assert out["raw_cloud"].shape[0] == len(out["y"]) == out["X_cloud"].shape[0] == len(out["meta"])
    out2 = cap_windows(ds, 1000)
    assert np.array_equal(out["X_cloud"], out2["X_cloud"])
