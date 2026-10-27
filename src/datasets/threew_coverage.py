"""Per-instance usable coverage of the five 3W cloud channels (descriptive, no models).
A channel is usable in an instance if it is not frozen over the whole instance; its coverage
is the share of present (non-missing, non-placeholder) timestamps. Written before any model
run to define the complete-case confirmatory subset (see plan/DEVIATIONS.md)."""
import json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO = ROOT / "data/raw/threew/repo/dataset"
CH = json.load(open(ROOT / "data/processed/threew/channels.json"))["cloud"]
SENTINEL = 1e30
rows = []
for f in sorted(REPO.glob("[0-9]/WELL-*.parquet")):
    df = pd.read_parquet(f, columns=CH + ["class"])
    lab = df["class"].notna().to_numpy()
    r = dict(file=f.name)
    for v in CH:
        x = df[v].to_numpy(dtype=float, copy=True)
        x[np.abs(x) > SENTINEL] = np.nan
        x = x[lab]
        p = ~np.isnan(x)
        frozen = bool(p.any() and np.nanstd(x) == 0)
        r[v] = 0.0 if (frozen or not p.any()) else float(p.mean())
    rows.append(r)
cov = pd.DataFrame(rows)
cov.to_csv(ROOT / "data/processed/threew/instance_coverage.csv", index=False)
print(cov[CH].describe().T[["mean", "min", "50%"]])
