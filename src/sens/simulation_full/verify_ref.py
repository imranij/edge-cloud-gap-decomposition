"""POST-PLAN: compare a verification run with the reference units computed before the run.
    python verify_ref.py <run_dir> <ref_dir>       exit 0 if every reference file was reproduced (max |diff| <= 0.05),
Reports the maximum absolute difference per file; 0 or ~1e-15 means the same numbers as the reference machine.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run, ref = Path(sys.argv[1]), Path(sys.argv[2])
worst, bad = 0.0, []
for f in sorted(ref.rglob("*.csv")):
    g = run / f.relative_to(ref)
    if not g.exists():
        bad.append(f"MISSING {g}"); continue
    a, b = pd.read_csv(f), pd.read_csv(g)
    cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    if len(a) != len(b) or set(a.columns) != set(b.columns):
        bad.append(f"SHAPE {f.name}: {a.shape} vs {b.shape}"); continue
    d = float(np.nanmax(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float))))
    nan_mismatch = int((a[cols].isna().to_numpy() != b[cols].isna().to_numpy()).sum())
    worst = max(worst, d)
    print(f"{f.relative_to(ref)}: max |diff| = {d:.3g}, NaN mismatches = {nan_mismatch}")
    if d > 0.05 or nan_mismatch:
        bad.append(f"DIFF {f.name}: {d:.3g}, NaN mismatches {nan_mismatch}")
print("WORST", worst)
if bad:
    print("VERIFICATION FAILED:\n" + "\n".join(bad)); sys.exit(1)
print("VERIFICATION OK" + (" (identical to machine precision)" if worst < 1e-9 else " (small numerical differences)"))
