#!/usr/bin/env python
"""Build every results figure of the paper:  python paper/figs/src/make_all.py [--analyze] [--only NAME ...]

Reads only results/ (summary CSVs written by src/analyze.py, split_*.parquet, arrays_*.npz); writes vector
PDFs to paper/figs/. Pair B / pair C are included when their files exist with >= MIN_SPLITS splits.
--analyze  first re-runs src/analyze.py (python src/analyze.py --n-perm 2000 --n-boot 2000) so the summary
           CSVs match the latest per-split files.
"""
from __future__ import annotations
import argparse
import importlib
import shutil
import subprocess
import sys
import time

import common as C

FIGS = ["fig_ladder", "fig_budget", "fig_routers", "fig_h2", "fig_h4", "fig_audit", "fig_arch"]
WIDTH_OK = {C.W1, C.W2, C.W_MAIN}


def pdf_width_in(path):
    if shutil.which("pdfinfo") is None:
        return None
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size"):
            return float(line.split()[2]) / 72.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.analyze:
        subprocess.run([sys.executable, str(C.ROOT / "src" / "analyze.py"), "--n-perm", "2000", "--n-boot", "2000"],
                       cwd=C.ROOT, check=True)
    print("data status:", C.pair_note())
    status = {}
    for name in (a.only or FIGS):
        t0 = time.time()
        print(f"[{name}]")
        mod = importlib.import_module(name)
        mod.main()
        if name == "fig_budget":                   # pair B deferral curves for the supplement
            mod.main("B")
        w = pdf_width_in(C.OUT / f"{name}.pdf")
        ok = w is not None and any(abs(w - x) < 0.01 for x in WIDTH_OK)
        status[name] = (w, ok, time.time() - t0)
    print("\nfigure            width_in  ok   seconds")
    for k, (w, ok, dt) in status.items():
        print(f"{k:<16} {w if w is None else round(w, 3)!s:>8}  {'yes' if ok else 'NO ':<4} {dt:6.1f}")
    if not all(v[1] for v in status.values()):
        sys.exit("a figure has a non-standard width")


if __name__ == "__main__":
    main()
