"""Post-plan deconfounded replication of H1-H5 (plan/postplan/DECONF_DECLARATION.md).

Links the pre-declared variant results into results_sens/deconf/<variant>/<pair> and runs the
unchanged src/analyze.py on them with the four variants as the 'confirmatory' set, so every
statistic and decision rule is computed by exactly the code that produced the confirmatory report.
Then writes results_sens/deconf/summary/compare_primary.csv (main vs deconfounded verdicts).
The output is labelled post-plan everywhere; it never replaces a confirmatory verdict.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DECL = {"hydraulic": ("hydskab", "hyd_blocks"), "skab": ("hydskab", "skab_filez"),
        "seu": ("threewseu", "seu_dm"), "threew": ("threewseu", "threew_dm")}


def main():
    out = ROOT / "results_sens" / "deconf"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    missing = []
    for parent, (area, var) in DECL.items():
        src = ROOT / "results_sens" / area / var
        if not src.exists():
            missing.append(str(src))
            continue
        dst = out / var
        dst.mkdir()
        for pair_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            shutil.copytree(pair_dir, dst / pair_dir.name)      # copy (glob does not follow symlinks reliably)
    if missing:
        raise SystemExit("missing variant results: " + ", ".join(missing))
    env = dict(os.environ)
    cmd = [sys.executable, str(ROOT / "src" / "analyze.py"), "--results", str(out),
           "--confirmatory", ",".join(v for _, v in DECL.values())]
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    main_p = pd.read_csv(ROOT / "results" / "summary" / "primary.csv")
    dec_p = pd.read_csv(out / "summary" / "primary.csv")
    cmp_ = main_p.merge(dec_p, on="hypothesis", suffixes=("_main", "_deconf"))
    cmp_.to_csv(out / "summary" / "compare_primary.csv", index=False)
    print(cmp_.to_string(index=False))


if __name__ == "__main__":
    main()
