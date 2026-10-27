"""POST-PLAN, descriptive. Refits nothing in the study; re-summarises the finished
pilot-simulation extension (results_sens/simulation/reps, 200 replicates per scenario, last file 13:11 UTC) with the
unchanged src/sens/simulation/summarize.py, writing into a SEPARATE folder so that the cited 10:22 UTC snapshot
(results_sens/simulation/rates.csv, tables.md, tests_per_replicate.csv; Table S-sim) is not touched.

    python src/extra/r6_sim_r200.py

Copy: results_sens/simulation_r200/ holds a symlink reps -> ../simulation/reps, copies of truth_draws_*.csv and a
copy of the per-replicate test cache (keyed by scenario, replicate and truth tag, so cached rows are reused only
where they are identical). Outputs: results_sens/simulation_r200/{truth,rates,tests_per_replicate,audit_*}.csv and
the rates quoted in the paper, results/extra/r6_sim_rates_r200.csv (R = 200 beside the snapshot value).
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results_sens" / "simulation"
DST = ROOT / "results_sens" / "simulation_r200"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sens" / "simulation"))


def main():
    DST.mkdir(exist_ok=True)
    if not (DST / "reps").exists():
        (DST / "reps").symlink_to(SRC / "reps", target_is_directory=True)
    for f in SRC.glob("truth_draws_*.csv"):
        shutil.copy2(f, DST / f.name)
    if not (DST / "tests_per_replicate.csv").exists():
        shutil.copy2(SRC / "tests_per_replicate.csv", DST / "tests_per_replicate.csv")
    import summarize as SM
    SM.OUT = DST
    SM.main()
    new = pd.read_csv(DST / "rates.csv"); old = pd.read_csv(SRC / "rates.csv")
    keys = ["h4_orig_supported", "h4_origR0_supported", "h4_cor_2stage_supported", "h4_cor_mixed_p_lt05",
            "h4_cor_mixed_supported", "h2_orig_supported", "h2_orig_p_lt05"]
    rows = []
    for k in keys:
        for sc in ("S0",):
            n = new[(new.key == k) & (new.scenario == sc)]; o = old[(old.key == k) & (old.scenario == sc)]
            if len(n):
                rows.append(dict(key=k, scenario=sc, design=n.design.iloc[0], quantity=n.quantity.iloc[0],
                                 rate_R200=float(n.rate.iloc[0]), n_rep_R200=int(n.n_rep.iloc[0]),
                                 rejections_R200=round(float(n.rate.iloc[0]) * int(n.n_rep.iloc[0])),
                                 mcse_R200=float(n.mcse.iloc[0]),
                                 rate_snapshot=float(o.rate.iloc[0]) if len(o) else float("nan"),
                                 n_rep_snapshot=int(o.n_rep.iloc[0]) if len(o) else -1))
    # H1 per-dataset NB coverage (all scenarios) and any-scenario rejection count of the proposed two-stage H4
    cov = new[new.key.astype(str).str.contains("cov", case=False) & new.test.astype(str).str.startswith("H1")]
    for _, r in cov.iterrows():
        rows.append(dict(key=r.key, scenario=r.scenario, design=r.design, quantity=r.quantity, rate_R200=r.rate,
                         n_rep_R200=r.n_rep, mcse_R200=r.mcse))
    for _, r in new[new.key == "h4_cor_2stage_supported"].iterrows():
        rows.append(dict(key="h4_cor_2stage_supported (all scenarios)", scenario=r.scenario, design=r.design,
                         quantity=r.quantity, rate_R200=r.rate, n_rep_R200=r.n_rep,
                         rejections_R200=round(r.rate * r.n_rep), mcse_R200=r.mcse))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "extra" / "r6_sim_rates_r200.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(out.to_string())


if __name__ == "__main__":
    main()
