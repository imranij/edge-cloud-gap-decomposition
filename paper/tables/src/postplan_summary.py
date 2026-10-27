#!/usr/bin/env python
"""POST-PLAN summaries for the manuscript. Reads only; fits no model.

    python paper/tables/src/postplan_summary.py

Reads the per-split outputs of every post-plan variant under results_sens/{hydskab,threewseu,holdout2,round3,hydabl}/<variant>/<pair>/
split_*.parquet (h2_q = 1 rows only) and summarises each quantity with the Nadeau-Bengio corrected mean and 95% CI,
using src/analyze.py's own nb_sizes / per_dataset_table, exactly as the confirmatory report does. A variant x pair is
COMPLETE only when it holds its full declared number of splits (EXPECTED_J); incomplete ones are written with
complete = False so that the table generators print a \\PENDING marker instead of a number. The leave-one-level-out
designs (hyd_cooler, hyd_accum, skab_lso) have one split per held-out level; they get the plain mean over folds and
no interval. Output: paper/tables/data/pp_variants.csv.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import analyze as A                                    # noqa: E402

OUT = ROOT / "paper" / "tables" / "data"
AREAS = {"hydskab": ["hyd_blocks", "hyd_runs", "hyd_weak", "hyd_severe", "hyd_stable", "hyd_cooler", "hyd_accum",
                     "skab_filez", "skab_base", "skab_lso", "skab_notemp", "skab_edgeP", "skab_cm_pressure",
                     "skab_cm_thermo", "skab_cm_flow"],
         "threewseu": ["seu_dm", "seu_zn", "threew_dm", "threew_dm__by_well", "threew_zn", "threew_zn__by_well",
                       "threew_inorm", "threew_inorm__by_well", "threew_noann"],
         # redesigned hold-outs, shortcut-augmented and single-channel variants, sensor-family ablation
         "holdout2": ["skab_lso2", "hyd_cooler2", "hyd_accum2",
                      # causal-offset features
                      "skab_lso2_base"],
         "round3": ["hyd_blocks_sc", "hyd_blocks_sc2", "skab_sc", "skab_e_flow", "skab_e_press", "skab_e_thermo"],
         "hydabl": ["hyd_blk_e_flow", "hyd_blk_e_press", "hyd_blk_e_temp", "hyd_blk_cm_flow", "hyd_blk_cm_press",
                    "hyd_blk_cm_temp",
                    # edge plus ONE flow sensor
                    "hyd_blk_e_fs1", "hyd_blk_e_fs2"]}
FOLD_ONLY = {"hyd_cooler": 3, "hyd_accum": 4, "skab_lso": 3, "skab_lso2": 3, "hyd_cooler2": 3, "hyd_accum2": 4,
             "skab_lso2_base": 3}
FIRST10 = {"skab_sc", "skab_e_flow", "skab_e_press", "skab_e_thermo"} | set(AREAS["hydabl"])  # first 10 splits


def expected_j(v: str) -> int:
    if v in FOLD_ONLY:
        return FOLD_ONLY[v]
    return 10 if v.startswith("threew") or v in FIRST10 else 20


def load_variant(area: str, v: str):
    frames = []
    base = ROOT / "results_sens" / area / v
    for pair in ("A", "B"):
        files = sorted((base / pair).glob("split_*.parquet")) if (base / pair).exists() else []
        files = [f for f in files if "_q" not in f.stem]
        if files:
            d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            d = d[(d.h2_q == 1.0) & (d.quantity != "tiny_settings")]
            d["dataset"] = v
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    rows = []
    for area, variants in AREAS.items():
        for v in variants:
            d = load_variant(area, v)
            if d is None:
                for pair in ("A", "B"):
                    rows.append(dict(dataset=v, area=area, pair=pair, quantity="_none", J=0,
                                     J_expected=expected_j(v), complete=False))
                continue
            d = d[d.quantity != "skipped"]
            sizes = A.nb_sizes(d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pdt = A.per_dataset_table(d, sizes, {v: "sensitivity"})
            if v in FOLD_ONLY:                         # no NB interval for one-split-per-level designs
                pdt["ci_lo"] = np.nan
                pdt["ci_hi"] = np.nan
                pdt["p_two_sided"] = np.nan
            pdt["area"] = area
            pdt["J_expected"] = expected_j(v)
            jmax = pdt.groupby("pair").J.max()
            pdt["J_pair"] = pdt.pair.map(jmax)
            pdt["complete"] = pdt.J_pair >= pdt.J_expected
            rows.extend(pdt.to_dict("records"))
            for pair in ("A", "B"):
                if pair not in set(pdt.pair):
                    rows.append(dict(dataset=v, area=area, pair=pair, quantity="_none", J=0,
                                     J_expected=expected_j(v), complete=False))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "pp_variants.csv", index=False)
    st = (out.groupby(["dataset", "pair"]).agg(J=("J_pair", "max"), J_expected=("J_expected", "max"),
                                              complete=("complete", "max")).reset_index())
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
