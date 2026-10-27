#!/usr/bin/env python
"""Regenerate every table of the paper in the right order:  python paper/tables/src/make_all_tables.py

make_extra_tables_2.py overwrites some rev_*.tex files written by make_extra_tables.py, so it runs
last. The layout is single-column (elsarticle): wide tables are written as
two stacked panels or with wrapped label columns, and the overview table is split into overview.tex
(Hydraulic, SKAB) and overview_b.tex (SEU, 3W). No model is fitted; every cell is read from results/ or (post-plan) results_sens/. The two
post-plan scripts run last and print \\PENDING for any run that is not yet complete.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for name in ("make_results_tables.py", "make_extra_tables.py", "make_extra_tables_2.py",
             "postplan_summary.py", "make_postplan_tables.py"):
    runpy.run_path(str(HERE / name), run_name="__main__")
