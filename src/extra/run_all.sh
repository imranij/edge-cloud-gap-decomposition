#!/usr/bin/env bash
# Post-plan analyses; NOT pre-specified. Reads results/ and data/processed/ only,
# refits no model, writes results/extra/*.csv, then rebuilds the figures and tables of the paper.
set -euo pipefail
cd "$(dirname "$0")"
for s in h4_nulls h3_feasibility h5_audit_breakdown lodo h2_extra seu_recordings oracle_reference \
         cost_model by_fault_type dataset_confounds \
         skab_breakdown fault_reach_test threew_labelling realised_budget system_cost stats_consistency router_paired; do
  echo "== $s"; python "$s.py" > /dev/null
done
cd ../..
python paper/tables/src/make_results_tables.py
python paper/tables/src/make_extra_tables.py
python paper/tables/src/make_extra_tables_2.py
python paper/figs/src/make_all.py
