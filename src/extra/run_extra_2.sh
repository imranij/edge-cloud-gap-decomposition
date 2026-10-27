#!/usr/bin/env bash
# POST-PLAN evaluation-only analyses (second batch). Reads results/, results_sens/ and data/ only, refits no model,
# writes results/extra/*.csv, then regenerates every table. survival_summary.py needs the variant summary
# (paper/tables/src/postplan_summary.py), so that runs first. round3_variants.py (shortcut-augmented and
# single-channel runs) runs before survival_summary.py, which reads its output for the stronger criterion (ii+).
set -euo pipefail
cd "$(dirname "$0")"
python ../../paper/tables/src/postplan_summary.py > /dev/null
for s in holdout_diagnostics overlap_corrected_tests audit_within_class router_by_well router_holm per_unit_escalation \
         arch_r3 capture_ratio_of_means skab_variants_timestrat round3_variants survival_summary; do
  echo "== $s"; python "$s.py" > /dev/null
done
cd ../..
python paper/tables/src/make_all_tables.py
