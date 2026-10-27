# Declaration of the deconfounded replication set (post-plan, written before any variant was built or run)

Recorded: 2026-10-12T01:43:53Z (UTC), after the confirmatory results had been reported and before
any sensitivity dataset variant existed or any variant model was fitted.

The confirmatory analysis (plan/ANALYSIS_PLAN.md) stands as reported. As a post-plan robustness check,
the five primary analyses (H1-H5) will be recomputed once, with the same code and decision rules, on
one pre-declared deconfounded variant per dataset. The variant for each dataset was chosen as the one
that most directly removes the leakage route documented in its dataset card, while changing as little
else as possible:

| Dataset | Variant | What changes | Why this one |
|---|---|---|---|
| hydraulic | hyd_blocks | group_id = 20 contiguous blocks of cycles in time (blocked-in-time splits); features unchanged | card: adjacent groups share pump condition and thermal history; blocked-in-time splits recommended |
| skab | skab_filez | every feature z-scored within its recording (file or anomaly-free chunk); grouping unchanged | card: normal windows come from other sessions than the anomaly-free chunks; within-file normalisation removes per-recording offsets |
| seu | seu_dm | features recomputed on de-meaned windows, mean feature dropped; grouping unchanged | card: recording-level DC offsets are label-confounded |
| threew | threew_dm | features recomputed on de-meaned windows, mean feature dropped; episode grouping unchanged | card: well identity (encoded in absolute pressure and temperature level) is confounded with event type |

Pairs A and B are run for each variant; pair A also at the H2 levels. 3W variants use the first 10
splits (two repeats) instead of 20 to bound compute; all other variants use the plan's J. Results are
labelled post-plan and cannot change any confirmatory verdict. All other variants (other groupings,
normalisations and edge definitions) are reported as secondary sensitivity analyses.
