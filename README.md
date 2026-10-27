# Decomposing the Edge-to-Cloud Accuracy Gap for Fault Detection in Condition Monitoring

Imran, King Fahd University of Petroleum and Minerals, Dhahran, Saudi Arabia.

This repository holds the code and the result files behind the paper. The paper asks how much of the
accuracy difference between an edge model, which sees only the sensors at one physical node, and a cloud
model, which sees all sensors, comes from the larger model and how much from the extra sensors. It also asks
whether a router can pick the windows that are worth sending to the cloud. The analysis was fixed in a written
plan before any model was fitted on the four confirmatory datasets (`plan/ANALYSIS_PLAN.md`, with its SHA-256
sum in `plan/FREEZE.txt`), and every later change is logged in `plan/DEVIATIONS.md`. The datasets are the UCI
hydraulic test rig, the Skoltech Anomaly Benchmark (SKAB), the Southeast University (SEU) gearbox data and the
real instances of the Petrobras 3W data. The analyses added after the plan, the sensitivity variants and the
simulation study of the proposed test designs are kept apart from the confirmatory run and labelled as
post-plan throughout. The tables in `paper/tables/` are written by scripts from files in `results/`
and `results_sens/`, and no number in them is typed by hand.

## What is in each folder

| folder | contents |
|---|---|
| `src/escalation/` | the library behind every run, with the shared feature extractor, the grouped splits, the model pairs, the routers, the factorial decomposition, the routing audit and the corrected resampled tests |
| `src/datasets/` | one build script and one check script per dataset, which turn the raw files into the processed arrays |
| `src/run_dataset.py`, `src/analyze.py` | the per-dataset runner of the plan and the confirmatory analysis of hypotheses H1 to H5 |
| `src/extra/` | post-plan analyses that refit no model and read only stored results |
| `src/sens/` | post-plan sensitivity variants (`hydskab`, `threewseu`, `designs`), router studies (`routers`), shortcut baselines (`shortcuts`), the simulation pilot (`simulation`) and the full validation of the proposed designs (`simulation_full`) |
| `tests/` | unit tests on synthetic data |
| `plan/` | the frozen plan, its addendum on the edge views, the freeze record, the deviations log and the post-plan records in `plan/postplan/` |
| `results/` | the confirmatory run, with the summaries in `summary/`, the post-plan outputs in `extra/` and the per-split outputs packed in `archives/` |
| `results_sens/` | the post-plan sensitivity runs, router studies and simulations, with summaries as plain files and per-split outputs packed in `runs.zip` |
| `results_timing/` | the single-split timing runs of pair C used for the compute rule in `plan/DEVIATIONS.md` |
| `results_pilots/` | the pilot results on SECOM and MetroPT-3 that suggested the hypotheses (see the README there) |
| `paper/tables/`, `paper/figs/` | the LaTeX tables and the figure PDFs of the paper, with the scripts that write them in `src/` |
| `data/` | the dataset cards and channel lists, and `data/README.md`, which gives the source of each raw dataset |
| `tools/` | `unpack_archives.py`, which restores the per-split outputs, and `code_fingerprint.py` |

Each dataset is run with three model pairs. In pair A the edge model is a logistic regression and the cloud
model a balanced random forest, in pair B both are histogram gradient boosting of two sizes, and in pair C
both are one-dimensional convolutional networks of two sizes. Some file names carry the prefixes `r4_` to
`r6_` or `round3`. These mark the batch in which a post-plan analysis was added, and the names were kept
because the paper cites the files by them.

## Setup

The code needs Python 3.13, and the results were produced with the package versions pinned in
`requirements.txt`, which install with

```
pip install -r requirements.txt
```

PyTorch is needed because `src/escalation/models.py` imports it for pair C. The CPU build is enough, since
all runs used the CPU only. Scripts find the repository root from their own location, so they can be run
from any folder.

The per-split outputs are stored as zip archives to keep the repository small. Running

```
python tools/unpack_archives.py
```

from any folder extracts every archive into the place it came from. The analysis, table and figure scripts
need this step, and it takes a few seconds.

The raw datasets are not redistributed, and `data/README.md` says where each one comes from and where it
goes under `data/raw/`. The SHA-256 sum of every raw file we read is in `data/processed/<dataset>/sha256.txt`.
The build scripts in `src/datasets/` then write the processed arrays, about 1.1 GB for the four datasets.

## Reproducing the results

The steps below are listed in the order the results depend on. The time column comes from the per-split logs
of the original run (`log.jsonl` in each run folder), on a machine with two processor cores and two threads
per process. A dash means the step was not timed as a whole.

| step | command | what it produces | time |
|---|---|---|---|
| 1 | `python src/datasets/<dataset>.py`, then `check_<dataset>.py` | processed arrays of each dataset, and for 3W `threew_subset.py` after `threew.py` | - |
| 2 | `python src/run_dataset.py <dataset> A` and `... A --h2` | pair A at full training data (20 splits) and at the H2 subsampling levels (60 splits) | hydraulic 5.5 min, SKAB 7.8 min, SEU 6.0 min, 3W 25.0 min |
| 3 | `python src/run_dataset.py <dataset> B` | pair B (20 splits) | 1.3, 3.6, 1.5 and 3.9 min |
| 4 | `python src/run_dataset.py threew A --group-col well` | the 3W sensitivity run grouped by well | 18.3 min |
| 5 | `python src/run_dataset.py <dataset> C` | pair C (10 splits, 12 for SEU) | 4.5, 1.3, 15.1 and 23.3 min |
| 6 | `python src/analyze.py` | `results/summary/`, including `REPORT.md` with the five confirmatory verdicts | 7 s |
| 7 | `python src/sens/hydskab/build.py`, `python src/sens/threewseu/build.py <variant>`, `src/sens/designs/build_*.py` | the sensitivity variants in `data/processed_sens/` | - |
| 8 | `python src/run_dataset.py <variant> <pair> --root data/processed_sens --out results_sens/<area>`, and `src/sens/designs/run_custom_holdout.py` for the hold-out designs | the sensitivity runs, summarised by `summarize.py` in each area | - |
| 9 | `src/sens/routers/p1_audit.py`, `p2_crossfit.py`, `p3_fixedtier.py`, `src/sens/designs/p1w_within_class.py`, `src/sens/shortcuts/run_shortcuts.py` | the corrected audit, the cross-fitted routers, the fixed-tier H2 variant and the shortcut baselines | - |
| 10 | `sh src/extra/run_all.sh`, then `sh src/extra/run_extra_2.sh` and the `r4_` to `r6_` scripts | the post-plan evaluation in `results/extra/` | - |
| 11 | `python src/sens/simulation/run.py` | the simulation pilot at 10 splits | - |
| 12 | `python src/sens/simulation_full/worker.py --procs <n>`, then `summarize_full.py` | the full validation, 13,000 replicate units and 340 truth chunks | about 97 CPU-hours |
| 13 | `python paper/tables/src/make_all_tables.py` | every table in `paper/tables/` | 32 s |
| 14 | `python paper/figs/src/make_all.py` | every figure in `paper/figs/` | 7 s |

Each script in steps 7 to 12 starts with a docstring that gives its exact command line, its inputs and its
outputs. The units of step 12 are deterministic, so the work can be split across machines with `--task` and
`--ntasks` and the union summarised. `plan/postplan/simulation_full.md` records the specification, which was
hashed before the run, and how the run was carried out.

## What was checked by re-running

We copied the repository into an empty folder, ran `tools/unpack_archives.py`, and used Python 3.13.16 with
the versions in `requirements.txt`. All 133 Python files passed a syntax check, and the unit tests gave 31
passes in about 100 s.

`src/analyze.py` rewrote every file in `results/summary/`, and all of them were identical to the committed
files, `REPORT.md` included. `make_all_tables.py` rewrote the 61 tables, and 59 were identical. In
`pp_controls.tex` and `pp_controls_b.tex` the shortcut column of the SKAB control variants shows a dash where
the committed tables show 0.961. That column is filled only after a check that the variant used the same test
folds as the main run, and the check reads per-split arrays of the sensitivity runs that are not included
here. The eight figures were redrawn and were identical to the committed PDFs pixel for pixel at 60 dpi.

Sixteen of the post-plan scripts in `src/extra/` were re-run and wrote files identical to those in
`results/extra/`. They are `h3_feasibility`, `h5_audit_breakdown`, `lodo`, `h2_extra`, `seu_recordings`,
`cost_model`, `fault_reach_test`, `system_cost`, `stats_consistency`, `router_paired`,
`overlap_corrected_tests`, `audit_within_class`, `router_holm`, `capture_ratio_of_means`, `r4_power` and
`r5_h4_factor`. The other scripts there need the processed feature arrays or the per-split arrays of the
sensitivity runs, so they were not re-run in this check.

For the simulation, the four reference replicates and their truth chunks (`results_sens/simulation_full/refs.zip`)
were recomputed with `worker.py` and compared with `verify_ref.py`, and all eight files agreed exactly. The two
base-scenario truth chunks in the canonical set differ from them by at most 4.2e-7, because those chunks were
computed on another machine, as `plan/postplan/simulation_full.md` records.

The comments and docstrings of the code were edited for this public copy. For the simulation files that
were hashed before the full run, `plan/postplan/CODE_FINGERPRINTS.txt` gives a fingerprint of each file's
syntax tree without docstrings. The fingerprints were computed on the hashed versions and match the files
here, which `python tools/code_fingerprint.py --check plan/postplan/CODE_FINGERPRINTS.txt` confirms. The
records `plan/postplan/DECONF_DECLARATION.md` and `plan/postplan/simulation_full.md` were edited in wording
only, so their sums no longer match `HASHES.txt` either. The frozen plan and its addendum are unchanged and
match `plan/FREEZE.txt`.

## What is not included

The raw datasets and the processed arrays built from them are not included, for the reasons given in
`data/README.md`. The test-set arrays of the sensitivity runs (`arrays_*.npz`, about 190 MB) are left out as
well, and so are the 13,000 replicate files of the full validation (about 440 MB). The replicates can be
regenerated exactly with `worker.py`, and their summaries are in `results_sens/simulation_full/summary/`.
The test-set arrays of the confirmatory run are included in `results/archives/`.

The launcher used to prepare the simulation for a computing cluster is not included, although
`plan/postplan/HASHES.txt` lists its sums, because the run was in the end carried out without the cluster.
The code of the pilot study belongs to an earlier exploratory project and is not part of this repository.
The paper's LaTeX source and our working notes are not part of the release either.

No licence is attached to this repository, which means the code may be viewed and cited while any reuse or
redistribution needs the author's permission.

## How to cite

Please cite both the paper and this archived copy of the code. The paper is

> Imran. Decomposing the edge-to-cloud accuracy gap for fault detection in condition monitoring. Working
> paper, King Fahd University of Petroleum and Minerals, 2026.

The archived copy of this repository is on Zenodo at
[10.5281/zenodo.23691982](https://doi.org/10.5281/zenodo.23691982), a DOI that always resolves to the
latest archived version. `CITATION.cff` holds the same details in a machine-readable form.
