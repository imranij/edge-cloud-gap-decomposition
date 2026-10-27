# SKAB: Skoltech Anomaly Benchmark (processed for edge-to-cloud escalation)

## Source
- Canonical home: https://github.com/waico/SKAB (shallow clone over HTTPS), commit `b2c0d46c2971dcbfe71e26087b6d231998bb91c2` (last commit 2024-08-11), data version "SKAB v0.9".
- Licence: GPL-3.0 (repository LICENSE; GitHub badge in README). The Kaggle DOI 10.34740/KAGGLE/DSV/1693952 is the owners' citation DOI; the Kaggle copy was not used, and only the GitHub content was read.
- Raw files and SHA-256: `sha256.txt` (raw copies under `data/raw/skab/repo/`).
- Citation (as given in the repository README):
```bibtex
@misc{skab,
  author = {Katser, Iurii D. and Kozitsin, Vyacheslav O.},
  title = {Skoltech Anomaly Benchmark (SKAB)},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/dsv/1693952}},
  DOI = {10.34740/KAGGLE/DSV/1693952}
}
```

## Testbed and sensors
Water-circulation testbed: tank, motor-driven centrifugal pump, pipe loop with two valves (inlet to the pump = `valve1`, outlet = `valve2`), flow meter and pressure sensor (docs/pictures/testbed.png in the repo; the figure carries numbered call-outs but the repository contains no legend text for them). All 8 channels are logged at nominal 1 Hz (one row per second, timestamps to the second; no other rate is documented).

| Channel | README description (verbatim meaning) | Placement | In edge view |
|---|---|---|---|
| Accelerometer1RMS | vibration acceleration, g | on the motor-pump machine (exact mount not given in the text) | yes |
| Accelerometer2RMS | vibration acceleration, g | on the motor-pump machine (exact mount not given in the text) | yes |
| Current | "the amperage on the electric motor", A | motor supply | yes |
| Voltage | "the voltage on the electric motor", V | motor supply | yes |
| Temperature | "the temperature of the engine body", deg C | motor casing | yes |
| Pressure | "the pressure in the loop after the water pump", bar | hydraulic loop, pump discharge | no |
| Thermocouple | "the temperature of the fluid in the circulation loop", deg C | fluid in the loop | no |
| Volume Flow RateRMS | "the circulation flow rate of the fluid inside the loop", L/min (README calls it `RateRMS`; CSV header is `Volume Flow RateRMS`) | hydraulic loop | no |

## Edge-node justification
The edge node is the **pump-motor drive unit**: the sensors that measure the electric motor itself (Current, Voltage, Temperature of the engine body) plus the two vibration accelerometers, which on a rotating machine can only be on the motor-pump set (the testbed has no other rotating or vibrating part; Katser and Kozitsin, SKAB README and testbed photo). This is the group an on-machine controller would see without any plant network. The three excluded channels are all quantities of the water loop (pressure after the pump, fluid temperature, loop flow rate): the README describes each as a property of "the loop" or "the fluid", i.e. process-side measurements that need the hydraulic instrumentation and would be gathered by a supervisory (cloud/plant) layer. The choice was fixed from the documentation text alone, before any model was run and without looking at which channels separate the classes. Honest caveats: (i) the README does not state where each accelerometer is mounted (motor housing or pump casing); both are on the same coupled machine, so the node is unchanged either way; (ii) Pressure is measured at the pump discharge and could be argued to belong to the pump; we treat it as loop-side because the README says "in the loop". A 6-channel variant (edge + Pressure) is an admissible sensitivity analysis but was NOT chosen as primary.

## Fault label
Per-second `anomaly` column (0/1) supplied by the owners; every experiment file contains one injected disturbance (README data/README.md lists the type per file), shown as a single contiguous run of anomaly = 1 per file (it extends to the end of the recording in `other/1` and `other/4`). The `changepoint` column is kept only as metadata (`changepoint_in_window`).
- Window label = 1 if at least 50% of the 60 seconds are labelled anomalous, 0 if none are; windows with 0 < fraction < 0.5 (boundary windows around onset and offset) are **dropped** (147 windows) because they are neither clean normal nor clean fault. `label_frac` records the exact fraction.
- `fault_type` = the experiment's injected disturbance for label 1 (from data/README.md: valve1_inlet_closure, valve2_outlet_closure, fluid_leak_or_addition, rotor_imbalance, water_increase, cavitation, warm_water_supply) and "normal" for label 0; `exp_fault_type` keeps the experiment type for every window. `condition` is empty (no operating condition is documented).

## Windowing
60-s windows (L = 60 samples at 1 Hz, so `raw_*` arrays need no resampling), 10-s stride (83% overlap), never crossing a group boundary or a timing break. The README argues for a 60-s scale in its leaderboard (NAB detection window = 60 s "to reflect the dynamics of the transition"). Timestamps are on a 1-s grid but about 5% of seconds are absent (single missing seconds); the series is put on a regular 1-s grid and the missing seconds are filled by forward fill.
Timing breaks: where consecutive samples are more than 5 s apart the series is cut into segments and no window spans the cut. This happens in 6 files (valve1/2, valve1/4, valve1/7, valve2/1, other/2, other/13), 9 breaks and 586 unrecorded seconds in total (largest: 247 s in other/2, right at the anomaly onset). Windows never include an interval longer than 4 consecutive imputed seconds.

## Groups (independent unit)
`group_id` = one experiment file (`valve1_k`, `valve2_k`, `other_k`; k as the file name; valve1 files are numbered 0-15 on disk although data/README.md lists 1-16). The anomaly-free recording (9405 samples, 9961 s) has no label variation and is split into 8 contiguous chunks of 1245 s (`free_0` ... `free_7`), close to the median experiment length (about 1200 s), so that anomaly-free groups have comparable weight; 1 trailing second is unused. Windows of one file/chunk always share a group_id. `order` is time order within the group; `session` (recording date) is provided as an extra grouping option.

## Counts
- Windows: 4533; groups: 42 (34 with positives, 8 anomaly-free); positives: 1371; prevalence: 0.3024.
- Per subset:
| subset       |   windows |   positives |   groups |
|:-------------|----------:|------------:|---------:|
| anomaly-free |       952 |           0 |        8 |
| other        |      1433 |         548 |       14 |
| valve1       |      1731 |         664 |       16 |
| valve2       |       417 |         159 |        4 |
- Positive windows per fault type:
| fault_type             |   windows |   groups |
|:-----------------------|----------:|---------:|
| cavitation             |        62 |        2 |
| fluid_leak_or_addition |       136 |        4 |
| rotor_imbalance        |       208 |        5 |
| valve1_inlet_closure   |       664 |       16 |
| valve2_outlet_closure  |       159 |        4 |
| warm_water_supply      |        32 |        1 |
| water_increase         |       110 |        2 |
- Positives per group (min / median / max over groups with positives): 17 / 43 / 62.
- Per-group table: see `meta.parquet` (group_by group_id).
- Raw samples read: 46806; 1-s grid seconds used: 49442.

## Imputation
Raw CSVs contain no missing values (0 NaN). The only imputation is the filling of absent seconds on the 1-s grid by forward fill within the segment: 2637 of 49442 grid seconds (5.33%); because rows are missing as whole rows, the share is identical for all 8 channels. The fallback to the training-free channel median was never triggered (no leading gaps, since each segment starts at a real sample). Per-window imputed share is in `meta.imputed_frac` (mean 0.054, max 0.38; 56 windows exceed 20%, concentrated in other/12, other/13 and valve2 which have denser dropouts). Imputed seconds inherit the previous label; changepoint flags are not copied to imputed seconds.

## Known caveats and leakage risks
1. **Time-position confound within each experiment.** Almost every labelled file is normal first (typically the first 560-580 s; exceptions other/2 at 104 s, other/13 at 495 s, other/4 at 796 s), then anomalous, then normal again (except other/1 and other/4, which end anomalous). Added after the plan (post-plan, from `meta.parquet`; `results/extra/skab_onset_timing.csv`, written by `src/extra/r4_skab_clock.py`): the injections are synchronised. In 30 of the 34 experiment files the first fault window starts at 560-580 s (28 at 570 s, one at 560 s, one at 580 s) and in 26 the last fault window starts at 980-990 s (20 at 990 s), so the anomaly occupies about 570 to 990 s in most files. Only other/1, other/2, other/4, other/13 and valve1/2 are timed differently. Inside 120-s bins of elapsed time a clock at 10-s resolution therefore separates the classes almost perfectly (positive rate 0.00 at every position up to 540 s and 0.97 from 570 s in the 480-599 s bin), so matching on such bins does not remove the clock. For the anomaly-free chunks `t_start_s` is the time since the start of the anomaly-free recording (up to 9,895 s), not since the chunk start. Anything that drifts with run time (motor-body temperature warm-up, fluid temperature) is correlated with the label inside a group; grouped CV does not remove this.
2. **Normal windows come from different sessions than anomaly-free chunks.** The anomaly-free file is from 2020-02-08, `other/5-14` from 2020-02-08 (later the same day), `other/1-4` from 2020-03-01 and valve1/valve2 from 2020-03-09. Group-wise splits can place a session in both train and test; use `session` for a leave-session-out sensitivity analysis.
3. **Anomaly-free chunks are contiguous slices of one recording**; neighbouring chunks are autocorrelated, so groups are not fully independent.
4. **83% overlap between windows** means windows must never be split across folds (grouping handles this).
5. Experiment files follow a similar fixed script (typically about 570 s normal, then about 400 s of disturbance), so the label is partly predictable from `order`/`t_start_s`; these columns are metadata, not features.
6. Only 4 `valve2` and 14 `other` experiments, with 1-5 experiments per specific fault type, so per-fault-type conclusions are low-powered; `other/*` fault types differ strongly in kind.
7. The Kaggle landing page and the position paper (referred to in README as submitted) were not accessible; sensor placement relies on the repository README and photo only. The repository states the testbed is under repair (v0.9 is the current release; v1.0 is announced).
8. Sampling-rate jitter (about 5% missing seconds) means 60 samples is not exactly 60 s; the grid fixes this.
