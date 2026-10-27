# SEU gearbox (Drivetrain Dynamics Simulator), dataset card

## Source
- Official release by the authors: https://github.com/cathysiyu/Mechanical-datasets, folder `gearbox/` (subfolders `gearset`, `bearingset`). Shallow sparse clone over HTTPS.
- Commit: `9732a2820f4cdf91fa7d1d0cc95442376660f5dd` (2018-01-16). Only `gearbox/` used. The repository's CWRU bearing copy was NOT used or downloaded.
- Licence: none stated in the repository (no LICENSE file; README gives no terms). Used for research with citation; redistribution of raw files is not asserted.
- Data collected by the group of Prof. Ruqiang Yan, Southeast University, on a Drivetrain Dynamics Simulator (repo file `gearbox/dds`).
- SHA-256 of all 20 raw CSVs: `sha256.txt`. Rebuild: `python3 -I src/datasets/seu.py`; checks: `python3 -I src/datasets/check_seu.py`.

## Citation
```bibtex
@article{shao2019highly,
  author  = {Shao, Siyu and McAleer, Stephen and Yan, Ruqiang and Baldi, Pierre},
  title   = {Highly Accurate Machine Fault Diagnosis Using Deep Transfer Learning},
  journal = {IEEE Transactions on Industrial Informatics},
  volume  = {15}, number = {4}, pages = {2446--2455}, year = {2019},
  doi     = {10.1109/TII.2018.2864759}
}
```
Volume/pages confirmed by a search listing; the DOI string could not be resolved when the card was written, so it is listed as unverified.

## Sensors and placement (README)
Each CSV has 8 columns after a ~16-line text header: 1 motor vibration; 2, 3, 4 planetary-gearbox vibration x, y, z; 5 motor torque; 6, 7, 8 parallel-gearbox vibration x, y, z. The README states that rows 2-4 are the "effective" signals. Names used: motor_vib, planet_x/y/z, motor_torque, par_x/y/z.
Sampling rate: NOT stated in the README or repo. The DAQ header gives "Frequency Limit 2000" and "Spectral Lines 1600"; with the usual 2.56 x frequency-limit convention this gives 5120 Hz (and 1600 lines at 1.25 Hz resolution = 4096-sample blocks, consistent). 5120 Hz is also the rate commonly quoted in the literature for this dataset, but I could not read the paper to confirm. It is recorded as "inferred" in channels.json. All frequency-domain features in the shared code are normalised to Nyquist, so no result depends on the absolute rate. Header "Total Data Rows 4194304" is wrong: each file has 1,048,560 data rows (about 204.8 s at 5120 Hz). All 8 channels share one rate, so no resampling; L = 1024 native samples.

## Edge node (chosen from documentation only)
Edge = the motor: motor_vib (channel 1) and motor_torque (channel 5). The README lists these two as the motor-side signals, distinct from the two gearboxes (planetary, parallel); motor vibration and torque are measured at the drive end, upstream of both gearboxes, which is the position of a drive-side condition-monitoring node in a drivetrain (Shao et al. 2019 describe the DDS test rig with motor, planetary and parallel gearboxes). Choice was made before any feature or label inspection. Cloud = all 8 channels, order: motor_vib, motor_torque, planet_x, planet_y, planet_z, par_x, par_y, par_z (edge first). Edge features are the first 34 columns of cloud (2 channels x 17 features, asserted).

## Label
Binary: health recordings 0, every fault recording 1. fault_type: normal, chipped, miss, root, surface (subset `gear`), ball, inner, outer, comb (subset `bearing`). `subset` column holds gear/bearing; `condition` is the speed-load configuration "20-0" or "30-2". Labels come from FILE NAMES; the titles inside the CSV headers are copy-paste artefacts and are often wrong (e.g. bearingset/ball_20_0.csv is titled "outer_20_0_1-10", comb_* titled "inner_*", Chipped_30_2 titled "c_20_0"), so they are ignored. Note the gear and bearing sets each have their own health recordings (4 health files total); the bearing and gear faults are in different test configurations.

## Windowing and grouping
- L = 1024 samples, stride 1024 (no overlap), time order retained (`order`, `start_sample`, `window_index_in_file`).
- Fixed rule: the first TRIM = 256 windows of every file are dropped (label-blind), then 400 windows evenly spaced (rounded linspace) over windows 256..1022 of each file. 20 files x 400 = 8000 windows. Reason for the trim: several recordings (health and comb in bearingset, Miss_20_0, Health_30_2, Root, Surface, outer, ...) start with a slowly decaying DC offset on motor_vib (down to about -2.5 at sample 0, settling over roughly 100-200 windows). It occurs in both classes, was identified from raw channel means only, and the same trim is applied to all files. Residual within-file drift of the window mean afterwards is at most 0.025. With the shared feature code the `mean` feature remains a recording-level offset, which is a leakage route (below).
- group_id = one recording file, e.g. `gear/Chipped_20_0` (20 groups; 16 fault, 4 health). Grouped CV must hold out whole files.

## Counts
Windows 8000; groups 20; positives 6400; prevalence 0.80; positives per group 400 (fault files) or 0 (health files). By type: normal 1600, each of the 8 fault types 800. Per subset: gear 4000 (3200 pos), bearing 4000 (3200 pos). Condition: 10 files each.

## Imputation
Rule implemented (forward fill within file, then channel median); share of non-finite values = 0.0 for all 8 channels, so nothing was imputed. No NaN/inf in any output.

## Payload (float32, per window, bytes.json)
raw_edge 8192, raw_cloud 32768, features_edge 136 (2 channels x 17 features x 4 B), features_cloud 544 (8 x 17 x 4 B).

## Caveats
1. Pseudo-replication: every fault is one seeded fault on one test rig, recorded once per condition; the 400 windows of a file are one physical realisation, strongly autocorrelated. Only 20 independent groups (4 health). Effective sample size is far smaller than 8000; inference must use group-level units (bootstrap/permutation over files).
2. Grouped CV with whole files held out means a fold test set contains only one or a few classes; with 4 health files, folds can lack negatives. Use stratified-group or leave-one-condition-out schemes and report fold composition.
3. Label-defined groups: group and label are perfectly aligned (each file is one class), so any recording-level artefact (DC offset, gain, session) is label-confounded. Residual offsets differ between files (motor_vib file means roughly -0.12 to -0.26) and the `mean` feature can exploit them; the drift/offset cannot be separated from the fault in this data.
4. Prevalence 0.8 is an artefact of the design (4 health vs 16 fault recordings), not a deployment prior; metrics need prevalence-aware reporting.
5. Bearing and gear fault types never co-occur within a rig configuration, so "subset" identification is possible from the planetary/parallel channels; edge-vs-cloud differences may reflect fault location rather than fault severity.
6. Condition (20-0 vs 30-2) shifts torque and vibration scale; per-condition splits are possible but the number of groups per condition is 10.
7. Sampling rate inferred, not documented; the DOI could not be verified.
