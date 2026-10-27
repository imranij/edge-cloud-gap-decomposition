# Dataset card: 3W (Petrobras), real instances only

## Source and licence
- Canonical home: https://github.com/petrobras/3W, accessed 2026-10-11 by a shallow (`--depth 1`), blob-filtered (`--filter=blob:none`), sparse git clone into `data/raw/threew/repo`.
- Commit: `d57c058af6baf65f1b285f3e56f46e4f018a07d9` (2026-10-05, "Merge pull request #226"). 3W Dataset version 2.0.0 (`dataset/dataset.ini`).
- Licence: the dataset files are under CC BY 4.0 (`dataset/LICENSE-CC-BY`). The code is under Apache-2.0.
- Files checked out: all 1,119 `dataset/<event>/WELL-*.parquet` files, plus `dataset.ini`, `dataset/README.md`, the licence files, `CITATION.md`, `3W_DATASET_STRUCTURE.md`, the v2.0.0 introduction resource, and `docs/doctoral_thesis_ricardo_vargas.pdf`. SHA-256 checksums are in `sha256.txt`, with a copy in `data/raw/threew/sha256.txt`.
- Excluded sources, counted from the git tree and never downloaded: 1,089 SIMULATED and 20 DRAWN instances.

```bibtex
@article{vargas2019threew,
  author  = {Vaz Vargas, Ricardo Emanuel and Munaro, Celso Jos{\'e} and Marques Ciarelli, Patrick and Gon{\c{c}}alves Medeiros, Andr{\'e} and Guberfain do Amaral, Bruno and Centurion Barrionuevo, Dyego and Dias de Ara{\'u}jo, Jean Carlos and Lins Ribeiro, Jorge and Pierezan Magalh{\~a}es, Lucas},
  title   = {A realistic and public dataset with rare undesirable real events in oil wells},
  journal = {Journal of Petroleum Science and Engineering}, volume = {181}, pages = {106223}, year = {2019},
  doi     = {10.1016/j.petrol.2019.106223}}
@article{vargas2026threew2,
  author  = {Vaz Vargas, Ricardo Emanuel and de Melo Junior, Afr{\^a}nio Jos{\'e} and de Campos Lima, C. B. and others},
  title   = {3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells},
  journal = {Scientific Data}, volume = {13}, pages = {949}, year = {2026},
  doi     = {10.1038/s41597-026-07225-z}}
```
The 2.0.0 article (`vargas2026threew2`) is the primary citation. The author list is copied from `CITATION.md`. The DOIs were not resolved when the card was written.

## Sensors and placement
All instances are sampled at 1 Hz. Every within-instance timestamp step is exactly 1 s, which was checked on all 32,872,250 real timestamps. The thesis (Vargas 2019, p. 55) also states "taxa de amostragem fixa (1 Hz)".

Variable definitions come from `dataset.ini [PARQUET_FILE_PROPERTIES]`. Placement comes from the same file and from Vargas's 2019 thesis, Sec. 2.1-2.2 (pp. 28-30). That section says:
- the oil flows from the reservoir through the production tubing, the wet Christmas tree (ANM), the production line and the production choke (CKP) to the platform;
- the ANM contains remotely operated valves and the TPT;
- the PDG and TPT contain pressure and temperature sensors;
- the PDG sensor "usually fails" and is costly to replace;
- the TPT sensors have "good reliability".

| channel | placement | in view |
|---|---|---|
| P-TPT, T-TPT | subsea Xmas tree (ANM), one TPT transducer | edge + cloud |
| P-ANULAR | well annulus (measurement point not documented) | cloud |
| P-MON-CKP | topside, upstream of the production choke | cloud |
| P-JUS-CKGL | topside, downstream of the gas-lift choke (gas-lift line) | cloud |
| P-PDG, T-PDG | downhole PDG on the tubing | excluded (coverage) |
| T-JUS-CKP, QGL, ABER-CKP, ABER-CKGL, T-MON-CKP, P-JUS-CKP, P-MON-CKGL | topside chokes and gas-lift line | excluded (coverage) |
| PT-P, P-MON-SDV-P, P-JUS-BS, QBS | Xmas tree, SDV, service pump | excluded (never present in real instances) |
| ESTADO-* (9 valve states, 0/0.5/1) | Xmas tree, DHSV, SDVs | excluded (discrete actuator states, frozen in nearly every instance) |

### Cloud-view rule
A variable joins the cloud view if it is analog and its usable coverage is at least 60% on both of these measures:
- the share of all real-instance timestamps;
- the per-instance mean.

A value counts as usable when it is non-missing and comes from an instance in which the variable is not frozen (constant over the whole instance). The rule, threshold and full table are in `missingness.csv`, which is produced by the build script.

Three cases are worth noting:
- P-PDG is 75% non-missing but frozen in 58% of instances, mostly at 0. That leaves 23% usable, so it is excluded.
- P-JUS-CKGL passes narrowly, with a per-instance mean of 0.605.
- Values with |v| > 1e12 are historian sentinels and are treated as missing. They are always -1.18e42: 305,679 values in P-PDG and 872,003 in T-PDG.

## Edge node
The edge node is the subsea wet Christmas tree's TPT, which provides P-TPT and T-TPT. These two channels come from one physical transducer that sits inside the ANM, the subsea equipment closest to the well that has a local electro-hydraulic control link. The thesis (Sec. 2.2) names the TPT as one of the three sensor-bearing pieces of equipment, along with the PDG and CKP, and as the one experts consider reliable. The TPT also has the highest coverage of any equipment group (usable 0.80/0.79 by timestamp).

The topside choke was considered as the alternative node. Its group is sparse: only P-MON-CKP is usable, T-JUS-CKP is about 50% present, and P-JUS-CKP, T-MON-CKP and ABER-CKP are 1-27% present. The PDG fails the coverage rule. The node was chosen from placement and coverage only. No separability was computed.

## Labels
Per-timestamp `class` codes follow `dataset.ini`: 0 is normal, 1-9 are the event types, and 101-109 are transients (`TRANSIENT_OFFSET = 100`). A timestamp counts as fault when its class is not 0.

The first 3,600 s of every real instance has class and state both NA, so 12 windows per instance are unlabelled. Windows with no labelled timestamp are dropped. Every kept window is fully labelled.

Within each window:
- `label_frac` is the number of fault seconds divided by the number of labelled seconds;
- `label` is 1 when `label_frac >= 0.5` (only 65 windows have 0 < frac < 1);
- `fault_type` is the event name of the window's majority non-zero class modulo 100, or "normal" when the label is 0.

Extra columns:
- `instance_event` and `instance_event_name`: the instance's folder;
- `majority_fault_class`;
- `transient_frac_of_fault`;
- `state_mode`: the documented "well operational status", 0 in 99% of instances;
- `t_start`, `n_labelled`, `well` and `episode_id`.

`condition` is "".

## Windowing and grouping
- Windows are 300 samples (300 s) with a stride of 300, starting at the first timestamp of each instance. Any trailing partial window is dropped.
- The raw arrays have L = 300 at the native 1 Hz, with no resampling.
- There were 95,382 labelled windows. Each instance was then capped at 150 windows using a label-stratified sample:
  - proportional allocation, with at least one window per label present;
  - `numpy.random.default_rng(2026)`, applied to files in sorted order;
  - windows kept in time order.
- `group_id` is the instance file stem. `order` is the original window index within the instance, which is increasing and can have gaps after the cap.

## Counts
- 57,346 windows from 1,119 groups (instances), 40 wells and 208 episodes.
- 17,749 positive windows, a prevalence of 0.310.
- 481 groups contain positives. 106 groups are mixed and 638 are all-negative. Positive groups hold between 2 and 147 positives (median 23).
- Features: 34 edge and 85 cloud, using the 17 shared features per channel.
- Payload per window: raw edge 2,400 B, raw cloud 6,000 B, edge features 136 B, cloud features 340 B.

| event (folder) | real instances | wells (instances) | hours | windows | positives |
|---|---|---|---|---|---|
| 0 normal | 594 | 00002(209) 00006(113) 00001(93) 00005(81) 00008(57) 00003(26) 00004(12) 00007(2) 00019(1) | 3377 | 32693 | 0 |
| 1 abrupt increase of BSW | 4 | 00006(2) 00001 00002 | 66 | 497 | 159 |
| 2 spurious closure of DHSV | 22 | 00011(13) 00003(3) 00012(2) 00002 00009 00010 00013 | 77 | 650 | 357 |
| 3 severe slugging | 32 | 00014(31) 00001 | 190 | 1875 | 1875 |
| 4 flow instability | 343 | 00002(112) 00010(83) 00004(43) 00005(38) 00001(36) 00014(21) 00007(10) | 1025 | 7901 | 7901 |
| 5 rapid productivity loss | 11 | 00020(6) 00016(4) 00015 | 122 | 941 | 749 |
| 6 quick restriction in PCK | 6 | 00002(3) 00004(3) | 22 | 184 | 64 |
| 7 scaling in PCK | 36 | 00024(19) 00022(8) 00023(4) 00006(2) 00021(2) 00001 | 2185 | 5311 | 4390 |
| 8 hydrate in production line | 14 | 00019(4) 00026(3) and 7 wells with 1 each | 1336 | 2100 | 1627 |
| 9 hydrate in service line | 57 | 15 wells (00041: 9) | 732 | 5194 | 627 |

43 of the 57 type-9 real instances contain no fault label.

## Imputation
Missing values are filled in two steps:
1. Forward fill within the instance. This is causal, and the unlabelled first hour is included as history.
2. Any value still missing is set to the channel median of all observed values in the kept windows. This is a fixed constant, independent of any training split.

| channel | share imputed | by forward fill | by median |
|---|---|---|---|
| P-TPT | 12.7% | 0.05% | 12.7% |
| T-TPT | 10.3% | 0.05% | 10.2% |
| P-ANULAR | 17.7% | 0.05% | 17.6% |
| P-MON-CKP | 32.3% | 0.02% | 32.3% |
| P-JUS-CKGL | 32.0% | 0.01% | 32.0% |

Almost all missingness affects whole instances. When a channel is missing for an instance, every window from it carries a constant median.

## Caveats and leakage risks
1. **Adjacent instances.** Consecutive instances of the same well are chained in time. The unlabelled first hour of an instance repeats the previous instance's last hour (911 overlapping pairs), but labelled parts overlap by at most 1 s. Grouping by instance therefore leaves neighbouring, near-continuous data in different folds. `episode_id` (208 chains) and `well` (40) support stricter grouped CV, and a well-level sensitivity analysis is advised.
2. **Event type and well are confounded.** Normal instances (folder 0) come from 9 wells, and 74% of their windows come from WELL-00002, 00006 and 00001. Label-0 windows from all folders span 40 wells, with 62% from those three. Severe slugging is 31/32 from WELL-00014, and types 3 and 4 are fault from start to end. Well identity, which sensor scale and missingness both encode, can therefore act as a shortcut.
3. **Missingness differs by label.** This was descriptive only. P-MON-CKP is fully imputed in 39% of label-0 windows against 17% of label-1 windows, and T-TPT in 13% against 4%. These shares were inspected after the edge node and rule were fixed, while auditing imputation. P-PDG had already been excluded for being frozen, at 0.58 of instances.
4. **Raw outliers remain.** Values are physically implausible in places: P-TPT reaches 2.9e9 Pa and pressures go negative. Only sentinels above 1e12 were removed.
5. **Few real instances for some events.** Types 1, 5 and 6 have 4-11 real instances, because SIMULATED and DRAWN instances are excluded.
6. **Mixed units.** Raw values are in Pa and degrees C, unscaled.
7. **Frozen instances stay in.** Frozen (constant) instances of the selected channels are kept: P-TPT 3.4%, T-TPT 1.1%.

## Rebuild
- `python -I src/datasets/threew.py` rebuilds everything from `data/raw/threew/repo`. It also writes `missingness.csv`, `instances.csv` and `summary.json`.
- `python -I src/datasets/check_threew.py` runs the assertions and prints the counts.
