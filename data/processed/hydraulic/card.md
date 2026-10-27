# Dataset card: hydraulic (ZeMA hydraulic test rig), task = internal pump leakage

## Source
- Condition monitoring of hydraulic systems, creator N. Helwig (ZeMA gGmbH), donated April 2018. Canonical homes: UCI ML Repository dataset 447 (DOI 10.24432/C5CW21) and Zenodo 10.5281/zenodo.1323611 (v1, 26 April 2018, data.zip md5 ac0b28f34686660219c4e43417053f9e). Licence: CC BY 4.0 (both pages).
- Provenance: the raw files in data/raw/hydraulic are a local copy of the UCI release. SHA-256 of every raw file is in sha256.txt. The Zenodo page lists only a zip md5, so the per-file hashes could not be cross-checked against the owner.
- Citation (requested in description.txt):
```
@inproceedings{helwig2015condition,
  author    = {Helwig, Nikolai and Pignanelli, Eliseo and Sch{\"u}tze, Andreas},
  title     = {Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics},
  booktitle = {Proc. I2MTC 2015, IEEE Int. Instrumentation and Measurement Technology Conference},
  address   = {Pisa, Italy}, year = {2015}, doi = {10.1109/I2MTC.2015.7151267}}
```

## Sensors and rates (documentation.txt, description.txt)
PS1-PS6 pressure (bar) and EPS1 motor power (W): 100 Hz (6000 samples/cycle). FS1-FS2 volume flow (l/min): 10 Hz (600). TS1-TS4 temperature (deg C), VS1 vibration (mm/s): 1 Hz (60). CE, CP, SE (virtual): 1 Hz (60). 2205 cycles of 60 s, no missing values.

## Edge node: the motor-pump drive = {EPS1, VS1}
Evidence read: documentation.txt and description.txt (UCI release); Helwig and Schuetze, SENSOR 2015, D8.1 (doi 10.5162/sensor2015/D8.1; text read at ama-science.org) which names EPS as "electrical pump input power"; Helwig and Schuetze, 10th Int. Fluid Power Conf. 2016, pp. 425-436 (TU Dresden, qucosa:29408), which states "the vibration sensor (VS1) installed at the main pump MP1", describes MP1 as an internal gear pump with a 3.3 kW motor, and describes EPS as motor power.
- EPS1: electrical input power of the motor that drives the main pump MP1; a drive-level quantity by its type. Included.
- VS1: vibration sensor installed at MP1 (explicit source statement). Included.
- Pressure sensors PS1-PS6, flow sensors FS1-FS2, temperature sensors TS1-TS4: no source we could read states which of them sits at the pump (the sensor-to-position map is only in the schematic figure of the I2MTC 2015 paper and the Helwig dissertation sec. 4.5, neither of which we could read as text; the dissertation PDF we obtained held only chapters 1-2). Pressure at the pump outlet is physically plausible for one PS sensor, but without a citation it is not placed. Excluded from the edge view, kept in the cloud view.
- The 2016 paper mentions a pump-housing temperature TS5; this is not in the released data.
- Edge = 2 channels (34 features, 6060 native samples = 24,240 bytes), cloud = 14 channels (238 features, 43,500 samples = 174,000 bytes). The choice was made from documentation only; no signal-to-label analysis was done.
- Cloud-only CE, CP, SE exclusion: they are virtual sensors computed "from physical sensor values" by a mathematical model. SE is described as hydraulic power over electrical power, CE as the oil temperature drop across the cooler relative to the maximum possible drop given the ambient temperature, CP as cooler heat-transfer rate. No source I read says which PS/FS/TS channels enter each, so I cannot state their parent sensors; the brief therefore excludes them. This also keeps cloud strictly physical and avoids derived quantities that already combine EPS1 with other channels.

## Label, windowing, groups
- One window = one 60 s load cycle (2205 windows, no sliding; stride = one cycle). Label 1 when pump leakage (profile column 3) is 1 (weak) or 2 (severe), 0 when 0. fault_type: normal / pump_leak_weak / pump_leak_severe. label_frac = label. condition = "" (the load cycle is constant). All five profile columns (cooler, valve, pump, accumulator, stable) are stored in meta, plus cycle (row index), run_id.
- Counts: 2205 windows, 984 positives (prevalence 0.4463): 1221 normal, 492 weak, 492 severe.
- group_id = maximal run of consecutive cycles with identical (cooler, valve, pump, accumulator); runs longer than 60 are split into ceil(len/60) contiguous near-equal chunks (each <= 60). 194 runs -> 203 groups (the runs of 200, 200 and 210 cycles became 4 chunks each: 50 or 52-53 cycles).
- Group sizes: 154 groups of 10, 36 singletons, 1 of 19, 12 of 50-53. No group mixes labels. Negative groups: 83 (0 positives). Positive groups: 120 (all-positive): 96 groups with 10 positives, 24 singletons with 1 positive; positives per positive group: min 1, median 10, max 10.
- order = cycle position within group; all groups are contiguous in time.

## Residual leakage between adjacent groups
Groups change by one component step at a time. Along the profile the pump setting stays fixed across several consecutive groups (the valve cycles 73 -> 80 -> 90 -> 100 in runs of 10 while pump stays 2, then 1, then 0), so 178 of 202 adjacent group pairs have the same label. Neighbouring groups therefore share pump condition, oil temperature history and thermal drift, and differ only in other components, so a grouped split that puts adjacent groups in train and test still gives time-neighbour leakage. The 36 singleton groups are the transition cycles where the rig was re-set and the stable flag is 1 (756 windows have stable = 1, spread over 59 groups, which includes the 36 singletons), so they carry transient dynamics. There is only one physical rig and one pump per setting, so no group is an independent unit in the strict sense: grouped CV removes within-run leakage but not rig-level or temporal confounding. Consider blocked-in-time splits as a sensitivity analysis.

## Resampling to L = 600 (raw_*.npy)
Time-aligned on a 0.1 s grid. 100 Hz channels: scipy.signal.decimate by 10 (order-8 Chebyshev-I anti-alias IIR, zero phase). 10 Hz: native. 1 Hz: np.interp onto t = 0, 0.1, ..., 59.9 s with sample k at t = k s (last value held beyond 59 s). Channel order: EPS1, VS1 | PS1-PS6, FS1, FS2, TS1-TS4. Features use native-rate data via multi_rate_features, edge columns first; raw_edge equals raw_cloud[:, :2].

## Imputation
Rule: forward fill along the cycle, then channel median. The release has no missing values; imputed share is 0.0 for every channel (channels.json). Arrays contain no NaN or inf.

## Caveats
- Sensor placement beyond EPS1 and VS1 could not be verified from accessible text (see above); a stakeholder with the Helwig dissertation sec. 4.5 or I2MTC schematic could enlarge the edge set, but that should be pre-registered, not data-driven.
- The 1 Hz VS1 and TS series are interpolated for the CNN arrays; features use native samples.
- Pump leakage is simulated with bypass orifices on a test rig, not a field fault; other components are varied at the same time (valve, cooler, accumulator), so they act as nuisance factors.
- No classifier was trained and no separability was inspected.
