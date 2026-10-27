# Addendum A: edge nodes and confirmatory units (recorded 2026-10-12, before any model run)

Taken from each dataset card; chosen from documentation only (no model was fitted).

| Dataset | Edge node | Edge sensors | Cloud sensors | Independent units (group_id) | Windows | Prevalence |
|---|---|---|---|---|---|---|
| hydraulic | Motor-pump drive (main pump MP1) | EPS1 (motor power, 100 Hz), VS1 (vibration, 1 Hz) | 14 physical channels: EPS1, VS1, PS1-PS6, FS1-FS2, TS1-TS4; virtual CE, CP, SE excluded | 203 runs of constant component settings, split into chunks of at most 60 cycles | 2,205 cycles | 0.446 (internal pump leakage) |
| skab | Pump-motor drive unit | Accelerometer1RMS, Accelerometer2RMS, Current, Temperature (motor body), Voltage | The five edge sensors plus Pressure, Thermocouple, Volume Flow RateRMS | 42: 34 experiment files and 8 contiguous chunks of the anomaly-free file | 4,533 | 0.302 |
| seu | Motor | motor vibration, motor torque | All 8 channels (adds planetary and parallel gearbox x, y, z) | 20 recording files (4 healthy) | 8,000 | 0.800 |
| threew | Subsea Christmas-tree transducer (TPT) | P-TPT, T-TPT | P-TPT, T-TPT, P-ANULAR, P-MON-CKP, P-JUS-CKGL | 94 episodes (back-to-back instances of one well), complete-case subset of 445 real instances from 13 wells | 25,488 | 0.362 |
