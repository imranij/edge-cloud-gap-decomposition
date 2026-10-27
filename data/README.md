# Data

No raw sensor data and no feature arrays are kept in this repository. The four public datasets are
downloaded from their original sources and placed under `data/raw/`, and the scripts in `src/datasets/`
build the processed arrays in `data/processed/`. What is kept is the documentation of each build, namely the
dataset card (`card.md`), the channel lists of the edge and cloud views (`channels.json`, `names_edge.json`,
`names_cloud.json`), the payload sizes (`bytes.json`) and the SHA-256 sum of every raw file that was read
(`sha256.txt`). The cards give the reasoning behind each edge view and list the known confounds of each
dataset. `data/processed_sens.zip` holds the same documentation for the sensitivity variants, which
`src/sens/*/build.py` derive from the processed arrays.

## Processed-data schema

Each dataset folder in `data/processed/<dataset>/` holds `X_edge.npy` and `X_cloud.npy` (one row per window,
float64 features, the edge columns first in the cloud matrix), `meta.parquet` (one row per window, with at
least `window_id`, `group_id`, `label`, `label_frac`, `fault_type`, `condition` and `order`), the raw windows
`raw_edge.npy` and `raw_cloud.npy` used only by the convolutional pair C, and the files listed above.
`src/escalation/data.py` loads this layout, and `tests/synthetic.py` builds a small synthetic copy of it for
the tests.

## UCI hydraulic test rig (`hydraulic`)

Condition monitoring of hydraulic systems, by N. Helwig, ZeMA gGmbH, UCI Machine Learning Repository dataset
447, <https://doi.org/10.24432/C5CW21>, also on Zenodo at <https://doi.org/10.5281/zenodo.1323611>,
under the CC BY 4.0 licence. The 17 sensor files and the three text files of the release go directly in
`data/raw/hydraulic/`, and their SHA-256 sums are in `data/processed/hydraulic/sha256.txt`. Build and check with

```
python src/datasets/hydraulic.py
python src/datasets/check_hydraulic.py
```

## Skoltech Anomaly Benchmark (`skab`)

<https://github.com/waico/SKAB>, commit `b2c0d46c2971dcbfe71e26087b6d231998bb91c2`, data version 0.9,
licence GPL-3.0. Clone the repository into `data/raw/skab/repo/` and check out that commit. The 40 files that
are read are listed with their sums in `data/processed/skab/sha256.txt`. Build and check with
`src/datasets/skab.py` and `src/datasets/check_skab.py`.

## Southeast University gearbox (`seu`)

<https://github.com/cathysiyu/Mechanical-datasets>, commit `9732a2820f4cdf91fa7d1d0cc95442376660f5dd`,
folder `gearbox/` only. The repository states no licence, so its files are used for research with citation
and are not copied here. Clone it so that the folder sits at `data/raw/seu/repo/gearbox/`. The 20 CSV files
and their sums are in `data/processed/seu/sha256.txt`. Build and check with `src/datasets/seu.py` and
`src/datasets/check_seu.py`.

## Petrobras 3W, real instances (`threew`)

<https://github.com/petrobras/3W>, commit `d57c058af6baf65f1b285f3e56f46e4f018a07d9`, 3W Dataset version
2.0.0. The data files are under CC BY 4.0. Clone the repository into `data/raw/threew/repo/`; only the 1,119
real instances are read, and their sums are in `data/processed/threew/sha256.txt`. `src/datasets/threew.py`
builds the full set in `data/processed/threew/`, and `src/datasets/threew_subset.py` then moves it to
`data/processed/threew_all/` and writes the confirmatory subset in `data/processed/threew/` (see `plan/DEVIATIONS.md`, 11 October 2026). The two
check scripts are `src/datasets/check_threew.py` and `src/datasets/check_threew_subset.py`.

## Citations

Please cite the original sources when using these datasets. The BibTeX entries are in the dataset cards.
