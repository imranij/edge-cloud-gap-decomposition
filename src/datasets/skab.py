"""Build the SKAB (Skoltech Anomaly Benchmark) arrays for the edge-to-cloud escalation study.

Rebuilds everything in data/processed/skab/ from data/raw/skab/repo/ (a shallow clone of
https://github.com/waico/SKAB, commit recorded in sha256.txt / card.md).

    python3 src/datasets/skab.py

No classifier is trained and no class-separability statistic is computed here.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import window_features  # noqa: E402

RAW = ROOT / "data" / "raw" / "skab"
REPO = RAW / "repo"
OUT = ROOT / "data" / "processed" / "skab"

# ----- design constants (frozen from documentation, before any model is run) -----------
EDGE = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Temperature", "Voltage"]
CLOUD_ONLY = ["Pressure", "Thermocouple", "Volume Flow RateRMS"]
CLOUD = EDGE + CLOUD_ONLY
FS = 1.0                 # Hz, nominal (README: one row per second, timestamps to the second)
WIN, STRIDE = 60, 10     # seconds -> L = 60 samples per window, native rate (no resampling)
MAX_FILL_GAP = 5         # a gap between two samples > 5 s (>4 missing seconds) is a segment break
FREE_CHUNKS = 8          # anomaly-free recording split in 8 contiguous chunks (~1245 s each)
POS_FRAC = 0.5           # label 1 iff >= 50% of the window's seconds are anomalous
FAULT_TYPES = {
    ("valve1", None): "valve1_inlet_closure",
    ("valve2", None): "valve2_outlet_closure",
    ("other", 1): "fluid_leak_or_addition", ("other", 2): "fluid_leak_or_addition",
    ("other", 3): "fluid_leak_or_addition", ("other", 4): "fluid_leak_or_addition",
    ("other", 5): "rotor_imbalance", ("other", 6): "rotor_imbalance", ("other", 7): "rotor_imbalance",
    ("other", 8): "rotor_imbalance", ("other", 9): "rotor_imbalance",
    ("other", 10): "water_increase", ("other", 11): "water_increase",
    ("other", 12): "cavitation", ("other", 13): "cavitation",
    ("other", 14): "warm_water_supply",
}


def fault_type(subset: str, k: int) -> str:
    return FAULT_TYPES.get((subset, k)) or FAULT_TYPES[(subset, None)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_grid(df: pd.DataFrame):
    """Return list of segments; each is a DataFrame on a regular 1-s grid (ffill within the
    segment) with columns CLOUD + ['anomaly','changepoint','imputed'] and index = seconds from
    the first sample of the file. Segments break where consecutive samples are > MAX_FILL_GAP apart."""
    t = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().round().astype(int).to_numpy()
    assert (np.diff(t) > 0).all(), "timestamps must be strictly increasing"
    cut = np.where(np.diff(t) > MAX_FILL_GAP)[0] + 1
    bounds = np.split(np.arange(len(df)), cut)
    segs = []
    for idx in bounds:
        sub = df.iloc[idx].copy()
        sub.index = t[idx]
        grid = np.arange(t[idx][0], t[idx][-1] + 1)
        has = np.isin(grid, t[idx])
        cols = CLOUD + (["anomaly", "changepoint"] if "anomaly" in sub else [])
        g = sub[cols].reindex(grid)
        if "changepoint" in g:
            g["changepoint"] = g["changepoint"].fillna(0.0)   # never duplicate a changepoint
        g = g.ffill()                                         # documented imputation rule
        g["imputed"] = (~has).astype(np.int8)
        segs.append(g)
    return segs


def windows_from(seg: pd.DataFrame):
    n = len(seg)
    vals = seg[CLOUD].to_numpy(np.float64)
    an = seg["anomaly"].to_numpy(np.float64) if "anomaly" in seg else np.zeros(n)
    cp = seg["changepoint"].to_numpy() if "changepoint" in seg else np.zeros(n)
    imp = seg["imputed"].to_numpy()
    t0 = int(seg.index[0])
    for s in range(0, n - WIN + 1, STRIDE):
        yield dict(raw=vals[s:s + WIN].T, frac=an[s:s + WIN].mean(), cp=int(cp[s:s + WIN].sum() > 0),
                   imp=imp[s:s + WIN].mean(), t_start=t0 + s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = []
    for subset in ["anomaly-free", "valve1", "valve2", "other"]:
        for p in sorted((REPO / "data" / subset).glob("*.csv"), key=lambda q: (len(q.stem), q.stem)):
            files.append((subset, p))
    # ---- sha256 of raw files ----
    lines = [f"# SKAB raw files; source https://github.com/waico/SKAB commit {commit}"]
    for p in [REPO / "README.md", REPO / "data" / "README.md", REPO / "LICENSE", REPO / "docs" / "pictures" / "testbed.png"] + [p for _, p in files]:
        lines.append(f"{sha256(p)}  {p.relative_to(RAW)}")
    (OUT / "sha256.txt").write_text("\n".join(lines) + "\n")

    rows, raws = [], []
    stats = dict(grid_seconds=0, imputed_seconds=0, dropped_boundary=0, files_with_breaks=[], n_samples=0, breaks=0,
                 lost_seconds_in_breaks=0, tail_unused=0)
    for subset, p in files:
        df = pd.read_csv(p, sep=";", parse_dates=["datetime"])
        assert list(df.columns[1:9]) == CLOUD or set(CLOUD) <= set(df.columns)
        assert not df[CLOUD].isna().any().any()
        stem = p.stem
        stats["n_samples"] += len(df)
        segs = to_grid(df)
        if len(segs) > 1:
            stats["files_with_breaks"].append(f"{subset}/{stem}")
            stats["breaks"] += len(segs) - 1
            starts = [int(s.index[0]) for s in segs]; ends = [int(s.index[-1]) for s in segs]
            stats["lost_seconds_in_breaks"] += sum(starts[i + 1] - ends[i] - 1 for i in range(len(segs) - 1))
        session = df["datetime"].iloc[0].strftime("%Y-%m-%d")
        if subset == "anomaly-free":
            seg = pd.concat(segs)
            assert len(segs) == 1
            L = len(seg) // FREE_CHUNKS
            units = [(f"free_{k}", seg.iloc[k * L:(k + 1) * L], "normal", "anomaly-free", k) for k in range(FREE_CHUNKS)]
            stats["tail_unused"] += len(seg) - L * FREE_CHUNKS
            units = [(g, [u], ft, sb, k) for g, u, ft, sb, k in units]
        else:
            k = int(stem)
            units = [(f"{subset}_{k}", segs, fault_type(subset, k), subset, k)]
        for gid, segl, ftype, sb, k in units:
            order = 0
            for si, seg in enumerate(segl):
                stats["grid_seconds"] += len(seg)
                stats["imputed_seconds"] += int(seg["imputed"].sum())
                for w in windows_from(seg):
                    f = w["frac"]
                    if 0 < f < POS_FRAC:
                        stats["dropped_boundary"] += 1
                        continue
                    lab = int(f >= POS_FRAC)
                    rows.append(dict(group_id=gid, label=lab, label_frac=float(f),
                                     fault_type=ftype if lab else "normal", condition="", order=order,
                                     subset=sb, file=f"{subset}/{p.name}", exp_fault_type=ftype, session=session,
                                     segment=si, t_start_s=w["t_start"], imputed_frac=float(w["imp"]),
                                     changepoint_in_window=w["cp"]))
                    raws.append(w["raw"])
                    order += 1
    meta = pd.DataFrame(rows)
    meta.insert(0, "window_id", np.arange(len(meta)))
    R = np.stack(raws).astype(np.float32)                    # (n, 8, 60), edge channels first
    assert np.isfinite(R).all()
    Xc, names_c = window_features(R, CLOUD)
    Xe, names_e = window_features(R[:, :len(EDGE), :], EDGE)
    assert names_c[:len(names_e)] == names_e and np.array_equal(Xc[:, :Xe.shape[1]], Xe)
    assert np.isfinite(Xc).all() and np.isfinite(Xe).all()
    meta.to_parquet(OUT / "meta.parquet", index=False)
    np.save(OUT / "X_edge.npy", Xe.astype(np.float32)); np.save(OUT / "X_cloud.npy", Xc.astype(np.float32))
    np.save(OUT / "raw_edge.npy", R[:, :len(EDGE), :]); np.save(OUT / "raw_cloud.npy", R)
    (OUT / "names_edge.json").write_text(json.dumps(names_e)); (OUT / "names_cloud.json").write_text(json.dumps(names_c))
    node = "pump-motor drive unit: motor current, motor voltage, motor-body temperature, two vibration accelerometers"
    (OUT / "channels.json").write_text(json.dumps(dict(edge=EDGE, cloud=CLOUD, rates_hz={c: FS for c in CLOUD},
                                                       edge_node=node), indent=1))
    (OUT / "bytes.json").write_text(json.dumps(dict(
        raw_edge=len(EDGE) * WIN * 4, raw_cloud=len(CLOUD) * WIN * 4,
        features_edge=Xe.shape[1] * 4, features_cloud=Xc.shape[1] * 4,
        note="per window, float32, native 1 Hz, 60-s window"), indent=1))

    # ---- card ----
    g = meta.groupby("group_id", sort=False).agg(n=("label", "size"), pos=("label", "sum"), ft=("exp_fault_type", "first"))
    n, npos = len(meta), int(meta.label.sum())
    imp_share = stats["imputed_seconds"] / stats["grid_seconds"]
    by_sub = meta.groupby("subset").agg(windows=("label", "size"), positives=("label", "sum"), groups=("group_id", "nunique"))
    by_ft = meta[meta.label == 1].groupby("fault_type").agg(windows=("label", "size"), groups=("group_id", "nunique"))
    card = f"""# SKAB: Skoltech Anomaly Benchmark (processed for edge-to-cloud escalation)

## Source
- Canonical home: https://github.com/waico/SKAB (shallow clone over HTTPS), commit `{commit}` (last commit 2024-08-11), data version "SKAB v0.9".
- Licence: GPL-3.0 (repository LICENSE; GitHub badge in README). The Kaggle DOI 10.34740/KAGGLE/DSV/1693952 is the owners' citation DOI; the Kaggle copy was not used, and only the GitHub content was read.
- Raw files and SHA-256: `sha256.txt` (raw copies under `data/raw/skab/repo/`).
- Citation (as given in the repository README):
```bibtex
@misc{{skab,
  author = {{Katser, Iurii D. and Kozitsin, Vyacheslav O.}},
  title = {{Skoltech Anomaly Benchmark (SKAB)}},
  year = {{2020}},
  publisher = {{Kaggle}},
  howpublished = {{\\url{{https://www.kaggle.com/dsv/1693952}}}},
  DOI = {{10.34740/KAGGLE/DSV/1693952}}
}}
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
- Window label = 1 if at least {int(POS_FRAC*100)}% of the 60 seconds are labelled anomalous, 0 if none are; windows with 0 < fraction < {POS_FRAC} (boundary windows around onset and offset) are **dropped** ({stats['dropped_boundary']} windows) because they are neither clean normal nor clean fault. `label_frac` records the exact fraction.
- `fault_type` = the experiment's injected disturbance for label 1 (from data/README.md: valve1_inlet_closure, valve2_outlet_closure, fluid_leak_or_addition, rotor_imbalance, water_increase, cavitation, warm_water_supply) and "normal" for label 0; `exp_fault_type` keeps the experiment type for every window. `condition` is empty (no operating condition is documented).

## Windowing
60-s windows (L = 60 samples at 1 Hz, so `raw_*` arrays need no resampling), 10-s stride (83% overlap), never crossing a group boundary or a timing break. The README argues for a 60-s scale in its leaderboard (NAB detection window = 60 s "to reflect the dynamics of the transition"). Timestamps are on a 1-s grid but about 5% of seconds are absent (single missing seconds); the series is put on a regular 1-s grid and the missing seconds are filled by forward fill.
Timing breaks: where consecutive samples are more than {MAX_FILL_GAP} s apart the series is cut into segments and no window spans the cut. This happens in {len(stats['files_with_breaks'])} files ({', '.join(stats['files_with_breaks'])}), {stats['breaks']} breaks and {stats['lost_seconds_in_breaks']} unrecorded seconds in total (largest: 247 s in other/2, right at the anomaly onset). Windows never include an interval longer than {MAX_FILL_GAP-1} consecutive imputed seconds.

## Groups (independent unit)
`group_id` = one experiment file (`valve1_k`, `valve2_k`, `other_k`; k as the file name; valve1 files are numbered 0-15 on disk although data/README.md lists 1-16). The anomaly-free recording (9405 samples, 9961 s) has no label variation and is split into {FREE_CHUNKS} contiguous chunks of {9961 // FREE_CHUNKS} s (`free_0` ... `free_{FREE_CHUNKS-1}`), close to the median experiment length (about 1200 s), so that anomaly-free groups have comparable weight; {stats['tail_unused']} trailing second is unused. Windows of one file/chunk always share a group_id. `order` is time order within the group; `session` (recording date) is provided as an extra grouping option.

## Counts
- Windows: {n}; groups: {meta.group_id.nunique()} ({(g.pos > 0).sum()} with positives, {(g.pos == 0).sum()} anomaly-free); positives: {npos}; prevalence: {npos / n:.4f}.
- Per subset:
{by_sub.to_markdown()}
- Positive windows per fault type:
{by_ft.to_markdown()}
- Positives per group (min / median / max over groups with positives): {int(g.pos[g.pos > 0].min())} / {int(g.pos[g.pos > 0].median())} / {int(g.pos.max())}.
- Per-group table: see `meta.parquet` (group_by group_id).
- Raw samples read: {stats['n_samples']}; 1-s grid seconds used: {stats['grid_seconds']}.

## Imputation
Raw CSVs contain no missing values (0 NaN). The only imputation is the filling of absent seconds on the 1-s grid by forward fill within the segment: {stats['imputed_seconds']} of {stats['grid_seconds']} grid seconds ({imp_share:.2%}); because rows are missing as whole rows, the share is identical for all 8 channels. The fallback to the training-free channel median was never triggered (no leading gaps, since each segment starts at a real sample). Per-window imputed share is in `meta.imputed_frac` (mean {meta.imputed_frac.mean():.3f}, max {meta.imputed_frac.max():.2f}; {(meta.imputed_frac > 0.2).sum()} windows exceed 20%, concentrated in other/12, other/13 and valve2 which have denser dropouts). Imputed seconds inherit the previous label; changepoint flags are not copied to imputed seconds.

## Known caveats and leakage risks
1. **Time-position confound within each experiment.** Almost every labelled file is normal first (typically the first 560-580 s; exceptions other/2 at 104 s, other/13 at 495 s, other/4 at 796 s), then anomalous, then normal again (except other/1 and other/4, which end anomalous). Anything that drifts with run time (motor-body temperature warm-up, fluid temperature) is correlated with the label inside a group; grouped CV does not remove this.
2. **Normal windows come from different sessions than anomaly-free chunks.** The anomaly-free file is from 2020-02-08, `other/5-14` from 2020-02-08 (later the same day), `other/1-4` from 2020-03-01 and valve1/valve2 from 2020-03-09. Group-wise splits can place a session in both train and test; use `session` for a leave-session-out sensitivity analysis.
3. **Anomaly-free chunks are contiguous slices of one recording**; neighbouring chunks are autocorrelated, so groups are not fully independent.
4. **83% overlap between windows** means windows must never be split across folds (grouping handles this).
5. Experiment files follow a similar fixed script (typically about 570 s normal, then about 400 s of disturbance), so the label is partly predictable from `order`/`t_start_s`; these columns are metadata, not features.
6. Only 4 `valve2` and 14 `other` experiments, with 1-5 experiments per specific fault type, so per-fault-type conclusions are low-powered; `other/*` fault types differ strongly in kind.
7. The Kaggle landing page and the position paper (referred to in README as submitted) were not accessible; sensor placement relies on the repository README and photo only. The repository states the testbed is under repair (v0.9 is the current release; v1.0 is announced).
8. Sampling-rate jitter (about 5% missing seconds) means 60 samples is not exactly 60 s; the grid fixes this.
"""
    (OUT / "card.md").write_text(card)
    print(f"windows {n}  groups {meta.group_id.nunique()}  positives {npos}  prevalence {npos/n:.4f}")
    print("imputed share", f"{imp_share:.4f}", "dropped boundary", stats["dropped_boundary"])


if __name__ == "__main__":
    main()
