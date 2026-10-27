"""Rebuild data/processed/hydraulic from data/raw/hydraulic (UCI/Zenodo release of the ZeMA
hydraulic test-rig data, Helwig et al. 2015). No model is trained here.

Task: internal pump leakage (label 1 if profile column 3 in {1,2}).
Window: one 60-s load cycle (one row of every sensor file).
Edge node: motor-pump drive = EPS1 (motor power, 100 Hz) + VS1 (vibration at pump MP1, 1 Hz).
Cloud view: edge + PS1-PS6, FS1-FS2, TS1-TS4 (14 physical sensors). CE, CP, SE excluded.
"""
import sys, json, hashlib, pathlib
import numpy as np, pandas as pd
from scipy.signal import decimate

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import multi_rate_features  # noqa: E402

RAW = ROOT / "data/raw/hydraulic"
OUT = ROOT / "data/processed/hydraulic"
OUT.mkdir(parents=True, exist_ok=True)

L = 600
MAXCHUNK = 60
EDGE = ["EPS1", "VS1"]
CLOUD_EXTRA = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4"]
CLOUD = EDGE + CLOUD_EXTRA
EXCLUDED = ["CE", "CP", "SE"]
RATE = {"EPS1": 100, "PS1": 100, "PS2": 100, "PS3": 100, "PS4": 100, "PS5": 100, "PS6": 100,
        "FS1": 10, "FS2": 10, "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1, "VS1": 1,
        "CE": 1, "CP": 1, "SE": 1}
NATIVE_LEN = {100: 6000, 10: 600, 1: 60}


def load(ch):
    a = pd.read_csv(RAW / f"{ch}.txt", sep="\t", header=None, dtype=np.float64).to_numpy()
    assert a.shape == (2205, NATIVE_LEN[RATE[ch]]), (ch, a.shape)
    return a


def impute(a):
    """Forward fill along the cycle, then channel median (training-free). Returns array, share."""
    m = ~np.isfinite(a)
    share = float(m.mean())
    if share > 0:
        df = pd.DataFrame(np.where(m, np.nan, a).T).ffill().T
        a = df.to_numpy()
        med = np.nanmedian(np.where(np.isfinite(a), a, np.nan))
        a = np.where(np.isfinite(a), a, med)
    return a, share


def resample(a, rate):
    if rate == 100:
        return decimate(a, 10, axis=-1, zero_phase=True)      # IIR Chebyshev-I order 8, zero phase
    if rate == 10:
        return a
    t_src = np.arange(60, dtype=float)                           # seconds, sample k at t = k s
    t_dst = np.arange(L) * 0.1                                   # 10 Hz grid, 0 .. 59.9 s
    return np.stack([np.interp(t_dst, t_src, r) for r in a])    # constant hold after t = 59 s


def main():
    # raw hashes
    files = sorted(p for p in RAW.iterdir() if p.is_file())
    with open(OUT / "sha256.txt", "w") as f:
        for p in files:
            f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")

    prof = np.loadtxt(RAW / "profile.txt", dtype=int)
    assert prof.shape == (2205, 5)
    n = len(prof)

    # groups: maximal runs of identical (cooler, valve, pump, accumulator); runs > 60 are
    # split into ceil(len/60) contiguous, near-equal chunks (each <= 60).
    key = [tuple(r[:4]) for r in prof]
    run_id = np.zeros(n, int)
    for i in range(1, n):
        run_id[i] = run_id[i - 1] + (key[i] != key[i - 1])
    gid, order = np.empty(n, object), np.zeros(n, int)
    for r in np.unique(run_id):
        idx = np.where(run_id == r)[0]
        k = int(np.ceil(len(idx) / MAXCHUNK))
        for c, part in enumerate(np.array_split(idx, k)):
            gid[part] = f"r{r:03d}c{c}"
            order[part] = np.arange(len(part))

    pump = prof[:, 2]
    label = (pump >= 1).astype(int)
    ftype = np.array(["normal", "pump_leak_weak", "pump_leak_severe"])[pump]
    meta = pd.DataFrame({
        "window_id": np.arange(n), "group_id": gid, "label": label,
        "label_frac": label.astype(float), "fault_type": ftype, "condition": "",
        "order": order, "cycle": np.arange(n), "run_id": run_id,
        "cooler": prof[:, 0], "valve": prof[:, 1], "pump": prof[:, 2],
        "accumulator": prof[:, 3], "stable": prof[:, 4]})
    meta.to_parquet(OUT / "meta.parquet", index=False)

    data, imp = {}, {}
    for ch in CLOUD:
        a, share = impute(load(ch))
        data[ch], imp[ch] = a, share

    # features at native rates; one block per channel group, edge blocks first
    blocks = [["EPS1"], ["VS1"], [f"PS{i}" for i in range(1, 7)], ["FS1", "FS2"],
              [f"TS{i}" for i in range(1, 5)]]
    groups_cloud = [(np.stack([data[c] for c in chs], axis=1), chs) for chs in blocks]
    Xc, nc = multi_rate_features(groups_cloud)
    Xe, ne = multi_rate_features(groups_cloud[:2])
    assert nc[:len(ne)] == ne and np.array_equal(Xc[:, :len(ne)], Xe)
    np.save(OUT / "X_edge.npy", Xe.astype(np.float32))
    np.save(OUT / "X_cloud.npy", Xc.astype(np.float32))
    json.dump(ne, open(OUT / "names_edge.json", "w"))
    json.dump(nc, open(OUT / "names_cloud.json", "w"))

    raw_cloud = np.stack([resample(data[c], RATE[c]) for c in CLOUD], axis=1).astype(np.float32)
    assert raw_cloud.shape == (n, len(CLOUD), L) and np.isfinite(raw_cloud).all()
    np.save(OUT / "raw_cloud.npy", raw_cloud)
    np.save(OUT / "raw_edge.npy", raw_cloud[:, :len(EDGE), :].copy())

    nat = lambda chs: int(sum(NATIVE_LEN[RATE[c]] for c in chs) * 4)
    json.dump({"raw_edge": nat(EDGE), "raw_cloud": nat(CLOUD),
               "features_edge": int(Xe.shape[1] * 4), "features_cloud": int(Xc.shape[1] * 4)},
              open(OUT / "bytes.json", "w"), indent=1)
    json.dump({"edge": EDGE, "cloud": CLOUD, "excluded": EXCLUDED, "rates_hz": {c: RATE[c] for c in CLOUD},
               "raw_L": L, "imputed_share": imp,
               "edge_node": "motor-pump drive: motor power EPS1 and pump vibration VS1"},
              open(OUT / "channels.json", "w"), indent=1)

    g = meta.groupby("group_id")
    pos = g["label"].sum()
    print(f"windows={n} groups={g.ngroups} positives={int(label.sum())} prevalence={label.mean():.4f}")
    print("max imputed share:", max(imp.values()))
    print("groups with 0 positives:", int((pos == 0).sum()), " all-positive:", int((pos == g.size()).sum()),
          " mixed:", int(((pos > 0) & (pos < g.size())).sum()))


if __name__ == "__main__":
    main()
