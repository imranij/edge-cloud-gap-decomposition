"""Build the SEU gearbox (DDS) dataset for the escalation study.
Rebuilds everything in data/processed/seu from data/raw/seu/repo/gearbox.
No model is trained and no class-separability statistic is computed here."""
import sys, json, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import window_features, multi_rate_features  # noqa: E402

RAW = ROOT / "data/raw/seu/repo/gearbox"
OUT = ROOT / "data/processed/seu"
L, STRIDE, CAP = 1024, 1024, 400
TRIM = 256   # leading windows dropped from EVERY file (label-blind): slow DC settling of the motor channel, see card.md
FS = 5120.0   # inferred, see card.md (2.56 x 2000 Hz frequency limit in the DAQ header)
CH = ["motor_vib", "planet_x", "planet_y", "planet_z", "motor_torque", "par_x", "par_y", "par_z"]
EDGE = ["motor_vib", "motor_torque"]
CLOUD = EDGE + [c for c in CH if c not in EDGE]       # edge first
FAULT_TYPES = {"chipped", "miss", "root", "surface", "ball", "inner", "outer", "comb"}


def read_file(p):
    lines = p.read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip(" \t,") == "Data") + 1
    sep = "," if "," in lines[start] else "\t"
    rows = [l.rstrip("\t,").split(sep) for l in lines[start:] if l.strip(" \t,")]
    a = np.array([[float(v) if v.strip() not in ("", "nan", "NaN") else np.nan for v in r] for r in rows])
    assert a.shape[1] == 8, (p, a.shape)
    return a


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(list((RAW / "gearset").glob("*.csv")) + list((RAW / "bearingset").glob("*.csv")),
                   key=lambda p: (p.parent.name, p.name))
    assert len(files) == 20
    with open(OUT / "sha256.txt", "w") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(RAW.parent)}\n")
    metas, raws = [], []
    imp_n = np.zeros(8); imp_tot = np.zeros(8)
    wid = 0
    for p in files:
        sub = p.parent.name.replace("set", "")          # gear | bearing
        m = re.match(r"([A-Za-z]+)_(\d+)_(\d+)\.csv", p.name)
        ft = m.group(1).lower(); cond = f"{m.group(2)}-{m.group(3)}"
        ft = "normal" if ft == "health" else ft
        assert ft == "normal" or ft in FAULT_TYPES, ft
        a = read_file(p)
        bad = ~np.isfinite(a)
        imp_n += bad.sum(0); imp_tot += len(a)
        if bad.any():                                   # ffill within file, then channel median
            df = pd.DataFrame(a).replace([np.inf, -np.inf], np.nan).ffill()
            a = df.fillna(df.median()).to_numpy()
        nw = len(a) // L
        # fixed rule: drop first TRIM windows of every file, then CAP windows evenly spaced among the
        # remaining non-overlapping windows (stride = L)
        idx = np.unique(np.round(np.linspace(TRIM, nw - 1, min(CAP, nw - TRIM))).astype(int))
        W = np.stack([a[i * STRIDE:i * STRIDE + L].T for i in idx]).astype(np.float32)   # (n, 8, L)
        raws.append(W)
        gid = f"{sub}/{p.stem}"
        for o, i in enumerate(idx):
            metas.append(dict(window_id=wid, group_id=gid, label=int(ft != "normal"),
                              label_frac=float(ft != "normal"), fault_type=ft, condition=cond,
                              order=o, subset=sub, source_file=f"{sub}set/{p.name}",
                              start_sample=int(i * STRIDE), window_index_in_file=int(i)))
            wid += 1
    meta = pd.DataFrame(metas)
    meta.to_parquet(OUT / "meta.parquet", index=False)
    W = np.concatenate(raws)
    ci = [CH.index(c) for c in CLOUD]
    Wc = np.ascontiguousarray(W[:, ci, :])
    We = np.ascontiguousarray(Wc[:, :len(EDGE), :])
    Xe, ne = multi_rate_features([(We, EDGE)])
    Xc, nc = multi_rate_features([(Wc, CLOUD)])
    assert ne == nc[:len(ne)] and np.array_equal(Xe, Xc[:, :Xe.shape[1]])
    for n_, a_ in [("X_edge", Xe), ("X_cloud", Xc), ("raw_edge", We), ("raw_cloud", Wc)]:
        np.save(OUT / f"{n_}.npy", a_.astype(np.float32))
    json.dump(ne, open(OUT / "names_edge.json", "w")); json.dump(nc, open(OUT / "names_cloud.json", "w"))
    json.dump({"edge": EDGE, "cloud": CLOUD, "rates_hz": {c: FS for c in CLOUD},
               "rate_note": "inferred 5120 Hz (2.56 x 2000 Hz DAQ frequency limit); not stated in repo README",
               "edge_node": "motor (motor vibration + motor torque)", "L": L, "stride": STRIDE},
              open(OUT / "channels.json", "w"), indent=1)
    nfe = Xe.shape[1] // 1; nfc = Xc.shape[1]
    json.dump({"raw_edge": len(EDGE) * L * 4, "raw_cloud": len(CLOUD) * L * 4,
               "features_edge": nfe * 4, "features_cloud": nfc * 4,
               "note": "native rate == L samples per channel per window; float32"},
              open(OUT / "bytes.json", "w"), indent=1)
    # descriptive counts
    g = meta.groupby("group_id").agg(n=("label", "size"), pos=("label", "sum"))
    cnt = dict(windows=len(meta), groups=len(g), positives=int(meta.label.sum()),
               prevalence=float(meta.label.mean()), imputed_share=dict(zip(CH, (imp_n / imp_tot).tolist())),
               by_fault_type=meta.fault_type.value_counts().to_dict(),
               by_subset_label=meta.groupby("subset").label.agg(["size", "sum"]).to_dict("index"),
               groups_by_label=meta.groupby("group_id").label.first().value_counts().to_dict())
    json.dump(cnt, open(OUT / "counts.json", "w"), indent=1, default=int)
    print(json.dumps(cnt, indent=1, default=int))


if __name__ == "__main__":
    main()
