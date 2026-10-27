"""Build data/processed/threew from the Petrobras 3W Dataset 2.0.0 (real instances only).

Source: sparse, shallow git clone of https://github.com/petrobras/3W in data/raw/threew/repo
(dataset/<event>/WELL-*.parquet, dataset/dataset.ini, docs). No classifier is fitted here.

Rules (see card.md for the rationale):
  * only REAL instances (file name starts with WELL-); SIMULATED and DRAWN are excluded;
  * 1 Hz, 300-s windows, 300-s stride, aligned to the first timestamp of each instance;
    a trailing partial window is dropped;
  * timestamp fault = class not in {0, NA}; label_frac = fault / labelled timestamps;
    label = 1 if label_frac >= 0.5; windows with no labelled timestamp are dropped;
  * cloud view = analog variables (ESTADO-* valve states excluded) whose USABLE coverage
    (non-missing, in an instance where the variable is not frozen, i.e. not constant over the
    whole instance) is >= 60 % of real-instance timestamps AND >= 60 % as a per-instance mean;
    edge view = the subsea Xmas-tree TPT (P-TPT, T-TPT), fixed from the documentation;
  * values with |v| > 1e12 (historian sentinels) are treated as missing;
  * per-instance cap of 150 windows, label-stratified, seed 2026;
  * imputation: forward fill within the instance (causal), then the channel median of all
    observed values in the kept windows (a fixed, training-free constant).
"""
from __future__ import annotations

import configparser
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import window_features  # noqa: E402

NAME = "threew"
RAW = ROOT / "data" / "raw" / NAME
REPO = RAW / "repo"
OUT = ROOT / "data" / "processed" / NAME
WIN = 300            # samples = seconds at 1 Hz
STRIDE = 300
CAP = 150            # windows per instance
SEED = 2026
COV_THR = 0.60
SENTINEL = 1e12      # |value| above this is a historian sentinel (e.g. -1.180116e42 in P-PDG); -> NaN
EDGE = ["P-TPT", "T-TPT"]
# cloud ordering after the edge channels: downhole -> annulus -> topside
PLACEMENT = {
    "P-TPT": "subsea Xmas tree, TPT (temperature and pressure transducer)",
    "T-TPT": "subsea Xmas tree, TPT (temperature and pressure transducer)",
    "PT-P": "subsea Xmas tree, downstream of the production wing valve",
    "P-PDG": "downhole, PDG (permanent downhole gauge) on the production tubing",
    "T-PDG": "downhole, PDG (permanent downhole gauge) on the production tubing",
    "P-ANULAR": "well annulus (measurement point not specified in the documentation)",
    "P-MON-CKP": "topside, upstream of the production choke (PCK)",
    "P-JUS-CKP": "topside, downstream of the production choke (PCK)",
    "T-MON-CKP": "topside, upstream of the production choke (PCK)",
    "T-JUS-CKP": "topside, downstream of the production choke (PCK); may mix several wells",
    "ABER-CKP": "topside, production choke (PCK) opening [%]",
    "P-MON-CKGL": "topside, upstream of the gas-lift choke (GLCK)",
    "P-JUS-CKGL": "topside, downstream of the gas-lift choke (GLCK), gas-lift line",
    "ABER-CKGL": "topside, gas-lift choke (GLCK) opening [%]",
    "QGL": "topside, gas-lift line flow rate",
    "P-MON-SDV-P": "topside, upstream of the production shutdown valve (SDV)",
    "P-JUS-BS": "service pump (SP) discharge",
    "QBS": "service pump (SP) flow rate",
}
ORDER_RANK = ["P-TPT", "T-TPT", "PT-P", "P-PDG", "T-PDG", "P-ANULAR", "P-MON-CKP", "P-JUS-CKP",
              "T-MON-CKP", "T-JUS-CKP", "ABER-CKP", "P-MON-SDV-P", "P-MON-CKGL", "P-JUS-CKGL",
              "ABER-CKGL", "QGL", "P-JUS-BS", "QBS"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def commit_hash() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return (REPO / ".git" / "HEAD").read_text().strip()


def load_ini():
    cp = configparser.ConfigParser()
    cp.read(REPO / "dataset" / "dataset.ini")
    names = [s.strip() for s in cp["EVENTS"]["NAMES"].replace("\n", " ").split(",")]
    ev = {int(cp[n]["LABEL"]): n.lower() for n in names}
    offset = int(cp["EVENTS"]["TRANSIENT_OFFSET"])
    variables = {k.upper(): v for k, v in cp["PARQUET_FILE_PROPERTIES"].items()}
    return ev, offset, variables, cp["VERSION"]["DATASET"]


def stratified_cap(labels: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices (sorted) of at most `cap` windows, label proportions kept, >=1 per present label."""
    n = len(labels)
    if n <= cap:
        return np.arange(n)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    k_pos = int(round(cap * len(pos) / n))
    if len(pos) > 0:
        k_pos = max(k_pos, 1)
    if len(neg) > 0:
        k_pos = min(k_pos, cap - 1)
    k_pos = min(k_pos, len(pos))
    k_neg = min(cap - k_pos, len(neg))
    sel = np.concatenate([rng.choice(pos, k_pos, replace=False) if k_pos else np.array([], int),
                          rng.choice(neg, k_neg, replace=False) if k_neg else np.array([], int)])
    return np.sort(sel.astype(int))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev_names, toff, var_doc, version = load_ini()
    files = sorted((REPO / "dataset").glob("[0-9]/WELL-*.parquet"))
    # the sparse checkout holds only WELL-* files; count the excluded sources from the git tree
    tree = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "HEAD", "dataset"],
                          capture_output=True, text=True, check=True).stdout.split()
    tree = [t for t in tree if t.endswith(".parquet") and t.split("/")[1].isdigit()]
    n_excl = {k: sum(Path(t).name.startswith(k) for t in tree) for k in ("SIMULATED", "DRAWN")}
    assert sum(Path(t).name.startswith("WELL-") for t in tree) == len(files)
    assert len(files) > 0

    # ---------- pass 1: missingness / coverage of every variable over real instances
    probe = pd.read_parquet(files[0])
    variables = [c for c in probe.columns if c not in ("class", "state")]
    analog = [v for v in variables if not v.startswith("ESTADO-")]
    present_ts = dict.fromkeys(variables, 0)
    usable_ts = dict.fromkeys(variables, 0)       # present AND not frozen over the whole instance
    inst_cov = {v: [] for v in variables}
    inst_use = {v: [] for v in variables}
    inst_frozen = dict.fromkeys(variables, 0)
    total_ts = 0
    inst_rows = []
    n_sentinel = dict.fromkeys(variables, 0)
    for f in files:
        df = pd.read_parquet(f)
        for v in variables:
            bad = df[v].abs() > SENTINEL
            if bad.any():
                n_sentinel[v] += int(bad.sum())
                df.loc[bad, v] = np.nan
        assert list(df.columns[:len(variables)]) == variables
        dt = np.diff(df.index.values.astype("datetime64[s]").astype(np.int64))
        assert (dt == 1).all(), f"non-1Hz sampling in {f.name}"
        total_ts += len(df)
        for v in variables:
            p = df[v].notna().to_numpy()
            present_ts[v] += int(p.sum())
            inst_cov[v].append(p.mean())
            frozen = bool(p.any() and np.nanstd(df[v].to_numpy(dtype=float)) == 0)
            inst_frozen[v] += frozen
            usable_ts[v] += 0 if frozen else int(p.sum())
            inst_use[v].append(0.0 if frozen else p.mean())
        cl = df["class"]
        inst_rows.append(dict(file=f.name, event=int(f.parent.name), well=f.name.split("_")[0],
                              n_ts=len(df), n_unlabelled=int(cl.isna().sum()),
                              n_fault_ts=int((cl.fillna(0) != 0).sum()),
                              t_start=df.index[0], t_end=df.index[-1],
                              t_lab_start=df.index[cl.notna().to_numpy()][0]))
    n_inst = len(files)
    miss = pd.DataFrame({
        "variable": variables,
        "kind": ["valve_state" if v.startswith("ESTADO-") else "analog" for v in variables],
        "placement": [PLACEMENT.get(v, "valve state (Xmas tree / topside / downhole valve)") for v in variables],
        "description": [var_doc.get(v, "") for v in variables],
        "present_frac_timestamps": [present_ts[v] / total_ts for v in variables],
        "present_frac_mean_per_instance": [float(np.mean(inst_cov[v])) for v in variables],
        "instances_all_missing_frac": [float(np.mean(np.array(inst_cov[v]) == 0)) for v in variables],
        "instances_frozen_frac": [inst_frozen[v] / n_inst for v in variables],
        "sentinel_values_set_missing": [n_sentinel[v] for v in variables],
        "usable_frac_timestamps": [usable_ts[v] / total_ts for v in variables],
        "usable_frac_mean_per_instance": [float(np.mean(inst_use[v])) for v in variables],
    })
    miss["selected_cloud"] = (miss.kind == "analog") & (miss.usable_frac_timestamps >= COV_THR) & \
        (miss.usable_frac_mean_per_instance >= COV_THR)
    miss = miss.sort_values("usable_frac_timestamps", ascending=False).reset_index(drop=True)
    miss.to_csv(OUT / "missingness.csv", index=False, float_format="%.4f")
    sel = set(miss.loc[miss.selected_cloud, "variable"])
    assert set(EDGE) <= sel, "edge channels must pass the coverage rule"
    CLOUD = EDGE + [v for v in ORDER_RANK if v in sel and v not in EDGE]
    assert set(CLOUD) == sel
    inst = pd.DataFrame(inst_rows)
    inst["event_name"] = inst.event.map(ev_names)
    # episodes: chains of time-overlapping or contiguous (<= 1 s gap) instances of the same well
    inst = inst.sort_values(["well", "t_start"]).reset_index(drop=True)
    ep, cur, last_end, last_well, lab_ov = [], -1, None, None, 0
    for r in inst.itertuples():
        if r.well != last_well or r.t_start > last_end + pd.Timedelta(seconds=1):
            cur += 1
            last_end = r.t_end
        else:
            lab_ov = max(lab_ov, int((last_end - r.t_lab_start).total_seconds()) + 1)
            last_end = max(last_end, r.t_end)
        last_well = r.well
        ep.append(f"{r.well}_ep{cur:04d}")
    inst["episode_id"] = ep
    episode_of = dict(zip(inst.file.str.replace(".parquet", "", regex=False), inst.episode_id))

    # ---------- pass 2: windows
    rng = np.random.default_rng(SEED)
    metas, raws, nanmask, ffmask = [], [], [], []
    n_lab_windows_total = 0
    for gi, f in enumerate(files):
        df = pd.read_parquet(f, columns=CLOUD + ["class", "state"])
        df[CLOUD] = df[CLOUD].mask(df[CLOUD].abs() > SENTINEL)
        orig_nan = df[CLOUD].isna().to_numpy()
        vals = df[CLOUD].ffill().to_numpy(dtype=np.float64)          # causal, within instance
        cls = df["class"].to_numpy(dtype="float64", na_value=np.nan)
        st = df["state"].to_numpy(dtype="float64", na_value=np.nan)
        nw = len(df) // WIN
        if nw == 0:
            continue
        L = nw * WIN
        cw = cls[:L].reshape(nw, WIN)
        lab = ~np.isnan(cw)
        n_lab = lab.sum(1)
        keep = np.flatnonzero(n_lab > 0)
        n_lab_windows_total += len(keep)
        fault = lab & (np.nan_to_num(cw) != 0)
        frac = fault.sum(1) / np.maximum(n_lab, 1)
        label = (frac >= 0.5).astype(int)
        idx = keep[stratified_cap(label[keep], CAP, rng)]
        W = vals[:L].reshape(nw, WIN, len(CLOUD)).transpose(0, 2, 1)[idx]
        M0 = orig_nan[:L].reshape(nw, WIN, len(CLOUD)).transpose(0, 2, 1)[idx]
        raws.append(W.astype(np.float32))
        nanmask.append(M0)
        ffmask.append(M0 & ~np.isnan(W))
        sw = st[:L].reshape(nw, WIN)[idx]
        ts = df.index[:L][idx * WIN]
        for j, w in enumerate(idx):
            c = cw[w][lab[w]].astype(int)
            nz = c[c != 0]
            if label[w] == 1:
                maj = int(pd.Series(nz).mode().iloc[0])
                ftype = ev_names[maj % toff]
            else:
                maj, ftype = 0, "normal"
            s = sw[j][~np.isnan(sw[j])]
            metas.append(dict(
                group_id=f.stem, label=int(label[w]), label_frac=float(frac[w]), fault_type=ftype,
                condition="", order=int(w), well=f.name.split("_")[0], episode_id=episode_of[f.stem],
                instance_event=int(f.parent.name),
                instance_event_name=ev_names[int(f.parent.name)], majority_fault_class=maj,
                transient_frac_of_fault=float((nz >= toff).mean()) if len(nz) else 0.0,
                n_labelled=int(n_lab[w]), state_mode=int(pd.Series(s).mode().iloc[0]) if len(s) else -1,
                t_start=str(ts[j]),
            ))
    raw = np.concatenate(raws)
    nanm = np.concatenate(nanmask)
    ffm = np.concatenate(ffmask)
    del raws, nanmask, ffmask
    meta = pd.DataFrame(metas)
    meta.insert(0, "window_id", np.arange(len(meta)))

    # ---------- imputation: remaining NaN -> channel median of observed values in kept windows
    med = {}
    imp = {}
    for c, ch in enumerate(CLOUD):
        obs = raw[:, c, :][~nanm[:, c, :]]
        med[ch] = float(np.median(obs)) if obs.size else 0.0
        still = np.isnan(raw[:, c, :])
        raw[:, c, :][still] = med[ch]
        imp[ch] = dict(missing_frac=float(nanm[:, c, :].mean()), ffill_frac=float(ffm[:, c, :].mean()),
                       median_frac=float(still.mean()), median_value=med[ch])
    assert np.isfinite(raw).all()
    raw = raw.astype(np.float32)

    # ---------- features (shared code), nested views
    ne = len(EDGE)
    Xc, names_c = window_features(raw, CLOUD)
    Xe, names_e = window_features(raw[:, :ne, :], EDGE)
    assert names_c[:len(names_e)] == names_e and np.array_equal(Xc[:, :Xe.shape[1]], Xe)
    Xc, Xe = Xc.astype(np.float32), Xe.astype(np.float32)
    assert np.isfinite(Xc).all() and np.isfinite(Xe).all()

    np.save(OUT / "X_edge.npy", Xe)
    np.save(OUT / "X_cloud.npy", Xc)
    np.save(OUT / "raw_edge.npy", np.ascontiguousarray(raw[:, :ne, :]))
    np.save(OUT / "raw_cloud.npy", raw)
    meta.to_parquet(OUT / "meta.parquet", index=False)
    json.dump(names_e, open(OUT / "names_edge.json", "w"))
    json.dump(names_c, open(OUT / "names_cloud.json", "w"))
    json.dump(dict(edge=EDGE, cloud=CLOUD, rates_hz={c: 1.0 for c in CLOUD},
                   edge_node="subsea wet Christmas tree (ANM): TPT temperature and pressure transducer",
                   placement={c: PLACEMENT[c] for c in CLOUD},
                   excluded=[v for v in variables if v not in CLOUD],
                   raw_length=WIN, resampling="none (native 1 Hz, 300 samples per window)",
                   imputation=imp),
              open(OUT / "channels.json", "w"), indent=1)
    json.dump(dict(raw_edge=ne * WIN * 4, raw_cloud=len(CLOUD) * WIN * 4,
                   features_edge=Xe.shape[1] * 4, features_cloud=Xc.shape[1] * 4,
                   note="float32 payload per 300-s window; raw at native 1 Hz"),
              open(OUT / "bytes.json", "w"), indent=1)

    # ---------- provenance
    raw_files = [REPO / "dataset" / "dataset.ini", REPO / "dataset" / "README.md",
                 REPO / "dataset" / "LICENSE-CC-BY", REPO / "LICENSE.md", REPO / "CITATION.md",
                 REPO / "3W_DATASET_STRUCTURE.md", REPO / "docs" / "doctoral_thesis_ricardo_vargas.pdf"] + files
    lines = [f"{sha256(p)}  {p.relative_to(RAW)}" for p in raw_files if p.exists()]
    (OUT / "sha256.txt").write_text("\n".join(lines) + "\n")
    (RAW / "sha256.txt").write_text("\n".join(lines) + "\n")

    # ---------- descriptive summaries for the card
    inst.to_csv(OUT / "instances.csv", index=False)
    ov = 0
    for w, g in inst.sort_values("t_start").groupby("well"):
        s, e = g.t_start.to_numpy(), g.t_end.to_numpy()
        ov += int((s[1:] <= e[:-1]).sum())
    max_lab_overlap_s = max(lab_ov, 0)
    summ = dict(commit=commit_hash(), dataset_version=version, n_real_instances=n_inst,
                n_excluded=n_excl, total_real_timestamps=int(total_ts),
                labelled_windows_before_cap=int(n_lab_windows_total), n_windows=len(meta),
                n_groups=int(meta.group_id.nunique()), n_wells=int(meta.well.nunique()),
                positives=int(meta.label.sum()), prevalence=float(meta.label.mean()),
                overlapping_instance_pairs_same_well=ov,
                max_overlap_into_labelled_part_s=max_lab_overlap_s,
                n_episodes=int(inst.episode_id.nunique()), channel_medians=med,
                sentinel_values_set_missing={v: k for v, k in n_sentinel.items() if k})
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
