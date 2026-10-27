"""POST-PLAN sensitivity variants for hydraulic and SKAB.

    python src/sens/hydskab/build.py [variant ...]      (default: all variants)

Builds data/processed_sens/<variant>/ with the processed schema (meta.parquet, X_edge.npy, X_cloud.npy,
names_*.json, bytes.json, channels.json, card.md). Features are copied from data/processed/<parent>
(no feature is recomputed); only rows, columns, group_id/order, or a per-group affine normalisation
change. raw_*.npy are not written (pairs A and B do not use them). Nothing here is confirmatory.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "processed"
DST = ROOT / "data" / "processed_sens"
SKAB_EDGE = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Temperature", "Voltage"]
SKAB_LOOP = ["Pressure", "Thermocouple", "Volume Flow RateRMS"]
HEADER = ("# Dataset variant `{name}` (POST-PLAN sensitivity analysis; not confirmatory)\n\n"
          "Parent: `data/processed/{parent}` (see its card.md for source, licence, channels, labels and windowing). "
          "Built by `src/sens/hydskab/build.py`. Pairs A and B only; raw_*.npy are not provided.\n\n")


def load_parent(parent):
    d = SRC / parent
    meta = pd.read_parquet(d / "meta.parquet")
    Xe = np.load(d / "X_edge.npy"); Xc = np.load(d / "X_cloud.npy")
    ne = json.loads((d / "names_edge.json").read_text()); nc = json.loads((d / "names_cloud.json").read_text())
    ch = json.loads((d / "channels.json").read_text()); by = json.loads((d / "bytes.json").read_text())
    assert np.array_equal(Xe, Xc[:, :len(ne)]) and nc[:len(ne)] == ne
    return meta, Xe, Xc, ne, nc, ch, by


def counts_md(meta):
    y = meta["label"].to_numpy(); g = meta["group_id"].astype(str).to_numpy()
    gp = len(np.unique(g[y == 1])); gn = len(np.unique(g[y == 0]))
    per = meta.groupby("group_id").agg(windows=("label", "size"), positives=("label", "sum"))
    mixed = int(((per.positives > 0) & (per.positives < per.windows)).sum())
    K = min(5, gp, gn)
    s = (f"## Counts\n- Windows: {len(meta)}; positives: {int(y.sum())}; prevalence: {y.mean():.4f}.\n"
         f"- Groups: {meta.group_id.nunique()}; groups containing positives: {gp}; groups containing negatives: {gn}; "
         f"groups mixing both labels: {mixed}.\n"
         f"- Resulting outer K = min(5, {gp}, {gn}) = {K}; repeats for J = 20: {math.ceil(20 / K)}.\n"
         f"- Group sizes (windows): min {per.windows.min()}, median {per.windows.median():.0f}, max {per.windows.max()}; "
         f"positives per positive group: min {per.positives[per.positives > 0].min() if gp else 0}, "
         f"median {per.positives[per.positives > 0].median() if gp else 0:.0f}, max {per.positives.max()}.\n")
    if meta.group_id.nunique() <= 25:
        s += "\n| group_id | windows | positives |\n|---|---:|---:|\n" + "".join(
            f"| {i} | {r.windows} | {r.positives} |\n" for i, r in per.iterrows())
    return s, K


def write(name, parent, meta, Xe, Xc, ne, nc, ch, by, what, why, how):
    meta = meta.reset_index(drop=True).copy()
    meta["window_id"] = np.arange(len(meta))
    Xe = np.ascontiguousarray(Xe, dtype=np.float32); Xc = np.ascontiguousarray(Xc, dtype=np.float32)
    assert nc[:len(ne)] == ne and np.array_equal(Xe, Xc[:, :len(ne)]), name
    assert Xe.shape == (len(meta), len(ne)) and Xc.shape == (len(meta), len(nc)), name
    assert np.isfinite(Xc).all(), name
    d = DST / name
    d.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(d / "meta.parquet", index=False)
    np.save(d / "X_edge.npy", Xe); np.save(d / "X_cloud.npy", Xc)
    (d / "names_edge.json").write_text(json.dumps(ne, indent=1))
    (d / "names_cloud.json").write_text(json.dumps(nc, indent=1))
    (d / "channels.json").write_text(json.dumps(ch, indent=1))
    (d / "bytes.json").write_text(json.dumps(by, indent=1))
    cm, K = counts_md(meta)
    card = (HEADER.format(name=name, parent=parent) + f"## What changed\n{what}\n\n## Why\n{why}\n\n"
            f"## How it was built\n{how}\n\n" + cm +
            f"\nEdge features: {len(ne)}; cloud features: {len(nc)}; edge = first {len(ne)} cloud columns "
            f"(checked). No NaN or inf.\n")
    (d / "card.md").write_text(card)
    print(f"{name}: n={len(meta)} pos={int(meta.label.sum())} groups={meta.group_id.nunique()} K={K} "
          f"edge={len(ne)} cloud={len(nc)}")


def regroup(meta, new_gid, new_order=None):
    meta = meta.copy()
    meta["group_id_orig"] = meta["group_id"]
    meta["group_id"] = new_gid
    if new_order is not None:
        meta["order_orig"] = meta["order"]
        meta["order"] = new_order
    return meta


# ------------------------------------------------------------------ hydraulic
def hyd_common():
    return load_parent("hydraulic")


def build_hyd_blocks():
    meta, Xe, Xc, ne, nc, ch, by = hyd_common()
    size = math.ceil(2205 / 20)
    blk = meta["cycle"].to_numpy() // size
    assert blk.max() == 19
    gid = np.array([f"b{b:02d}" for b in blk])
    order = meta["cycle"].to_numpy() - blk * size
    m = regroup(meta, gid, order)
    write("hyd_blocks", "hydraulic", m, Xe, Xc, ne, nc, ch, by,
          "Grouping only: group_id = 'b%02d' % (cycle // 111), i.e. 20 contiguous blocks of 111 load cycles "
          "(the last block has 96). Features, labels and windows are unchanged. order = cycle position within the "
          "block (used only by router R9); the main-run group_id and order are kept as group_id_orig and order_orig. "
          "Blocks may contain both labels (allowed by the grouped stratified splitter).",
          "Pre-declared deconfounded variant (plan/postplan/DECONF_DECLARATION.md). The parent card "
          "documents that adjacent groups share pump condition and thermal history, so a grouped split that puts "
          "neighbouring groups in train and test leaks; blocked-in-time grouping keeps 111-cycle stretches together.",
          f"meta.cycle // {size}; copied X_edge/X_cloud unchanged.")


def build_hyd_runs():
    meta, Xe, Xc, ne, nc, ch, by = hyd_common()
    gid = np.array([f"run{r:03d}" for r in meta["run_id"]])
    order = meta.groupby("run_id").cumcount().to_numpy()
    m = regroup(meta, gid, order)
    write("hyd_runs", "hydraulic", m, Xe, Xc, ne, nc, ch, by,
          "Grouping only: group_id = 'run%03d' % run_id, the 194 maximal runs of constant (cooler, valve, pump, "
          "accumulator) setting WITHOUT the main-run chunking of runs longer than 60 cycles (the 200-, 200- and "
          "210-cycle runs stay whole). order = position within the run.",
          "'without chunking constant-setting runs'. The main grouping split three long runs into "
          "4 chunks each, so chunks of one run could fall in train and test.",
          "group_id from meta.run_id; features unchanged.")


def build_hyd_drop(name, drop_type, keep_word):
    meta, Xe, Xc, ne, nc, ch, by = hyd_common()
    keep = (meta["fault_type"] != drop_type).to_numpy()
    write(name, "hydraulic", meta[keep], Xe[keep], Xc[keep], ne, nc, ch, by,
          f"Rows only: the {int((~keep).sum())} windows with fault_type = {drop_type} are removed, so the positive "
          f"class is {keep_word} leakage only. Main grouping (group_id unchanged), features unchanged.",
          "report weak and severe leakage separately.",
          f"meta.fault_type != '{drop_type}'.")


def build_hyd_stable():
    meta, Xe, Xc, ne, nc, ch, by = hyd_common()
    keep = (meta["stable"] == 0).to_numpy()
    write("hyd_stable", "hydraulic", meta[keep], Xe[keep], Xc[keep], ne, nc, ch, by,
          f"Rows only: the {int((~keep).sum())} transitional cycles (meta.stable == 1, i.e. static conditions possibly "
          "not yet reached; includes the 36 singleton transition groups) are removed. Main grouping, features unchanged. "
          "Removed rows: 732 normal, 12 weak, 12 severe.",
          "'the result with transitional cycles excluded'.",
          "meta.stable == 0.")


def build_hyd_level(name, col):
    meta, Xe, Xc, ne, nc, ch, by = hyd_common()
    gid = np.array([f"{col}{v}" for v in meta[col]])
    m = regroup(meta, gid)
    lv = sorted(meta[col].unique())
    write(name, "hydraulic", m, Xe, Xc, ne, nc, ch, by,
          f"Grouping only: group_id = {col} level ({len(lv)} groups: {', '.join(map(str, lv))}), a "
          "hold-out-nuisance-condition design: each outer fold tests on one level never seen in training. order is "
          "kept from the main run (position within the original group; groups are not contiguous in time here).",
          "hold-out-nuisance-condition splits. With one group per level every repeat is the same "
          f"partition, so only the first repeat (--splits 0:{len(lv)}) is run and reported per fold, without NB "
          "intervals (SENS_BRIEF).",
          f"group_id from meta.{col}; features unchanged.")


# ------------------------------------------------------------------ SKAB
def skab_common():
    return load_parent("skab")


def cols_of(nc, channels):
    idx = [i for c in channels for i, n in enumerate(nc) if n.split("__")[0] == c]
    return idx, [nc[i] for i in idx]


def skab_channels(edge_ch, cloud_ch, ch, by):
    ch = dict(ch); ch["edge"] = list(edge_ch); ch["cloud"] = list(cloud_ch)
    ch["rates_hz"] = {c: 1.0 for c in cloud_ch}
    by = dict(raw_edge=len(edge_ch) * 60 * 4, raw_cloud=len(cloud_ch) * 60 * 4,
              features_edge=len(edge_ch) * 17 * 4, features_cloud=len(cloud_ch) * 17 * 4,
              note="per window, float32, native 1 Hz, 60-s window (raw arrays not provided in this variant)")
    return ch, by


def skab_subset(name, edge_ch, cloud_ch, what, why):
    meta, Xe, Xc, ne, nc, ch, by = skab_common()
    assert cloud_ch[:len(edge_ch)] == edge_ch
    ic, ncn = cols_of(nc, cloud_ch)
    ie, nen = cols_of(nc, edge_ch)
    Xc2 = Xc[:, ic]; Xe2 = Xc[:, ie]
    ch2, by2 = skab_channels(edge_ch, cloud_ch, ch, by)
    write(name, "skab", meta, Xe2, Xc2, nen, ncn, ch2, by2, what, why,
          f"Column selection from the parent X_cloud by channel name; edge = {edge_ch}; cloud = {cloud_ch} (edge first). "
          "Feature values unchanged. bytes.json recomputed for the new channel counts (17 features x 4 bytes per channel; "
          "60 float32 samples per channel).")


def build_skab_filez():
    meta, Xe, Xc, ne, nc, ch, by = skab_common()
    Z = Xc.astype(np.float64).copy()
    floored = 0
    for g, idx in meta.groupby("group_id").indices.items():
        mu = Z[idx].mean(axis=0); sd = Z[idx].std(axis=0, ddof=0)
        floored += int((sd < 1e-6).sum())
        Z[idx] = (Z[idx] - mu) / np.maximum(sd, 1e-6)
    Zc = Z.astype(np.float32)
    write("skab_filez", "skab", meta, Zc[:, :len(ne)], Zc, ne, nc, ch, by,
          "Feature values only: every one of the 136 cloud feature columns is z-scored within its group_id (one "
          "experiment file, or one 1245-s chunk of the anomaly-free recording), using that group's mean and population "
          "sd over all its windows (sd floor 1e-6). The cloud matrix is normalised once and the edge matrix is its first "
          f"85 columns, so the nested-view property holds exactly. {floored} (group, column) pairs hit the sd floor "
          "(constant columns become 0). Rows, labels, grouping unchanged.\n\n**This normalisation is NON-CAUSAL**: each "
          "window is normalised with statistics of its whole recording, including later windows and the anomalous "
          "segment itself. It removes recording-level (and therefore session-level) offsets and scales, but it also uses "
          "information a deployed edge node would not have.",
          "Pre-declared deconfounded variant (DECONF_DECLARATION.md) ('features normalised within each "
          "file'). The parent card documents that normal windows come from other sessions than the anomaly-free chunks, "
          "so absolute levels can identify the session/recording rather than the fault.",
          "per group: (x - mean_g) / max(sd_g, 1e-6) on float64, cast to float32.")


def build_skab_base():
    meta, Xe, Xc, ne, nc, ch, by = skab_common()
    Z = Xc.astype(np.float64).copy()
    lines = []
    for f, idx in meta.groupby("file").indices.items():
        sub = meta.iloc[idx]
        b = idx[(sub["t_start_s"] <= 40).to_numpy()]
        assert len(b) == 5, (f, len(b))
        assert (meta.loc[b, "label"] == 0).all() and (meta.loc[b, "label_frac"] == 0).all(), f
        Z[idx] = Z[idx] - Z[b].mean(axis=0)
        lines.append(f)
    Zc = Z.astype(np.float32)
    write("skab_base", "skab", meta, Zc[:, :len(ne)], Zc, ne, nc, ch, by,
          "Feature values only: from every feature, subtract the mean of that feature over the recording's first five "
          "windows (t_start_s in {0, 10, 20, 30, 40}, i.e. the first 100 s of the recording). 'Recording' = source file "
          f"(meta.file, {len(lines)} files); the eight anomaly-free chunks share one recording and therefore one baseline "
          "(the first 100 s of anomaly-free.csv, which lie in free_0). Verified: every recording has exactly 5 such "
          "windows and all 175 are normal (label 0 and label_frac 0, i.e. no anomalous second). Normalised once on the "
          "cloud matrix; edge = first 85 columns.\n\nCausality: the baseline uses only the first 100 s of each recording, so "
          "it is causal for every window starting at or after 40 s; the five baseline windows themselves use up to 60 s of "
          "data after their own end. No scaling, only an offset.",
          "a deployable alternative to within-file normalisation that removes per-recording "
          "offsets without using future or anomalous data.",
          "per file: x - mean(x over windows with t_start_s <= 40), float64, cast to float32.")


def build_skab_lso():
    meta, Xe, Xc, ne, nc, ch, by = skab_common()
    m = regroup(meta, meta["session"].astype(str).to_numpy())
    write("skab_lso", "skab", m, Xe, Xc, ne, nc, ch, by,
          "Grouping only: group_id = session (recording date: 2020-02-08 = anomaly-free + other/5-14; 2020-03-01 = "
          "other/1-4; 2020-03-09 = valve1 + valve2). Each session contains both classes, so K = 3 and each outer fold "
          "tests on one whole session (leave-session-out). order kept (time order within the file). Features unchanged.",
          "leave-session-out. With one group per session every repeat is the same partition, so only "
          "--splits 0:3 is run and reported per fold, without NB intervals.",
          "group_id from meta.session.")


def main(argv):
    B = dict(
        hyd_blocks=build_hyd_blocks, skab_filez=build_skab_filez, hyd_runs=build_hyd_runs,
        hyd_weak=lambda: build_hyd_drop("hyd_weak", "pump_leak_severe", "weak"),
        hyd_severe=lambda: build_hyd_drop("hyd_severe", "pump_leak_weak", "severe"),
        hyd_stable=build_hyd_stable,
        hyd_cooler=lambda: build_hyd_level("hyd_cooler", "cooler"),
        hyd_accum=lambda: build_hyd_level("hyd_accum", "accumulator"),
        skab_lso=build_skab_lso,
        skab_notemp=lambda: skab_subset(
            "skab_notemp", [c for c in SKAB_EDGE if c != "Temperature"],
            [c for c in SKAB_EDGE if c != "Temperature"] + ["Pressure", "Volume Flow RateRMS"],
            "Columns only: the motor-body Temperature channel is removed from edge and cloud, and the fluid "
            "Thermocouple channel is removed from cloud (edge = 4 channels, 68 features; cloud = 6 channels, 102 "
            "features). Rows, labels and grouping unchanged.",
            "Hypothesis tested: temperature warm-up drifts with run time and the label follows a "
            "fixed script inside each file (parent card caveat 1), so temperature may act as a clock."),
        skab_base=build_skab_base,
        skab_edgeP=lambda: skab_subset(
            "skab_edgeP", SKAB_EDGE + ["Pressure"], SKAB_EDGE + ["Pressure", "Thermocouple", "Volume Flow RateRMS"],
            "Columns only: the alternative admissible edge named in the parent card: edge = the five drive channels "
            "plus Pressure (pump discharge), 6 channels, 102 features; cloud = all eight channels with the six edge "
            "channels first (136 features, same values as the parent, reordered).",
            "alternative edge definition. The parent card states that Pressure is measured at the pump "
            "discharge and a 6-channel edge is admissible but was not chosen as primary."),
    )
    for ch in SKAB_LOOP:
        tag = {"Pressure": "pressure", "Thermocouple": "thermo", "Volume Flow RateRMS": "flow"}[ch]
        B[f"skab_cm_{tag}"] = (lambda ch=ch, tag=tag: skab_subset(
            f"skab_cm_{tag}", SKAB_EDGE, SKAB_EDGE + [c for c in SKAB_LOOP if c != ch],
            f"Columns only: cloud = all eight channels minus the loop channel {ch} (7 channels, 119 features); edge "
            "unchanged (5 drive channels, 85 features).",
            "which loop channel carries the SKAB pair A "
            "model x information interaction of about 0.12."))
    for n in (argv or list(B)):
        B[n]()


if __name__ == "__main__":
    main(sys.argv[1:])
