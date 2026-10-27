"""POST-PLAN: which cloud-only sensor family carries the
hydraulic information step? Builds six variants of the declared deconfounded variant hyd_blocks
(blocked-in-time grouping, features unchanged) that differ only in the cloud view:
  hyd_blk_cm_flow / _cm_press / _cm_temp : cloud = all sensors minus FS1-FS2 / PS1-PS6 / TS1-TS4
  hyd_blk_e_flow / _e_press / _e_temp    : cloud = edge (EPS1, VS1) plus only FS1-FS2 / PS1-PS6 / TS1-TS4
The edge view, labels, groups and splits are those of hyd_blocks.
"""
import json
import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "processed_sens" / "hyd_blocks"
FAM = {"flow": ["FS1", "FS2"], "press": [f"PS{i}" for i in range(1, 7)], "temp": [f"TS{i}" for i in range(1, 5)]}
EDGE = ["EPS1", "VS1"]


def build(name, cloud_channels):
    dst = ROOT / "data" / "processed_sens" / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    names = json.loads((SRC / "names_cloud.json").read_text())
    Xc = np.load(SRC / "X_cloud.npy")
    keep = [i for i, n in enumerate(names) if n.split("__")[0] in cloud_channels]
    order = sorted(keep, key=lambda i: cloud_channels.index(names[i].split("__")[0]))  # edge channels first
    np.save(dst / "X_cloud.npy", np.ascontiguousarray(Xc[:, order], dtype=np.float32))
    nc = [names[i] for i in order]
    (dst / "names_cloud.json").write_text(json.dumps(nc))
    for f in ("meta.parquet", "X_edge.npy", "names_edge.json"):
        shutil.copy2(SRC / f, dst / f)
    ne = json.loads((SRC / "names_edge.json").read_text())
    assert nc[:len(ne)] == ne
    b = json.loads((SRC / "bytes.json").read_text()); b["features_cloud"] = 4 * len(nc); b.pop("raw_cloud", None)
    (dst / "bytes.json").write_text(json.dumps(b, indent=1))
    ch = json.loads((SRC / "channels.json").read_text()); ch["cloud"] = cloud_channels
    (dst / "channels.json").write_text(json.dumps(ch, indent=1))
    (dst / "card.md").write_text(
        f"# {name} (post-plan sensitivity variant; not confirmatory)\n\nCopy of hyd_blocks (blocked-in-time grouping, "
        f"20 contiguous blocks; features, labels and edge view unchanged). Cloud view = {', '.join(cloud_channels)} "
        f"({len(nc)} features). Built by src/sens/designs/build_hyd_ablation.py.\n")
    print(name, len(nc))


ALL = EDGE + FAM["press"] + FAM["flow"] + FAM["temp"]
for fam, chans in FAM.items():
    build(f"hyd_blk_cm_{fam}", [c for c in ALL if c not in chans])
    build(f"hyd_blk_e_{fam}", EDGE + chans)
