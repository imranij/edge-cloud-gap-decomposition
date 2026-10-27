"""POST-PLAN: SKAB single-loop-channel clouds and
shortcut-augmented edges. Features copied from data/processed (or data/processed_sens/hyd_blocks),
only the views change; labels, groups and splits are unchanged.
  skab_e_flow / skab_e_press / skab_e_thermo : cloud = the five drive channels + one loop channel (R3)
  skab_sc     : edge = drive features + t_start_s (time in file); cloud = edge + t_start_s + loop channels (R4)
  hyd_blocks_sc : hyd_blocks with the cycle index added to the edge (and hence the cloud) view (R4)
"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def write(name, src, Xe, ne, Xc, nc, note, meta=None):
    dst = ROOT / "data" / "processed_sens" / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    assert nc[:len(ne)] == ne
    np.save(dst / "X_edge.npy", np.ascontiguousarray(Xe, dtype=np.float32))
    np.save(dst / "X_cloud.npy", np.ascontiguousarray(Xc, dtype=np.float32))
    (dst / "names_edge.json").write_text(json.dumps(ne)); (dst / "names_cloud.json").write_text(json.dumps(nc))
    if meta is None:
        shutil.copy2(src / "meta.parquet", dst / "meta.parquet")
    else:
        meta.to_parquet(dst / "meta.parquet", index=False)
    b = json.loads((src / "bytes.json").read_text())
    b["features_edge"] = 4 * len(ne); b["features_cloud"] = 4 * len(nc)
    b.pop("raw_edge", None); b.pop("raw_cloud", None)
    (dst / "bytes.json").write_text(json.dumps(b, indent=1))
    shutil.copy2(src / "channels.json", dst / "channels.json")
    (dst / "card.md").write_text(f"# {name} (post-plan sensitivity variant; not confirmatory)\n\n{note}\nBuilt by src/sens/designs/build_round3_variants.py.\n")
    print(name, len(ne), len(nc))


# SKAB single loop channel
S = ROOT / "data" / "processed" / "skab"
ne = json.loads((S / "names_edge.json").read_text()); nc = json.loads((S / "names_cloud.json").read_text())
Xe = np.load(S / "X_edge.npy"); Xc = np.load(S / "X_cloud.npy")
loop = {"flow": "Volume Flow RateRMS", "press": "Pressure", "thermo": "Thermocouple"}
for k, ch in loop.items():
    keep = list(range(len(ne))) + [i for i, n in enumerate(nc) if n.split("__")[0] == ch]
    assert len(keep) == len(ne) + 17, (k, len(keep))
    write(f"skab_e_{k}", S, Xe, ne, Xc[:, keep], [nc[i] for i in keep],
          f"Cloud view = the five drive channels plus {ch} only (edge unchanged).")
# SKAB shortcut-augmented edge
m = pd.read_parquet(S / "meta.parquet")
t = m["t_start_s"].to_numpy(float)[:, None]
loop_cols = [i for i in range(len(nc)) if i >= len(ne)]
write("skab_sc", S, np.hstack([Xe, t]), ne + ["time_in_file__t_start_s"],
      np.hstack([Xc[:, :len(ne)], t, Xc[:, loop_cols]]), ne + ["time_in_file__t_start_s"] + [nc[i] for i in loop_cols],
      "Shortcut-augmented views: the time-in-file variable t_start_s (the strongest SKAB shortcut, BA 0.961 alone) "
      "is added to the edge view and therefore to the cloud view. The information step then measures what the "
      "loop channels add beyond the drive channels AND the window's position in its recording.")
# hydraulic shortcut-augmented edge on the blocked grouping
H = ROOT / "data" / "processed_sens" / "hyd_blocks"
ne = json.loads((H / "names_edge.json").read_text()); nc = json.loads((H / "names_cloud.json").read_text())
Xe = np.load(H / "X_edge.npy"); Xc = np.load(H / "X_cloud.npy")
m = pd.read_parquet(H / "meta.parquet")
c = m["cycle"].to_numpy(float)[:, None]
rest = list(range(len(ne), len(nc)))
write("hyd_blocks_sc", H, np.hstack([Xe, c]), ne + ["cycle_index__cycle"],
      np.hstack([Xc[:, :len(ne)], c, Xc[:, rest]]), ne + ["cycle_index__cycle"] + [nc[i] for i in rest],
      "hyd_blocks (20 contiguous blocks) with the global cycle index (the hydraulic shortcut variable) added to the "
      "edge view and therefore to the cloud view.")
