"""Size of the confounds documented in the dataset cards, computed
from labels and metadata alone. Post-plan analysis, not pre-specified. No classifier is
trained here: the shortcut BASELINES (well-only, position-only, cycle-index-only
predictors under the paper's splits) are new experiments and are run by
src/sens/shortcuts/run_shortcuts.py instead.

    python src/extra/dataset_confounds.py   -> results/extra/dataset_confounds.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X


def main():
    rows = []
    # 3W: well identity and label
    m = X.dataset("threew")["meta"]
    fr = m.groupby("well").label.mean()
    rows += [dict(dataset="threew", stat="wells", value=m.well.nunique()),
             dict(dataset="threew", stat="min per-well fault share", value=fr.min(), note=fr.idxmin()),
             dict(dataset="threew", stat="max per-well fault share", value=fr.max(), note=fr.idxmax())]
    neg = m[m.label == 0].well.value_counts(normalize=True)
    rows.append(dict(dataset="threew", stat="share of normal windows from the three largest wells",
                     value=neg.iloc[:3].sum(), note=";".join(neg.index[:3])))
    allnorm = m.groupby("instance").label.max()
    wells_allnorm = m[m.instance.isin(allnorm[allnorm == 0].index)].well.nunique()
    rows.append(dict(dataset="threew", stat="wells contributing all-normal instances", value=wells_allnorm))
    # SKAB: position within experiment files
    s = X.dataset("skab")["meta"]
    exp = s[s.subset != "anomaly-free"] if "subset" in s else s
    rhos = [X.spearman(g.order, g.label) for _, g in exp.groupby("group_id") if g.label.nunique() == 2]
    rows += [dict(dataset="skab", stat="median within-file Spearman(position, label)", value=float(np.nanmedian(rhos)),
                  note=f"{len(rhos)} files with both classes"),
             dict(dataset="skab", stat="sessions", value=s.session.nunique(), note=";".join(sorted(map(str, s.session.unique()))))]
    for sess, g in s.groupby("session"):
        rows.append(dict(dataset="skab", stat=f"session {sess}: windows / positives / groups",
                         value=len(g), note=f"{int(g.label.sum())} positives; {g.group_id.nunique()} groups"))
    # hydraulic: adjacent groups and transitional cycles
    h = X.dataset("hydraulic")["meta"].sort_values("cycle")
    gl = h.groupby("group_id", sort=False).agg(first=("cycle", "min"), label=("label", "first")).sort_values("first")
    same = int((gl.label.values[1:] == gl.label.values[:-1]).sum())
    rows += [dict(dataset="hydraulic", stat="adjacent group pairs with the same label", value=same, note=f"of {len(gl) - 1}"),
             dict(dataset="hydraulic", stat="cycles flagged stable = 1 (possibly not yet stable)", value=int((h.stable == 1).sum()))]
    # SEU: recording-level offset of motor vibration (the 'mean' feature, edge view)
    d = X.dataset("seu")
    j = d["names_edge"].index("motor_vib__mean")
    off = pd.Series(np.asarray(d["X_edge"][:, j], float)).groupby(d["meta"].group_id.values).mean()
    lab = d["meta"].groupby("group_id").label.first()
    rows += [dict(dataset="seu", stat="motor_vib mean feature, per-recording mean, healthy min..max",
                  value=float(off[lab[lab == 0].index].min()), note=f"max {off[lab[lab == 0].index].max():.3f}"),
             dict(dataset="seu", stat="motor_vib mean feature, per-recording mean, faulty min..max",
                  value=float(off[lab[lab == 1].index].min()), note=f"max {off[lab[lab == 1].index].max():.3f}")]
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "dataset_confounds.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.to_string())


if __name__ == "__main__":
    main()
