"""POST-PLAN, evaluation only. Refits no model.

The corrected audit with ONE null, the within-class permutation null (validation gains permuted among
windows of the same label), as computed for all 400 cells in results_sens/routers_wc/p1w_cells.csv
(src/sens/designs/p1w_within_class.py). The across-window null used earlier
(results_sens/routers/p1_cells.csv) is summarised beside it for comparison.

For each null and each outcome (routing beats random: Cap(R6) > 0.10; escalation beats the edge: R6 at
10% above B_eE) the balanced accuracy of the GO calls is given
  (a) over all cells, with undefined cells (no finite fold-averaged C, or no null in the block) as NO-GO,
  (b) over the defined cells only,
with permutation p-values (10,000 permutations, calls permuted within datasets and within dataset x
pair x level blocks), for all cells and for full training data (q = 1, pairs A and B).
Fold-averaging rule as run: the plain mean of the finite fold C values (not D-weighted).
Output: results/extra/audit_within_class.csv
    python src/extra/audit_within_class.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

N_PERM = 10_000


def ba_stats(call, out):
    call = np.asarray(call, int); out = np.asarray(out, int)
    tp = ((call == 1) & (out == 1)).sum(-1); fp = ((call == 1) & (out == 0)).sum(-1)
    fn = ((call == 0) & (out == 1)).sum(-1); tn = ((call == 0) & (out == 0)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def perm_p(call, out, blocks, seed):
    call = np.asarray(call, int); out = np.asarray(out, int)
    obs = float(ba_stats(call, out))
    if not np.isfinite(obs):
        return obs, np.nan
    rng = np.random.default_rng(seed)
    P = np.empty((N_PERM, len(call)), dtype=np.int8)
    for b in np.unique(blocks):
        ix = np.flatnonzero(blocks == b)
        P[:, ix] = call[ix][rng.permuted(np.tile(np.arange(len(ix)), (N_PERM, 1)), axis=1)]
    bp = ba_stats(P, out[None, :])
    return obs, float((1 + (bp >= obs - 1e-12).sum()) / (N_PERM + 1))


def main():
    src = {"within-class null": X.ROOT / "results_sens" / "routers_wc" / "p1w_cells.csv",
           "across-window null": X.ROOT / "results_sens" / "routers" / "p1_cells.csv"}
    rows, seed = [], 9600
    for null, f in src.items():
        d = pd.read_csv(f)
        d["defined"] = np.isfinite(d.C_cor) & np.isfinite(d.thr95)
        for subset, m in [("all cells", np.ones(len(d), bool)), ("q = 1, pairs A and B", (d.q == 1.0).to_numpy())]:
            for cells, mm in [("undefined as NO-GO", m), ("defined cells only", m & d.defined.to_numpy())]:
                sub = d[mm]
                for oc, ol in [("out_capture", "routing beats random"), ("out_edge", "escalation beats the edge")]:
                    c, o = sub.go_cor.to_numpy(), sub[oc].to_numpy().astype(int)
                    ba, p_ds = perm_p(c, o, sub.dataset.to_numpy(), seed)
                    _, p_blk = perm_p(c, o, sub.block.to_numpy(), seed + 1)
                    seed += 2
                    rows.append(dict(null=null, subset=subset, cells=cells, outcome=ol, n=len(sub),
                                     n_undefined=int((~sub.defined).sum()), n_go=int(c.sum()), base_rate=o.mean(),
                                     ba=ba, p_within_dataset=p_ds, p_within_block=p_blk,
                                     undefined_by_dataset="; ".join(f"{k} {v}" for k, v in
                                                                    (~d[m].defined).groupby(d[m].dataset).sum().items() if v)))
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "audit_within_class.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["undefined_by_dataset"]).round(3).to_string())
    print(out.undefined_by_dataset.unique())


if __name__ == "__main__":
    main()
