"""POST-PLAN, descriptive, evaluation only. Refits no model.

1. Off-schedule SKAB files. Four experiment files do not follow the 560-990 s injection schedule
   (results/extra/skab_onset_timing.csv): other/2 (first fault window at 354 s), other/4 (810 s), other/13 (653 s)
   and valve1/2 (666 s). For each file and each of three runs, with the stored predictions thresholded at the stored
   thresholds (edge p_ef_ev >= t_ef_ev; cloud p_cf_cv >= t_cf_cv):
     pre-specified   results/skab/<pair>/arrays_<j>.npz, j = 0..19, meta data/processed/skab/meta.parquet
                     (every file is in the test fold once per repeat, so four appearances; pooled over them);
     causal offset   results_sens/hydskab/skab_base/<pair>/arrays_<j>.npz, meta data/processed_sens/skab_base;
     offset session hold-out  results_sens/holdout2/skab_lso2_base/<pair>/arrays_<j>.npz (one appearance).
   Recall, specificity, and the first alarm (start time of the first window predicted positive in the file;
   median over appearances) against the actual onset (first fault window). The time-in-file comparator is the
   fixed schedule rule "positive if 560 s <= window start <= 990 s", read off the documented injection schedule
   (Section 4.1 of the paper; not fitted), and the depth-3 time-in-file tree of the shortcut baselines
   (a_t_start_tree3 of src/sens/shortcuts/run_shortcuts.py). The tree's per-window predictions were not stored, so
   they are reconstructed as in src/extra/r4_skab_clock.py: the deterministic metadata baseline is refitted with the
   same code, seed and fit/validation/test indices, and its test balanced accuracy must reproduce the stored value
   on every split (assertion; results_sens/shortcuts/shortcuts_per_split.csv for the pre-specified splits,
   results_sens/shortcuts_controls/per_split.csv, variant skab_lso2_base, for the session hold-out). The causal-
   offset run has the same windows, labels, groups, start times and test folds as the pre-specified run (asserted),
   so its tree predictions are those of the pre-specified splits. No edge, cloud or router model is refitted.
2. Schedule rule and time-in-file tree by family on the offset session hold-out: recall and specificity
   per held-out session and disturbance family, beside the rows of holdout_skab_by_type.csv.
3. Edge false alarms along the anomaly-free recording: the offset session hold-out fold that holds the
   anomaly-free recording (2020-02-08); false-alarm rate of edge and cloud per 1,200-s segment of the recording.
Outputs: results/extra/r6_skab_offschedule.csv, r6_skab_schedule_rule_by_family.csv,
         r6_skab_anomaly_free_drift.csv
    python src/extra/r6_skab_offschedule.py
"""
from __future__ import annotations
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/extra/common.py first
import common as X                                               # noqa: E402
for _p in ("src", "src/sens", "src/sens/shortcuts", "src/sens/designs"):
    sys.path.append(str(ROOT / _p))
OUT = ROOT / "results" / "extra"
FILES = ["other/2.csv", "other/4.csv", "other/13.csv", "valve1/2.csv"]
SCHED = (560, 990)
RUNS = [("pre-specified", ROOT / "results" / "skab", ROOT / "data" / "processed" / "skab" / "meta.parquet"),
        ("causal offset", ROOT / "results_sens" / "hydskab" / "skab_base",
         ROOT / "data" / "processed_sens" / "skab_base" / "meta.parquet"),
        ("offset session hold-out", ROOT / "results_sens" / "holdout2" / "skab_lso2_base",
         ROOT / "data" / "processed_sens" / "skab_base" / "meta.parquet")]


def arrays(base, pair):
    return sorted(f for f in (base / pair).glob("arrays_*.npz") if "_q" not in f.stem)


def rule(t):
    return ((t >= SCHED[0]) & (t <= SCHED[1])).astype(int)


def tree_predictions():
    """Per run: {split: (test_idx, 0/1 predictions of the time-in-file tree)}, reproduced against stored test BA."""
    from escalation.data import load_dataset
    from escalation.metrics import threshold_max_ba, balanced_accuracy
    from run_shortcuts import tree
    import shortcuts_under_controls as SU

    def fit_pred(t, y, sp):
        m = tree(); m.fit(t[sp["fit"]].reshape(-1, 1), y[sp["fit"]])
        k = list(m.classes_).index(1)
        thr = threshold_max_ba(m.predict_proba(t[sp["val"]].reshape(-1, 1))[:, k], y[sp["val"]])
        return (m.predict_proba(t[sp["test"]].reshape(-1, 1))[:, k] >= thr).astype(int)

    out = {}
    a = X.dataset("skab"); ma = a["meta"].reset_index(drop=True)
    t = ma.t_start_s.to_numpy(float); y = np.asarray(a["y"]).astype(int)
    st = pd.read_csv(ROOT / "results_sens" / "shortcuts" / "shortcuts_per_split.csv")
    st = st[(st.dataset == "skab") & (st.shortcut == "a_t_start_tree3")].set_index("split").ba_test
    pre = {}
    for j, sp in X.splits("skab").items():
        p = fit_pred(t, y, sp)
        assert abs(balanced_accuracy(y[sp["test"]], p) - st[j]) < 1e-9, ("pre-specified", j)
        pre[j] = (np.asarray(sp["test"]), p)
    out["pre-specified"] = pre
    b = load_dataset("skab_base", ROOT / "data" / "processed_sens"); mb = b["meta"].reset_index(drop=True)
    assert np.array_equal(mb.t_start_s.to_numpy(), ma.t_start_s.to_numpy()) and np.array_equal(np.asarray(b["y"]), np.asarray(a["y"]))
    assert np.array_equal(np.asarray(b["groups"]), np.asarray(a["groups"])) and np.array_equal(mb.file.to_numpy(), ma.file.to_numpy())
    for pair in ("A", "B"):
        for j, (idx, _) in pre.items():
            with np.load(ROOT / "results_sens" / "hydskab" / "skab_base" / pair / f"arrays_{j}.npz") as z:
                assert np.array_equal(np.sort(z["test_idx"]), np.sort(idx)), ("causal offset", pair, j)
    out["causal offset"] = pre
    sc = pd.read_csv(ROOT / "results_sens" / "shortcuts_controls" / "per_split.csv")
    sc = sc[(sc.variant == "skab_lso2_base") & (sc.shortcut == "a_t_start_tree3")].set_index("split").ba_test
    yb = np.asarray(b["y"]).astype(int); tb = mb.t_start_s.to_numpy(float)
    ho = {}
    for sp in SU.holdout2_splits(b, "session"):
        p = fit_pred(tb, yb, sp)
        assert abs(balanced_accuracy(yb[sp["test"]], p) - sc[sp["split_id"]]) < 1e-9, ("hold-out", sp["split_id"])
        ho[sp["split_id"]] = (np.asarray(sp["test"]), p)
    out["offset session hold-out"] = ho
    return out


def tree_lookup(TP, run, j, test_idx):
    idx, p = TP[run][j]
    pos = pd.Series(np.arange(len(idx)), index=idx)
    return p[pos.loc[test_idx].to_numpy()]


def main():
    onset = pd.read_csv(OUT / "skab_onset_timing.csv").set_index("file")
    TP = tree_predictions()
    rows = []
    for run, base, mp in RUNS:
        meta = pd.read_parquet(mp)
        for pair in ("A", "B"):
            acc = {f: dict(y=[], e=[], c=[], r=[], k=[], fa_e=[], fa_c=[], fa_r=[], fa_k=[], n=0) for f in FILES}
            for fz in arrays(base, pair):
                z = np.load(fz)
                yk = tree_lookup(TP, run, int(fz.stem.split("_")[1]), z["test_idx"])
                mt = meta.iloc[z["test_idx"]].reset_index(drop=True)
                assert np.array_equal(mt.label.to_numpy().astype(int), z["y_test"].astype(int))
                ye = (z["p_ef_ev"] >= float(z["t_ef_ev"])).astype(int)
                yc = (z["p_cf_cv"] >= float(z["t_cf_cv"])).astype(int)
                for f in FILES:
                    ix = np.flatnonzero(mt.file.to_numpy() == f)
                    if not len(ix):
                        continue
                    ix = ix[np.argsort(mt.t_start_s.to_numpy()[ix], kind="stable")]
                    t = mt.t_start_s.to_numpy()[ix]; y = mt.label.to_numpy()[ix].astype(int)
                    assert int(t[y == 1].min()) == int(onset.loc[f, "first_pos"]), (run, f)
                    a = acc[f]; a["n"] += 1
                    for key, pred, fk in (("e", ye[ix], "fa_e"), ("c", yc[ix], "fa_c"), ("r", rule(t), "fa_r"),
                                          ("k", yk[ix], "fa_k")):
                        a[key].append(pred)
                        a[fk].append(float(t[pred == 1].min()) if pred.any() else np.nan)
                    a["y"].append(y)
            for f in FILES:
                a = acc[f]
                if not a["n"]:
                    continue
                y = np.concatenate(a["y"])
                rec = dict(run=run, pair=pair, file=f, onset_s=int(onset.loc[f, "first_pos"]),
                           in_schedule_window=bool(SCHED[0] <= onset.loc[f, "first_pos"] <= SCHED[1]),
                           appearances=a["n"], n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
                for key, lab in (("e", "edge"), ("c", "cloud"), ("r", "schedule_rule"), ("k", "time_tree")):
                    p = np.concatenate(a[key])
                    rec[f"recall_{lab}"] = float(p[y == 1].mean())
                    rec[f"spec_{lab}"] = float(1 - p[y == 0].mean())
                    fa = np.array(a["fa_" + key], float)
                    rec[f"first_alarm_{lab}_median_s"] = float(np.nanmedian(fa)) if np.isfinite(fa).any() else np.nan
                    rec[f"first_alarm_{lab}_min_s"] = float(np.nanmin(fa)) if np.isfinite(fa).any() else np.nan
                    rec[f"first_alarm_{lab}_max_s"] = float(np.nanmax(fa)) if np.isfinite(fa).any() else np.nan
                rows.append(rec)
    o = pd.DataFrame(rows)
    o.to_csv(OUT / "r6_skab_offschedule.csv", index=False)

    # 2. schedule rule by family on the offset session hold-out (same windows as holdout_skab_by_type.csv)
    run, base, mp = RUNS[2]
    meta = pd.read_parquet(mp)
    bt = pd.read_csv(OUT / "holdout_skab_by_type.csv")
    bt = bt[bt.variant == "skab_lso2_base"]
    fam = []
    for pair in ("A", "B"):
        for fz in arrays(base, pair):
            j = int(fz.stem.split("_")[1]); z = np.load(fz)
            mt = meta.iloc[z["test_idx"]].reset_index(drop=True)
            y = z["y_test"].astype(int); r = rule(mt.t_start_s.to_numpy())
            kk = tree_lookup(TP, run, j, z["test_idx"])
            for t, g in mt.groupby("exp_fault_type"):
                ix = g.index.to_numpy(); pos, neg = ix[y[ix] == 1], ix[y[ix] == 0]
                ref = bt[(bt.pair == pair) & (bt.split == j) & (bt.family == t)].iloc[0]
                assert ref.n_pos == len(pos) and ref.n_neg == len(neg)
                fam.append(dict(pair=pair, split=j, held_out=ref.held_out, family=t, n_pos=len(pos), n_neg=len(neg),
                                recall_rule=r[pos].mean() if len(pos) else np.nan,
                                spec_rule=1 - r[neg].mean() if len(neg) else np.nan,
                                recall_tree=kk[pos].mean() if len(pos) else np.nan,
                                spec_tree=1 - kk[neg].mean() if len(neg) else np.nan,
                                recall_cloud=ref.recall_cloud, spec_cloud=ref.spec_cloud,
                                recall_edge=ref.recall_edge, spec_edge=ref.spec_edge))
    F = pd.DataFrame(fam)
    F.to_csv(OUT / "r6_skab_schedule_rule_by_family.csv", index=False)

    # 3. false alarms along the anomaly-free recording, offset session hold-out
    dr = []
    for pair in ("A", "B"):
        for fz in arrays(base, pair):
            z = np.load(fz); mt = meta.iloc[z["test_idx"]].reset_index(drop=True)
            af = (mt.subset == "anomaly-free").to_numpy()
            if not af.any():
                continue
            ye = (z["p_ef_ev"] >= float(z["t_ef_ev"])).astype(int)[af]
            yc = (z["p_cf_cv"] >= float(z["t_cf_cv"])).astype(int)[af]
            t = mt.t_start_s.to_numpy()[af]
            for lo in range(0, int(t.max()) + 1, 1200):
                ix = (t >= lo) & (t < lo + 1200)
                if ix.any():
                    dr.append(dict(pair=pair, split=int(fz.stem.split("_")[1]), segment_start_s=lo, n=int(ix.sum()),
                                   false_alarm_edge=float(ye[ix].mean()), false_alarm_cloud=float(yc[ix].mean())))
            dr.append(dict(pair=pair, split=int(fz.stem.split("_")[1]), segment_start_s=-1, n=int(af.sum()),
                           false_alarm_edge=float(ye.mean()), false_alarm_cloud=float(yc.mean())))
    D = pd.DataFrame(dr)
    D.to_csv(OUT / "r6_skab_anomaly_free_drift.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(o.round(3).to_string()); print(F.round(3).to_string()); print(D.round(3).to_string())


if __name__ == "__main__":
    main()
