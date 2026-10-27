"""POST-PLAN, evaluation only. Refits no study model.

SKAB injection timing and the time-in-file clock.

1. Timing of the injected anomalies (data/processed/skab/meta.parquet, experiment files only): per file the
   t_start_s of the first and the last positive window, and the positive rate at each 10-s position inside the
   two 120-s bins that hold the usual onset (480-599 s) and end (960-1079 s).
2. The depth-3 time-in-file tree (shortcut a_t_start_tree3 of src/sens/shortcuts/run_shortcuts.py: class-balanced
   DecisionTreeClassifier(max_depth=3, random_state=0) on t_start_s, fitted on the fit part, threshold maximising
   BA on the validation part) re-evaluated on the plan's splits. It is a deterministic metadata baseline; this
   script refits it with the same code, seed and splits and checks that its test BA reproduces the stored value in
   results_sens/shortcuts/shortcuts_per_split.csv on every split (assertion), so the per-window predictions it
   scores are those of the stored baseline. No edge, cloud or router model is refitted.
   From these predictions:
   - within-bin BA of the clock under the matched-elapsed-time rule of src/extra/skab_breakdown.py (120-s bins,
     experiment files only, bins with >= 5 windows of each class), beside the within-bin BA of the four factorial
     models of pairs A and B (results/skab/{A,B}/arrays_<j>.npz); pairs A and B share their splits;
   - the number of splits in which each bin qualifies;
   - the clock's BA on experiment-file windows only, and with chunk-relative time for the anomaly-free
     chunks (sensitivity, R2 m3).
3. The experiment-file-only information step (file_info) and the all-window step (all_info) of
   results/extra/skab_breakdown.csv are copied beside it for the text.
Outputs: results/extra/skab_onset_timing.csv (per file), results/extra/skab_position_rate.csv (10-s positions),
         results/extra/skab_clock_timestrat.csv (per split), results/extra/skab_clock_summary.csv (summary)
    python src/extra/r4_skab_clock.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import common as X
from escalation.metrics import threshold_max_ba, balanced_accuracy

BIN_S, MIN_PER_CLASS = 120, 5
MODELS = {"eE": "ef_ev", "cE": "cf_ev", "eC": "ef_cv", "cC": "cf_cv"}


def ba(y, pred):
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def clock_tree(t, y, sp):
    fit, val, test = sp["fit"], sp["val"], sp["test"]
    m = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    m.fit(t[fit].reshape(-1, 1), y[fit])
    k = list(m.classes_).index(1)
    pv = m.predict_proba(t[val].reshape(-1, 1))[:, k]
    thr = threshold_max_ba(pv, y[val])
    return m, thr


def main():
    ds = X.dataset("skab")
    meta = ds["meta"].reset_index(drop=True)
    y_all = np.asarray(ds["y"]).astype(int)
    t_all = meta.t_start_s.to_numpy(float)
    exp = (meta.subset != "anomaly-free").to_numpy()

    # 1. timing of the injections
    e = meta[exp]
    g = e[e.label == 1].groupby("file").t_start_s.agg(first_pos="min", last_pos="max")
    g["n_windows"] = e.groupby("file").size()
    g["t_max"] = e.groupby("file").t_start_s.max()
    g = g.reset_index()
    g.to_csv(X.OUT / "skab_onset_timing.csv", index=False)
    n_files = int(e.file.nunique())
    pos = []
    for lo, hi in ((480, 600), (960, 1080)):
        for t in range(lo, hi, 10):
            m = exp & (t_all == t)
            pos.append(dict(bin_lo=lo, t_start_s=t, n=int(m.sum()), pos_rate=float(y_all[m].mean()) if m.any() else np.nan))
    pos = pd.DataFrame(pos)
    pos.to_csv(X.OUT / "skab_position_rate.csv", index=False)

    # chunk-relative time for the anomaly-free chunks (sensitivity)
    t_rel = t_all.copy()
    free = ~exp
    t_rel[free] = t_all[free] - meta[free].groupby("group_id").t_start_s.transform("min").to_numpy(float)

    stored = pd.read_csv(X.ROOT / "results_sens" / "shortcuts" / "shortcuts_per_split.csv")
    stored = stored[(stored.dataset == "skab") & (stored.shortcut == "a_t_start_tree3")].set_index("split").ba_test
    sp = X.splits("skab")
    rows, binq = [], {}
    for j in range(20):
        s = sp[j]
        mdl, thr = clock_tree(t_all, y_all, s)
        za = X.arrays("skab", "A", j)
        idx = za["test_idx"]
        assert np.array_equal(np.sort(idx), np.sort(s["test"]))
        k = list(mdl.classes_).index(1)
        pc = (mdl.predict_proba(t_all[idx].reshape(-1, 1))[:, k] >= thr).astype(int)
        y = za["y_test"].astype(int)
        assert np.array_equal(y, y_all[idx])
        ba_all = balanced_accuracy(y, pc)
        assert abs(ba_all - stored[j]) < 1e-9, (j, ba_all, stored[j])
        infile = exp[idx]
        bins = (t_all[idx] // BIN_S).astype(int)
        ok = [b for b in np.unique(bins[infile])
              if ((bins == b) & infile & (y == 0)).sum() >= MIN_PER_CLASS and ((bins == b) & infile & (y == 1)).sum() >= MIN_PER_CLASS]
        for b in ok:
            binq[b] = binq.get(b, 0) + 1
        rec = dict(split=j, clock_ba_all=ba_all, clock_ba_file=ba(y[infile], pc[infile]), n_bins=len(ok),
                   bins=" ".join(str(b) for b in ok))
        rec["clock_timestrat"] = float(np.mean([ba(y[(bins == b) & infile], pc[(bins == b) & infile]) for b in ok])) if ok else np.nan
        # chunk-relative variant (refit on relative time; sensitivity only)
        mr, thr_r = clock_tree(t_rel, y_all, s)
        kr = list(mr.classes_).index(1)
        pr = (mr.predict_proba(t_rel[idx].reshape(-1, 1))[:, kr] >= thr_r).astype(int)
        rec["clock_rel_ba_all"] = balanced_accuracy(y, pr)
        rec["clock_rel_ba_file"] = ba(y[infile], pr[infile])
        for pair in ("A", "B"):
            z = X.arrays("skab", pair, j)
            assert np.array_equal(z["test_idx"], idx)
            for sm, kk in MODELS.items():
                pred = (z[f"p_{kk}"] >= float(z[f"t_{kk}"])).astype(int)
                rec[f"{pair}_{sm}_timestrat"] = float(np.mean([ba(y[(bins == b) & infile], pred[(bins == b) & infile])
                                                              for b in ok])) if ok else np.nan
            rec[f"{pair}_cC_all"] = balanced_accuracy(y, (z["p_cf_cv"] >= float(z["t_cf_cv"])).astype(int))
            rec[f"{pair}_cC_minus_clock_timestrat"] = rec[f"{pair}_cC_timestrat"] - rec["clock_timestrat"]
        rows.append(rec)
    per = pd.DataFrame(rows)
    per.to_csv(X.OUT / "skab_clock_timestrat.csv", index=False)

    sz = X.sizes()
    summ = []

    def add(name, v, pair="A"):
        s = sz[(sz.dataset == "skab") & (sz.pair == pair)].iloc[0]
        t = X.S.nb_test(np.asarray(v, float), s.n_train, s.n_test)
        summ.append(dict(quantity=name, J=t["J"], mean=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"]))

    add("clock_ba_all", per.clock_ba_all)
    add("clock_ba_file", per.clock_ba_file)
    add("clock_rel_ba_all", per.clock_rel_ba_all)
    add("clock_rel_ba_file", per.clock_rel_ba_file)
    add("clock_timestrat", per.clock_timestrat)
    for pair in ("A", "B"):
        for sm in MODELS:
            add(f"{pair}_{sm}_timestrat", per[f"{pair}_{sm}_timestrat"], pair)
        add(f"{pair}_cC_minus_clock_timestrat", per[f"{pair}_cC_minus_clock_timestrat"], pair)
    bd = pd.read_csv(X.OUT / "skab_breakdown.csv")
    for pair in ("A", "B"):
        for q in ("file_info", "all_info", "timestrat_info"):
            r = bd[(bd.pair == pair) & (bd.quantity == q)].iloc[0]
            summ.append(dict(quantity=f"{pair}_{q}", J=r.J, mean=r["mean"], ci_lo=r.ci_lo, ci_hi=r.ci_hi))
    summ.append(dict(quantity="n_experiment_files", mean=n_files))
    summ.append(dict(quantity="n_files_first_pos_560_580", mean=int(g.first_pos.between(560, 580).sum())))
    summ.append(dict(quantity="n_files_first_pos_570", mean=int((g.first_pos == 570).sum())))
    summ.append(dict(quantity="n_files_last_pos_980_990", mean=int(g.last_pos.between(980, 990).sum())))
    summ.append(dict(quantity="n_files_last_pos_990", mean=int((g.last_pos == 990).sum())))
    for b in sorted(binq):
        summ.append(dict(quantity=f"bin{b}_{b*BIN_S}-{b*BIN_S+BIN_S-1}s_qualifying_splits", mean=binq[b]))
    out = pd.DataFrame(summ)
    out.to_csv(X.OUT / "skab_clock_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(g.sort_values("first_pos").to_string())
    print(pos.to_string())
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
