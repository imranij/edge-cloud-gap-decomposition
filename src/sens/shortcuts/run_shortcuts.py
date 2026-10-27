"""Shortcut baselines under the paper's own pair-A splits.
Trivial predictors fit on the fit set, threshold by escalation.metrics.threshold_max_ba on the validation
set, balanced accuracy on the test fold. Nothing here changes the confirmatory pipeline."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "sens"))
import common as C
from escalation.metrics import threshold_max_ba, balanced_accuracy
from escalation.stats import nb_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")
OUT = ROOT / "results_sens" / "shortcuts"

def tree(): return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
def logreg(): return LogisticRegression(class_weight="balanced", max_iter=2000)

def sk_proba(model, Xf, yf, Xo):
    model.fit(Xf, yf)
    return model.predict_proba(Xo)[:, list(model.classes_).index(1)]

def well_rate(wf, yf, wo):
    s = pd.Series(yf).groupby(np.asarray(wf)).mean()
    return np.array([s.get(w, yf.mean()) for w in wo], float)

def build_specs(name, ds):
    m = ds["meta"]; names = ds["names_cloud"]; Xc = ds["X_cloud"]
    def col(c): return m[c].to_numpy(float).reshape(-1, 1)
    ix = lambda cols: Xc[:, [names.index(c) for c in cols]].astype(float)
    S = []
    if name == "hydraulic":
        S += [("a_cycle_1nn", col("cycle"), lambda: KNeighborsClassifier(n_neighbors=1)),
              ("b_cycle_tree3", col("cycle"), tree),
              ("c_settings_tree3", m[["cooler", "valve", "accumulator"]].to_numpy(float), tree)]
    elif name == "skab":
        sess = OneHotEncoder(sparse_output=False).fit_transform(m[["session"]])
        S += [("a_t_start_tree3", col("t_start_s"), tree),
              ("b_order_tree3", col("order"), tree),
              ("c_session_logreg", sess, logreg)]
    elif name == "seu":
        e2 = ["motor_vib__mean", "motor_torque__mean"]
        eight = [c for c in names if c.endswith("__mean")]
        assert len(eight) == 8, eight
        S += [("a_edge2_means_logreg", ix(e2), lambda: make_pipeline(StandardScaler(), logreg())),
              ("b_cloud8_means_logreg", ix(eight), lambda: make_pipeline(StandardScaler(), logreg()))]
    elif name == "threew":
        wl = m["well"].astype(str).to_numpy()
        S += [("a_well_rate", wl, "wellrate"),
              ("b_well_onehot_logreg", wl.reshape(-1, 1), "wellonehot")]
    return S

def run_one(name, specs, y, sp):
    fit, val, test = sp["fit"], sp["val"], sp["test"]
    rows = []
    for key, X, mk in specs:
        if mk == "wellrate":
            pv = well_rate(X[fit], y[fit], X[val]); pt = well_rate(X[fit], y[fit], X[test])
        elif mk == "wellonehot":
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[fit])
            f = lambda i: enc.transform(X[i])
            mdl = logreg(); mdl.fit(f(fit), y[fit])
            g = lambda i: mdl.predict_proba(f(i))[:, list(mdl.classes_).index(1)]
            pv, pt = g(val), g(test)
        else:
            mdl = mk(); mdl.fit(X[fit], y[fit])
            k = list(mdl.classes_).index(1)
            pv, pt = mdl.predict_proba(X[val])[:, k], mdl.predict_proba(X[test])[:, k]
        t = threshold_max_ba(pv, y[val])
        rows.append((key, balanced_accuracy(y[test], (pt >= t).astype(int)), balanced_accuracy(y[val], (pv >= t).astype(int)), t))
    return rows

def main():
    allrows = []
    jobs = [(n, None, n) for n in C.DATASETS] + [("threew", "well", "threew__by_well")]
    for name, gcol, label in jobs:
        ds = C.dataset(name, "data/processed", gcol)
        y = ds["y"]
        specs = build_specs(name, ds)
        if gcol == "well":
            specs = [s for s in specs]  # same two shortcuts, under by-well splits
        ids = C.stored_split_ids(name, "A", None, gcol)
        n_ok = 0
        for sid in ids:
            sp = C.reconstruct_split(name, "A", sid, None, gcol)
            if sp is None: continue
            n_ok += 1
            for key, ba, bav, t in run_one(name, specs, y, sp):
                allrows.append(dict(dataset=label, shortcut=key, split=sid, repeat=sp["repeat"], fold=sp["fold"],
                                    n_fit=len(sp["fit"]), n_val=len(sp["val"]), n_test=len(sp["test"]),
                                    ba_test=ba, ba_val=bav, threshold=t))
        print(label, "splits", n_ok, flush=True)
    per = pd.DataFrame(allrows)
    per.to_csv(OUT / "shortcuts_per_split.csv", index=False)
    pdset = pd.read_csv(ROOT / "results/summary/per_dataset.csv")
    def ref(ds, pair, q):
        r = pdset[(pdset.dataset == ds) & (pdset.pair == pair) & (pdset.quantity == q) & pdset.router.isna()]
        return (r["mean"].iloc[0], r["ci_lo"].iloc[0], r["ci_hi"].iloc[0]) if len(r) else (np.nan,) * 3
    out = []
    for (ds, key), g in per.groupby(["dataset", "shortcut"], sort=False):
        ntr = float((g.n_fit + g.n_val).mean()); nte = float(g.n_test.mean())
        r = nb_test(g.ba_test.to_numpy(), ntr, nte)
        row = dict(dataset=ds, shortcut=key, J=r["J"], ba_mean=r["mean"], sd=r["sd"], se_nb=r["se"],
                   ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n_train=ntr, n_test=nte, ba_val_mean=g.ba_val.mean())
        for pair in "AB":
            for q, nm in (("ba_ef_ev", "edge"), ("ba_cf_cv", "cloud")):
                m_, lo, hi = ref(ds, pair, q)
                row.update({f"{nm}_{pair}": m_, f"{nm}_{pair}_lo": lo, f"{nm}_{pair}_hi": hi})
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / "shortcuts.csv", index=False)
    print(pd.DataFrame(out)[["dataset", "shortcut", "J", "ba_mean", "ci_lo", "ci_hi", "edge_A", "cloud_A", "edge_B", "cloud_B"]].round(3).to_string())

if __name__ == "__main__":
    main()
