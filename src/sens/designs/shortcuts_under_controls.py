"""POST-PLAN: the shortcut baselines re-scored under the control
splits. For every hydraulic control variant (and the 3W de-meaned variant under episode and by-well
grouping) the pair-A splits of that variant are rebuilt with escalation.splits.make_splits (the same
call run_dataset.py makes), checked against the stored test indices of the variant run, and the
shortcut predictors of src/sens/shortcuts/run_shortcuts.py are refitted on them. The variant's own
edge and cloud balanced accuracies (pair A and B) are placed next to each shortcut.
Writes results_sens/shortcuts_controls/{per_split.csv, summary.csv}.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "sens")); sys.path.insert(0, str(ROOT / "src" / "sens" / "shortcuts"))
from escalation.config import PLAN                      # noqa: E402
from escalation.data import load_dataset                # noqa: E402
from escalation.splits import make_splits, inner_split  # noqa: E402
from escalation.stats import nb_test                    # noqa: E402
import run_shortcuts as RS                              # noqa: E402

OUT = ROOT / "results_sens" / "shortcuts_controls"
OUT.mkdir(parents=True, exist_ok=True)
# (variant, parent for the shortcut specs, group_col, result area)
JOBS = [("hyd_blocks", "hydraulic", None, "hydskab"), ("hyd_runs", "hydraulic", None, "hydskab"),
        ("hyd_stable", "hydraulic", None, "hydskab"), ("hyd_weak", "hydraulic", None, "hydskab"),
        ("hyd_severe", "hydraulic", None, "hydskab"),
        ("threew_dm", "threew", None, "threewseu"), ("threew_dm", "threew", "well", "threewseu")]


HOLDOUT2 = {"skab_lso2": ("skab", "session"), "hyd_cooler2": ("hydraulic", "cooler"),
            "hyd_accum2": ("hydraulic", "accumulator"), "skab_lso2_base": ("skab", "session")}
H2ROOT = {"skab_lso2_base": ("skab_base", "data/processed_sens")}


def holdout2_splits(ds, unit_col):
    """The redesigned hold-out splits of src/sens/designs/run_custom_holdout.py."""
    units = ds["meta"][unit_col].astype(str).to_numpy()
    out = []
    for j, u in enumerate(sorted(np.unique(units))):
        test = np.flatnonzero(units == u); train = np.flatnonzero(units != u)
        fit, val, K_in, reason = inner_split(ds["y"], ds["groups"], train, j, PLAN)
        out.append(dict(split_id=j, fit=fit, val=val, test=np.sort(test), skip=reason))
    return out


def model_ba(area, label, pair):
    d = ROOT / "results_sens" / area / label / pair
    fs = sorted(p for p in d.glob("split_*.parquet") if "_q" not in p.name)
    if not fs:
        return {}
    df = pd.concat([pd.read_parquet(f) for f in fs])
    per = df.drop_duplicates("split")
    out = {}
    for q, nm in (("ba_ef_ev", "edge"), ("ba_cf_cv", "cloud")):
        v = df[(df.quantity == q) & (df.router == "")].sort_values("split")["value"].to_numpy()
        r = nb_test(v, per.n_train.mean(), per.n_test.mean())
        out.update({f"{nm}_{pair}": r["mean"], f"{nm}_{pair}_lo": r["ci_lo"], f"{nm}_{pair}_hi": r["ci_hi"],
                    f"J_{pair}": r["J"]})
    return out


def main():
    rows = []
    for design, (parent, unit_col) in HOLDOUT2.items():
        src_name, src_root = H2ROOT.get(design, (parent, "data/processed"))
        ds = load_dataset(src_name, ROOT / src_root)
        y = ds["y"]; specs = RS.build_specs(parent, ds); n = 0
        for sp in holdout2_splits(ds, unit_col):
            fa = ROOT / "results_sens" / "holdout2" / design / "A" / f"arrays_{sp['split_id']}.npz"
            fb = ROOT / "results_sens" / "holdout2" / design / "B" / f"arrays_{sp['split_id']}.npz"
            fx = fa if fa.exists() else fb
            if sp["skip"] or not fx.exists():
                continue
            with np.load(fx) as z:
                assert np.array_equal(z["test_idx"], sp["test"]), (design, sp["split_id"])
            n += 1
            for key, ba, bav, t in RS.run_one(parent, specs, y, sp):
                rows.append(dict(variant=design, shortcut=key, split=sp["split_id"], n_fit=len(sp["fit"]),
                                 n_val=len(sp["val"]), n_test=len(sp["test"]), ba_test=ba, ba_val=bav, threshold=t))
        print(design, "folds verified and scored:", n, flush=True)
    for var, parent, gcol, area in JOBS:
        ds = load_dataset(var, ROOT / "data" / "processed_sens", gcol)
        label = f"{var}__by_{gcol}" if gcol else var
        y = ds["y"]
        specs = RS.build_specs(parent, ds)
        sps = make_splits(y, ds["groups"], "tabular", PLAN)
        stored = ROOT / "results_sens" / area / label / "A"
        n = 0
        for sp in sps:
            fa = stored / f"arrays_{sp['split_id']}.npz"
            if sp["skip"] or not fa.exists():
                continue
            with np.load(fa) as z:
                assert np.array_equal(z["test_idx"], sp["test"]), (label, sp["split_id"])
            n += 1
            for key, ba, bav, t in RS.run_one(parent, specs, y, sp):
                rows.append(dict(variant=label, shortcut=key, split=sp["split_id"], n_fit=len(sp["fit"]),
                                 n_val=len(sp["val"]), n_test=len(sp["test"]), ba_test=ba, ba_val=bav, threshold=t))
        print(label, "splits verified and scored:", n, flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_split.csv", index=False)
    out = []
    for (label, key), g in per.groupby(["variant", "shortcut"], sort=False):
        r = nb_test(g.ba_test.to_numpy(), float((g.n_fit + g.n_val).mean()), float(g.n_test.mean()))
        if label in HOLDOUT2:
            area = "holdout2"
        else:
            area = "hydskab" if label.startswith("hyd") else "threewseu"
        row = dict(variant=label, shortcut=key, J=r["J"], ba_mean=r["mean"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                   per_fold=";".join(f"{v:.3f}" for v in g.sort_values("split").ba_test))
        for pair in "AB":
            row.update(model_ba(area, label, pair))
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(OUT / "summary.csv", index=False)
    cols = [c for c in ["variant", "shortcut", "J", "ba_mean", "ci_lo", "ci_hi", "edge_A", "cloud_A", "edge_B", "cloud_B"] if c in s]
    print(s[cols].round(3).to_string())


if __name__ == "__main__":
    main()
