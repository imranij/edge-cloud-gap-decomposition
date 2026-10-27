"""POST-PLAN summaries for area threewseu. Reads results_sens/threewseu and the
stored main run (results/); refits nothing. Writes results_sens/threewseu/summary/*.csv.

    python src/sens/threewseu/summarize.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import analyze as AN                                       # noqa: E402
from escalation import stats as S                          # noqa: E402

VAR = ROOT / "results_sens" / "threewseu"
OUT = VAR / "summary"
OUT.mkdir(parents=True, exist_ok=True)
MAINR = ROOT / "results"
MODELS = ["ef_ev", "cf_cv", "cf_ev", "ef_cv"]

KEYQ = [("ba_ef_ev", "", None), ("ba_cf_cv", "", None), ("ba_cf_ev", "", None), ("ba_ef_cv", "", None),
        ("total_step", "", None), ("model_step_ev", "", None), ("info_step_ef", "", None), ("model_step_cv", "", None), ("info_step_cf", "", None),
        ("shapley_model", "", None), ("shapley_info", "", None), ("interaction", "", None),
        ("ba_stacker_dep", "", None), ("ba_oracle_dep", "", None), ("unresolved_dep", "", None),
        ("self_oracle_headroom", "", None)] + \
       [("C", f"R{i}", None) for i in range(10)] + \
       [("capture", "R1", None), ("capture", "R6", None),
        ("ba_at_budget", "R1", 0.1), ("ba_at_budget", "R1", 0.3), ("ba_at_budget", "R6", 0.1),
        ("ba_at_budget", "R6", 0.3), ("rho_pos", "R1", None), ("audit_go", "R6", None), ("outcome", "R6", None)]


def parent_of(ds: str) -> str:
    base, _, by = ds.partition("__")
    p = base.split("_")[0]
    return f"{p}__{by}" if by else p


def load_main(parent: str) -> pd.DataFrame:
    files = sorted((MAINR / parent).glob("*/split_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[~df.quantity.isin(["tiny_settings", "skipped"])]


def sel(df, q, r, b):
    m = (df.quantity == q) & (df.router == r)
    m &= np.isclose(df.budget, b) if b is not None else df.budget.isna() | True
    return df[m].set_index("split")["value"]


def nbrow(v, sz):
    v = v.dropna()
    if len(v) == 0:
        return dict(J=0, mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    t = S.nb_test(v.to_numpy(float), sz["n_train"], sz["n_test"])
    return dict(J=t["J"], mean=t["mean"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], p=t["p"])


def main():
    df, skipped = AN.load_results(VAR)
    skipped.to_csv(OUT / "skipped.csv", index=False)
    main_rows = df[df.h2_q == 1.0]
    sizes = AN.nb_sizes(main_rows)
    roles = {d: "sensitivity" for d in main_rows.dataset.unique()}
    pdt = AN.per_dataset_table(main_rows, sizes, roles)
    pdt.to_csv(OUT / "per_variant.csv", index=False)
    sizes.to_csv(OUT / "sizes.csv", index=False)
    ref = pd.read_csv(MAINR / "summary" / "per_dataset.csv")

    rows = []
    for (ds, pair), d in main_rows.groupby(["dataset", "pair"]):
        par = parent_of(ds)
        sz = sizes[(sizes.dataset == ds) & (sizes.pair == pair)].iloc[0]
        mm = load_main(par)
        mm = mm[(mm.pair == pair) & (mm.h2_q == 1.0)]
        msz = AN.nb_sizes(mm).iloc[0] if len(mm) else None
        for q, r, b in KEYQ:
            v = sel(d, q, r, b)
            if v.empty:
                continue
            a = nbrow(v, sz)
            rr = ref[(ref.dataset == par) & (ref.pair == pair) & (ref.quantity == q) & (ref.router.fillna("") == r)]
            rr = rr[np.isclose(rr.budget, b)] if b is not None else rr[rr.budget.isna()]
            row = dict(variant=ds, pair=pair, quantity=q, router=r, budget=b, J=a["J"], mean=a["mean"],
                       ci_lo=a["ci_lo"], ci_hi=a["ci_hi"], p_two_sided=a["p"], parent=par)
            if len(rr):
                row.update(main_J=int(rr.J.iloc[0]), main_mean=rr["mean"].iloc[0], main_ci_lo=rr.ci_lo.iloc[0],
                           main_ci_hi=rr.ci_hi.iloc[0])
            if msz is not None:
                mv = sel(mm, q, r, b)
                common = v.index.intersection(mv.index)
                if len(common):
                    ms = nbrow(mv.loc[common], msz)
                    dd = nbrow(v.loc[common] - mv.loc[common], sz)
                    row.update(main_same_splits_J=ms["J"], main_same_splits_mean=ms["mean"],
                               main_same_ci_lo=ms["ci_lo"], main_same_ci_hi=ms["ci_hi"],
                               diff_mean=dd["mean"], diff_ci_lo=dd["ci_lo"], diff_ci_hi=dd["ci_hi"], diff_p=dd["p"])
            rows.append(row)
    key = pd.DataFrame(rows)
    key.to_csv(OUT / "key_quantities.csv", index=False)

    # H2 levels (pair A, C of R6) for the variants run with --h2, and the single-dataset slope
    h2rows = []
    for ds in sorted(df[df.h2_q != 1.0].dataset.unique()):
        res, by, m01 = AN.h2(df, [ds])
        for _, r in by.iterrows():
            h2rows.append(dict(variant=ds, h2_q=r.h2_q, J=int(r.J), mean_C_R6=r.mean_C, sd_C_R6=r.sd_C,
                               mean_npos=r.mean_npos, slope=res["slope"], slope_ci_lo=res["ci_lo"],
                               slope_ci_hi=res["ci_hi"], slope_model=res["model"]))
    pd.DataFrame(h2rows).to_csv(OUT / "h2_levels.csv", index=False)

    # per-dataset H1 / H3 / H4 / H5 contributions for the declared variants (descriptive, no pooling here)
    hp = []
    for ds in ("seu_dm", "threew_dm"):
        if ds not in set(main_rows.dataset):
            continue
        p1, _, _ = AN.h1(main_rows, sizes, [ds])
        p3, _, _ = AN.h3(main_rows, sizes, [ds])
        r4, _ = AN.h4(df, [ds])
        r5, _ = AN.h5(df, [ds], 10000, 10000, 9000)
        hp.append(dict(variant=ds, H1_shapley_info=p1["mean"].iloc[0], H1_ci_lo=p1.ci_lo.iloc[0],
                       H1_ci_hi=p1.ci_hi.iloc[0], H1_p_one_sided=p1.p_one_sided.iloc[0],
                       H3_dC_R7_R6=p3["mean"].iloc[0] if len(p3) else np.nan,
                       H3_ci_lo=p3.ci_lo.iloc[0] if len(p3) else np.nan, H3_ci_hi=p3.ci_hi.iloc[0] if len(p3) else np.nan,
                       H4_slope=r4["slope"], H4_ci_lo=r4["ci_lo"], H4_ci_hi=r4["ci_hi"], H4_model=r4["model"],
                       H5_n_cells=r5["n_cells"], H5_n_go=r5["n_go"], H5_n_outcome_pos=r5["n_outcome_pos"],
                       H5_ba=r5["ba"], H5_p_perm=r5["p_perm"], H5_note=r5.get("note", "")))
    pd.DataFrame(hp).to_csv(OUT / "declared_hypothesis_contributions.csv", index=False)

    # SEU healthy-recording specificity, as src/extra/seu_recordings.py does for the main run
    sp = []
    for ds in sorted(d for d in main_rows.dataset.unique() if d.startswith("seu")):
        for pair in ("A", "B"):
            for f in sorted((VAR / ds / pair).glob("arrays_*.npz")):
                if "_q" in f.stem:
                    continue
                z = np.load(f)
                y, grp = z["y_test"].astype(int), z["groups_test"]
                neg = np.unique(grp[y == 0])
                assert len(neg) == 1, (ds, pair, f, neg)
                for k in MODELS:
                    pred = (z[f"p_{k}"] >= float(z[f"t_{k}"])).astype(int)
                    sp.append(dict(variant=ds, pair=pair, split=int(f.stem.split("_")[1]), healthy=neg[0], model=k,
                                   specificity=float((pred[y == 0] == 0).mean()),
                                   recall=float((pred[y == 1] == 1).mean())))
    sp = pd.DataFrame(sp)
    if len(sp):
        sp.to_csv(OUT / "seu_specificity_per_split.csv", index=False)
        summ = sp.groupby(["variant", "pair", "model"]).agg(
            n_cases=("specificity", "size"), n_distinct_healthy=("healthy", "nunique"),
            share_correct=("specificity", lambda v: float((v >= 0.5).mean())),
            share_extreme=("specificity", lambda v: float(((v <= 0.1) | (v >= 0.9)).mean())),
            mean_specificity=("specificity", "mean"), mean_recall=("recall", "mean")).reset_index()
        per_rec = sp.groupby(["variant", "pair", "model", "healthy"]).agg(
            n=("specificity", "size"), mean_spec=("specificity", "mean"),
            share_correct=("specificity", lambda v: float((v >= 0.5).mean()))).reset_index()
        rc = per_rec.groupby(["variant", "pair", "model"]).agg(
            recordings_mostly_correct=("share_correct", lambda v: int((v > 0.5).sum()))).reset_index()
        summ = summ.merge(rc, on=["variant", "pair", "model"])
        summ.to_csv(OUT / "seu_specificity.csv", index=False)
        per_rec.to_csv(OUT / "seu_specificity_by_recording.csv", index=False)

    # 3W: recall per fault type and specificity on the test windows (pooled over splits), plus fold composition
    meta = pd.read_parquet(ROOT / "data" / "processed" / "threew" / "meta.parquet")
    ft, comp = [], []
    for ds in sorted(d for d in main_rows.dataset.unique() if d.startswith("threew")):
        for pair in ("A", "B"):
            for f in sorted((VAR / ds / pair).glob("arrays_*.npz")):
                if "_q" in f.stem:
                    continue
                z = np.load(f)
                ti = z["test_idx"]; y = z["y_test"].astype(int)
                m = meta.iloc[ti]
                comp.append(dict(variant=ds, pair=pair, split=int(f.stem.split("_")[1]), n_test=len(ti),
                                 test_wells=m.well.nunique(), test_wells_with_pos=m[y == 1].well.nunique(),
                                 test_wells_with_neg=m[y == 0].well.nunique(), test_episodes=m.episode_id.nunique()))
                for k in MODELS:
                    pred = (z[f"p_{k}"] >= float(z[f"t_{k}"])).astype(int)
                    for t, g in m.assign(pred=pred, y=y).groupby("fault_type"):
                        ft.append(dict(variant=ds, pair=pair, split=int(f.stem.split("_")[1]), model=k, fault_type=t,
                                       n=len(g), correct=int((g.pred == g.y).sum())))
    if ft:
        ft = pd.DataFrame(ft)
        agg = ft.groupby(["variant", "pair", "model", "fault_type"]).agg(n=("n", "sum"), correct=("correct", "sum"),
                                                                         splits=("split", "nunique")).reset_index()
        agg["share_correct"] = agg.correct / agg.n
        agg.to_csv(OUT / "threew_by_fault_type.csv", index=False)
        pd.DataFrame(comp).to_csv(OUT / "threew_fold_composition.csv", index=False)
    print(key[key.quantity.isin(["ba_ef_ev", "ba_cf_cv", "shapley_model", "shapley_info"]) |
              ((key.quantity == "C") & key.router.isin(["R1", "R6"]))]
          [["variant", "pair", "quantity", "router", "J", "mean", "ci_lo", "ci_hi", "main_mean",
            "main_same_splits_mean", "diff_mean", "diff_ci_lo", "diff_ci_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
