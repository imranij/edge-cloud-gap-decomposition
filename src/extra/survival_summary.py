"""POST-PLAN, evaluation only. Refits no model.

One row per dataset applying the survival criterion stated in Section 5.2 of the paper, under the declared
deconfounded variant (plan/postplan/DECONF_DECLARATION.md):
  (i)  the Shapley information step's corrected 95% CI excludes zero for both pairs A and B, and
  (ii) the deployed cloud beats the best metadata shortcut baseline scored on the same splits, i.e. the
       corrected 95% CI of the per-split difference B_cC - BA(shortcut) excludes zero, for both pairs.
The step "survives" when (i) and (ii) hold for both pairs; otherwise the row says which part fails for
which pair.

The STRONGER form of (ii), (ii+), where the shortcut-augmented runs
exist (summarised by src/extra/round3_variants.py -> results/extra/round3_variants.csv):
  (ii+) the deployed cloud beats an edge given the shortcut variable as extra features, on the same splits
        (corrected 95% CI of the per-split difference excludes zero), for both pairs.
Hydraulic: hyd_blocks_sc2 (cycle index and component settings added to both views; block-grouped splits, J = 20),
against the hyd_blocks cloud. SKAB: skab_sc (time in file added to both views; pre-specified features and splits,
first 10 splits), against the pre-specified cloud on the same splits, since no augmented run exists for the
declared z-scored variant. The information step inside the augmented run (what the cloud-only sensors add beyond
the edge sensors AND the shortcut) is reported beside it. 3W and SEU have no augmented run: 3W already fails (i)
and the weaker (ii), and SEU has no metadata shortcut, so (ii+) cannot change either verdict.
verdict_strong applies (i), (ii) and (ii+); it equals verdict where no augmented run exists.

Two refinements:
  * the criterion is evaluated and reported PER PAIR (columns met_A, met_B; verdict_pairs). A dataset meets it for
    both pairs only if met_A and met_B.
  * (ii+) is REDEFINED: the deployed cloud must beat the BETTER of the augmented edge and the shortcut alone, per
    split (max of the two balanced accuracies on that split), paired over splits with the corrected CI
    (columns cloud_minus_best_{p}, cmb_{p}_lo/hi, crit_ii_best_{p}). Reason: the pair A edge is a logistic
    regression, which cannot use non-monotone shortcuts, so "cloud beats the augmented edge" alone was passed by
    SKAB pair A although the shortcut alone beat the cloud. The earlier (ii+) columns are kept for reference.
  * a dataset whose best shortcut beats its cloud with an interval excluding zero for both pairs is labelled
    "benchmark-confounded, not testable": its labels can be separated from its collection protocol better than by
    any model, so condition (ii) cannot say whether the cloud-only sensors carry information.
  * an extra, non-declared row for the causal SKAB offset variant (skab_base), whose test folds equal the
    pre-specified ones, and the pre-specified SKAB cloud minus the shortcut (m4) as columns of the SKAB rows.

Sources: paper/tables/data/pp_variants.csv (information step; postplan_summary.py),
results_sens/{hydskab,threewseu}/<variant>/<pair>/split_*.parquet (cloud BA per split),
results_sens/shortcuts/shortcuts_per_split.csv (pre-specified splits; the SKAB and SEU declared variants
change only features, and their test folds equal the pre-specified ones, checked here),
results_sens/shortcuts_controls/per_split.csv (shortcuts re-scored on the hyd_blocks and threew_dm splits).
Output: results/extra/survival_summary.csv
    python src/extra/survival_summary.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as X

RSS = X.ROOT / "results_sens"
DECL = {"hydraulic": ("hydskab", "hyd_blocks", "controls"), "skab": ("hydskab", "skab_filez", "main"),
        "seu": ("threewseu", "seu_dm", "main"), "threew": ("threewseu", "threew_dm", "controls"),
        "skab_offset": ("hydskab", "skab_base", "main")}      # extra row, not the declared variant
DSKEY = {"skab_offset": "skab"}
AUG = {"hydraulic": "hyd_blocks_sc2", "skab": "skab_sc"}
AUG_REF = {"hydraulic": ("hydskab", "hyd_blocks"), "skab": None}   # None: pre-specified run results/skab


def edge_per_split_aug(av, pair):
    d = RSS / "round3" / av / pair
    fs = sorted(f for f in d.glob("split_*.parquet") if "_q" not in f.stem)
    x = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    x = x[(x.h2_q == 1.0) & (x.quantity == "ba_ef_ev") & (x.router.fillna("") == "") & x.budget.isna()]
    return x.set_index("split").value.sort_index()


def ref_cloud_per_split(ds, pair):
    if AUG_REF.get(ds):
        area, v = AUG_REF[ds]
        return cloud_per_split(area, v, pair)
    fs = sorted(f for f in (X.RES / "skab" / pair).glob("split_*.parquet") if "_q" not in f.stem)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[(d.h2_q == 1.0) & (d.quantity == "ba_cf_cv") & (d.router.fillna("") == "") & d.budget.isna()]
    return d.set_index("split").value.sort_index(), float(d.n_train.mean()), float(d.n_test.mean())


def cloud_per_split(area, v, pair):
    fs = sorted(f for f in (RSS / area / v / pair).glob("split_*.parquet") if "_q" not in f.stem)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[(d.h2_q == 1.0) & (d.quantity == "ba_cf_cv") & (d.router.fillna("") == "") & d.budget.isna()]
    return d.set_index("split").value.sort_index(), float(d.n_train.mean()), float(d.n_test.mean())


def main():
    pv = pd.read_csv(X.ROOT / "paper" / "tables" / "data" / "pp_variants.csv")
    pv["router"] = pv["router"].fillna("")
    sc_main = pd.read_csv(RSS / "shortcuts" / "shortcuts_per_split.csv")
    sc_ctrl = pd.read_csv(RSS / "shortcuts_controls" / "per_split.csv")
    rows = []
    for ds_row, (area, v, src) in DECL.items():
        ds = DSKEY.get(ds_row, ds_row)
        if src == "main":
            for pair in ("A", "B"):
                for j in range(20):
                    same = np.array_equal(np.load(RSS / area / v / pair / f"arrays_{j}.npz")["test_idx"],
                                          X.arrays(ds, pair, j)["test_idx"])
                    assert same, (v, pair, j)
            sc = sc_main[sc_main.dataset == ds]
        else:
            sc = sc_ctrl[sc_ctrl.variant == v]
        best = sc.groupby("shortcut").ba_test.mean().idxmax()
        scb = sc[sc.shortcut == best].set_index("split").ba_test.sort_index()
        rec = dict(dataset=ds_row, variant=v, best_shortcut=best, shortcut_ba=scb.mean())
        ok_i, ok_ii = {}, {}
        for pair in ("A", "B"):
            r = pv[(pv.dataset == v) & (pv.pair == pair) & (pv.quantity == "shapley_info") & (pv.router == "")
                   & pv.budget.isna()].iloc[0]
            rec[f"info_{pair}"], rec[f"info_{pair}_lo"], rec[f"info_{pair}_hi"] = r["mean"], r.ci_lo, r.ci_hi
            ok_i[pair] = bool(r.ci_lo > 0)
            cl, ntr, nte = cloud_per_split(area, v, pair)
            diff = (cl - scb.reindex(cl.index)).to_numpy(float)
            t = X.S.nb_test(diff, ntr, nte)
            rec[f"cloud_{pair}"] = cl.mean()
            rec[f"cloud_minus_shortcut_{pair}"], rec[f"cms_{pair}_lo"], rec[f"cms_{pair}_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
            ok_ii[pair] = bool(t["ci_lo"] > 0)
        rec.update({f"crit_i_{p}": ok_i[p] for p in "AB"}, **{f"crit_ii_{p}": ok_ii[p] for p in "AB"})
        both = {p: ok_i[p] and ok_ii[p] for p in "AB"}
        if all(both.values()):
            verdict = "survives"
        elif any(both.values()):
            verdict = "pair " + "".join(p for p in "AB" if both[p]) + " only"
        else:
            verdict = "does not survive"
        rec["verdict"] = verdict
        confounded = all(rec[f"cms_{p}_hi"] < 0 for p in "AB")
        rec["benchmark_confounded"] = confounded
        if ds == "skab":   # m4: the pre-specified cloud minus the shortcut, same splits
            for pair in ("A", "B"):
                cl0, ntr0, nte0 = ref_cloud_per_split("skab", pair)
                t = X.S.nb_test((cl0 - scb.reindex(cl0.index)).to_numpy(float), ntr0, nte0)
                rec[f"prespec_cloud_minus_shortcut_{pair}"] = t["mean"]
                rec[f"pcms_{pair}_lo"], rec[f"pcms_{pair}_hi"] = t["ci_lo"], t["ci_hi"]
        # (ii+) stronger form, against the shortcut-augmented edge
        r3 = pd.read_csv(X.OUT / "round3_variants.csv")
        av = AUG.get(ds) if ds_row == ds else None
        rec["aug_variant"] = av if av else ""
        ok_s = {}
        for pair in ("A", "B"):
            g = r3[(r3.variant == av) & (r3.pair == pair)] if av else r3.iloc[0:0]
            if len(g) and bool(g.complete.iloc[0]):
                g = g.iloc[0]
                rec[f"aug_complete_{pair}"] = True
                rec[f"aug_J_{pair}"] = int(g.J)
                rec[f"cloud_minus_aug_edge_{pair}"] = g.ref_cloud_minus_aug_edge
                rec[f"cmae_{pair}_lo"], rec[f"cmae_{pair}_hi"] = g.rcae_lo, g.rcae_hi
                rec[f"info_aug_{pair}"], rec[f"info_aug_{pair}_lo"], rec[f"info_aug_{pair}_hi"] = g["info"], g.info_lo, g.info_hi
                # cloud minus the better (per split) of the augmented edge and the shortcut alone
                cl, ntr_r, nte_r = ref_cloud_per_split(ds, pair)
                ae = edge_per_split_aug(av, pair)
                spl = ae.index
                best_comp = np.maximum(ae.to_numpy(float), scb.reindex(spl).to_numpy(float))
                t = X.S.nb_test(cl.reindex(spl).to_numpy(float) - best_comp, ntr_r, nte_r)
                rec[f"cloud_minus_best_{pair}"], rec[f"cmb_{pair}_lo"], rec[f"cmb_{pair}_hi"] = t["mean"], t["ci_lo"], t["ci_hi"]
                rec[f"aug_edge_{pair}"] = float(ae.mean())
                rec[f"shortcut_same_splits_{pair}"] = float(scb.reindex(spl).mean())
                rec[f"share_splits_shortcut_better_{pair}"] = float((scb.reindex(spl).to_numpy(float) > ae.to_numpy(float)).mean())
                rec[f"crit_ii_aug_edge_only_{pair}"] = bool(g.rcae_lo > 0)
                ok_s[pair] = bool(t["ci_lo"] > 0)
                rec[f"crit_ii_best_{pair}"] = ok_s[pair]
            else:
                rec[f"aug_complete_{pair}"] = False if av else np.nan
                ok_s[pair] = None
            rec[f"crit_ii_strong_{pair}"] = ok_s[pair]
        for p in "AB":
            met = ok_i[p] and ok_ii[p] and (ok_s[p] if av else True)
            rec[f"met_{p}"] = met
            fails = [n for n, ok in (("(i)", ok_i[p]), ("(ii)", ok_ii[p]), ("(ii+)", ok_s[p] if av else True)) if not ok]
            rec[f"fails_{p}"] = " ".join(fails)
        if confounded:
            rec["verdict_pairs"] = "benchmark-confounded, not testable"
        elif rec["met_A"] and rec["met_B"]:
            rec["verdict_pairs"] = "met for both pairs"
        elif rec["met_A"] or rec["met_B"]:
            pm = "A" if rec["met_A"] else "B"; pf = "B" if pm == "A" else "A"
            rec["verdict_pairs"] = f"pair {pm} met; pair {pf} failed {rec['fails_' + pf]}"
        else:
            rec["verdict_pairs"] = "not met for either pair"
        if not av:
            rec["verdict_strong"] = verdict
        elif any(v is None for v in ok_s.values()):
            rec["verdict_strong"] = "pending"
        else:
            both_s = {p: ok_i[p] and ok_ii[p] and ok_s[p] for p in "AB"}
            rec["verdict_strong"] = ("survives" if all(both_s.values()) else
                                     "pair " + "".join(p for p in "AB" if both_s[p]) + " only" if any(both_s.values())
                                     else "does not survive")
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(X.OUT / "survival_summary.csv", index=False)
    pd.set_option("display.width", 300)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    main()
