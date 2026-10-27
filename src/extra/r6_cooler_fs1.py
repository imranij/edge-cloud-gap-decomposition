"""POST-PLAN: the wired design (edge sensors plus FS1) under the
redesigned cooler hold-out. Reads results_sens/holdout2/hyd_cooler2_fs1/{A,B} (run by
src/sens/designs/run_custom_holdout.py, design hyd_cooler2_fs1; no model is fitted here) and the gateway
rows of results/extra/r5_gateway_holdout.csv, applies the decision rule fixed before these runs were
summarised (supplement, cooler hold-out with FS1), and writes results/extra/r6_cooler_fs1.csv and paper/tables/sup_r6_cooler_fs1.tex."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
rows = []
for pair in "AB":
    for f in sorted(glob.glob(str(ROOT / f"results_sens/holdout2/hyd_cooler2_fs1/{pair}/split_*.parquet"))):
        d = pd.read_parquet(f)
        z = np.load(f.replace("split_", "arrays_").replace(".parquet", ".npz"))
        val = lambda q: float(d[(d.quantity == q) & (d.router == "")].value.iloc[0])
        y = z["y_test"]
        rows.append(dict(pair=pair, fold=int(d.split.iloc[0]), held_out_cooler=str(d.held_out.iloc[0]),
                         ba_edge=val("ba_ef_ev"), ba_wired=val("ba_ef_cv"), ba_cloud_fs1=val("ba_cf_cv"),
                         auc_wired=roc_auc_score(y, z["p_ef_cv"]), auc_cloud_fs1=roc_auc_score(y, z["p_cf_cv"]),
                         ppr_wired=float((z["p_ef_cv"] >= z["t_ef_cv"]).mean())))
df = pd.DataFrame(rows)
gw = pd.read_csv(ROOT / "results/extra/r5_gateway_holdout.csv")
gw = gw[gw.split >= 0][["pair", "split", "gateway", "cloud"]].rename(columns={"split": "fold", "gateway": "ba_gateway_all",
                                                                               "cloud": "ba_cloud_gateway_all"})
df = df.merge(gw, on=["pair", "fold"], how="left")
out = []
for pair, g in df.groupby("pair"):
    m_w, m_g = g.ba_wired.mean(), g.ba_gateway_all.mean()
    rule = bool(m_w >= m_g - 1e-12 and g.ba_wired.min() >= 0.75)
    out.append(dict(pair=pair, mean_wired=m_w, mean_gateway_all=m_g, min_fold_wired=g.ba_wired.min(),
                    mean_cloud_fs1=g.ba_cloud_fs1.mean(), mean_cloud_gateway_all=g.ba_cloud_gateway_all.mean(),
                    rule_replace_limitation=rule))
summ = pd.DataFrame(out)
df.to_csv(ROOT / "results/extra/r6_cooler_fs1.csv", index=False)
summ.to_csv(ROOT / "results/extra/r6_cooler_fs1_summary.csv", index=False)
L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
     r"Pair & Held out & Edge & Wired & AUC wired & Gateway & Cloud, FS1 & Cloud gateway \\",
     r"\midrule"]
for _, r in df.sort_values(["pair", "fold"]).iterrows():
    L.append(f"{r.pair} & {r.held_out_cooler}\\% & {r.ba_edge:.3f} & {r.ba_wired:.3f} & {r.auc_wired:.3f} & "
             f"{r.ba_gateway_all:.3f} & {r.ba_cloud_fs1:.3f} & {r.ba_cloud_gateway_all:.3f} \\\\")
for _, s in summ.iterrows():
    L.append(f"{s.pair} & mean & -- & {s.mean_wired:.3f} & -- & {s.mean_gateway_all:.3f} & {s.mean_cloud_fs1:.3f} & "
             f"{s.mean_cloud_gateway_all:.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "paper/tables/sup_r6_cooler_fs1.tex").write_text("\n".join(L) + "\n")
print(df.round(3).to_string(index=False)); print(summ.round(4).to_string(index=False))
