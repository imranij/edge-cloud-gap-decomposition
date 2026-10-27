"""Confirmatory 3W subset (decided before any model run; see plan/DEVIATIONS.md).
Keeps instances whose five cloud channels are all usable (>= 90% present and not frozen over
the labelled period), and groups windows by episode (chains of back-to-back instances of one
well), so near-continuous recordings never straddle folds. The full build stays in
data/processed/threew_all/."""
import json, shutil, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]
ALL, OUT = ROOT / "data/processed/threew_all", ROOT / "data/processed/threew"
if not ALL.exists():
    shutil.move(str(OUT), str(ALL))
OUT.mkdir(parents=True, exist_ok=True)
CH = json.load(open(ALL / "channels.json"))["cloud"]
cov = pd.read_csv(ALL / "instance_coverage.csv")
ok = set(cov.loc[(cov[CH] >= 0.9).all(axis=1), "file"].str.replace(".parquet", "", regex=False))
m = pd.read_parquet(ALL / "meta.parquet")
keep = m.group_id.isin(ok).to_numpy()
idx = np.flatnonzero(keep)
mm = m.iloc[idx].copy()
mm["instance"] = mm["group_id"]
mm["group_id"] = mm["episode_id"]
mm = mm.sort_values(["group_id", "t_start"]).copy()
order_idx = mm.index.to_numpy()                       # positions in the full arrays
mm["order"] = mm.groupby("group_id").cumcount()
mm["window_id"] = np.arange(len(mm))
mm.reset_index(drop=True).to_parquet(OUT / "meta.parquet")
for f in ["X_edge.npy", "X_cloud.npy", "raw_edge.npy", "raw_cloud.npy"]:
    A = np.load(ALL / f, mmap_mode="r")
    np.save(OUT / f, np.asarray(A[order_idx]))
for f in ["names_edge.json", "names_cloud.json", "channels.json", "bytes.json", "card.md", "sha256.txt",
          "missingness.csv", "instances.csv", "instance_coverage.csv", "summary.json"]:
    if (ALL / f).exists():
        shutil.copy(ALL / f, OUT / f)
with open(OUT / "card.md", "a") as fh:
    fh.write("\n\n## Confirmatory subset (fixed before any model run)\n\n"
             f"Instances with all five cloud channels usable (>= 90% present, not frozen): {len(ok)} of 1,119. "
             f"Windows: {len(mm)}; prevalence {mm.label.mean():.3f}; groups = episodes: {mm.group_id.nunique()} "
             f"({mm[mm.label==1].group_id.nunique()} with positives, {mm[mm.label==0].group_id.nunique()} with negatives); "
             f"wells: {mm.well.nunique()}. Reason: the build imputed whole-instance missing channels with the median, and "
             "missingness differed by label, so cloud-only channels could carry an imputation shortcut. Grouping by episode "
             "keeps back-to-back recordings of one well in one fold.\n")
print(len(mm), mm.group_id.nunique(), round(mm.label.mean(), 3), mm.well.nunique())
