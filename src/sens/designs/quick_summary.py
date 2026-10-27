"""Quick NB-corrected summary of key quantities for post-plan variant runs (helper)."""
import sys, glob
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from escalation import stats as S

def summ(path):
    fs = [f for f in sorted(glob.glob(f"{path}/split_*.parquet")) if "_q" not in f]
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs]); per = d.drop_duplicates("split")
    ntr, nte = per.n_train.mean(), per.n_test.mean(); out = {"J": len(fs)}
    for q in ["ba_ef_ev", "ba_cf_cv", "shapley_info", "shapley_model"]:
        v = d[(d.quantity == q) & (d.router == "")].sort_values("split").value.to_numpy()
        t = S.nb_test(v, ntr, nte); out[q] = f"{t['mean']:.3f} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]"
    for r in ["R1", "R6"]:
        v = d[(d.quantity == "C") & (d.router == r)].sort_values("split").value.to_numpy()
        out[f"C_{r}"] = f"{S.nb_test(v, ntr, nte)['mean']:.3f}"
    return out

if __name__ == "__main__":
    rows = []
    for p in sys.argv[1:]:
        r = summ(p)
        if r: rows.append(dict(run=p.split("results_sens/")[-1], **r))
    print(pd.DataFrame(rows).to_string(index=False))
