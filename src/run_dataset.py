"""Per-dataset entry point of the frozen plan (Section 10).

    python src/run_dataset.py <dataset> <pair> [--h2] [--splits a:b] [--root data/processed]
                              [--out results] [--group-col COL] [--tiny]

Writes, per split, results/<dataset>/<pair>/split_<id>[_q<q>].parquet (long format) and
arrays_<id>[_q<q>].npz (test-set arrays), skipping splits already written (resume). --h2 runs the
H2 subsampling levels q in {0.1, 0.25, 0.5} for pair A (q = 1.0 is the main run).
--group-col replaces group_id by another meta column (sensitivity analyses, e.g. 3W by well);
results then go to results/<dataset>__by_<col>/. --tiny is for the test suite only: it may not
write under <repo>/results, and every split it writes carries a `tiny_settings` row = 1 that makes
analyze.py refuse the results. Every file is written to a temporary name and renamed into place;
the parquet is renamed last, so its existence marks a completed split.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from escalation import metrics as M                       # noqa: E402
from escalation import routers as RT                      # noqa: E402
from escalation.config import PLAN, tiny, BUDGETS, CAPTURE_RANGE, H2_LEVELS, EXPLANATION_TOKEN_BYTES  # noqa: E402
from escalation.data import load_dataset, cap_windows     # noqa: E402
from escalation.splits import make_splits, h2_subsample, check_split  # noqa: E402
from escalation.models import make_model, footprint, PAIR_KIND, FAMILY_NAMES  # noqa: E402
from escalation.decompose import (factorial_ba, factorial_steps, threshold_composite, combination,  # noqa: E402
                                  self_oracle_headroom, KEYS)
from escalation.audit import run_audit                    # noqa: E402

FAM_VIEW = {"ef_ev": ("edge", "edge"), "cf_ev": ("cloud", "edge"), "ef_cv": ("edge", "cloud"),
            "cf_cv": ("cloud", "cloud")}


def recall_at_budget(y, pred_edge, pred_cloud, s, f: float) -> float:
    """System recall when the top round(f n) windows by s are escalated (ties in expectation)."""
    y = np.asarray(y).astype(int)
    pos = (y == 1).astype(float) / max(int(y.sum()), 1)
    r = pos * ((np.asarray(pred_cloud) == y).astype(float) - (np.asarray(pred_edge) == y).astype(float))
    k = int(round(f * len(y)))
    return float(np.mean(np.asarray(pred_edge)[y == 1] == 1) + M.gain_curve(r, s)[k])


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def view_arrays(ds, pair):
    if PAIR_KIND[pair] == "cnn":
        if ds["raw_edge"] is None:
            raise SystemExit(f"{ds['name']}: pair C needs raw_edge.npy/raw_cloud.npy")
        return {"edge": ds["raw_edge"], "cloud": ds["raw_cloud"]}
    return {"edge": ds["X_edge"], "cloud": ds["X_cloud"]}


def take(a, ix):
    """Rows ix (sorted index arrays, as produced by splits.py) as an in-memory float32 array."""
    return np.asarray(a[ix], dtype=np.float32)


def fit_models(ds, pair, keys, fit, val, test, seed, cfg, extra_twin=False):
    V = view_arrays(ds, pair)
    y = ds["y"]
    cache = {v: (take(V[v], fit), take(V[v], val), take(V[v], test)) for v in {FAM_VIEW[k][1] for k in keys}}
    models, p_val, p_test = {}, {}, {}
    todo = [(k, FAM_VIEW[k][0], FAM_VIEW[k][1], seed) for k in keys]
    if extra_twin:
        todo.append(("twin", "cloud", "cloud", seed + cfg.twin_offset))
    for key, fam, view, sd in todo:
        Xf, Xv, Xt = cache[view]
        m = make_model(pair, fam, sd, cfg).fit(Xf, y[fit], Xv, y[val])
        models[key] = m
        p_val[key] = m.predict_proba(Xv)
        p_test[key] = m.predict_proba(Xt)
    shapes = {v: cache[v][0].shape[1:] for v in cache}
    return models, p_val, p_test, shapes


def router_metrics(rows_add, y, pe, pc, g, scores: dict):
    for name, s in scores.items():
        rows_add("C", name, np.nan, M.weighted_concordance(g, s))
        rows_add("capture", name, np.nan, M.capture(g, s, *CAPTURE_RANGE))
        rows_add("capture_0_1", name, np.nan, M.capture(g, s, 0.0, 1.0))
        for f in BUDGETS:
            rows_add("ba_at_budget", name, f, M.ba_at_budget(y, pe, pc, s, f))
            rows_add("recall_at_budget", name, f, recall_at_budget(y, pe, pc, s, f))


def run_split(ds, pair, sp, cfg, q=None):
    """Returns (rows, arrays, info). q = None is the main run (h2_q = 1.0)."""
    y, groups = ds["y"], ds["groups"]
    sid = sp["split_id"]
    seed = cfg.model_seed + sid
    fit, val, test = sp["fit"], sp["val"], sp["test"]
    h2 = q is not None
    if h2:
        fit, val = h2_subsample(y, groups, fit, val, q, sid, cfg)
        reason = check_split(y, groups, fit, val, test)
        if reason:
            return None, None, dict(skip=reason)
    yv, yt = y[val], y[test]
    base = dict(dataset=ds["name"], pair=pair, split=sid, repeat=sp["repeat"], fold=sp["fold"],
                h2_q=float(q if h2 else 1.0), n_train=int(len(fit) + len(val)), n_test=int(len(test)),
                n_pos_train=int(y[fit].sum() + y[val].sum()))
    rows = []

    def add(quantity, router="", budget=np.nan, value=np.nan):
        rows.append(dict(base, quantity=quantity, router=router, budget=float(budget), value=float(value)))

    add("tiny_settings", value=float(cfg.tiny))
    add("n_fit", value=len(fit)); add("n_val", value=len(val))
    # positive groups per part before (sp) and after (kept) H2 subsampling; equal in the main run
    for nm, full, kept in (("fit", sp["fit"], fit), ("val", sp["val"], val)):
        add(f"n_pos_groups_{nm}", value=len(np.unique(groups[full][y[full] == 1])))
        add(f"n_pos_groups_{nm}_kept", value=len(np.unique(groups[kept][y[kept] == 1])))
    add("n_pos_fit", value=y[fit].sum()); add("n_pos_val", value=yv.sum()); add("n_pos_test", value=yt.sum())

    keys = ("ef_ev", "cf_cv") if h2 else KEYS
    twin = (pair == "A") and not h2
    models, pv, pt, shapes = fit_models(ds, pair, keys, fit, val, test, seed, cfg, extra_twin=twin)
    allk = list(keys) + (["twin"] if twin else [])
    thr = {k: M.threshold_max_ba(pv[k], yv) for k in allk}
    thr_c = {k: threshold_composite(pv[k], yv) for k in allk}
    for k in allk:
        add("threshold", k, value=thr[k]); add("threshold_composite", k, value=thr_c[k])

    ba = factorial_ba(yt, pt, thr, keys)
    for k, v in ba.items():
        add(f"ba_{k}", value=v)
    ba_comp = factorial_ba(yt, pt, thr_c, keys)
    for k, v in ba_comp.items():
        add(f"comp_ba_{k}", value=v)
    if not h2:
        for k, v in factorial_steps(ba).items():
            add(k, value=v)
        for k, v in factorial_steps(ba_comp).items():
            add(f"comp_{k}", value=v)
        for edge_key, suf in (("ef_ev", "_dep"), ("cf_ev", "_upg")):
            res, _ = combination(yt, pt[edge_key], pt["cf_cv"], thr[edge_key], thr["cf_cv"],
                                 pv[edge_key], pv["cf_cv"], yv, groups[val], seed, cfg, suffix=suf)
            for k, v in res.items():
                add(k, value=v)
        if twin:
            for k, v in self_oracle_headroom(yt, pt["cf_cv"], pt["twin"], thr["cf_cv"], thr["twin"]).items():
                add(k, value=v)

    # ---------------- routers (deployed edge)
    pe_t = (pt["ef_ev"] >= thr["ef_ev"]).astype(int); pc_t = (pt["cf_cv"] >= thr["cf_cv"]).astype(int)
    g = M.escalation_gain(yt, pe_t, pc_t)
    g_upg = None
    if not h2:
        pu_t = (pt["cf_ev"] >= thr["cf_ev"]).astype(int)
        g_upg = M.escalation_gain(yt, pu_t, pc_t)
    Xe = ds["X_edge"]
    Z_val = RT.router_inputs(Xe[val], pv["ef_ev"]); Z_test = RT.router_inputs(Xe[test], pt["ef_ev"])
    tg = RT.router_targets(yv, pv["ef_ev"], pv["cf_cv"], thr["ef_ev"], thr["cf_cv"])
    scores = {"R1": RT.r1_confidence(pt["ef_ev"], thr["ef_ev"])}
    if not h2:
        scores["R0"] = RT.r0_random(len(test), cfg.router_seed + sid)
        scores["R2"] = RT.r2_entropy(pt["ef_ev"])
        scores["R3"] = RT.r3_conformal(pv["ef_ev"], yv, thr["ef_ev"], pt["ef_ev"])
        scores.update(RT.learned_routers(Z_val, Z_test, tg, seed, cfg))
    else:
        scores["R6"] = RT.fit_regressor_router(Z_val, tg["g"], seed, cfg,   # weights rescaled to mean 1 inside
                                               sample_weight=tg["w"]).predict(Z_test)
    if pair == "A" and not h2:
        lr = models["ef_ev"]
        E_val = RT.explanation_inputs(Xe[val], lr, Xe[fit], cfg.router_seed + sid, cfg)
        E_test = RT.explanation_inputs(Xe[test], lr, Xe[fit], cfg.router_seed + sid, cfg)
        scores["R7"] = RT.r7_router(Z_val, Z_test, E_val, E_test, tg, seed, cfg)
        Xc = ds["X_cloud"]
        cr = RT.crossfit_cloud_correct(lambda: make_model("A", "cloud", seed, cfg), np.asarray(Xc[fit]), y[fit],
                                       groups[fit], thr["cf_cv"], cfg.l2d_folds)
        add("l2d_cloud_right_fit", value=cr.mean())
        net = RT.train_l2d(lr.scaler.transform(Xe[fit]), y[fit], cr, seed, cfg)
        scores["R8"] = RT.l2d_score(net, lr.scaler.transform(Xe[test]))
    if not h2:
        scores["R9"] = RT.r9_periodic(ds["order"][test])
    scores["oracle"] = RT.oracle_score(g)
    router_metrics(add, yt, pe_t, pc_t, g, scores)
    add("rho_pos", "R1", value=spearman(scores["R1"][yt == 1], g[yt == 1]))
    cap6 = M.capture(g, scores["R6"], *CAPTURE_RANGE)
    add("outcome", "R6", value=float(np.isfinite(cap6) and cap6 > cfg.outcome_cut))
    add("dcap_R6_R1", "R6", value=cap6 - M.capture(g, scores["R1"], *CAPTURE_RANGE))
    add("g_sum", value=g.sum())
    if pair in ("A", "B"):
        au = run_audit(Z_val, tg["g"], tg["w"], groups[val], sid, cfg)
        for k in ("audit_go", "audit_C_val", "audit_C_lo", "audit_C_hi"):
            add(k, "R6", value=au[k])

    arrays = dict(y_test=yt, groups_test=groups[test].astype(str), test_idx=test, order_test=ds["order"][test],
                  g_dep=g, **{f"p_{k}": pt[k] for k in allk}, **{f"pval_{k}": pv[k] for k in allk},
                  **{f"t_{k}": np.array(thr[k]) for k in allk}, **{f"tcomp_{k}": np.array(thr_c[k]) for k in allk},
                  **{f"score_{k}": v for k, v in scores.items()})
    if g_upg is not None:
        arrays["g_upg"] = g_upg
    info = dict(models=models, shapes=shapes)
    return rows, arrays, info


def write_environment(out: Path):
    import sklearn, scipy, torch, imblearn, statsmodels
    p = out / "ENVIRONMENT.txt"
    out.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join([f"python {platform.python_version()}", f"platform {platform.platform()}",
                            f"numpy {np.__version__}", f"pandas {pd.__version__}", f"scipy {scipy.__version__}",
                            f"scikit-learn {sklearn.__version__}", f"imbalanced-learn {imblearn.__version__}",
                            f"statsmodels {statsmodels.__version__}", f"torch {torch.__version__}",
                            f"torch threads {torch.get_num_threads()}"]) + "\n")


def write_footprint(path: Path, ds, pair, info):
    fp = {}
    for k, m in info["models"].items():
        fam, view = FAM_VIEW.get(k, ("cloud", "cloud"))
        shp = info["shapes"][view]
        fp[k] = dict(family=FAMILY_NAMES[(pair, fam)], view=view, input_shape=list(map(int, shp)),
                     **footprint(m, tuple(shp) if len(shp) > 1 else int(shp[0])))
    payload = dict(ds["bytes"]); payload["explanation_token"] = EXPLANATION_TOKEN_BYTES
    path.write_text(json.dumps(dict(dataset=ds["name"], pair=pair, footprint=fp, payload_bytes=payload),
                               indent=1, default=float))


def _atomic_parquet(df: pd.DataFrame, fn: Path):
    tmp = fn.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, fn)


def _atomic_npz(fn: Path, **arrays):
    tmp = fn.with_name(fn.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, fn)


def _atomic_text(fn: Path, writer):
    tmp = fn.with_name(fn.name + ".tmp")
    writer(tmp)
    os.replace(tmp, fn)


def check_tiny_out(tiny_flag: bool, out) -> None:
    """--tiny may never write into the repository's results/ tree."""
    if not tiny_flag:
        return
    o, r = Path(out).resolve(), (ROOT / "results").resolve()
    if o == r or r in o.parents:
        raise SystemExit("--tiny may not write to results/ (test-only settings); pass --out <scratch dir>")


def parse_slice(s, n):
    if not s:
        return 0, n
    a, b = s.split(":")
    return int(a or 0), int(b) if b else n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("pair", choices=["A", "B", "C"])
    ap.add_argument("--h2", action="store_true"); ap.add_argument("--splits", default="")
    ap.add_argument("--root", default="data/processed"); ap.add_argument("--out", default="results")
    ap.add_argument("--group-col", default=None); ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--max-windows", type=int, default=None, help="pair-C compute cap (DEVIATIONS.md)")
    a = ap.parse_args(argv)
    check_tiny_out(a.tiny, a.out)
    cfg = tiny() if a.tiny else PLAN
    if a.h2 and a.pair != "A":
        raise SystemExit("--h2 uses pair A only")
    import torch
    torch.set_num_threads(2)
    ds = load_dataset(a.dataset, a.root, a.group_col)
    if a.max_windows:
        if a.pair != "C":
            raise SystemExit("--max-windows is defined for pair C only")
        ds = cap_windows(ds, a.max_windows)
        print(json.dumps({"cap": ds.get("cap")}), flush=True)
    if a.group_col:
        ds["name"] = f"{a.dataset}__by_{a.group_col}"
    out_root = Path(a.out)
    write_environment(out_root)
    od = out_root / ds["name"] / a.pair
    od.mkdir(parents=True, exist_ok=True)
    if ds.get("cap"):
        (od / "cap.json").write_text(json.dumps(ds["cap"]))
    splits = make_splits(ds["y"], ds["groups"], PAIR_KIND[a.pair], cfg)
    lo, hi = parse_slice(a.splits, len(splits))
    levels = list(H2_LEVELS) if a.h2 else [None]
    log = open(od / "log.jsonl", "a")
    for sp in splits[lo:hi]:
        for q in levels:
            tag = f"{sp['split_id']}" + (f"_q{q}" if q is not None else "")
            fn = od / f"split_{tag}.parquet"
            if fn.exists():
                continue
            t0 = time.time()
            if sp["skip"]:
                rows, arrays, info = None, None, dict(skip=sp["skip"])
            else:
                rows, arrays, info = run_split(ds, a.pair, sp, cfg, q)
            dt = time.time() - t0
            if rows is None:
                sk = dict(dataset=ds["name"], pair=a.pair, split=sp["split_id"], repeat=sp["repeat"],
                          fold=sp["fold"], h2_q=float(q or 1.0), n_train=0, n_test=0, n_pos_train=0,
                          router="", budget=np.nan)
                _atomic_parquet(pd.DataFrame([dict(sk, quantity="skipped", value=1.0),
                                              dict(sk, quantity="tiny_settings", value=float(cfg.tiny))]), fn)
                rec = dict(split=tag, skipped=info["skip"], seconds=dt, tiny=cfg.tiny)
            else:
                df = pd.DataFrame(rows)
                df = pd.concat([df, pd.DataFrame([dict(rows[0], quantity="seconds", router="", budget=np.nan,
                                                       value=dt)])], ignore_index=True)
                _atomic_npz(od / f"arrays_{tag}.npz", **arrays)
                fpp = od / "footprint.json"
                if q is None and not fpp.exists():
                    _atomic_text(fpp, lambda t: write_footprint(t, ds, a.pair, info))
                _atomic_parquet(df, fn)                  # last: its existence marks the split as done
                rec = dict(split=tag, seconds=round(dt, 2), K=sp["K"], K_in=sp["K_in"], tiny=cfg.tiny)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
    log.close()


if __name__ == "__main__":
    main()
