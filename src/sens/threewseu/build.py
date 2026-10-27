"""POST-PLAN sensitivity variants: de-meaned and normalised 3W and SEU, and the 3W cloud without P-ANULAR.

    python src/sens/threewseu/build.py check            # stored X reproduces from raw (per dataset)
    python src/sens/threewseu/build.py <variant> [...]  # build data/processed_sens/<variant>/

Variants (features always via escalation.features.window_features, one call per view, edge first):
  seu_dm, threew_dm     every window de-meaned per channel; the all-zero `mean` columns dropped (16 per channel)
  seu_zn, threew_zn     every window z-normalised per channel (sd floor 1e-6); only scale-free features kept
                        (skew, kurt, crest, shape, impulse, zcr, six spectral: 12 per channel)
  threew_inorm          per instance x channel robust normalisation (median, IQR floor 1e-6, over all samples of
                        the instance's kept windows), then all 17 features. NON-CAUSAL (uses future samples).
  threew_noann          stored features, cloud view without the 17 P-ANULAR columns.
Not confirmatory; nothing here changes a pre-specified verdict.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from escalation.features import window_features, FEATURES   # noqa: E402

SRC = ROOT / "data" / "processed"
DST = ROOT / "data" / "processed_sens"
CHUNK = 1000
SCALE_FREE = ["skew", "kurt", "crest", "shape", "impulse", "zcr", "spec_centroid", "spec_spread",
              "spec_entropy", "band_lo", "band_mid", "band_hi"]
DM_KEEP = [f for f in FEATURES if f != "mean"]


def channels(ds):
    return json.loads((SRC / ds / "channels.json").read_text())


def feats(ds, view, transform=None, rows=None):
    """Features of the stored raw windows of one view, in chunks. transform: f(float64 chunk, row idx) -> chunk."""
    R = np.load(SRC / ds / f"raw_{view}.npy", mmap_mode="r")
    names = channels(ds)[view]
    n = R.shape[0]
    out = []
    for a in range(0, n, CHUNK):
        ix = np.arange(a, min(a + CHUNK, n))
        W = np.asarray(R[a:a + len(ix)], dtype=np.float64)
        if transform is not None:
            W = transform(W, ix)
        X, nm = window_features(W, names)
        out.append(X)
    return np.concatenate(out), nm


def demean(W, ix):
    return W - W.mean(axis=2, keepdims=True)


def znorm(W, ix):
    mu = W.mean(axis=2, keepdims=True)
    sd = W.std(axis=2, keepdims=True)
    return (W - mu) / np.maximum(sd, 1e-6)


def check(ds):
    res = {}
    for view in ("edge", "cloud"):
        X, nm = feats(ds, view)
        S = np.load(SRC / ds / f"X_{view}.npy", mmap_mode="r")
        stored_names = json.loads((SRC / ds / f"names_{view}.json").read_text())
        same_names = nm == stored_names
        exact = bool(np.array_equal(X, S))
        close = bool(np.allclose(X, S, rtol=1e-5, atol=1e-6))
        d = np.abs(X.astype(np.float64) - S.astype(np.float64))
        rel = d / np.maximum(np.abs(S.astype(np.float64)), 1e-6)
        res[view] = dict(names_equal=same_names, exact=exact, allclose_rtol1e5_atol1e6=close,
                         max_abs_diff=float(d.max()), max_rel_diff=float(rel.max()),
                         n_cells_differ=int((d > 0).sum()), shape=list(X.shape))
    return res


def keep_cols(names, keep):
    idx = [i for i, n in enumerate(names) if n.split("__", 1)[1] in keep]
    return idx, [names[i] for i in idx]


def write_variant(variant, parent, Xe, ne, Xc, nc, card_text, extra_bytes_note=""):
    d = DST / variant
    d.mkdir(parents=True, exist_ok=True)
    assert nc[:len(ne)] == ne, "edge columns must lead the cloud columns"
    assert np.array_equal(Xc[:, :len(ne)], Xe)
    np.save(d / "X_edge.npy", Xe.astype(np.float32))
    np.save(d / "X_cloud.npy", Xc.astype(np.float32))
    (d / "names_edge.json").write_text(json.dumps(ne, indent=1))
    (d / "names_cloud.json").write_text(json.dumps(nc, indent=1))
    shutil.copy(SRC / parent / "meta.parquet", d / "meta.parquet")
    shutil.copy(SRC / parent / "channels.json", d / "channels.json")
    b = json.loads((SRC / parent / "bytes.json").read_text())
    b["features_edge"] = 4 * len(ne)
    b["features_cloud"] = 4 * len(nc)
    b["note"] = b.get("note", "") + f"; POST-PLAN variant {variant}: feature bytes recomputed. {extra_bytes_note}".strip()
    (d / "bytes.json").write_text(json.dumps(b, indent=1))
    meta = pd.read_parquet(d / "meta.parquet")
    counts = composition(meta)
    (d / "card.md").write_text(card_text + "\n\n" + counts + "\n")
    return d


def composition(meta):
    lines = [f"## Counts (POST-PLAN variant; windows and grouping identical to the parent)",
             f"- windows {len(meta)}, positives {int(meta.label.sum())}, prevalence {meta.label.mean():.3f}"]
    for col in [c for c in ("group_id", "well") if c in meta]:
        g = meta.groupby("label")[col].nunique()
        K = int(min(5, g.get(0, 0), g.get(1, 0)))
        lines.append(f"- grouping `{col}`: {meta[col].nunique()} groups; groups with negatives {int(g.get(0, 0))}, "
                     f"with positives {int(g.get(1, 0))}; K = min(5, pos, neg) = {K}")
    return "\n".join(lines)


HEAD = ("# POST-PLAN sensitivity variant `{v}` (parent: data/processed/{p})\n\n"
        "Built by src/sens/threewseu/build.py ({item}). Not confirmatory: it cannot change any "
        "pre-specified verdict. Windows, labels, meta (all columns, row order, window_id) and channels are copied "
        "unchanged from the parent; raw_*.npy are not included (pairs A and B do not use them). "
        "Features are recomputed with escalation.features.window_features from the parent's stored raw windows "
        "(one call per view, channels in the stored order, edge channels first) unless stated otherwise.\n\n"
        "## What changed and why\n{what}\n")


def build(variant):
    if variant in ("seu_dm", "threew_dm", "seu_zn", "threew_zn"):
        parent, kind = variant.split("_")
        tf, keep = (demean, DM_KEEP) if kind == "dm" else (znorm, SCALE_FREE)
        Xe, ne = feats(parent, "edge", tf)
        Xc, nc = feats(parent, "cloud", tf)
        ie, ne = keep_cols(ne, keep); ic, nc = keep_cols(nc, keep)
        Xe, Xc = Xe[:, ie], Xc[:, ic]
        if kind == "dm":
            what = ("Every window is de-meaned per channel (x - mean over the window's samples, float64) before the "
                    "features, so no feature can carry the absolute level of a channel. The `mean` feature is then "
                    "identically zero and its column is dropped: 16 features per channel. Note that after "
                    "de-meaning `rms` equals `std`; both columns are kept because the declaration "
                    "specify only the mean drop. Scale (std, p2p, mad) is still present.")
        else:
            what = ("Every window is z-normalised per channel ((x - mean) / max(sd, 1e-6), float64) before the "
                    "features, then only the scale-free features are kept: " + ", ".join(SCALE_FREE) +
                    " (12 per channel). mean, std, rms, p2p and mad are dropped. Neither absolute level nor "
                    "absolute scale is visible to the models.")
        item = ("DECLARED deconfounded variant, plan/postplan/DECONF_DECLARATION.md" if kind == "dm"
                else "secondary sensitivity analysis")
    elif variant == "threew_inorm":
        parent = "threew"
        meta = pd.read_parquet(SRC / parent / "meta.parquet")
        inst = meta["instance"].to_numpy()
        stats = {}
        for view in ("cloud",):
            R = np.load(SRC / parent / f"raw_{view}.npy", mmap_mode="r")
            for k in pd.unique(inst):
                ix = np.flatnonzero(inst == k)
                W = np.asarray(R[ix], dtype=np.float64)               # (nw, C, L)
                A = W.transpose(1, 0, 2).reshape(W.shape[1], -1)
                q1, med, q3 = np.percentile(A, [25, 50, 75], axis=1)
                stats[k] = (med, np.maximum(q3 - q1, 1e-6))
        nce = len(channels(parent)["edge"])          # edge channels are the first cloud channels

        def tf_view(view):
            def f(W, ix):
                out = np.empty_like(W)
                for k in pd.unique(inst[ix]):
                    m = inst[ix] == k
                    med, iqr = stats[k]
                    if view == "edge":
                        med, iqr = med[:nce], iqr[:nce]
                    out[m] = (W[m] - med[None, :, None]) / iqr[None, :, None]
                return out
            return f
        assert channels(parent)["cloud"][:nce] == channels(parent)["edge"]
        Xe, ne = feats(parent, "edge", tf_view("edge"))
        Xc, nc = feats(parent, "cloud", tf_view("cloud"))
        what = ("Per-instance robust normalisation: for each instance (meta.instance, 445) and channel, the "
                "median and the IQR (floor 1e-6) are computed over all samples of that instance's kept windows, "
                "and every sample is mapped to (x - median) / IQR before all 17 features are computed. "
                "This is NON-CAUSAL: the statistics use samples later in the same instance, which a deployed node "
                "would not have. Instances whose channel is median-imputed or frozen become all-zero for that "
                "channel (IQR 0 -> floor; x - median = 0).")
        item = "secondary sensitivity analysis"
    elif variant == "threew_noann":
        parent = "threew"
        Xe = np.load(SRC / parent / "X_edge.npy"); Xc = np.load(SRC / parent / "X_cloud.npy")
        ne = json.loads((SRC / parent / "names_edge.json").read_text())
        nc = json.loads((SRC / parent / "names_cloud.json").read_text())
        ic = [i for i, n in enumerate(nc) if not n.startswith("P-ANULAR__")]
        Xc, nc = Xc[:, ic], [nc[i] for i in ic]
        what = ("Stored features unchanged; the cloud view drops the 17 P-ANULAR columns (annulus pressure, "
                "measurement point undocumented). Edge view unchanged. Cloud = 4 channels x 17 = 68 features.")
        item = "alternative cloud view; secondary sensitivity analysis"
    else:
        raise SystemExit(f"unknown variant {variant}")
    card = HEAD.format(v=variant, p=parent, item=item, what=what)
    card += (f"\n## Shapes\n- X_edge {list(Xe.shape)}, X_cloud {list(Xc.shape)}; edge columns lead cloud columns.\n")
    if parent == "threew":
        card += ("- Grouping: group_id as stored (episode). The by-well analysis runs this same directory with "
                 "`--group-col well` (results under <variant>__by_well).\n")
    d = write_variant(variant, parent, Xe, ne, Xc, nc, card)
    print(variant, Xe.shape, Xc.shape, "->", d)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        out = {ds: check(ds) for ds in (sys.argv[2:] or ["seu", "threew"])}
        print(json.dumps(out, indent=1))
    else:
        for v in sys.argv[1:]:
            build(v)
