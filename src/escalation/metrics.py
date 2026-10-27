"""Metrics for edge-to-cloud escalation. Any change here should be matched by
tests/test_metrics.py.

Notation (binary task, y in {0,1}):
  w_i  balanced-accuracy weight: 1/(2 n_pos) for positives, 1/(2 n_neg) for negatives
  g_i  gain of escalating window i: w_i * (1[cloud right] - 1[edge right])
  BA(E) = BA(edge) + sum_{i in E} g_i  for any escalation set E (exact identity)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


def ba_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("balanced accuracy needs both classes")
    return np.where(y == 1, 0.5 / n1, 0.5 / n0)


def balanced_accuracy(y, pred) -> float:
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    return float((ba_weights(y) * (pred == y)).sum())


def recall_at_far(y, score, far: float) -> float:
    """Recall when the threshold is set so the false-alarm rate on negatives is <= far
    (threshold-free diagnostic, computed on the evaluation set itself)."""
    y = np.asarray(y).astype(int); s = np.asarray(score, float)
    neg = np.sort(s[y == 0])[::-1]
    k = int(np.floor(far * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf          # alarms strictly above thr
    return float((s[y == 1] > thr).mean())


def escalation_gain(y, pred_edge, pred_cloud, weights=None) -> np.ndarray:
    y = np.asarray(y).astype(int)
    w = ba_weights(y) if weights is None else np.asarray(weights, float)
    return w * ((np.asarray(pred_cloud) == y).astype(float) - (np.asarray(pred_edge) == y).astype(float))


def ba_with_escalation(y, pred_edge, pred_cloud, escalate) -> float:
    return balanced_accuracy(y, np.where(np.asarray(escalate, bool), pred_cloud, pred_edge))


def oracle_predictions(y, pred_edge, pred_cloud) -> np.ndarray:
    """Per-window oracle: the cloud decision when it is right, else the edge decision."""
    y = np.asarray(y).astype(int)
    return np.where(np.asarray(pred_cloud) == y, pred_cloud, pred_edge)


def weighted_concordance(g, s) -> float:
    """C = sum_{i<j} |g_i-g_j| kappa_ij / sum_{i<j} |g_i-g_j|, kappa_ij = 1 if s orders the pair
    like g, 1/2 if s ties, 0 if reversed. Exact O(n log n) form:
        S = sum_i g_i (R_i - (n+1)/2),  R = average ranks of s (ascending),
        D = sum_k g_(k) (2k - n - 1) with g sorted ascending,  C = 1/2 + S / D.
    Identity: averaged over all budgets k = 1..n, a router's gain over random is
    (2C - 1) times the oracle ranking's gain over random."""
    g = np.asarray(g, float); s = np.asarray(s, float)
    n = len(g)
    gs = np.sort(g)
    D = float((gs * (2 * np.arange(1, n + 1) - n - 1)).sum())
    if D <= 0:
        return float("nan")
    R = rankdata(s, method="average")
    S = float((g * (R - (n + 1) / 2.0)).sum())
    return 0.5 + S / D


def gain_curve(g, s) -> np.ndarray:
    """Expected cumulative gain G(k), k = 0..n, escalating in descending order of s,
    with random tie-breaking taken in expectation (each tie block contributes its mean gain
    per position)."""
    g = np.asarray(g, float); s = np.asarray(s, float)
    order = np.argsort(-s, kind="stable")
    gs, ss = g[order], s[order]
    vals = gs.copy()
    # average gains within blocks of tied scores
    start = 0
    n = len(gs)
    while start < n:
        end = start + 1
        while end < n and ss[end] == ss[start]:
            end += 1
        if end - start > 1:
            vals[start:end] = gs[start:end].mean()
        start = end
    return np.concatenate([[0.0], np.cumsum(vals)])


def capture(g, s, f_lo: float = 0.0, f_hi: float = 1.0) -> float:
    """Share of the oracle ranking's advantage over random that s captures, summed over
    budgets k/n in [f_lo, f_hi]. With [0, 1] this equals 2C - 1 (up to the k = 0 term)."""
    g = np.asarray(g, float); n = len(g)
    k = np.arange(n + 1)
    sel = (k / n >= f_lo) & (k / n <= f_hi) & (k > 0)
    rand = k / n * g.sum()
    G = gain_curve(g, s)
    Go = gain_curve(g, g)
    num = (G - rand)[sel].sum(); den = (Go - rand)[sel].sum()
    return float(num / den) if den > 0 else float("nan")


def ba_at_budget(y, pred_edge, pred_cloud, s, f: float) -> float:
    """BA when the top round(f n) windows by s are escalated (ties in expectation)."""
    g = escalation_gain(y, pred_edge, pred_cloud)
    k = int(round(f * len(g)))
    return balanced_accuracy(y, pred_edge) + float(gain_curve(g, s)[k])


def threshold_max_ba(p_val, y_val, grid=None) -> float:
    """Validation threshold maximising balanced accuracy (primary rule of the frozen plan).
    Ties are broken towards the threshold closest to 0.5."""
    p_val = np.asarray(p_val, float); y_val = np.asarray(y_val).astype(int)
    if grid is None:
        grid = np.unique(np.concatenate([np.quantile(p_val, np.linspace(0, 1, 201)),
                                         np.linspace(0.01, 0.99, 99)]))
    best_t, best_v = 0.5, -1.0
    for t in grid:
        v = balanced_accuracy(y_val, (p_val >= t).astype(int))
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_v = float(t), v
    return best_t


def calibrate_to_threshold(p, t) -> np.ndarray:
    """Monotone map sending threshold t to 0.5 (rank-preserving), as in the original pipeline."""
    p = np.asarray(p, float)
    lo = 0.5 * p / max(t, 1e-12)
    hi = 0.5 + 0.5 * (p - t) / max(1 - t, 1e-12)
    return np.clip(np.where(p < t, lo, hi), 0.0, 1.0)
