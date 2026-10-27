"""Inference for repeated grouped cross-validation.

Repeated resplits of one dataset overlap, so the naive paired t-test understates the
variance. Every per-dataset test uses the Nadeau-Bengio corrected resampled t-test:
    se = sd(d) * sqrt(1/J + n_test/n_train)
Note the floor sd * sqrt(n_test/n_train): no number of repeats pushes se below it, so a
+-margin equivalence test can be infeasible by design. power_floor() reports this.
Pooling across datasets uses a DerSimonian-Laird random-effects model on the per-dataset
estimates and their corrected standard errors.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def nb_test(d, n_train: float, n_test: float, margin: float | None = None, alpha: float = 0.05) -> dict:
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    J = len(d)
    if J < 2:
        return dict(J=J, mean=float(d.mean()) if J else np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    p=np.nan, p_eq=np.nan, se_naive=np.nan, p_naive=np.nan, se_floor=np.nan)
    m = float(d.mean()); sd = float(d.std(ddof=1))
    se = sd * np.sqrt(1.0 / J + n_test / n_train)
    se0 = sd / np.sqrt(J)
    df = J - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    out = dict(J=J, mean=m, sd=sd, se=se, ci_lo=m - tcrit * se, ci_hi=m + tcrit * se,
               p=float(2 * stats.t.sf(abs(m / se), df)) if se > 0 else (0.0 if m != 0 else 1.0),
               se_naive=se0, p_naive=float(2 * stats.t.sf(abs(m / se0), df)) if se0 > 0 else np.nan,
               se_floor=sd * np.sqrt(n_test / n_train))
    if margin is not None and se > 0:
        p_lo = stats.t.sf((m + margin) / se, df)      # H0: mean <= -margin
        p_hi = stats.t.cdf((m - margin) / se, df)     # H0: mean >= +margin
        out["p_eq"] = float(max(p_lo, p_hi))
        out["eq_feasible"] = bool(out["se_floor"] * stats.t.ppf(1 - alpha, df) < margin)
    return out


def one_sided_p(est: dict, direction: str = "greater") -> float:
    """One-sided p from an nb_test/pool result (t with J-1 df, or z when df is None)."""
    se = est["se"]; m = est["mean"]
    df = est.get("df", est.get("J", 2) - 1)
    z = m / se if se and se > 0 else np.inf * np.sign(m)
    dist = stats.t(df) if df else stats.norm
    return float(dist.sf(z) if direction == "greater" else dist.cdf(z))


def pool_random_effects(means, ses, margin: float | None = None, alpha: float = 0.05) -> dict:
    """DerSimonian-Laird random-effects pooling of per-dataset estimates."""
    y = np.asarray(means, float); s = np.asarray(ses, float)
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    y, s = y[ok], s[ok]
    k = len(y)
    if k == 0:
        return dict(k=0, mean=np.nan, se=np.nan)
    w = 1.0 / s ** 2
    ybar = float((w * y).sum() / w.sum())
    Q = float((w * (y - ybar) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - (k - 1)) / c) if (k > 1 and c > 0) else 0.0
    ws = 1.0 / (s ** 2 + tau2)
    m = float((ws * y).sum() / ws.sum()); se = float(np.sqrt(1.0 / ws.sum()))
    z = stats.norm.ppf(1 - alpha / 2)
    out = dict(k=k, mean=m, se=se, ci_lo=m - z * se, ci_hi=m + z * se, tau2=tau2, Q=Q, df=None,
               p=float(2 * stats.norm.sf(abs(m / se))))
    if margin is not None:
        out["p_eq"] = float(max(stats.norm.sf((m + margin) / se), stats.norm.cdf((m - margin) / se)))
    return out


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def power_floor(sd: float, n_train: float, n_test: float, margin: float, J: int, alpha: float = 0.05) -> dict:
    """Is a +-margin TOST achievable at all, and with how many repeats (if the true difference is 0)?"""
    floor = sd * np.sqrt(n_test / n_train)
    feasible = floor * stats.t.ppf(1 - alpha, max(J - 1, 1)) < margin
    need = None
    if feasible:
        for JJ in range(2, 1001):
            se = sd * np.sqrt(1.0 / JJ + n_test / n_train)
            if se * stats.t.ppf(1 - alpha, JJ - 1) < margin:
                need = JJ; break
    return dict(se_floor=float(floor), feasible=bool(feasible), repeats_needed=need)


def pool_hksj(means, ses, margin: float | None = None, alpha: float = 0.05) -> dict:
    """Random-effects pooling with the Hartung-Knapp-Sidik-Jonkman variance (t, k-1 df),
    DerSimonian-Laird tau^2, and the conservative max(1, q) modification. Primary pooling
    rule of the frozen plan."""
    y = np.asarray(means, float); s = np.asarray(ses, float)
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    y, s = y[ok], s[ok]
    k = len(y)
    if k < 2:
        return dict(k=k, mean=float(y[0]) if k else np.nan, se=float(s[0]) if k else np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, p=np.nan, df=None)
    base = pool_random_effects(y, s)
    tau2 = base["tau2"]
    ws = 1.0 / (s ** 2 + tau2)
    m = float((ws * y).sum() / ws.sum())
    q = float((ws * (y - m) ** 2).sum() / (k - 1))
    se = float(np.sqrt(max(1.0, q) / ws.sum()))
    df = k - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    out = dict(k=k, mean=m, se=se, ci_lo=m - tc * se, ci_hi=m + tc * se, tau2=tau2, q=q, df=df,
               p=float(2 * stats.t.sf(abs(m / se), df)))
    if margin is not None:
        out["p_eq"] = float(max(stats.t.sf((m + margin) / se, df), stats.t.cdf((m - margin) / se, df)))
    return out
