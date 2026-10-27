"""Shared per-channel window features. Every dataset uses exactly this code, so
edge and cloud views differ only in WHICH channels they see, never in how a
channel is summarised. Edge features are therefore a column subset of cloud
features (nested views), which the factorial decomposition relies on.

Input  W : float array (n_windows, n_channels, length), one sampling rate per channel group.
Output X : float32 array (n_windows, n_channels * N_FEAT), names "<channel>__<feature>".
"""
from __future__ import annotations
import numpy as np

FEATURES = ["mean", "std", "rms", "skew", "kurt", "p2p", "crest", "mad",
            "shape", "impulse", "zcr", "spec_centroid", "spec_spread",
            "spec_entropy", "band_lo", "band_mid", "band_hi"]
N_FEAT = len(FEATURES)
_EPS = 1e-12


def channel_features(x: np.ndarray) -> np.ndarray:
    """x: (n_windows, length) for ONE channel. Returns (n_windows, N_FEAT)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("channel_features expects (n_windows, length)")
    n, L = x.shape
    mu = x.mean(axis=1)
    xc = x - mu[:, None]
    sd = xc.std(axis=1)
    rms = np.sqrt((x ** 2).mean(axis=1))
    m3 = (xc ** 3).mean(axis=1)
    m4 = (xc ** 4).mean(axis=1)
    skew = np.where(sd > _EPS, m3 / np.maximum(sd, _EPS) ** 3, 0.0)
    kurt = np.where(sd > _EPS, m4 / np.maximum(sd, _EPS) ** 4 - 3.0, 0.0)
    p2p = x.max(axis=1) - x.min(axis=1)
    absx = np.abs(x)
    peak = absx.max(axis=1)
    mean_abs = absx.mean(axis=1)
    crest = np.where(rms > _EPS, peak / np.maximum(rms, _EPS), 0.0)
    mad = np.abs(xc).mean(axis=1)
    shape = np.where(mean_abs > _EPS, rms / np.maximum(mean_abs, _EPS), 0.0)
    impulse = np.where(mean_abs > _EPS, peak / np.maximum(mean_abs, _EPS), 0.0)
    sgn = np.sign(xc)
    zcr = (np.abs(np.diff(sgn, axis=1)) > 0).mean(axis=1) if L > 1 else np.zeros(n)
    # spectrum of the de-meaned window; frequencies normalised to [0, 1] of Nyquist
    if L >= 4:
        P = np.abs(np.fft.rfft(xc, axis=1)) ** 2
        P = P[:, 1:]                                  # drop DC
        f = np.linspace(0.0, 1.0, P.shape[1] + 1)[1:]
        tot = P.sum(axis=1)
        Pn = np.where(tot[:, None] > _EPS, P / np.maximum(tot[:, None], _EPS), 0.0)
        cen = (Pn * f).sum(axis=1)
        spread = np.sqrt(np.maximum((Pn * (f - cen[:, None]) ** 2).sum(axis=1), 0.0))
        ent = -(Pn * np.log(Pn + _EPS)).sum(axis=1) / np.log(max(P.shape[1], 2))
        b1 = f <= 1 / 3
        b2 = (f > 1 / 3) & (f <= 2 / 3)
        b3 = f > 2 / 3
        lo, mid, hi = Pn[:, b1].sum(axis=1), Pn[:, b2].sum(axis=1), Pn[:, b3].sum(axis=1)
    else:
        cen = spread = ent = lo = mid = hi = np.zeros(n)
    out = np.column_stack([mu, sd, rms, skew, kurt, p2p, crest, mad, shape, impulse,
                           zcr, cen, spread, ent, lo, mid, hi])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def window_features(W: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """W: (n_windows, n_channels, length) at ONE sampling rate. Returns X and names."""
    W = np.asarray(W)
    if W.ndim != 3 or W.shape[1] != len(channel_names):
        raise ValueError("W must be (n_windows, n_channels, length) and match channel_names")
    blocks, names = [], []
    for c, name in enumerate(channel_names):
        blocks.append(channel_features(W[:, c, :]))
        names += [f"{name}__{f}" for f in FEATURES]
    return np.concatenate(blocks, axis=1), names


def multi_rate_features(groups: list[tuple[np.ndarray, list[str]]]) -> tuple[np.ndarray, list[str]]:
    """Channels sampled at different rates: pass [(W_rate1, names1), (W_rate2, names2), ...],
    each W of shape (n_windows, n_channels_k, length_k). Columns are concatenated in order."""
    Xs, names = [], []
    for W, nm in groups:
        X, n = window_features(W, nm)
        Xs.append(X)
        names += n
    return np.concatenate(Xs, axis=1), names
