"""POST-PLAN: generators of the full validation.
Post-plan, not pre-specified. Hashed in plan/postplan/HASHES.txt before the full run.

Three families. Every dataset has an edge view of D_E = 6 features and a cloud view [x_e, x_c] with
D_C = 6 cloud-only features; s = 2y - 1; a, b as in src/sens/simulation/generator.py.

1. Base (S0, S1, S1b, S2, S3): src/sens/simulation/generator.py UNCHANGED (40 groups of 25 windows, labels
   constant within a group, i.i.d. group effects). Imported, not copied.

2. Temporal dependence (T*), mimicking the 20 contiguous 111-cycle blocks of the hydraulic block-grouped
   splits. The label sequence is the rig's own (2,205 cycles; run-length code HYD_RLE below, read from
   data/processed_sens/hyd_blocks/meta.parquet sorted by cycle), and group = cycle // 111 (20 blocks, the last
   of 96 cycles, 16 of them mixing both labels), exactly as the hyd_blocks variant. Features:
       x_e = mu_e s a + u_g + e,     x_c = mu_c s b + v_g + e'
   with block effects u_g, v_g following a stationary AR(1) over the block index,
       u_0 ~ N(0, tau^2 I),  u_g = phi u_{g-1} + sqrt(1 - phi^2) eta_g,  eta_g ~ N(0, tau^2 I),
   independently for u and v, so adjacent blocks have lag-one correlation phi (0, 0.3, 0.6, 0.9).
   T0_*: mu_c = 0 (no information); T1_*: mu_c = 1.0. T0s_0.9: the null at phi = 0.9 with tau = 1 (stress).

3. Confounded (C*), SKAB-like files with an injection schedule. 40 files of 25 windows; files 0-11 are
   anomaly-free recordings (all negative, file type 0); files 12-39 are experiment files (file type 1) whose
   windows are positive from an onset at a fraction U(0.1, 0.9) of the file onwards. t = time in file
   (window position / 25). Edge as above with i.i.d. file effects (tau = 0.5). Cloud-only features carry NO
   fault information (mu_c = 0) and one confound that reaches only the cloud view, along d = (0,0,0,1,1,0)/sqrt 2:
       C_clock:  + kappa (t - 0.5) d        (a sensor that drifts during each run; label-correlated
                                               because the anomaly occupies the later part of each run)
       C_offset: + delta (filetype - 0.8) d  (the anomaly-free recordings sit at another offset, as SKAB's
                                               separately recorded anomaly-free file)
   C_ref: mu_c = 1.0 and no confound (a real information step under the same schedule).
   Metadata for the shortcut baselines: t and the file type.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation"))
from generator import SCENARIOS as BASE_SCENARIOS, generate as base_generate, A_VEC, B_VEC, D_E, D_C  # noqa: E402

# hydraulic rig label sequence (2,205 cycles), run-length coded (label, length)
HYD_RLE = [(0, 210), (1, 82), (0, 51), (1, 82), (0, 51), (1, 82), (0, 51), (1, 82), (0, 251), (1, 82), (0, 51),
           (1, 82), (0, 51), (1, 82), (0, 51), (1, 82), (0, 251), (1, 82), (0, 51), (1, 82), (0, 51), (1, 82),
           (0, 51), (1, 82), (0, 50)]
HYD_BLOCK = 111
CONF_VEC = np.array([0, 0, 0, 1, 1, 0], float) / np.sqrt(2)


def hyd_labels() -> np.ndarray:
    y = np.concatenate([np.full(n, v, int) for v, n in HYD_RLE])
    assert len(y) == 2205 and y.sum() == 984
    return y


@dataclass(frozen=True)
class TParams:
    mu_e: float = 1.0
    mu_c: float = 0.0
    phi: float = 0.0
    tau: float = 0.5


@dataclass(frozen=True)
class CParams:
    mu_e: float = 1.0
    mu_c: float = 0.0
    kappa: float = 0.0       # clock strength
    delta: float = 0.0       # file-type offset
    tau: float = 0.5
    n_files: int = 40
    n_free: int = 12
    m: int = 25
    onset_lo: float = 0.1
    onset_hi: float = 0.9


T_SCENARIOS = {
    "T0_0.0": TParams(mu_c=0.0, phi=0.0), "T0_0.3": TParams(mu_c=0.0, phi=0.3),
    "T0_0.6": TParams(mu_c=0.0, phi=0.6), "T0_0.9": TParams(mu_c=0.0, phi=0.9),
    "T0s_0.9": TParams(mu_c=0.0, phi=0.9, tau=1.0),
    "T1_0.0": TParams(mu_c=1.0, phi=0.0), "T1_0.3": TParams(mu_c=1.0, phi=0.3),
    "T1_0.6": TParams(mu_c=1.0, phi=0.6), "T1_0.9": TParams(mu_c=1.0, phi=0.9),
}
# kappa and delta: set by the calibration recorded in plan/postplan/simulation_full.md (before hashing)
C_SCENARIOS = {
    "C_ref": CParams(mu_c=1.0),
    "C_clock": CParams(kappa=10.0),
    "C_offset": CParams(delta=8.0),
}


def family(scen: str) -> str:
    if scen in BASE_SCENARIOS:
        return "base"
    if scen in T_SCENARIOS:
        return "T"
    if scen in C_SCENARIOS:
        return "C"
    raise KeyError(scen)


def ar1(rng, G: int, d: int, phi: float, tau: float) -> np.ndarray:
    out = np.empty((G, d))
    out[0] = rng.normal(0.0, tau, d)
    k = np.sqrt(1.0 - phi ** 2)
    for g in range(1, G):
        out[g] = phi * out[g - 1] + k * rng.normal(0.0, tau, d)
    return out


def sample_T(p: TParams, rng, prefix: str = "b") -> dict:
    y = hyd_labels()
    n = len(y)
    gi = np.arange(n) // HYD_BLOCK
    G = gi.max() + 1
    u, v = ar1(rng, G, D_E, p.phi, p.tau), ar1(rng, G, D_C, p.phi, p.tau)
    s = (2 * y - 1).astype(float)
    xe = p.mu_e * s[:, None] * A_VEC[None, :] + u[gi] + rng.normal(size=(n, D_E))
    xc = p.mu_c * s[:, None] * B_VEC[None, :] + v[gi] + rng.normal(size=(n, D_C))
    return dict(X_edge=xe.astype(np.float32), X_cloud=np.column_stack([xe, xc]).astype(np.float32), y=y,
                groups=np.array([f"{prefix}{k:02d}" for k in gi]), cycle=np.arange(n, dtype=float))


def sample_C(p: CParams, rng, prefix: str = "f") -> dict:
    F, m = p.n_files, p.m
    ftype = (np.arange(F) >= p.n_free).astype(int)
    onset = np.where(ftype == 1, np.round(rng.uniform(p.onset_lo, p.onset_hi, F) * m), m + 1).astype(int)
    fi = np.repeat(np.arange(F), m)
    pos = np.tile(np.arange(m), F)
    y = (pos >= onset[fi]).astype(int)
    t = pos / m
    u = rng.normal(0.0, p.tau, (F, D_E)); v = rng.normal(0.0, p.tau, (F, D_C))
    s = (2 * y - 1).astype(float)
    n = len(y)
    xe = p.mu_e * s[:, None] * A_VEC[None, :] + u[fi] + rng.normal(size=(n, D_E))
    conf = p.kappa * (t - 0.5) + p.delta * (ftype[fi] - 0.8)
    xc = p.mu_c * s[:, None] * B_VEC[None, :] + v[fi] + conf[:, None] * CONF_VEC[None, :] + rng.normal(size=(n, D_C))
    return dict(X_edge=xe.astype(np.float32), X_cloud=np.column_stack([xe, xc]).astype(np.float32), y=y,
                groups=np.array([f"{prefix}{k:03d}" for k in fi]), t=t, ftype=ftype[fi].astype(float))


def generate(scen: str, seed: int) -> dict:
    fam = family(scen)
    if fam == "base":
        d = base_generate(scen, seed)
        d["family"] = "base"
        return d
    rng = np.random.default_rng(seed)
    if fam == "T":
        p = T_SCENARIOS[scen]; d = sample_T(p, rng)
    else:
        p = C_SCENARIOS[scen]; d = sample_C(p, rng)
    d.update(params=asdict(p), seed=int(seed), name=scen, family=fam)
    return d
