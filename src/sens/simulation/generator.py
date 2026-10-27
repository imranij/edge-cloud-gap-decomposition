"""POST-PLAN: synthetic two-view generator with known ground truth.
Post-plan, not pre-specified.

One synthetic dataset has G groups (half all-positive, half all-negative; labels constant within a
group, like the hydraulic rig and SEU) of m windows each. With s = 2y - 1 in {-1, +1}:

  edge view (d_e = 6)       x_e = mu_e * s * a + u_g + e,     a = (1, 1, 0, 0, 0, 0) / sqrt(2)
  cloud-only (d_c = 6)      x_c = mu_c(x) * s * b + v_g + e', b = (1, 1, 1, 0, 0, 0) / sqrt(3)
  gate                      mu_c(x) = mu_c * (1 + lam * sign(x_e[:, 2])),  lam in [0, 1]
  group effects             u_g, v_g ~ N(0, tau^2 I_6) per group;  e, e' ~ N(0, I_6) per window
  cloud view                [x_e, x_c]  (edge columns first, as in the processed real data)

x_e[:, 2] = u_g[2] + e[2] carries no class signal (a[2] = 0) and is symmetric about 0, so
P(sign = +1) = 1/2 and the average separation E[mu_c(x)] = mu_c for every lam. lam = 0 spreads the
cloud-only information uniformly (a router can learn nothing about where escalation helps beyond
what p_E gives); lam = 1 puts all of it where x_e[:, 2] > 0 (an edge-visible region), so the gain
of escalation is predictable from an edge feature that confidence ignores.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np

D_E = 6
D_C = 6
A_VEC = np.array([1, 1, 0, 0, 0, 0], float) / np.sqrt(2)
B_VEC = np.array([1, 1, 1, 0, 0, 0], float) / np.sqrt(3)
GATE_COL = 2


@dataclass(frozen=True)
class Params:
    mu_e: float = 1.0
    mu_c: float = 0.0
    lam: float = 0.0
    tau: float = 0.5


# Scenario values after the pilot (src/sens/simulation/pilot.py; results_sens/simulation/pilot.csv and
# pilot_round2.csv, 30 draws each). The initial values (0.6 / 1.2 for both lam) gave information
# steps of 0.035 / 0.102 (lam = 0) and 0.022 / 0.053 (lam = 1); the values below target about 0.03 and
# 0.08 at both lam (pilot: 0.035, 0.076, 0.030, 0.081). S0 has a slightly NEGATIVE step (-0.010): the
# cloud LR fits six pure-noise columns.
SCENARIOS: dict[str, Params] = {
    "S0": Params(mu_c=0.0, lam=0.0),      # no information, nothing to learn
    "S1": Params(mu_c=0.6, lam=0.0),      # information ~0.03, not learnable beyond confidence
    "S1b": Params(mu_c=1.0, lam=0.0),     # information ~0.08, not learnable beyond confidence
    "S2": Params(mu_c=0.8, lam=1.0),      # information ~0.03, concentrated where x_e[:, 2] > 0
    "S3": Params(mu_c=2.5, lam=1.0),      # information ~0.08, concentrated where x_e[:, 2] > 0
}


def sample(p: Params, rng: np.random.Generator, n_pos_groups: int, n_neg_groups: int, m: int,
           group_prefix: str = "g") -> dict:
    """Windows of n_pos_groups positive and n_neg_groups negative groups, m windows each."""
    G = n_pos_groups + n_neg_groups
    yg = np.r_[np.ones(n_pos_groups, int), np.zeros(n_neg_groups, int)]
    u = rng.normal(0.0, p.tau, size=(G, D_E))
    v = rng.normal(0.0, p.tau, size=(G, D_C))
    gi = np.repeat(np.arange(G), m)
    y = yg[gi]
    s = (2 * y - 1).astype(float)
    xe = p.mu_e * s[:, None] * A_VEC[None, :] + u[gi] + rng.normal(size=(len(gi), D_E))
    muc = p.mu_c * (1.0 + p.lam * np.sign(xe[:, GATE_COL]))
    xc = (muc * s)[:, None] * B_VEC[None, :] + v[gi] + rng.normal(size=(len(gi), D_C))
    groups = np.array([f"{group_prefix}{k:04d}" for k in gi])
    return dict(X_edge=xe.astype(np.float32), X_cloud=np.column_stack([xe, xc]).astype(np.float32),
                y=y, groups=groups)


def generate(scenario: str | Params, seed: int, G: int = 40, m: int = 25) -> dict:
    """One synthetic dataset (G/2 positive and G/2 negative groups, m windows each), seeded."""
    p = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)
    d = sample(p, rng, G // 2, G - G // 2, m)
    d.update(params=asdict(p), seed=int(seed), name=str(scenario) if isinstance(scenario, str) else "custom",
             names_edge=[f"e{k}" for k in range(D_E)],
             names_cloud=[f"e{k}" for k in range(D_E)] + [f"c{k}" for k in range(D_C)])
    return d
