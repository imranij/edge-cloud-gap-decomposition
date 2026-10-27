"""Constants of the frozen analysis plan (plan/ANALYSIS_PLAN.md), in one place.

`PLAN` holds the confirmatory settings. `tiny()` returns a reduced copy used ONLY by the test
suite (`--tiny` flag of run_dataset.py) so that smoke tests finish in seconds; it is never the
default and results written with it are flagged in the run log.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, field

BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)
CAPTURE_RANGE = (0.05, 0.30)
H2_LEVELS = (0.1, 0.25, 0.5)
PILOTS = {"secom", "metropt3", "metropt-3", "metropt_3", "metropt"}
CONFIRMATORY = ("hydraulic", "skab", "seu", "threew")
EXPLANATION_TOKEN_BYTES = 96
PERIODIC_PHI = 0.618034


@dataclass(frozen=True)
class Config:
    # Section 3
    k_max: int = 5
    k_min: int = 3
    j_tabular: int = 20
    j_cnn: int = 10
    k_in_max: int = 4
    outer_seed: int = 1000
    inner_seed: int = 2000
    model_seed: int = 3000
    h2_seed: int = 4000
    router_seed: int = 5000        # R0 random, R7 perturbations (not fixed by the plan)
    audit_seed: int = 8000         # audit bootstrap (not fixed by the plan)
    # Section 4
    lr_C: float = 1.0
    lr_max_iter: int = 5000
    brf_trees: int = 300
    brf_min_leaf: int = 2
    hgb_small: dict = field(default_factory=lambda: dict(max_iter=50, max_depth=3, learning_rate=0.1))
    hgb_large: dict = field(default_factory=lambda: dict(max_iter=400, max_leaf_nodes=31, learning_rate=0.05,
                                                         l2_regularization=1.0))
    cnn_epochs: int = 30
    cnn_patience: int = 5
    cnn_batch: int = 128
    cnn_lr: float = 1e-3
    cnn_kernel: int = 7
    twin_offset: int = 7919
    # Section 5
    stacker_gb: dict = field(default_factory=lambda: dict(max_iter=100, max_depth=3))
    stacker_cv: int = 3
    # Section 6
    router_gb: dict = field(default_factory=lambda: dict(max_iter=50, max_depth=3, learning_rate=0.1))
    r7_top: int = 8
    r7_perturb: int = 5
    r7_sd: float = 0.05
    r7_clip: float = 10.0
    l2d_folds: int = 3
    l2d_epochs: int = 500
    l2d_lr: float = 0.01
    l2d_wd: float = 1e-4
    # Section 7 audit
    audit_folds: int = 5
    audit_boot: int = 1000
    audit_go: float = 0.52
    outcome_cut: float = 0.10
    tiny: bool = False


PLAN = Config()


def tiny() -> Config:
    """Test-only reduced settings (never used for confirmatory runs)."""
    return replace(PLAN, brf_trees=20, hgb_small=dict(max_iter=10, max_depth=3, learning_rate=0.1),
                   hgb_large=dict(max_iter=20, max_leaf_nodes=15, learning_rate=0.1, l2_regularization=1.0),
                   cnn_epochs=2, cnn_patience=1, stacker_gb=dict(max_iter=10, max_depth=3),
                   router_gb=dict(max_iter=10, max_depth=3, learning_rate=0.1), l2d_epochs=50,
                   audit_boot=50, lr_max_iter=500, tiny=True)
