"""Section 6: routers R0-R9 and the oracle ranking. Higher score = escalated first.

Notation: p_E edge probability (deployed edge), t_E its validation threshold,
p~ = calibrate_to_threshold(p_E, t_E). Validation-trained routers (R3-R7) see the validation
set; R8 is trained on the fit set with cross-fitted cloud correctness.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .config import PLAN, Config, PERIODIC_PHI
from . import models as _models  # noqa: F401  (installs the exact fast HGB weighted binning)
from .metrics import ba_weights, calibrate_to_threshold, escalation_gain

_EPS = 1e-12


def p_of_y(p, y):
    p = np.asarray(p, float); y = np.asarray(y).astype(int)
    return np.where(y == 1, p, 1.0 - p)


# ------------------------------------------------------------------ simple routers
def r0_random(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def r1_confidence(p_e, t_e) -> np.ndarray:
    return -np.abs(calibrate_to_threshold(p_e, t_e) - 0.5)


def r2_entropy(p_e) -> np.ndarray:
    p = np.clip(np.asarray(p_e, float), _EPS, 1 - _EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def r3_conformal(p_e_val, y_val, t_e, p_e_test) -> np.ndarray:
    """Split conformal: nonconformity a = 1 - p~(y) on validation; p-value of label c for a test
    window is (1 + #{a_val >= a_c}) / (n_val + 1); score = min(p_0, p_1)."""
    a_val = np.sort(1.0 - p_of_y(calibrate_to_threshold(p_e_val, t_e), y_val))
    n = len(a_val)
    pt = calibrate_to_threshold(p_e_test, t_e)

    def pval(a):
        ge = n - np.searchsorted(a_val, a - 1e-15, side="left")     # count a_val >= a
        return (1.0 + ge) / (n + 1.0)
    return np.minimum(pval(1.0 - (1.0 - pt)), pval(1.0 - pt))


def r9_periodic(order) -> np.ndarray:
    v = np.asarray(order, float) * PERIODIC_PHI
    return -(v - np.floor(v))


def oracle_score(g) -> np.ndarray:
    return np.asarray(g, float).copy()


# ------------------------------------------------------------------ learned regressors (R4-R7)
def _gbr(seed: int, cfg: Config):
    return HistGradientBoostingRegressor(**cfg.router_gb, early_stopping=False, random_state=seed)


def router_inputs(X_edge, p_e) -> np.ndarray:
    return np.column_stack([np.asarray(X_edge, float), np.asarray(p_e, float)])


def router_targets(y_val, p_e_val, p_c_val, t_e, t_c) -> dict:
    y = np.asarray(y_val).astype(int)
    pe = (np.asarray(p_e_val) >= t_e).astype(int); pc = (np.asarray(p_c_val) >= t_c).astype(int)
    d01 = (pc == y).astype(float) - (pe == y).astype(float)
    return dict(diff01=d01, diffprob=p_of_y(p_c_val, y) - p_of_y(p_e_val, y),
                g=escalation_gain(y, pe, pc), w=ba_weights(y))


def mean_one(sample_weight):
    """Rescale weights to mean 1 (relative weights unchanged). sklearn's HGB grower discards any
    child whose hessian sum (= weight sum for squared error) is below a fixed 1e-3, so BA weights
    that sum to 1 would drop leaves holding < 0.1% of the weight on large validation sets. With
    l2_regularization = 0 leaf values and split gains are scale-free, so this only removes that
    artefact."""
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, float)
    return w / w.mean()


def fit_regressor_router(Z_val, target, seed: int, cfg: Config = PLAN, sample_weight=None):
    m = _gbr(seed, cfg)
    m.fit(Z_val, target, sample_weight=mean_one(sample_weight))
    return m


def learned_routers(Z_val, Z_test, tg: dict, seed: int, cfg: Config = PLAN) -> dict:
    """R4 Diff-01, R5 Diff-Prob, R6 metric-weighted gain (BA sample weights from val labels,
    rescaled to mean 1 inside fit_regressor_router)."""
    out = {}
    out["R4"] = fit_regressor_router(Z_val, tg["diff01"], seed, cfg).predict(Z_test)
    out["R5"] = fit_regressor_router(Z_val, tg["diffprob"], seed, cfg).predict(Z_test)
    out["R6"] = fit_regressor_router(Z_val, tg["g"], seed, cfg, sample_weight=tg["w"]).predict(Z_test)
    return out


# ------------------------------------------------------------------ R7 explanation/stability/drift
def explanation_inputs(X, lr_model, X_fit_ref, seed: int, cfg: Config = PLAN) -> np.ndarray:
    """Inputs added by R7 for pair A. Attributions a = coef * z, z the standardised edge features
    (the LR's own fit-set scaler). Returns (n, top + 6):
      top-`r7_top` sorted |a| / L1 (zero padded), log L1, attribution entropy, signed sum, max |a|,
      stability (mean cosine of a with attributions under `r7_perturb` Gaussian perturbations of
      sd 0.05 in standardised units), drift (mean clipped |robust z| against fit median and IQR)."""
    X = np.asarray(X, float)
    z = lr_model.scaler.transform(X)
    coef = lr_model.coef
    a = z * coef
    absa = np.abs(a)
    l1 = absa.sum(axis=1)
    pn = absa / np.maximum(l1, _EPS)[:, None]
    top = -np.sort(-pn, axis=1)[:, :cfg.r7_top]
    if top.shape[1] < cfg.r7_top:
        top = np.pad(top, ((0, 0), (0, cfg.r7_top - top.shape[1])))
    ent = -(pn * np.log(pn + _EPS)).sum(axis=1)
    rng = np.random.default_rng(seed)
    na = np.linalg.norm(a, axis=1)
    cos = np.zeros(len(X))
    for _ in range(cfg.r7_perturb):
        ap = (z + rng.normal(0.0, cfg.r7_sd, size=z.shape)) * coef
        cos += (a * ap).sum(axis=1) / np.maximum(na * np.linalg.norm(ap, axis=1), _EPS)
    stab = cos / cfg.r7_perturb
    Xf = np.asarray(X_fit_ref, float)
    med = np.median(Xf, axis=0)
    iqr = np.subtract(*np.percentile(Xf, [75, 25], axis=0))
    iqr = np.where(iqr > _EPS, iqr, 1.0)
    drift = np.clip(np.abs((X - med) / iqr), 0, cfg.r7_clip).mean(axis=1)
    return np.column_stack([top, np.log(l1 + _EPS), ent, a.sum(axis=1), absa.max(axis=1), stab, drift])


def r7_router(Z_val, Z_test, E_val, E_test, tg: dict, seed: int, cfg: Config = PLAN) -> np.ndarray:
    m = fit_regressor_router(np.column_stack([Z_val, E_val]), tg["g"], seed, cfg, sample_weight=tg["w"])
    return m.predict(np.column_stack([Z_test, E_test]))


# ------------------------------------------------------------------ R8 learning to defer
class L2D(nn.Module):
    """Linear model with three logits (negative, positive, defer)."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 3)

    def forward(self, x):
        return self.lin(x)


def l2d_loss(logits, y, cloud_right, w_y):
    """Mozannar-Sontag cross-entropy surrogate: -w_y log p(y) - 1[cloud right] log p(defer)."""
    logp = torch.log_softmax(logits, dim=1)
    return (-(w_y * logp.gather(1, y[:, None]).squeeze(1)) - cloud_right * logp[:, 2]).mean()


def train_l2d(Z_fit, y_fit, cloud_right, seed: int, cfg: Config = PLAN):
    """Z_fit standardised edge features. Returns the trained module."""
    torch.manual_seed(seed)
    y = np.asarray(y_fit).astype(int)
    X = torch.as_tensor(np.asarray(Z_fit, np.float32))
    yt = torch.as_tensor(y)
    w = torch.as_tensor(ba_weights(y) * len(y), dtype=torch.float32)
    cr = torch.as_tensor(np.asarray(cloud_right, np.float32))
    net = L2D(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.l2d_lr, weight_decay=cfg.l2d_wd)
    for _ in range(cfg.l2d_epochs):
        opt.zero_grad()
        loss = l2d_loss(net(X), yt, cr, w)
        loss.backward()
        opt.step()
    net.final_loss = float(loss.detach())
    return net


@torch.no_grad()
def l2d_score(net, Z) -> np.ndarray:
    lg = net(torch.as_tensor(np.asarray(Z, np.float32))).numpy()
    return lg[:, 2] - np.maximum(lg[:, 0], lg[:, 1])


def crossfit_cloud_correct(make_cloud, X_cloud_fit, y_fit, groups_fit, t_c: float, n_folds: int = 3) -> np.ndarray:
    """1[cloud right] on the fit set from n-fold group cross-fitting of the cloud model
    (cloud family, cloud view), thresholded at the deployed cloud's validation threshold."""
    y = np.asarray(y_fit).astype(int)
    out = np.zeros(len(y))
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups_fit))))
    for tr, te in gkf.split(X_cloud_fit, y, groups_fit):
        if len(np.unique(y[tr])) < 2:                  # cannot fit: predict the only class seen
            pred = np.full(len(te), y[tr][0])
        else:
            m = make_cloud().fit(X_cloud_fit[tr], y[tr])
            pred = (m.predict_proba(X_cloud_fit[te]) >= t_c).astype(int)
        out[te] = (pred == y[te]).astype(float)
    return out
