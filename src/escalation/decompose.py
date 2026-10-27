"""Section 5: factorial decomposition, stacker, combination and unresolved steps, self-oracle.

Factorial models are keyed (family, view): ef_ev = deployed edge, cf_ev = upgraded edge,
ef_cv = edge family on the cloud view, cf_cv = deployed cloud.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from .config import PLAN, Config
from . import models as _models  # noqa: F401  (installs the exact fast HGB weighted binning)
from .metrics import balanced_accuracy, oracle_predictions, ba_weights

KEYS = ("ef_ev", "cf_ev", "ef_cv", "cf_cv")


def factorial_ba(y_test, probs: dict, thr: dict, keys=KEYS) -> dict:
    """Balanced accuracy of each factorial model at its own validation threshold."""
    return {k: balanced_accuracy(y_test, (np.asarray(probs[k]) >= thr[k]).astype(int)) for k in keys}


def factorial_steps(ba: dict) -> dict:
    """ba: BA of the four factorial models. Returns the model, information and Shapley steps and
    the interaction. shapley_model + shapley_info == ba['cf_cv'] - ba['ef_ev'] exactly."""
    out = dict(model_step_ev=ba["cf_ev"] - ba["ef_ev"], model_step_cv=ba["cf_cv"] - ba["ef_cv"],
               info_step_ef=ba["ef_cv"] - ba["ef_ev"], info_step_cf=ba["cf_cv"] - ba["cf_ev"],
               interaction=ba["cf_cv"] - ba["cf_ev"] - ba["ef_cv"] + ba["ef_ev"],
               total_step=ba["cf_cv"] - ba["ef_ev"])
    out["shapley_model"] = 0.5 * (out["model_step_ev"] + out["model_step_cv"])
    out["shapley_info"] = 0.5 * (out["info_step_ef"] + out["info_step_cf"])
    return out


def threshold_composite(p_val, y_val, grid=None) -> float:
    """Sensitivity rule: maximise BA + 0.25 F1 + 0.10 recall on validation (ties towards 0.5),
    on the same grid as metrics.threshold_max_ba."""
    p = np.asarray(p_val, float); y = np.asarray(y_val).astype(int)
    if grid is None:
        grid = np.unique(np.concatenate([np.quantile(p, np.linspace(0, 1, 201)), np.linspace(0.01, 0.99, 99)]))
    P = p[None, :] >= np.asarray(grid)[:, None]
    tp = (P & (y == 1)).sum(1); fp = (P & (y == 0)).sum(1)
    n1 = max((y == 1).sum(), 1); n0 = max((y == 0).sum(), 1)
    rec = tp / n1; spec = 1 - fp / n0
    ba = 0.5 * (rec + spec)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    v = ba + 0.25 * f1 + 0.10 * rec
    best = v.max()
    cand = np.asarray(grid)[np.abs(v - best) <= 1e-12]
    return float(cand[np.argmin(np.abs(cand - 0.5))])


# ------------------------------------------------------------------ stacker
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Stacker:
    """'logistic': class-balanced logistic regression on the logits of both models;
    'gb': HistGradientBoosting (max_iter 100, max_depth 3, class-balanced weights) on both
    probabilities. Decision threshold 0.5."""

    def __init__(self, kind: str, seed: int, cfg: Config = PLAN):
        self.kind, self.seed, self.cfg = kind, seed, cfg

    def _Z(self, pa, pb):
        return np.column_stack([_logit(pa), _logit(pb)]) if self.kind == "logistic" else np.column_stack([pa, pb])

    def fit(self, pa, pb, y):
        y = np.asarray(y).astype(int)
        self.const = None
        if len(np.unique(y)) < 2:
            self.const = int(y[0]); return self
        if self.kind == "logistic":
            self.m = LogisticRegression(class_weight="balanced", max_iter=1000)
            self.m.fit(self._Z(pa, pb), y)
        else:
            self.m = HistGradientBoostingClassifier(**self.cfg.stacker_gb, early_stopping=False,
                                                    random_state=self.seed)
            self.m.fit(self._Z(pa, pb), y, sample_weight=ba_weights(y) * len(y))
        return self

    def predict_proba(self, pa, pb):
        if self.const is not None:
            return np.full(len(pa), float(self.const))
        return self.m.predict_proba(self._Z(pa, pb))[:, 1]

    def predict(self, pa, pb):
        return (self.predict_proba(pa, pb) >= 0.5).astype(int)


def _cv_folds(y, groups, k: int, seed: int):
    ng = len(np.unique(groups))
    if ng < 2:
        return []
    k = min(k, ng)
    try:
        sg = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(sg.split(np.zeros(len(y)), y, groups))
    except ValueError:
        return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))


def select_stacker(pa_val, pb_val, y_val, g_val, seed: int, cfg: Config = PLAN):
    """3-fold grouped CV inside validation; pick the candidate with the higher out-of-fold BA
    (pooled over folds; ties -> logistic), refit on all of validation. Returns (stacker, cv)."""
    y = np.asarray(y_val).astype(int)
    folds = _cv_folds(y, np.asarray(g_val), cfg.stacker_cv, seed)
    cv = {}
    for kind in ("logistic", "gb"):
        if not folds:
            cv[kind] = np.nan; continue
        oof = np.zeros(len(y), int)
        for tr, te in folds:
            oof[te] = Stacker(kind, seed, cfg).fit(pa_val[tr], pb_val[tr], y[tr]).predict(pa_val[te], pb_val[te])
        cv[kind] = balanced_accuracy(y, oof)
    kind = "gb" if (np.isfinite(cv["gb"]) and cv["gb"] > cv["logistic"] + 1e-12) else "logistic"
    return Stacker(kind, seed, cfg).fit(pa_val, pb_val, y), cv


def combination(y_test, pa_test, pb_test, ta, tb, pa_val, pb_val, y_val, g_val, seed, cfg: Config = PLAN,
                suffix: str = "") -> tuple[dict, np.ndarray]:
    """Combination step BA(stacker) - BA(cloud) and unresolved BA(oracle) - BA(stacker), with
    a = edge (deployed or upgraded) and b = cloud."""
    st, cv = select_stacker(pa_val, pb_val, y_val, g_val, seed, cfg)
    ps = st.predict(pa_test, pb_test)
    pe = (pa_test >= ta).astype(int); pc = (pb_test >= tb).astype(int)
    ba_st = balanced_accuracy(y_test, ps)
    ba_c = balanced_accuracy(y_test, pc)
    ba_or = balanced_accuracy(y_test, oracle_predictions(y_test, pe, pc))
    out = {f"ba_stacker{suffix}": ba_st, f"stacker_gb{suffix}": float(st.kind == "gb"),
           f"stacker_cv_logistic{suffix}": cv["logistic"], f"stacker_cv_gb{suffix}": cv["gb"],
           f"comb_step{suffix}": ba_st - ba_c, f"ba_oracle{suffix}": ba_or, f"unresolved{suffix}": ba_or - ba_st}
    return out, ps


def self_oracle_headroom(y_test, p_cloud, p_twin, t_cloud, t_twin) -> dict:
    a = (p_cloud >= t_cloud).astype(int); b = (p_twin >= t_twin).astype(int)
    ba_a, ba_b = balanced_accuracy(y_test, a), balanced_accuracy(y_test, b)
    ba_o = balanced_accuracy(y_test, oracle_predictions(y_test, a, b))
    return dict(ba_twin=ba_b, ba_oracle_self=ba_o, self_oracle_headroom=ba_o - max(ba_a, ba_b))
