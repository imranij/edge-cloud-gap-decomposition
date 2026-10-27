"""Section 4: model families of pairs A, B and C.

Every model exposes fit(X_fit, y_fit, X_val=None, y_val=None) -> self and predict_proba(X) -> p (n,),
the probability of the positive class. `make_model(pair, family, seed, cfg)` is the factory;
family is 'edge' or 'cloud' (the family, not the view: views are chosen by the caller).
"""
from __future__ import annotations
import io
import pickle
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from .config import PLAN, Config
from .metrics import balanced_accuracy, ba_weights

torch.set_num_threads(2)


def balanced_sample_weight(y) -> np.ndarray:
    """Class-balanced weights n / (2 n_c) (mean 1)."""
    return ba_weights(y) * len(y)


# ---------------------------------------------------------------- tabular families
class LRModel:
    """Pair A edge family: standardised, L2, C = 1, class-balanced logistic regression."""
    kind = "lr"

    def __init__(self, seed: int, cfg: Config = PLAN):
        self.est = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=cfg.lr_C, l1_ratio=0.0, class_weight="balanced",
                                                    max_iter=cfg.lr_max_iter, random_state=seed))

    def fit(self, X, y, X_val=None, y_val=None):
        self.est.fit(np.asarray(X), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(np.asarray(X))[:, 1]

    @property
    def scaler(self):
        return self.est[0]

    @property
    def coef(self):
        return self.est[-1].coef_.ravel()


class BRFModel:
    """Pair A cloud family: balanced random forest."""
    kind = "brf"

    def __init__(self, seed: int, cfg: Config = PLAN):
        self.est = BalancedRandomForestClassifier(
            n_estimators=cfg.brf_trees, min_samples_leaf=cfg.brf_min_leaf, max_features="sqrt",
            sampling_strategy="all", replacement=True, bootstrap=False, n_jobs=2, random_state=seed)

    def fit(self, X, y, X_val=None, y_val=None):
        self.est.fit(np.asarray(X), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(np.asarray(X))[:, 1]


class HGBModel:
    """Pair B families: HistGradientBoosting with class-balanced sample weights. Early stopping
    is disabled so that max_iter is the number of iterations the plan states."""
    kind = "hgb"

    def __init__(self, seed: int, params: dict):
        self.est = HistGradientBoostingClassifier(**params, early_stopping=False, random_state=seed)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y).astype(int)
        self.est.fit(np.asarray(X), y, sample_weight=balanced_sample_weight(y))
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(np.asarray(X))[:, 1]


# ---------------------------------------------------------------- CNN families
class TinyCNN(nn.Module):
    """Pair C edge family: two conv layers (8, 16 filters, stride 2), GAP, linear head."""

    def __init__(self, c_in: int, k: int = 7):
        super().__init__()
        p = k // 2
        self.features = nn.Sequential(
            nn.Conv1d(c_in, 8, k, stride=2, padding=p), nn.ReLU(),
            nn.Conv1d(8, 16, k, stride=2, padding=p), nn.ReLU())
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=2))


class LargeCNN(nn.Module):
    """Pair C cloud family: four conv blocks (32, 32, 64, 64; conv-BN-ReLU, stride 2), GAP,
    dropout 0.2, linear head."""

    def __init__(self, c_in: int, k: int = 7):
        super().__init__()
        p = k // 2
        layers, c = [], c_in
        for co in (32, 32, 64, 64):
            layers += [nn.Conv1d(c, co, k, stride=2, padding=p, bias=False), nn.BatchNorm1d(co), nn.ReLU()]
            c = co
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        return self.head(self.drop(self.features(x).mean(dim=2)))


class CNNModel:
    """Adam (lr 1e-3, batch 128), class-balanced cross-entropy, <= 30 epochs, early stopping on
    validation balanced accuracy (argmax decision, patience 5, best weights restored),
    per-channel standardisation with fit-set statistics."""
    kind = "cnn"

    def __init__(self, seed: int, size: str, cfg: Config = PLAN):
        self.seed, self.size, self.cfg = seed, size, cfg
        self.net = None
        self.history = []

    def _std(self, X):
        X = np.asarray(X, dtype=np.float32)
        return torch.from_numpy((X - self.mu) / self.sd)

    def fit(self, X, y, X_val=None, y_val=None):
        cfg = self.cfg
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y).astype(int)
        self.mu = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        sd = X.std(axis=(0, 2), keepdims=True)
        self.sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
        c_in = X.shape[1]
        self.net = (TinyCNN if self.size == "tiny" else LargeCNN)(c_in, cfg.cnn_kernel)
        Xt = self._std(X); yt = torch.from_numpy(y)
        n1 = max(int(y.sum()), 1); n0 = max(len(y) - int(y.sum()), 1)
        w = torch.tensor([len(y) / (2 * n0), len(y) / (2 * n1)], dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.cnn_lr)
        best, best_state, bad = -np.inf, None, 0
        has_val = X_val is not None and y_val is not None and len(np.unique(y_val)) == 2
        for ep in range(cfg.cnn_epochs):
            self.net.train()
            perm = rng.permutation(len(y))
            for b in range(0, len(y), cfg.cnn_batch):
                ix = torch.from_numpy(perm[b:b + cfg.cnn_batch])
                if len(ix) < 2 and isinstance(self.net, LargeCNN):
                    continue                                   # batch-norm needs > 1 sample
                opt.zero_grad()
                loss = lossf(self.net(Xt[ix]), yt[ix])
                loss.backward()
                opt.step()
            if not has_val:
                continue
            v = balanced_accuracy(y_val, (self.predict_proba(X_val) >= 0.5).astype(int))
            self.history.append(v)
            if v > best + 1e-12:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.cnn_patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.epochs_run = ep + 1
        return self

    @torch.no_grad()
    def predict_proba(self, X, batch: int = 1024):
        self.net.eval()
        out = []
        for b in range(0, len(X), batch):
            xb = self._std(np.asarray(X[b:b + batch], dtype=np.float32))
            out.append(torch.softmax(self.net(xb), dim=1)[:, 1].numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


# ---------------------------------------------------------------- factory
PAIR_KIND = {"A": "tabular", "B": "tabular", "C": "cnn"}
FAMILY_NAMES = {("A", "edge"): "logistic regression", ("A", "cloud"): "balanced random forest",
                ("B", "edge"): "small HGB", ("B", "cloud"): "large HGB",
                ("C", "edge"): "tiny 1D-CNN", ("C", "cloud"): "large 1D-CNN"}


def make_model(pair: str, family: str, seed: int, cfg: Config = PLAN):
    if pair == "A":
        return LRModel(seed, cfg) if family == "edge" else BRFModel(seed, cfg)
    if pair == "B":
        return HGBModel(seed, cfg.hgb_small if family == "edge" else cfg.hgb_large)
    if pair == "C":
        return CNNModel(seed, "tiny" if family == "edge" else "large", cfg)
    raise ValueError(pair)


# ---------------------------------------------------------------- footprint
def _tree_expected_visits(children_left, children_right, counts) -> float:
    """Expected number of internal (split) nodes visited by a window, weighting branches by the
    training counts that reached them."""
    counts = np.asarray(counts, float)
    root = counts[0] if counts[0] > 0 else 1.0
    internal = np.asarray(children_left) != -1
    return float(counts[internal].sum() / root)


def footprint(model, n_features_or_shape) -> dict:
    """Parameter count, serialised bytes and multiply-accumulates (MACs) per window. For trees,
    parameters are the node count and MACs are the expected number of nodes visited (comparisons)."""
    if isinstance(model, CNNModel):
        buf = io.BytesIO(); torch.save(model.net.state_dict(), buf)
        c, L = n_features_or_shape
        macs = [0]

        def hook(mod, inp, out):
            if isinstance(mod, nn.Conv1d):
                macs[0] += out.shape[1] * out.shape[2] * (mod.in_channels // mod.groups) * mod.kernel_size[0]
            elif isinstance(mod, nn.Linear):
                macs[0] += mod.in_features * mod.out_features
            elif isinstance(mod, nn.BatchNorm1d):
                macs[0] += out.shape[1] * out.shape[2]
        hs = [m.register_forward_hook(hook) for m in model.net.modules()
              if isinstance(m, (nn.Conv1d, nn.Linear, nn.BatchNorm1d))]
        model.net.eval()
        with torch.no_grad():
            model.net(torch.zeros(1, c, L))
        for h in hs:
            h.remove()
        params = int(sum(p.numel() for p in model.net.parameters()))
        return dict(params=params, bytes=len(buf.getvalue()), macs=int(macs[0] + 2 * c * L), unit="MAC")
    nbytes = len(pickle.dumps(model.est))
    d = int(n_features_or_shape if np.isscalar(n_features_or_shape) else np.prod(n_features_or_shape))
    if isinstance(model, LRModel):
        return dict(params=int(3 * d + 1), bytes=nbytes, macs=int(2 * d), unit="MAC")
    if isinstance(model, BRFModel):
        nodes = visits = 0
        for t in model.est.estimators_:
            tr = t.tree_
            nodes += tr.node_count
            visits += _tree_expected_visits(tr.children_left, tr.children_right, tr.n_node_samples)
        return dict(params=int(nodes), bytes=nbytes, macs=float(visits), unit="nodes visited")
    if isinstance(model, HGBModel):
        nodes = visits = 0
        for it in model.est._predictors:
            for pred in it:
                nd = pred.nodes
                nodes += len(nd)
                left = np.where(nd["is_leaf"], -1, nd["left"].astype(np.int64))
                visits += _tree_expected_visits(left, nd["right"], nd["count"])
        return dict(params=int(nodes), bytes=nbytes, macs=float(visits), unit="nodes visited")
    raise TypeError(type(model))


# ---------------------------------------------------------------- fast weighted binning
# scikit-learn 1.9 computes HistGradientBoosting bin thresholds with weighted percentiles when
# sample weights are given, one percentile at a time through the array-API layer, which costs
# seconds per fit (most of a pair-B split). `_fast_find_binning_thresholds` returns exactly the
# same thresholds (same sort, same averaged-inverted-CDF rule; verified against the library
# function in tests/test_models.py) with vectorised numpy, and is installed in place of the
# library function. The unweighted path is delegated to the original.
from sklearn.ensemble._hist_gradient_boosting import binning as _skbin  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

_ORIG_FIND = getattr(_skbin, "_orig_find_binning_thresholds", _skbin._find_binning_thresholds)


def _fast_find_binning_thresholds(col_data, max_bins, sample_weight=None):
    if sample_weight is None:
        return _ORIG_FIND(col_data, max_bins, sample_weight)
    missing_mask = np.isnan(col_data)
    if missing_mask.any():
        col_data = col_data[~missing_mask]
        sample_weight = sample_weight[~missing_mask]
    nnz = sample_weight != 0
    col_data = col_data[nnz]; sample_weight = sample_weight[nnz]
    sort_idx = np.argsort(col_data)
    col_data = col_data[sort_idx]; sample_weight = sample_weight[sort_idx]
    distinct = np.unique(col_data).astype(_skbin.X_DTYPE)
    if len(distinct) == 1:
        return np.asarray([])
    if len(distinct) <= max_bins:
        thr = sliding_window_view(distinct, 2).mean(axis=1)
    else:
        arr = np.asarray(col_data, dtype=np.float64)
        sw = np.asarray(sample_weight, dtype=np.float64)
        sidx = np.argsort(arr, axis=0)                       # as xp.argsort(stable=False)
        sws = sw[sidx]
        cdf = np.cumsum(sws)
        pr = np.linspace(0, 100, num=max_bins + 1)[1:-1]
        adj = pr / 100 * cdf[-1]
        adj = np.where(adj == 0, np.nextafter(adj, adj + 1), adj)
        mx = len(arr) - 1
        pi = np.clip(np.searchsorted(cdf, adj), 0, mx)
        frac_above = cdf[pi] - adj
        is_above = frac_above > np.finfo(np.float64).eps
        pip1 = np.clip(pi + 1, 0, mx)
        thr = np.where(is_above, arr[sidx[pi]], (arr[sidx[pi]] + arr[sidx[pip1]]) / 2)
    # same post-processing as the library function
    unique_bin_val = np.unique(thr)
    if unique_bin_val.shape[0] != thr.shape[0]:
        thr = unique_bin_val
    np.clip(thr, a_min=None, a_max=_skbin.ALMOST_INF, out=thr)
    return thr


if not hasattr(_skbin, "_orig_find_binning_thresholds"):
    _skbin._orig_find_binning_thresholds = _ORIG_FIND
    _skbin._find_binning_thresholds = _fast_find_binning_thresholds
