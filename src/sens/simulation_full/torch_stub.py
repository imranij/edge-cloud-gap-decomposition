"""If PyTorch is not installed, register a minimal stand-in so that src/escalation/models.py and routers.py import.
The simulation uses only the logistic-regression and gradient-boosting code paths of those modules; the stand-in
covers their module-level torch use (set_num_threads, nn.Module base classes, the no_grad decorator) and raises if
any torch function is actually called."""
import sys
import types


def ensure() -> bool:
    try:
        import torch  # noqa: F401
        return False
    except ImportError:
        pass

    def _missing(*a, **k):
        raise RuntimeError("PyTorch is not installed; this code path needs it")

    t = types.ModuleType("torch")
    t.set_num_threads = lambda n: None
    t.no_grad = lambda: (lambda f: f)
    for cls in ("Tensor", "dtype", "device", "Size", "Generator"):   # array-API helpers (scipy, sklearn) test
        setattr(t, cls, type(cls, (), {}))                             # issubclass(x, torch.Tensor) etc.
    t.__getattr__ = lambda name: _missing
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    nn.__getattr__ = lambda name: _missing
    t.nn = nn
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = nn
    return True
