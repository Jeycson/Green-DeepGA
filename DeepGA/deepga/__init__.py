"""DeepGA research framework.

Algorithm implementations remain in the top-level ``variants`` package during
the compatibility transition. ``ExperimentManager`` is imported lazily so that
low-level primitives can be imported without loading all variants.
"""

__all__ = ["ExperimentManager", "data", "utils", "core", "training", "evolution", "experiment"]


def __getattr__(name):
    if name == "ExperimentManager":
        from .experiment.manager import ExperimentManager
        return ExperimentManager
    try:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    except Exception:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

