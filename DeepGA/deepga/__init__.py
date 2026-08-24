"""DeepGA research framework.

Algorithm implementations remain in the top-level ``variants`` package during
the compatibility transition. ``ExperimentManager`` is imported lazily so that
low-level primitives can be imported without loading all variants.
"""

__all__ = ["ExperimentManager"]


def __getattr__(name):
    if name == "ExperimentManager":
        from .experiment.manager import ExperimentManager
        return ExperimentManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
