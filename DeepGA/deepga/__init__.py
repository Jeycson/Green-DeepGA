# -*- coding: utf-8 -*-
"""
DeepGA: Framework evolutivo de búsqueda de arquitecturas neuronales (NAS) y optimización general.
Versión preliminar centrada en DeepGA V10 (ACO-Enhanced Pheromone-Guided Evolution).
"""

__version__ = "0.1.0"

from deepga.config import DeepGAConfig, DeepGAV10Config

__all__ = [
    "__version__",
    "DeepGAConfig",
    "DeepGAV10Config",
    "DeepGASearch",
    "DeepGAV10Search",
    "BaseEvaluator",
    "CNNEvaluator",
    "CustomEvaluator",
]


def __getattr__(name: str):
    """Carga perezosa (lazy-loading) de componentes pesados para evitar dependencias innecesarias."""
    if name in ("DeepGASearch", "DeepGAV10Search"):
        from deepga.search import DeepGASearch, DeepGAV10Search
        return DeepGASearch if name == "DeepGASearch" else DeepGAV10Search
    elif name in ("BaseEvaluator", "CNNEvaluator", "CustomEvaluator"):
        from deepga.core.evaluator import BaseEvaluator, CNNEvaluator, CustomEvaluator
        mapping = {
            "BaseEvaluator": BaseEvaluator,
            "CNNEvaluator": CNNEvaluator,
            "CustomEvaluator": CustomEvaluator
        }
        return mapping[name]
    raise AttributeError(f"El módulo 'deepga' no tiene el atributo '{name}'.")
