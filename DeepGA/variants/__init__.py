# -*- coding: utf-8 -*-
"""
Módulo de variantes para DeepGA / MODeepGA.
"""

from .DeepGA import deepGA, final_evaluation
from .DeepGA_V2 import green_DeepGA_v2
from .DeepGA_V3 import green_DeepGA_v3

__all__ = ["deepGA", "green_DeepGA_v2", "green_DeepGA_v3", "final_evaluation"]
