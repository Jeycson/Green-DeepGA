# -*- coding: utf-8 -*-
"""
Módulo de variantes para DeepGA / MODeepGA.
"""

from .DeepGA import deepGA, final_evaluation
from .DeepGA_V2 import green_DeepGA_v2
from .DeepGA_V3 import green_DeepGA_v3
from .DeepGA_V4 import green_DeepGA_v4
from .DeepGA_V5 import green_DeepGA_v5

__all__ = ["deepGA", "green_DeepGA_v2", "green_DeepGA_v3", "green_DeepGA_v4", "green_DeepGA_v5", "final_evaluation"]


