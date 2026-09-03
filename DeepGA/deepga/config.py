# -*- coding: utf-8 -*-
"""
Módulo de Configuración para DeepGA (Enfoque Versión 10).
Los valores por defecto corresponden exactamente a la configuración óptima
calibrada mediante irace ('irace_tuning/best_configuration_4.json').

La modificación de parámetros está protegida bajo 'expert_mode'.
"""

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

# Ruta al archivo de calibración óptima por irace
IRACE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "irace_tuning",
    "best_configuration_4.json"
)


def _load_irace_defaults() -> Dict[str, Any]:
    """Carga los parámetros óptimos desde best_configuration_4.json si existe."""
    if os.path.exists(IRACE_CONFIG_PATH):
        try:
            with open(IRACE_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Respaldo con los valores exactos de best_configuration_4.json
    return {
        "variant": "v10",
        "lr": 0.0009,
        "pop_size": 17,
        "generations": 32,
        "t_size": 5,
        "cr": 0.5436,
        "mr": 0.2321,
        "mr_min": 0.175,
        "mr_max": 0.9218,
        "final_epoch": 9,
        "pool_candidates_factor": 3,
        "kappa": 0.1464,
        "rho": 0.2824,
        "alpha": 0.527,
        "top_k_ratio": 0.2815,
        "n_islands": 1,
        "migration_interval": 32,
        "migration_size": 1
    }


IRACE_DEFAULTS = _load_irace_defaults()


@dataclass
class DeepGAV10Config:
    """
    Configuración para DeepGA Versión 10 (ACO-Enhanced Pheromone-Guided Evolution).
    Por defecto utiliza los hiperparámetros óptimos de irace (best_configuration_4).

    Si se modifican los hiperparámetros óptimos sin 'expert_mode=True',
    se emitirá un aviso advirtiendo sobre posibles pérdidas de rendimiento.
    """

    # --- Modo de Protección ---
    expert_mode: bool = False           # Debe ser True para modificar parámetros deliberadamente

    # --- Parámetros Calibrados Óptimos por irace (best_configuration_4) ---
    pop_size: int = IRACE_DEFAULTS.get("pop_size", 17)
    generations: int = IRACE_DEFAULTS.get("generations", 32)
    crossover_rate: float = IRACE_DEFAULTS.get("cr", 0.5436)
    mutation_rate: float = IRACE_DEFAULTS.get("mr", 0.2321)
    tournament_size: int = IRACE_DEFAULTS.get("t_size", 5)
    learning_rate: float = IRACE_DEFAULTS.get("lr", 0.0009)
    final_epochs: int = IRACE_DEFAULTS.get("final_epoch", 9)

    # Parámetros específicos de Feromonas y Subrogado V10
    alpha: float = IRACE_DEFAULTS.get("alpha", 0.527)
    rho: float = IRACE_DEFAULTS.get("rho", 0.2824)
    top_k_ratio: float = IRACE_DEFAULTS.get("top_k_ratio", 0.2815)
    mr_min: float = IRACE_DEFAULTS.get("mr_min", 0.175)
    mr_max: float = IRACE_DEFAULTS.get("mr_max", 0.9218)
    pool_candidates_factor: int = IRACE_DEFAULTS.get("pool_candidates_factor", 3)
    kappa: float = IRACE_DEFAULTS.get("kappa", 0.1464)

    # --- Parámetros de Estructura y Penalización ---
    weight_params: float = 0.10
    seed: Optional[int] = 42

    # --- Espacio de Búsqueda Arquitectónica (CNNs) ---
    min_conv: int = 2
    max_conv: int = 5
    min_full: int = 1
    max_full: int = 3
    max_params: int = 1_500_000

    # --- Entrenamiento y Recursos ---
    train_epochs: int = 2               # Épocas de evaluación rápida durante GA
    batch_size: int = 64
    device: str = "auto"
    memory_cleanup: bool = True
    preload_gpu: bool = False

    # --- Especificaciones del Dataset / Problema ---
    dataset_name: str = "custom"
    n_channels: int = 3
    n_classes: int = 10
    image_size: int = 32

    # --- Salidas y Monitoreo ---
    checkpoint_dir: str = "./checkpoints_v10"
    output_dir: str = "./results_v10"
    track_carbon: bool = True
    save_txt: bool = True
    execution: int = 1

    def __post_init__(self):
        """Valida si se han alterado los hiperparámetros óptimos de irace."""
        # Claves críticas calibradas por irace
        critical_keys = {
            "pop_size": IRACE_DEFAULTS.get("pop_size", 17),
            "generations": IRACE_DEFAULTS.get("generations", 32),
            "crossover_rate": IRACE_DEFAULTS.get("cr", 0.5436),
            "mutation_rate": IRACE_DEFAULTS.get("mr", 0.2321),
            "tournament_size": IRACE_DEFAULTS.get("t_size", 5),
            "learning_rate": IRACE_DEFAULTS.get("lr", 0.0009),
            "alpha": IRACE_DEFAULTS.get("alpha", 0.527),
            "rho": IRACE_DEFAULTS.get("rho", 0.2824),
            "top_k_ratio": IRACE_DEFAULTS.get("top_k_ratio", 0.2815),
            "final_epochs": IRACE_DEFAULTS.get("final_epoch", 9)
        }

        modified = []
        for key, opt_val in critical_keys.items():
            curr_val = getattr(self, key)
            if abs(curr_val - opt_val) > 1e-5:
                modified.append(f"{key}: {curr_val} (óptimo irace: {opt_val})")

        if modified and not self.expert_mode:
            warnings.warn(
                "\n⚠️ [ADVERTENCIA DE CONFIGURACIÓN ÓPTIMA]:\n"
                "Se han modificado parámetros respecto a la configuración óptima encontrada por irace:\n"
                + "\n".join([f"   • {m}" for m in modified]) +
                "\nModificar estos valores sin comprender sus efectos puede degradar la convergencia.\n"
                "Para suprimir este aviso en código, active 'expert_mode=True'.",
                UserWarning,
                stacklevel=2
            )

    @classmethod
    def optimal(cls) -> "DeepGAV10Config":
        """Retorna la configuración óptima estricta encontrada por irace."""
        return cls(expert_mode=False)

    @classmethod
    def custom(cls, **kwargs) -> "DeepGAV10Config":
        """Permite a un usuario avanzado definir parámetros personalizados en modo experto."""
        kwargs["expert_mode"] = True
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a un diccionario estándar."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepGAV10Config":
        """Instancia la configuración a partir de un diccionario."""
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, file_path: str):
        """Guarda la configuración en un archivo JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_json(cls, file_path: str) -> "DeepGAV10Config":
        """Carga una configuración desde un archivo JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Alias amigable
DeepGAConfig = DeepGAV10Config
