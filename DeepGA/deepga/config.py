# -*- coding: utf-8 -*-
"""
Módulo de Configuración para DeepGA (Enfoque Versión 10).
Proporciona estructuras de datos fuertemente tipadas y serializables
para controlar todos los aspectos de la búsqueda evolutiva guiada por feromonas (V10).
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class DeepGAV10Config:
    """
    Configuración completa para DeepGA Versión 10 (ACO-Enhanced Pheromone-Guided Evolution).
    """

    # --- Parámetros Generales de la Población y Búsqueda ---
    pop_size: int = 10                  # N: Tamaño de la población de candidatos
    generations: int = 5                # T: Número de generaciones evolutivas
    crossover_rate: float = 0.80        # cr: Probabilidad de cruce genético
    mutation_rate: float = 0.15         # mr: Probabilidad de mutación base
    tournament_size: int = 3            # t_size: Tamaño de torneo para selección
    weight_params: float = 0.10         # w: Penalización de complejidad de parámetros en fitness
    seed: Optional[int] = 42            # Semilla pseudoaleatoria para reproducibilidad

    # --- Parámetros Específicos de V10 (ACO y Feromonas Arquitectónicas) ---
    alpha: float = 1.2                  # Sensibilidad / peso de feromonas en selección probabilística
    rho: float = 0.10                   # Tasa de evaporación de feromonas por generación
    top_k_ratio: float = 0.35           # Fracción de individuos élite que depositan feromonas
    mr_min: float = 0.10                # Tasa de mutación adaptativa mínima
    mr_max: float = 0.85                # Tasa de mutación adaptativa máxima
    pool_candidates_factor: int = 5     # Factor de sobre-muestreo para evaluación con meta-modelo
    kappa: float = 0.10                 # Factor de exploración UCB en el modelo subrogado

    # --- Espacio de Búsqueda Arquitectónica (para CNNs) ---
    min_conv: int = 2                   # Mínimo de bloques convolucionales
    max_conv: int = 5                   # Máximo de bloques convolucionales
    min_full: int = 1                   # Mínimo de capas densas (Fully Connected)
    max_full: int = 3                   # Máximo de capas densas
    max_params: int = 1_500_000         # Límite máximo permitido de parámetros

    # --- Configuración de Entrenamiento ---
    learning_rate: float = 0.001        # Tasa de aprendizaje (Adam)
    train_epochs: int = 2               # Épocas de entrenamiento rápido durante el GA
    final_epochs: int = 10              # Épocas para re-entrenar la mejor arquitectura encontrada
    batch_size: int = 64                # Tamaño del lote
    device: str = "auto"                # Dispositivo: "auto", "cuda", "cpu"
    memory_cleanup: bool = True         # Liberar memoria de GPU tras evaluar cada candidato
    preload_gpu: bool = False           # Precargar dataset completo en VRAM para máxima velocidad

    # --- Especificaciones del Dataset / Problema ---
    dataset_name: str = "custom"        # Nombre identificador del dataset
    n_channels: int = 3                 # Canales de entrada (3=RGB, 1=Grises)
    n_classes: int = 10                 # Número de clases a clasificar
    image_size: int = 32                # Resolución cuadrada de entrada (e.g. 32, 64, 128, 224)

    # --- Rutas de Salida y Monitoreo ---
    checkpoint_dir: str = "./checkpoints_v10"
    output_dir: str = "./results_v10"
    track_carbon: bool = True           # Medir emisiones de CO2 con CodeCarbon
    save_txt: bool = True               # Guardar resúmenes legibles en .txt
    execution: int = 1                  # Identificador numérico de ejecución

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


# Alias amigable para usar como configuración general
DeepGAConfig = DeepGAV10Config
