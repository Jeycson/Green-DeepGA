# -*- coding: utf-8 -*-
""" Created on 2026
    Operadores Evolutivos Multi-Objetivo para Modelo de Islas (MO-Island-ACO) en MO-DeepGA V11:
    - Modelo de Islas Multi-Objetivo (Multi-Island Pareto Neuroevolution).
    - Matrices de Feromonas Desacopladas por Cada Isla (Sin Contaminación Cruzada).
    - Migración en Anillo de Soluciones No Dominadas del Frente de Pareto.
    - Medición de Diversidad Inter-Islas y Cobertura de Nichos Arquitectónicos.
"""

import random
import math
from copy import deepcopy
import numpy as np

from deepga.core.encoding import Encoding
from .operators_mo_v10 import (
    MOPheromoneMatrix,
    mo_guided_adaptive_mutation_v10,
    compute_mo_adaptive_mutation_rate,
    crossover_v7,
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    tournament_selection_mo,
    calculate_hypervolume_2d
)
from .operators_v11 import compute_individual_structural_vector, compute_population_diversity


def compute_mo_inter_island_diversity(islands_pop: list, max_conv: int = 5, max_full: int = 4) -> float:
    """
    Calcula la distancia estructural promedio entre los mejores individuos del Frente de Pareto de cada isla.
    Un valor alto indica que las diferentes islas exploran nichos Pareto complementarios.
    """
    n_islands = len(islands_pop)
    if n_islands < 2:
        return 0.0

    island_leaders = []
    for island in islands_pop:
        if len(island) > 0:
            fronts = fast_non_dominated_sort(island)
            if fronts and len(fronts[0]) > 0:
                island_leaders.append(fronts[0][0][0])  # Genoma del primer no dominado

    if len(island_leaders) < 2:
        return 0.0

    leader_vectors = [compute_individual_structural_vector(ldr, max_conv, max_full) for ldr in island_leaders]
    distances = []
    for i in range(len(leader_vectors)):
        for j in range(i + 1, len(leader_vectors)):
            dist = float(np.linalg.norm(leader_vectors[i] - leader_vectors[j]))
            distances.append(dist)

    return float(np.mean(distances)) if distances else 0.0


def perform_island_migration_mo(islands_pop: list, migration_size: int = 2) -> tuple:
    """
    Ejecuta la migración de individuos Pareto-óptimos entre islas en topología de anillo:
    - Selecciona las mejores soluciones no dominadas (F1) de cada isla.
    - Se transfieren copias profundas de los individuos.
    - Las matrices de feromonas NO se transfieren (aislamiento estricto de feromonas).
    - En la isla receptora, se realiza reemplazo no dominado (NSGA-II) para mantener el tamaño.
    """
    n_islands = len(islands_pop)
    if n_islands < 2 or migration_size <= 0:
        return islands_pop, []

    # 1. Seleccionar migrantes no dominados de cada isla
    emigrants_per_island = []
    for island_idx, pop in enumerate(islands_pop):
        ranked_f = fast_non_dominated_sort(pop)
        for front in ranked_f:
            calculate_crowding_distance(front)

        # Tomar los de Frente 1 con mayor crowding distance
        front1 = ranked_f[0] if ranked_f else pop
        front1_sorted = sorted(front1, key=lambda x: x[3].get('crowding_dist', 0.0), reverse=True)
        num_to_send = min(migration_size, len(front1_sorted))
        emigrants = [deepcopy(ind) for ind in front1_sorted[:num_to_send]]
        emigrants_per_island.append(emigrants)

    # 2. Distribuir a la siguiente isla en anillo: Isla i -> Isla (i + 1) % n_islands
    new_islands_pop = []
    migration_records = []

    for dst_idx in range(n_islands):
        target_pop_size = len(islands_pop[dst_idx])
        src_idx = (dst_idx - 1 + n_islands) % n_islands

        incoming_immigrants = emigrants_per_island[src_idx]

        # Fusionar población local con inmigrantes
        combined = deepcopy(islands_pop[dst_idx]) + incoming_immigrants
        combined_fronts = fast_non_dominated_sort(combined)

        # Truncamiento NSGA-II para preservar tamaño de isla
        new_island_pop = []
        for front in combined_fronts:
            calculate_crowding_distance(front)
            if len(new_island_pop) + len(front) <= target_pop_size:
                new_island_pop.extend(front)
            else:
                front.sort(key=lambda x: x[3].get('crowding_dist', 0.0), reverse=True)
                needed = target_pop_size - len(new_island_pop)
                new_island_pop.extend(front[:needed])
                break

        new_islands_pop.append(new_island_pop)
        migration_records.append({
            'from_island': src_idx,
            'to_island': dst_idx,
            'num_migrants': len(incoming_immigrants),
            'migrants_acc': [round(m[1], 2) for m in incoming_immigrants],
            'migrants_carbon': [round(m[2], 4) for m in incoming_immigrants]
        })

    return new_islands_pop, migration_records
