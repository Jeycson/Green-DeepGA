# -*- coding: utf-8 -*-
""" Created on Oct 2024    @author: user
    Operadores Evolutivos y de Migración para DeepGA V11 (Island Model + ACO Pheromones):
    - Modelo de Islas Aisladas (Island Model NAS).
    - Matrices de Feromonas Independientes y Desacopladas por Cada Isla (Cero Contaminación Cruzada).
    - Migración Periódica en Anillo (Ring Migration) de Individuos Élite.
    - Preservación y Monitoreo de Diversidad Inter-Islas.
    - Cruce Topológico Emparejado V7 y Mutación Adaptativa Guiada V10.
"""

import random
import math
from copy import deepcopy
import numpy as np

# Reutilizar operadores base probados de V10 y V7
from Operators_V10 import (
    PheromoneMatrix,
    guided_adaptive_mutation_v10,
    compute_adaptive_mutation_rate,
    FSIZES,
    NFILTERS,
    PSIZES,
    PTYPE,
    NEURONS
)
from Operators_V7 import genome_to_graph, graph_to_genome, crossover_v7
from EncodingClass import Encoding


def compute_individual_structural_vector(genome: Encoding, max_conv: int = 5, max_full: int = 4) -> np.ndarray:
    """
    Convierte un genoma en un vector numérico normalizado de estructura para cálculo de distancias
    y medición de diversidad inter-islas.
    """
    vec = [
        float(genome.n_conv) / float(max_conv),
        float(genome.n_full) / float(max_full)
    ]
    # Filtros y kernels
    for i in range(max_conv):
        if i < genome.n_conv and i < len(genome.first_level):
            l = genome.first_level[i]
            vec.append(float(l.get('nfilters', 16)) / 256.0)
            vec.append(float(l.get('fsize', 3)) / 9.0)
            pool_val = 1.0 if l.get('pool') == 'max' else (0.5 if l.get('pool') == 'avg' else 0.0)
            vec.append(pool_val)
        else:
            vec.extend([0.0, 0.0, 0.0])

    # Neuronas FC
    fc_start = genome.n_conv
    for j in range(max_full):
        idx = fc_start + j
        if idx < len(genome.first_level):
            vec.append(float(genome.first_level[idx].get('neurons', 32)) / 256.0)
        else:
            vec.append(0.0)

    # Conexiones residuales
    max_skips = (max_conv * (max_conv - 1)) // 2
    skips = getattr(genome, 'second_level', [])
    for k in range(max_skips):
        if k < len(skips):
            vec.append(float(skips[k]))
        else:
            vec.append(0.0)

    return np.array(vec, dtype=np.float32)


def compute_population_diversity(population: list, max_conv: int = 5, max_full: int = 4) -> float:
    """
    Calcula la diversidad estructural promedio dentro de una población (distancia euclidiana media).
    """
    if len(population) < 2:
        return 0.0

    vectors = [compute_individual_structural_vector(ind[0], max_conv, max_full) for ind in population]
    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = float(np.linalg.norm(vectors[i] - vectors[j]))
            distances.append(dist)

    return float(np.mean(distances)) if distances else 0.0


def compute_inter_island_diversity(islands_pop: list, max_conv: int = 5, max_full: int = 4) -> float:
    """
    Calcula la distancia estructural promedio entre los mejores individuos de las diferentes islas.
    Un valor alto indica que las islas están explorando nichos arquitectónicos distintos.
    """
    n_islands = len(islands_pop)
    if n_islands < 2:
        return 0.0

    island_leaders = [max(island, key=lambda x: x[1])[0] for island in islands_pop if len(island) > 0]
    if len(island_leaders) < 2:
        return 0.0

    leader_vectors = [compute_individual_structural_vector(ldr, max_conv, max_full) for ldr in island_leaders]
    distances = []
    for i in range(len(leader_vectors)):
        for j in range(i + 1, len(leader_vectors)):
            dist = float(np.linalg.norm(leader_vectors[i] - leader_vectors[j]))
            distances.append(dist)

    return float(np.mean(distances)) if distances else 0.0


def perform_island_migration(islands_pop: list, migration_size: int = 2, topology: str = "ring") -> tuple:
    """
    Ejecuta la migración de individuos entre islas de acuerdo a la topología especificada (por defecto anillo).
    
    Regla fundamental:
    - Se transfieren ÚNICAMENTE copias profundas de los individuos (genoma + fitness + métricas).
    - Las matrices de feromonas NO se tocan ni se migran para evitar contaminación cruzada.
    
    Retorna:
    - new_islands_pop: Lista de poblaciones actualizadas por isla manteniendo el tamaño de cada una.
    - migration_records: Lista de diccionarios describiendo el flujo de migrantes en esta generación.
    """
    n_islands = len(islands_pop)
    if n_islands < 2 or migration_size <= 0:
        return islands_pop, []

    # 1. Seleccionar los mejores `migration_size` individuos de cada isla como emisarios
    emigrants_per_island = []
    for island_idx, pop in enumerate(islands_pop):
        sorted_island = sorted(pop, key=lambda x: x[1], reverse=True)
        num_to_send = min(migration_size, len(sorted_island))
        # Copia profunda de los emisarios
        emigrants = [deepcopy(ind) for ind in sorted_island[:num_to_send]]
        emigrants_per_island.append(emigrants)

    # 2. Distribuir inmigrantes según la topología (anillo por defecto: Isla i -> Isla (i + 1) % n_islands)
    new_islands_pop = []
    migration_records = []

    for dst_island_idx in range(n_islands):
        target_pop_size = len(islands_pop[dst_island_idx])
        src_island_idx = (dst_island_idx - 1 + n_islands) % n_islands

        incoming_immigrants = emigrants_per_island[src_island_idx]

        # Fusionar población local con inmigrantes
        current_pop = deepcopy(islands_pop[dst_island_idx])
        initial_worst_fit = min(ind[1] for ind in current_pop) if current_pop else 0.0

        current_pop.extend(incoming_immigrants)
        # Ordenar por fitness y conservar los mejores N_island (elitismo con absorción de inmigrantes)
        current_pop.sort(key=lambda x: x[1], reverse=True)
        final_island_pop = current_pop[:target_pop_size]

        # Verificar si algún inmigrante logró integrarse en el top N de la isla
        successful_immigrants = sum(1 for ind in incoming_immigrants if ind in final_island_pop)

        rec = {
            "source_island": src_island_idx,
            "target_island": dst_island_idx,
            "num_immigrants_sent": len(incoming_immigrants),
            "num_immigrants_accepted": successful_immigrants,
            "immigrant_fitnesses": [round(ind[1], 4) for ind in incoming_immigrants],
            "immigrant_accuracies": [round(ind[2], 2) for ind in incoming_immigrants]
        }
        migration_records.append(rec)
        new_islands_pop.append(final_island_pop)

    return new_islands_pop, migration_records
