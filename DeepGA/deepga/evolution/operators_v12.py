# -*- coding: utf-8 -*-
""" Created on 2026
    Operadores Evolutivos y de Diversidad para DeepGA V12 (Pure Multi-Island NAS sin Feromonas):
    - Modelo de Islas Aisladas sin Feromonas (Eliminación de sesgos y atractores artificiales).
    - Mutación Estocástica Estructural y Paramétrica Balanceada (Capa +, Capa -, Filtros, Kernels, Skips).
    - Tasa de Mutación Adaptativa guiada por Diversidad Poblacional (Srinivas + Diversity Feedback).
    - Cruce Topológico de Grafos Coherente V7 (Graph-Based Topological Recombination).
    - Migración Espaciada con Reemplazo por Nichos (Deterministic Crowding / Distance-based Replacement).
    - Mecanismo Anti-Estancamiento por Isla (Hipermutación de Choque e Inyección de Inmigrantes Aleatorios).
    - Métricas Continuas de Diversidad Intra e Inter-Islas.
"""

import random
import math
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from deepga.core.encoding import Encoding
from .operators_v7 import genome_to_graph, graph_to_genome, crossover_v7

'''Catálogo de Hiperparámetros Discretos'''
FSIZES = [3, 5, 7, 9]
NFILTERS = [8, 16, 32, 64, 128, 256]
PSIZES = [2, 3]
PTYPE = ['max', 'avg', 'off']
NEURONS = [16, 32, 64, 128, 256]


def generate_random_encoding(min_conv: int = 2, max_conv: int = 5,
                             min_full: int = 1, max_full: int = 4) -> Encoding:
    """
    Genera un genoma arquitectónico estocástico uniforme sin sesgo de feromona.
    """
    return Encoding(min_conv, max_conv, min_full, max_full)


def compute_individual_structural_vector(genome: Encoding, max_conv: int = 5, max_full: int = 4) -> np.ndarray:
    """
    Convierte un genoma en un vector numérico normalizado de características estructurales
    para el cálculo de distancias de nicho y diversidad poblacional.
    """
    vec = [
        float(genome.n_conv) / float(max_conv),
        float(genome.n_full) / float(max_full)
    ]

    # Convolucionales
    for i in range(max_conv):
        if i < genome.n_conv and i < len(genome.first_level):
            l = genome.first_level[i]
            vec.append(float(l.get('nfilters', 16)) / 256.0)
            vec.append(float(l.get('fsize', 3)) / 9.0)
            p_val = 1.0 if l.get('pool') == 'max' else (0.5 if l.get('pool') == 'avg' else 0.0)
            vec.append(p_val)
        else:
            vec.extend([0.0, 0.0, 0.0])

    # Densas
    fc_start = genome.n_conv
    for j in range(max_full):
        idx = fc_start + j
        if idx < len(genome.first_level):
            vec.append(float(genome.first_level[idx].get('neurons', 32)) / 256.0)
        else:
            vec.append(0.0)

    # Skip-connections
    max_skips = (max_conv * (max_conv - 1)) // 2
    skips = getattr(genome, 'second_level', [])
    for k in range(max_skips):
        if k < len(skips):
            vec.append(float(skips[k]))
        else:
            vec.append(0.0)

    return np.array(vec, dtype=np.float32)


def compute_structural_distance(ind1: Encoding, ind2: Encoding, max_conv: int = 5, max_full: int = 4) -> float:
    """Calcula la distancia euclidiana estructural normalizada entre dos genomas."""
    v1 = compute_individual_structural_vector(ind1, max_conv, max_full)
    v2 = compute_individual_structural_vector(ind2, max_conv, max_full)
    return float(np.linalg.norm(v1 - v2))


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
    Calcula la distancia estructural promedio entre los líderes de las diferentes islas.
    Un valor alto confirma que cada isla está explorando nichos arquitectónicos distintos.
    """
    n_islands = len(islands_pop)
    if n_islands < 2:
        return 0.0

    leaders = [max(island, key=lambda x: x[1])[0] for island in islands_pop if len(island) > 0]
    if len(leaders) < 2:
        return 0.0

    leader_vectors = [compute_individual_structural_vector(ldr, max_conv, max_full) for ldr in leaders]
    distances = []
    for i in range(len(leader_vectors)):
        for j in range(i + 1, len(leader_vectors)):
            dist = float(np.linalg.norm(leader_vectors[i] - leader_vectors[j]))
            distances.append(dist)

    return float(np.mean(distances)) if distances else 0.0


def compute_diversity_adaptive_mutation_rate(parent_fitness: float, f_max: float, f_avg: float, f_min: float,
                                             current_diversity: float, target_diversity: float = 0.25,
                                             mr_base: float = 0.40, mr_min: float = 0.15, mr_max: float = 0.85) -> float:
    """
    Calcula la tasa de mutación adaptativa combinando el fitness relativo del individuo
    con la diversidad observada en la subpoblación local.
    Si la diversidad cae por debajo de `target_diversity`, se eleva automáticamente mr.
    """
    eps = 1e-6
    if parent_fitness >= f_avg:
        mr = mr_min + (mr_base - mr_min) * ((f_max - parent_fitness) / (f_max - f_avg + eps))
    else:
        mr = mr_base + (mr_max - mr_base) * ((f_avg - parent_fitness) / (f_avg - f_min + eps))

    # Refuerzo por retroalimentación de diversidad
    if current_diversity < target_diversity:
        deficit_ratio = (target_diversity - current_diversity) / max(eps, target_diversity)
        mr += 0.25 * deficit_ratio

    return float(max(mr_min, min(mr_max, mr)))


def stochastic_mutation_v12(genome: Encoding, mr: float,
                            min_conv: int = 2, max_conv: int = 5,
                            min_full: int = 1, max_full: int = 4,
                            structural_prob: float = 0.20):
    """
    Mutación estocástica pura V12 para DeepGA:
    1. Mutación paramétrica uniforme (filtros, kernels, pool, neuronas).
    2. Mutación estructural (adición/eliminación de capas conv o FC con probabilidad structural_prob).
    3. Mutación de skip-connections residuales.
    """
    # 1. Mutación Estructural (Agregar o Eliminar Capas)
    if random.random() < (mr * structural_prob):
        # Decidir si modificar capas convolucionales o densas
        if random.random() < 0.5:
            # Modificación de capas convolucionales
            if genome.n_conv < max_conv and (random.random() < 0.5 or genome.n_conv <= min_conv):
                # Añadir capa convolucional
                new_conv = {
                    'nfilters': random.choice(NFILTERS),
                    'fsize': random.choice(FSIZES),
                    'pool': random.choice(PTYPE),
                    'psize': random.choice(PSIZES)
                }
                insert_pos = random.randint(0, genome.n_conv)
                genome.first_level.insert(insert_pos, new_conv)
                genome.n_conv += 1
                
                # Regenerar skip-connections para el nuevo tamaño
                total_skips = (genome.n_conv * (genome.n_conv - 1)) // 2
                new_skips = [random.choice([0, 1]) if random.random() < 0.25 else 0 for _ in range(total_skips)]
                genome.second_level = new_skips
            elif genome.n_conv > min_conv:
                # Eliminar capa convolucional
                del_pos = random.randint(0, genome.n_conv - 1)
                del genome.first_level[del_pos]
                genome.n_conv -= 1

                total_skips = (genome.n_conv * (genome.n_conv - 1)) // 2
                new_skips = [random.choice([0, 1]) if random.random() < 0.25 else 0 for _ in range(total_skips)]
                genome.second_level = new_skips
        else:
            # Modificación de capas densas
            if genome.n_full < max_full and (random.random() < 0.5 or genome.n_full <= min_full):
                new_fc = {'neurons': random.choice(NEURONS)}
                genome.first_level.append(new_fc)
                genome.n_full += 1
            elif genome.n_full > min_full:
                fc_del_idx = genome.n_conv + random.randint(0, genome.n_full - 1)
                if fc_del_idx < len(genome.first_level):
                    del genome.first_level[fc_del_idx]
                    genome.n_full -= 1

    # 2. Mutación Paramétrica Uniforme de Capas Convolucionales
    for i in range(min(genome.n_conv, len(genome.first_level))):
        layer = genome.first_level[i]
        if random.random() < mr:
            layer['nfilters'] = random.choice(NFILTERS)
        if random.random() < mr:
            layer['fsize'] = random.choice(FSIZES)
        if random.random() < mr:
            layer['pool'] = random.choice(PTYPE)
        if random.random() < mr:
            layer['psize'] = random.choice(PSIZES)

    # 3. Mutación Paramétrica de Capas Densas
    fc_start = genome.n_conv
    for j in range(genome.n_full):
        idx = fc_start + j
        if idx < len(genome.first_level):
            if random.random() < mr:
                genome.first_level[idx]['neurons'] = random.choice(NEURONS)

    # 4. Mutación de Conexiones Residuales (Skip-Connections)
    skips = getattr(genome, 'second_level', [])
    for k in range(len(skips)):
        if random.random() < mr:
            skips[k] = 1 - skips[k]  # Invertir bit (0 -> 1 o 1 -> 0)
    genome.second_level = skips


def perform_island_migration_v12(islands_pop: list, migration_size: int = 1,
                                 topology: str = "ring", max_conv: int = 5,
                                 max_full: int = 4) -> Tuple[list, list]:
    """
    Migración en anillo con política de Reemplazo por Nicho / Deterministic Crowding:
    1. Se selecciona el mejor individuo de cada isla de origen.
    2. En la isla de destino, el inmigrante compite directamente contra el individuo local
       que sea más parecido estructuralmente a él (menor distancia genómica).
    3. Si el inmigrante tiene mejor fitness que su vecino más cercano, lo reemplaza;
       preservando así la diversidad de los nichos arquitectónicos disimilares.
    """
    n_islands = len(islands_pop)
    if n_islands < 2 or migration_size <= 0:
        return islands_pop, []

    # Seleccionar emigrantes élite de cada isla
    emigrants_per_island = []
    for island_idx, pop in enumerate(islands_pop):
        sorted_pop = sorted(pop, key=lambda x: x[1], reverse=True)
        count = min(migration_size, len(sorted_pop))
        emigrants_per_island.append([deepcopy(ind) for ind in sorted_pop[:count]])

    new_islands_pop = []
    migration_records = []

    for dst_idx in range(n_islands):
        target_pop = deepcopy(islands_pop[dst_idx])
        src_idx = (dst_idx - 1 + n_islands) % n_islands
        incoming = emigrants_per_island[src_idx]

        accepted_count = 0
        for immigrant in incoming:
            imm_genome, imm_fit = immigrant[0], immigrant[1]
            
            # Buscar el individuo más similar estructuralmente en la población destino (Crowding)
            closest_idx = 0
            min_dist = float('inf')
            for local_idx, local_ind in enumerate(target_pop):
                dist = compute_structural_distance(imm_genome, local_ind[0], max_conv, max_full)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = local_idx

            # Si el inmigrante supera al más similar (o al peor de la isla), lo reemplaza
            worst_local_fit = min(ind[1] for ind in target_pop)
            if imm_fit > target_pop[closest_idx][1]:
                target_pop[closest_idx] = immigrant
                accepted_count += 1
            elif imm_fit > worst_local_fit:
                worst_idx = min(range(len(target_pop)), key=lambda i: target_pop[i][1])
                target_pop[worst_idx] = immigrant
                accepted_count += 1

        # Reordenar la población de la isla
        target_pop.sort(key=lambda x: x[1], reverse=True)
        new_islands_pop.append(target_pop)

        migration_records.append({
            "source_island": src_idx,
            "target_island": dst_idx,
            "num_immigrants_sent": len(incoming),
            "num_immigrants_accepted": accepted_count,
            "immigrant_fitnesses": [round(ind[1], 4) for ind in incoming],
            "immigrant_accuracies": [round(ind[2], 2) for ind in incoming]
        })

    return new_islands_pop, migration_records


def apply_island_anti_stagnation(island_pop: list, island_idx: int, stagnation_count: int,
                                 stagnation_limit: int = 4, min_conv: int = 2, max_conv: int = 5,
                                 min_full: int = 1, max_full: int = 4) -> Tuple[list, bool]:
    """
    Mecanismo de Cataclismo / Hipermutación de Choque:
    Si una isla no ha mejorado su mejor fitness en `stagnation_limit` generaciones,
    aplica hipermutación a la mitad inferior de la población e inyecta un genoma aleatorio.
    """
    if stagnation_count < stagnation_limit or len(island_pop) < 2:
        return island_pop, False

    print(f"   🌋 [Anti-Stagnation Trigger | Isla {island_idx+1}] Estancada por {stagnation_count} gens. Aplicando Hipermutación de Choque e Inmigrante Aleatorio...", flush=True)
    
    # Conservar el líder intacto (elitismo)
    sorted_pop = sorted(island_pop, key=lambda x: x[1], reverse=True)
    leader = sorted_pop[0]

    # Mutación agresiva para los demás individuos (mr = 0.75)
    for k in range(1, len(sorted_pop) - 1):
        ind_genome = deepcopy(sorted_pop[k][0])
        stochastic_mutation_v12(ind_genome, mr=0.75, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full, structural_prob=0.50)
        sorted_pop[k][0] = ind_genome

    # El peor individuo es reemplazado por un nuevo genoma estocástico fresco (Random Immigrant)
    fresh_genome = generate_random_encoding(min_conv, max_conv, min_full, max_full)
    sorted_pop[-1][0] = fresh_genome

    return sorted_pop, True
