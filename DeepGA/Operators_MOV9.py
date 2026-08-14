# -*- coding: utf-8 -*-
""" Created on 2026
    Operadores Evolutivos Multi-Objetivo para MO-DeepGA V9:
    - Dominancia de Pareto (Precisión vs Huella de Carbono).
    - Ordenamiento No Dominado Rápido (Fast Non-Dominated Sorting - NSGA-II).
    - Cálculo de Distancia de Apiñamiento (Crowding Distance).
    - Selección por Torneo Multi-Objetivo (Pareto Rank + Crowding Distance).
    - Probabilidad de Mutación Adaptativa Multi-Objetivo (Basada en Rango de Pareto + Apiñamiento).
    - Mutación Especializada Multi-Objetivo (Micro-ajustes para Frente 1 vs Macro-estructural para dominados).
    - Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
    - Métrica de Hipervolumen 2D (Hypervolume Indicator).
"""

import random
import math
import numpy as np
from copy import deepcopy
from EncodingClass import Encoding
from Operators_V7 import genome_to_graph, graph_to_genome, crossover_v7

'''Hyperparameters configuration'''
FSIZES = [3, 5, 7, 9]
NFILTERS = [8, 16, 32, 64, 128, 256]
PSIZES = [2, 3]
PTYPE = ['max', 'avg']
NEURONS = [16, 32, 64, 128, 256]


# =====================================================================
# 1. DOMINANCIA DE PARETO Y ORDENAMIENTO NO DOMINADO (NSGA-II)
# =====================================================================

def dominates(p, q) -> bool:
    """
    Determina si el individuo p domina al individuo q (p ≻ q) en el espacio bi-objetivo:
    - Objetivo 1: Precisión / Accuracy (p[1] >= q[1]) -> MAXIMIZAR
    - Objetivo 2: Huella de Carbono / gCO2eq (p[2] <= q[2]) -> MINIMIZAR

    Estructura de individuo: [genome, accuracy, carbon_gco2, metrics_dict, ...]
    """
    p_acc, p_carb = p[1], p[2]
    q_acc, q_carb = q[1], q[2]

    # p no es peor que q en ninguno de los objetivos
    not_worse = (p_acc >= q_acc) and (p_carb <= q_carb)
    # p es estrictamente mejor que q en al menos un objetivo
    strictly_better = (p_acc > q_acc) or (p_carb < q_carb)

    return not_worse and strictly_better


def fast_non_dominated_sort(pop: list) -> list:
    """
    Algoritmo de Ordenamiento No Dominado Rápido (NSGA-II, Deb et al., 2002).
    Clasifica la población en frentes de Pareto F = [F1, F2, ..., Fk].
    F1 es el frente no dominado (óptimos de Pareto).
    """
    population_size = len(pop)
    if population_size == 0:
        return []

    # S[p]: conjunto de individuos que domina p
    S = [[] for _ in range(population_size)]
    # n[p]: contador de individuos que dominan a p
    n = [0 for _ in range(population_size)]
    # rank[p]: número de frente al que pertenece p (1-indexed)
    rank = [0 for _ in range(population_size)]

    fronts = [[]]

    for p_idx in range(population_size):
        p = pop[p_idx]
        for q_idx in range(population_size):
            if p_idx == q_idx:
                continue
            q = pop[q_idx]

            if dominates(p, q):
                S[p_idx].append(q_idx)
            elif dominates(q, p):
                n[p_idx] += 1

        if n[p_idx] == 0:
            rank[p_idx] = 1
            fronts[0].append(p_idx)

    current_front_idx = 0
    while len(fronts[current_front_idx]) > 0:
        next_front = []
        for p_idx in fronts[current_front_idx]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    rank[q_idx] = current_front_idx + 2
                    next_front.append(q_idx)

        current_front_idx += 1
        fronts.append(next_front)

    # Eliminar el último frente vacío
    if len(fronts[-1]) == 0:
        fronts.pop()

    # Reconstruir frentes con los objetos de la población
    ranked_fronts = []
    for f_idx, front_indices in enumerate(fronts):
        front_members = []
        for idx in front_indices:
            ind = pop[idx]
            # Si el individuo tiene diccionario de métricas en ind[3], actualizar rank
            if len(ind) > 3 and isinstance(ind[3], dict):
                ind[3]['pareto_rank'] = f_idx + 1
            front_members.append(ind)
        ranked_fronts.append(front_members)

    return ranked_fronts


def calculate_crowding_distance(front: list) -> list:
    """
    Calcula la Distancia de Apiñamiento (Crowding Distance) para preservar
    la diversidad a lo largo de cada frente de Pareto en NSGA-II.

    Asigna distancia infinita a los extremos y la distancia normalizada a los puntos intermedios.
    Retorna la lista con los individuos anotados con su crowding distance.
    """
    n = len(front)
    if n == 0:
        return []
    if n == 1:
        if len(front[0]) > 3 and isinstance(front[0][3], dict):
            front[0][3]['crowding_dist'] = float('inf')
        return front
    if n == 2:
        for ind in front:
            if len(ind) > 3 and isinstance(ind[3], dict):
                ind[3]['crowding_dist'] = float('inf')
        return front

    # Inicializar distancias en 0.0
    distances = {id(ind): 0.0 for ind in front}

    # Objetivo 1: Accuracy (índice 1) - MAXIMIZAR
    front_acc = sorted(front, key=lambda x: x[1])
    min_acc, max_acc = front_acc[0][1], front_acc[-1][1]
    acc_range = max_acc - min_acc
    if acc_range < 1e-6:
        acc_range = 1e-6

    distances[id(front_acc[0])] = float('inf')
    distances[id(front_acc[-1])] = float('inf')

    for i in range(1, n - 1):
        if distances[id(front_acc[i])] != float('inf'):
            diff = (front_acc[i + 1][1] - front_acc[i - 1][1]) / acc_range
            distances[id(front_acc[i])] += diff

    # Objetivo 2: Huella de Carbono (índice 2) - MINIMIZAR
    front_carb = sorted(front, key=lambda x: x[2])
    min_carb, max_carb = front_carb[0][2], front_carb[-1][2]
    carb_range = max_carb - min_carb
    if carb_range < 1e-6:
        carb_range = 1e-6

    distances[id(front_carb[0])] = float('inf')
    distances[id(front_carb[-1])] = float('inf')

    for i in range(1, n - 1):
        if distances[id(front_carb[i])] != float('inf'):
            diff = (front_carb[i + 1][2] - front_carb[i - 1][2]) / carb_range
            distances[id(front_carb[i])] += diff

    # Guardar en metadata del individuo
    for ind in front:
        cd = distances[id(ind)]
        if len(ind) > 3 and isinstance(ind[3], dict):
            ind[3]['crowding_dist'] = cd

    return front


def get_crowding_distance(ind: list) -> float:
    """Helper para obtener la distancia de apiñamiento de un individuo."""
    if len(ind) > 3 and isinstance(ind[3], dict):
        return ind[3].get('crowding_dist', 0.0)
    return 0.0


def get_pareto_rank(ind: list) -> int:
    """Helper para obtener el rango de Pareto de un individuo (1 es óptimo)."""
    if len(ind) > 3 and isinstance(ind[3], dict):
        return ind[3].get('pareto_rank', 1)
    return 1


def compare_crowding(ind1: list, ind2: list) -> int:
    """
    Operador de Comparación de Apiñamiento de NSGA-II (Crowded-Comparison Operator ≺_n).
    Retorna:
      -1 si ind1 es mejor que ind2
       1 si ind2 es mejor que ind1
       0 si son indistinguibles
    """
    r1, r2 = get_pareto_rank(ind1), get_pareto_rank(ind2)
    if r1 < r2:
        return -1
    elif r1 > r2:
        return 1

    d1, d2 = get_crowding_distance(ind1), get_crowding_distance(ind2)
    if d1 > d2:
        return -1
    elif d1 < d2:
        return 1
    return 0


# =====================================================================
# 2. SELECCIÓN POR TORNEO MULTI-OBJETIVO
# =====================================================================

def tournament_selection_mo(pop: list, t_size: int = 2) -> list:
    """
    Selección por Torneo Multi-Objetivo (NSGA-II):
    Elige al mejor candidato según (1) Menor Rango de Pareto, (2) Mayor Distancia de Apiñamiento.
    """
    participants = random.sample(pop, min(t_size, len(pop)))
    best = participants[0]
    for candidate in participants[1:]:
        if compare_crowding(candidate, best) < 0:
            best = candidate
    return best


# =====================================================================
# 3. MUTACIÓN ADAPTATIVA MULTI-OBJETIVO (V9)
# =====================================================================

def compute_mo_adaptive_mutation_rate(rank: int, max_rank: int, crowding_dist: float,
                                      t: int, T: int, mr_base: float = 0.5,
                                      mr_min: float = 0.10, mr_max: float = 0.85,
                                      use_temporal: bool = True) -> float:
    """
    Calcula la tasa de mutación adaptativa para un individuo en un contexto multi-objetivo:
    - Individuos en Frente 1 (Pareto Óptimo, rank = 1):
      Tasa baja (mr_min .. mr_base * 0.5) para proteger las soluciones no dominadas.
      Si la distancia de apiñamiento es baja (zona congestionada), aumenta ligeramente mr para dispersar el frente.
    - Individuos dominados (rank > 1):
      Tasa alta (mr_base .. mr_max) proporcional al grado de dominancia para explorar activamente.
    - Modulación temporal (Annealing):
      Reduce gradualmente la tasa hacia las últimas generaciones.
    """
    if rank == 1:
        # Frente de Pareto: proteger bloques genéticos de élite
        # Si está muy apiñado (crowding_dist < 0.5), damos un pequeño empuje para diversificar
        crowd_factor = 0.0
        if crowding_dist != float('inf') and crowding_dist < 0.5:
            crowd_factor = 0.05 * (0.5 - crowding_dist)
        mr = mr_min + crowd_factor
    else:
        # Individuo dominado: tasa proporcional al rango para escapar
        rank_ratio = float(rank - 1) / max(1.0, float(max_rank - 1))
        mr = mr_base + (mr_max - mr_base) * rank_ratio

    if use_temporal and T > 1:
        # Modulación temporal suave
        time_factor = 1.0 - 0.25 * (float(t) / float(T))
        mr *= time_factor

    return float(max(mr_min, min(mr_max, mr)))


def mo_adaptive_mutation_v9(x, mr_rate: float, rank: int = 1,
                            min_conv: int = 2, max_conv: int = 5,
                            min_full: int = 1, max_full: int = 4):
    """
    Mutación Adaptativa Especializada Multi-Objetivo V9:
    1. Si ocurre mutación según `mr_rate`:
       - Frente de Pareto (rank == 1):
         * 60% Micro-ajustes de filtros / kernels / pooling (optimización fina de precisión/energía).
         * 25% Inversión de 1 conexión residual skip (modulación de flujo de gradiente).
         * 15% Modificación estructural de capas.
       - Frentes dominados (rank > 1):
         * 50% Modificación estructural (agregar/remover capas) para buscar nuevas fronteras.
         * 35% Modificación de hiperparámetros de capas.
         * 15% Conexiones residuales.
    """
    if random.uniform(0, 1) > mr_rate:
        return False, 'none'

    conv_layers, fc_layers, adj_matrix = genome_to_graph(x)
    is_pareto_elite = (rank == 1)

    if is_pareto_elite:
        mut_choice = random.choices(['modify_layer', 'flip_skip', 'add_remove_layer'], weights=[0.60, 0.25, 0.15])[0]
    else:
        mut_choice = random.choices(['add_remove_layer', 'modify_layer', 'flip_skip'], weights=[0.50, 0.35, 0.15])[0]

    if mut_choice == 'modify_layer':
        if random.uniform(0, 1) < 0.7 and len(conv_layers) > 0:
            idx = random.randint(0, len(conv_layers) - 1)
            conv_layers[idx]['nfilters'] = random.choice(NFILTERS)
            conv_layers[idx]['fsize'] = random.choice(FSIZES)
            conv_layers[idx]['pool'] = random.choice(['max', 'avg', 'off'])
            conv_layers[idx]['psize'] = random.choice(PSIZES)
        elif len(fc_layers) > 0:
            idx = random.randint(0, len(fc_layers) - 1)
            fc_layers[idx]['neurons'] = random.choice(NEURONS)

    elif mut_choice == 'add_remove_layer':
        if random.uniform(0, 1) < 0.5:
            # Operar en capas convolucionales
            if len(conv_layers) < max_conv and random.uniform(0, 1) < 0.5:
                new_layer = {
                    'type': 'conv',
                    'nfilters': random.choice(NFILTERS),
                    'fsize': random.choice(FSIZES),
                    'pool': random.choice(['max', 'avg', 'off']),
                    'psize': random.choice(PSIZES)
                }
                insert_idx = random.randint(0, len(conv_layers))
                conv_layers.insert(insert_idx, new_layer)
                new_n = len(conv_layers)
                new_adj = [[0 for _ in range(new_n)] for _ in range(new_n)]
                for i in range(new_n):
                    for j in range(new_n):
                        old_i = i if i < insert_idx else i - 1
                        old_j = j if j < insert_idx else j - 1
                        if 0 <= old_i < len(adj_matrix) and 0 <= old_j < len(adj_matrix[old_i]):
                            new_adj[j][i] = adj_matrix[old_j][old_i]
                adj_matrix = new_adj
            elif len(conv_layers) > min_conv:
                del_idx = random.randint(0, len(conv_layers) - 1)
                conv_layers.pop(del_idx)
                adj_matrix = [row[:del_idx] + row[del_idx+1:] for idx, row in enumerate(adj_matrix) if idx != del_idx]
        else:
            # Operar en capas densas
            if len(fc_layers) < max_full and random.uniform(0, 1) < 0.5:
                fc_layers.append({'type': 'fc', 'neurons': random.choice(NEURONS)})
            elif len(fc_layers) > min_full:
                fc_layers.pop()

    else:  # 'flip_skip'
        n_c = len(conv_layers)
        if n_c > 2:
            i = random.randint(2, n_c - 1)
            j = random.randint(0, i - 2)
            if j < len(adj_matrix) and i < len(adj_matrix[j]):
                adj_matrix[j][i] = 1 - adj_matrix[j][i]

    mutated = graph_to_genome(conv_layers, fc_layers, adj_matrix, min_conv, max_conv, min_full, max_full)
    x.n_conv = mutated.n_conv
    x.n_full = mutated.n_full
    x.first_level = mutated.first_level
    x.second_level = mutated.second_level
    return True, mut_choice


# =====================================================================
# 4. MÉTRICA DE HIPERVOLUMEN 2D (HYPERVOLUME INDICATOR)
# =====================================================================

def calculate_hypervolume_2d(pareto_front: list, ref_point: tuple = (0.0, 50.0)) -> float:
    """
    Calcula el Hipervolumen (HV) 2D dominado por el frente de Pareto con respecto
    a un punto de referencia nadir (ref_acc, ref_carbon).

    - Objetivo 1: Accuracy (%) -> ref_point[0] (ej. 0.0%)
    - Objetivo 2: Carbon Footprint (gCO2eq) -> ref_point[1] (ej. 50.0 gCO2eq)

    Transforma el objetivo de minimización de carbono en maximización del ahorro:
      saving = max(0, ref_carbon - ind_carbon)
      gain_acc = max(0, ind_acc - ref_acc)
    Calcula el área 2D no superpuesta cubierta por el frente.
    """
    if len(pareto_front) == 0:
        return 0.0

    ref_acc, ref_carb = ref_point

    # Filtrar puntos válidos que mejoran al punto de referencia
    points = []
    for ind in pareto_front:
        acc = ind[1]
        carb = ind[2]
        if acc >= ref_acc and carb <= ref_carb:
            gain_acc = acc - ref_acc
            gain_carb_saving = ref_carb - carb
            points.append((gain_acc, gain_carb_saving))

    if len(points) == 0:
        return 0.0

    # Ordenar por ganancia de precisión ascendente (o ganancia de ahorro descendente)
    points.sort(key=lambda p: p[0])

    # Calcular área escalonada
    hv = 0.0
    prev_saving = 0.0
    for i in range(len(points) - 1, -1, -1):
        curr_acc, curr_saving = points[i]
        if curr_saving > prev_saving:
            hv += curr_acc * (curr_saving - prev_saving)
            prev_saving = curr_saving

    return round(float(hv), 4)
