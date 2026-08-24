# -*- coding: utf-8 -*-
""" Created on Sep 2024    @author: user
    Operadores Evolutivos Adaptativos para DeepGA V9:
    - Probabilidad de Mutación Adaptativa Individual (Performance-based Srinivas & Patnaik + Annealing).
    - Selección Inteligente del Tipo de Mutación (Micro-ajuste para altos rendimientos vs Macro-estructural para bajos).
    - Cruce Topológico Emparejado V7 (Graph-Based Crossover).
"""

import random
import math
from copy import deepcopy
from deepga.core.encoding import Encoding
from .operators_v7 import genome_to_graph, graph_to_genome, crossover_v7

'''Hyperparameters configuration'''
FSIZES = [3, 5, 7, 9]
NFILTERS = [8, 16, 32, 64, 128, 256]
PSIZES = [2, 3]
PTYPE = ['max', 'avg']
NEURONS = [16, 32, 64, 128, 256]


def compute_adaptive_mutation_rate(parent_fitness: float, f_max: float, f_avg: float, f_min: float,
                                   t: int, T: int, mr_base: float = 0.5,
                                   mr_min: float = 0.10, mr_max: float = 0.85,
                                   use_temporal: bool = True) -> float:
    """
    Calcula la probabilidad de mutación adaptativa para un individuo en función de su fitness
    relativo respecto a la población y el avance generacional (Srinivas & Patnaik + Annealing).

    - Individuo de alto rendimiento (fitness >= f_avg): mr disminuye hacia mr_min para proteger sus bloques genéticos.
    - Individuo de bajo rendimiento (fitness < f_avg): mr aumenta hacia mr_max para escapar de óptimos locales.
    - Avance temporal: reduce progresivamente la tasa a medida que nos acercamos a la generación final.
    """
    eps = 1e-6
    if parent_fitness >= f_avg:
        # Proteger individuos buenos reduciendo la tasa de mutación
        mr = mr_min + (mr_base - mr_min) * ((f_max - parent_fitness) / (f_max - f_avg + eps))
    else:
        # Promover la exploración en individuos de menor rendimiento
        mr = mr_base + (mr_max - mr_base) * ((f_avg - parent_fitness) / (f_avg - f_min + eps))

    if use_temporal and T > 1:
        # Modulación temporal suave (recocido simulado genético)
        time_factor = 1.0 - 0.25 * (float(t) / float(T))
        mr *= time_factor

    return float(max(mr_min, min(mr_max, mr)))


def adaptive_mutation_v9(x, mr_rate: float, parent_fitness: float = 0.5, f_avg: float = 0.5,
                         min_conv: int = 2, max_conv: int = 5, min_full: int = 1, max_full: int = 4):
    """
    Operador de mutación adaptativa estructurada:
    1. Evalúa si ocurre mutación usando la tasa adaptativa personalizada `mr_rate`.
    2. Adapta la intensidad de la mutación:
       - Alto rendimiento (parent_fitness >= f_avg): Realiza micro-ajustes (parámetros o un bit de skip).
       - Bajo rendimiento (parent_fitness < f_avg): Realiza cambios macro-estructurales (agregar/quitar capas).
    """
    if random.uniform(0, 1) > mr_rate:
        return False, 'none'

    conv_layers, fc_layers, adj_matrix = genome_to_graph(x)
    is_high_performer = (parent_fitness >= f_avg)

    if is_high_performer:
        # Alto rendimiento: 55% micro-ajuste de hiperparámetros, 25% invertir 1 conexión skip, 20% estructural
        mut_choice = random.choices(['modify_layer', 'flip_skip', 'add_remove_layer'], weights=[0.55, 0.25, 0.20])[0]
    else:
        # Bajo rendimiento: 50% mutación estructural (arquitectura), 40% hiperparámetros, 10% skip
        mut_choice = random.choices(['add_remove_layer', 'modify_layer', 'flip_skip'], weights=[0.50, 0.40, 0.10])[0]

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
