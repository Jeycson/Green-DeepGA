# -*- coding: utf-8 -*-
""" Created on 2026
    Operadores Evolutivos Multi-Objetivo Guiados por Feromonas (MO-ACO-NAS) para MO-DeepGA V10:
    - Matriz de Feromonas Arquitectónicas Multi-Objetivo (MOPheromoneMatrix).
    - Depósito de Feromonas Proporcional al Rango de Pareto (F1 Élites No Dominadas + Crowding Distance).
    - Evaporación Dinámica de Feromonas para Evitar Estancamiento Prematuro.
    - Inicialización y Mutación Adaptativa Guiadas por Distribución de Feromonas Pareto.
    - Re-exportación de Operadores Multi-Objetivo (NSGA-II, Cruce V7, Dominancia e Hipervolumen).
"""

import random
import math
from copy import deepcopy
import numpy as np
from deepga.core.encoding import Encoding
from .operators_v7 import genome_to_graph, graph_to_genome, crossover_v7
from .operators_mo_v9 import (
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    tournament_selection_mo,
    compute_mo_adaptive_mutation_rate,
    calculate_hypervolume_2d
)

'''Configuración de opciones discretas de hiperparámetros'''
FSIZES = [3, 5, 7, 9]
NFILTERS = [8, 16, 32, 64, 128, 256]
PSIZES = [2, 3]
PTYPE = ['max', 'avg', 'off']
NEURONS = [16, 32, 64, 128, 256]


# =====================================================================
# 1. MATRIZ DE FEROMONAS ARQUITECTÓNICAS MULTI-OBJETIVO (MO-ACO)
# =====================================================================

class MOPheromoneMatrix:
    """
    Matriz de Feromonas Arquitectónicas para Búsqueda Neuroevolutiva Multi-Objetivo (MO-ACO-NAS).
    Mantiene y actualiza las intensidades de feromona en base a las soluciones del Frente de Pareto:
    - Recompensa fuertemente a los individuos en el Frente 1 (No dominados).
    - Bonifica a las soluciones con mayor Distancia de Apiñamiento (diversidad en la frontera).
    - Guía la mutación y la inicialización hacia motivos arquitectónicos precisos y de bajo carbono.
    """
    def __init__(self, min_conv: int = 2, max_conv: int = 5,
                 min_full: int = 1, max_full: int = 4,
                 tau_0: float = 1.0, min_tau: float = 0.05, max_tau: float = 10.0,
                 rho: float = 0.10, alpha: float = 1.2):
        self.min_conv = min_conv
        self.max_conv = max_conv
        self.min_full = min_full
        self.max_full = max_full
        self.tau_0 = tau_0
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.rho = rho          # Tasa de evaporación
        self.alpha = alpha      # Sensibilidad de muestreo

        # 1. Feromonas para macroestructura
        self.conv_count_tau = {k: self.tau_0 for k in range(min_conv, max_conv + 1)}
        self.full_count_tau = {k: self.tau_0 for k in range(min_full, max_full + 1)}

        # 2. Feromonas por capa convolucional
        self.conv_layers_tau = []
        for _ in range(max_conv):
            layer_tau = {
                'nfilters': {nf: self.tau_0 for nf in NFILTERS},
                'fsize': {fs: self.tau_0 for fs in FSIZES},
                'pool': {pt: self.tau_0 for pt in PTYPE},
                'psize': {ps: self.tau_0 for ps in PSIZES}
            }
            self.conv_layers_tau.append(layer_tau)

        # 3. Feromonas por capa densa (FC)
        self.fc_layers_tau = []
        for _ in range(max_full):
            self.fc_layers_tau.append({
                'neurons': {n: self.tau_0 for n in NEURONS}
            })

        # 4. Feromonas para Skip-Connections (grafo triangular inferior)
        self.skip_tau = {}
        for dst in range(2, max_conv):
            for src in range(0, dst - 1):
                conn_key = f"{src}->{dst}"
                self.skip_tau[conn_key] = {0: self.tau_0, 1: self.tau_0}

    def _sample_categorical(self, tau_dict: dict, alpha: float = None) -> any:
        """Muestrea una opción basada en la intensidad de feromona con ponderación exponencial."""
        a = self.alpha if alpha is None else alpha
        options = list(tau_dict.keys())
        tau_values = [max(1e-5, float(tau_dict[k])) for k in options]
        powered = [t ** a for t in tau_values]
        sum_powered = sum(powered)
        if sum_powered <= 0:
            return random.choice(options)
        probs = [p / sum_powered for p in powered]
        return random.choices(options, weights=probs, k=1)[0]

    def evaporate(self, rho: float = None):
        """Aplica la evaporación natural de feromonas a todas las tablas."""
        r = self.rho if rho is None else rho

        def _evap_dict(d):
            for k in d:
                d[k] = max(self.min_tau, min(self.max_tau, (1.0 - r) * d[k]))

        _evap_dict(self.conv_count_tau)
        _evap_dict(self.full_count_tau)

        for l_tau in self.conv_layers_tau:
            for sub in l_tau.values():
                _evap_dict(sub)

        for fc_tau in self.fc_layers_tau:
            _evap_dict(fc_tau['neurons'])

        for sk_dict in self.skip_tau.values():
            _evap_dict(sk_dict)

    def deposit_pareto(self, ranked_fronts: list, top_k_ratio: float = 0.35, delta_factor: float = 1.5):
        """
        Deposita feromonas proporcionalmente a la jerarquía de Pareto:
        - Frente 1 (Pareto Óptimo): Depósito máximo (delta_factor * 1.5), ponderado por crowding distance.
        - Frentes dominados (F2, F3): Depósito decreciente (delta_factor / rank^1.5).
        - Frentes lejanos: Sin depósito para no reforzar estructuras subóptimas.
        """
        if not ranked_fronts:
            return

        for rank_idx, front in enumerate(ranked_fronts):
            rank = rank_idx + 1
            if rank > 3:
                break  # Solo los 3 primeros frentes depositan feromonas

            for ind in front:
                genome = ind[0]
                cd = ind[3].get('crowding_dist', 1.0) if len(ind) > 3 and isinstance(ind[3], dict) else 1.0
                crowd_mult = 1.2 if (cd == float('inf') or cd > 1.0) else 1.0

                if rank == 1:
                    delta_tau = delta_factor * 1.5 * crowd_mult
                elif rank == 2:
                    delta_tau = delta_factor * 0.6 * crowd_mult
                else:
                    delta_tau = delta_factor * 0.2

                # 1. Macroestructura
                n_c = genome.n_conv
                n_f = genome.n_full
                if n_c in self.conv_count_tau:
                    self.conv_count_tau[n_c] = min(self.max_tau, self.conv_count_tau[n_c] + delta_tau)
                if n_f in self.full_count_tau:
                    self.full_count_tau[n_f] = min(self.max_tau, self.full_count_tau[n_f] + delta_tau)

                # 2. Capas convolucionales
                for idx in range(min(n_c, len(genome.first_level), self.max_conv)):
                    layer = genome.first_level[idx]
                    nf = layer.get('nfilters')
                    fs = layer.get('fsize')
                    pt = layer.get('pool', 'off')
                    ps = layer.get('psize', 2)

                    if idx < len(self.conv_layers_tau):
                        lt = self.conv_layers_tau[idx]
                        if nf in lt['nfilters']:
                            lt['nfilters'][nf] = min(self.max_tau, lt['nfilters'][nf] + delta_tau)
                        if fs in lt['fsize']:
                            lt['fsize'][fs] = min(self.max_tau, lt['fsize'][fs] + delta_tau)
                        if pt in lt['pool']:
                            lt['pool'][pt] = min(self.max_tau, lt['pool'][pt] + delta_tau)
                        if ps in lt['psize']:
                            lt['psize'][ps] = min(self.max_tau, lt['psize'][ps] + delta_tau)

                # 3. Capas densas
                for f_idx in range(min(n_f, self.max_full)):
                    real_idx = n_c + f_idx
                    if real_idx < len(genome.first_level) and f_idx < len(self.fc_layers_tau):
                        fc_layer = genome.first_level[real_idx]
                        neu = fc_layer.get('neurons')
                        if neu in self.fc_layers_tau[f_idx]['neurons']:
                            self.fc_layers_tau[f_idx]['neurons'][neu] = min(
                                self.max_tau, self.fc_layers_tau[f_idx]['neurons'][neu] + delta_tau
                            )

                # 4. Skip-connections
                pos = 0
                prev = -1
                for dst in range(n_c):
                    if prev < 1:
                        prev += 1
                    elif prev >= 1:
                        for src in range(prev - 1):
                            bit_idx = pos + src
                            if bit_idx < len(genome.second_level):
                                bit_val = genome.second_level[bit_idx]
                                conn_key = f"{src}->{dst}"
                                if conn_key in self.skip_tau and bit_val in self.skip_tau[conn_key]:
                                    self.skip_tau[conn_key][bit_val] = min(
                                        self.max_tau, self.skip_tau[conn_key][bit_val] + delta_tau
                                    )
                        pos += (prev - 1)
                        prev += 1

    def generate_guided_encoding(self, min_conv: int = None, max_conv: int = None,
                                min_full: int = None, max_full: int = None,
                                alpha: float = None) -> Encoding:
        """Crea un individuo Encoding nuevo utilizando las distribuciones de feromonas actuales."""
        min_c = self.min_conv if min_conv is None else min_conv
        max_c = self.max_conv if max_conv is None else max_conv
        min_f = self.min_full if min_full is None else min_full
        max_f = self.max_full if max_full is None else max_full

        # Macroestructura guiada
        n_c = self._sample_categorical(self.conv_count_tau, alpha)
        n_f = self._sample_categorical(self.full_count_tau, alpha)
        n_c = max(min_c, min(max_c, n_c))
        n_f = max(min_f, min(max_f, n_f))

        first_level = []
        # Capas Convolucionales guiadas
        for i in range(n_c):
            lt = self.conv_layers_tau[i] if i < len(self.conv_layers_tau) else self.conv_layers_tau[-1]
            layer = {
                'type': 'conv',
                'nfilters': self._sample_categorical(lt['nfilters'], alpha),
                'fsize': self._sample_categorical(lt['fsize'], alpha),
                'pool': self._sample_categorical(lt['pool'], alpha),
                'psize': self._sample_categorical(lt['psize'], alpha)
            }
            first_level.append(layer)

        # Capas Densas guiadas
        for j in range(n_f):
            ft = self.fc_layers_tau[j] if j < len(self.fc_layers_tau) else self.fc_layers_tau[-1]
            layer = {
                'type': 'fc',
                'neurons': self._sample_categorical(ft['neurons'], alpha)
            }
            first_level.append(layer)

        # Skip connections guiadas
        second_level = []
        prev = -1
        for dst in range(n_c):
            if prev < 1:
                prev += 1
            elif prev >= 1:
                for src in range(prev - 1):
                    conn_key = f"{src}->{dst}"
                    if conn_key in self.skip_tau:
                        val = self._sample_categorical(self.skip_tau[conn_key], alpha)
                    else:
                        val = random.choice([0, 1])
                    second_level.append(val)
                prev += 1

        encoding = Encoding(min_c, max_c, min_f, max_f)
        encoding.n_conv = n_c
        encoding.n_full = n_f
        encoding.first_level = first_level
        encoding.second_level = second_level
        return encoding

    def get_top_motifs_summary(self) -> dict:
        """Extrae las decisiones arquitectónicas más reforzadas por el rastro de feromonas."""
        best_conv_count = max(self.conv_count_tau.items(), key=lambda x: x[1])[0]
        best_fc_count = max(self.full_count_tau.items(), key=lambda x: x[1])[0]

        top_conv_rules = []
        for i, lt in enumerate(self.conv_layers_tau):
            best_nf = max(lt['nfilters'].items(), key=lambda x: x[1])[0]
            best_fs = max(lt['fsize'].items(), key=lambda x: x[1])[0]
            best_pt = max(lt['pool'].items(), key=lambda x: x[1])[0]
            top_conv_rules.append({
                'layer': i,
                'favored_filters': best_nf,
                'favored_kernel': best_fs,
                'favored_pool': best_pt
            })

        top_skips = []
        for conn, sk_dict in self.skip_tau.items():
            tau_on = sk_dict.get(1, 1.0)
            tau_off = sk_dict.get(0, 1.0)
            prob_on = tau_on / (tau_on + tau_off)
            if prob_on > 0.60:
                top_skips.append((conn, round(prob_on * 100, 1)))

        return {
            'favored_conv_count': best_conv_count,
            'favored_fc_count': best_fc_count,
            'layer_motifs': top_conv_rules,
            'reinforced_skip_connections': top_skips
        }


# =====================================================================
# 2. MUTACIÓN ADAPTATIVA GUIADA POR FEROMONAS MULTI-OBJETIVO (V10)
# =====================================================================

def mo_guided_adaptive_mutation_v10(x, pheromones: MOPheromoneMatrix, mr_rate: float,
                                     rank: int = 1,
                                     min_conv: int = 2, max_conv: int = 5,
                                     min_full: int = 1, max_full: int = 4):
    """
    Operador de Mutación Adaptativa Guiada por Feromonas Multi-Objetivo (V10):
    1. Decide si mutar según la tasa adaptativa por rango `mr_rate`.
    2. Frente 1 (Pareto Óptimo): Micro-ajustes guiados por feromonas reforzadas.
    3. Frentes Dominados (Rank > 1): Re-estructuración macro guiada por feromonas para saltar hacia la frontera de Pareto.
    4. Cada nuevo hiperparámetro se muestrea de las distribuciones de feromonas.
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
            lt = pheromones.conv_layers_tau[idx] if idx < len(pheromones.conv_layers_tau) else pheromones.conv_layers_tau[-1]
            conv_layers[idx]['nfilters'] = pheromones._sample_categorical(lt['nfilters'])
            conv_layers[idx]['fsize'] = pheromones._sample_categorical(lt['fsize'])
            conv_layers[idx]['pool'] = pheromones._sample_categorical(lt['pool'])
            conv_layers[idx]['psize'] = pheromones._sample_categorical(lt['psize'])
        elif len(fc_layers) > 0:
            idx = random.randint(0, len(fc_layers) - 1)
            ft = pheromones.fc_layers_tau[idx] if idx < len(pheromones.fc_layers_tau) else pheromones.fc_layers_tau[-1]
            fc_layers[idx]['neurons'] = pheromones._sample_categorical(ft['neurons'])

    elif mut_choice == 'add_remove_layer':
        if random.uniform(0, 1) < 0.5:
            # Capas Convolucionales
            if len(conv_layers) < max_conv and random.uniform(0, 1) < 0.5:
                insert_idx = random.randint(0, len(conv_layers))
                lt = pheromones.conv_layers_tau[insert_idx] if insert_idx < len(pheromones.conv_layers_tau) else pheromones.conv_layers_tau[-1]
                new_layer = {
                    'type': 'conv',
                    'nfilters': pheromones._sample_categorical(lt['nfilters']),
                    'fsize': pheromones._sample_categorical(lt['fsize']),
                    'pool': pheromones._sample_categorical(lt['pool']),
                    'psize': pheromones._sample_categorical(lt['psize'])
                }
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
            # Capas Densas
            if len(fc_layers) < max_full and random.uniform(0, 1) < 0.5:
                f_idx = len(fc_layers)
                ft = pheromones.fc_layers_tau[f_idx] if f_idx < len(pheromones.fc_layers_tau) else pheromones.fc_layers_tau[-1]
                fc_layers.append({'type': 'fc', 'neurons': pheromones._sample_categorical(ft['neurons'])})
            elif len(fc_layers) > min_full:
                fc_layers.pop()

    else:  # 'flip_skip'
        n_c = len(conv_layers)
        if n_c > 2:
            i = random.randint(2, n_c - 1)
            j = random.randint(0, i - 2)
            if j < len(adj_matrix) and i < len(adj_matrix[j]):
                conn_key = f"{j}->{i}"
                if conn_key in pheromones.skip_tau:
                    adj_matrix[j][i] = pheromones._sample_categorical(pheromones.skip_tau[conn_key])
                else:
                    adj_matrix[j][i] = 1 - adj_matrix[j][i]

    mutated = graph_to_genome(conv_layers, fc_layers, adj_matrix, min_conv, max_conv, min_full, max_full)
    x.n_conv = mutated.n_conv
    x.n_full = mutated.n_full
    x.first_level = mutated.first_level
    x.second_level = mutated.second_level
    return True, mut_choice
