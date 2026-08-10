# -*- coding: utf-8 -*-
""" Created on Sep 2024    @author: user
    Operadores Evolutivos Estructurados para DeepGA V7:
    - Cruce Topológico Emparejado (Coherent Graph-Based Crossover).
    - Sincronización estricta entre primer nivel (capas) y segundo nivel (conexiones residuales).
    - Preservación de bloques funcionales (Building Blocks) de las CNNs parentales.
"""

import random
import math
from copy import deepcopy
from EncodingClass import Encoding

'''Hyperparameters configuration'''
FSIZES = [3, 5, 7, 9]
NFILTERS = [8, 16, 32, 64, 128, 256]
PSIZES = [2, 3]
PTYPE = ['max', 'avg']
NEURONS = [16, 32, 64, 128, 256]


def genome_to_graph(genome):
    """
    Descompone un genoma de DeepGA en:
    - conv_layers: lista de diccionarios de capas convolucionales.
    - fc_layers: lista de diccionarios de capas densas (FC).
    - adj_matrix: matriz de adyacencia (n_conv x n_conv) para las conexiones residuales.
    """
    n_c = genome.n_conv
    n_f = genome.n_full
    conv_layers = [deepcopy(genome.first_level[i]) for i in range(min(n_c, len(genome.first_level)))]
    fc_layers = []
    for j in range(n_f):
        idx = n_c + j
        if idx < len(genome.first_level):
            fc_layers.append(deepcopy(genome.first_level[idx]))
        else:
            fc_layers.append({'type': 'fc', 'neurons': random.choice(NEURONS)})

    # Construir matriz de adyacencia para las conexiones residuales (skip-connections)
    adj_matrix = [[0 for _ in range(n_c)] for _ in range(n_c)]
    skips = getattr(genome, 'second_level', [])
    pos = 0
    for i in range(2, n_c):
        num_incoming = i - 1
        for j in range(num_incoming):
            if pos < len(skips):
                adj_matrix[j][i] = int(skips[pos])
                pos += 1

    return conv_layers, fc_layers, adj_matrix


def graph_to_genome(conv_layers, fc_layers, adj_matrix, min_conv=2, max_conv=5, min_full=1, max_full=4):
    """
    Reconstruye un genoma Encoding a partir de las capas y la matriz de adyacencia,
    asegurando que first_level y second_level estén perfectamente sincronizados y sin incompatibilidades.
    """
    n_c = max(min_conv, min(max_conv, len(conv_layers)))
    n_f = max(min_full, min(max_full, len(fc_layers)))

    conv_layers = conv_layers[:n_c]
    fc_layers = fc_layers[:n_f]

    new_encoding = Encoding(n_c, n_c, n_f, n_f)
    new_encoding.n_conv = n_c
    new_encoding.n_full = n_f
    new_encoding.first_level = conv_layers + fc_layers

    # Reconstruir second_level directamente desde la matriz de adyacencia
    second_level = []
    for i in range(2, n_c):
        for j in range(i - 1):
            if j < len(adj_matrix) and i < len(adj_matrix[j]):
                second_level.append(int(adj_matrix[j][i]))
            else:
                second_level.append(0)

    new_encoding.second_level = second_level
    return new_encoding


def crossover_v7(x, y, min_conv: int = 2, max_conv: int = 5, min_full: int = 1, max_full: int = 4):
    """
    Operador de Cruce Topológico Emparejado (DeepGA V7):
    1. Trata las redes como subgrafos dirigidos coherentes donde cada capa lleva asociadas sus conexiones.
    2. Realiza un corte jerárquico dividiendo en prefijo (extracción de bajo nivel) y sufijo (alto nivel).
    3. Preserva 100% de los subgrafos residuales internos de cada padre.
    4. Mapea proporcionalmente las conexiones residuales que atraviesan la frontera del corte.
    5. Garantiza dimensiones válidas en todo momento sin estructuras incompatibles.
    """
    conv1, fc1, adj1 = genome_to_graph(x)
    conv2, fc2, adj2 = genome_to_graph(y)

    n1 = len(conv1)
    n2 = len(conv2)

    # 1. Puntos de corte en la parte convolucional (preservando jerarquía de capas)
    if n1 > 2 and n2 > 2:
        k1 = random.randint(1, n1 - 1)
        k2 = random.randint(1, n2 - 1)
    else:
        k1 = 1
        k2 = 1

    # Ajuste de tamaño para asegurar que ambos hijos estén en [min_conv, max_conv]
    len_c1 = k1 + (n2 - k2)
    len_c2 = k2 + (n1 - k1)

    if len_c1 < min_conv or len_c1 > max_conv or len_c2 < min_conv or len_c2 > max_conv:
        split_ratio = random.uniform(0.4, 0.6)
        k1 = max(1, min(n1 - 1, int(round(n1 * split_ratio))))
        k2 = max(1, min(n2 - 1, int(round(n2 * split_ratio))))

    # Construcción de capas convolucionales
    # Hijo 1: Prefijo P1 [0..k1-1] + Sufijo P2 [k2..n2-1]
    child1_conv = deepcopy(conv1[:k1]) + deepcopy(conv2[k2:])
    # Hijo 2: Prefijo P2 [0..k2-1] + Sufijo P1 [k1..n1-1]
    child2_conv = deepcopy(conv2[:k2]) + deepcopy(conv1[k1:])

    # Ajustar a rango permitido [min_conv, max_conv]
    if len(child1_conv) > max_conv:
        child1_conv = child1_conv[:max_conv]
    elif len(child1_conv) < min_conv:
        while len(child1_conv) < min_conv:
            child1_conv.append(deepcopy(conv1[-1]))

    if len(child2_conv) > max_conv:
        child2_conv = child2_conv[:max_conv]
    elif len(child2_conv) < min_conv:
        while len(child2_conv) < min_conv:
            child2_conv.append(deepcopy(conv2[-1]))

    nc1 = len(child1_conv)
    nc2 = len(child2_conv)

    # 2. Mapeo Topológico Coherente de Conexiones Residuales (Segundo Nivel)
    # Hijo 1
    adj_c1 = [[0 for _ in range(nc1)] for _ in range(nc1)]
    for i in range(2, nc1):
        for j in range(i - 1):
            if i < k1:
                # Región 1: Conexión interna del prefijo -> Preservada de P1
                if j < len(adj1) and i < len(adj1[j]):
                    adj_c1[j][i] = adj1[j][i]
            elif j >= k1:
                # Región 2: Conexión interna del sufijo -> Preservada de P2
                j_p2 = k2 + (j - k1)
                i_p2 = k2 + (i - k1)
                if j_p2 < len(adj2) and i_p2 < len(adj2[j_p2]):
                    adj_c1[j][i] = adj2[j_p2][i_p2]
            else:
                # Región 3: Conexión inter-módulo (del prefijo al sufijo)
                j_mapped = int(round(j * (float(k2) / float(k1)))) if k1 > 0 else 0
                i_p2 = k2 + (i - k1)
                if j_mapped < len(adj2) and i_p2 < len(adj2[j_mapped]):
                    adj_c1[j][i] = adj2[j_mapped][i_p2]

    # Hijo 2
    adj_c2 = [[0 for _ in range(nc2)] for _ in range(nc2)]
    for i in range(2, nc2):
        for j in range(i - 1):
            if i < k2:
                # Región 1: Conexión interna del prefijo -> Preservada de P2
                if j < len(adj2) and i < len(adj2[j]):
                    adj_c2[j][i] = adj2[j][i]
            elif j >= k2:
                # Región 2: Conexión interna del sufijo -> Preservada de P1
                j_p1 = k1 + (j - k2)
                i_p1 = k1 + (i - k2)
                if j_p1 < len(adj1) and i_p1 < len(adj1[j_p1]):
                    adj_c2[j][i] = adj1[j_p1][i_p1]
            else:
                # Región 3: Conexión inter-módulo (del prefijo al sufijo)
                j_mapped = int(round(j * (float(k1) / float(k2)))) if k2 > 0 else 0
                i_p1 = k1 + (i - k2)
                if j_mapped < len(adj1) and i_p1 < len(adj1[j_mapped]):
                    adj_c2[j][i] = adj1[j_mapped][i_p1]

    # 3. Recombinación de capas densas (Fully Connected)
    m1 = len(fc1)
    m2 = len(fc2)
    f1_cut = max(1, min(m1, random.randint(1, m1)))
    f2_cut = max(1, min(m2, random.randint(1, m2)))

    child1_fc = deepcopy(fc1[:f1_cut]) + deepcopy(fc2[f2_cut:])
    child2_fc = deepcopy(fc2[:f2_cut]) + deepcopy(fc1[f1_cut:])

    if len(child1_fc) > max_full:
        child1_fc = child1_fc[:max_full]
    elif len(child1_fc) < min_full:
        while len(child1_fc) < min_full:
            child1_fc.append(deepcopy(fc1[-1]))

    if len(child2_fc) > max_full:
        child2_fc = child2_fc[:max_full]
    elif len(child2_fc) < min_full:
        while len(child2_fc) < min_full:
            child2_fc.append(deepcopy(fc2[-1]))

    # 4. Reconstruir los genomas descendientes con sincronización estricta
    child1 = graph_to_genome(child1_conv, child1_fc, adj_c1, min_conv, max_conv, min_full, max_full)
    child2 = graph_to_genome(child2_conv, child2_fc, adj_c2, min_conv, max_conv, min_full, max_full)

    return child1, child2


def mutation_v7(x, min_conv: int = 2, max_conv: int = 5, min_full: int = 1, max_full: int = 4):
    """
    Operador de mutación consistente con la estructura topológica de V7.
    """
    conv_layers, fc_layers, adj_matrix = genome_to_graph(x)

    mutation_type = random.choice(['modify_layer', 'add_remove_layer', 'flip_skip'])

    if mutation_type == 'modify_layer':
        # Modificar hiperparámetros de una capa existente
        if random.uniform(0, 1) < 0.7 and len(conv_layers) > 0:
            idx = random.randint(0, len(conv_layers) - 1)
            conv_layers[idx]['nfilters'] = random.choice(NFILTERS)
            conv_layers[idx]['fsize'] = random.choice(FSIZES)
            conv_layers[idx]['pool'] = random.choice(['max', 'avg', 'off'])
            conv_layers[idx]['psize'] = random.choice(PSIZES)
        elif len(fc_layers) > 0:
            idx = random.randint(0, len(fc_layers) - 1)
            fc_layers[idx]['neurons'] = random.choice(NEURONS)

    elif mutation_type == 'add_remove_layer':
        # Agregar o eliminar una capa respetando los límites
        if random.uniform(0, 1) < 0.5:
            # Operar en capas convolucionales
            if len(conv_layers) < max_conv and random.uniform(0, 1) < 0.5:
                # Agregar capa conv
                new_layer = {
                    'type': 'conv',
                    'nfilters': random.choice(NFILTERS),
                    'fsize': random.choice(FSIZES),
                    'pool': random.choice(['max', 'avg', 'off']),
                    'psize': random.choice(PSIZES)
                }
                insert_idx = random.randint(0, len(conv_layers))
                conv_layers.insert(insert_idx, new_layer)
                # Expandir matriz de adyacencia
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
                # Eliminar capa conv
                del_idx = random.randint(0, len(conv_layers) - 1)
                conv_layers.pop(del_idx)
                # Reducir matriz de adyacencia
                adj_matrix = [row[:del_idx] + row[del_idx+1:] for idx, row in enumerate(adj_matrix) if idx != del_idx]
        else:
            # Operar en capas densas
            if len(fc_layers) < max_full and random.uniform(0, 1) < 0.5:
                fc_layers.append({'type': 'fc', 'neurons': random.choice(NEURONS)})
            elif len(fc_layers) > min_full:
                fc_layers.pop()

    else:
        # Mutación en conexiones residuales (segundo nivel)
        n_c = len(conv_layers)
        if n_c > 2:
            i = random.randint(2, n_c - 1)
            j = random.randint(0, i - 2)
            if j < len(adj_matrix) and i < len(adj_matrix[j]):
                adj_matrix[j][i] = 1 - adj_matrix[j][i]  # Invertir bit

    mutated = graph_to_genome(conv_layers, fc_layers, adj_matrix, min_conv, max_conv, min_full, max_full)
    x.n_conv = mutated.n_conv
    x.n_full = mutated.n_full
    x.first_level = mutated.first_level
    x.second_level = mutated.second_level
    return x
