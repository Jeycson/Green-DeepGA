# -*- coding: utf-8 -*-
""" Created on Sep 1 2024    @author: user
    Variante con Podado por Familiaridad y Entrenamiento Secuencial (DeepGA_V4).
"""

from deepga.evolution.operators import *
from deepga.core.encoding import Encoding
from deepga.core.decoding import *
from deepga.training.engine import *
from torch.utils.data import DataLoader
import timeit
import torch
import pickle
import torch.nn as nn
import os
from pathlib import Path
import pandas as pd
import copy
import random


def calculate_genetic_distance(e1, e2) -> float:
    """
    Calcula una distancia estructural normalizada entre dos genomas de DeepGA [0.0 - 1.0]:
    - Diferencia en número de capas convolucionales y densas.
    - Diferencia en filtros, tamaños de kernel y pooling capa por capa.
    - Distancia Hamming en conexiones residuales (skip-connections).
    """
    # 1. Distancia en cantidad de capas
    conv_diff = abs(e1.n_conv - e2.n_conv) / max(1, max(e1.n_conv, e2.n_conv))
    full_diff = abs(e1.n_full - e2.n_full) / max(1, max(e1.n_full, e2.n_full))

    # 2. Distancia en configuración de capas convolucionales
    min_c = min(e1.n_conv, e2.n_conv)
    layer_diff = 0.0
    for i in range(min_c):
        l1 = e1.first_level[i]
        l2 = e2.first_level[i]
        f_diff = abs(l1.get('nfilters', 16) - l2.get('nfilters', 16)) / 256.0
        k_diff = abs(l1.get('fsize', 3) - l2.get('fsize', 3)) / 9.0
        p_diff = 0.0 if l1.get('pool') == l2.get('pool') else 0.5
        layer_diff += (f_diff + k_diff + p_diff) / 3.0
    layer_diff = layer_diff / max(1, min_c)

    # 3. Distancia en conexiones residuales (Hamming en second_level)
    s1 = getattr(e1, 'second_level', [])
    s2 = getattr(e2, 'second_level', [])
    min_s = min(len(s1), len(s2))
    hamming = sum(1 for a, b in zip(s1[:min_s], s2[:min_s]) if a != b)
    hamming += abs(len(s1) - len(s2))
    max_s = max(1, max(len(s1), len(s2)))
    skip_diff = hamming / max_s

    # Ponderación total [0, 1]
    total_distance = 0.25 * conv_diff + 0.25 * full_diff + 0.30 * layer_diff + 0.20 * skip_diff
    return float(total_distance)


def check_kinship_pruning(candidate_e, parent1_fit: float, parent2_fit: float,
                          evaluated_history: list, prune_threshold: float,
                          k_neighbors: int = 3, similarity_radius: float = 0.30):
    """
    Evalúa si un candidato debe ser podado antes de entrenar en GPU:
    1. Linaje Parental: Si ambos padres fueron deficientes (ambos con fitness < prune_threshold).
    2. Vecindario Genético (k-NN sobre el historial en cacheM): Si sus parientes más cercanos
       tienen un fitness promedio bajo.
    """
    # 1. Nivel 1: Filtro Parental Directo
    if parent1_fit is not None and parent2_fit is not None:
        if parent1_fit < prune_threshold and parent2_fit < prune_threshold:
            return True, f"Linaje parental deficiente (P1: {parent1_fit:.4f}, P2: {parent2_fit:.4f} < {prune_threshold:.4f})"

    # 2. Nivel 2: Vecinos estructurales más cercanos en el historial
    if len(evaluated_history) >= k_neighbors:
        distances = []
        for hist_e, fit_val, acc_val in evaluated_history:
            dist = calculate_genetic_distance(candidate_e, hist_e)
            distances.append((dist, fit_val, acc_val))

        distances.sort(key=lambda x: x[0])
        nearest = distances[:k_neighbors]

        close_relatives = [x for x in nearest if x[0] <= similarity_radius]
        if len(close_relatives) >= 2:
            avg_rel_fit = sum(x[1] for x in close_relatives) / len(close_relatives)
            if avg_rel_fit < prune_threshold:
                return True, f"Parientes genéticos deficientes (Fit prom: {avg_rel_fit:.4f} < {prune_threshold:.4f}, dist={close_relatives[0][0]:.2f})"

    return False, "Aprobado para entrenamiento"


def _evaluate_individual(e, n_channels: int, out_size: int, n_classes: int, device: torch.device,
                         num_epochs: int, loss_func, train_dl: DataLoader, val_dl: DataLoader,
                         lr: float, w: float, max_params: int):
    """Decodifica y entrena secuencialmente una única CNN."""
    network = decoding(e, n_channels, out_size, n_classes)
    cnn = CNN(e, network[0], network[1], network[2])
    acc_list = []
    fit, acc, pars, _ = training('1', device, cnn, num_epochs, loss_func,
                                 train_dl, val_dl, lr, w, max_params, acc_list)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fit, acc, pars


def green_DeepGA_v4(execution: int, memoryC: bool, train_epochs: int, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                    min_conv: int, max_conv: int, min_full: int, max_full: int, max_params: int, cr: float, mr: float,  
                    N: int, T: int, t_size: int, w: float, device: torch.device, chck_dir: str,
                    n_channels: int = 3, n_classes: int = 10, out_size: int = 32, loss_func=None,
                    k_neighbors: int = 3, prune_quantile: float = 0.25,
                    max_prune_retries: int = 3, **kwargs):
    """
    Algoritmo DeepGA V4: Podado por Familiaridad y Linaje con Entrenamiento Secuencial (no paralelo).
    
    Args:
        k_neighbors: Número de parientes arquitecturales más cercanos a inspeccionar en historial.
        prune_quantile: Cuantil de corte inferior de la población considerado 'deficiente' (ej. 0.25 = peor 25%).
        max_prune_retries: Intentos máximos de mutación exploratoria antes de forzar entrenamiento.
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):  
        os.makedirs(chck_dir)      

    '''Initialize population'''
    chkpoint_obj = Path(chck_dir + str(execution) + "_checkpoint.pkl")
    if chkpoint_obj.exists():
        print("Re-Initialize population (DeepGA V4)")
        with open(chck_dir + str(execution) + "_checkpoint.pkl", "rb") as p:
            values = pickle.load(p)
        start = timeit.default_timer() - values['time']
        pop = values['pop']
        bestAcc = values['bestAcc']
        bestF = values['bestF']
        bestParams = values['bestParams']
        t = values['t']
        if t == T:
            print('The maximum number of generations has been reached. Please run a new execution.')
            leader = max(pop, key = lambda x: x[1])
        evals = values['evals']
        cacheM = values['cacheM']
        meanfitpop = values['meanfitpop']
        meanAccpop = values['meanAccpop']
        meanParpop = values['meanParpop']
        evaluated_history = values.get('evaluated_history', [])
        pruned_count = values.get('pruned_count', 0)
    else:
        print('Initialize population (DeepGA V4 - Kinship Pruning + Sequential Training)')
        start = timeit.default_timer()
        pop = []
        bestAcc = []
        bestF = []
        bestParams = []
        t = 0
        evals = 0
        pruned_count = 0
        cacheM = {}
        evaluated_history = []  # Lista de tuplas: (genome_obj, fitness, accuracy)
        meanfitpop = []
        meanAccpop = []
        meanParpop = []

        while len(pop) < N:
            e1 = Encoding(min_conv, max_conv, min_full, max_full)
            if memoryC:
                strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
                if strIDe1 in cacheM:
                    fit1, acc1, pars1 = cacheM[strIDe1]
                    print(f"[Cache Init] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                    pop.append([e1, fit1, acc1, pars1])
                    evaluated_history.append((e1, fit1, acc1))
                else:
                    fit1, acc1, pars1 = _evaluate_individual(
                        e1, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    cacheM[strIDe1] = [fit1, acc1, pars1]
                    evals += 1
                    evaluated_history.append((e1, fit1, acc1))
                    print(f"[Sequential Init V4] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                    pop.append([e1, fit1, acc1, pars1])
            else:
                fit1, acc1, pars1 = _evaluate_individual(
                    e1, n_channels, out_size, n_classes, device1,
                    num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                )
                evals += 1
                evaluated_history.append((e1, fit1, acc1))
                print(f"[Sequential Init V4] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                pop.append([e1, fit1, acc1, pars1])

    '''Genetic Algorithm'''
    print('--------------------------------------------')
    while t < T:
        print(f'Generation: {t} (DeepGA V4)')

        # Calcular umbral dinámico de poda basado en el cuantil inferior de la población actual
        all_fitnesses = sorted([ind[1] for ind in pop])
        cutoff_idx = max(0, int(len(all_fitnesses) * prune_quantile))
        prune_threshold = all_fitnesses[cutoff_idx]
        print(f" [V4 Pruning Threshold (peor {int(prune_quantile*100)}%)]: {prune_threshold:.4f}")

        # Parents Selection
        parents = []
        while len(parents) < int(N/2):
            tournament = random.sample(pop, t_size)
            p1 = selection(tournament, 'max')
            tournament = random.sample(pop, t_size)
            p2 = selection(tournament, 'max')
            while p1 == p2:
                tournament = random.sample(pop, t_size)
                p2 = selection(tournament, 'max')

            parents.append(p1)
            parents.append(p2)

        # Reproduction con Podado por Familiaridad
        offspring = []
        iter_parents = 0
        while len(offspring) < int(N/2):
            parent_1_obj = parents[iter_parents]
            parent_2_obj = parents[iter_parents + 1]
            p1 = parent_1_obj[0]
            p2 = parent_2_obj[0]
            p1_fit = parent_1_obj[1]
            p2_fit = parent_2_obj[1]

            if cr >= random.uniform(0, 1):
                c1, c2 = crossover(p1, p2)
            else:
                c1 = deepcopy(p1)
                c2 = deepcopy(p2)

            if mr >= random.uniform(0, 1):
                mutation(c1)

            if mr >= random.uniform(0, 1):
                mutation(c2)

            # --- FILTRO DE PODA POR FAMILIARIDAD (KINSHIP PRUNING) ---
            candidate_list = [c1, c2]
            processed_candidates = []

            for cand_idx, candidate in enumerate(candidate_list):
                curr_c = candidate
                retries = 0
                while retries < max_prune_retries:
                    is_pruned, reason = check_kinship_pruning(
                        candidate_e=curr_c,
                        parent1_fit=p1_fit,
                        parent2_fit=p2_fit,
                        evaluated_history=evaluated_history,
                        prune_threshold=prune_threshold,
                        k_neighbors=k_neighbors
                    )
                    if is_pruned:
                        pruned_count += 1
                        print(f" [PODA V4] Hijo {cand_idx+1} descartado antes de GPU: {reason}. Re-mutando intento {retries+1}/{max_prune_retries}...")
                        curr_c = deepcopy(curr_c)
                        mutation(curr_c)
                        retries += 1
                    else:
                        break
                processed_candidates.append(curr_c)

            c1, c2 = processed_candidates[0], processed_candidates[1]

            # Evaluar de forma secuencial las CNNs supervivientes
            candidates = [c1, c2]
            eval_results = []

            for cand_idx, c in enumerate(candidates):
                if memoryC:
                    strID = str([c.n_conv, c.n_full, c.first_level, c.second_level])
                    if strID in cacheM:
                        fit, acc, pars = cacheM[strID]
                        print(f"[Cache Offspring] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")
                    else:
                        fit, acc, pars = _evaluate_individual(
                            c, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                        )
                        cacheM[strID] = [fit, acc, pars]
                        evaluated_history.append((c, fit, acc))
                        evals += 1
                        print(f"[Sequential Offspring V4] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")
                else:
                    fit, acc, pars = _evaluate_individual(
                        c, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    evaluated_history.append((c, fit, acc))
                    evals += 1
                    print(f"[Sequential Offspring V4] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")

                eval_results.append([fit, acc, pars])

            fit1, acc1, pars1 = eval_results[0]
            fit2, acc2, pars2 = eval_results[1]

            offspring.append([c1, fit1, acc1, pars1])
            offspring.append([c2, fit2, acc2, pars2])

            iter_parents += 2

        # Replacement with elitism
        pop = pop + offspring
        pop.sort(reverse = True, key = lambda x: x[1])
        pop = pop[:N]

        leader = max(pop, key = lambda x: x[1])
        bestAcc.append(leader[2])
        bestF.append(leader[1])
        bestParams.append(leader[3])
        meanfitpop.append(sum([q[1] for q in pop])/N)
        meanAccpop.append(sum([q[2] for q in pop])/N)
        meanParpop.append(sum([q[3] for q in pop])/N)

        t += 1

        # Checkpoint con estado de V4
        time = timeit.default_timer() - start
        current_state: dict = dict(pop=pop, bestAcc=bestAcc, bestF=bestF,
                                   bestParams=bestParams, t=t, evals=evals,
                                   pruned_count=pruned_count,
                                   evaluated_history=evaluated_history,
                                   time=time, cacheM=cacheM, meanfitpop=meanfitpop,
                                   meanAccpop=meanAccpop, meanParpop=meanParpop)
        with open(chck_dir + str(execution) + "_checkpoint.pkl", "wb") as p:
            pickle.dump(current_state, p)

        print(f"--- Fin Gen {t-1} | Podas acumuladas: {pruned_count} | Evaluaciones GPU: {evals} ---")
        print('Best fitness: ', leader[1])
        print('Best accuracy: ', leader[2])
        print('Best No. of Params: ', leader[3])
        print('No. of Conv. Layers: ', leader[0].n_conv)
        print('No. of FC Layers: ', leader[0].n_full)
        print('--------------------------------------------')

    bestind = copy.deepcopy(leader)
    results = pd.DataFrame(list(zip(bestAcc, bestF, bestParams, meanfitpop, meanAccpop, meanParpop)),
                           columns = ['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar'])
    print(results)

    # Guardar automáticamente la arquitectura del mejor modelo de esta variante (V4)
    try:
        from deepga.utils.model_utils import save_best_model
        save_best_model(
            variant="v4",
            execution=execution,
            bestind=bestind,
            in_channels=n_channels,
            out_size=out_size,
            n_classes=n_classes,
            chck_dir=chck_dir
        )
    except Exception as e:
        print(f"Nota al guardar mejor modelo V4: {e}")

    return results, pop, bestind


def final_evaluation(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     max_params: int, w: float, device: torch.device, train_epochs: int, loss_func, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, variant: str = "v4", auto_download: bool = False):
    from variants.DeepGA import final_evaluation as fe
    return fe(execution, bestind, train_dl, val_dl, lr, max_params, w, device, train_epochs, loss_func, chck_dir, n_channels, n_classes, out_size, variant=variant, auto_download=auto_download)
