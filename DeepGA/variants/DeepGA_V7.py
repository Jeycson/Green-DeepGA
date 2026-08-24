# -*- coding: utf-8 -*-
""" Created on Sep 2024    @author: user
    Variante con Operadores Evolutivos Topológicamente Estructurados (DeepGA_V7).
    Implementa Cruce y Mutación Emparejados donde el primer y segundo nivel
    se recombinan como subgrafos coherentes, evitando incompatibilidades estructurales.
"""

from deepga.evolution.operators import selection
from deepga.evolution.operators_v7 import crossover_v7, mutation_v7
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
from copy import deepcopy
import random


def _evaluate_individual(e, n_channels: int, out_size: int, n_classes: int, device: torch.device,
                         num_epochs: int, loss_func, train_dl: DataLoader, val_dl: DataLoader,
                         lr: float, w: float, max_params: int):
    """Decodifica y entrena secuencialmente una única CNN en GPU."""
    network = decoding(e, n_channels, out_size, n_classes)
    cnn = CNN(e, network[0], network[1], network[2])
    acc_list = []
    fit, acc, pars, _ = training('1', device, cnn, num_epochs, loss_func,
                                 train_dl, val_dl, lr, w, max_params, acc_list)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fit, acc, pars


def green_DeepGA_v7(execution: int, memoryC: bool, train_epochs: int, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                    min_conv: int, max_conv: int, min_full: int, max_full: int, max_params: int, cr: float, mr: float,  
                    N: int, T: int, t_size: int, w: float, device: torch.device, chck_dir: str,
                    n_channels: int = 3, n_classes: int = 10, out_size: int = 32, loss_func=None, **kwargs):
    """
    Algoritmo DeepGA V7:
    - Operadores Genéticos Estructurados (Cruce y Mutación Topológica V7).
    - Primer nivel (capas) y segundo nivel (conexiones residuales) acoplados.
    - Preservación de subgrafos funcionales y recombinación modular.
    - Entrenamiento secuencial optimizado en GPU / VRAM.
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):  
        os.makedirs(chck_dir)      

    '''Initialize population'''
    chkpoint_obj = Path(chck_dir + str(execution) + "_checkpoint.pkl")
    if chkpoint_obj.exists():
        print("Re-Initialize population (DeepGA V7 - Structured Operators)")
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
    else:
        print('Initialize population (DeepGA V7 - Structured Topological Operators)')
        start = timeit.default_timer()
        pop = []
        bestAcc = []
        bestF = []
        bestParams = []
        t = 0
        evals = 0
        cacheM = {}
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
                else:
                    fit1, acc1, pars1 = _evaluate_individual(
                        e1, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    cacheM[strIDe1] = [fit1, acc1, pars1]
                    evals += 1
                    print(f"[Sequential Init V7] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                    pop.append([e1, fit1, acc1, pars1])
            else:
                fit1, acc1, pars1 = _evaluate_individual(
                    e1, n_channels, out_size, n_classes, device1,
                    num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                )
                evals += 1
                print(f"[Sequential Init V7] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                pop.append([e1, fit1, acc1, pars1])

    '''Genetic Algorithm con Operadores V7'''
    print('--------------------------------------------')
    while t < T:
        print(f'Generation: {t} (DeepGA V7 - Structured Crossover & Mutation)')

        # Parents Selection (Torneo)
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

        # Reproducción con Cruce Topológico V7
        offspring = []
        iter_parents = 0
        while len(offspring) < int(N/2):
            p1 = parents[iter_parents][0]
            p2 = parents[iter_parents + 1][0]

            if cr >= random.uniform(0, 1):
                # Cruce Topológicamente Estructurado (primer y segundo nivel emparejados)
                c1, c2 = crossover_v7(p1, p2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
            else:
                c1 = copy.deepcopy(p1)
                c2 = copy.deepcopy(p2)

            if mr >= random.uniform(0, 1):
                mutation_v7(c1, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)

            if mr >= random.uniform(0, 1):
                mutation_v7(c2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)

            # Evaluación secuencial de los descendientes
            candidates = [c1, c2]
            eval_results = []

            for c in candidates:
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
                        evals += 1
                        print(f"[GPU Offspring V7] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")
                else:
                    fit, acc, pars = _evaluate_individual(
                        c, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    evals += 1
                    print(f"[GPU Offspring V7] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")

                eval_results.append([fit, acc, pars])

            fit1, acc1, pars1 = eval_results[0]
            fit2, acc2, pars2 = eval_results[1]

            offspring.append([c1, fit1, acc1, pars1])
            offspring.append([c2, fit2, acc2, pars2])

            iter_parents += 2

        # Reemplazo con Elitismo
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

        # Checkpoint de V7
        time = timeit.default_timer() - start
        current_state: dict = dict(pop=pop, bestAcc=bestAcc, bestF=bestF,
                                   bestParams=bestParams, t=t, evals=evals,
                                   time=time, cacheM=cacheM, meanfitpop=meanfitpop,
                                   meanAccpop=meanAccpop, meanParpop=meanParpop)
        with open(chck_dir + str(execution) + "_checkpoint.pkl", "wb") as p:
            pickle.dump(current_state, p)

        print(f"--- Fin Gen {t-1} | Evaluaciones GPU: {evals} ---")
        print('Best fitness: ', leader[1])
        print('Best accuracy: ', leader[2])
        print('Best No. of Params: ', leader[3])
        print('No. of Conv. Layers: ', leader[0].n_conv)
        print('No. of FC Layers: ', leader[0].n_full)
        print('--------------------------------------------')

    bestind = copy.deepcopy(leader)
    results = pd.DataFrame(list(zip(bestAcc, bestF, bestParams, meanfitpop, meanAccpop, meanParpop)),
                           columns = ['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar'])

    # Guardar automáticamente la arquitectura del mejor modelo de esta variante (V7)
    try:
        from deepga.utils.model_utils import save_best_model
        save_best_model(
            variant="v7",
            execution=execution,
            bestind=bestind,
            in_channels=n_channels,
            out_size=out_size,
            n_classes=n_classes,
            chck_dir=chck_dir
        )
    except Exception as e:
        print(f"Nota al guardar mejor modelo V7: {e}")

    return results, pop, bestind


def final_evaluation(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     max_params: int, w: float, device: torch.device, train_epochs: int, loss_func, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, variant: str = "v7", auto_download: bool = False):
    from variants.DeepGA import final_evaluation as fe
    return fe(execution, bestind, train_dl, val_dl, lr, max_params, w, device, train_epochs, loss_func, chck_dir, n_channels, n_classes, out_size, variant=variant, auto_download=auto_download)
