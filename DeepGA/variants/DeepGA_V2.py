# -*- coding: utf-8 -*-
""" Created on Sep 1 2024    @author: user
    Variante con paralelización del entrenamiento de CNNs generadas (DeepGA_V2).
"""

from Operators import *
from EncodingClass import Encoding
from Decoding import *
from DistributedTraining import *
from torch.utils.data import DataLoader
import timeit
import torch
import pickle
import torch.nn as nn
import os
from pathlib import Path
import pandas as pd
import copy
from concurrent.futures import ThreadPoolExecutor


def _evaluate_individual(e, n_channels: int, out_size: int, n_classes: int, device: torch.device,
                         num_epochs: int, loss_func, train_dl: DataLoader, val_dl: DataLoader,
                         lr: float, w: float, max_params: int):
    """
    Decodifica y entrena concurrentemente una única CNN generada por DeepGA.
    """
    network = decoding(e, n_channels, out_size, n_classes)
    cnn = CNN(e, network[0], network[1], network[2])
    acc_list = []
    fit, acc, pars, _ = training('1', device, cnn, num_epochs, loss_func,
                                 train_dl, val_dl, lr, w, max_params, acc_list)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fit, acc, pars


def green_DeepGA_v2(execution: int, memoryC: bool, train_epochs: int, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                    min_conv: int, max_conv: int, min_full: int, max_full: int, max_params: int, cr: float, mr: float,  
                    N: int, T: int, t_size: int, w: float, device: torch.device, chck_dir: str,
                    n_channels: int = 3, n_classes: int = 10, out_size: int = 32, loss_func=None,
                    num_workers: int = 2):
    """
    Algoritmo DeepGA con entrenamiento paralelo de las CNNs generadas.
    
    Args:
        num_workers: Número de CNNs a entrenar concurrentemente (por defecto 2).
                     En Google Colab (GPU T4) permite entrenar los pares de hijos en paralelo.
    """
    num_epochs = train_epochs # Epochs to train each individual during the GA
    device1 = device

    # Configuración de dispositivos según disponibilidad (multi-GPU o single-GPU compartida)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        worker_devices = [torch.device(f"cuda:{i % gpu_count}") for i in range(num_workers)]
    else:
        worker_devices = [device1] * num_workers

    # Indicate path for checkpoint
    if not os.path.exists(chck_dir):  
        os.makedirs(chck_dir)      

    '''Initialize population'''
    # Check if checkpoint is available
    chkpoint_obj = Path(chck_dir + str(execution) + "_checkpoint.pkl")
    if chkpoint_obj.exists():
        print("Re-Initialize population")
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
        print('Initialize population (Parallel Training)')
        start = timeit.default_timer()
        pop = []
        bestAcc = []
        bestF = []
        bestParams = []
        t = 0  # Generaciones
        evals = 0 # Evaluaciones
        cacheM = {}
        meanfitpop = []
        meanAccpop = []
        meanParpop = []

        while len(pop) < N:
            batch_needed = N - len(pop)
            batch_genomes = []

            # Generar lote de genomas candidatos
            while len(batch_genomes) < min(batch_needed, num_workers) and (len(pop) + len(batch_genomes)) < N:
                e1 = Encoding(min_conv, max_conv, min_full, max_full)
                if memoryC:
                    strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
                    if strIDe1 in cacheM:
                        fit1, acc1, pars1 = cacheM[strIDe1]
                        print(f"[Cache Init] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                        pop.append([e1, fit1, acc1, pars1])
                    else:
                        batch_genomes.append((e1, strIDe1))
                else:
                    batch_genomes.append((e1, None))

            if not batch_genomes:
                continue

            # Entrenar en paralelo los candidatos que no estaban en caché
            actual_workers = len(batch_genomes)
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = [
                    executor.submit(
                        _evaluate_individual,
                        e, n_channels, out_size, n_classes,
                        worker_devices[i % len(worker_devices)],
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    for i, (e, _) in enumerate(batch_genomes)
                ]
                results = [f.result() for f in futures]

            for (e, strID), (fit, acc, pars) in zip(batch_genomes, results):
                if memoryC and strID is not None:
                    cacheM[strID] = [fit, acc, pars]
                evals += 1
                print(f"[Parallel Init] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")
                pop.append([e, fit, acc, pars])

    '''Genetic Algorithm'''
    print('--------------------------------------------')
    while t < T:
        print('Generation: ', t)

        #Parents Selection
        parents = []
        while len(parents) < int(N/2):
            #Tournament Selection
            tournament = random.sample(pop, t_size)
            p1 = selection(tournament, 'max')
            tournament = random.sample(pop, t_size)
            p2 = selection(tournament, 'max')
            while p1 == p2:
                tournament = random.sample(pop, t_size)
                p2 = selection(tournament, 'max')

            parents.append(p1)
            parents.append(p2)

        #Reproduction
        offspring = []
        iter_parents = 0
        while len(offspring) < int(N/2):
            #Crossover + Mutation
            p1 = parents[iter_parents][0]
            p2 = parents[iter_parents + 1][0]
            if cr >= random.uniform(0,1): #Crossover
                c1, c2 = crossover(p1, p2)
            else:
                c1 = deepcopy(p1)
                c2 = deepcopy(p2)

            #Mutation
            if mr >= random.uniform(0,1):
                mutation(c1)

            if mr >= random.uniform(0,1):
                mutation(c2)

            #Evaluate offspring en paralelo
            candidates = [c1, c2]
            eval_results = [None, None]
            to_train = []

            for idx, c in enumerate(candidates):
                if memoryC:
                    strID = str([c.n_conv, c.n_full, c.first_level, c.second_level])
                    if strID in cacheM:
                        eval_results[idx] = cacheM[strID]
                        print(f"[Cache Offspring] Fit: {eval_results[idx][0]:.4f}, Acc: {eval_results[idx][1]:.2f}%, Params: {eval_results[idx][2]}")
                    else:
                        to_train.append((idx, c, strID))
                else:
                    to_train.append((idx, c, None))

            # Si alguna o ambas CNNs requieren entrenamiento, ejecutarlas en paralelo
            if to_train:
                with ThreadPoolExecutor(max_workers=len(to_train)) as executor:
                    futures = [
                        (idx, strID, executor.submit(
                            _evaluate_individual,
                            c, n_channels, out_size, n_classes,
                            worker_devices[i % len(worker_devices)],
                            num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                        ))
                        for i, (idx, c, strID) in enumerate(to_train)
                    ]
                    for idx, strID, fut in futures:
                        fit, acc, pars = fut.result()
                        eval_results[idx] = [fit, acc, pars]
                        if memoryC and strID is not None:
                            cacheM[strID] = [fit, acc, pars]
                        evals += 1
                        print(f"[Parallel Offspring] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")

            fit1, acc1, pars1 = eval_results[0]
            fit2, acc2, pars2 = eval_results[1]

            offspring.append([c1, fit1, acc1, pars1])
            offspring.append([c2, fit2, acc2, pars2])

            iter_parents += 2

        #Replacement with elitism
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

        # Making checkpoint
        time = timeit.default_timer() - start
        current_state: dict = dict(pop=pop, bestAcc=bestAcc,bestF=bestF,
                                   bestParams=bestParams, t=t, evals=evals,
                                   time=time, cacheM=cacheM, meanfitpop=meanfitpop,
                                   meanAccpop=meanAccpop, meanParpop=meanParpop)
        with open(chck_dir + str(execution) + "_checkpoint.pkl", "wb") as p:
            pickle.dump(current_state, p)

        print('Best fitness: ', leader[1])
        print('Best accuracy: ', leader[2])
        print('Best No. of Params: ', leader[3])
        print('No. of Conv. Layers: ', leader[0].n_conv)
        print('No. of FC Layers: ', leader[0].n_full)
        print('--------------------------------------------')

    bestind = copy.deepcopy(leader)
    results = pd.DataFrame(list(zip(bestAcc, bestF, bestParams, meanfitpop,  meanAccpop, meanParpop)),
                           columns = ['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar'])
    print(results)

    # Guardar automáticamente la arquitectura del mejor modelo de esta variante (V2)
    try:
        from model_utils import save_best_model
        save_best_model(
            variant="v2",
            execution=execution,
            bestind=bestind,
            in_channels=n_channels,
            out_size=out_size,
            n_classes=n_classes,
            chck_dir=chck_dir
        )
    except Exception as e:
        print(f"Nota al guardar mejor modelo V2: {e}")

    return results, pop, bestind  


def final_evaluation(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     max_params: int, w: float, device: torch.device, train_epochs: int, loss_func, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, variant: str = "v2", auto_download: bool = False):
  
    chkpoint_obj = Path(os.path.join(chck_dir, f"Model_Exec_{execution}_Epoch_{train_epochs}_point.pkl"))
    if not chkpoint_obj.exists():
        print("Training final model from best individual") 
        start = timeit.default_timer()
        #Decoding the networks of the best individual
        network1 = decoding(bestind[0], n_channels, out_size, n_classes)
        acc_list = []
        #Creating the CNNs
        cnnfin = CNN(bestind[0], network1[0], network1[1], network1[2])
        #Evaluate individuals
        fitfin, accfin, parsfin, CNNModel = training('1', device, cnnfin, train_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
        print(fitfin, accfin, parsfin)
        stop = timeit.default_timer()
        execution_timeS = (stop-start)
        execution_timeH = execution_timeS/3600
        print("Execution time: ", execution_timeS, " seconds")
        print("Execution time: ", execution_timeH, " hours")
        print("Accuracy: ", accfin)

        current_state_model: dict = dict(modelo=CNNModel)
        with open(chkpoint_obj, "wb") as p:
            pickle.dump(current_state_model, p)

        # Guardar también formato PyTorch .pth con pesos entrenados
        try:
            from model_utils import save_best_model
            save_best_model(
                variant=variant,
                execution=execution,
                bestind=bestind,
                in_channels=n_channels,
                out_size=out_size,
                n_classes=n_classes,
                chck_dir=chck_dir,
                trained_model=CNNModel,
                auto_download=auto_download
            )
        except Exception as e:
            print(f"Nota al exportar pesos finales: {e}")

    else: 
        print("Loading final model")
        with open(chkpoint_obj, "rb") as p:
            values = pickle.load(p)
        CNNModel = values['modelo']
    
    return CNNModel