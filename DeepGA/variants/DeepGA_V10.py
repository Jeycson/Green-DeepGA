# -*- coding: utf-8 -*-
""" Created on Oct 2024    @author: user
    Variante DeepGA V10 (ACO-Enhanced DeepGA / Pheromone-Guided Evolution):
    - Matriz de Feromonas Arquitectónicas (ACO-NAS Pheromone Trail Matrix).
    - Depósito de Feromonas Proporcional al Fitness por Individuos Élite.
    - Evaporación Dinámica de Feromonas para Evitar Estancamiento Prematuro.
    - Mutación e Inicialización Guiadas por Probabilidad (Softmax / Boltzmann sobre Feromonas).
    - Asistencia por Meta-Modelo Subrogado (Surrogate-Assisted Search con Random Forest + UCB).
    - Operador de Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
    - Probabilidad de Mutación Adaptativa Individual (Srinivas & Patnaik + Modulación Temporal).
"""

from Operators import selection
from Operators_V10 import (
    PheromoneMatrix,
    guided_adaptive_mutation_v10,
    compute_adaptive_mutation_rate,
    crossover_v7
)
from EncodingClass import Encoding
from Decoding import decoding, CNN
from DistributedTraining import training
from torch.utils.data import DataLoader
import timeit
import torch
import pickle
import torch.nn as nn
import os
from pathlib import Path
import pandas as pd
import numpy as np
import copy
from copy import deepcopy
import random

# Importación de modelos para el subrogado
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_genome_features(e, max_conv: int = 5, max_full: int = 4) -> np.ndarray:
    """
    Convierte un genoma de DeepGA (Encoding) en un vector numérico de características
    de tamaño fijo para ser consumido por el modelo subrogado.
    """
    features = []

    # 1. Dimensiones macroestructurales
    features.append(float(e.n_conv) / float(max_conv))
    features.append(float(e.n_full) / float(max_full))

    # 2. Características por capa convolucional (fijo a max_conv)
    total_filters = 0.0
    total_kernels = 0.0
    pool_count = 0.0

    for i in range(max_conv):
        if i < e.n_conv and i < len(e.first_level):
            layer = e.first_level[i]
            nf = float(layer.get('nfilters', 16))
            ks = float(layer.get('fsize', 3))
            ps = float(layer.get('psize', 2))
            pool_val = str(layer.get('pool', 'off')).lower().strip()

            if pool_val == 'max':
                pool_code = 1.0
                has_pool = 1.0
            elif pool_val == 'avg':
                pool_code = 0.5
                has_pool = 1.0
            else:
                pool_code = 0.0
                has_pool = 0.0

            features.extend([nf / 256.0, ks / 9.0, ps / 5.0, pool_code, 1.0])
            total_filters += nf
            total_kernels += ks
            pool_count += has_pool
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # 3. Características por capa densa (fijo a max_full)
    total_neurons = 0.0
    fc_start = e.n_conv
    for j in range(max_full):
        idx = fc_start + j
        if idx < len(e.first_level):
            fc_layer = e.first_level[idx]
            neurons = float(fc_layer.get('neurons', 32))
            features.extend([neurons / 256.0, 1.0])
            total_neurons += neurons
        else:
            features.extend([0.0, 0.0])

    # 4. Conexiones residuales (segundo nivel)
    max_skips = (max_conv * (max_conv - 1)) // 2
    skips = getattr(e, 'second_level', [])
    for k in range(max_skips):
        if k < len(skips):
            features.append(float(skips[k]))
        else:
            features.append(0.0)

    # 5. Métricas globales derivadas
    features.append(total_filters / (256.0 * max_conv))
    features.append(total_kernels / (9.0 * max_conv))
    features.append(total_neurons / (256.0 * max_full))
    features.append(pool_count / float(max_conv))
    features.append(float(sum(skips)) / max(1.0, float(max_skips)))

    return np.array(features, dtype=np.float32)


class SurrogatePredictor:
    """
    Meta-modelo subrogado basado en Random Forest para predecir
    el fitness de arquitecturas DeepGA con estimación de incertidumbre.
    """
    def __init__(self, n_estimators: int = 50, max_depth: int = 8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.train_x_history = []
        self.train_y_history = []

    def add_sample(self, genome_features: np.ndarray, fitness_val: float):
        self.train_x_history.append(genome_features)
        self.train_y_history.append(fitness_val)

    def train(self):
        """Entrena el modelo subrogado con todas las muestras evaluadas en GPU hasta el momento."""
        if not SKLEARN_AVAILABLE or len(self.train_x_history) < 5:
            return False

        X = np.array(self.train_x_history)
        y = np.array(self.train_y_history)

        self.model = RandomForestRegressor(
            n_estimators=min(self.n_estimators, max(10, len(X))),
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict(self, genome_features: np.ndarray):
        """Retorna la predicción de fitness y la incertidumbre (desviación estándar entre árboles)."""
        if not self.is_trained or self.model is None:
            return 0.5, 1.0

        X = genome_features.reshape(1, -1)
        tree_preds = [tree.predict(X)[0] for tree in self.model.estimators_]
        mu = float(np.mean(tree_preds))
        sigma = float(np.std(tree_preds))
        return mu, sigma

    def acquisition_score(self, genome_features: np.ndarray, kappa: float = 0.1) -> float:
        """Función de adquisición Upper Confidence Bound (UCB)."""
        mu, sigma = self.predict(genome_features)
        return mu + (kappa * sigma)


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


def green_DeepGA_v10(execution: int, memoryC: bool, train_epochs: int, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     min_conv: int, max_conv: int, min_full: int, max_full: int, max_params: int, cr: float, mr: float,  
                     N: int, T: int, t_size: int, w: float, device: torch.device, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, loss_func=None,
                     pool_candidates_factor: int = 5, kappa: float = 0.1,
                     mr_min: float = 0.10, mr_max: float = 0.85,
                     rho: float = 0.10, alpha: float = 1.2, top_k_ratio: float = 0.35,
                     test_dl: DataLoader = None, dataset_name: str = "CIFAR-10", seed: int = None,
                     energy_kwh: float = None, emissions_g_co2: float = None, save_txt: bool = True, **kwargs):
    """
    Algoritmo DeepGA V10 (ACO-Enhanced Pheromone-Guided Evolution):
    - Matriz de Feromonas Arquitectónicas (ACO-NAS Pheromone Matrix).
    - Depósito y Evaporación de Feromonas guiado por Élite.
    - Mutación e Inicialización Guiadas por Probabilidad (Softmax).
    - Asistencia por Meta-Modelo Subrogado (Random Forest + UCB).
    - Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
    - Probabilidad de Mutación Adaptativa Individual (Srinivas & Patnaik).
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):  
        os.makedirs(chck_dir)      

    surrogate = SurrogatePredictor()
    pheromones = PheromoneMatrix(
        min_conv=min_conv, max_conv=max_conv,
        min_full=min_full, max_full=max_full,
        rho=rho, alpha=alpha
    )

    '''Initialize population'''
    chkpoint_file = os.path.join(chck_dir, f"checkpoint_v10_exec_{execution}.pkl")
    legacy_chkpoint = os.path.join(chck_dir, f"{execution}_checkpoint.pkl")

    values = None
    if os.path.exists(chkpoint_file):
        try:
            with open(chkpoint_file, "rb") as p:
                loaded = pickle.load(p)
            if isinstance(loaded, dict) and 'pop' in loaded and 'islands_pop' not in loaded:
                values = loaded
        except Exception:
            values = None
    elif os.path.exists(legacy_chkpoint):
        try:
            with open(legacy_chkpoint, "rb") as p:
                loaded = pickle.load(p)
            if isinstance(loaded, dict) and 'pop' in loaded and 'islands_pop' not in loaded:
                values = loaded
        except Exception:
            values = None

    if values is not None:
        print("Re-Initialize population (DeepGA V10 - ACO Pheromones + Surrogate + Adaptive Mutation)", flush=True)
        start = timeit.default_timer() - values['time']
        pop = values['pop']
        bestAcc = values['bestAcc']
        bestF = values['bestF']
        bestParams = values['bestParams']
        t = values['t']
        leader = max(pop, key=lambda x: x[1]) if pop else None
        if t >= T:
            print(f'The maximum number of generations has already been reached ({t}/{T}). Returning best individual from checkpoint.', flush=True)
        evals = values.get('evals', 0)
        cacheM = values.get('cacheM', {})
        meanfitpop = values.get('meanfitpop', [])
        meanAccpop = values.get('meanAccpop', [])
        meanParpop = values.get('meanParpop', [])
        evaluated_history = values.get('evaluated_history', [])
        total_screened_candidates = values.get('total_screened_candidates', 0)
        prediction_errors = values.get('prediction_errors', [])
        adaptive_mr_history = values.get('adaptive_mr_history', [])
        mutation_events = values.get('mutation_events', {'modify_layer': 0, 'flip_skip': 0, 'add_remove_layer': 0})

        # Re-alimentar datos al subrogado y matriz de feromonas
        for ind_e, fit_val, _ in evaluated_history:
            feats = extract_genome_features(ind_e, max_conv, max_full)
            surrogate.add_sample(feats, fit_val)
        surrogate.train()
        pheromones.deposit(pop, top_k_ratio=top_k_ratio)
    else:
        print('Initialize population (DeepGA V10 - ACO Pheromones + Adaptive Mutation + Surrogate + Structured Crossover)', flush=True)
        start = timeit.default_timer()
        pop = []
        bestAcc = []
        bestF = []
        bestParams = []
        t = 0
        evals = 0
        total_screened_candidates = 0
        prediction_errors = []
        adaptive_mr_history = []
        mutation_events = {'modify_layer': 0, 'flip_skip': 0, 'add_remove_layer': 0}
        cacheM = {}
        evaluated_history = []
        meanfitpop = []
        meanAccpop = []
        meanParpop = []

        # Generación 0: Inicialización guiada y recolección de datos base
        while len(pop) < N:
            e1 = pheromones.generate_guided_encoding(min_conv, max_conv, min_full, max_full, alpha=1.0)
            if memoryC:
                strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
                if strIDe1 in cacheM:
                    fit1, acc1, pars1 = cacheM[strIDe1]
                    print(f"[Cache Init V10] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}", flush=True)
                    pop.append([e1, fit1, acc1, pars1])
                    evaluated_history.append((e1, fit1, acc1))
                else:
                    print(f"[Init V10] Evaluando individuo {len(pop)+1}/{N} en GPU...", flush=True)
                    fit1, acc1, pars1 = _evaluate_individual(
                        e1, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    cacheM[strIDe1] = [fit1, acc1, pars1]
                    evals += 1
                    evaluated_history.append((e1, fit1, acc1))
                    print(f"[Sequential Init V10] Ind {len(pop)+1}/{N} -> Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1:,}", flush=True)
                    pop.append([e1, fit1, acc1, pars1])
            else:
                print(f"[Init V10] Evaluando individuo {len(pop)+1}/{N} en GPU...", flush=True)
                fit1, acc1, pars1 = _evaluate_individual(
                    e1, n_channels, out_size, n_classes, device1,
                    num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                )
                evals += 1
                evaluated_history.append((e1, fit1, acc1))
                print(f"[Sequential Init V10] Ind {len(pop)+1}/{N} -> Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1:,}", flush=True)
                pop.append([e1, fit1, acc1, pars1])

            # Agregar muestra al subrogado
            feats = extract_genome_features(e1, max_conv, max_full)
            surrogate.add_sample(feats, fit1)

        # Entrenar primer subrogado y depositar primer rastro de feromonas
        surrogate.train()
        pheromones.deposit(pop, top_k_ratio=top_k_ratio)
        print(f"✓ Modelo Subrogado y Matriz de Feromonas inicializados con {len(surrogate.train_x_history)} arquitecturas.", flush=True)
        leader = max(pop, key=lambda x: x[1]) if pop else None

    '''Genetic Algorithm Asistido por Feromonas (ACO) y Subrogado V10'''
    print('--------------------------------------------', flush=True)
    while t < T:
        print(f'Generation: {t} (DeepGA V10 - ACO Pheromone-Guided + Surrogate-Assisted)', flush=True)

        # 1. Evaporación y Actualización de Feromonas
        pheromones.evaporate()
        pheromones.deposit(pop, top_k_ratio=top_k_ratio)

        # 2. Re-entrenar el subrogado
        surrogate.train()

        # Estadísticas de fitness de la población actual
        pop_fitnesses = [ind[1] for ind in pop]
        f_max = max(pop_fitnesses)
        f_min = min(pop_fitnesses)
        f_avg = sum(pop_fitnesses) / len(pop_fitnesses)

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

        # Reproducción: Pool en CPU con Cruce V7 + Mutación Guiada por Feromonas + Pre-Screening por Subrogado
        offspring = []
        iter_parents = 0
        candidates_per_pair = max(4, pool_candidates_factor * 2)
        gen_applied_mrs = []

        while len(offspring) < int(N/2):
            p1_obj = parents[iter_parents]
            p2_obj = parents[iter_parents + 1]
            p1, f1 = p1_obj[0], p1_obj[1]
            p2, f2 = p2_obj[0], p2_obj[1]

            mr_adapt1 = compute_adaptive_mutation_rate(f1, f_max, f_avg, f_min, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            mr_adapt2 = compute_adaptive_mutation_rate(f2, f_max, f_avg, f_min, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            gen_applied_mrs.extend([mr_adapt1, mr_adapt2])

            # Generar pool de candidatos en CPU guiados por feromonas
            candidate_pool = []
            for _ in range(candidates_per_pair // 2):
                if cr >= random.uniform(0, 1):
                    c1, c2 = crossover_v7(p1, p2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                else:
                    c1 = deepcopy(p1)
                    c2 = deepcopy(p2)

                # Mutación adaptativa guiada por feromonas (V10)
                mut1_applied, mut1_type = guided_adaptive_mutation_v10(
                    c1, pheromones, mr_adapt1, parent_fitness=f1, f_avg=f_avg,
                    min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full
                )
                if mut1_applied and mut1_type in mutation_events:
                    mutation_events[mut1_type] += 1

                mut2_applied, mut2_type = guided_adaptive_mutation_v10(
                    c2, pheromones, mr_adapt2, parent_fitness=f2, f_avg=f_avg,
                    min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full
                )
                if mut2_applied and mut2_type in mutation_events:
                    mutation_events[mut2_type] += 1

                candidate_pool.append(c1)
                candidate_pool.append(c2)

            total_screened_candidates += len(candidate_pool)

            # Pre-filtrado inteligente con el Meta-Modelo Subrogado
            if surrogate.is_trained:
                scored_candidates = []
                for cand in candidate_pool:
                    feats = extract_genome_features(cand, max_conv, max_full)
                    score = surrogate.acquisition_score(feats, kappa=kappa)
                    pred_mu, _ = surrogate.predict(feats)
                    scored_candidates.append((cand, score, pred_mu, feats))

                # Elegir los 2 mejores candidatos según adquisición UCB
                scored_candidates.sort(key=lambda item: item[1], reverse=True)
                selected_pair = scored_candidates[:2]
            else:
                selected_pair = [(c, 0.5, 0.5, extract_genome_features(c, max_conv, max_full)) for c in candidate_pool[:2]]

            # Evaluación secuencial en GPU de los 2 candidatos ganadores
            for cand, score, pred_mu, feats in selected_pair:
                if memoryC:
                    strID = str([cand.n_conv, cand.n_full, cand.first_level, cand.second_level])
                    if strID in cacheM:
                        fit, acc, pars = cacheM[strID]
                    else:
                        fit, acc, pars = _evaluate_individual(
                            cand, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                        )
                        cacheM[strID] = [fit, acc, pars]
                        evals += 1
                else:
                    fit, acc, pars = _evaluate_individual(
                        cand, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    evals += 1

                # Registrar error de predicción del subrogado
                if surrogate.is_trained:
                    pred_err = abs(pred_mu - fit)
                    prediction_errors.append(pred_err)

                surrogate.add_sample(feats, fit)
                evaluated_history.append((cand, fit, acc))
                offspring.append([cand, fit, acc, pars])

            iter_parents += 2

        # Population replacement (elitismo generacional)
        pop.extend(offspring)
        pop.sort(key=lambda x: x[1], reverse=True)
        pop = pop[:N]

        # Estadísticas de la generación
        leader = max(pop, key=lambda x: x[1])
        bestF.append(leader[1])
        bestAcc.append(leader[2])
        bestParams.append(leader[3])

        mean_fit = sum(ind[1] for ind in pop) / len(pop)
        mean_acc = sum(ind[2] for ind in pop) / len(pop)
        mean_par = sum(ind[3] for ind in pop) / len(pop)
        mean_gen_mr = float(np.mean(gen_applied_mrs)) if gen_applied_mrs else mr

        meanfitpop.append(mean_fit)
        meanAccpop.append(mean_acc)
        meanParpop.append(mean_par)
        adaptive_mr_history.append(mean_gen_mr)

        t += 1
        time = timeit.default_timer() - start

        # Checkpoint por generación
        current_state = dict(pop=pop, bestAcc=bestAcc, bestF=bestF,
                             bestParams=bestParams, t=t, evals=evals,
                             total_screened_candidates=total_screened_candidates,
                             prediction_errors=prediction_errors,
                             adaptive_mr_history=adaptive_mr_history,
                             mutation_events=mutation_events,
                             evaluated_history=evaluated_history,
                             time=time, cacheM=cacheM, meanfitpop=meanfitpop,
                             meanAccpop=meanAccpop, meanParpop=meanParpop)
        with open(os.path.join(chck_dir, f"checkpoint_v10_exec_{execution}.pkl"), "wb") as p:
            pickle.dump(current_state, p)

        avg_mae = np.mean(prediction_errors[-10:]) if prediction_errors else 0.0
        motifs = pheromones.get_top_motifs_summary()
        print(f"--- Fin Gen {t-1} | mr_adapt: {mean_gen_mr:.3f} | Explorados CPU: {total_screened_candidates} | GPU Evals: {evals} | MAE Subrogado: {avg_mae:.4f} ---", flush=True)
        print(f"    Pheromone Favors: Conv={motifs['favored_conv_count']}, FC={motifs['favored_fc_count']}, Active Skips={len(motifs['reinforced_skip_connections'])}", flush=True)
        print('Best fitness: ', leader[1], flush=True)
        print('Best accuracy: ', leader[2], flush=True)
        print('Best No. of Params: ', leader[3], flush=True)
        print('No. of Conv. Layers: ', leader[0].n_conv, flush=True)
        print('No. of FC Layers: ', leader[0].n_full, flush=True)
        print('--------------------------------------------', flush=True)

    bestind = copy.deepcopy(leader)
    results = pd.DataFrame(list(zip(bestAcc, bestF, bestParams, meanfitpop, meanAccpop, meanParpop, adaptive_mr_history)),
                           columns=['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar', 'MeanAdaptiveMR'])

    # Estadísticas de Feromonas y Subrogado en V10
    top_motifs = pheromones.get_top_motifs_summary()
    aco_stats = {
        'total_cpu_screened': total_screened_candidates,
        'total_gpu_evaluations': evals,
        'exploration_multiplier': round(total_screened_candidates / max(1, evals), 2),
        'mean_absolute_error_fit': round(float(np.mean(prediction_errors)), 4) if prediction_errors else 0.0,
        'surrogate_training_samples': len(surrogate.train_x_history),
        'adaptive_mr_history': adaptive_mr_history,
        'mutation_events': mutation_events,
        'pheromone_motifs': top_motifs,
        'evaporation_rate': rho,
        'pheromone_sensitivity': alpha
    }

    # Guardar automáticamente la arquitectura del mejor modelo de esta variante (V10)
    try:
        from model_utils import save_best_model, calculate_cnn_metrics, compute_classification_metrics, save_experiment_record
        save_best_model(
            variant="v10",
            execution=execution,
            bestind=bestind,
            in_channels=n_channels,
            out_size=out_size,
            n_classes=n_classes,
            chck_dir=chck_dir
        )

        # Calcular métricas del modelo y de clasificación
        time_sec = timeit.default_timer() - start
        cnn_metrics = calculate_cnn_metrics(bestind, n_channels, out_size, n_classes)
        target_eval_dl = test_dl if test_dl is not None else val_dl
        cls_metrics = compute_classification_metrics(cnn_metrics["model"], target_eval_dl, device1)

        calc_energy = energy_kwh if energy_kwh is not None else (0.150 * (time_sec / 3600.0))
        calc_co2 = emissions_g_co2 if emissions_g_co2 is not None else (calc_energy * 430.0)

        exp_data = {
            "dataset": dataset_name,
            "method": "DeepGA_V10",
            "seed": seed if seed is not None else execution,
            "gen": T,
            "pop": N,
            "mig": "N/A",
            "epoch": num_epochs,
            "time": time_sec,
            "energy": calc_energy,
            "co2": calc_co2,
            "fitness": leader[1],
            "val_acc": leader[2],
            "test_acc": cls_metrics["accuracy"] if test_dl is not None else "N/A",
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1": cls_metrics["f1"],
            "params": cnn_metrics["total_params"],
            "memory": cnn_metrics["model_size_mb"],
            "flops": cnn_metrics["estimated_flops"],
            "evaluations": evals
        }

        if save_txt:
            save_experiment_record(exp_data, chck_dir=chck_dir, print_console=True)
    except Exception as e:
        print(f"Nota al guardar mejor modelo V10 / reporte: {e}", flush=True)

    return results, pop, bestind, aco_stats


def final_evaluation(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     max_params: int, w: float, device: torch.device, train_epochs: int, loss_func, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, variant: str = "v10", auto_download: bool = False):
    from variants.DeepGA import final_evaluation as fe
    return fe(execution, bestind, train_dl, val_dl, lr, max_params, w, device, train_epochs, loss_func, chck_dir, n_channels, n_classes, out_size, variant=variant, auto_download=auto_download)
