# -*- coding: utf-8 -*-
""" Created on Sep 2024    @author: user
    Variante DeepGA V9:
    - Asistencia por Meta-Modelo Subrogado (Surrogate-Assisted Search con Random Forest + UCB).
    - Operador de Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
    - Probabilidad de Mutación Adaptativa Individual (Srinivas & Patnaik + Modulación Temporal).
    - Mutación Especializada según Desempeño (Micro-ajustes para élites vs Macro-estructural para bajos rendimientos).
"""

from Operators import selection
from Operators_V9 import crossover_v7, adaptive_mutation_v9, compute_adaptive_mutation_rate
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

            # Codificación numérica de pooling: 0.0 = 'off', 0.5 = 'avg', 1.0 = 'max'
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

    # 3. Características por capa densa / Fully Connected (fijo a max_full)
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
        """
        Retorna la predicción de fitness y la incertidumbre (desviación estándar entre árboles).
        """
        if not self.is_trained or self.model is None:
            return 0.5, 1.0

        X = genome_features.reshape(1, -1)
        tree_preds = [tree.predict(X)[0] for tree in self.model.estimators_]
        mu = float(np.mean(tree_preds))
        sigma = float(np.std(tree_preds))
        return mu, sigma

    def acquisition_score(self, genome_features: np.ndarray, kappa: float = 0.1) -> float:
        """
        Función de adquisición Upper Confidence Bound (UCB):
        Score = Media Predicha + (kappa * Incertidumbre)
        """
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


def green_DeepGA_v9(execution: int, memoryC: bool, train_epochs: int, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                    min_conv: int, max_conv: int, min_full: int, max_full: int, max_params: int, cr: float, mr: float,  
                    N: int, T: int, t_size: int, w: float, device: torch.device, chck_dir: str,
                    n_channels: int = 3, n_classes: int = 10, out_size: int = 32, loss_func=None,
                    pool_candidates_factor: int = 5, kappa: float = 0.1,
                    mr_min: float = 0.10, mr_max: float = 0.85, **kwargs):
    """
    Algoritmo DeepGA V9:
    - Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
    - Probabilidad de Mutación Adaptativa Individual (Srinivas & Patnaik + Annealing).
    - Mutación Especializada según Desempeño (Micro vs Macro).
    - Asistencia por Meta-Modelo Subrogado (Random Forest + UCB).
    - Entrenamiento Secuencial en GPU / VRAM.
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):  
        os.makedirs(chck_dir)      

    surrogate = SurrogatePredictor()

    '''Initialize population'''
    chkpoint_obj = Path(chck_dir + str(execution) + "_checkpoint.pkl")
    if chkpoint_obj.exists():
        print("Re-Initialize population (DeepGA V9 - Surrogate + Adaptive Mutation)")
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
        total_screened_candidates = values.get('total_screened_candidates', 0)
        prediction_errors = values.get('prediction_errors', [])
        adaptive_mr_history = values.get('adaptive_mr_history', [])
        mutation_events = values.get('mutation_events', {'modify_layer': 0, 'flip_skip': 0, 'add_remove_layer': 0})

        # Re-alimentar datos al subrogado
        for ind_e, fit_val, _ in evaluated_history:
            feats = extract_genome_features(ind_e, max_conv, max_full)
            surrogate.add_sample(feats, fit_val)
        surrogate.train()
    else:
        print('Initialize population (DeepGA V9 - Adaptive Mutation + Surrogate + Structured Crossover)')
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

        # Generación 0: Inicialización y recolección de datos base para el subrogado
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
                    print(f"[Sequential Init V9] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                    pop.append([e1, fit1, acc1, pars1])
            else:
                fit1, acc1, pars1 = _evaluate_individual(
                    e1, n_channels, out_size, n_classes, device1,
                    num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                )
                evals += 1
                evaluated_history.append((e1, fit1, acc1))
                print(f"[Sequential Init V9] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}")
                pop.append([e1, fit1, acc1, pars1])

            # Agregar muestra al dataset de entrenamiento del subrogado
            feats = extract_genome_features(e1, max_conv, max_full)
            surrogate.add_sample(feats, fit1)

        # Entrenar primer modelo subrogado con la población inicial
        surrogate.train()
        print(f"✓ Modelo Subrogado inicializado con {len(surrogate.train_x_history)} arquitecturas.")

    '''Genetic Algorithm Asistido por Subrogado y Mutación Adaptativa V9'''
    print('--------------------------------------------')
    while t < T:
        print(f'Generation: {t} (DeepGA V9 - Adaptive Mutation + Surrogate Assisted)')

        # Re-entrenar el subrogado al inicio de la generación con los datos acumulados
        surrogate.train()

        # Estadísticas de fitness de la población actual para guiar la mutación adaptativa
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

        # Reproducción: Pool en CPU con Cruce Estructurado + Mutación Adaptativa + Pre-Screening por Subrogado
        offspring = []
        iter_parents = 0
        candidates_per_pair = max(4, pool_candidates_factor * 2)
        gen_applied_mrs = []

        while len(offspring) < int(N/2):
            p1_obj = parents[iter_parents]
            p2_obj = parents[iter_parents + 1]
            p1, f1 = p1_obj[0], p1_obj[1]
            p2, f2 = p2_obj[0], p2_obj[1]

            # 1. Calcular tasas de mutación adaptativas individuales según el desempeño del padre y la generación
            mr_adapt1 = compute_adaptive_mutation_rate(f1, f_max, f_avg, f_min, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            mr_adapt2 = compute_adaptive_mutation_rate(f2, f_max, f_avg, f_min, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            gen_applied_mrs.extend([mr_adapt1, mr_adapt2])

            # 2. Generar pool de candidatos en CPU con Cruce V7 y Mutación Adaptativa
            candidate_pool = []
            for _ in range(candidates_per_pair // 2):
                if cr >= random.uniform(0, 1):
                    c1, c2 = crossover_v7(p1, p2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                else:
                    c1 = deepcopy(p1)
                    c2 = deepcopy(p2)

                # Mutación adaptativa especializada (intensidad y tasa según desempeño de padres)
                mut1_applied, mut1_type = adaptive_mutation_v9(c1, mr_adapt1, parent_fitness=f1, f_avg=f_avg,
                                                               min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                if mut1_applied and mut1_type in mutation_events:
                    mutation_events[mut1_type] += 1

                mut2_applied, mut2_type = adaptive_mutation_v9(c2, mr_adapt2, parent_fitness=f2, f_avg=f_avg,
                                                               min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                if mut2_applied and mut2_type in mutation_events:
                    mutation_events[mut2_type] += 1

                candidate_pool.extend([c1, c2])

            total_screened_candidates += len(candidate_pool)

            # 3. Evaluar todo el pool en CPU usando el Subrogado (< 5 ms)
            scored_candidates = []
            for cand in candidate_pool:
                cand_feats = extract_genome_features(cand, max_conv, max_full)
                pred_fit, uncertainty = surrogate.predict(cand_feats)
                acq_score = surrogate.acquisition_score(cand_feats, kappa=kappa)
                scored_candidates.append({
                    'genome': cand,
                    'pred_fit': pred_fit,
                    'uncertainty': uncertainty,
                    'acq_score': acq_score,
                    'features': cand_feats
                })

            # 4. Ordenar por score de adquisición del subrogado y seleccionar los 2 mejores
            scored_candidates.sort(key=lambda x: x['acq_score'], reverse=True)
            top_selected = scored_candidates[:2]

            print(f" [🧬 V9 Adaptativo] Pool: {len(candidate_pool)} candidatos (mr_p1={mr_adapt1:.2f}, mr_p2={mr_adapt2:.2f}). "
                  f"Top 1 Pred: {top_selected[0]['pred_fit']:.4f} (±{top_selected[0]['uncertainty']:.3f}), "
                  f"Top 2 Pred: {top_selected[1]['pred_fit']:.4f} (±{top_selected[1]['uncertainty']:.3f})")

            # 5. Entrenar en GPU ÚNICAMENTE los 2 mejores candidatos seleccionados
            selected_results = []
            for item in top_selected:
                cand_genome = item['genome']
                pred_f = item['pred_fit']
                feats = item['features']

                if memoryC:
                    strID = str([cand_genome.n_conv, cand_genome.n_full, cand_genome.first_level, cand_genome.second_level])
                    if strID in cacheM:
                        fit, acc, pars = cacheM[strID]
                        print(f"[Cache Offspring] Fit: {fit:.4f}, Acc: {acc:.2f}%, Params: {pars}")
                    else:
                        fit, acc, pars = _evaluate_individual(
                            cand_genome, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                        )
                        cacheM[strID] = [fit, acc, pars]
                        evals += 1
                        evaluated_history.append((cand_genome, fit, acc))
                        print(f"[GPU Offspring V9] Fit Real: {fit:.4f} (Pred: {pred_f:.4f}), Acc: {acc:.2f}%, Params: {pars}")
                else:
                    fit, acc, pars = _evaluate_individual(
                        cand_genome, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    evals += 1
                    evaluated_history.append((cand_genome, fit, acc))
                    print(f"[GPU Offspring V9] Fit Real: {fit:.4f} (Pred: {pred_f:.4f}), Acc: {acc:.2f}%, Params: {pars}")

                # Registrar error de predicción y agregar al dataset del subrogado
                error = abs(fit - pred_f)
                prediction_errors.append(error)
                surrogate.add_sample(feats, fit)

                selected_results.append((cand_genome, fit, acc, pars))

            for cg, f, a, p in selected_results:
                offspring.append([cg, f, a, p])

            iter_parents += 2

        # Registrar tasa de mutación adaptativa promedio de la generación
        mean_gen_mr = float(np.mean(gen_applied_mrs)) if gen_applied_mrs else mr
        adaptive_mr_history.append(mean_gen_mr)

        # Replacement con elitismo
        pop = pop + offspring
        pop.sort(reverse=True, key=lambda x: x[1])
        pop = pop[:N]

        leader = max(pop, key=lambda x: x[1])
        bestAcc.append(leader[2])
        bestF.append(leader[1])
        bestParams.append(leader[3])
        meanfitpop.append(sum([q[1] for q in pop])/N)
        meanAccpop.append(sum([q[2] for q in pop])/N)
        meanParpop.append(sum([q[3] for q in pop])/N)

        t += 1

        # Checkpoint con estado de V9
        time = timeit.default_timer() - start
        current_state: dict = dict(pop=pop, bestAcc=bestAcc, bestF=bestF,
                                   bestParams=bestParams, t=t, evals=evals,
                                   total_screened_candidates=total_screened_candidates,
                                   prediction_errors=prediction_errors,
                                   adaptive_mr_history=adaptive_mr_history,
                                   mutation_events=mutation_events,
                                   evaluated_history=evaluated_history,
                                   time=time, cacheM=cacheM, meanfitpop=meanfitpop,
                                   meanAccpop=meanAccpop, meanParpop=meanParpop)
        with open(chck_dir + str(execution) + "_checkpoint.pkl", "wb") as p:
            pickle.dump(current_state, p)

        avg_mae = np.mean(prediction_errors[-10:]) if prediction_errors else 0.0
        print(f"--- Fin Gen {t-1} | mr_adapt_prom: {mean_gen_mr:.3f} | Total Explorados en CPU: {total_screened_candidates} | GPU Evals: {evals} | MAE Subrogado: {avg_mae:.4f} ---")
        print('Best fitness: ', leader[1])
        print('Best accuracy: ', leader[2])
        print('Best No. of Params: ', leader[3])
        print('No. of Conv. Layers: ', leader[0].n_conv)
        print('No. of FC Layers: ', leader[0].n_full)
        print('--------------------------------------------')

    bestind = copy.deepcopy(leader)
    results = pd.DataFrame(list(zip(bestAcc, bestF, bestParams, meanfitpop, meanAccpop, meanParpop, adaptive_mr_history)),
                           columns = ['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar', 'MeanAdaptiveMR'])

    # Estadísticas de Subrogado y Mutación Adaptativa en V9
    surrogate_stats = {
        'total_cpu_screened': total_screened_candidates,
        'total_gpu_evaluations': evals,
        'exploration_multiplier': round(total_screened_candidates / max(1, evals), 2),
        'mean_absolute_error_fit': round(float(np.mean(prediction_errors)), 4) if prediction_errors else 0.0,
        'surrogate_training_samples': len(surrogate.train_x_history),
        'adaptive_mr_history': adaptive_mr_history,
        'mutation_events': mutation_events
    }

    return results, pop, bestind, surrogate_stats
