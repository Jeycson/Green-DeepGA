# -*- coding: utf-8 -*-
""" Created on Oct 2024    @author: user
    Variante DeepGA V11 (Multi-Island Evolutionary ACO-NAS with Isolated Pheromone Trails):
    - Modelo de Islas (Island Model) con 3 islas independientes por defecto.
    - Matriz de Feromonas Arquitectónicas Desacoplada e Independiente por Cada Isla.
    - Cero Contaminación Cruzada de Feromonas (las feromonas NO migran, sólo genomas evaluados).
    - Migración Periódica en Anillo (Ring Migration): 2 individuos cada 10 generaciones.
    - Mutación Adaptativa Guiada por Feromonas Locales (V10) y Cruce Topológico V7.
    - Asistencia por Meta-Modelo Subrogado (Random Forest + UCB) para filtrado rápido en CPU.
    - Métricas de Diversidad Inter-Islas para monitoreo continuo de nichos evolutivos.
"""

from Operators import selection
from Operators_V11 import (
    PheromoneMatrix,
    guided_adaptive_mutation_v10,
    compute_adaptive_mutation_rate,
    crossover_v7,
    perform_island_migration,
    compute_inter_island_diversity,
    compute_population_diversity
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


def green_DeepGA_v11(execution: int, memoryC: bool, train_epochs: int, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     min_conv: int, max_conv: int, min_full: int, max_full: int, max_params: int, cr: float, mr: float,  
                     N: int, T: int, t_size: int, w: float, device: torch.device, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, loss_func=None,
                     n_islands: int = 3, migration_interval: int = 10, migration_size: int = 2,
                     pool_candidates_factor: int = 4, kappa: float = 0.1,
                     mr_min: float = 0.10, mr_max: float = 0.85,
                     rho: float = 0.10, alpha: float = 1.2, top_k_ratio: float = 0.35,
                     test_dl: DataLoader = None, dataset_name: str = "CIFAR-10", seed: int = None,
                     energy_kwh: float = None, emissions_g_co2: float = None, save_txt: bool = True, **kwargs):
    """
    Algoritmo DeepGA V11 (Multi-Island Model con Feromonas Aisladas e Independientes):
    - n_islands: Cantidad de islas evolutivas (por defecto: 3).
    - migration_interval: Frecuencia de migración en generaciones (por defecto: cada 10 generaciones).
    - migration_size: Cantidad de individuos élite que migran por isla (por defecto: 2).
    - Matrices de Feromonas: Cada isla posee su propia PheromoneMatrix independiente. Las matrices
      NUNCA migran ni se promedian para mantener la máxima diversidad del espacio de búsqueda.
    - Cruce Topológico V7 y Mutación Adaptativa Guiada por Feromonas Locales V10.
    - Meta-Modelo Subrogado para optimización de recursos GPU en CPU.
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):  
        os.makedirs(chck_dir)

    # Configuración de tamaño de subpoblación por isla
    n_islands = max(2, int(n_islands))
    pop_per_island = max(4, int(N // n_islands))
    total_effective_N = pop_per_island * n_islands

    # Meta-modelo subrogado global
    surrogate = SurrogatePredictor()

    # Cada isla tiene su propia matriz de feromonas independiente (desacoplada)
    islands_pheromones = [
        PheromoneMatrix(
            min_conv=min_conv, max_conv=max_conv,
            min_full=min_full, max_full=max_full,
            rho=rho, alpha=alpha
        )
        for _ in range(n_islands)
    ]

    '''Inicialización de Poblaciones Multi-Isla'''
    chkpoint_obj = Path(chck_dir + str(execution) + "_checkpoint.pkl")
    if chkpoint_obj.exists():
        print(f"Re-Initialize population (DeepGA V11 - {n_islands} Islands | Migration every {migration_interval} gens | {migration_size} migrants)", flush=True)
        with open(chck_dir + str(execution) + "_checkpoint.pkl", "rb") as p:
            values = pickle.load(p)
        start = timeit.default_timer() - values['time']
        islands_pop = values['islands_pop']
        bestAcc = values['bestAcc']
        bestF = values['bestF']
        bestParams = values['bestParams']
        t = values['t']
        if t == T:
            print('The maximum number of generations has been reached. Please run a new execution.', flush=True)
        evals = values['evals']
        cacheM = values['cacheM']
        meanfitpop = values['meanfitpop']
        meanAccpop = values['meanAccpop']
        meanParpop = values['meanParpop']
        evaluated_history = values.get('evaluated_history', [])
        total_screened_candidates = values.get('total_screened_candidates', 0)
        prediction_errors = values.get('prediction_errors', [])
        adaptive_mr_history = values.get('adaptive_mr_history', [])
        migrations_log = values.get('migrations_log', [])
        inter_island_diversity_history = values.get('inter_island_diversity_history', [])

        # Re-entrenar subrogado y re-depositar feromonas locales por isla
        for ind_e, fit_val, _ in evaluated_history:
            feats = extract_genome_features(ind_e, max_conv, max_full)
            surrogate.add_sample(feats, fit_val)
        surrogate.train()

        for k in range(n_islands):
            if k < len(islands_pop):
                islands_pheromones[k].deposit(islands_pop[k], top_k_ratio=top_k_ratio)
    else:
        print(f"Initialize population (DeepGA V11 - {n_islands} Islands | {pop_per_island} ind/island | Total N: {total_effective_N})", flush=True)
        start = timeit.default_timer()
        islands_pop = [[] for _ in range(n_islands)]
        bestAcc = []
        bestF = []
        bestParams = []
        t = 0
        evals = 0
        total_screened_candidates = 0
        prediction_errors = []
        adaptive_mr_history = []
        migrations_log = []
        inter_island_diversity_history = []
        cacheM = {}
        evaluated_history = []
        meanfitpop = []
        meanAccpop = []
        meanParpop = []

        # Generación 0: Inicialización independiente por isla
        for island_idx in range(n_islands):
            print(f"\n🏝️  Inicializando Isla {island_idx + 1}/{n_islands} ({pop_per_island} individuos)...", flush=True)
            island_ph = islands_pheromones[island_idx]

            while len(islands_pop[island_idx]) < pop_per_island:
                e1 = island_ph.generate_guided_encoding(min_conv, max_conv, min_full, max_full, alpha=1.0)
                if memoryC:
                    strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
                    if strIDe1 in cacheM:
                        fit1, acc1, pars1 = cacheM[strIDe1]
                        print(f"  [Cache Init V11 | Isla {island_idx+1}] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}", flush=True)
                        islands_pop[island_idx].append([e1, fit1, acc1, pars1])
                        evaluated_history.append((e1, fit1, acc1))
                    else:
                        print(f"  [Init V11 | Isla {island_idx+1}] Evaluando individuo {len(islands_pop[island_idx])+1}/{pop_per_island} en GPU...", flush=True)
                        fit1, acc1, pars1 = _evaluate_individual(
                            e1, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                        )
                        cacheM[strIDe1] = [fit1, acc1, pars1]
                        evals += 1
                        evaluated_history.append((e1, fit1, acc1))
                        print(f"  [Sequential Init V11 | Isla {island_idx+1}] Ind {len(islands_pop[island_idx])+1}/{pop_per_island} -> Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1:,}", flush=True)
                        islands_pop[island_idx].append([e1, fit1, acc1, pars1])
                else:
                    print(f"  [Init V11 | Isla {island_idx+1}] Evaluando individuo {len(islands_pop[island_idx])+1}/{pop_per_island} en GPU...", flush=True)
                    fit1, acc1, pars1 = _evaluate_individual(
                        e1, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params
                    )
                    evals += 1
                    evaluated_history.append((e1, fit1, acc1))
                    print(f"  [Sequential Init V11 | Isla {island_idx+1}] Ind {len(islands_pop[island_idx])+1}/{pop_per_island} -> Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1:,}", flush=True)
                    islands_pop[island_idx].append([e1, fit1, acc1, pars1])

                # Muestra al subrogado
                feats = extract_genome_features(e1, max_conv, max_full)
                surrogate.add_sample(feats, fit1)

            # Depositar primer rastro de feromonas exclusivo de esta isla
            island_ph.deposit(islands_pop[island_idx], top_k_ratio=top_k_ratio)

        surrogate.train()
        print(f"\n✓ Inicializadas {n_islands} Islas independientes ({total_effective_N} individuos totales en GPU).", flush=True)

    '''Ciclo Evolutivo Multi-Isla DeepGA V11'''
    print('============================================', flush=True)
    while t < T:
        print(f'\n--- Generación: {t} (DeepGA V11 - {n_islands} Islas | Migración cada {migration_interval} gens) ---', flush=True)

        # 1. Re-entrenar subrogado global
        surrogate.train()

        gen_applied_mrs = []
        island_bests = []

        # 2. Evolución intra-isla con feromonas locales y reproducción en cada isla
        for island_idx in range(n_islands):
            island_pop = islands_pop[island_idx]
            island_ph = islands_pheromones[island_idx]

            # 2.1 Evaporación y depósito exclusivo sobre la matriz de esta isla
            island_ph.evaporate()
            island_ph.deposit(island_pop, top_k_ratio=top_k_ratio)

            # Estadísticas de fitness de la isla actual
            isl_fitnesses = [ind[1] for ind in island_pop]
            f_max = max(isl_fitnesses)
            f_min = min(isl_fitnesses)
            f_avg = sum(isl_fitnesses) / len(isl_fitnesses)

            # Selección por torneo dentro de la isla
            effective_t_size = max(2, min(t_size, len(island_pop)))
            parents = []
            num_offspring_pairs = max(1, pop_per_island // 4)
            while len(parents) < (num_offspring_pairs * 2):
                t1 = random.sample(island_pop, effective_t_size)
                p1 = selection(t1, 'max')
                t2 = random.sample(island_pop, effective_t_size)
                p2 = selection(t2, 'max')
                while p1 == p2 and len(island_pop) > 1:
                    t2 = random.sample(island_pop, effective_t_size)
                    p2 = selection(t2, 'max')
                parents.append(p1)
                parents.append(p2)

            # Reproducción y screening en CPU usando las feromonas de esta isla
            candidates_per_pair = max(4, pool_candidates_factor * 2)
            offspring = []
            iter_parents = 0

            while iter_parents < len(parents):
                p1_obj = parents[iter_parents]
                p2_obj = parents[iter_parents + 1]
                p1, f1 = p1_obj[0], p1_obj[1]
                p2, f2 = p2_obj[0], p2_obj[1]

                mr_adapt1 = compute_adaptive_mutation_rate(f1, f_max, f_avg, f_min, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
                mr_adapt2 = compute_adaptive_mutation_rate(f2, f_max, f_avg, f_min, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
                gen_applied_mrs.extend([mr_adapt1, mr_adapt2])

                # Generar candidatos guiados por la matriz de feromonas de la isla actual
                candidate_pool = []
                for _ in range(candidates_per_pair // 2):
                    if cr >= random.uniform(0, 1):
                        c1, c2 = crossover_v7(p1, p2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                    else:
                        c1 = deepcopy(p1)
                        c2 = deepcopy(p2)

                    # Mutación guiada por las feromonas exclusivas de esta isla
                    guided_adaptive_mutation_v10(
                        c1, island_ph, mr_adapt1, parent_fitness=f1, f_avg=f_avg,
                        min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full
                    )
                    guided_adaptive_mutation_v10(
                        c2, island_ph, mr_adapt2, parent_fitness=f2, f_avg=f_avg,
                        min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full
                    )

                    candidate_pool.append(c1)
                    candidate_pool.append(c2)

                total_screened_candidates += len(candidate_pool)

                # Pre-filtrado inteligente por subrogado (UCB)
                if surrogate.is_trained:
                    scored_candidates = []
                    for cand in candidate_pool:
                        feats = extract_genome_features(cand, max_conv, max_full)
                        score = surrogate.acquisition_score(feats, kappa=kappa)
                        pred_mu, _ = surrogate.predict(feats)
                        scored_candidates.append((cand, score, pred_mu, feats))

                    scored_candidates.sort(key=lambda item: item[1], reverse=True)
                    selected_pair = scored_candidates[:2]
                else:
                    selected_pair = [(c, 0.5, 0.5, extract_genome_features(c, max_conv, max_full)) for c in candidate_pool[:2]]

                # Evaluación en GPU de los candidatos ganadores
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

                    if surrogate.is_trained:
                        pred_err = abs(pred_mu - fit)
                        prediction_errors.append(pred_err)

                    surrogate.add_sample(feats, fit)
                    evaluated_history.append((cand, fit, acc))
                    offspring.append([cand, fit, acc, pars])

                iter_parents += 2

            # Reemplazo elitista intra-isla
            island_pop.extend(offspring)
            island_pop.sort(key=lambda x: x[1], reverse=True)
            islands_pop[island_idx] = island_pop[:pop_per_island]

            best_in_island = islands_pop[island_idx][0]
            island_bests.append(best_in_island)
            print(f"  🏝️  [Isla {island_idx+1}] Mejor Fit: {best_in_island[1]:.4f} | Acc: {best_in_island[2]:.2f}% | Params: {best_in_island[3]:,} | Capas: Conv={best_in_island[0].n_conv}, FC={best_in_island[0].n_full}", flush=True)

        # 3. Migración de Individuos entre Islas (Cada `migration_interval` generaciones)
        #    IMPORTANTE: No se migran las matrices de feromonas, sólo copias de individuos.
        migration_occurred = False
        current_gen_records = []
        if (t + 1) % migration_interval == 0 and (t + 1) < T:
            print(f"\n🚢 [MIGRACIÓN INTER-ISLAS - Generación {t+1}] Migrando {migration_size} individuos por isla (Topología Anillo)...", flush=True)
            islands_pop, current_gen_records = perform_island_migration(
                islands_pop=islands_pop,
                migration_size=migration_size,
                topology="ring"
            )
            migration_occurred = True
            migrations_log.append({
                "generation": t + 1,
                "records": current_gen_records
            })
            for r in current_gen_records:
                print(f"   ✈️  Isla {r['source_island']+1} ➔ Isla {r['target_island']+1}: {r['num_immigrants_sent']} enviados | {r['num_immigrants_accepted']} integrados en élite local | Fits: {r['immigrant_fitnesses']}", flush=True)
            print("   🔒 [Aislamiento Estricto] Las matrices de feromonas permanecen desacopladas sin contaminación cruzada.\n", flush=True)

        # 4. Estadísticas Globales y Diversidad
        all_individuals = [ind for island in islands_pop for ind in island]
        global_leader = max(all_individuals, key=lambda x: x[1])

        bestF.append(global_leader[1])
        bestAcc.append(global_leader[2])
        bestParams.append(global_leader[3])

        mean_fit = sum(ind[1] for ind in all_individuals) / len(all_individuals)
        mean_acc = sum(ind[2] for ind in all_individuals) / len(all_individuals)
        mean_par = sum(ind[3] for ind in all_individuals) / len(all_individuals)
        mean_gen_mr = float(np.mean(gen_applied_mrs)) if gen_applied_mrs else mr

        inter_div = compute_inter_island_diversity(islands_pop, max_conv, max_full)
        inter_island_diversity_history.append(inter_div)

        meanfitpop.append(mean_fit)
        meanAccpop.append(mean_acc)
        meanParpop.append(mean_par)
        adaptive_mr_history.append(mean_gen_mr)

        t += 1
        time = timeit.default_timer() - start

        # Checkpoint multi-isla por generación
        current_state = dict(
            islands_pop=islands_pop,
            bestAcc=bestAcc,
            bestF=bestF,
            bestParams=bestParams,
            t=t,
            evals=evals,
            total_screened_candidates=total_screened_candidates,
            prediction_errors=prediction_errors,
            adaptive_mr_history=adaptive_mr_history,
            migrations_log=migrations_log,
            inter_island_diversity_history=inter_island_diversity_history,
            evaluated_history=evaluated_history,
            time=time,
            cacheM=cacheM,
            meanfitpop=meanfitpop,
            meanAccpop=meanAccpop,
            meanParpop=meanParpop
        )
        with open(chck_dir + str(execution) + "_checkpoint.pkl", "wb") as p:
            pickle.dump(current_state, p)

        avg_mae = np.mean(prediction_errors[-10:]) if prediction_errors else 0.0
        print(f"--- Fin Gen {t-1} | Diversidad Inter-Islas: {inter_div:.4f} | CPU Screened: {total_screened_candidates} | GPU Evals: {evals} | MAE Subrogado: {avg_mae:.4f} ---", flush=True)
        print(f"🏆 LÍDER GLOBAL V11 -> Fitness: {global_leader[1]:.4f} | Acc: {global_leader[2]:.2f}% | Params: {global_leader[3]:,} | Conv: {global_leader[0].n_conv}, FC: {global_leader[0].n_full}", flush=True)
        print('============================================', flush=True)

    # Consolidación de población final y mejor individuo global
    all_final_pop = [ind for island in islands_pop for ind in island]
    all_final_pop.sort(key=lambda x: x[1], reverse=True)
    bestind = copy.deepcopy(all_final_pop[0])

    results = pd.DataFrame(
        list(zip(bestAcc, bestF, bestParams, meanfitpop, meanAccpop, meanParpop, adaptive_mr_history, inter_island_diversity_history)),
        columns=['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar', 'MeanAdaptiveMR', 'InterIslandDiversity']
    )

    # Resumen de motivos de feromonas por cada isla
    islands_summary = []
    for k in range(n_islands):
        island_leader = max(islands_pop[k], key=lambda x: x[1])
        island_motifs = islands_pheromones[k].get_top_motifs_summary()
        isl_div = compute_population_diversity(islands_pop[k], max_conv, max_full)
        islands_summary.append({
            'island_id': k + 1,
            'best_fitness': round(island_leader[1], 4),
            'best_accuracy': round(island_leader[2], 2),
            'best_params': island_leader[3],
            'conv_layers': island_leader[0].n_conv,
            'fc_layers': island_leader[0].n_full,
            'intra_island_diversity': round(isl_div, 4),
            'pheromone_motifs': island_motifs
        })

    island_stats = {
        'n_islands': n_islands,
        'pop_per_island': pop_per_island,
        'total_population': total_effective_N,
        'migration_interval': migration_interval,
        'migration_size': migration_size,
        'total_migrations_performed': len(migrations_log),
        'migrations_log': migrations_log,
        'islands_summary': islands_summary,
        'inter_island_diversity_history': inter_island_diversity_history,
        'final_inter_island_diversity': round(inter_island_diversity_history[-1], 4) if inter_island_diversity_history else 0.0,
        'total_cpu_screened': total_screened_candidates,
        'total_gpu_evaluations': evals,
        'exploration_multiplier': round(total_screened_candidates / max(1, evals), 2),
        'mean_absolute_error_fit': round(float(np.mean(prediction_errors)), 4) if prediction_errors else 0.0,
        'surrogate_training_samples': len(surrogate.train_x_history),
        'evaporation_rate': rho,
        'pheromone_sensitivity': alpha
    }

    # Guardar automáticamente la arquitectura del mejor modelo de esta variante (V11)
    try:
        from model_utils import save_best_model, calculate_cnn_metrics, compute_classification_metrics, save_experiment_record
        save_best_model(
            variant="v11",
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
            "method": "DeepGA_V11",
            "seed": seed if seed is not None else execution,
            "gen": T,
            "pop": total_effective_N,
            "mig": f"{migration_interval}/{migration_size}",
            "epoch": num_epochs,
            "time": time_sec,
            "energy": calc_energy,
            "co2": calc_co2,
            "fitness": bestind[1],
            "val_acc": bestind[2],
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
        print(f"Nota al guardar mejor modelo V11 / reporte: {e}", flush=True)

    return results, all_final_pop, bestind, island_stats


def final_evaluation(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     max_params: int, w: float, device: torch.device, train_epochs: int, loss_func, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, variant: str = "v11", auto_download: bool = False):
    from variants.DeepGA import final_evaluation as fe
    return fe(execution, bestind, train_dl, val_dl, lr, max_params, w, device, train_epochs, loss_func, chck_dir, n_channels, n_classes, out_size, variant=variant, auto_download=auto_download)
