# -*- coding: utf-8 -*-
""" Created on 2026
    Variante DeepGA V12 (Pure Multi-Island Evolutionary NAS with Diversity Preservation & Memory Optimization):
    - Modelo de Islas Puro (Sin Feromonas Artificiales para Evitar Estancamiento Prematuro).
    - Cruce Topológico Basado en Grafos V7 y Mutación Estocástica Estructural/Paramétrica V12.
    - Tasa de Mutación Adaptativa Dinámica con Retroalimentación de Diversidad Local.
    - Migración Espaciada con Reemplazo por Similitud Estructural (Deterministic Crowding).
    - Mecanismo Anti-Estancamiento Activo (Hipermutación y Random Immigrant por Isla).
    - Meta-Modelo Subrogado de Doble Vía (1 por Explotación UCB + 1 por Máxima Incertidumbre/Novedad).
    - Optimización Avanzada de Memoria VRAM (AMP FP16, GAP / Adaptive Pooling y OOM Guard para imágenes grandes como 256x256).
    - Monitoreo Continuo de Diversidad Intra e Inter-Islas.
"""

import os
import copy
from copy import deepcopy
import random
import timeit
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from EncodingClass import Encoding
from Decoding import decoding, CNN
from DistributedTraining import training
from Operators import selection
from Operators_V12 import (
    generate_random_encoding,
    stochastic_mutation_v12,
    compute_diversity_adaptive_mutation_rate,
    crossover_v7,
    perform_island_migration_v12,
    compute_population_diversity,
    compute_inter_island_diversity,
    apply_island_anti_stagnation
)
from memory_optimizer import GPUMemoryOptimizer, safe_train_val_with_amp

# Importación de modelos para el subrogado
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_genome_features(e: Encoding, max_conv: int = 5, max_full: int = 4) -> np.ndarray:
    """
    Convierte un genoma de DeepGA en un vector numérico de características de tamaño fijo
    para el modelo subrogado.
    """
    features = []
    features.append(float(e.n_conv) / float(max_conv))
    features.append(float(e.n_full) / float(max_full))

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

            pool_code = 1.0 if pool_val == 'max' else (0.5 if pool_val == 'avg' else 0.0)
            has_pool = 1.0 if pool_val in ['max', 'avg'] else 0.0

            features.extend([nf / 256.0, ks / 9.0, ps / 5.0, pool_code, 1.0])
            total_filters += nf
            total_kernels += ks
            pool_count += has_pool
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    fc_start = e.n_conv
    total_neurons = 0.0
    for j in range(max_full):
        idx = fc_start + j
        if idx < len(e.first_level):
            fc_layer = e.first_level[idx]
            neurons = float(fc_layer.get('neurons', 32))
            features.extend([neurons / 256.0, 1.0])
            total_neurons += neurons
        else:
            features.extend([0.0, 0.0])

    max_skips = (max_conv * (max_conv - 1)) // 2
    skips = getattr(e, 'second_level', [])
    for k in range(max_skips):
        if k < len(skips):
            features.append(float(skips[k]))
        else:
            features.append(0.0)

    features.append(total_filters / (256.0 * max_conv))
    features.append(total_kernels / (9.0 * max_conv))
    features.append(total_neurons / (256.0 * max_full))
    features.append(pool_count / float(max_conv))
    features.append(float(sum(skips)) / max(1.0, float(max_skips)))

    return np.array(features, dtype=np.float32)


class DiversitySurrogatePredictor:
    """
    Meta-modelo subrogado basado en Random Forest diseñado para equilibrar
    la explotación del fitness con la exploración activa de novedad (alta incertidumbre).
    """
    def __init__(self, n_estimators: int = 50, max_depth: int = 8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.train_x = []
        self.train_y = []

    def add_sample(self, genome_features: np.ndarray, fitness_val: float):
        self.train_x.append(genome_features)
        self.train_y.append(fitness_val)

    def train(self) -> bool:
        if not SKLEARN_AVAILABLE or len(self.train_x) < 5:
            return False

        X = np.array(self.train_x)
        y = np.array(self.train_y)

        self.model = RandomForestRegressor(
            n_estimators=min(self.n_estimators, max(10, len(X))),
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict(self, genome_features: np.ndarray) -> Tuple[float, float]:
        if not self.is_trained or self.model is None:
            return 0.5, 1.0

        X = genome_features.reshape(1, -1)
        preds = [tree.predict(X)[0] for tree in self.model.estimators_]
        mu = float(np.mean(preds))
        sigma = float(np.std(preds))
        return mu, sigma

    def select_dual_candidates(self, candidate_pool: list, max_conv: int, max_full: int,
                               kappa: float = 0.2) -> list:
        """
        Selecciona 2 candidatos de forma balanceada:
        - Candidato 1: Mejor score de explotación / UCB (mu + kappa * sigma).
        - Candidato 2: Mayor incertidumbre / novedad (argmax sigma) para explorar nichos desconocidos.
        """
        if not self.is_trained or len(candidate_pool) < 2:
            return [(c, 0.5, 0.5, extract_genome_features(c, max_conv, max_full)) for c in candidate_pool[:2]]

        scored = []
        for cand in candidate_pool:
            feats = extract_genome_features(cand, max_conv, max_full)
            mu, sigma = self.predict(feats)
            ucb = mu + (kappa * sigma)
            scored.append({
                "cand": cand,
                "mu": mu,
                "sigma": sigma,
                "ucb": ucb,
                "feats": feats
            })

        # 1. Candidato por explotación (mejor UCB)
        scored_by_ucb = sorted(scored, key=lambda x: x["ucb"], reverse=True)
        best_exploitation = scored_by_ucb[0]

        # 2. Candidato por exploración (mayor incertidumbre entre los restantes)
        remaining = [item for item in scored_by_ucb if item["cand"] is not best_exploitation["cand"]]
        if remaining:
            best_exploration = max(remaining, key=lambda x: x["sigma"])
        else:
            best_exploration = best_exploitation

        selected = [
            (best_exploitation["cand"], best_exploitation["ucb"], best_exploitation["mu"], best_exploitation["feats"]),
            (best_exploration["cand"], best_exploration["ucb"], best_exploration["mu"], best_exploration["feats"])
        ]
        return selected


def _evaluate_individual_v12(e: Encoding, n_channels: int, out_size: int, n_classes: int,
                             device: torch.device, num_epochs: int, loss_func,
                             train_dl: DataLoader, val_dl: DataLoader, lr: float, w: float,
                             max_params: int, use_amp: bool = True, max_spatial_size: int = 4):
    """
    Decodifica y entrena una CNN en GPU con aceleración AMP y Adaptive Spatial Pooling para imágenes grandes.
    """
    try:
        network = decoding(e, n_channels, out_size, n_classes, max_spatial_size=max_spatial_size)
        cnn = CNN(e, network[0], network[1], network[2], max_spatial_size=max_spatial_size)
        
        acc_list = []
        fit, acc, pars, _ = training(
            '1', device, cnn, num_epochs, loss_func,
            train_dl, val_dl, lr, w, max_params, acc_list,
            use_amp=use_amp
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        print(f"⚠️  [OOM Warning] Fallo en evaluación de candidato por VRAM: {err}. Asignando fitness de seguridad.", flush=True)
        GPUMemoryOptimizer.cleanup_gpu_memory()
        fit = 0.0
        acc = 0.0
        pars = max_params

    # Liberación explícita de memoria GPU
    GPUMemoryOptimizer.cleanup_gpu_memory(force_sync=False)

    return fit, acc, pars


def green_DeepGA_v12(execution: int, memoryC: bool, train_epochs: int,
                     train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     min_conv: int, max_conv: int, min_full: int, max_full: int,
                     max_params: int, cr: float, mr: float,
                     N: int, T: int, t_size: int, w: float,
                     device: torch.device, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32,
                     loss_func=None, n_islands: int = 3,
                     migration_interval: int = 12, migration_size: int = 1,
                     pool_candidates_factor: int = 4, kappa: float = 0.2,
                     mr_min: float = 0.15, mr_max: float = 0.85,
                     target_diversity: float = 0.25, stagnation_limit: int = 4,
                     use_amp: bool = True, max_spatial_size: int = 4,
                     test_dl: DataLoader = None, dataset_name: str = "CIFAR-10", seed: int = None,
                     energy_kwh: float = None, emissions_g_co2: float = None, save_txt: bool = True, **kwargs):
    """
    Algoritmo DeepGA V12 (Pure Multi-Island Model con Preservación de Diversidad y Optimización de Memoria):
    - n_islands: Cantidad de islas evolutivas independientes (por defecto: 3).
    - Sin feromonas artificiales: Se recupera la estocasticidad pura y la diversidad de búsqueda.
    - migration_interval: Intervalo de migración más espaciado (por defecto: 12 generaciones).
    - migration_size: Tamaño de migración controlado (por defecto: 1 individuo élite por isla).
    - Reemplazo por Deterministic Crowding: El inmigrante reemplaza a su vecino más similar.
    - Mecanismo Anti-Estancamiento: Hipermutación y Random Immigrant si una isla se estanca `stagnation_limit` gens.
    - Subrogado de Doble Vía: 1 candidato por explotación + 1 por máxima incertidumbre (novedad).
    - Optimización de VRAM: AMP FP16 + Adaptive Spatial Pooling para soportar imágenes de alta resolución (ej. 256x256).
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):
        os.makedirs(chck_dir)

    # Configuración de subpoblaciones por isla
    n_islands = max(2, int(n_islands))
    pop_per_island = max(4, int(N // n_islands))
    total_effective_N = pop_per_island * n_islands

    # Meta-modelo subrogado enfocado en diversidad
    surrogate = DiversitySurrogatePredictor()

    '''Inicialización de Poblaciones Multi-Isla V12'''
    chkpoint_file = os.path.join(chck_dir, f"checkpoint_v12_exec_{execution}.pkl")
    legacy_chkpoint = os.path.join(chck_dir, f"{execution}_checkpoint.pkl")

    values = None
    if os.path.exists(chkpoint_file):
        try:
            with open(chkpoint_file, "rb") as p:
                loaded = pickle.load(p)
            if isinstance(loaded, dict) and 'islands_pop' in loaded:
                values = loaded
        except Exception:
            values = None
    elif os.path.exists(legacy_chkpoint):
        try:
            with open(legacy_chkpoint, "rb") as p:
                loaded = pickle.load(p)
            if isinstance(loaded, dict) and 'islands_pop' in loaded:
                values = loaded
        except Exception:
            values = None

    if values is not None:
        print(f"Re-Initialize population (DeepGA V12 - {n_islands} Pure Islands | Migration every {migration_interval} gens | Memory-Optimized)", flush=True)
        start = timeit.default_timer() - values['time']
        islands_pop = values['islands_pop']
        bestAcc = values['bestAcc']
        bestF = values['bestF']
        bestParams = values['bestParams']
        t = values['t']
        if t >= T:
            print(f'The maximum number of generations ({t}/{T}) has already been reached. Returning best individual from checkpoint.', flush=True)
        evals = values.get('evals', 0)
        cacheM = values.get('cacheM', {})
        meanfitpop = values.get('meanfitpop', [])
        meanAccpop = values.get('meanAccpop', [])
        meanParpop = values.get('meanParpop', [])
        evaluated_history = values.get('evaluated_history', [])
        total_screened_candidates = values.get('total_screened_candidates', 0)
        prediction_errors = values.get('prediction_errors', [])
        adaptive_mr_history = values.get('adaptive_mr_history', [])
        migrations_log = values.get('migrations_log', [])
        inter_island_diversity_history = values.get('inter_island_diversity_history', [])
        stagnation_trackers = values.get('stagnation_trackers', [0 for _ in range(n_islands)])

        # Re-entrenar subrogado
        for ind_e, fit_val, _ in evaluated_history:
            feats = extract_genome_features(ind_e, max_conv, max_full)
            surrogate.add_sample(feats, fit_val)
        surrogate.train()
    else:
        print(f"Initialize population (DeepGA V12 - {n_islands} Pure Islands | {pop_per_island} ind/island | Total N: {total_effective_N} | Memory AMP: {use_amp})", flush=True)
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
        stagnation_trackers = [0 for _ in range(n_islands)]
        cacheM = {}
        evaluated_history = []
        meanfitpop = []
        meanAccpop = []
        meanParpop = []

        # Generación 0: Inicialización estocástica uniforme por isla
        for island_idx in range(n_islands):
            print(f"\n🏝️  Inicializando Isla {island_idx + 1}/{n_islands} ({pop_per_island} individuos estocásticos puros)...", flush=True)
            while len(islands_pop[island_idx]) < pop_per_island:
                e1 = generate_random_encoding(min_conv, max_conv, min_full, max_full)
                if memoryC:
                    strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
                    if strIDe1 in cacheM:
                        fit1, acc1, pars1 = cacheM[strIDe1]
                        print(f"  [Cache Init V12 | Isla {island_idx+1}] Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1}", flush=True)
                        islands_pop[island_idx].append([e1, fit1, acc1, pars1])
                        evaluated_history.append((e1, fit1, acc1))
                    else:
                        fit1, acc1, pars1 = _evaluate_individual_v12(
                            e1, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, w, max_params,
                            use_amp=use_amp, max_spatial_size=max_spatial_size
                        )
                        cacheM[strIDe1] = [fit1, acc1, pars1]
                        evals += 1
                        evaluated_history.append((e1, fit1, acc1))
                        print(f"  [Init V12 | Isla {island_idx+1}] Ind {len(islands_pop[island_idx])+1}/{pop_per_island} -> Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1:,}", flush=True)
                        islands_pop[island_idx].append([e1, fit1, acc1, pars1])
                else:
                    fit1, acc1, pars1 = _evaluate_individual_v12(
                        e1, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, w, max_params,
                        use_amp=use_amp, max_spatial_size=max_spatial_size
                    )
                    evals += 1
                    evaluated_history.append((e1, fit1, acc1))
                    print(f"  [Init V12 | Isla {island_idx+1}] Ind {len(islands_pop[island_idx])+1}/{pop_per_island} -> Fit: {fit1:.4f}, Acc: {acc1:.2f}%, Params: {pars1:,}", flush=True)
                    islands_pop[island_idx].append([e1, fit1, acc1, pars1])

                feats = extract_genome_features(e1, max_conv, max_full)
                surrogate.add_sample(feats, fit1)

        surrogate.train()
        print(f"\n✓ Inicializadas {n_islands} Islas independientes sin feromonas ({total_effective_N} individuos totales evaluados).", flush=True)

    '''Ciclo Evolutivo Multi-Isla DeepGA V12'''
    print('============================================', flush=True)
    while t < T:
        print(f'\n--- Generación: {t} (DeepGA V12 - {n_islands} Islas Puras | Diversidad & Memoria Optimizada) ---', flush=True)

        # 1. Re-entrenar subrogado
        surrogate.train()

        gen_applied_mrs = []
        island_bests = []

        # 2. Evolución intra-isla sin feromonas
        for island_idx in range(n_islands):
            island_pop = islands_pop[island_idx]

            # 2.1 Medición de diversidad intra-isla actual
            intra_div = compute_population_diversity(island_pop, max_conv, max_full)

            isl_fitnesses = [ind[1] for ind in island_pop]
            f_max = max(isl_fitnesses)
            f_min = min(isl_fitnesses)
            f_avg = sum(isl_fitnesses) / len(isl_fitnesses)

            # 2.2 Selección por torneo dentro de la isla
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

            # 2.3 Reproducción estocástica pura con cruce topológico V7 y mutación balanceada
            candidates_per_pair = max(4, pool_candidates_factor * 2)
            offspring = []
            iter_parents = 0

            while iter_parents < len(parents):
                p1_obj = parents[iter_parents]
                p2_obj = parents[iter_parents + 1]
                p1, f1 = p1_obj[0], p1_obj[1]
                p2, f2 = p2_obj[0], p2_obj[1]

                mr_adapt1 = compute_diversity_adaptive_mutation_rate(
                    f1, f_max, f_avg, f_min, intra_div, target_diversity=target_diversity,
                    mr_base=mr, mr_min=mr_min, mr_max=mr_max
                )
                mr_adapt2 = compute_diversity_adaptive_mutation_rate(
                    f2, f_max, f_avg, f_min, intra_div, target_diversity=target_diversity,
                    mr_base=mr, mr_min=mr_min, mr_max=mr_max
                )
                gen_applied_mrs.extend([mr_adapt1, mr_adapt2])

                candidate_pool = []
                for _ in range(candidates_per_pair // 2):
                    if cr >= random.uniform(0, 1):
                        c1, c2 = crossover_v7(p1, p2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                    else:
                        c1 = deepcopy(p1)
                        c2 = deepcopy(p2)

                    # Mutación estocástica estructural y paramétrica sin sesgos de feromona
                    stochastic_mutation_v12(c1, mr_adapt1, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                    stochastic_mutation_v12(c2, mr_adapt2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)

                    candidate_pool.append(c1)
                    candidate_pool.append(c2)

                total_screened_candidates += len(candidate_pool)

                # 2.4 Pre-filtrado inteligente de doble vía (Explotación UCB + Novedad Incertidumbre)
                selected_pair = surrogate.select_dual_candidates(
                    candidate_pool, max_conv=max_conv, max_full=max_full, kappa=kappa
                )

                # 2.5 Evaluación acelerada en GPU con AMP y Adaptive Pooling
                for cand, score, pred_mu, feats in selected_pair:
                    if memoryC:
                        strID = str([cand.n_conv, cand.n_full, cand.first_level, cand.second_level])
                        if strID in cacheM:
                            fit, acc, pars = cacheM[strID]
                        else:
                            fit, acc, pars = _evaluate_individual_v12(
                                cand, n_channels, out_size, n_classes, device1,
                                num_epochs, loss_func, train_dl, val_dl, lr, w, max_params,
                                use_amp=use_amp, max_spatial_size=max_spatial_size
                            )
                            cacheM[strID] = [fit, acc, pars]
                            evals += 1
                    else:
                        fit, acc, pars = _evaluate_individual_v12(
                            cand, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, w, max_params,
                            use_amp=use_amp, max_spatial_size=max_spatial_size
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
            old_best_fit = island_pop[0][1] if island_pop else 0.0
            island_pop.extend(offspring)
            island_pop.sort(key=lambda x: x[1], reverse=True)
            islands_pop[island_idx] = island_pop[:pop_per_island]

            best_in_island = islands_pop[island_idx][0]
            island_bests.append(best_in_island)

            # Monitoreo de estancamiento por isla
            if best_in_island[1] > (old_best_fit + 1e-4):
                stagnation_trackers[island_idx] = 0
            else:
                stagnation_trackers[island_idx] += 1

            # 2.6 Aplicar Anti-Estancamiento si la isla no mejora
            islands_pop[island_idx], triggered = apply_island_anti_stagnation(
                islands_pop[island_idx], island_idx, stagnation_trackers[island_idx],
                stagnation_limit=stagnation_limit, min_conv=min_conv, max_conv=max_conv,
                min_full=min_full, max_full=max_full
            )
            if triggered:
                stagnation_trackers[island_idx] = 0

            print(f"  🏝️  [Isla {island_idx+1}] Mejor Fit: {best_in_island[1]:.4f} | Acc: {best_in_island[2]:.2f}% | Params: {best_in_island[3]:,} | Div: {intra_div:.4f} | Conv={best_in_island[0].n_conv}, FC={best_in_island[0].n_full}", flush=True)

        # 3. Migración Espaciada con Reemplazo por Similitud (Deterministic Crowding)
        if (t + 1) % migration_interval == 0 and (t + 1) < T:
            print(f"\n🚢 [MIGRACIÓN POR CROWDING - Gen {t+1}] Migrando {migration_size} individuo por isla con reemplazo por nicho...", flush=True)
            islands_pop, current_gen_records = perform_island_migration_v12(
                islands_pop=islands_pop,
                migration_size=migration_size,
                topology="ring",
                max_conv=max_conv,
                max_full=max_full
            )
            migrations_log.append({
                "generation": t + 1,
                "records": current_gen_records
            })
            for r in current_gen_records:
                print(f"   ✈️  Isla {r['source_island']+1} ➔ Isla {r['target_island']+1}: {r['num_immigrants_sent']} enviado | {r['num_immigrants_accepted']} integrado por Crowding | Fits: {r['immigrant_fitnesses']}", flush=True)

        # 4. Estadísticas Globales y Diversidad Inter-Islas
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

        # Checkpoint multi-isla V12
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
            stagnation_trackers=stagnation_trackers,
            evaluated_history=evaluated_history,
            time=time,
            cacheM=cacheM,
            meanfitpop=meanfitpop,
            meanAccpop=meanAccpop,
            meanParpop=meanParpop
        )
        with open(os.path.join(chck_dir, f"checkpoint_v12_exec_{execution}.pkl"), "wb") as p:
            pickle.dump(current_state, p)

        avg_mae = np.mean(prediction_errors[-10:]) if prediction_errors else 0.0
        vram_stat = GPUMemoryOptimizer.get_vram_info(device1)
        print(f"--- Fin Gen {t-1} | Diversidad Inter-Islas: {inter_div:.4f} | VRAM Libre: {vram_stat.get('free_vram_gb', 0.0)} GB | GPU Evals: {evals} | MAE Subrogado: {avg_mae:.4f} ---", flush=True)
        print(f"🏆 LÍDER GLOBAL V12 -> Fitness: {global_leader[1]:.4f} | Acc: {global_leader[2]:.2f}% | Params: {global_leader[3]:,} | Conv: {global_leader[0].n_conv}, FC: {global_leader[0].n_full}", flush=True)
        print('============================================', flush=True)

    # Consolidación final
    all_final_pop = [ind for island in islands_pop for ind in island]
    all_final_pop.sort(key=lambda x: x[1], reverse=True)
    bestind = copy.deepcopy(all_final_pop[0])

    results = pd.DataFrame(
        list(zip(bestAcc, bestF, bestParams, meanfitpop, meanAccpop, meanParpop, adaptive_mr_history, inter_island_diversity_history)),
        columns=['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar', 'MeanAdaptiveMR', 'InterIslandDiversity']
    )

    islands_summary = []
    for k in range(n_islands):
        island_leader = max(islands_pop[k], key=lambda x: x[1])
        isl_div = compute_population_diversity(islands_pop[k], max_conv, max_full)
        islands_summary.append({
            'island_id': k + 1,
            'best_fitness': round(island_leader[1], 4),
            'best_accuracy': round(island_leader[2], 2),
            'best_params': island_leader[3],
            'conv_layers': island_leader[0].n_conv,
            'fc_layers': island_leader[0].n_full,
            'intra_island_diversity': round(isl_div, 4)
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
        'surrogate_training_samples': len(surrogate.train_x),
        'use_amp': use_amp,
        'max_spatial_size': max_spatial_size
    }

    # Guardar automáticamente la arquitectura del mejor modelo de V12
    try:
        from model_utils import save_best_model, calculate_cnn_metrics, compute_classification_metrics, save_experiment_record
        save_best_model(
            variant="v12",
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
            "method": "DeepGA_V12",
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
        print(f"Nota al guardar mejor modelo V12 / reporte: {e}", flush=True)

    return results, all_final_pop, bestind, island_stats


def final_evaluation(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     max_params: int, w: float, device: torch.device, train_epochs: int, loss_func, chck_dir: str,
                     n_channels: int = 3, n_classes: int = 10, out_size: int = 32, variant: str = "v12", auto_download: bool = False):
    from variants.DeepGA import final_evaluation as fe
    return fe(execution, bestind, train_dl, val_dl, lr, max_params, w, device, train_epochs, loss_func, chck_dir, n_channels, n_classes, out_size, variant=variant, auto_download=auto_download)
