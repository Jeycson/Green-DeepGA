# -*- coding: utf-8 -*-
""" Created on 2026
    Variante Multi-Objetivo MO-DeepGA V9 (Green Multi-Objective Neuroevolution):
    - Balance Bi-Objetivo: Maximizar Precisión vs Minimizar Huella de Carbono (gCO2eq / Energía).
    - Asistencia por Meta-Modelo Subrogado Multi-Objetivo (Dual Random Forest + MO-UCB/LCB).
    - Pre-Screening Rápido en CPU: Filtra candidatos en CPU y entrena en GPU solo los Pareto-prometedores.
    - Operador de Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
    - Probabilidad de Mutación Adaptativa Multi-Objetivo (Basada en Rango de Pareto + Crowding Distance).
    - Mutación Especializada Multi-Objetivo (Micro-ajustes para Frente 1 vs Macro-estructural para dominados).
    - Reemplazo Elitista NSGA-II con Ordenamiento No Dominado Rápido y Distancia de Apiñamiento.
    - Métrica de Hipervolumen 2D (HV) y Selección Automática de Modelos: Verde, Máxima Precisión y Equilibrado (Knee Point).
"""

import os
import copy
from copy import deepcopy
import random
import time
import timeit
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from EncodingClass import Encoding
from Decoding import decoding, CNN
from DistributedTraining import training
from Operators_MOV9 import (
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    tournament_selection_mo,
    compute_mo_adaptive_mutation_rate,
    mo_adaptive_mutation_v9,
    crossover_v7,
    calculate_hypervolume_2d
)

# Importación de modelos para el subrogado multi-objetivo
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =====================================================================
# 1. EXTRACCIÓN DE CARACTERÍSTICAS DE GENOMAS (PARA EL SUBROGADO)
# =====================================================================

def extract_genome_features(e, max_conv: int = 5, max_full: int = 4) -> np.ndarray:
    """
    Convierte un genoma de DeepGA (Encoding) en un vector numérico de características
    de tamaño fijo para ser consumido por el meta-modelo subrogado dual.
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


# =====================================================================
# 2. META-MODELO SUBROGADO MULTI-OBJETIVO (DUAL RANDOM FOREST)
# =====================================================================

class MOSurrogatePredictor:
    """
    Meta-modelo subrogado Multi-Objetivo basado en dos Random Forest Regressors:
    1. Modelo de Precisión (Accuracy Predictor): Estima la precisión esperada con incertidumbre UCB.
    2. Modelo de Huella de Carbono (Carbon Predictor): Estima las emisiones gCO2eq esperadas con incertidumbre LCB.
    """
    def __init__(self, n_estimators: int = 50, max_depth: int = 8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_acc = None
        self.model_carb = None
        self.is_trained = False
        self.train_x_history = []
        self.train_y_acc_history = []
        self.train_y_carb_history = []

    def add_sample(self, genome_features: np.ndarray, acc_val: float, carb_val: float):
        self.train_x_history.append(genome_features)
        self.train_y_acc_history.append(acc_val)
        self.train_y_carb_history.append(carb_val)

    def train(self) -> bool:
        """Entrena ambos subrogados con las evaluaciones reales obtenidas en GPU."""
        if not SKLEARN_AVAILABLE or len(self.train_x_history) < 5:
            return False

        X = np.array(self.train_x_history)
        y_acc = np.array(self.train_y_acc_history)
        y_carb = np.array(self.train_y_carb_history)

        n_est = min(self.n_estimators, max(10, len(X)))

        self.model_acc = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_acc.fit(X, y_acc)

        self.model_carb = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=self.max_depth,
            random_state=self.random_state + 1,
            n_jobs=-1
        )
        self.model_carb.fit(X, y_carb)

        self.is_trained = True
        return True

    def predict_accuracy(self, genome_features: np.ndarray) -> tuple:
        """Retorna (mu_acc, sigma_acc)."""
        if not self.is_trained or self.model_acc is None:
            return 50.0, 10.0

        X = genome_features.reshape(1, -1)
        tree_preds = [tree.predict(X)[0] for tree in self.model_acc.estimators_]
        return float(np.mean(tree_preds)), float(np.std(tree_preds))

    def predict_carbon(self, genome_features: np.ndarray) -> tuple:
        """Retorna (mu_carb, sigma_carb)."""
        if not self.is_trained or self.model_carb is None:
            return 5.0, 2.0

        X = genome_features.reshape(1, -1)
        tree_preds = [tree.predict(X)[0] for tree in self.model_carb.estimators_]
        return float(np.mean(tree_preds)), float(np.std(tree_preds))

    def acquisition_scores(self, genome_features: np.ndarray, kappa: float = 0.1) -> tuple:
        """
        Calcula los scores de adquisición optimistas:
        - Acc UCB (Maximizar): mu_acc + kappa * sigma_acc
        - Carb LCB (Minimizar): max(0.01, mu_carb - kappa * sigma_carb)
        """
        mu_acc, sigma_acc = self.predict_accuracy(genome_features)
        mu_carb, sigma_carb = self.predict_carbon(genome_features)

        acq_acc = mu_acc + (kappa * sigma_acc)
        acq_carb = max(0.001, mu_carb - (kappa * sigma_carb))
        return acq_acc, acq_carb, mu_acc, mu_carb, sigma_acc, sigma_carb

    def select_top_candidates(self, candidate_pool: list, n_select: int = 2,
                              kappa: float = 0.1, max_conv: int = 5, max_full: int = 4) -> list:
        """
        Pre-evalúa en CPU todo el pool de candidatos usando el subrogado dual.
        Aplica ordenamiento no dominado y distancia de apiñamiento sobre las predicciones
        para retornar los `n_select` candidatos más prometedores en el frente de Pareto predicho.
        """
        if not self.is_trained or len(candidate_pool) <= n_select:
            return [{'genome': c, 'pred_acc': 50.0, 'pred_carb': 5.0, 'features': extract_genome_features(c, max_conv, max_full)} for c in candidate_pool[:n_select]]

        scored_pool = []
        for cand in candidate_pool:
            feats = extract_genome_features(cand, max_conv, max_full)
            acq_acc, acq_carb, mu_acc, mu_carb, sig_acc, sig_carb = self.acquisition_scores(feats, kappa)
            scored_pool.append({
                'genome': cand,
                'acq_acc': acq_acc,
                'acq_carb': acq_carb,
                'pred_acc': mu_acc,
                'pred_carb': mu_carb,
                'sigma_acc': sig_acc,
                'sigma_carb': sig_carb,
                'features': feats
            })

        # Estructurar para ordenamiento no dominado en espacio subrogado
        # ind_format = [item, acq_acc (max), acq_carb (min), metrics_dict]
        proxy_pop = [[item, item['acq_acc'], item['acq_carb'], {'pareto_rank': 1, 'crowding_dist': 0.0}] for item in scored_pool]
        fronts = fast_non_dominated_sort(proxy_pop)

        selected_items = []
        for front in fronts:
            front = calculate_crowding_distance(front)
            # Ordenar frente por crowding distance descendente
            front.sort(key=lambda x: x[3].get('crowding_dist', 0.0), reverse=True)
            for proxy_ind in front:
                selected_items.append(proxy_ind[0])
                if len(selected_items) >= n_select:
                    break
            if len(selected_items) >= n_select:
                break

        return selected_items[:n_select]


# =====================================================================
# 3. CÁLCULO DE HUELLA DE CARBONO Y ENERGÍA POR INDIVIDUO (GREEN AI)
# =====================================================================

def calculate_individual_carbon_and_energy(model: nn.Module, genome,
                                          train_time_sec: float,
                                          num_epochs: int,
                                          num_samples: int,
                                          in_channels: int = 3,
                                          out_size: int = 32,
                                          country_iso_code: str = "MEX",
                                          gpu_tdp_watts: float = 150.0) -> dict:
    """
    Calcula de manera exhaustiva el impacto ambiental y computacional de una arquitectura CNN:
    1. Parámetros totales y entrenables.
    2. FLOPs / MACs por imagen de inferencia.
    3. FLOPs totales de entrenamiento (forward + backward).
    4. Consumo energético real estimado (kWh).
    5. Huella de carbono en gramos de CO2 equivalente (gCO2eq).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4.0) / (1024.0 * 1024.0)

    # Estimación de MACs / FLOPs forward
    macs_est = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            # Aproximación del tamaño de mapa espacial (out_size / 2 promedio)
            spatial_factor = max(4, out_size // 2)
            macs_est += m.out_channels * m.in_channels * kh * kw * spatial_factor * spatial_factor
        elif isinstance(m, nn.Linear):
            macs_est += m.in_features * m.out_features

    flops_per_sample = macs_est * 2
    # Entrenamiento completo: ~3x FLOPs por muestra (1 forward + 2 backward) * muestras * épocas
    total_train_flops = 3 * flops_per_sample * num_samples * max(1, num_epochs)

    # 1. Componente temporal de energía (GPU Power en kW * horas de entrenamiento)
    power_kw = gpu_tdp_watts / 1000.0
    energy_time_kwh = power_kw * (train_time_sec / 3600.0)

    # 2. Componente computacional Green AI (PUE 1.2, eficiencia de hardware 10 TFLOPs/W)
    pue = 1.2
    hardware_efficiency_tflops_w = 10.0
    energy_comp_kwh = (total_train_flops / (1e12 * hardware_efficiency_tflops_w * 3600.0 * 1000.0)) * pue

    # Energía combinada calibrada
    total_energy_kwh = max(1e-7, (energy_time_kwh * 0.7) + (energy_comp_kwh * 0.3))

    # Factor de emisión de carbono por país (gCO2eq / kWh)
    carbon_intensity_table = {
        "MEX": 430.0,
        "USA": 380.0,
        "COL": 190.0,
        "ESP": 160.0,
        "CHL": 350.0,
        "ARG": 340.0,
        "FRA": 55.0,
        "DEU": 350.0
    }
    carbon_factor = carbon_intensity_table.get(country_iso_code.upper(), 430.0)
    carbon_gco2 = total_energy_kwh * carbon_factor

    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    fc_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    skip_count = sum(1 for bit in genome.second_level if bit == 1) if hasattr(genome, "second_level") else 0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 4),
        "flops_per_sample": flops_per_sample,
        "total_train_flops": total_train_flops,
        "train_time_sec": round(train_time_sec, 3),
        "energy_kwh": total_energy_kwh,
        "carbon_gco2": round(carbon_gco2, 5),
        "conv_count": conv_count,
        "fc_count": fc_count,
        "skip_count": skip_count,
        "pareto_rank": 1,
        "crowding_dist": 0.0
    }


def _evaluate_individual_mo(e, in_channels: int, out_size: int, n_classes: int,
                            device: torch.device, num_epochs: int, loss_func,
                            train_dl: DataLoader, val_dl: DataLoader, lr: float,
                            max_params: int, country_iso_code: str = "MEX"):
    """
    Decodifica y entrena secuencialmente una CNN en GPU para el enfoque multi-objetivo.
    Retorna:
    - accuracy (float): Precisión en validación [0, 100]% (Objetivo 1 -> Maximizar).
    - carbon_gco2 (float): Huella de carbono en gCO2eq (Objetivo 2 -> Minimizar).
    - metrics_dict (dict): Detalles de parámetros, FLOPs, energía y tiempo.
    """
    network = decoding(e, in_channels, out_size, n_classes)
    cnn = CNN(e, network[0], network[1], network[2])

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    acc_list = []
    # Entrenamiento con w=0.0 para evaluar accuracy pura
    _, accuracy, params, trained_model = training(
        '1', device, cnn, num_epochs, loss_func,
        train_dl, val_dl, lr, 0.0, max_params, acc_list
    )

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time_sec = time.perf_counter() - t_start

    num_samples = len(train_dl.dataset) if hasattr(train_dl, 'dataset') else len(train_dl) * 64

    metrics_dict = calculate_individual_carbon_and_energy(
        model=trained_model,
        genome=e,
        train_time_sec=train_time_sec,
        num_epochs=num_epochs,
        num_samples=num_samples,
        in_channels=in_channels,
        out_size=out_size,
        country_iso_code=country_iso_code
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    carbon_gco2 = metrics_dict["carbon_gco2"]
    return accuracy, carbon_gco2, metrics_dict


# =====================================================================
# 4. ALGORITMO EVOLUTIVO MULTI-OBJETIVO MO-DEEPGA V9
# =====================================================================

def green_MODeepGA_v9(execution: int, memoryC: bool, train_epochs: int,
                      train_dl: DataLoader, val_dl: DataLoader, lr: float,
                      min_conv: int, max_conv: int, min_full: int, max_full: int,
                      max_params: int, cr: float, mr: float,
                      N: int, T: int, t_size: int, w: float,
                      device: torch.device, chck_dir: str,
                      n_channels: int = 3, n_classes: int = 10, out_size: int = 32,
                      loss_func=None, pool_candidates_factor: int = 5,
                      kappa: float = 0.1, mr_min: float = 0.10, mr_max: float = 0.85,
                      country_iso_code: str = "MEX", **kwargs):
    """
    Algoritmo Multi-Objetivo MO-DeepGA V9:
    - Espacio Bi-Objetivo: Precisión (↑ Max) vs Huella de Carbono gCO2eq (↓ Min).
    - Meta-Modelo Subrogado Multi-Objetivo (Dual Random Forest + MO-UCB/LCB).
    - Pre-Screening en CPU de Candidatos y Evaluación Selectiva en GPU.
    - Cruce Topológico Coherente V7 y Mutación Adaptativa según Rango de Pareto.
    - Reemplazo No Dominado NSGA-II con Elitismo y Distancia de Apiñamiento.
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):
        os.makedirs(chck_dir)

    surrogate = MOSurrogatePredictor()

    '''Inicialización de Población Multi-Objetivo'''
    chkpoint_obj = Path(chck_dir + str(execution) + "_mo_v9_checkpoint.pkl")
    if chkpoint_obj.exists():
        print("Re-Initialize population (MO-DeepGA V9 - Multi-Objective Surrogate + NSGA-II)")
        with open(chck_dir + str(execution) + "_mo_v9_checkpoint.pkl", "rb") as p:
            values = pickle.load(p)
        start = timeit.default_timer() - values['time']
        pop = values['pop']
        pareto_front = values.get('pareto_front', [])
        hv_history = values.get('hv_history', [])
        bestAcc_history = values.get('bestAcc_history', [])
        minCarb_history = values.get('minCarb_history', [])
        knee_history = values.get('knee_history', [])
        t = values['t']
        evals = values['evals']
        cacheM = values['cacheM']
        total_screened_candidates = values.get('total_screened_candidates', 0)
        prediction_errors_acc = values.get('prediction_errors_acc', [])
        prediction_errors_carb = values.get('prediction_errors_carb', [])
        adaptive_mr_history = values.get('adaptive_mr_history', [])
        mutation_events = values.get('mutation_events', {'modify_layer': 0, 'flip_skip': 0, 'add_remove_layer': 0})
        evaluated_history = values.get('evaluated_history', [])

        # Re-entrenar el subrogado con las muestras guardadas
        for ind_e, acc_v, carb_v in evaluated_history:
            feats = extract_genome_features(ind_e, max_conv, max_full)
            surrogate.add_sample(feats, acc_v, carb_v)
        surrogate.train()
    else:
        print('Initialize population (MO-DeepGA V9 - Bi-Objective: Precision vs Carbon Footprint)')
        start = timeit.default_timer()
        pop = []
        pareto_front = []
        hv_history = []
        bestAcc_history = []
        minCarb_history = []
        knee_history = []
        t = 0
        evals = 0
        total_screened_candidates = 0
        prediction_errors_acc = []
        prediction_errors_carb = []
        adaptive_mr_history = []
        mutation_events = {'modify_layer': 0, 'flip_skip': 0, 'add_remove_layer': 0}
        cacheM = {}
        evaluated_history = []

        # Generación 0: Inicialización y evaluación base en GPU
        while len(pop) < N:
            e1 = Encoding(min_conv, max_conv, min_full, max_full)
            if memoryC:
                strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
                if strIDe1 in cacheM:
                    acc1, carb1, metrics1 = cacheM[strIDe1]
                    print(f"[Cache Init MO] Acc: {acc1:.2f}%, Carbon: {carb1:.4f} gCO2eq, Params: {metrics1['total_params']}")
                    pop.append([e1, acc1, carb1, metrics1])
                    evaluated_history.append((e1, acc1, carb1))
                else:
                    acc1, carb1, metrics1 = _evaluate_individual_mo(
                        e1, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, max_params, country_iso_code
                    )
                    cacheM[strIDe1] = [acc1, carb1, metrics1]
                    evals += 1
                    evaluated_history.append((e1, acc1, carb1))
                    print(f"[GPU Init MO-V9] Acc: {acc1:.2f}%, Carbon: {carb1:.4f} gCO2eq, Params: {metrics1['total_params']}, Time: {metrics1['train_time_sec']}s")
                    pop.append([e1, acc1, carb1, metrics1])
            else:
                acc1, carb1, metrics1 = _evaluate_individual_mo(
                    e1, n_channels, out_size, n_classes, device1,
                    num_epochs, loss_func, train_dl, val_dl, lr, max_params, country_iso_code
                )
                evals += 1
                evaluated_history.append((e1, acc1, carb1))
                print(f"[GPU Init MO-V9] Acc: {acc1:.2f}%, Carbon: {carb1:.4f} gCO2eq, Params: {metrics1['total_params']}, Time: {metrics1['train_time_sec']}s")
                pop.append([e1, acc1, carb1, metrics1])

            # Alimentar dataset del subrogado
            feats = extract_genome_features(e1, max_conv, max_full)
            surrogate.add_sample(feats, acc1, carb1)

        # Entrenar primer modelo subrogado dual
        surrogate.train()
        print(f"✓ Subrogado Multi-Objetivo (Accuracy & Carbon) inicializado con {len(surrogate.train_x_history)} muestras.")

    '''Bucle Evolutivo Multi-Objetivo (NSGA-II + Subrogado V9)'''
    print('--------------------------------------------')
    while t < T:
        print(f'Generation: {t} (MO-DeepGA V9 - Precision vs Carbon Footprint)')

        # 1. Re-entrenar subrogado al inicio de cada generación
        surrogate.train()

        # 2. Ordenamiento no dominado y cálculo de crowding distance de la población actual
        ranked_fronts = fast_non_dominated_sort(pop)
        for front in ranked_fronts:
            calculate_crowding_distance(front)

        max_rank = len(ranked_fronts)
        current_pareto_front = ranked_fronts[0]

        # 3. Selección de Padres por Torneo Multi-Objetivo
        parents = []
        while len(parents) < int(N / 2):
            p1 = tournament_selection_mo(pop, t_size)
            p2 = tournament_selection_mo(pop, t_size)
            while p1 == p2:
                p2 = tournament_selection_mo(pop, t_size)
            parents.append(p1)
            parents.append(p2)

        # 4. Reproducción con Cruce V7, Mutación Adaptativa MO y Pre-Screening por Subrogado
        offspring = []
        iter_parents = 0
        candidates_per_pair = max(4, pool_candidates_factor * 2)
        gen_applied_mrs = []

        while len(offspring) < int(N / 2):
            p1_obj = parents[iter_parents]
            p2_obj = parents[iter_parents + 1]
            p1_genome, p1_acc, p1_carb, p1_info = p1_obj[0], p1_obj[1], p1_obj[2], p1_obj[3]
            p2_genome, p2_acc, p2_carb, p2_info = p2_obj[0], p2_obj[1], p2_obj[2], p2_obj[3]

            r1, cd1 = p1_info.get('pareto_rank', 1), p1_info.get('crowding_dist', 0.0)
            r2, cd2 = p2_info.get('pareto_rank', 1), p2_info.get('crowding_dist', 0.0)

            # Calcular tasas de mutación adaptativa personalizadas por rango de Pareto
            mr_adapt1 = compute_mo_adaptive_mutation_rate(r1, max_rank, cd1, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            mr_adapt2 = compute_mo_adaptive_mutation_rate(r2, max_rank, cd2, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            gen_applied_mrs.extend([mr_adapt1, mr_adapt2])

            # Generar pool de candidatos en CPU
            candidate_pool = []
            for _ in range(candidates_per_pair // 2):
                if cr >= random.uniform(0, 1):
                    c1, c2 = crossover_v7(p1_genome, p2_genome, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                else:
                    c1 = deepcopy(p1_genome)
                    c2 = deepcopy(p2_genome)

                mut1_applied, mut1_type = mo_adaptive_mutation_v9(c1, mr_adapt1, rank=r1, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                if mut1_applied and mut1_type in mutation_events:
                    mutation_events[mut1_type] += 1

                mut2_applied, mut2_type = mo_adaptive_mutation_v9(c2, mr_adapt2, rank=r2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                if mut2_applied and mut2_type in mutation_events:
                    mutation_events[mut2_type] += 1

                candidate_pool.extend([c1, c2])

            total_screened_candidates += len(candidate_pool)

            # Pre-screening en CPU: seleccionar los 2 mejores candidatos sobre el frente de Pareto predicho
            top_selected = surrogate.select_top_candidates(
                candidate_pool, n_select=2, kappa=kappa, max_conv=max_conv, max_full=max_full
            )

            print(f" [🧬 MO-V9 Subrogado] Pool CPU: {len(candidate_pool)} cands (mr_p1={mr_adapt1:.2f}, mr_p2={mr_adapt2:.2f}). "
                  f"Top 1 Pred: Acc={top_selected[0]['pred_acc']:.2f}% (±{top_selected[0].get('sigma_acc', 0):.2f}), Carb={top_selected[0]['pred_carb']:.4f}g | "
                  f"Top 2 Pred: Acc={top_selected[1]['pred_acc']:.2f}% (±{top_selected[1].get('sigma_acc', 0):.2f}), Carb={top_selected[1]['pred_carb']:.4f}g")

            # Entrenar en GPU ÚNICAMENTE los 2 mejores candidatos seleccionados
            for item in top_selected:
                cand_genome = item['genome']
                pred_a = item.get('pred_acc', 50.0)
                pred_c = item.get('pred_carb', 5.0)
                feats = item['features']

                if memoryC:
                    strID = str([cand_genome.n_conv, cand_genome.n_full, cand_genome.first_level, cand_genome.second_level])
                    if strID in cacheM:
                        acc, carb, metrics = cacheM[strID]
                        print(f"[Cache Offspring MO] Acc: {acc:.2f}%, Carbon: {carb:.4f} gCO2eq, Params: {metrics['total_params']}")
                    else:
                        acc, carb, metrics = _evaluate_individual_mo(
                            cand_genome, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, max_params, country_iso_code
                        )
                        cacheM[strID] = [acc, carb, metrics]
                        evals += 1
                        evaluated_history.append((cand_genome, acc, carb))
                        print(f"[GPU Offspring MO-V9] Acc Real: {acc:.2f}% (Pred: {pred_a:.2f}%), Carbon Real: {carb:.4f}g (Pred: {pred_c:.4f}g)")
                else:
                    acc, carb, metrics = _evaluate_individual_mo(
                        cand_genome, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, max_params, country_iso_code
                    )
                    evals += 1
                    evaluated_history.append((cand_genome, acc, carb))
                    print(f"[GPU Offspring MO-V9] Acc Real: {acc:.2f}% (Pred: {pred_a:.2f}%), Carbon Real: {carb:.4f}g (Pred: {pred_c:.4f}g)")

                # Registrar errores del subrogado
                prediction_errors_acc.append(abs(acc - pred_a))
                prediction_errors_carb.append(abs(carb - pred_c))
                surrogate.add_sample(feats, acc, carb)

                offspring.append([cand_genome, acc, carb, metrics])

            iter_parents += 2

        # 5. Reemplazo Elitista NSGA-II (Población Actual + Descendencia -> Top N)
        combined_pop = pop + offspring
        combined_fronts = fast_non_dominated_sort(combined_pop)

        new_pop = []
        for front in combined_fronts:
            calculate_crowding_distance(front)
            if len(new_pop) + len(front) <= N:
                new_pop.extend(front)
            else:
                # Ordenar el último frente por crowding distance descendente
                front.sort(key=lambda x: x[3].get('crowding_dist', 0.0), reverse=True)
                needed = N - len(new_pop)
                new_pop.extend(front[:needed])
                break

        pop = new_pop

        # 6. Actualizar métricas del Frente de Pareto de la Generación
        gen_fronts = fast_non_dominated_sort(pop)
        pareto_front = gen_fronts[0]

        # Calcular Hipervolumen 2D
        # Punto de referencia: (0% Accuracy, 50.0 gCO2eq)
        max_carbon_seen = max([ind[2] for ind in pop] + [10.0])
        ref_point = (0.0, max(20.0, max_carbon_seen * 1.2))
        hv = calculate_hypervolume_2d(pareto_front, ref_point=ref_point)
        hv_history.append(hv)

        # Mejor precisión absoluta
        best_acc_ind = max(pareto_front, key=lambda x: x[1])
        bestAcc_history.append(best_acc_ind[1])

        # Modelo más verde (menor huella de carbono)
        greenest_ind = min(pareto_front, key=lambda x: x[2])
        minCarb_history.append(greenest_ind[2])

        # Solución de compromiso equilibrada (Knee Point)
        # Normalizamos Acc y Carbon para encontrar la distancia mínima al punto ideal utópico (100%, 0g)
        all_accs = [p[1] for p in pareto_front]
        all_carbs = [p[2] for p in pareto_front]
        min_a, max_a = min(all_accs), max(all_accs)
        min_c, max_c = min(all_carbs), max(all_carbs)
        range_a = max(1e-5, max_a - min_a)
        range_c = max(1e-5, max_c - min_c)

        knee_ind = pareto_front[0]
        min_dist = float('inf')
        for ind in pareto_front:
            norm_acc_gap = (max_a - ind[1]) / range_a
            norm_carb_gap = (ind[2] - min_c) / range_c
            dist_to_ideal = math.sqrt(norm_acc_gap**2 + norm_carb_gap**2)
            if dist_to_ideal < min_dist:
                min_dist = dist_to_ideal
                knee_ind = ind

        knee_history.append((knee_ind[1], knee_ind[2]))

        mean_gen_mr = float(np.mean(gen_applied_mrs)) if gen_applied_mrs else mr
        adaptive_mr_history.append(mean_gen_mr)

        t += 1

        # Checkpoint multi-objetivo
        time_elapsed = timeit.default_timer() - start
        current_state = dict(
            pop=pop, pareto_front=pareto_front, hv_history=hv_history,
            bestAcc_history=bestAcc_history, minCarb_history=minCarb_history,
            knee_history=knee_history, t=t, evals=evals,
            total_screened_candidates=total_screened_candidates,
            prediction_errors_acc=prediction_errors_acc,
            prediction_errors_carb=prediction_errors_carb,
            adaptive_mr_history=adaptive_mr_history,
            mutation_events=mutation_events,
            evaluated_history=evaluated_history,
            time=time_elapsed, cacheM=cacheM
        )
        with open(chck_dir + str(execution) + "_mo_v9_checkpoint.pkl", "wb") as p:
            pickle.dump(current_state, p)

        mae_acc = np.mean(prediction_errors_acc[-10:]) if prediction_errors_acc else 0.0
        mae_carb = np.mean(prediction_errors_carb[-10:]) if prediction_errors_carb else 0.0

        print(f"--- Fin Gen {t-1} | Frente Pareto (F1): {len(pareto_front)} redes | HV: {hv:.2f} | mr_adapt: {mean_gen_mr:.3f} | GPU Evals: {evals} ---")
        print(f"  🏆 Mejor Precisión:        Acc={best_acc_ind[1]:.2f}%, Carbon={best_acc_ind[2]:.4f} gCO2eq, Params={best_acc_ind[3]['total_params']}")
        print(f"  🌿 Más Ecológica (Green):   Acc={greenest_ind[1]:.2f}%, Carbon={greenest_ind[2]:.4f} gCO2eq, Params={greenest_ind[3]['total_params']}")
        print(f"  ⚖️ Equilibrada (Knee):      Acc={knee_ind[1]:.2f}%, Carbon={knee_ind[2]:.4f} gCO2eq, Params={knee_ind[3]['total_params']}")
        print(f"  🔍 MAE Subrogado:           Acc={mae_acc:.2f}%, Carbon={mae_carb:.4f} gCO2eq")
        print('--------------------------------------------')

    # Dataframe de evolución multi-objetivo
    results_df = pd.DataFrame({
        'Generation': list(range(len(hv_history))),
        'Hypervolume': hv_history,
        'BestAccuracy': bestAcc_history,
        'MinCarbon_gCO2': minCarb_history,
        'KneeAccuracy': [k[0] for k in knee_history],
        'KneeCarbon_gCO2': [k[1] for k in knee_history],
        'MeanAdaptiveMR': adaptive_mr_history
    })

    # Estadísticas consolidadas
    surrogate_stats = {
        'total_cpu_screened': total_screened_candidates,
        'total_gpu_evaluations': evals,
        'exploration_multiplier': round(total_screened_candidates / max(1, evals), 2),
        'mean_absolute_error_acc': round(float(np.mean(prediction_errors_acc)), 4) if prediction_errors_acc else 0.0,
        'mean_absolute_error_carb': round(float(np.mean(prediction_errors_carb)), 4) if prediction_errors_carb else 0.0,
        'surrogate_training_samples': len(surrogate.train_x_history),
        'adaptive_mr_history': adaptive_mr_history,
        'mutation_events': mutation_events
    }

    mo_stats = {
        'pareto_front_size': len(pareto_front),
        'final_hypervolume': hv_history[-1] if hv_history else 0.0,
        'best_accuracy_individual': best_acc_ind,
        'greenest_individual': greenest_ind,
        'knee_point_individual': knee_ind,
        'all_pareto_solutions': pareto_front
    }

    # Guardar automáticamente los 3 modelos representativos del Frente de Pareto
    try:
        from model_utils import save_best_model
        save_best_model(variant="mo_v9_best_acc", execution=execution, bestind=best_acc_ind, in_channels=n_channels, out_size=out_size, n_classes=n_classes, chck_dir=chck_dir)
        save_best_model(variant="mo_v9_greenest", execution=execution, bestind=greenest_ind, in_channels=n_channels, out_size=out_size, n_classes=n_classes, chck_dir=chck_dir)
        save_best_model(variant="mo_v9_knee", execution=execution, bestind=knee_ind, in_channels=n_channels, out_size=out_size, n_classes=n_classes, chck_dir=chck_dir)
    except Exception as e:
        print(f"Nota al guardar modelos del Frente de Pareto: {e}")

    # Retornamos results_df, población completa, individuo knee (o frente completo), surrogate_stats y mo_stats
    return results_df, pareto_front, knee_ind, surrogate_stats, mo_stats


def final_evaluation_mo(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader,
                        lr: float, max_params: int, device: torch.device, train_epochs: int,
                        loss_func, chck_dir: str, n_channels: int = 3, n_classes: int = 10,
                        out_size: int = 32, variant: str = "mo_v9", auto_download: bool = False):
    """Re-entrenamiento final para un modelo seleccionado del Frente de Pareto."""
    from variants.DeepGA import final_evaluation as fe
    return fe(execution, bestind, train_dl, val_dl, lr, max_params, 0.0, device, train_epochs, loss_func, chck_dir, n_channels, n_classes, out_size, variant=variant, auto_download=auto_download)
