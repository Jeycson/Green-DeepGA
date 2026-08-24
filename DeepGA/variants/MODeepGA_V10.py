# -*- coding: utf-8 -*-
""" Created on 2026
    Variante Multi-Objetivo MO-DeepGA V10 (ACO-Enhanced Multi-Objective DeepGA):
    - Balance Bi-Objetivo: Maximizar Precisión vs Minimizar Huella de Carbono (gCO2eq / Energía).
    - Matriz de Feromonas Arquitectónicas Multi-Objetivo (MO-ACO Pheromone Trail Matrix).
    - Depósito de Feromonas Proporcional al Rango de Pareto (F1 Élites No Dominadas + Crowding Distance).
    - Mutación e Inicialización Guiadas por Distribución de Feromonas Pareto.
    - Asistencia por Meta-Modelo Subrogado Multi-Objetivo (Dual Random Forest + MO-UCB/LCB en CPU).
    - Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
    - Reemplazo Elitista NSGA-II con Ordenamiento No Dominado Rápido y Distancia de Apiñamiento.
    - Métrica de Hipervolumen 2D (HV) y Extracción de Motivos Arquitectónicos Favorecidos por ACO.
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

from deepga.core.encoding import Encoding
from deepga.core.decoding import decoding, CNN
from deepga.training.engine import training
from deepga.evolution.operators_mo_v10 import (
    MOPheromoneMatrix,
    mo_guided_adaptive_mutation_v10,
    compute_mo_adaptive_mutation_rate,
    crossover_v7,
    dominates,
    fast_non_dominated_sort,
    calculate_crowding_distance,
    tournament_selection_mo,
    calculate_hypervolume_2d
)
from variants.MODeepGA_V9 import (
    extract_genome_features,
    MOSurrogatePredictor,
    calculate_individual_carbon_and_energy,
    _evaluate_individual_mo
)


def green_MODeepGA_v10(execution: int, memoryC: bool, train_epochs: int,
                       train_dl: DataLoader, val_dl: DataLoader, lr: float,
                       min_conv: int, max_conv: int, min_full: int, max_full: int,
                       max_params: int, cr: float, mr: float,
                       N: int, T: int, t_size: int, w: float,
                       device: torch.device, chck_dir: str,
                       n_channels: int = 3, n_classes: int = 10, out_size: int = 32,
                       loss_func=None, pool_candidates_factor: int = 5,
                       kappa: float = 0.1, mr_min: float = 0.10, mr_max: float = 0.85,
                       rho: float = 0.10, alpha: float = 1.2, top_k_ratio: float = 0.35,
                       country_iso_code: str = "MEX", **kwargs):
    """
    Algoritmo Multi-Objetivo MO-DeepGA V10:
    - Espacio Bi-Objetivo: Precisión (↑ Max) vs Huella de Carbono gCO2eq (↓ Min).
    - Guía Heurística de Feromonas ACO Multi-Objetivo sobre el Frente de Pareto.
    - Meta-Modelo Subrogado Multi-Objetivo (Dual Random Forest + MO-UCB/LCB).
    - Pre-Screening en CPU de Candidatos y Selección No Dominada.
    - Cruce Coherente V7 y Mutación Guiada por Feromonas Pareto.
    - Reemplazo NSGA-II con Elitismo y Distancia de Apiñamiento.
    """
    num_epochs = train_epochs
    device1 = device

    if not os.path.exists(chck_dir):
        os.makedirs(chck_dir)

    surrogate = MOSurrogatePredictor()
    pheromones = MOPheromoneMatrix(
        min_conv=min_conv, max_conv=max_conv,
        min_full=min_full, max_full=max_full,
        rho=rho, alpha=alpha
    )

    '''Inicialización de Población Multi-Objetivo'''
    chkpoint_obj = Path(chck_dir + str(execution) + "_mo_v10_checkpoint.pkl")
    if chkpoint_obj.exists():
        print("Re-Initialize population (MO-DeepGA V10 - ACO Pheromones + Multi-Objective Surrogate + NSGA-II)", flush=True)
        with open(chck_dir + str(execution) + "_mo_v10_checkpoint.pkl", "rb") as p:
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

        # Re-entrenar el subrogado y depositar feromonas
        for ind_e, acc_v, carb_v in evaluated_history:
            feats = extract_genome_features(ind_e, max_conv, max_full)
            surrogate.add_sample(feats, acc_v, carb_v)
        surrogate.train()
        ranked_f = fast_non_dominated_sort(pop)
        pheromones.deposit_pareto(ranked_f, top_k_ratio=top_k_ratio)
    else:
        print('Initialize population (MO-DeepGA V10 - ACO-Guided Bi-Objective Search)', flush=True)
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

        # Generación 0: Inicialización guiada por feromonas base y evaluación en GPU
        while len(pop) < N:
            e1 = pheromones.generate_guided_encoding(min_conv, max_conv, min_full, max_full, alpha=1.0)
            if memoryC:
                strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
                if strIDe1 in cacheM:
                    acc1, carb1, metrics1 = cacheM[strIDe1]
                    print(f"[Cache Init MO-V10] Acc: {acc1:.2f}%, Carbon: {carb1:.4f} gCO2eq, Params: {metrics1['total_params']}")
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
                    print(f"[GPU Init MO-V10] Ind {len(pop)+1}/{N} -> Acc: {acc1:.2f}%, Carbon: {carb1:.4f} gCO2eq, Params: {metrics1['total_params']:,}, Time: {metrics1['train_time_sec']}s")
                    pop.append([e1, acc1, carb1, metrics1])
            else:
                acc1, carb1, metrics1 = _evaluate_individual_mo(
                    e1, n_channels, out_size, n_classes, device1,
                    num_epochs, loss_func, train_dl, val_dl, lr, max_params, country_iso_code
                )
                evals += 1
                evaluated_history.append((e1, acc1, carb1))
                print(f"[GPU Init MO-V10] Ind {len(pop)+1}/{N} -> Acc: {acc1:.2f}%, Carbon: {carb1:.4f} gCO2eq, Params: {metrics1['total_params']:,}, Time: {metrics1['train_time_sec']}s")
                pop.append([e1, acc1, carb1, metrics1])

            # Alimentar dataset del subrogado
            feats = extract_genome_features(e1, max_conv, max_full)
            surrogate.add_sample(feats, acc1, carb1)

        # Entrenar primer modelo subrogado dual y depositar feromonas iniciales
        surrogate.train()
        ranked_f = fast_non_dominated_sort(pop)
        pheromones.deposit_pareto(ranked_f, top_k_ratio=top_k_ratio)
        print(f"✓ Subrogado Multi-Objetivo y Matriz de Feromonas inicializados con {len(surrogate.train_x_history)} muestras.", flush=True)

    '''Bucle Evolutivo Multi-Objetivo Guiado por Feromonas (NSGA-II + ACO + Subrogado V10)'''
    print('--------------------------------------------', flush=True)
    while t < T:
        print(f'Generation: {t} (MO-DeepGA V10 - ACO Pheromones + NSGA-II + Surrogate)', flush=True)

        # 1. Evaporación y Actualización de Feromonas sobre el Frente de Pareto
        pheromones.evaporate()
        ranked_fronts = fast_non_dominated_sort(pop)
        for front in ranked_fronts:
            calculate_crowding_distance(front)
        pheromones.deposit_pareto(ranked_fronts, top_k_ratio=top_k_ratio)

        # 2. Re-entrenar subrogado al inicio de cada generación
        surrogate.train()

        max_rank = len(ranked_fronts)
        current_pareto_front = ranked_fronts[0]

        # 3. Selección de Padres por Torneo Multi-Objetivo (NSGA-II)
        parents = []
        while len(parents) < int(N / 2):
            p1 = tournament_selection_mo(pop, t_size)
            p2 = tournament_selection_mo(pop, t_size)
            while p1 == p2:
                p2 = tournament_selection_mo(pop, t_size)
            parents.append(p1)
            parents.append(p2)

        # 4. Reproducción: Cruce V7 + Mutación Guiada por Feromonas Pareto + Pre-Screening por Subrogado
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

            mr_adapt1 = compute_mo_adaptive_mutation_rate(r1, max_rank, cd1, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            mr_adapt2 = compute_mo_adaptive_mutation_rate(r2, max_rank, cd2, t, T, mr_base=mr, mr_min=mr_min, mr_max=mr_max)
            gen_applied_mrs.extend([mr_adapt1, mr_adapt2])

            # Generar pool de candidatos en CPU con Mutación Guiada por Feromonas
            candidate_pool = []
            for _ in range(candidates_per_pair // 2):
                if cr >= random.uniform(0, 1):
                    c1, c2 = crossover_v7(p1_genome, p2_genome, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                else:
                    c1 = deepcopy(p1_genome)
                    c2 = deepcopy(p2_genome)

                mut1_applied, mut1_type = mo_guided_adaptive_mutation_v10(c1, pheromones, mr_adapt1, rank=r1, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                if mut1_applied and mut1_type in mutation_events:
                    mutation_events[mut1_type] += 1

                mut2_applied, mut2_type = mo_guided_adaptive_mutation_v10(c2, pheromones, mr_adapt2, rank=r2, min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full)
                if mut2_applied and mut2_type in mutation_events:
                    mutation_events[mut2_type] += 1

                candidate_pool.extend([c1, c2])

            total_screened_candidates += len(candidate_pool)

            # Pre-screening en CPU: seleccionar los 2 mejores candidatos sobre el frente de Pareto predicho
            top_selected = surrogate.select_top_candidates(
                candidate_pool, n_select=2, kappa=kappa, max_conv=max_conv, max_full=max_full
            )

            print(f" [🐜 MO-V10 ACO+Subrogado] Pool: {len(candidate_pool)} cands (mr_p1={mr_adapt1:.2f}, mr_p2={mr_adapt2:.2f}). "
                  f"Top 1 Pred: Acc={top_selected[0]['pred_acc']:.2f}%, Carb={top_selected[0]['pred_carb']:.4f}g | "
                  f"Top 2 Pred: Acc={top_selected[1]['pred_acc']:.2f}%, Carb={top_selected[1]['pred_carb']:.4f}g")

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
                        print(f"[Cache Offspring MO-V10] Acc: {acc:.2f}%, Carbon: {carb:.4f} gCO2eq, Params: {metrics['total_params']}")
                    else:
                        acc, carb, metrics = _evaluate_individual_mo(
                            cand_genome, n_channels, out_size, n_classes, device1,
                            num_epochs, loss_func, train_dl, val_dl, lr, max_params, country_iso_code
                        )
                        cacheM[strID] = [acc, carb, metrics]
                        evals += 1
                        evaluated_history.append((cand_genome, acc, carb))
                        print(f"[GPU Offspring MO-V10] Acc Real: {acc:.2f}% (Pred: {pred_a:.2f}%), Carbon Real: {carb:.4f}g (Pred: {pred_c:.4f}g)")
                else:
                    acc, carb, metrics = _evaluate_individual_mo(
                        cand_genome, n_channels, out_size, n_classes, device1,
                        num_epochs, loss_func, train_dl, val_dl, lr, max_params, country_iso_code
                    )
                    evals += 1
                    evaluated_history.append((cand_genome, acc, carb))
                    print(f"[GPU Offspring MO-V10] Acc Real: {acc:.2f}% (Pred: {pred_a:.2f}%), Carbon Real: {carb:.4f}g (Pred: {pred_c:.4f}g)")

                # Registrar errores del subrogado
                prediction_errors_acc.append(abs(acc - pred_a))
                prediction_errors_carb.append(abs(carb - pred_c))
                surrogate.add_sample(feats, acc, carb)

                offspring.append([cand_genome, acc, carb, metrics])

            iter_parents += 2

        # 5. Reemplazo Elitista NSGA-II
        combined_pop = pop + offspring
        combined_fronts = fast_non_dominated_sort(combined_pop)

        new_pop = []
        for front in combined_fronts:
            calculate_crowding_distance(front)
            if len(new_pop) + len(front) <= N:
                new_pop.extend(front)
            else:
                front.sort(key=lambda x: x[3].get('crowding_dist', 0.0), reverse=True)
                needed = N - len(new_pop)
                new_pop.extend(front[:needed])
                break

        pop = new_pop

        # 6. Actualizar métricas del Frente de Pareto de la Generación
        gen_fronts = fast_non_dominated_sort(pop)
        pareto_front = gen_fronts[0]

        # Calcular Hipervolumen 2D
        max_carbon_seen = max([ind[2] for ind in pop] + [10.0])
        ref_point = (0.0, max(20.0, max_carbon_seen * 1.2))
        hv = calculate_hypervolume_2d(pareto_front, ref_point=ref_point)
        hv_history.append(hv)

        best_acc_ind = max(pareto_front, key=lambda x: x[1])
        bestAcc_history.append(best_acc_ind[1])

        greenest_ind = min(pareto_front, key=lambda x: x[2])
        minCarb_history.append(greenest_ind[2])

        # Knee Point
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

        # Checkpoint multi-objetivo V10
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
        with open(chck_dir + str(execution) + "_mo_v10_checkpoint.pkl", "wb") as p:
            pickle.dump(current_state, p)

        mae_acc = np.mean(prediction_errors_acc[-10:]) if prediction_errors_acc else 0.0
        mae_carb = np.mean(prediction_errors_carb[-10:]) if prediction_errors_carb else 0.0
        p_motifs = pheromones.get_top_motifs_summary()

        print(f"--- Fin Gen {t-1} | Frente Pareto (F1): {len(pareto_front)} redes | HV: {hv:.2f} | mr_adapt: {mean_gen_mr:.3f} | GPU Evals: {evals} ---")
        print(f"  🏆 Mejor Precisión:        Acc={best_acc_ind[1]:.2f}%, Carbon={best_acc_ind[2]:.4f} gCO2eq, Params={best_acc_ind[3]['total_params']}")
        print(f"  🌿 Más Ecológica (Green):   Acc={greenest_ind[1]:.2f}%, Carbon={greenest_ind[2]:.4f} gCO2eq, Params={greenest_ind[3]['total_params']}")
        print(f"  ⚖️ Equilibrada (Knee):      Acc={knee_ind[1]:.2f}%, Carbon={knee_ind[2]:.4f} gCO2eq, Params={knee_ind[3]['total_params']}")
        print(f"  🐜 Feromonas Favorecen:     Conv={p_motifs.get('favored_conv_count')}, FC={p_motifs.get('favored_fc_count')}, Skips Activos={len(p_motifs.get('reinforced_skip_connections', []))}")
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

    surrogate_stats = {
        'total_cpu_screened': total_screened_candidates,
        'total_gpu_evaluations': evals,
        'exploration_multiplier': round(total_screened_candidates / max(1, evals), 2),
        'mean_absolute_error_acc': round(float(np.mean(prediction_errors_acc)), 4) if prediction_errors_acc else 0.0,
        'mean_absolute_error_carb': round(float(np.mean(prediction_errors_carb)), 4) if prediction_errors_carb else 0.0,
        'surrogate_training_samples': len(surrogate.train_x_history),
        'adaptive_mr_history': adaptive_mr_history,
        'mutation_events': mutation_events,
        'pheromone_motifs': pheromones.get_top_motifs_summary()
    }

    mo_stats = {
        'pareto_front_size': len(pareto_front),
        'final_hypervolume': hv_history[-1] if hv_history else 0.0,
        'best_accuracy_individual': best_acc_ind,
        'greenest_individual': greenest_ind,
        'knee_point_individual': knee_ind,
        'all_pareto_solutions': pareto_front,
        'pheromone_motifs': pheromones.get_top_motifs_summary()
    }

    # Guardar automáticamente los 3 modelos representativos del Frente de Pareto
    try:
        from deepga.utils.model_utils import save_best_model
        save_best_model(variant="mo_v10_best_acc", execution=execution, bestind=best_acc_ind, in_channels=n_channels, out_size=out_size, n_classes=n_classes, chck_dir=chck_dir)
        save_best_model(variant="mo_v10_greenest", execution=execution, bestind=greenest_ind, in_channels=n_channels, out_size=out_size, n_classes=n_classes, chck_dir=chck_dir)
        save_best_model(variant="mo_v10_knee", execution=execution, bestind=knee_ind, in_channels=n_channels, out_size=out_size, n_classes=n_classes, chck_dir=chck_dir)
    except Exception as e:
        print(f"Nota al guardar modelos del Frente de Pareto V10: {e}")

    return results_df, pareto_front, knee_ind, surrogate_stats, mo_stats


def final_evaluation_mo(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader,
                        lr: float, max_params: int, device: torch.device, train_epochs: int,
                        loss_func, chck_dir: str, n_channels: int = 3, n_classes: int = 10,
                        out_size: int = 32, variant: str = "mo_v10", auto_download: bool = False):
    """Re-entrenamiento final para un modelo seleccionado del Frente de Pareto."""
    from variants.DeepGA import final_evaluation as fe
    return fe(execution, bestind, train_dl, val_dl, lr, max_params, 0.0, device, train_epochs, loss_func, chck_dir, n_channels, n_classes, out_size, variant=variant, auto_download=auto_download)
