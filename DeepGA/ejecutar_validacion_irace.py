# -*- coding: utf-8 -*-
"""
Script de Validación Automatizada para Resultados de irace en Green DeepGA:
- Ejecuta N repeticiones independientes (por defecto: 10 corridas con 10 semillas distintas).
- Recibe todos los hiperparámetros optimizados por irace vía CLI o desde un archivo JSON (best_configuration.json).
- Registra métricas completas por corrida: Test Accuracy, Macro F1, Consumo Energético (kWh),
  Huella de Carbono (gCO2eq), Parámetros, FLOPs y Tiempo.
- Genera matrices de confusión y calcula estadísticas globales (Media ± Desviación Estándar).
- Exporta resúmenes en CSV y TXT compatibles con Excel.

Uso rápido (cargando el JSON de irace):
    python ejecutar_validacion_irace.py --config-json irace_tuning/best_configuration.json --data-root ./Datasets/Tumour --num-runs 10

Uso pasando todos los argumentos por CLI:
    python ejecutar_validacion_irace.py --variant v12 --lr 0.0004 --pop-size 14 --generations 10 \
        --t-size 2 --cr 0.75 --mr 0.27 --mr-min 0.17 --mr-max 0.70 --final-epochs 25 \
        --pool-candidates-factor 5 --kappa 0.17 --n-islands 2 --migration-interval 5 \
        --migration-size 1 --target-diversity 0.24 --stagnation-limit 5 \
        --data-root ./Datasets/Tumour --num-runs 10
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Añadir el directorio raíz
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




def parse_args():
    parser = argparse.ArgumentParser(
        description="Validación Estadística Automatizada (10 corridas) para Configuración Óptima de irace en DeepGA"
    )

    # --- Archivo de Configuración JSON Opcional ---
    parser.add_argument("--config-json", type=str, default=None,
                        help="Ruta al archivo JSON generado por irace (ej. irace_tuning/best_configuration.json)")
    parser.add_argument("--config-name", "--config-tag", type=str, default=None, dest="config_name",
                        help="Nombre/etiqueta única para la configuración (default: nombre del JSON o derivado de params)")
    parser.add_argument("--force-restart", "--no-resume", action="store_true", default=False, dest="force_restart",
                        help="Fuerza un inicio limpio ignorando cualquier checkpoint previo existente")

    # --- Control de Repeticiones y Semillas ---
    parser.add_argument("--num-runs", "--repetitions", type=int, default=10, dest="num_runs",
                        help="Número de repeticiones independientes (default: 10)")
    parser.add_argument("--start-seed", type=int, default=1,
                        help="Semilla inicial (default: 1)")
    parser.add_argument("--start-exec", type=int, default=1,
                        help="Número identificador de ejecución inicial (default: 1)")

    # --- Dataset & Hardware ---
    parser.add_argument("--data-root", "--dataset", type=str, default="./Datasets/Covid", dest="data_root",
                        help="Ruta o nombre del dataset (default: ./Datasets/Covid)")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Resolución de imágenes (default: 28 para MNIST/MedMNIST, 64 otros)")
    parser.add_argument("--in-channels", type=int, default=None, choices=[1, 3],
                        help="Canales de entrada: 1 para Grayscale/Covid, 3 para RGB")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamaño de lote (default: 32)")
    parser.add_argument("--output-dir", "--chck-dir", type=str, default="./checkpoints_validacion/", dest="output_dir",
                        help="Directorio donde se guardarán modelos, reportes y resúmenes")
    parser.add_argument("--country-iso", type=str, default="MEX",
                        help="Código ISO del país para tracking de emisiones de carbono")
    parser.add_argument("--device", type=str, default=None,
                        help="Dispositivo PyTorch ('cuda', 'cpu' o None para auto-detección)")

    # --- Hiperparámetros Globales de irace ---
    parser.add_argument("--variant", "--Variant", type=str, default="v12", dest="variant",
                        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "mo_v9", "mo_v10", "mo_v11"],
                        help="Variante de DeepGA ganadora de irace (default: v12)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate de Adam (default: 1e-4)")
    parser.add_argument("--pop-size", "--pop_size", type=int, default=12, dest="pop_size",
                        help="Tamaño de la población N (default: 12)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Número de generaciones evolutivas (default: 5)")
    parser.add_argument("--train-epochs", "--train_epochs", type=int, default=2, dest="train_epochs",
                        help="Épocas de evaluación rápida durante el GA (default: 2)")
    parser.add_argument("--final-epochs", "--final-epoch", "--final_epoch", type=int, default=25, dest="final_epochs",
                        help="Épocas de re-entrenamiento completo de la red ganadora (default: 25)")
    parser.add_argument("--t-size", "--t_size", type=int, default=3, dest="t_size",
                        help="Tamaño de torneo de selección (default: 3)")
    parser.add_argument("--cr", type=float, default=0.70,
                        help="Probabilidad de cruce / crossover rate (default: 0.70)")
    parser.add_argument("--mr", type=float, default=0.50,
                        help="Probabilidad de mutación / mutation rate base (default: 0.50)")
    parser.add_argument("--mr-min", "--mr_min", type=float, default=0.10, dest="mr_min",
                        help="Tasa mínima de mutación adaptativa (default: 0.10)")
    parser.add_argument("--mr-max", "--mr_max", type=float, default=0.85, dest="mr_max",
                        help="Tasa máxima de mutación adaptativa (default: 0.85)")
    parser.add_argument("--w", type=float, default=0.30,
                        help="Ponderación w de penalización de complejidad de parámetros (default: 0.30)")

    # --- Meta-Modelo Subrogado (Random Forest + UCB) ---
    parser.add_argument("--pool-candidates-factor", "--pool_candidates_factor", type=int, default=5, dest="pool_candidates_factor",
                        help="Factor de candidatos a predecir por el subrogado (default: 5)")
    parser.add_argument("--kappa", type=float, default=0.10,
                        help="Parámetro kappa de exploración UCB en el subrogado (default: 0.10)")

    # --- Matriz de Feromonas ACO (V10 y V11) ---
    parser.add_argument("--rho", type=float, default=0.10,
                        help="Tasa de evaporación de feromonas rho en V10/V11 (default: 0.10)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ponderación alpha de feromonas en mutación guiada (default: 1.0)")
    parser.add_argument("--top-k-ratio", "--top_k_ratio", type=float, default=0.20, dest="top_k_ratio",
                        help="Ratio de individuos élite para depósito de feromonas (default: 0.20)")

    # --- Arquitectura Multi-Isla (V11 y V12) ---
    parser.add_argument("--n-islands", "--n_islands", type=int, default=3, dest="n_islands",
                        help="Número de islas evolutivas (default: 3)")
    parser.add_argument("--migration-interval", "--migration_interval", type=int, default=12, dest="migration_interval",
                        help="Intervalo de generaciones para migración (default: 12)")
    parser.add_argument("--migration-size", "--migration_size", type=int, default=1, dest="migration_size",
                        help="Cantidad de individuos que migran por isla (default: 1)")

    # --- Preservación de Diversidad & Anti-Estancamiento (V12) ---
    parser.add_argument("--target-diversity", "--target_diversity", type=float, default=0.25, dest="target_diversity",
                        help="Diversidad estructural objetivo intra-isla (default: 0.25)")
    parser.add_argument("--stagnation-limit", "--stagnation_limit", type=int, default=4, dest="stagnation_limit",
                        help="Límite de generaciones sin mejora antes de reiniciar isla (default: 4)")

    # --- Otros límites estructurales ---
    parser.add_argument("--min-conv", type=int, default=2)
    parser.add_argument("--max-conv", type=int, default=5)
    parser.add_argument("--min-full", type=int, default=1)
    parser.add_argument("--max-full", type=int, default=4)
    parser.add_argument("--max-params", type=int, default=2000000)
    parser.add_argument("--use-amp", action="store_true", default=True)

    return parser.parse_args()


def load_json_config(args, json_path: str):
    """Carga hiperparámetros desde un archivo JSON (como best_configuration.json de irace)."""
    if not os.path.exists(json_path):
        print(f"⚠️ Advertencia: No se encontró el archivo JSON '{json_path}'. Se usarán los argumentos CLI.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Mapeos de nombres alternativos entre irace y el script
    key_mapping = {
        "variant": "variant",
        "Variant": "variant",
        "lr": "lr",
        "pop_size": "pop_size",
        "pop-size": "pop_size",
        "generations": "generations",
        "t_size": "t_size",
        "t-size": "t_size",
        "cr": "cr",
        "mr": "mr",
        "mr_min": "mr_min",
        "mr-min": "mr_min",
        "mr_max": "mr_max",
        "mr-max": "mr_max",
        "final_epoch": "final_epochs",
        "final_epochs": "final_epochs",
        "final-epochs": "final_epochs",
        "train_epochs": "train_epochs",
        "train-epochs": "train_epochs",
        "pool_candidates_factor": "pool_candidates_factor",
        "pool-candidates-factor": "pool_candidates_factor",
        "kappa": "kappa",
        "rho": "rho",
        "alpha": "alpha",
        "top_k_ratio": "top_k_ratio",
        "top-k-ratio": "top_k_ratio",
        "n_islands": "n_islands",
        "n-islands": "n_islands",
        "migration_interval": "migration_interval",
        "migration-interval": "migration_interval",
        "migration_size": "migration_size",
        "migration-size": "migration_size",
        "target_diversity": "target_diversity",
        "target-diversity": "target_diversity",
        "stagnation_limit": "stagnation_limit",
        "stagnation-limit": "stagnation_limit",
        "w": "w",
        "batch_size": "batch_size",
        "batch-size": "batch_size",
        "min_conv": "min_conv",
        "max_conv": "max_conv",
        "min_full": "min_full",
        "max_full": "max_full",
        "max_params": "max_params"
    }

    print(f"📖 Cargando configuración óptima desde: {json_path}")
    loaded_count = 0
    for k, v in cfg.items():
        attr_name = key_mapping.get(k, k.replace("-", "_"))
        if hasattr(args, attr_name):
            current_val = getattr(args, attr_name)
            if isinstance(current_val, int) and not isinstance(v, bool):
                v = int(float(v))
            elif isinstance(current_val, float):
                v = float(v)
            elif isinstance(current_val, bool):
                v = bool(v)
            setattr(args, attr_name, v)
            loaded_count += 1
            print(f"   • {attr_name:<24} = {v}")

    print(f"✅ Se aplicaron {loaded_count} parámetros desde el JSON.\n")


def resolve_auto_dataset_params(args):
    """Detecta automáticamente canales y tamaño de imagen si no fueron especificados."""
    if args.img_size is not None:
        effective_img_size = args.img_size
    elif "mnist" in str(args.data_root).lower():
        effective_img_size = 28
    else:
        effective_img_size = 64

    if args.in_channels is not None:
        effective_in_channels = args.in_channels
    elif any(k in str(args.data_root).lower() for k in ["covid", "breast", "bread", "grayscale"]):
        effective_in_channels = 1
    elif any(k in str(args.data_root).lower() for k in ["tumour", "tumor", "cifar", "rgb", "derma", "blood", "path"]):
        effective_in_channels = 3
    else:
        effective_in_channels = 1

    return effective_img_size, effective_in_channels


def main():
    args = parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from deepga.experiment.manager import ExperimentManager

    # Si se especificó un archivo JSON, cargarlo
    if args.config_json:
        load_json_config(args, args.config_json)

    # Determinar nombre / tag para la configuración evaluada
    if args.config_name:
        config_name = args.config_name
    elif args.config_json:
        config_name = Path(args.config_json).stem
    else:
        config_name = f"cfg_{args.variant.lower()}_gen{args.generations}_pop{args.pop_size}"

    config_clean = str(config_name).replace("-", "_").replace(" ", "_").replace(".", "_").lower()

    img_size, in_channels = resolve_auto_dataset_params(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detección de dispositivo
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    end_seed = args.start_seed + args.num_runs - 1

    print("\n" + "=" * 80)
    print("   VALIDACIÓN AUTOMATIZADA DE RESULTADOS DE IRACE EN GREEN DEEPGA")
    print("=" * 80)
    print(f"📌 Configuración (Tag):     {config_name}")
    print(f"📌 Variante:                 {args.variant.upper()}")
    print(f"📌 Total de Repeticiones:    {args.num_runs} corridas independientes")
    print(f"📌 Rango de Semillas:        {args.start_seed} a {end_seed}")
    print(f"📌 Execution Inicial:        {args.start_exec}")
    print(f"📌 Dataset / Resolución:     {args.data_root} | {img_size}x{img_size} ({in_channels} canal/es)")
    print(f"📌 Dispositivo de Cómputo:   {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" and torch.cuda.is_available() else " (CPU)"))
    print(f"📌 Directorio de Resultados: {out_dir.resolve()}")
    if args.force_restart:
        print(f"📌 Reinicio Forzado:         ACTIVADO (ignora checkpoints anteriores)")
    print("-" * 80)
    print("📋 HIPERPARÁMETROS OPTIMIZADOS A VALIDAR:")
    print(f"   • Población (pop_size)     : {args.pop_size:<6} | Generaciones (generations) : {args.generations}")
    print(f"   • Tasa Aprendizaje (lr)    : {args.lr:<6} | Peso Penalización (w)    : {args.w}")
    print(f"   • Crossover Rate (cr)      : {args.cr:<6} | Mutation Rate (mr)       : {args.mr}")
    print(f"   • Rango Mutación Adaptativa: [{args.mr_min}, {args.mr_max}] | Tamaño Torneo (t_size)  : {args.t_size}")
    print(f"   • Épocas GA (train_epochs) : {args.train_epochs:<6} | Épocas Finales (final_epoch) : {args.final_epochs}")
    print(f"   • Factor Subrogado (pool)  : {args.pool_candidates_factor:<6} | Kappa Exploración UCB    : {args.kappa}")
    if args.variant.lower() in ["v10", "v11", "mo_v10", "mo_v11"]:
        print(f"   • Feromonas (rho / alpha)  : {args.rho} / {args.alpha} | Top K Ratio             : {args.top_k_ratio}")
    if args.variant.lower() in ["v11", "v12", "mo_v11"]:
        print(f"   • Islas (n_islands)        : {args.n_islands:<6} | Intervalo Migración       : {args.migration_interval}")
        print(f"   • Tamaño Migración         : {args.migration_size}")
    if args.variant.lower() in ["v12"]:
        print(f"   • Diversidad Objetivo      : {args.target_diversity:<6} | Límite Estancamiento     : {args.stagnation_limit}")
    print("=" * 80 + "\n")

    manager = ExperimentManager(
        country_iso_code=args.country_iso,
        track_carbon=True
    )

    results_list = []
    current_exec = args.start_exec
    start_time_all = time.time()

    summary_csv_path = out_dir / f"validacion_irace_{config_clean}_{args.num_runs}_corridas.csv"
    summary_txt_path = out_dir / f"validacion_irace_{config_clean}_{args.num_runs}_corridas.txt"
    default_summary_csv = out_dir / "validacion_irace_10_corridas.csv"
    default_summary_txt = out_dir / "validacion_irace_10_corridas.txt"
    stats_csv_path = out_dir / f"estadisticas_globales_{config_clean}_validacion.csv"
    default_stats_csv = out_dir / "estadisticas_globales_validacion.csv"

    for run_idx, seed in enumerate(range(args.start_seed, end_seed + 1), start=1):
        print("\n" + "#" * 80)
        print(f"▶ [CORRIDA {run_idx}/{args.num_runs}] Semilla: {seed} | Execution ID: {current_exec} | Variante: {args.variant.upper()} | Config: {config_name}")
        print("#" * 80)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t_run_start = time.time()

        try:
            resultados = manager.run_deepga(
                variant=args.variant,
                execution=current_exec,
                seed=seed,
                config_name=config_name,
                force_restart=args.force_restart,
                population_size=args.pop_size,
                generations=args.generations,
                train_epochs=args.train_epochs,
                lr=args.lr,
                w=args.w,
                cr=args.cr,
                mr=args.mr,
                t_size=args.t_size,
                min_conv=args.min_conv,
                max_conv=args.max_conv,
                min_full=args.min_full,
                max_full=args.max_full,
                max_params=args.max_params,
                batch_size=args.batch_size,
                pool_candidates_factor=args.pool_candidates_factor,
                kappa=args.kappa,
                mr_min=args.mr_min,
                mr_max=args.mr_max,
                rho=args.rho,
                alpha=args.alpha,
                top_k_ratio=args.top_k_ratio,
                n_islands=args.n_islands,
                migration_interval=args.migration_interval,
                migration_size=args.migration_size,
                target_diversity=args.target_diversity,
                stagnation_limit=args.stagnation_limit,
                use_amp=args.use_amp,
                data_root=args.data_root,
                img_size=img_size,
                in_channels=in_channels,
                chck_dir=str(out_dir) + "/",
                device=device,
                save_best_model_file=True,
                save_txt_report=True,
                train_final_model=True,
                final_train_epochs=args.final_epochs,
                auto_download=False
            )

            t_run_elapsed = time.time() - t_run_start

            val_acc = resultados.get("best_val_accuracy", resultados.get("best_accuracy", 0.0))
            if isinstance(val_acc, (int, float)) and val_acc <= 1.0 and val_acc > 0.0:
                val_acc = val_acc * 100.0

            test_acc = resultados.get("final_test_accuracy", None)
            if test_acc is not None and isinstance(test_acc, (int, float)) and test_acc <= 1.0 and test_acc > 0.0:
                test_acc = test_acc * 100.0

            prec = resultados.get("final_precision", None)
            rec = resultados.get("final_recall", None)
            f1 = resultados.get("final_f1", None)

            energy_kwh = resultados.get("energy_consumed_kwh", 0.0)
            co2_g = resultados.get("carbon_emissions_g_co2", 0.0)
            total_params = resultados.get("best_total_params", 0)
            flops = resultados.get("best_estimated_flops", 0)
            model_size_mb = resultados.get("best_model_size_mb", 0.0)
            saved_model_path = resultados.get("saved_model_path", "")

            test_dl = resultados.get("test_dataloader")
            class_names = resultados.get("class_names", [f"Clase_{i}" for i in range(10)])
            if test_dl is not None and saved_model_path and os.path.exists(saved_model_path):
                cm_save_path = out_dir / f"matriz_confusion_{args.variant.lower()}_{config_clean}_seed_{seed}_exec_{current_exec}.png"
                try:
                    manager.generate_confusion_matrix(
                        model_or_path=saved_model_path,
                        dataloader=test_dl,
                        class_names=class_names,
                        title=f"Matriz de Confusión - {args.variant.upper()} ({config_name}, Semilla {seed})",
                        save_fig_path=str(cm_save_path),
                        auto_download_plot=False
                    )
                except Exception as ex_cm:
                    print(f"ℹ️ (Nota sobre matriz de confusión: {ex_cm})")

            run_row = {
                "Run": run_idx,
                "Config": config_name,
                "Seed": seed,
                "Execution": current_exec,
                "Variant": args.variant.upper(),
                "Val_Accuracy_%": round(float(val_acc), 2) if val_acc is not None else np.nan,
                "Test_Accuracy_%": round(float(test_acc), 2) if test_acc is not None else np.nan,
                "Precision_%": round(float(prec), 2) if prec is not None else np.nan,
                "Recall_%": round(float(rec), 2) if rec is not None else np.nan,
                "F1_Macro_%": round(float(f1), 2) if f1 is not None else np.nan,
                "Energy_kWh": round(float(energy_kwh), 6),
                "CO2_g": round(float(co2_g), 4),
                "Total_Params": int(total_params),
                "FLOPs": int(flops),
                "Model_Size_MB": round(float(model_size_mb), 3),
                "Time_Seconds": round(float(t_run_elapsed), 2),
                "Saved_Model": os.path.basename(saved_model_path) if saved_model_path else ""
            }

            results_list.append(run_row)

            print(f"\n✅ [CORRIDA {run_idx} EXITOSA]")
            print(f"   • Val Accuracy : {val_acc:.2f}% | Test Accuracy: {test_acc:.2f}%" if test_acc is not None else f"   • Val Accuracy: {val_acc:.2f}%")
            if f1 is not None:
                print(f"   • Macro F1     : {f1:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}%")
            print(f"   • Energía      : {energy_kwh:.6f} kWh | Huella CO₂: {co2_g:.4f} gCO₂eq")
            print(f"   • Parámetros   : {total_params:,} | FLOPs: {flops:,}")
            print(f"   • Tiempo       : {t_run_elapsed:.2f} s ({t_run_elapsed/60.0:.2f} min)")

            df_current = pd.DataFrame(results_list)
            df_current.to_csv(summary_csv_path, index=False)
            df_current.to_csv(summary_txt_path, sep="\t", index=False)
            df_current.to_csv(default_summary_csv, index=False)
            df_current.to_csv(default_summary_txt, sep="\t", index=False)

        except Exception as e:
            print(f"\n❌ [ERROR] Falló la corrida #{run_idx} con Semilla {seed} y Exec {current_exec}: {e}")
            import traceback
            traceback.print_exc()

        current_exec += 1

    total_time = time.time() - start_time_all

    if not results_list:
        print("\n❌ No se completaron corridas exitosamente.")
        return

    df_results = pd.DataFrame(results_list)

    print("\n" + "=" * 80)
    print(f"        RESUMEN ESTADÍSTICO DE LAS {len(results_list)} REPETICIONES (VALIDACIÓN IRACE)")
    print("=" * 80)
    print(df_results[["Run", "Seed", "Val_Accuracy_%", "Test_Accuracy_%", "F1_Macro_%", "Energy_kWh", "CO2_g", "Total_Params", "Time_Seconds"]].to_string(index=False))
    print("-" * 80)

    numeric_cols = [
        "Val_Accuracy_%", "Test_Accuracy_%", "Precision_%", "Recall_%", "F1_Macro_%",
        "Energy_kWh", "CO2_g", "Total_Params", "FLOPs", "Model_Size_MB", "Time_Seconds"
    ]

    stats_summary = []
    print("📊 MÉTRICAS PROMEDIO Y DESVIACIÓN ESTÁNDAR (Mean ± Std):")
    for col in numeric_cols:
        if col in df_results.columns and not df_results[col].dropna().empty:
            mean_val = df_results[col].mean()
            std_val = df_results[col].std(ddof=1) if len(df_results[col].dropna()) > 1 else 0.0
            min_val = df_results[col].min()
            max_val = df_results[col].max()

            stats_summary.append({
                "Métrica": col,
                "Media (Mean)": round(mean_val, 4),
                "Desv. Est. (Std)": round(std_val, 4),
                "Mínimo (Min)": round(min_val, 4),
                "Máximo (Max)": round(max_val, 4)
            })

            unit = "%" if "%" in col else ("kWh" if "kWh" in col else ("gCO2" if "CO2" in col else ("s" if "Time" in col else "")))
            print(f"   • {col:<18} : {mean_val:>10.4f} ± {std_val:<8.4f} {unit}  [Min: {min_val:.2f}, Max: {max_val:.2f}]")

    df_stats = pd.DataFrame(stats_summary)
    df_stats.to_csv(stats_csv_path, index=False)
    df_stats.to_csv(default_stats_csv, index=False)

    print("\n" + "=" * 80)
    print(f"🎉 VALIDACIÓN FINALIZADA CON ÉXITO")
    print(f"⏱️ Tiempo Total Acumulado: {total_time/60.0:.2f} minutos ({total_time:.1f} s)")
    print(f"💾 Archivos guardados en: {out_dir.resolve()}/")
    print(f"   1. Tabla corrida por corrida (Excel CSV): {summary_csv_path.name}")
    print(f"   2. Tabla corrida por corrida (TSV):       {summary_txt_path.name}")
    print(f"   3. Estadísticas Globales (Mean ± Std):   {stats_csv_path.name}")
    print(f"   4. Modelos entrenados y Matrices de Confusión por semilla")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
