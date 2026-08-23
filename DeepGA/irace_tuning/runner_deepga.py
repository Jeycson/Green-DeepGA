# -*- coding: utf-8 -*-
"""
Runner de DeepGA adaptado para irace (Automated Algorithm Configuration).
Ejecuta las variantes V10, V11 y V12 de DeepGA con los hiperparámetros
especificados por irace en parameters.txt.

Evaluación Multi-Objetivo Normalizada (Green NAS):
- Considera conjuntamente el rendimiento predictivo (Macro F1) y el consumo energético (kWh).
- Normaliza las métricas en base al archivo 'baseline.csv' (promedios de 10 corridas originales)
  para otorgar exactamente la misma ponderación representativa a datasets pesados
  (ej. PathMNIST) que a datasets ligeros (ej. BreadMNIST).
- Retorna a irace un único costo flotante a MINIMIZAR por STDOUT.
"""

import os
import sys
import argparse
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple

# Añadir el directorio raíz de DeepGA al sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_all_seeds(seed: int):
    """Establece todas las semillas para reproducibilidad exacta."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def normalize_dataset_name(name_or_path: str) -> str:
    """
    Normaliza un nombre o ruta de dataset a una clave canónica
    para indexación y matching robusto con baseline.csv.
    """
    clean = str(name_or_path).strip().lower()
    clean = clean.replace("\\", "/").split("/")[-1]
    clean = clean.replace("-", "").replace("_", "").replace(" ", "")
    clean = clean.replace(".19", "").replace("19", "")

    # Mapeo de alias comunes
    if any(k in clean for k in ["breadmnist", "breastmnist", "breast", "bread"]):
        return "breadmnist"
    if any(k in clean for k in ["dermamnist", "derma"]):
        return "dermamnist"
    if any(k in clean for k in ["bloodmnist", "blood"]):
        return "bloodmnist"
    if any(k in clean for k in ["organcmnist", "organc", "organ_c"]):
        return "organcmnist"
    if any(k in clean for k in ["pathmnist", "path"]):
        return "pathmnist"
    if any(k in clean for k in ["cifar10", "cifar"]):
        return "cifar10"
    if "covid" in clean:
        return "covid"
    if any(k in clean for k in ["tumour3", "tumours3", "tumor3", "tumors3"]):
        return "tumour3"
    if any(k in clean for k in ["tumour", "tumours", "tumor", "tumors"]):
        return "tumour"

    return clean


def resolve_dataset_path(dataset_name_or_path: str) -> str:
    """
    Localiza la ruta absoluta del dataset especificado.
    Busca en variables de entorno, directorios estándar y rutas relativas.
    """
    # 1. Si es una ruta existente
    candidate_path = Path(dataset_name_or_path).expanduser()
    if candidate_path.exists() and candidate_path.is_dir():
        return str(candidate_path.resolve())

    # 2. Variable de entorno
    env_dir = os.environ.get("DEEPGA_DATA_DIR", "")
    if env_dir:
        p = Path(env_dir) / dataset_name_or_path
        if p.exists():
            return str(p.resolve())

    # 3. Normalizar nombre para búsqueda de alias de carpetas
    name_clean = dataset_name_or_path.strip().lower()
    search_names = [dataset_name_or_path]
    if "tumour_3" in name_clean or "tumours_3" in name_clean or "tumor_3" in name_clean:
        search_names = ["Tumour_3", "Tumours_3", "tumour_3", "tumours_3", "tumor_3"]
    elif "tumour" in name_clean or "tumours" in name_clean or "tumor" in name_clean:
        search_names = ["Tumour", "Tumours", "tumour", "tumours", "tumor"]
    elif "covid" in name_clean:
        search_names = ["COVID", "covid", "covid-19", "COVID-19"]

    # 4. Directorios de búsqueda comunes
    search_dirs = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT.parent / "dataset",
        PROJECT_ROOT.parent / "datasets",
        PROJECT_ROOT.parent / "Datasets",
        Path.home() / "Documents" / "Datasets",
        Path.home() / "Datasets",
        Path.home() / "Downloads",
        Path.home() / "Downloads" / "Datasets",
        Path("/data"),
        Path("/datasets"),
    ]

    for base in search_dirs:
        for sname in search_names:
            candidate = base / sname
            if candidate.exists() and candidate.is_dir():
                return str(candidate.resolve())

    # Retornar original para datasets nativos (ej: cifar10, pathmnist, etc.)
    return dataset_name_or_path


def load_baseline_metrics(baseline_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Carga el archivo baseline.csv con los promedios de 10 corridas de la versión original.
    Columnas esperadas: Dataset, Enegy/Energy, Macro F1
    """
    baseline_map: Dict[str, Dict[str, Any]] = {}
    if not baseline_path.exists():
        return baseline_map

    try:
        df = pd.read_csv(baseline_path)
        col_map = {}
        for col in df.columns:
            c_clean = col.strip().lower().replace("_", " ")
            if "dataset" in c_clean or "nombre" in c_clean or "name" in c_clean:
                col_map["dataset"] = col
            elif "enegy" in c_clean or "energy" in c_clean or "energia" in c_clean or "kwh" in c_clean:
                col_map["energy"] = col
            elif "f1" in c_clean or "macro" in c_clean:
                col_map["f1"] = col

        ds_col = col_map.get("dataset", df.columns[0])
        en_col = col_map.get("energy", df.columns[1] if len(df.columns) > 1 else None)
        f1_col = col_map.get("f1", df.columns[2] if len(df.columns) > 2 else None)

        for _, row in df.iterrows():
            d_name = str(row[ds_col]).strip()
            e_val = float(row[en_col]) if en_col is not None and pd.notna(row[en_col]) else 0.005
            f_val = float(row[f1_col]) if f1_col is not None and pd.notna(row[f1_col]) else 75.0

            key = normalize_dataset_name(d_name)
            baseline_map[key] = {
                "dataset_orig": d_name,
                "energy_kwh": e_val,
                "macro_f1": f_val
            }
    except Exception as err:
        sys.stderr.write(f"Aviso al procesar baseline.csv ({baseline_path}): {err}\n")

    return baseline_map


def get_dataset_baseline(dataset_name: str, baseline_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recupera las métricas baseline (energy_kwh, macro_f1) para un dataset.
    Si no existe exactamente, usa substring matching o el promedio global de baseline.csv.
    """
    key = normalize_dataset_name(dataset_name)
    if key in baseline_map:
        return {**baseline_map[key], "is_fallback": False}

    for b_key, b_val in baseline_map.items():
        if b_key in key or key in b_key:
            return {**b_val, "is_fallback": False}

    # Fallback dinámico representativo promediado de baseline.csv
    if baseline_map:
        avg_energy = float(np.mean([v["energy_kwh"] for v in baseline_map.values()]))
        avg_f1 = float(np.mean([v["macro_f1"] for v in baseline_map.values()]))
    else:
        avg_energy = 0.00923
        avg_f1 = 71.16

    return {
        "dataset_orig": dataset_name,
        "energy_kwh": avg_energy,
        "macro_f1": avg_f1,
        "is_fallback": True
    }


def calculate_normalized_cost(
    f1_measured: float,
    energy_measured: float,
    baseline_f1: float,
    baseline_energy: float,
    weight_f1: float = 0.60,
    weight_energy: float = 0.40
) -> Tuple[float, Dict[str, Any]]:
    """
    Calcula el Costo Multi-Objetivo Normalizado (MINIMIZACIÓN para irace):

    Fórmula:
      Norm_Error  = (100.0 - F1_run) / max(100.0 - F1_base, 1.0)
      Norm_Energy = Energy_run / max(Energy_base, 1e-7)
      Cost        = (weight_f1 * Norm_Error) + (weight_energy * Norm_Energy)

    Garantías:
    - En el baseline exacto: Cost = weight_f1*(1.0) + weight_energy*(1.0) = 1.000000
    - Mejor Macro F1 (> baseline) -> Norm_Error < 1.0 -> reduce costo.
    - Menor Consumo (< baseline)  -> Norm_Energy < 1.0 -> reduce costo.
    - La normalización por el baseline propio de cada dataset otorga exactamente
      la MISMA ponderación representativa a datasets pesados (ej. PathMNIST, 0.032 kWh)
      que a datasets ligeros (ej. BreadMNIST, 0.00023 kWh).
    """
    f1_clean = max(0.0, min(100.0, float(f1_measured)))
    energy_clean = max(0.0, float(energy_measured))

    base_f1_clean = max(1.0, min(99.9, float(baseline_f1)))
    base_energy_clean = max(1e-7, float(baseline_energy))

    # Error relativo al margen de mejora del baseline
    base_error = max(1.0, 100.0 - base_f1_clean)
    curr_error = 100.0 - f1_clean
    norm_error = curr_error / base_error

    # Consumo relativo al baseline
    norm_energy = energy_clean / base_energy_clean

    # Costo final ponderado
    cost = (weight_f1 * norm_error) + (weight_energy * norm_energy)
    cost = max(0.0, float(cost))

    details = {
        "f1_measured": f1_clean,
        "f1_baseline": base_f1_clean,
        "norm_error": norm_error,
        "energy_measured_kwh": energy_clean,
        "energy_baseline_kwh": base_energy_clean,
        "norm_energy": norm_energy,
        "weight_f1": weight_f1,
        "weight_energy": weight_energy,
        "final_cost": cost
    }

    return cost, details


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGA irace Target Runner (Multi-Objective Normalized)")

    # Argumentos de instancia y ejecución irace
    parser.add_argument("--instance", type=str, required=True, help="Nombre o ruta del dataset")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--config-id", type=str, default="0", help="ID de configuración irace")
    parser.add_argument("--instance-id", type=str, default="0", help="ID de instancia irace")

    # Variante (restringida a v10, v11, v12 en parameters.txt)
    parser.add_argument("--variant", type=str, default="v12", choices=["v10", "v11", "v12", "v9", "v8", "v7", "v6", "v5", "v4", "v3", "v2", "v1"],
                        help="Variante de DeepGA a ejecutar")

    # Hiperparámetros de parameters.txt
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument("--pop-size", "--pop_size", dest="pop_size", type=int, default=12, help="Tamaño de población N")
    parser.add_argument("--generations", type=int, default=8, help="Número de generaciones T")
    parser.add_argument("--t-size", "--t_size", dest="t_size", type=int, default=3, help="Tamaño de torneo")
    parser.add_argument("--cr", type=float, default=0.75, help="Probabilidad de cruce")
    parser.add_argument("--mr", type=float, default=0.45, help="Probabilidad base de mutación")
    parser.add_argument("--mr-min", "--mr_min", dest="mr_min", type=float, default=0.10, help="Tasa mínima de mutación")
    parser.add_argument("--mr-max", "--mr_max", dest="mr_max", type=float, default=0.85, help="Tasa máxima de mutación")
    parser.add_argument("--final-epochs", "--final_epochs", "--final-epoch", dest="final_epochs", type=int, default=10,
                        help="Épocas de entrenamiento final del modelo ganador")

    # Meta-modelo Subrogado
    parser.add_argument("--pool-candidates-factor", "--pool_candidates_factor", dest="pool_candidates_factor",
                        type=int, default=4, help="Factor de candidatos a pre-evaluar")
    parser.add_argument("--kappa", type=float, default=0.15, help="Parámetro UCB de exploración del subrogado")

    # Parámetros V10 y V11 (Feromonas ACO)
    parser.add_argument("--rho", type=float, default=0.10, help="Tasa de evaporación de feromonas")
    parser.add_argument("--alpha", type=float, default=1.20, help="Exponente de atracción de feromonas")
    parser.add_argument("--top-k-ratio", "--top_k_ratio", dest="top_k_ratio", type=float, default=0.35, help="Fracción élite para feromonas")

    # Parámetros V11 y V12 (Multi-Isla)
    parser.add_argument("--n-islands", "--n_islands", dest="n_islands", type=int, default=3, help="Número de islas")
    parser.add_argument("--migration-interval", "--migration_interval", dest="migration_interval", type=int, default=8, help="Intervalo de migración")
    parser.add_argument("--migration-size", "--migration_size", dest="migration_size", type=int, default=1, help="Cantidad de migrantes por isla")

    # Parámetros V12 (Diversidad y Anti-Estancamiento)
    parser.add_argument("--target-diversity", "--target_diversity", dest="target_diversity", type=float, default=0.25, help="Umbral de diversidad")
    parser.add_argument("--stagnation-limit", "--stagnation_limit", dest="stagnation_limit", type=int, default=4, help="Límite de estancamiento")

    # Parámetros adicionales de entrenamiento del GA
    parser.add_argument("--train-epochs", "--train_epochs", dest="train_epochs", type=int, default=3, help="Épocas por individuo en GA")
    parser.add_argument("--w", type=float, default=0.05, help="Peso de penalización por parámetros")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32, choices=[16, 32, 64], help="Tamaño de batch")
    parser.add_argument("--train-final-model", action="store_true", default=True, help="Entrenar completamente el modelo ganador")
    parser.add_argument("--no-final-train", dest="train_final_model", action="store_false", help="Desactivar reentrenamiento final")

    # Espacio Arquitectónico
    parser.add_argument("--min-conv", type=int, default=2, help="Mínimo de capas convolucionales")
    parser.add_argument("--max-conv", type=int, default=5, help="Máximo de capas convolucionales")
    parser.add_argument("--min-full", type=int, default=1, help="Mínimo de capas densas")
    parser.add_argument("--max-full", type=int, default=3, help="Máximo de capas densas")
    parser.add_argument("--max-params", type=int, default=2000000, help="Tope máximo de parámetros")

    # Hardware y Dataset
    parser.add_argument("--img-size", type=int, default=64, help="Resolución de imagen")
    parser.add_argument("--in-channels", type=int, default=3, help="Número de canales de entrada")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Uso de AMP FP16")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Desactivar AMP")
    parser.add_argument("--max-spatial-size", type=int, default=4, help="Tope espacial pooling previo a FC")

    # Configuración de Evaluación Multi-Objetivo y Baseline
    parser.add_argument("--baseline-file", type=str, default=str(CURRENT_DIR / "baseline.csv"), help="Ruta al archivo baseline.csv")
    parser.add_argument("--weight-f1", type=float, default=0.60, help="Ponderación del rendimiento Macro F1 (por defecto: 0.60)")
    parser.add_argument("--weight-energy", type=float, default=0.40, help="Ponderación del consumo energético (por defecto: 0.40)")

    # Directorios de control
    parser.add_argument("--log-dir", type=str, default="./irace_logs", help="Directorio de logs detallados")
    parser.add_argument("--chck-dir", type=str, default="./irace_checkpoints", help="Directorio de checkpoints temporales")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Configurar reproducibilidad
    set_all_seeds(args.seed)

    # 2. Cargar tabla baseline de referencia
    baseline_path = Path(args.baseline_file)
    baseline_map = load_baseline_metrics(baseline_path)
    base_info = get_dataset_baseline(args.instance, baseline_map)
    base_f1 = base_info["macro_f1"]
    base_energy = base_info["energy_kwh"]

    # 3. Resolver ruta del dataset
    data_dir = resolve_dataset_path(args.instance)
    if not os.path.exists(data_dir) and not any(k in args.instance.lower() for k in ["mnist", "cifar"]):
        sys.stderr.write(f"ERROR: Dataset no encontrado en '{data_dir}' (instancia: {args.instance})\n")
        print("2.000000")  # Costo penalizado de fallo
        sys.exit(0)

    # 4. Preparar directorios de logs y checkpoints
    log_dir = Path(args.log_dir)
    chck_dir = Path(args.chck_dir) / f"c{args.config_id}_i{args.instance_id}_s{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    chck_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / f"run_c{args.config_id}_i{args.instance_id}_s{args.seed}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    cost = 2.000000  # Costo de seguridad por defecto

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            # Redirigir salidas internas al log detallado para no contaminar STDOUT de irace
            sys.stdout = log_file
            sys.stderr = log_file

            print(f"==================================================================", flush=True)
            print(f"  EJECUCIÓN DEEPGA - IRACE MULTI-OBJECTIVE NORMALIZED RUNNER", flush=True)
            print(f"==================================================================", flush=True)
            print(f"Config ID:         {args.config_id}", flush=True)
            print(f"Instance ID:       {args.instance_id}", flush=True)
            print(f"Instance Name:     {args.instance} (Ruta: {data_dir})", flush=True)
            print(f"Seed:              {args.seed}", flush=True)
            print(f"Variant:           {args.variant.upper()}", flush=True)
            print(f"Baseline Data:     F1_base={base_f1:.3f}%, Energy_base={base_energy:.6f} kWh (Fallback: {base_info.get('is_fallback', False)})", flush=True)
            print(f"Ponderaciones:     w_F1={args.weight_f1:.2f}, w_Energy={args.weight_energy:.2f}", flush=True)
            print(f"Hiperparámetros:   lr={args.lr}, pop={args.pop_size}, gen={args.generations}, t_size={args.t_size}, cr={args.cr}, mr={args.mr}", flush=True)
            print(f"                   mr_min={args.mr_min}, mr_max={args.mr_max}, final_epochs={args.final_epochs}", flush=True)
            print(f"==================================================================\n", flush=True)

            if TORCH_AVAILABLE:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = "cpu"

            from experiment_manager import ExperimentManager
            manager = ExperimentManager(
                country_iso_code="MEX",
                track_carbon=True
            )

            # Ejecución de DeepGA
            resultados = manager.run_deepga(
                variant=args.variant,
                execution=int(args.seed % 100000),
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
                max_spatial_size=args.max_spatial_size,
                data_root=data_dir,
                img_size=args.img_size,
                in_channels=args.in_channels,
                chck_dir=str(chck_dir) + "/",
                device=device,
                save_best_model_file=False,
                save_txt_report=False,
                train_final_model=args.train_final_model,
                final_train_epochs=args.final_epochs,
                auto_download=False
            )

            # Extraer métricas obtenidas
            # 1. Macro F1 (o accuracy si F1 no está disponible)
            f1_measured = resultados.get("f1")
            if f1_measured is None or np.isnan(f1_measured):
                f1_measured = resultados.get("final_test_accuracy")
            if f1_measured is None or np.isnan(f1_measured):
                f1_measured = resultados.get("best_accuracy", 0.0)

            # Normalizar a escala 0.0 - 100.0 si venía en [0.0, 1.0]
            if f1_measured is not None and f1_measured <= 1.0 and f1_measured > 0.0:
                f1_measured = f1_measured * 100.0
            elif f1_measured is None:
                f1_measured = 0.0

            # 2. Consumo Energético (kWh)
            energy_measured = resultados.get("energy_consumed_kwh", 0.0)
            if energy_measured is None or energy_measured <= 0.0:
                elapsed_s = resultados.get("execution_time_seconds", 1.0)
                energy_measured = 0.150 * (elapsed_s / 3600.0)

            # 3. Cálculo del Costo Normalizado respecto al Baseline
            cost, details = calculate_normalized_cost(
                f1_measured=f1_measured,
                energy_measured=energy_measured,
                baseline_f1=base_f1,
                baseline_energy=base_energy,
                weight_f1=args.weight_f1,
                weight_energy=args.weight_energy
            )

            print("\n------------------------------------------------------------------", flush=True)
            print("📊 DESGLOSE DE MÉTRICAS MULTI-OBJETIVO Y NORMALIZACIÓN", flush=True)
            print("------------------------------------------------------------------", flush=True)
            print(f"Dataset Instancia:         {args.instance}", flush=True)
            print(f"Macro F1 Medido:           {f1_measured:.3f}% (Baseline: {base_f1:.3f}%)", flush=True)
            print(f"Error Normalizado:         {details['norm_error']:.4f} (Factor: x{details['weight_f1']})", flush=True)
            print(f"Consumo Energético Medido: {energy_measured:.6f} kWh (Baseline: {base_energy:.6f} kWh)", flush=True)
            print(f"Energía Normalizada:       {details['norm_energy']:.4f} (Factor: x{details['weight_energy']})", flush=True)
            print(f"Tiempo de Ejecución:       {resultados.get('execution_time_seconds', 0.0):.2f} s", flush=True)
            print(f"Parámetros de la Red:      {resultados.get('best_total_params', 0):,}", flush=True)
            print(f"🎯 Costo Final para irace: {cost:.6f}", flush=True)
            print("------------------------------------------------------------------", flush=True)

    except Exception as e:
        sys.stdout = original_stdout
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n❌ EXCEPCIÓN DURANTE EJECUCIÓN:\n{traceback.format_exc()}\n")
        # Devolver costo penalizado (2.0 = peor que baseline 1.0)
        print("2.000000")
        sys.exit(0)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Limpiar checkpoints temporales
        try:
            for item in chck_dir.glob("*"):
                if item.is_file():
                    item.unlink()
            if chck_dir.exists():
                chck_dir.rmdir()
        except Exception:
            pass

    # IMPORTANTE: STDOUT debe contener ÚNICAMENTE el valor numérico del costo para irace
    print(f"{cost:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()

