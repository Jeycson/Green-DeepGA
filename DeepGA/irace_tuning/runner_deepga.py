# -*- coding: utf-8 -*-
"""
Runner de DeepGA adaptado para irace (Automated Algorithm Configuration).
Ejecuta las variantes V10, V11 y V12 de DeepGA con los hiperparámetros
especificados por irace y reporta el costo (1.0 - accuracy) para minimización.

Soporta:
- Variantes: v10 (ACO Pheromones + Surrogate), v11 (Multi-Island + Pheromones), v12 (Pure Multi-Island + Diversity)
- Datasets: Tumours y Tumours_3 (detección automática en local o remoto)
- Registro de logs limpio para no contaminar STDOUT de irace.
"""

import os
import sys
import argparse
import random
import traceback
from pathlib import Path

# Añadir el directorio raíz de DeepGA al sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from experiment_manager import ExperimentManager


def set_all_seeds(seed: int):
    """Establece todas las semillas para reproducibilidad exacta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_dataset_path(dataset_name_or_path: str) -> str:
    """
    Localiza la ruta absoluta del dataset especificado (Tumour / Tumour_3).
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

    # 3. Normalizar nombre para búsqueda de alias
    name_clean = dataset_name_or_path.strip().lower()
    search_names = [dataset_name_or_path]
    if "tumour_3" in name_clean or "tumours_3" in name_clean or "tumor_3" in name_clean:
        search_names = ["Tumour_3", "Tumours_3", "tumour_3", "tumours_3", "tumor_3"]
    elif "tumour" in name_clean or "tumours" in name_clean or "tumor" in name_clean:
        search_names = ["Tumour", "Tumours", "tumour", "tumours", "tumor"]

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

    # Si no se encuentra, retornar el valor original para que el loader falle con mensaje descriptivo
    return dataset_name_or_path


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGA irace Target Runner")

    # Argumentos de instancia y ejecución irace
    parser.add_argument("--instance", type=str, required=True, help="Nombre o ruta del dataset (ej: Tumour, Tumour_3)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--config-id", type=str, default="0", help="ID de la configuración evaluada")
    parser.add_argument("--instance-id", type=str, default="0", help="ID de la instancia")

    # Variante (restringida a v10, v11, v12)
    parser.add_argument("--variant", type=str, default="v12", choices=["v10", "v11", "v12"],
                        help="Variante de DeepGA a ejecutar (v10, v11 o v12)")

    # Hiperparámetros de Entrenamiento / Algoritmo Genético
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate para entrenamiento de cada red")
    parser.add_argument("--w", type=float, default=0.05, help="Peso de penalización por parámetros (0.01 a 0.3)")
    parser.add_argument("--train-epochs", type=int, default=3, help="Épocas de entrenamiento por individuo en el GA")
    parser.add_argument("--pop-size", type=int, default=12, help="Tamaño total de la población N")
    parser.add_argument("--generations", type=int, default=8, help="Número de generaciones T")
    parser.add_argument("--batch-size", type=int, default=32, choices=[16, 32, 64], help="Tamaño de batch")
    parser.add_argument("--t-size", type=int, default=3, help="Tamaño de torneo de selección")
    parser.add_argument("--cr", type=float, default=0.75, help="Probabilidad de cruce")
    parser.add_argument("--mr", type=float, default=0.45, help="Probabilidad base de mutación")
    parser.add_argument("--mr-min", type=float, default=0.10, help="Tasa mínima de mutación adaptativa")
    parser.add_argument("--mr-max", type=float, default=0.85, help="Tasa máxima de mutación adaptativa")

    # Espacio Arquitectónico
    parser.add_argument("--min-conv", type=int, default=2, help="Mínimo de capas convolucionales")
    parser.add_argument("--max-conv", type=int, default=5, help="Máximo de capas convolucionales")
    parser.add_argument("--min-full", type=int, default=1, help="Mínimo de capas densas (FC)")
    parser.add_argument("--max-full", type=int, default=3, help="Máximo de capas densas (FC)")
    parser.add_argument("--max-params", type=int, default=2000000, help="Tope máximo de parámetros")

    # Meta-modelo Subrogado
    parser.add_argument("--pool-candidates-factor", type=int, default=4, help="Factor de candidatos a pre-evaluar")
    parser.add_argument("--kappa", type=float, default=0.15, help="Parámetro UCB de exploración del subrogado")

    # Parámetros V10 y V11 (Feromonas ACO)
    parser.add_argument("--rho", type=float, default=0.10, help="Tasa de evaporación de feromonas (V10/V11)")
    parser.add_argument("--alpha", type=float, default=1.2, help="Exponente de atracción de feromonas (V10/V11)")
    parser.add_argument("--top-k-ratio", type=float, default=0.35, help="Fracción de élite para depósito de feromonas (V10/V11)")

    # Parámetros V11 y V12 (Multi-Isla)
    parser.add_argument("--n-islands", type=int, default=3, help="Número de islas independientes (V11/V12)")
    parser.add_argument("--migration-interval", type=int, default=8, help="Intervalo de migración en generaciones (V11/V12)")
    parser.add_argument("--migration-size", type=int, default=1, help="Cantidad de migrantes por isla (V11/V12)")

    # Parámetros V12 (Diversidad y Anti-estancamiento)
    parser.add_argument("--target-diversity", type=float, default=0.25, help="Umbral de diversidad estructural intra-isla (V12)")
    parser.add_argument("--stagnation-limit", type=int, default=4, help="Generaciones sin mejora antes de anti-estancamiento (V12)")

    # Opciones de Hardware y Dataset
    parser.add_argument("--img-size", type=int, default=64, help="Resolución de las imágenes (por defecto: 64)")
    parser.add_argument("--in-channels", type=int, default=3, help="Número de canales (3 para RGB)")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Uso de AMP FP16")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--max-spatial-size", type=int, default=4, help="Tope espacial previo a Fully-Connected")

    # Directorios de control
    parser.add_argument("--log-dir", type=str, default="./irace_logs", help="Directorio para logs detallados")
    parser.add_argument("--chck-dir", type=str, default="./irace_checkpoints", help="Directorio temporal de checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Configurar reproducibilidad
    set_all_seeds(args.seed)

    # 2. Resolver directorio del dataset
    data_dir = resolve_dataset_path(args.instance)
    if not os.path.exists(data_dir):
        # En caso de error fatal en ruta, escribir a stderr y retornar peor costo
        sys.stderr.write(f"ERROR: Dataset no encontrado en '{data_dir}' (instancia: {args.instance})\n")
        print("1.000000") # Peor costo (1.0 = 0% accuracy)
        sys.exit(1)

    # 3. Preparar directorios de logs y checkpoints
    log_dir = Path(args.log_dir)
    chck_dir = Path(args.chck_dir) / f"c{args.config_id}_i{args.instance_id}_s{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    chck_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / f"run_c{args.config_id}_i{args.instance_id}_s{args.seed}.log"

    # Redirigir stdout/stderr de la neuroevolución al archivo de log para mantener limpio el STDOUT de irace
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    cost = 1.0

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            # Redirigir salidas
            sys.stdout = log_file
            sys.stderr = log_file

            print(f"==================================================", flush=True)
            print(f"  EJECUCIÓN DEEPGA - IRACE TUNING RUNNER", flush=True)
            print(f"==================================================", flush=True)
            print(f"Config ID:     {args.config_id}", flush=True)
            print(f"Instance ID:   {args.instance_id}", flush=True)
            print(f"Instance Name: {args.instance} ({data_dir})", flush=True)
            print(f"Seed:          {args.seed}", flush=True)
            print(f"Variant:       {args.variant.upper()}", flush=True)
            print(f"Hyperparams:   lr={args.lr}, w={args.w}, epochs={args.train_epochs}, pop={args.pop_size}, gen={args.generations}, batch={args.batch_size}", flush=True)
            print(f"==================================================\n", flush=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            manager = ExperimentManager(
                country_iso_code="MEX",
                track_carbon=False # Desactivado para máxima velocidad en irace
            )

            # Ejecutar DeepGA
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
                save_best_model_file=False, # No guardar .pth pesados por cada intento
                save_txt_report=False,
                train_final_model=False,    # Evaluación basada en la mejor validación del GA
                auto_download=False
            )

            best_acc = resultados.get("best_accuracy", 0.0)
            if best_acc is None or np.isnan(best_acc):
                best_acc = 0.0

            # Si el accuracy viene en formato porcentaje (> 1.0), normalizar a [0.0, 1.0]
            if best_acc > 1.0:
                best_acc = best_acc / 100.0

            # irace MINIMIZA el costo: Costo = 1.0 - Accuracy
            cost = max(0.0, min(1.0, 1.0 - float(best_acc)))

            print(f"\n✅ Finalizado con éxito.", flush=True)
            print(f"📊 Mejor Accuracy alcanzada: {best_acc * 100.0:.2f}%", flush=True)
            print(f"🎯 Costo retornado a irace (1.0 - acc): {cost:.6f}", flush=True)

    except Exception as e:
        # En caso de error, capturar traceback en el log y devolver costo penalizado 1.0
        sys.stdout = original_stdout
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n❌ EXCEPCIÓN DURANTE EJECUCIÓN:\n{traceback.format_exc()}\n")
        # Imprimir únicamente el costo penalizado para irace
        print("1.000000")
        sys.exit(0)
    finally:
        # Restaurar stdout y stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Limpiar checkpoints temporales para ahorrar espacio en disco
        try:
            for item in chck_dir.glob("*"):
                if item.is_file():
                    item.unlink()
            if chck_dir.exists():
                chck_dir.rmdir()
        except Exception:
            pass

    # IMPORTANTE: La ÚNICA salida por STDOUT debe ser el número flotante del costo para irace
    print(f"{cost:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
