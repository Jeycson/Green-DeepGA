# -*- coding: utf-8 -*-
"""
Script de Evaluación y Re-entrenamiento Final de la Mejor Configuración de DeepGA.
Utiliza la configuración óptima descubierta por irace para:
1. Ejecutar la neuroevolución con los hiperparámetros ganadores.
2. Entrenar completamente la red neuronal encontrada durante varias épocas finales.
3. Evaluar el modelo sobre el conjunto de prueba (Test Set).
4. Generar Matriz de Confusión, métricas detalladas (Accuracy, Precision, Recall, F1)
   y guardar el modelo entrenado (.pth).
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Añadir el directorio raíz de DeepGA
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from experiment_manager import ExperimentManager
from runner_deepga import resolve_dataset_path, set_all_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación y Entrenamiento Final de Mejor Configuración DeepGA")

    # Opción para cargar configuración desde archivo JSON
    parser.add_argument("--config-json", type=str, default=None,
                        help="Ruta a archivo JSON con la configuración (ej: best_configuration.json)")

    # Dataset a evaluar
    parser.add_argument("--dataset", type=str, default="Tumour_3",
                        help="Nombre o ruta del dataset a evaluar (Tumour o Tumour_3)")
    parser.add_argument("--final-epochs", type=int, default=25,
                        help="Número de épocas de entrenamiento completo para el modelo ganador")
    parser.add_argument("--execution", type=int, default=1,
                        help="Identificador de la ejecución")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--output-dir", type=str, default="./best_model_results",
                        help="Directorio de guardado para checkpoints, modelos y gráficos")

    # Hiperparámetros (pueden pasarse directamente por CLI o cargarse vía JSON)
    parser.add_argument("--variant", type=str, default="v12", choices=["v10", "v11", "v12"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w", type=float, default=0.05)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=14)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--t-size", type=int, default=3)
    parser.add_argument("--cr", type=float, default=0.75)
    parser.add_argument("--mr", type=float, default=0.45)
    parser.add_argument("--mr-min", type=float, default=0.10)
    parser.add_argument("--mr-max", type=float, default=0.85)

    parser.add_argument("--min-conv", type=int, default=2)
    parser.add_argument("--max-conv", type=int, default=5)
    parser.add_argument("--min-full", type=int, default=1)
    parser.add_argument("--max-full", type=int, default=3)
    parser.add_argument("--max-params", type=int, default=2000000)

    parser.add_argument("--pool-candidates-factor", type=int, default=4)
    parser.add_argument("--kappa", type=float, default=0.15)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--top-k-ratio", type=float, default=0.35)

    parser.add_argument("--n-islands", type=int, default=3)
    parser.add_argument("--migration-interval", type=int, default=8)
    parser.add_argument("--migration-size", type=int, default=1)
    parser.add_argument("--target-diversity", type=float, default=0.25)
    parser.add_argument("--stagnation-limit", type=int, default=4)

    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--use-amp", action="store_true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # Si se especificó un JSON de configuración, sobrescribir parámetros
    if args.config_json and os.path.exists(args.config_json):
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            for k, v in cfg.items():
                attr = k.replace("-", "_")
                if hasattr(args, attr):
                    setattr(args, attr, v)
        print(f"📖 Configuración cargada desde: {args.config_json}")

    set_all_seeds(args.seed)

    # Localizar dataset
    data_dir = resolve_dataset_path(args.dataset)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No se encontró el dataset en '{data_dir}' (dataset: {args.dataset})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("   EVALUACIÓN FINAL DEEPGA CON CONFIGURACIÓN ÓPTIMA IRACE")
    print("=" * 65)
    print(f"📌 Dispositivo:        {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU)"))
    print(f"📌 Variante:           {args.variant.upper()}")
    print(f"📌 Dataset:            {data_dir}")
    print(f"📌 Hiperparámetros:    lr={args.lr}, w={args.w}, epochs_ga={args.train_epochs}, pop={args.pop_size}, gen={args.generations}, batch={args.batch_size}")
    print(f"📌 Épocas finales:     {args.final_epochs}")
    print(f"📌 Carpeta de salida:  {out_dir.resolve()}")
    print("=" * 65 + "\n")

    manager = ExperimentManager(
        country_iso_code="MEX",
        track_carbon=True
    )

    resultados = manager.run_deepga(
        variant=args.variant,
        execution=args.execution,
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
        data_root=data_dir,
        img_size=args.img_size,
        in_channels=args.in_channels,
        chck_dir=str(out_dir) + "/",
        device=device,
        save_best_model_file=True,
        save_txt_report=True,
        train_final_model=True,
        final_train_epochs=args.final_epochs,
        auto_download=False
    )

    saved_model = resultados.get("saved_model_path")
    class_names = resultados.get("class_names", ["BENIGN", "MALIGNANT", "NORMAL"])

    # Comparación con baseline.csv
    baseline_csv = CURRENT_DIR / "baseline.csv"
    from runner_deepga import load_baseline_metrics, get_dataset_baseline, calculate_normalized_cost
    b_map = load_baseline_metrics(baseline_csv)
    b_info = get_dataset_baseline(args.dataset, b_map)

    f1_val = resultados.get("f1", resultados.get("best_accuracy", 0.0))
    if f1_val is not None and f1_val <= 1.0 and f1_val > 0.0:
        f1_val = f1_val * 100.0
    energy_val = resultados.get("energy_consumed_kwh", 0.0)
    norm_cost, details = calculate_normalized_cost(f1_val, energy_val, b_info["macro_f1"], b_info["energy_kwh"])

    print("\n" + "=" * 65)
    print("🎉 RESULTADOS FINALES DE LA ARQUITECTURA GANADORA")
    print("=" * 65)
    print(f"🏆 Mejor Accuracy en Validación (GA): {resultados.get('best_accuracy', 0.0) * 100:.2f}%")
    if resultados.get("final_test_accuracy") is not None:
        print(f"🎯 Accuracy en Test Set Independiente: {resultados.get('final_test_accuracy') * 100:.2f}%")
    print(f"📊 Macro F1 alcanzado:                {f1_val:.3f}% (Baseline: {b_info['macro_f1']:.3f}%)")
    print(f"⚡ Consumo de Energía:                 {energy_val:.6f} kWh (Baseline: {b_info['energy_kwh']:.6f} kWh)")
    print(f"🎯 Costo Multi-Objetivo Normalizado:  {norm_cost:.6f} (Baseline = 1.000000)")
    print(f"📦 Parámetros totales de la red:      {resultados.get('best_total_params', 0):,}")
    print(f"💾 Tamaño estimado del modelo:        {resultados.get('best_model_size_mb', 0.0):.2f} MB")
    print(f"⚡ FLOPs estimados:                   {resultados.get('best_estimated_flops', 0):,}")
    print(f"💾 Modelo guardado en:                 {saved_model}")
    print("=" * 65 + "\n")

    # Generar matriz de confusión sobre el test set si está disponible
    if resultados.get("test_dataloader") is not None and saved_model:
        print("📊 Generando Matriz de Confusión sobre el Test Set...")
        cm_path = out_dir / f"matriz_confusion_final_{args.variant}_{args.dataset}.png"
        manager.generate_confusion_matrix(
            model_or_path=saved_model,
            dataloader=resultados["test_dataloader"],
            class_names=class_names,
            title=f"Matriz de Confusión - {args.variant.upper()} ({args.dataset})",
            save_fig_path=str(cm_path),
            auto_download_plot=False
        )
        print(f"📊 Gráfico guardado en: {cm_path.resolve()}")


if __name__ == "__main__":
    main()
