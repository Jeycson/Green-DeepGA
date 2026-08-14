# -*- coding: utf-8 -*-
"""
Ejemplo de Ejecución Multi-Objetivo (Green MO-DeepGA V9):
1. Optimización Bi-Objetivo simultánea: Maximizar Precisión vs Minimizar Huella de Carbono (gCO2eq).
2. Asistencia por Meta-Modelo Subrogado Multi-Objetivo (Dual Random Forest + MO-UCB/LCB en CPU).
3. Búsqueda Evolutiva NSGA-II con Elitismo, Rango de Pareto y Distancia de Apiñamiento.
4. Generación automática del Gráfico 2D del Frente de Pareto (Accuracy vs gCO2eq).
5. Guardado automático de los 3 modelos clave del Frente de Pareto:
   - 🏆 Máxima Precisión (Best Accuracy)
   - 🌿 Más Verde / Ultra-Low Carbon (Greenest)
   - ⚖️ Compromiso Óptimo (Knee Point)
6. Evaluación de Matriz de Confusión sobre el Test Set (10,000 imágenes).

Uso rápido:
    python ejemplo_mo.py

Uso personalizado (CLI):
    python ejemplo_mo.py --variant mo_v9 --generations 5 --pop-size 12 --train-epochs 2 --data-root ./data
"""

import os
import argparse
import torch
from experiment_manager import ExperimentManager

# Clases oficiales de CIFAR-10
CLASS_NAMES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecución Multi-Objetivo de DeepGA (Precisión vs Huella de Carbono)")
    parser.add_argument("--variant", type=str, default="mo_v11", choices=["mo_v9", "mo_v10", "mo_v11"],
                        help="Variante Multi-Objetivo a ejecutar (por defecto: mo_v11)")
    parser.add_argument("--execution", type=int, default=1,
                        help="Número identificador de la ejecución (por defecto: 1)")
    parser.add_argument("--pop-size", type=int, default=12,
                        help="Tamaño de la población N (por defecto: 12)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Número de generaciones T (por defecto: 5)")
    parser.add_argument("--train-epochs", type=int, default=2,
                        help="Épocas de entrenamiento por individuo en el GA (por defecto: 2)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño del batch (por defecto: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (por defecto: 1e-4)")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Ruta local donde se almacena CIFAR-10 o dataset personalizado")
    parser.add_argument("--img-size", type=int, default=32,
                        help="Resolución de las imágenes (por defecto: 32)")
    parser.add_argument("--in-channels", type=int, default=3, choices=[1, 3],
                        help="Número de canales de entrada (3 para RGB, 1 para Grayscale)")
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/",
                        help="Directorio donde se guardarán los modelos y gráficos (por defecto: ./checkpoints/)")
    parser.add_argument("--country-iso", type=str, default="MEX",
                        help="Código ISO del país para cálculo de huella de carbono (por defecto: MEX)")
    parser.add_argument("--track-carbon", action="store_true", default=True,
                        help="Activa la medición de huella de carbono (por defecto: True)")
    parser.add_argument("--no-preload-gpu", action="store_true", default=False,
                        help="Desactiva la precarga completa en VRAM")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65, flush=True)
    print("      GREEN MULTI-OBJECTIVE DEEPGA (MO-DEEPGA V9)", flush=True)
    print("   Optimización Bi-Objetivo: Precisión vs Huella de Carbono", flush=True)
    print("=" * 65, flush=True)
    print(f"📌 Dispositivo:               {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU)"), flush=True)
    print(f"📌 Dataset:                   {os.path.abspath(args.data_root)}", flush=True)
    print(f"📌 Resolución / Canales:      {args.img_size}x{args.img_size} | {args.in_channels} canales", flush=True)
    print(f"📌 Directorio de Checkpoints: {os.path.abspath(args.chck_dir)}", flush=True)
    print(f"📌 Variante Multi-Objetivo:   {args.variant.upper()}", flush=True)
    print(f"📌 País / Matriz Energética:  {args.country_iso}", flush=True)
    print("=" * 65 + "\n", flush=True)

    # 1. Instanciar Gestor de Experimentos Multi-Objetivo
    manager = ExperimentManager(
        country_iso_code=args.country_iso,
        track_carbon=args.track_carbon
    )

    # 2. Ejecutar Neuroevolución Multi-Objetivo
    resultados = manager.run_deepga(
        variant=args.variant,
        execution=args.execution,
        population_size=args.pop_size,
        generations=args.generations,
        train_epochs=args.train_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_root=args.data_root,
        img_size=args.img_size,
        in_channels=args.in_channels,
        chck_dir=args.chck_dir,
        preload_gpu=not args.no_preload_gpu,
        save_best_model_file=True,
        save_txt_report=True,
        train_final_model=False
    )

    # 3. Generar y Guardar Gráfica 2D del Frente de Pareto
    print("\n📈 Generando Gráfica del Frente de Pareto (Precisión vs Huella de Carbono)...", flush=True)
    pareto_plot_path = os.path.join(args.chck_dir, f"pareto_front_{args.variant.lower()}_exec_{args.execution}.png")
    manager.generate_pareto_front_plot(
        metrics_summary=resultados,
        title=f"Frente de Pareto MO-DeepGA V9 (Precisión vs gCO2eq)",
        save_fig_path=pareto_plot_path,
        auto_download_plot=False
    )
    print(f"📊 Gráfico del Frente de Pareto guardado en: {os.path.abspath(pareto_plot_path)}", flush=True)

    # 4. Reporte de los Modelos Clave del Frente de Pareto
    mo_stats = resultados.get("mo_stats", {})
    b_acc = mo_stats.get("best_accuracy_individual")
    b_grn = mo_stats.get("greenest_individual")
    b_kne = mo_stats.get("knee_point_individual")

    print("\n" + "=" * 65, flush=True)
    print("        SOLUCIONES CLAVE DEL FRENTE DE PARETO (F1)", flush=True)
    print("=" * 65, flush=True)
    if b_acc:
        print(f"🏆 Modelo de Máxima Precisión:")
        print(f"   - Precisión: {b_acc[1]:.2f}% | Huella CO2: {b_acc[2]:.4f} gCO2eq | Parámetros: {b_acc[3]['total_params']:,} | FLOPs: {b_acc[3]['flops_per_sample']:,}")
    if b_grn:
        print(f"\n🌿 Modelo Más Verde (Ultra-Low Carbon):")
        print(f"   - Precisión: {b_grn[1]:.2f}% | Huella CO2: {b_grn[2]:.4f} gCO2eq | Parámetros: {b_grn[3]['total_params']:,} | FLOPs: {b_grn[3]['flops_per_sample']:,}")
    if b_kne:
        print(f"\n⚖️ Modelo Equilibrado (Knee Point / Compromiso Óptimo):")
        print(f"   - Precisión: {b_kne[1]:.2f}% | Huella CO2: {b_kne[2]:.4f} gCO2eq | Parámetros: {b_kne[3]['total_params']:,} | FLOPs: {b_kne[3]['flops_per_sample']:,}")
    print("=" * 65, flush=True)

    txt_report = resultados.get("txt_report_path")
    if txt_report:
        print(f"\n📄 Reporte de texto completo guardado en: {os.path.abspath(txt_report)}", flush=True)

    print("\n✨ Proceso Multi-Objetivo completado exitosamente.")


if __name__ == "__main__":
    main()
