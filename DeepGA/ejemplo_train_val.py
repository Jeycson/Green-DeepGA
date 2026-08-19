# -*- coding: utf-8 -*-
"""
Ejemplo de Ejecución de DeepGA con Partición en 2 Conjuntos (Train y Validación):
---------------------------------------------------------------------------------
Este script entrena y valida redes neuronales evolucionadas con DeepGA utilizando
ÚNICAMENTE 2 particiones del dataset (Train y Validation), sin separar conjunto de test.

Objetivo:
- Maximizar la cantidad de imágenes disponibles para el entrenamiento (ej. 85%-90% Train / 10%-15% Val,
  o en CIFAR-10 los 50,000 datos completos de train y 10,000 de validación).
- Monitorear métricas de entrenamiento y validación exclusivamente:
  * Precisión de Validación alcanzada (%)
  * Huella de Carbono (gCO2eq) y Consumo de Energía (kWh)
  * Tiempo de ejecución y cómputo GPU
  * Parámetros de la CNN, FLOPs y tamaño en memoria
  * Matriz de Confusión calculada sobre el conjunto de VALIDACIÓN
  * Para variantes Multi-Objetivo: Frente de Pareto y soluciones no dominadas

Uso rápido:
    python ejemplo_train_val.py

Uso personalizado (CLI):
    python ejemplo_train_val.py --variant mo_v11 --generations 5 --pop-size 12 --train-epochs 2 --val-ratio 0.15 --data-root ./data
"""

import os
import argparse

try:
    import torch
    from experiment_manager import ExperimentManager
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecución de DeepGA con Partición en 2 Conjuntos (Train y Validación Solamente)"
    )
    parser.add_argument("--variant", type=str, default="v12",
                        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "mo_v9", "mo_v10", "mo_v11"],
                        help="Variante de DeepGA a ejecutar (por defecto: v12)")
    parser.add_argument("--execution", type=int, default=1,
                        help="Número identificador de la ejecución (por defecto: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla aleatoria / ID de ejecución (por defecto: igual a --execution)")
    parser.add_argument("--pop-size", type=int, default=12,
                        help="Tamaño de la población N total (por defecto: 12)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Número de generaciones T (por defecto: 5)")
    parser.add_argument("--train-epochs", type=int, default=2,
                        help="Épocas de entrenamiento por individuo en el GA (por defecto: 2)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño del lote / batch size (por defecto: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (por defecto: 1e-4)")
    parser.add_argument("--w", type=float, default=0.3,
                        help="Peso de penalización por parámetros para variantes mono-objetivo (por defecto: 0.3)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Proporción del dataset para Validación en el split de 2 (por defecto: 0.15 -> 85%% Train / 15%% Val)")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Ruta al dataset (ej. ./data o ./dataset/covid)")
    parser.add_argument("--img-size", type=int, default=32,
                        help="Resolución de las imágenes en píxeles (por defecto: 32)")
    parser.add_argument("--in-channels", type=int, default=3, choices=[1, 3],
                        help="Canales de entrada: 3 para RGB o 1 para escala de grises")
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/",
                        help="Directorio de checkpoints y reportes (por defecto: ./checkpoints/)")
    parser.add_argument("--country-iso", type=str, default="MEX",
                        help="Código ISO del país para intensidad de huella de carbono (por defecto: MEX)")
    parser.add_argument("--track-carbon", action="store_true", default=True,
                        help="Activa medición de consumo energético y huella de carbono")
    parser.add_argument("--no-preload-gpu", action="store_true", default=False,
                        help="Desactiva la precarga completa en VRAM")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_multiobjective = args.variant.lower().startswith("mo")

    print("\n" + "=" * 70, flush=True)
    print("       DEEPGA - ENTRENAMIENTO Y VALIDACIÓN (PARTICIÓN 2-SPLIT)", flush=True)
    print("        (Más datos asignados a Train | Sin partición de Test)", flush=True)
    print("=" * 70, flush=True)
    print(f"📌 Dispositivo:               {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU)"), flush=True)
    print(f"📌 Variante Seleccionada:     {args.variant.upper()}", flush=True)
    print(f"📌 Tipo de Optimización:      {'Multi-Objetivo (Precisión vs Carbono)' if is_multiobjective else 'Mono-Objetivo (Fitness ponderado)'}", flush=True)
    print(f"📌 Dataset Origen:            {os.path.abspath(args.data_root)}", flush=True)
    print(f"📌 Partición 2-Split:         {round((1.0 - args.val_ratio) * 100)}% Entrenamiento | {round(args.val_ratio * 100)}% Validación", flush=True)
    print(f"📌 Resolución / Canales:      {args.img_size}x{args.img_size} px | {args.in_channels} canal(es)", flush=True)
    print(f"📌 Población / Generaciones:  N={args.pop_size} individuos | T={args.generations} generaciones", flush=True)
    print(f"📌 Épocas por individuo:      {args.train_epochs} épocas", flush=True)
    print(f"📌 Checkpoints y Reportes:    {os.path.abspath(args.chck_dir)}", flush=True)
    print("=" * 70 + "\n", flush=True)

    actual_seed = args.seed if args.seed is not None else args.execution
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    # 1. Instanciar el Gestor de Experimentos
    manager = ExperimentManager(
        country_iso_code=args.country_iso,
        track_carbon=args.track_carbon
    )

    # 2. Ejecutar Neuroevolución con Partición 2-Split
    # Se utiliza use_2split=True para entregar el máximo de imágenes a Train y evaluar en Val
    resultados = manager.run_deepga(
        variant=args.variant,
        execution=actual_seed,
        population_size=args.pop_size,
        generations=args.generations,
        train_epochs=args.train_epochs,
        lr=args.lr,
        w=args.w,
        batch_size=args.batch_size,
        data_root=args.data_root,
        img_size=args.img_size,
        in_channels=args.in_channels,
        chck_dir=args.chck_dir,
        preload_gpu=not args.no_preload_gpu,
        save_best_model_file=True,
        save_txt_report=True,
        train_final_model=False,
        use_2split=True,
        val_ratio=args.val_ratio
    )

    val_loader = resultados.get("val_dataloader")
    class_names = resultados.get("class_names", [f"Clase_{i}" for i in range(10)])

    # 3. Mostrar Resumen Exclusivo de Métricas de Entrenamiento y Validación
    print("\n" + "=" * 70, flush=True)
    print("       📊 MÉTRICAS DE ENTRENAMIENTO Y VALIDACIÓN OBTENIDAS", flush=True)
    print("=" * 70, flush=True)
    print(f"🎯 Precisión en Validación:        {resultados['best_val_accuracy']:.2f}%")
    if not is_multiobjective:
        print(f"🏆 Mejor Fitness Obtenido:         {resultados['best_fitness']:.4f}")
    print(f"⏱️ Tiempo Total de Búsqueda:       {resultados['execution_time_seconds']:.2f} s ({resultados['execution_time_minutes']:.2f} min)")
    print(f"⚡ Consumo de Energía Eléctrica:   {resultados['energy_consumed_kwh']:.6f} kWh")
    print(f"🌿 Huella de Carbono Total:        {resultados['carbon_emissions_g_co2']:.4f} gCO2eq")
    print(f"🧠 Parámetros Totales de la Red:   {resultados['best_total_params']:,}")
    print(f"🧠 Parámetros Entrenables:         {resultados['best_trainable_params']:,}")
    print(f"💾 Tamaño Estimado del Modelo:     {resultados['best_model_size_mb']:.2f} MB")
    print(f"🔢 FLOPs Estimados por Muestra:    {resultados['best_estimated_flops']:,}")
    print(f"🏗️ Arquitectura Convolucional:     {resultados['conv_layers_count']} capas Conv | {resultados['fc_layers_count']} capas FC | {resultados['skip_connections_count']} skips")

    # 4. Sección especial si es Multi-Objetivo
    if is_multiobjective and "mo_stats" in resultados:
        mo = resultados["mo_stats"]
        print("\n" + "-" * 70, flush=True)
        print("        🌐 DETALLE DEL FRENTE DE PARETO (F1 - SOLUCIONES NO DOMINADAS)", flush=True)
        print("-" * 70, flush=True)
        print(f"📐 Tamaño del Frente de Pareto:    {mo.get('pareto_front_size', len(resultados.get('pareto_front', [])))} arquitecturas")
        print(f"📈 Hipervolumen 2D (HV):           {mo.get('final_hypervolume', 0.0):.2f}")

        b_acc = mo.get("best_accuracy_individual")
        b_grn = mo.get("greenest_individual")
        b_kne = mo.get("knee_point_individual")

        if b_acc:
            print(f"🏆 Modelo Máxima Precisión (Val):  Acc={b_acc[1]:.2f}% | Carbono={b_acc[2]:.4f} gCO2eq | Params={b_acc[3]['total_params']:,}")
        if b_grn:
            print(f"🌿 Modelo Más Verde (Ultra-Low):   Acc={b_grn[1]:.2f}% | Carbono={b_grn[2]:.4f} gCO2eq | Params={b_grn[3]['total_params']:,}")
        if b_kne:
            print(f"⚖️ Modelo Equilibrado (Knee Point): Acc={b_kne[1]:.2f}% | Carbono={b_kne[2]:.4f} gCO2eq | Params={b_kne[3]['total_params']:,}")

        # Generar gráfico del Frente de Pareto
        pareto_img = os.path.join(args.chck_dir, f"pareto_front_{args.variant.lower()}_exec_{args.execution}_2split.png")
        try:
            manager.generate_pareto_front_plot(
                metrics_summary=resultados,
                title=f"Frente de Pareto (Train/Val 2-Split) - {args.variant.upper()}",
                save_fig_path=pareto_img
            )
            print(f"\n📈 Gráfico del Frente de Pareto guardado en: {os.path.abspath(pareto_img)}", flush=True)
        except Exception as e:
            print(f"Nota sobre gráfico de Pareto: {e}")

    # 5. Generar Matriz de Confusión EXCLUSIVAMENTE sobre el Conjunto de VALIDACIÓN
    saved_model_path = resultados.get("saved_model_path")
    if saved_model_path and os.path.exists(saved_model_path) and val_loader is not None:
        print("\n" + "-" * 70, flush=True)
        print("       🔍 MATRIZ DE CONFUSIÓN SOBRE EL CONJUNTO DE VALIDACIÓN", flush=True)
        print("-" * 70, flush=True)
        cm_path = os.path.join(args.chck_dir, f"confusion_matrix_val_{args.variant.lower()}_exec_{args.execution}.png")
        try:
            manager.generate_confusion_matrix(
                model_or_path=saved_model_path,
                dataloader=val_loader,
                class_names=class_names,
                device=device,
                title=f"Matriz de Confusión (Validación) - DeepGA {args.variant.upper()}",
                save_fig_path=cm_path
            )
            print(f"📊 Matriz de Confusión (Validación) guardada en: {os.path.abspath(cm_path)}", flush=True)
        except Exception as e:
            print(f"Nota sobre matriz de confusión de validación: {e}")

    txt_report = resultados.get("txt_report_path")
    if txt_report:
        print(f"\n📄 Reporte de texto completo guardado en: {os.path.abspath(txt_report)}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("✨ Ejecución completada exitosamente sin conjunto de prueba independiente.", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
