# -*- coding: utf-8 -*-
"""
Ejemplo de ejecución Local (o servidor local/remoto) con ExperimentManager:
1. Detección automática de hardware (GPU CUDA / CPU).
2. Neuroevolución con guardado automático del mejor modelo por versión (.pth y .pkl).
3. Entrenamiento final del modelo ganador.
4. Matriz de confusión y reporte de clasificación sobre el Test Set independiente (10,000 imágenes).
5. Descarga / guardado automático del dataset CIFAR-10 en la carpeta local del proyecto (./data).
6. Opciones de inferencia y empaquetado de modelos.

Uso rápido:
    python ejemplo_local.py
    
Uso personalizado (CLI opcional):
    python ejemplo_local.py --variant v9 --generations 5 --pop-size 10 --train-epochs 2 --final-epochs 10 --data-root ./data
"""

import os
import argparse
import torch
from experiment_manager import ExperimentManager

# Clases oficiales de CIFAR-10
CLASS_NAMES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecución local de DeepGA con ExperimentManager")
    parser.add_argument("--variant", type=str, default="v12", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "mo_v9", "mo_v10", "mo_v11"],
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
    parser.add_argument("--final-epochs", type=int, default=10,
                        help="Épocas de entrenamiento final para el modelo ganador (por defecto: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamaño del lote / batch size (por defecto: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (por defecto: 1e-4)")
    parser.add_argument("--w", type=float, default=0.3,
                        help="Peso de penalización por parámetros (por defecto: 0.3)")
    parser.add_argument("--n-islands", type=int, default=3,
                        help="Número de islas evolutivas independientes para V11/V12 (por defecto: 3)")
    parser.add_argument("--migration-interval", type=int, default=12,
                        help="Intervalo de generaciones para migración en V11/V12 (por defecto: 12)")
    parser.add_argument("--migration-size", type=int, default=1,
                        help="Número de individuos que migran por isla en V11/V12 (por defecto: 1)")
    parser.add_argument("--target-diversity", type=float, default=0.25,
                        help="Umbral objetivo de diversidad estructural intra-isla (por defecto: 0.25)")
    parser.add_argument("--stagnation-limit", type=int, default=4,
                        help="Límite de generaciones sin mejora antes de anti-estancamiento (por defecto: 4)")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Activa Mixed Precision FP16 para reducir el uso de VRAM a la mitad (por defecto: True)")
    parser.add_argument("--max-spatial-size", type=int, default=4,
                        help="Tope de dimensión espacial antes de Linear para evitar explosión de pesos (por defecto: 4)")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Ruta local donde se almacena CIFAR-10 o dataset personalizado (ej. ./dataset/covid)")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Resolución de las imágenes (por defecto: 64; probado en 128, 256)")
    parser.add_argument("--in-channels", type=int, default=3, choices=[1, 3],
                        help="Número de canales de entrada (3 para RGB, 1 para Grayscale)")
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/",
                        help="Directorio donde se guardarán los modelos y gráficos (por defecto: ./checkpoints/)")
    parser.add_argument("--country-iso", type=str, default="MEX",
                        help="Código ISO del país para tracking de huella de carbono (por defecto: MEX)")
    parser.add_argument("--track-carbon", action="store_true", default=True,
                        help="Activa la medición de huella de carbono (por defecto: True)")
    parser.add_argument("--device", type=str, default=None,
                        help="Dispositivo a utilizar: 'cuda', 'cuda:0', 'cpu' o None (auto-detección)")
    parser.add_argument("--no-preload-gpu", action="store_true", default=False,
                        help="Desactiva la precarga completa en VRAM si la GPU tiene memoria muy limitada")
    return parser.parse_args()


def main():
    args = parse_args()

    # Detección de dispositivo
    if args.device is not None:
        if "cuda" in args.device and not torch.cuda.is_available():
            print(f"⚠️ ADVERTENCIA: Se especificó --device {args.device} pero CUDA no está disponible en este entorno PyTorch. Usando CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60, flush=True)
    print("      EJECUCIÓN LOCAL DE DEEPGA / EXPERIMENT MANAGER", flush=True)
    print("=" * 60, flush=True)
    print(f"📌 Dispositivo detectado:    {device}" + (f" ({torch.cuda.get_device_name(0)})" if (device.type == "cuda" and torch.cuda.is_available()) else " (CPU)"), flush=True)
    print(f"📌 Ruta del dataset:         {os.path.abspath(args.data_root)}", flush=True)
    print(f"📌 Resolución / Canales:     {args.img_size}x{args.img_size} | {args.in_channels} canales", flush=True)
    print(f"📌 Directorio de checkpoints: {os.path.abspath(args.chck_dir)}", flush=True)
    print(f"📌 Variante seleccionada:     {args.variant.upper()}", flush=True)
    print("=" * 60 + "\n", flush=True)

    seed = args.seed if args.seed is not None else args.execution
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Instanciar el gestor de experimentos
    manager = ExperimentManager(
        country_iso_code=args.country_iso,
        track_carbon=args.track_carbon
    )

    # 2. Ejecutar la neuroevolución con entrenamiento y guardado del mejor modelo
    resultados = manager.run_deepga(
        variant=args.variant,
        execution=args.execution,
        seed=seed,
        device=device,
        population_size=args.pop_size,
        generations=args.generations,
        train_epochs=args.train_epochs,
        lr=args.lr,
        w=args.w,
        batch_size=args.batch_size,
        n_islands=args.n_islands,
        migration_interval=args.migration_interval,
        migration_size=args.migration_size,
        data_root=args.data_root,
        img_size=args.img_size,
        in_channels=args.in_channels,
        chck_dir=args.chck_dir,
        preload_gpu=not args.no_preload_gpu,
        save_best_model_file=True, # Guarda automáticamente best_model_{variant}_exec_{execution}.pth y .pkl
        save_txt_report=True,      # Guarda reporte completo de resultados en archivo .txt
        train_final_model=True,    # Entrena el ganador para tener los pesos listos
        final_train_epochs=args.final_epochs,
        auto_download=False        # False en local (los archivos quedan guardados en chck_dir)
    )

    # 3. Acceder al modelo guardado y reporte de texto
    ruta_modelo = resultados["saved_model_path"]
    ruta_txt = resultados.get("txt_report_path")
    ruta_valores = resultados.get("raw_values_path")
    class_names = resultados.get("class_names", CLASS_NAMES)
    print(f"\n✅ Modelo ganador guardado en: {ruta_modelo}", flush=True)
    if ruta_txt:
        print(f"📄 Reporte de experimento (.txt): {os.path.abspath(ruta_txt)}", flush=True)
    if ruta_valores:
        print(f"📋 Archivo SOLO VALORES para Excel (.txt): {os.path.abspath(ruta_valores)}", flush=True)

    # 4. Generar Matriz de Confusión y Reporte sobre el Test Set independiente
    print("\n📊 Generando Matriz de Confusión sobre el Test Set independiente...", flush=True)
    save_fig_path = os.path.join(args.chck_dir, f"matriz_confusion_{args.variant.lower()}_exec_{args.execution}.png")
    cm, reporte = manager.generate_confusion_matrix(
        model_or_path=ruta_modelo,
        dataloader=resultados["test_dataloader"],
        class_names=class_names,
        title=f"Matriz de Confusión - DeepGA {resultados['variant']} (Exec {args.execution})",
        save_fig_path=save_fig_path,
        auto_download_plot=False
    )
    print(f"📊 Gráfico de Matriz de Confusión guardado en: {os.path.abspath(save_fig_path)}", flush=True)

    # 5. Ejemplo de inferencia con una imagen propia (opcional):
    # if os.path.exists("mi_imagen.jpg"):
    #     pred_clase, confianza, probs = manager.predict_image(
    #         model_or_path=ruta_modelo,
    #         image_path="mi_imagen.jpg",
    #         class_names=CLASS_NAMES
    #     )
    #     print(f"\n🔮 Predicción: {pred_clase} ({confianza:.2f}% de certeza)")

    # 6. Empaquetar todos los modelos guardados en un ZIP (opcional):
    # zip_path = manager.download_all_models(chck_dir=args.chck_dir, zip_name="modelos_deepga_local.zip")

    print("\n✨ Proceso completado exitosamente.")


if __name__ == "__main__":
    main()
