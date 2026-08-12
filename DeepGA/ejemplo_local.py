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
    parser.add_argument("--variant", type=str, default="v9", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"],
                        help="Variante de DeepGA a ejecutar (por defecto: v9)")
    parser.add_argument("--execution", type=int, default=1,
                        help="Número identificador de la ejecución (por defecto: 1)")
    parser.add_argument("--pop-size", type=int, default=10,
                        help="Tamaño de la población N (por defecto: 10)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Número de generaciones T (por defecto: 5)")
    parser.add_argument("--train-epochs", type=int, default=2,
                        help="Épocas de entrenamiento por individuo en el GA (por defecto: 2)")
    parser.add_argument("--final-epochs", type=int, default=10,
                        help="Épocas de entrenamiento final para el modelo ganador (por defecto: 10)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño del lote / batch size (por defecto: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (por defecto: 1e-4)")
    parser.add_argument("--w", type=float, default=0.3,
                        help="Peso de penalización por parámetros (por defecto: 0.3)")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Ruta local donde se almacena o descargará CIFAR-10 (por defecto: ./data)")
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/",
                        help="Directorio donde se guardarán los modelos y gráficos (por defecto: ./checkpoints/)")
    parser.add_argument("--country-iso", type=str, default="MEX",
                        help="Código ISO del país para tracking de huella de carbono (por defecto: MEX)")
    parser.add_argument("--track-carbon", action="store_true", default=True,
                        help="Activa la medición de huella de carbono (por defecto: True)")
    parser.add_argument("--no-preload-gpu", action="store_true", default=False,
                        help="Desactiva la precarga completa en VRAM si la GPU tiene memoria muy limitada")
    return parser.parse_args()


def main():
    args = parse_args()

    # Detección de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("      EJECUCIÓN LOCAL DE DEEPGA / EXPERIMENT MANAGER")
    print("=" * 60)
    print(f"📌 Dispositivo detectado:    {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU)"))
    print(f"📌 Ruta del dataset CIFAR-10: {os.path.abspath(args.data_root)}")
    print(f"📌 Directorio de checkpoints: {os.path.abspath(args.chck_dir)}")
    print(f"📌 Variante seleccionada:     {args.variant.upper()}")
    print("=" * 60 + "\n")

    # 1. Instanciar el gestor de experimentos
    manager = ExperimentManager(
        country_iso_code=args.country_iso,
        track_carbon=args.track_carbon
    )

    # 2. Ejecutar la neuroevolución con entrenamiento y guardado del mejor modelo
    #    (Si CIFAR-10 no existe en data_root, se descargará automáticamente la primera vez)
    resultados = manager.run_deepga(
        variant=args.variant,
        execution=args.execution,
        population_size=args.pop_size,
        generations=args.generations,
        train_epochs=args.train_epochs,
        lr=args.lr,
        w=args.w,
        batch_size=args.batch_size,
        data_root=args.data_root,
        chck_dir=args.chck_dir,
        preload_gpu=not args.no_preload_gpu,
        save_best_model_file=True, # Guarda automáticamente best_model_{variant}_exec_{execution}.pth y .pkl
        train_final_model=True,    # Entrena el ganador para tener los pesos listos
        final_train_epochs=args.final_epochs,
        auto_download=False        # False en local (los archivos quedan guardados en chck_dir)
    )

    # 3. Acceder al modelo guardado
    ruta_modelo = resultados["saved_model_path"]
    print(f"\n✅ Modelo ganador guardado en: {ruta_modelo}")

    # 4. Generar Matriz de Confusión y Reporte sobre el Test Set independiente (10,000 imágenes)
    #    (Estas 10,000 imágenes NUNCA fueron vistas durante la neuroevolución ni durante la validación)
    print("\n📊 Generando Matriz de Confusión sobre el Test Set independiente (10,000 imágenes)...")
    save_fig_path = os.path.join(args.chck_dir, f"matriz_confusion_{args.variant.lower()}.png")
    cm, reporte = manager.generate_confusion_matrix(
        model_or_path=ruta_modelo,
        dataloader=resultados["test_dataloader"],  # 10,000 imágenes oficiales de prueba
        class_names=CLASS_NAMES,
        title=f"Matriz de Confusión (Test Set 10k) - DeepGA {resultados['variant']}",
        save_fig_path=save_fig_path,
        auto_download_plot=False
    )
    print(f"📊 Gráfico de Matriz de Confusión guardado en: {os.path.abspath(save_fig_path)}")

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
