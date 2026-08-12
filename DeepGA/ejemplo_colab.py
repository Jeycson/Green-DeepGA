# -*- coding: utf-8 -*-
"""
Ejemplo de ejecución en Google Colab / Local:
1. Neuroevolución con guardado automático del mejor modelo por versión (.pth y .pkl).
2. Entrenamiento final del modelo ganador y descarga automática a tu navegador.
3. Inferencia con imágenes propias.
4. Matriz de confusión y reporte de clasificación (Precision, Recall, F1).
5. Descarga de todos los modelos en un archivo ZIP.
"""

import torch
from experiment_manager import ExperimentManager

# Clases de CIFAR-10 (o cámbialas por las de tu dataset)
CLASS_NAMES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# 1. Instanciar el gestor
manager = ExperimentManager(country_iso_code="MEX", track_carbon=True)

# 2. Ejecutar la neuroevolución guardando y entrenando el mejor modelo
#    (Nota: Pon auto_download=True si deseas que se descargue automáticamente en Colab)
resultados = manager.run_deepga(
    variant="v10",             # "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9" o "v10"
    execution=1,
    population_size=10,        # N: Tamaño de población
    generations=5,             # T: Número de generaciones
    train_epochs=2,            # Épocas durante el GA
    lr=1e-4,                   # Tasa de aprendizaje
    w=0.3,                     # Peso de penalización
    batch_size=64,
    chck_dir="./checkpoints/",
    save_best_model_file=True, # Guarda automáticamente best_model_v9_exec_1.pth
    train_final_model=True,    # Entrena el ganador para tener los pesos listos
    final_train_epochs=10,     # Épocas de entrenamiento final (ej. 10 a 30)
    auto_download=True        # Activa True en Colab para descargar directamente
)

# 3. Acceder al modelo guardado
ruta_modelo = resultados["saved_model_path"]
print(f"\n✅ Modelo guardado en: {ruta_modelo}")

# 4. Generar Matriz de Confusión y Reporte sobre el Test Set independiente (10,000 imágenes)
#    (Estas 10,000 imágenes NUNCA fueron vistas durante la neuroevolución ni durante la validación)
print("\n📊 Generando Matriz de Confusión sobre el Test Set independiente (10,000 imágenes)...")
cm, reporte = manager.generate_confusion_matrix(
    model_or_path=ruta_modelo,
    dataloader=resultados["test_dataloader"],  # 10,000 imágenes oficiales de prueba
    class_names=CLASS_NAMES,
    title=f"Matriz de Confusión (Test Set 10k) - DeepGA {resultados['variant']}",
    save_fig_path="./checkpoints/matriz_confusion_v9.png",
    auto_download_plot=False
)

# 5. Inferencia con tus propias imágenes
# Ejemplo:
# pred_clase, confianza, probabilidades = manager.predict_image(
#     model_or_path=ruta_modelo,
#     image_path="mi_imagen.jpg",
#     class_names=CLASS_NAMES
# )
# print(f"\n🔮 Predicción de tu imagen: {pred_clase} ({confianza:.2f}% de certeza)")

# 6. Descargar el modelo o empaquetar todos los modelos ganadores en un ZIP
# manager.download_model(variant="v9", execution=1)
# manager.download_all_models(chck_dir="./checkpoints/", zip_name="modelos_deepga.zip")
