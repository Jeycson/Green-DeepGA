# -*- coding: utf-8 -*-
"""
Ejemplo de uso de la librería DeepGA Versión 10:
Demuestra cómo utilizar la nueva API por código para:
1. Configurar hiperparámetros tipados con DeepGAConfig.
2. Indicar un dataset (carpeta de imágenes o 'cifar10').
3. Ejecutar la búsqueda evolutiva V10 (ACO-Enhanced).
4. Obtener y guardar el mejor modelo PyTorch resultante.
"""

from deepga import DeepGAConfig, DeepGASearch


def main():
    print("=" * 65)
    print("🧬 DeepGA V10: Demostración de la API de Librería")
    print("=" * 65)

    # 1. Definición de parámetros mediante código
    config = DeepGAConfig(
        pop_size=4,                  # Población pequeña para prueba rápida
        generations=2,               # 2 generaciones de demostración
        train_epochs=1,              # 1 época por arquitectura
        final_epochs=1,              # 1 época de re-entrenamiento final
        batch_size=32,
        learning_rate=0.001,
        alpha=1.2,                   # Peso de feromonas V10
        rho=0.10,                    # Tasa de evaporación V10
        seed=42,
        output_dir="./results_v10_demo"
    )

    # 2. Instanciación del optimizador
    searcher = DeepGASearch(config=config)

    # 3. Ejecución de la búsqueda
    # Puedes pasar 'cifar10' o la ruta a tu propia carpeta de imágenes:
    # searcher.fit(dataset="/ruta/a/mi/dataset")
    print("Iniciando búsqueda con dataset...")
    searcher.fit(dataset="cifar10")

    # 4. Inspección del mejor modelo
    best_model = searcher.get_best_model()
    print("\n🏆 Mejor arquitectura PyTorch encontrada:")
    print(best_model)

    # 5. Guardado del modelo
    save_path = "./results_v10_demo/mejor_modelo_v10.pt"
    searcher.save(save_path)
    print(f"\n✅ Modelo y pesos guardados en: {save_path}")


if __name__ == "__main__":
    main()
