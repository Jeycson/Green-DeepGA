# -*- coding: utf-8 -*-
"""
Ejemplo: Cómo utilizar DeepGA V10 para optimizar MODELOS NO-CNN.
Muestra dos mecanismos concretos:
  1. Mecanismo PyTorch: Inyectar una arquitectura personalizada (ej. Red Densa / MLP para datos tabulares).
  2. Mecanismo Black-Box / Scikit-Learn: Inyectar cualquier modelo o función con CustomEvaluator.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deepga import DeepGAConfig, DeepGASearch, CustomEvaluator, ModelEvaluator


# ==============================================================================
# CASO 1: Optimizar una Arquitectura PyTorch NO-CNN (ej. MLP para Tabular)
# ==============================================================================
def demo_modelo_pytorch_personalizado():
    print("\n" + "=" * 65)
    print("🔹 CASO 1: Modelo PyTorch Personalizado (MLP en lugar de CNN)")
    print("=" * 65)

    # 1. Crear datos sintéticos tabulares (1000 muestras, 20 variables, 2 clases)
    X = torch.randn(1000, 20)
    y = torch.randint(0, 2, (1000,))
    train_ds = TensorDataset(X[:800], y[:800])
    val_ds = TensorDataset(X[800:], y[800:])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. El desarrollador define su función constructora de modelo:
    #    Recibe el genoma 'e' y retorna su propia arquitectura nn.Module
    def mi_mlp_builder(genome) -> nn.Module:
        # Usa las capas densas determinadas por el algoritmo genético
        layers = []
        in_features = 20  # dimensiones de entrada

        # Extraer neuronas sugeridas por los genes
        for i in range(getattr(genome, 'n_full', 2)):
            neurons = 32 * (i + 1)
            layers.append(nn.Linear(in_features, neurons))
            layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_features = neurons

        layers.append(nn.Linear(in_features, 2))  # capa de salida
        return nn.Sequential(*layers)

    # 3. Configuración óptima de DeepGA V10 (calibrada por irace)
    #    Usamos expert_mode=True si queremos reducir generaciones para la demo rápida
    config = DeepGAConfig.custom(
        pop_size=6,
        generations=2,
        train_epochs=1,
        final_epochs=1,
        batch_size=32
    )

    searcher = DeepGASearch(config=config)

    # 4. Ajustar pasando los DataLoaders y la función 'model_builder'
    searcher.fit(
        train_dl=train_dl,
        val_dl=val_dl,
        model_builder=mi_mlp_builder
    )

    # 5. Obtener el modelo óptimo MLP resultante
    mejor_mlp = searcher.get_best_model()
    print("\n✅ Mejor MLP obtenido:")
    print(mejor_mlp)


# ==============================================================================
# CASO 2: Optimizar Modelos de Scikit-Learn o Funciones Arbitrarias
# ==============================================================================
def demo_modelo_sklearn_arbitrario():
    print("\n" + "=" * 65)
    print("🔹 CASO 2: Optimizar Modelos de Machine Learning (Scikit-Learn)")
    print("=" * 65)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # El desarrollador define una función de evaluación arbitraria
    def evaluar_random_forest(genome):
        # Mapea los genes del individuo a hiperparámetros de RandomForest
        n_trees = int(getattr(genome, 'n_conv', 2) * 20 + 10)
        max_depth = int(getattr(genome, 'n_full', 1) * 3 + 2)

        clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_val, y_val)

        # Retorna: (fitness, metric_principal, complejidad)
        return float(accuracy), float(accuracy), n_trees

    custom_eval = CustomEvaluator(eval_fn=evaluar_random_forest)

    config = DeepGAConfig.custom(pop_size=6, generations=2)
    searcher = DeepGASearch(config=config)

    # Iniciar búsqueda con el evaluador personalizado
    searcher.fit(evaluator=custom_eval)

    best_genome = searcher.get_best_genome()
    print(f"\n✅ Mejor configuración encontrada: n_conv={best_genome.n_conv}, n_full={best_genome.n_full}")


if __name__ == "__main__":
    demo_modelo_pytorch_personalizado()
    demo_modelo_sklearn_arbitrario()
