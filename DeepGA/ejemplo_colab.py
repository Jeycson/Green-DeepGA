# -*- coding: utf-8 -*-
"""
Ejemplo de ejecución en Google Colab:
Pruebas de variantes de DeepGA en CIFAR-10 con medición de huella de carbono,
tiempos de ejecución y variables de las redes convolucionales generadas.
"""

from experiment_manager import ExperimentManager

# 1. Instanciar el gestor
manager = ExperimentManager(country_iso_code="MEX", track_carbon=True)

# 2. Ejecutar la neuroevolución con los parámetros que desees probar
resultados = manager.run_deepga(
    execution=1,
    population_size=10,    # N: Tamaño de población
    generations=5,         # T: Número de generaciones
    train_epochs=2,        # Épocas de entrenamiento por CNN
    lr=1e-4,               # Tasa de aprendizaje
    w=0.3,                 # Peso de penalización de parámetros
    batch_size=64,
    chck_dir="./checkpoints/"
)

# 3. Acceder a las métricas del experimento
print("\nAcceso a variables individuales:")
print("Huella de Carbono (gCO2eq):", resultados["carbon_emissions_g_co2"])
print("Energía Consumida (kWh):", resultados["energy_consumed_kwh"])
print("Tiempo de Ejecución (segundos):", resultados["execution_time_seconds"])
print("Parámetros Totales de la CNN:", resultados["best_total_params"])
print("FLOPs Estimados de la CNN:", resultados["best_estimated_flops"])
print("Tamaño en Memoria RAM (MB):", resultados["best_model_size_mb"])
print("Capas Convolucionales:", resultados["conv_layers_count"])
print("Capas Densas:", resultados["fc_layers_count"])

# 4. Historial generacional (DataFrame de Pandas que ya genera tu DeepGA original)
df_historial = resultados["history_dataframe"]
print("\nHistorial Generacional:")
print(df_historial)
