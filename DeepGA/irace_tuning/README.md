# 🧬 Optimización Automática de Hiperparámetros de DeepGA con irace

Este paquete contiene la integración completa y lista para usar de **irace (Iterated Racing)** con **DeepGA**, restringido a las tres variantes con mejor rendimiento arquitectónico: **V10**, **V11** y **V12**, sobre los datasets médicos de **Tumours** y **Tumours_3**.

---

## 🎯 ¿Por qué DeepGA tenía problemas para subir el Accuracy y cómo lo soluciona irace?

En los experimentos iniciales, se detectaron cuatro factores clave que frenan el accuracy en datasets médicos:
1. **Penalización de complejidad (`w`) excesiva**: En la función de fitness $f = (1-w)\cdot \text{Acc} + w \cdot \frac{\text{max\_params}-\text{params}}{\text{max\_params}}$, si $w \ge 0.3$, el algoritmo genético prefiere redes casi vacías con bajo accuracy solo porque tienen pocos parámetros. Al calibrar $w \in [0.01, 0.20]$, se obliga a DeepGA a priorizar la precisión diagnóstica.
2. **Épocas de evaluación en el GA (`train_epochs`)**: 1 época no permite a las capas convolucionales aprender patrones sutiles en imágenes médicas. irace busca el balance óptimo (2 a 5 épocas) para obtener señales de fitness reales sin disparar el tiempo de cómputo.
3. **Tasa de aprendizaje (`lr`) y Batch Size**: Ajuste fino del optimizador Adam ($\text{lr} \in [5\times 10^{-5}, 5\times 10^{-3}]$) y batches de 16, 32 o 64 según el tamaño del dataset.
4. **Parámetros evolutivos especializados**:
   - **V10**: Evaporación de feromonas (`rho`), exponente de atracción (`alpha`) y depósito élite (`top_k_ratio`).
   - **V11**: Modelo multi-isla con feromonas aisladas (`n_islands`, `migration_interval`, `migration_size`).
   - **V12**: Modelo multi-isla puro con Deterministic Crowding, preservación de diversidad (`target_diversity`) y anti-estancamiento (`stagnation_limit`).

---

## 📁 Estructura de Archivos

```
irace_tuning/
├── target-runner              # Ejecutable wrapper llamado directamente por irace
├── runner_deepga.py           # Motor en Python que entrena DeepGA y retorna el costo (1.0 - acc)
├── parameters.txt             # Espacio de búsqueda y parámetros condicionales (v10, v11, v12)
├── scenario.txt               # Configuración del escenario de irace (budget, tests, logs)
├── instances.txt              # Lista de instancias a optimizar (Tumour, Tumour_3)
├── run_irace.sh               # Script Bash de un solo clic para lanzar irace
├── run_irace.R                # Script en R alternativo con ejecución programática
├── analyze_results.R          # Extrae las configuraciones élite y genera reportes
├── evaluate_best_config.py    # Re-entrena la red ganadora, genera matriz de confusión y guarda el .pth
├── test_target_runner.py      # Test rápido de auto-diagnóstico (1 corrida seca)
├── requirements.txt           # Dependencias de Python requeridas
└── README.md                  # Esta guía
```

---

## 🚀 Guía de Instalación y Ejecución en Otra Máquina

### Paso 1: Clonar o Copiar la Carpeta del Proyecto
Copie todo el directorio `DeepGA/` a la máquina de destino (servidor, clúster o PC con GPU).

### Paso 2: Instalar Dependencias del Sistema y R

#### En Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y r-base r-base-dev python3 python3-pip python3-venv
```

#### En Fedora / RHEL / CentOS:
```bash
sudo dnf install -y R R-devel python3 python3-pip python3-virtualenv
```

### Paso 3: Instalar el Paquete `irace` en R
Ejecute en la terminal:
```bash
Rscript -e "install.packages('irace', repos='https://cloud.r-project.org')"
```

### Paso 4: Configurar el Entorno Virtual de Python
```bash
cd DeepGA
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r irace_tuning/requirements.txt
```

### Paso 5: Ubicación de los Datasets (`Tumour` y `Tumour_3`)
El runner busca automáticamente las carpetas `Tumour` y `Tumour_3` en:
- `./dataset/Tumour` o `./data/Tumour`
- `~/Documents/Datasets/Tumour`
- `~/Downloads/Tumour`

O puede definir explícitamente la variable de entorno:
```bash
export DEEPGA_DATA_DIR="/ruta/absoluta/a/tus/Datasets"
```

---

## 🧪 Paso 6: Ejecutar Prueba de Diagnóstico (Dry Run)

Antes de iniciar la búsqueda completa, verifique que el entorno esté 100% operativo:
```bash
cd irace_tuning
python test_target_runner.py
```
Si todo es correcto, verá: `✅ TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO`.

---

## 🏁 Paso 7: Ejecutar la Optimización con irace

Para lanzar el proceso de sintonización automática:

```bash
cd irace_tuning
./run_irace.sh
```

O si prefiere ejecutarlo directamente con R:
```bash
Rscript run_irace.R
```

> **Nota sobre el Presupuesto (`maxExperiments`)**:
> En `scenario.txt`, `maxExperiments = 300` realiza 300 evaluaciones de DeepGA. Si dispone de menos tiempo o recursos, puede reducirlo en `scenario.txt` a `100` o `150`. Si dispone de múltiples GPUs o núcleos, aumente `parallel = 2` o `parallel = 4`.

---

## 📊 Paso 8: Analizar Resultados e Identificar la Mejor Configuración

Una vez concluida la ejecución de irace (se genera el archivo `irace.Rdata`), ejecute:

```bash
Rscript analyze_results.R
```

Este comando:
1. Imprime la tabla con todas las configuraciones élite finales.
2. Muestra los hiperparámetros ganadores exactos.
3. Genera automáticamente el archivo `best_configuration.json` y `best_configuration_cmd.txt`.

---

## 🏆 Paso 9: Entrenar el Modelo Ganador y Generar Gráficos

Para tomar la mejor configuración encontrada por irace y entrenar a fondo la arquitectura ganadora (ej. 25-30 épocas) con matriz de confusión y guardado de pesos `.pth`:

```bash
python evaluate_best_config.py --config-json best_configuration.json --dataset Tumour_3 --final-epochs 25
```

Los artefactos resultantes se guardarán en `./best_model_results/`:
- `best_model_v12_exec_1.pth` (Pesos de la CNN de alta precisión)
- `best_model_v12_exec_1.pkl` (Genoma evolutivo)
- `matriz_confusion_final_v12_Tumour_3.png` (Matriz de confusión)
- Reporte de clasificación con métricas de Precisión, Sensibilidad (Recall), Especificidad y F1-Score.
