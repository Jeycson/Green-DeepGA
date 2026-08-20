#!/usr/bin/env bash
# ==============================================================================
# AUTOMATIZACIÓN DE EXPERIMENTOS GREEN DEEPGA (.SH)
# ==============================================================================
# Descripción:
#   Script bash para automatizar la ejecución secuencial de experimentos en Linux/macOS.
#   - Semillas: Varía de uno en uno desde 103 hasta 110.
#   - Un ciclo: Evalúa consecutivamente las variantes v1, v10, v11 y v12.
#   - Dataset: El mismo dataset para todas las evaluaciones.
#   - Execution: Se actualiza en cada corrida para evitar colisiones y
#     guardar modelos/reportes independientes.
# ==============================================================================

set -e

START_SEED=${1:-103}
END_SEED=${2:-110}
START_EXEC=${3:-165}

CURRENT_EXEC=$START_EXEC

DATA_ROOT="./Datasets/Covid"
IN_CHANNELS=1
POP_SIZE=12
GENERATIONS=5
TRAIN_EPOCHS=2
FINAL_EPOCHS=10
BATCH_SIZE=32
LR="1e-4"
W="0.3"
CHCK_DIR="./checkpoints/"
COUNTRY_ISO="MEX"

# Detección de python
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
fi

echo "============================================================================"
echo "         AUTOMATIZACIÓN DE EXPERIMENTOS GREEN DEEPGA"
echo "============================================================================"
echo "  Rango de Semillas:        $START_SEED a $END_SEED (paso de 1 en 1 por ciclo)"
echo "  Execution Inicial:        $START_EXEC (incrementa de 1 en 1 por variante)"
echo "  Variantes por Ciclo:      v1, v10, v11, v12"
echo "  Ruta del Dataset:         $DATA_ROOT"
echo "  Canales de Entrada:       $IN_CHANNELS (Grayscale / Covid)"
echo "  Población / Generaciones: $POP_SIZE individuos / $GENERATIONS generaciones"
echo "  Épocas (GA / Final):      $TRAIN_EPOCHS épocas / $FINAL_EPOCHS épocas"
echo "  Directorio Checkpoints:   $CHCK_DIR"
echo "  Intérprete Python:        $PYTHON_CMD"
echo "============================================================================"

TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

for SEED in $(seq $START_SEED $END_SEED); do
    echo ""
    echo "============================================================================"
    echo "  [CICLO DE EXPERIMENTACIÓN] SEMILLA ACTUAL = $SEED"
    echo "============================================================================"
    
    for VAR in v1 v10 v11 v12; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        echo ""
        echo "------------------------------------------------------------------------"
        echo "  [Corrida #$TOTAL_RUNS] Variante: $VAR | Semilla: $SEED | Execution: $CURRENT_EXEC"
        echo "------------------------------------------------------------------------"
        
        if $PYTHON_CMD ejemplo_local.py \
            --execution "$CURRENT_EXEC" \
            --variant "$VAR" \
            --seed "$SEED" \
            --data-root "$DATA_ROOT" \
            --pop-size "$POP_SIZE" \
            --generations "$GENERATIONS" \
            --in-channels "$IN_CHANNELS" \
            --train-epochs "$TRAIN_EPOCHS" \
            --final-epochs "$FINAL_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --w "$W" \
            --chck-dir "$CHCK_DIR" \
            --country-iso "$COUNTRY_ISO"; then
            
            SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
            echo "[OK] Variante $VAR finalizada con éxito (Semilla: $SEED, Exec: $CURRENT_EXEC)."
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            echo "[ERROR] Falló la corrida de $VAR con Semilla $SEED y Exec $CURRENT_EXEC."
        fi
        
        CURRENT_EXEC=$((CURRENT_EXEC + 1))
    done
done

echo ""
echo "============================================================================"
echo "                   RESUMEN DE LA AUTOMATIZACIÓN"
echo "============================================================================"
echo "  Total de Corridas Realizadas: $TOTAL_RUNS"
echo "  Corridas Exitosas:            $SUCCESS_RUNS"
echo "  Corridas Fallidas:            $FAILED_RUNS"
echo "  Última Ejecución Utilizada:   $CURRENT_EXEC"
echo ""
echo "  Archivos generados en: $CHCK_DIR"
echo "    - Reportes con texto:             exp_*.txt y reporte_*.txt"
echo "    - Reportes SOLO VALORES (Excel):  exp_*_values.txt y exp_*_values.csv"
echo "    - Resumen acumulativo TSV:        experiments_summary_values.txt"
echo "    - Resumen acumulativo Excel CSV:  experiments_summary_values.csv"
echo "============================================================================"
