#!/usr/bin/env bash
# ==============================================================================
# SCRIPT DE VALIDACIÓN AUTOMATIZADA PARA IRACE EN GREEN DEEPGA (10 CORRIDAS)
# ==============================================================================
# Uso:
#   bash ejecutar_validacion_irace.sh
#   bash ejecutar_validacion_irace.sh ./Datasets/Tumour 10 1
# ==============================================================================

set -e

DATA_ROOT=${1:-"./Datasets/Covid"}
NUM_RUNS=${2:-10}
START_SEED=${3:-1}
JSON_CONFIG=${4:-"irace_tuning/best_configuration.json"}

# Detección de python
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
fi

echo "============================================================================"
echo "    INICIANDO VALIDACIÓN AUTOMATIZADA DE IRACE (GREEN DEEPGA)"
echo "============================================================================"
echo "  Dataset:            $DATA_ROOT"
echo "  Repeticiones:       $NUM_RUNS corridas independientes"
echo "  Semilla Inicial:    $START_SEED"
echo "  Configuración JSON: $JSON_CONFIG"
echo "  Python:             $PYTHON_CMD"
echo "============================================================================"

if [ -f "$JSON_CONFIG" ]; then
    $PYTHON_CMD ejecutar_validacion_irace.py \
        --config-json "$JSON_CONFIG" \
        --data-root "$DATA_ROOT" \
        --num-runs "$NUM_RUNS" \
        --start-seed "$START_SEED"
else
    echo "⚠️ No se encontró $JSON_CONFIG. Pasando argumentos por defecto..."
    $PYTHON_CMD ejecutar_validacion_irace.py \
        --data-root "$DATA_ROOT" \
        --num-runs "$NUM_RUNS" \
        --start-seed "$START_SEED"
fi
