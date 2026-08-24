#!/usr/bin/env bash
# ==============================================================================
# Script de Análisis Post-Optimización de irace para Linux / Conda
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

RDATA_FILE=${1:-"irace.Rdata"}

# Comprobar Rscript
if ! command -v Rscript &> /dev/null; then
    echo "[ERROR] No se encontró el comando 'Rscript' en tu entorno actual."
    echo "Si estás en Conda, puedes instalarlo con: conda install -c r r-base r-irace"
    exit 1
fi

Rscript analyze_results.R "$RDATA_FILE"
