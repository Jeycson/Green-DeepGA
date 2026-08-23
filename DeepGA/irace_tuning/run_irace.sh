#!/usr/bin/env bash
###############################################################################
# Script de Lanzamiento Automatizado para irace + DeepGA
# Optimización de Hiperparámetros (v10, v11, v12) sobre Tumour y Tumour_3
###############################################################################

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DIR}"

    echo "======================================================================"
    echo "    LANZADOR DE IRACE PARA OPTIMIZACIÓN DE HIPERPARÁMETROS DEEPGA"
    echo "    Variantes: V10, V11, V12 | Datasets: Benchmark 7 Datasets (baseline.csv)"
    echo "======================================================================"

# 1. Verificar si existe R y el paquete irace
if ! command -v Rscript &> /dev/null; then
    echo "❌ Error: 'Rscript' no está instalado en el sistema."
    echo "   Instale R con: sudo apt install r-base r-base-dev (Ubuntu/Debian) o sudo dnf install R (Fedora/RHEL)"
    exit 1
fi

# Verificar paquete irace en R
IRACE_INSTALLED=$(Rscript -e "cat(as.character('irace' %in% installed.packages()[,'Package']))" 2>/dev/null || echo "FALSE")
if [ "${IRACE_INSTALLED}" != "TRUE" ]; then
    echo "⚠️  El paquete 'irace' no está instalado en R. Intentando instalar automáticamente..."
    Rscript -e "install.packages('irace', repos='https://cloud.r-project.org')"
fi

# 2. Dar permisos de ejecución al target-runner
chmod +x ./target-runner

# 3. Crear carpetas de logs y checkpoints
mkdir -p ./irace_logs
mkdir -p ./irace_checkpoints

# 4. Parámetros opcionales pasados por línea de comandos
EXTRA_ARGS="$*"

echo ""
echo "🚀 Iniciando proceso de búsqueda con irace..."
echo "📄 Escenario: scenario.txt"
echo "⚙️  Parámetros: parameters.txt"
echo "📂 Instancias Train (5): train_instances.txt"
echo "📂 Instancias Test (2):  test_instances.txt"
echo "----------------------------------------------------------------------"

# 5. Ejecutar irace
if command -v irace &> /dev/null; then
    irace --scenario scenario.txt ${EXTRA_ARGS}
else
    # Ejecutar mediante Rscript si el binario irace no está en el PATH
    Rscript -e "library(irace); irace.cmdline(c('--scenario', 'scenario.txt', unlist(strsplit('${EXTRA_ARGS}', ' '))))"
fi

echo ""
echo "======================================================================"
echo "✨ Búsqueda de irace finalizada con éxito."
echo "📊 Los resultados y configuraciones élite se han guardado en: irace.Rdata"
echo "🔍 Para analizar los resultados detallados, ejecute:"
echo "   Rscript analyze_results.R"
echo "======================================================================"
