@echo off
rem ===========================================================================
rem Lanzador de irace para DeepGA en Windows (CMD / PowerShell)
rem Variantes: V10, V11, V12 | Datasets: Tumour, Tumour_3
rem ===========================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ======================================================================
echo     LANZADOR DE IRACE PARA OPTIMIZACION DE HIPERPARAMETROS DEEPGA
echo     Variantes: V10, V11, V12 ^| Datasets: Tumour, Tumour_3
echo ======================================================================

rem 1. Verificar si existe Rscript
where Rscript >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 'Rscript' no esta en el PATH del sistema.
    echo Asegurese de tener R instalado y agregado a las Variables de Entorno.
    pause
    exit /b 1
)

rem 2. Crear carpetas de logs y checkpoints
if not exist "irace_logs" mkdir "irace_logs"
if not exist "irace_checkpoints" mkdir "irace_checkpoints"

rem 3. Ejecutar irace mediante Rscript
echo.
echo Iniciando proceso de optimizacion con irace...
echo Escenario: scenario.txt
echo Parametros: parameters.txt
echo Instancias: instances.txt
echo ----------------------------------------------------------------------

Rscript -e "if (!require('irace', quietly=TRUE)) install.packages('irace', repos='https://cloud.r-project.org'); library(irace); s <- readScenario('scenario.txt'); irace(scenario=s)"

echo.
echo ======================================================================
echo  Optimizacion de irace finalizada.
echo  Los resultados se han guardado en: irace.Rdata
echo  Para analizar las configuraciones ganadoras ejecute:
echo    Rscript analyze_results.R
echo ======================================================================
