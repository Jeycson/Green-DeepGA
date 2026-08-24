@echo off
REM ==============================================================================
REM VALIDACION AUTOMATIZADA PARA RESULTADOS DE IRACE EN GREEN DEEPGA (10 CORRIDAS)
REM ==============================================================================

set DATA_ROOT=%1
if "%DATA_ROOT%"=="" set DATA_ROOT=./Datasets/Covid

set NUM_RUNS=%2
if "%NUM_RUNS%"=="" set NUM_RUNS=10

set START_SEED=%3
if "%START_SEED%"=="" set START_SEED=1

set JSON_CONFIG=%4
if "%JSON_CONFIG%"=="" set JSON_CONFIG=irace_tuning/best_configuration.json

echo ============================================================================
echo     INICIANDO VALIDACION AUTOMATIZADA DE IRACE (GREEN DEEPGA)
echo ============================================================================
echo   Dataset:            %DATA_ROOT%
echo   Repeticiones:       %NUM_RUNS% corridas
echo   Semilla Inicial:    %START_SEED%
echo   Configuracion JSON: %JSON_CONFIG%
echo ============================================================================

if exist "%JSON_CONFIG%" (
    python ejecutar_validacion_irace.py --config-json "%JSON_CONFIG%" --data-root "%DATA_ROOT%" --num-runs %NUM_RUNS% --start-seed %START_SEED%
) else (
    echo Advertencia: No se encontro %JSON_CONFIG%.
    python ejecutar_validacion_irace.py --data-root "%DATA_ROOT%" --num-runs %NUM_RUNS% --start-seed %START_SEED%
)

pause
