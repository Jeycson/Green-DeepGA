@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

:: ============================================================================
:: AUTOMATIZACIÓN DE EXPERIMENTOS GREEN DEEPGA (.BAT)
:: ============================================================================
:: Descripción:
::   Script por lotes para automatizar la ejecución secuencial de experimentos.
::   - Semillas: Varía de uno en uno desde 103 hasta 110.
::   - Un ciclo: Evalúa consecutivamente las variantes v1, v10, v11 y v12.
::   - Dataset: El mismo dataset para todas las evaluaciones.
::   - Execution: Se actualiza en cada corrida para evitar colisiones y
::     guardar modelos/reportes independientes.
:: ============================================================================

:: Configuración de Rango de Semillas (por defecto: 103 a 110)
set START_SEED=103
set END_SEED=110

:: Si el usuario pasa argumentos por consola, se pueden sobreescribir:
:: Ejemplo: ejecutar_experimentos.bat 103 105
if not "%~1"=="" set START_SEED=%~1
if not "%~2"=="" set END_SEED=%~2

:: ============================================================================
:: CONFIGURACIÓN DE HIPERPARÁMETROS DEL EXPERIMENTO
:: ============================================================================
set DATA_ROOT=./data
set POP_SIZE=12
set GENERATIONS=5
set TRAIN_EPOCHS=2
set FINAL_EPOCHS=10
set BATCH_SIZE=32
set LR=1e-4
set W=0.3
set CHCK_DIR=./checkpoints/
set COUNTRY_ISO=MEX

:: Detección automática del intérprete de Python (entorno virtual o global)
set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
)

echo.
echo ============================================================================
echo         AUTOMATIZACIÓN DE EXPERIMENTOS GREEN DEEPGA
echo ============================================================================
echo   Rango de Semillas:        !START_SEED! a !END_SEED! (paso de 1 en 1)
echo   Variantes por Ciclo:      v1, v10, v11, v12
echo   Ruta del Dataset:         !DATA_ROOT!
echo   Población / Generaciones: !POP_SIZE! individuos / !GENERATIONS! generaciones
echo   Épocas (GA / Final):      !TRAIN_EPOCHS! épocas / !FINAL_EPOCHS! épocas
echo   Directorio Checkpoints:   !CHCK_DIR!
echo   Intérprete Python:        !PYTHON_CMD!
echo ============================================================================
echo.

set TOTAL_RUNS=0
set SUCCESS_RUNS=0
set FAILED_RUNS=0

:: ============================================================================
:: BUCLE PRINCIPAL: VARIANDO LA SEMILLA DE UNO EN UNO (103 a 110)
:: ============================================================================
for /L %%S in (!START_SEED!, 1, !END_SEED!) do (
    echo.
    echo ============================================================================
    echo   [CICLO DE EXPERIMENTACIÓN] INICIANDO CON SEMILLA = %%S
    echo ============================================================================
    
    :: Bucle interno: Evaluación de v1, v10, v11, v12 en el mismo dataset
    for %%V in (v1 v10 v11 v12) do (
        set /a TOTAL_RUNS+=1
        
        echo.
        echo ------------------------------------------------------------------------
        echo   [Corrida #!TOTAL_RUNS!] Variante: %%V ^| Semilla: %%S ^| Execution: %%S
        echo ------------------------------------------------------------------------
        
        !PYTHON_CMD! ejemplo_local.py ^
            --variant %%V ^
            --seed %%S ^
            --execution %%S ^
            --data-root !DATA_ROOT! ^
            --pop-size !POP_SIZE! ^
            --generations !GENERATIONS! ^
            --train-epochs !TRAIN_EPOCHS! ^
            --final-epochs !FINAL_EPOCHS! ^
            --batch-size !BATCH_SIZE! ^
            --lr !LR! ^
            --w !W! ^
            --chck-dir !CHCK_DIR! ^
            --country-iso !COUNTRY_ISO!
            
        if !errorlevel! equ 0 (
            set /a SUCCESS_RUNS+=1
            echo.
            echo [OK] Variante %%V finalizada con éxito (Semilla: %%S, Exec: %%S).
        ) else (
            set /a FAILED_RUNS+=1
            echo.
            echo [ERROR] Falló la corrida de %%V con Semilla %%S (Código: !errorlevel!).
        )
    )
)

echo.
echo ============================================================================
echo                    RESUMEN DE LA AUTOMATIZACIÓN
echo ============================================================================
echo   Total de Corridas Realizadas: !TOTAL_RUNS!
echo   Corridas Exitosas:            !SUCCESS_RUNS!
echo   Corridas Fallidas:            !FAILED_RUNS!
echo.
echo   Archivos generados en: !CHCK_DIR!
echo     - Reportes con texto:             exp_*.txt y reporte_*.txt
echo     - Reportes SOLO VALORES (Excel):  exp_*_values.txt y exp_*_values.csv
echo     - Resumen acumulativo TSV:        experiments_summary_values.txt
echo     - Resumen acumulativo Excel CSV:  experiments_summary_values.csv
echo ============================================================================
echo.
pause
