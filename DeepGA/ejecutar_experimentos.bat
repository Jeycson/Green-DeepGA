@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM ============================================================================
REM AUTOMATIZACIÓN DE EXPERIMENTOS GREEN DEEPGA (.BAT)
REM ============================================================================
REM Descripción:
REM   Script por lotes para automatizar la ejecución secuencial de experimentos.
REM   - Semillas: Varía de uno en uno desde 103 hasta 110 (o personalizable).
REM   - Un ciclo: Evalúa consecutivamente las variantes v1, v10, v11 y v12 con la misma semilla.
REM   - Execution: Contador incremental independiente (ej. 165, 166, 167, 168, 169...)
REM     que avanza de 1 en 1 en CADA corrida individual de variante.
REM
REM Parámetros opcionales por línea de comando:
REM   ejecutar_experimentos.bat [START_SEED] [END_SEED] [START_EXEC]
REM   Ejemplo: ejecutar_experimentos.bat 103 110 165
REM ============================================================================

REM Configuración de Rango de Semillas y Execution Inicial
set START_SEED=103
set END_SEED=110
set START_EXEC=165

REM Sobreescritura opcional por argumentos CLI:
if not "%~1"=="" set START_SEED=%~1
if not "%~2"=="" set END_SEED=%~2
if not "%~3"=="" set START_EXEC=%~3

REM Inicializar contador dinámico de execution
set CURRENT_EXEC=%START_EXEC%

REM ============================================================================
REM CONFIGURACIÓN DE HIPERPARÁMETROS DEL EXPERIMENTO
REM ============================================================================
set DATA_ROOT=./Datasets/Covid
set IN_CHANNELS=1
set POP_SIZE=12
set GENERATIONS=5
set TRAIN_EPOCHS=2
set FINAL_EPOCHS=10
set BATCH_SIZE=32
set LR=1e-4
set W=0.3
set CHCK_DIR=./checkpoints/
set COUNTRY_ISO=MEX

REM Intérprete Python (por defecto 'python' para usar el entorno activo ej. torch_gpu)
set PYTHON_CMD=python

echo.
echo ============================================================================
echo         AUTOMATIZACION DE EXPERIMENTOS GREEN DEEPGA
echo ============================================================================
echo   Rango de Semillas:        %START_SEED% a %END_SEED% (paso de 1 en 1 por ciclo)
echo   Execution Inicial:        %START_EXEC% (incrementa de 1 en 1 por variante)
echo   Variantes por Ciclo:      v1, v10, v11, v12
echo   Ruta del Dataset:         %DATA_ROOT%
echo   Canales de Entrada:       %IN_CHANNELS% (Grayscale / Covid)
echo   Poblacion / Generaciones: %POP_SIZE% individuos / %GENERATIONS% generaciones
echo   Epocas (GA / Final):      %TRAIN_EPOCHS% epocas / %FINAL_EPOCHS% epocas
echo   Directorio Checkpoints:   %CHCK_DIR%
echo   Interprete Python:        %PYTHON_CMD%
echo ============================================================================
echo.

set TOTAL_RUNS=0
set SUCCESS_RUNS=0
set FAILED_RUNS=0

REM ============================================================================
REM BUCLE PRINCIPAL: VARIANDO LA SEMILLA DE UNO EN UNO POR CICLO
REM ============================================================================
for /L %%S in (%START_SEED%, 1, %END_SEED%) do (
    echo.
    echo ============================================================================
    echo   [CICLO DE EXPERIMENTACION] SEMILLA = %%S
    echo ============================================================================
    
    for %%V in (v1 v10 v11 v12) do (
        set /a TOTAL_RUNS+=1
        
        echo.
        echo ------------------------------------------------------------------------
        echo   [Corrida #!TOTAL_RUNS!] Variante: %%V ^| Semilla: %%S ^| Execution: !CURRENT_EXEC!
        echo ------------------------------------------------------------------------
        
        %PYTHON_CMD% ejemplo_local.py --execution !CURRENT_EXEC! --variant %%V --seed %%S --data-root %DATA_ROOT% --pop-size %POP_SIZE% --generations %GENERATIONS% --in-channels %IN_CHANNELS% --train-epochs %TRAIN_EPOCHS% --final-epochs %FINAL_EPOCHS% --batch-size %BATCH_SIZE% --lr %LR% --w %W% --chck-dir %CHCK_DIR% --country-iso %COUNTRY_ISO%
        
        if !errorlevel! equ 0 (
            set /a SUCCESS_RUNS+=1
            echo.
            echo [OK] Variante %%V finalizada con exito (Semilla: %%S, Exec: !CURRENT_EXEC!).
        ) else (
            set /a FAILED_RUNS+=1
            echo.
            echo [ERROR] Fallo la corrida de %%V con Semilla %%S y Exec !CURRENT_EXEC! (Codigo: !errorlevel!).
        )
        
        set /a CURRENT_EXEC+=1
    )
)

echo.
echo ============================================================================
echo                    RESUMEN DE LA AUTOMATIZACION
echo ============================================================================
echo   Total de Corridas Realizadas: !TOTAL_RUNS!
echo   Corridas Exitosas:            !SUCCESS_RUNS!
echo   Corridas Fallidas:            !FAILED_RUNS!
echo   Ultima Ejecucion Utilizada:   !CURRENT_EXEC!
echo.
echo   Archivos generados en: %CHCK_DIR%
echo     - Reportes con texto:             exp_*.txt y reporte_*.txt
echo     - Reportes SOLO VALORES (Excel):  exp_*_values.txt y exp_*_values.csv
echo     - Resumen acumulativo TSV:        experiments_summary_values.txt
echo     - Resumen acumulativo Excel CSV:  experiments_summary_values.csv
echo ============================================================================
echo.
pause
