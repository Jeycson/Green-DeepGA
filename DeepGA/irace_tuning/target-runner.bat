@echo off
rem ===========================================================================
rem target-runner para irace y DeepGA (Windows Batch)
rem ===========================================================================
setlocal enabledelayedexpansion

rem Forzar UTF-8 para evitar errores de codificación con caracteres especiales
chcp 65001 >nul 2>nul

set "TARGET_RUNNER_DIR=%~dp0"
set "PROJECT_ROOT=%TARGET_RUNNER_DIR%.."

rem Crear directorios de logs si no existen
if not exist "%TARGET_RUNNER_DIR%irace_logs" mkdir "%TARGET_RUNNER_DIR%irace_logs" >nul 2>nul
if not exist "%TARGET_RUNNER_DIR%irace_checkpoints" mkdir "%TARGET_RUNNER_DIR%irace_checkpoints" >nul 2>nul

rem Detección de Python (prioriza venv activo o .venv en la raíz)
set "PYTHON_EXE=python"
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
    )
) else if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
)

set "CONFIG_ID=%~1"
set "INSTANCE_ID=%~2"
set "SEED=%~3"
set "INSTANCE=%~4"

if "%CONFIG_ID%"=="" (
    echo 2.000000
    exit /b 0
)

shift
shift
shift
shift

set "PARAMS="
:parse_args
if "%~1"=="" goto execute_runner
set "PARAMS=!PARAMS! %1"
shift
goto parse_args

:execute_runner
set "RAW_COST="
rem Ejecutar runner redirigiendo stderr al archivo de errores
for /f "delims=" %%i in ('"%PYTHON_EXE%" "%TARGET_RUNNER_DIR%runner_deepga.py" --instance "%INSTANCE%" --seed %SEED% --config-id "%CONFIG_ID%" --instance-id "%INSTANCE_ID%" %PARAMS% 2^>^> "%TARGET_RUNNER_DIR%irace_logs\runner_errors.log"') do (
    set "RAW_COST=%%i"
)

rem Si la salida no es vacía, imprimir el costo; de lo contrario devolver costo de penalización
if defined RAW_COST (
    echo !RAW_COST!
) else (
    echo 2.000000
)

exit /b 0
