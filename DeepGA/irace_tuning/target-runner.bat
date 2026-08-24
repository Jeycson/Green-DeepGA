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

rem Detección robusta de Python (prioriza venv activo, .venv, venv, conda o PATH)
set "PYTHON_EXE="
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
)
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
)
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
)
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%\env\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\env\Scripts\python.exe"
)
if not defined PYTHON_EXE if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
)
if not defined PYTHON_EXE (
    for %%P in (
        "%USERPROFILE%\anaconda3\python.exe"
        "%USERPROFILE%\miniconda3\python.exe"
        "%LOCALAPPDATA%\anaconda3\python.exe"
        "%LOCALAPPDATA%\miniconda3\python.exe"
        "C:\ProgramData\anaconda3\python.exe"
        "C:\ProgramData\miniconda3\python.exe"
        "C:\anaconda3\python.exe"
        "C:\miniconda3\python.exe"
    ) do (
        if not defined PYTHON_EXE if exist "%%~P" set "PYTHON_EXE=%%~P"
    )
)
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        where py >nul 2>nul
        if !ERRORLEVEL! equ 0 set "PYTHON_EXE=py"
    )
)

if not defined PYTHON_EXE (
    echo 2.000000
    exit /b 0
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
set "OUT_TMP=%TARGET_RUNNER_DIR%irace_logs\out_%CONFIG_ID%_%INSTANCE_ID%_%SEED%.tmp"
set "ERR_LOG=%TARGET_RUNNER_DIR%irace_logs\runner_errors.log"

rem Ejecutar runner redirigiendo stdout a archivo temporal y stderr al archivo de errores
"%PYTHON_EXE%" "%TARGET_RUNNER_DIR%runner_deepga.py" --instance "%INSTANCE%" --seed %SEED% --config-id "%CONFIG_ID%" --instance-id "%INSTANCE_ID%" %PARAMS% > "%OUT_TMP%" 2>> "%ERR_LOG%"

set "RAW_COST="
if exist "%OUT_TMP%" (
    for /f "usebackq delims=" %%a in ("%OUT_TMP%") do (
        set "RAW_COST=%%a"
    )
    del /f /q "%OUT_TMP%" >nul 2>nul
)

rem Si la salida no es vacía, imprimir el costo; de lo contrario devolver costo de penalización
if defined RAW_COST (
    echo !RAW_COST!
) else (
    echo 2.000000
)

exit /b 0
