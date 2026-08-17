@echo off
rem ===========================================================================
rem target-runner para irace y DeepGA (Windows Batch)
rem ===========================================================================
setlocal enabledelayedexpansion

set TARGET_RUNNER_DIR=%~dp0
set PROJECT_ROOT=%TARGET_RUNNER_DIR%..

if not exist "%TARGET_RUNNER_DIR%irace_logs" mkdir "%TARGET_RUNNER_DIR%irace_logs"
if not exist "%TARGET_RUNNER_DIR%irace_checkpoints" mkdir "%TARGET_RUNNER_DIR%irace_checkpoints"

rem Detección de intérprete de Python
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
    echo 1.000000
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
for /f "tokens=*" %%a in ('"%PYTHON_EXE%" "%TARGET_RUNNER_DIR%runner_deepga.py" --instance "%INSTANCE%" --seed %SEED% --config-id "%CONFIG_ID%" --instance-id "%INSTANCE_ID%" %PARAMS% 2^>^> "%TARGET_RUNNER_DIR%irace_logs\runner_errors.log"') do (
    set "COST=%%a"
)

if "%COST%"=="" (
    echo 1.000000
) else (
    echo %COST%
)

exit /b 0
