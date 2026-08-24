@echo off
rem ===========================================================================
rem Lanzador de irace para DeepGA en Windows (CMD / PowerShell)
rem Detección automática de R y Rscript sin requerir permisos de administrador
rem Variantes: V10, V11, V12 | Datasets: Tumour, Tumour_3
rem ===========================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ======================================================================
echo     LANZADOR DE IRACE PARA OPTIMIZACION DE HIPERPARAMETROS DEEPGA
echo     Variantes: V10, V11, V12 ^| Datasets: Tumour, Tumour_3
echo ======================================================================

rem 1. Localizar Rscript.exe (primero en PATH, luego en rutas comunes de Windows)
set "RSCRIPT_EXE="

where Rscript >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "RSCRIPT_EXE=Rscript"
    echo [OK] Rscript detectado en el PATH del sistema.
) else (
    rem Buscar en Program Files
    for /d %%D in ("C:\Program Files\R\R-*") do (
        if exist "%%D\bin\x64\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\x64\Rscript.exe"
        if exist "%%D\bin\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\Rscript.exe"
    )
    rem Buscar en AppData de usuario
    if not defined RSCRIPT_EXE (
        for /d %%D in ("%LOCALAPPDATA%\Programs\R\R-*") do (
            if exist "%%D\bin\x64\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\x64\Rscript.exe"
            if exist "%%D\bin\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\Rscript.exe"
        )
    )
    rem Buscar en C:\R
    if not defined RSCRIPT_EXE (
        for /d %%D in ("C:\R\R-*") do (
            if exist "%%D\bin\x64\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\x64\Rscript.exe"
            if exist "%%D\bin\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\Rscript.exe"
        )
    )
    rem Buscar en Program Files (x86)
    if not defined RSCRIPT_EXE (
        for /d %%D in ("C:\Program Files (x86)\R\R-*") do (
            if exist "%%D\bin\x64\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\x64\Rscript.exe"
            if exist "%%D\bin\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\Rscript.exe"
        )
    )
)

if not defined RSCRIPT_EXE (
    echo.
    echo [ERROR] No se pudo encontrar Rscript.exe automaticamente.
    echo Por favor indique la ruta a Rscript.exe o agregue R a su PATH.
    echo.
    pause
    exit /b 1
)

echo [OK] Usando Rscript desde: !RSCRIPT_EXE!

rem Agregar el directorio de R al PATH de la sesion actual
for %%F in ("!RSCRIPT_EXE!") do set "R_BIN_DIR=%%~dpF"
set "PATH=!R_BIN_DIR!;%PATH%"

rem 2. Detectar y activar entorno virtual de Python o Conda
if defined CONDA_PREFIX (
    echo [OK] Entorno Conda activo detectado: !CONDA_DEFAULT_ENV! (!CONDA_PREFIX!)
) else if not defined VIRTUAL_ENV (
    if exist "..\.venv\Scripts\activate.bat" (
        call "..\.venv\Scripts\activate.bat"
        echo [OK] Entorno virtual .venv activado.
    ) else if exist "..\venv\Scripts\activate.bat" (
        call "..\venv\Scripts\activate.bat"
        echo [OK] Entorno virtual venv activado.
    ) else if exist "..\env\Scripts\activate.bat" (
        call "..\env\Scripts\activate.bat"
        echo [OK] Entorno virtual env activado.
    )
)

rem 3. Crear carpetas de logs y checkpoints
if not exist "irace_logs" mkdir "irace_logs"
if not exist "irace_checkpoints" mkdir "irace_checkpoints"

rem 4. Ejecutar irace mediante el script run_irace.R
echo.
echo Iniciando proceso de optimizacion con irace...
echo Escenario: scenario.txt
echo Parametros: parameters.txt
echo Instancias: instances.txt
echo ----------------------------------------------------------------------

"!RSCRIPT_EXE!" run_irace.R

echo.
echo ======================================================================
echo  Proceso de irace finalizado.
echo ======================================================================
pause
