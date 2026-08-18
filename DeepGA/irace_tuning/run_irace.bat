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
    echo Por favor indique la ruta a Rscript.exe o abra PowerShell y agregue R a su PATH de usuario.
    echo.
    pause
    exit /b 1
)

echo [OK] Usando Rscript desde: !RSCRIPT_EXE!

rem Agregar el directorio de R al PATH de la sesion actual
for %%F in ("!RSCRIPT_EXE!") do set "R_BIN_DIR=%%~dpF"
set "PATH=!R_BIN_DIR!;%PATH%"

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

"!RSCRIPT_EXE!" -e "if (!require('irace', quietly=TRUE)) install.packages('irace', repos='https://cloud.r-project.org'); library(irace); s <- readScenario('scenario.txt'); irace(scenario=s)"

echo.
echo ======================================================================
echo  Optimizacion de irace finalizada.
echo  Los resultados se han guardado en: irace.Rdata
echo  Para analizar las configuraciones ganadoras ejecute:
echo    analyze_results.bat  (o Rscript analyze_results.R)
echo ======================================================================
pause
