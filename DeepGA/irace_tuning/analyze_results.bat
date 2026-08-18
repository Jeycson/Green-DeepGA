@echo off
rem ===========================================================================
rem Analizador de Resultados de irace para DeepGA en Windows
rem ===========================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "RSCRIPT_EXE="
where Rscript >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "RSCRIPT_EXE=Rscript"
) else (
    for /d %%D in ("C:\Program Files\R\R-*") do (
        if exist "%%D\bin\x64\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\x64\Rscript.exe"
        if exist "%%D\bin\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\Rscript.exe"
    )
    if not defined RSCRIPT_EXE (
        for /d %%D in ("%LOCALAPPDATA%\Programs\R\R-*") do (
            if exist "%%D\bin\x64\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\x64\Rscript.exe"
            if exist "%%D\bin\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\Rscript.exe"
        )
    )
    if not defined RSCRIPT_EXE (
        for /d %%D in ("C:\R\R-*") do (
            if exist "%%D\bin\x64\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\x64\Rscript.exe"
            if exist "%%D\bin\Rscript.exe" set "RSCRIPT_EXE=%%D\bin\Rscript.exe"
        )
    )
)

if not defined RSCRIPT_EXE (
    echo [ERROR] No se pudo encontrar Rscript.exe.
    pause
    exit /b 1
)

"!RSCRIPT_EXE!" analyze_results.R %*
pause
