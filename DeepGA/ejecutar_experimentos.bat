@echo off
chcp 65001 >nul
python ejecutar_experimentos.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Ocurrió un problema durante la ejecución de los experimentos.
)
pause
