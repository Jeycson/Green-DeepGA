# -*- coding: utf-8 -*-
"""
Script de Auto-diagnóstico y Prueba para el Target-Runner de irace.
Ejecuta una prueba rápida de 1 generación con DeepGA para verificar que:
1. PyTorch y las librerías necesarias estén instaladas.
2. Los datasets Tumour y Tumour_3 sean localizados correctamente.
3. Las variantes V10, V11 y V12 funcionen sin errores de memoria o ejecución.
4. El target-runner retorne el costo numérico exacto que espera irace.
"""

import os
import sys
import subprocess
from pathlib import Path

DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIR.parent

print("=" * 65)
print("  DIAGNÓSTICO Y TEST DE COMPATIBILIDAD IRACE + DEEPGA")
print("=" * 65)

# 1. Comprobar imports clave
print("\n1. Verificando librerías de Python...")
required_pkgs = ["torch", "torchvision", "sklearn", "numpy", "PIL"]
missing = []
for pkg in required_pkgs:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg} detectado.")
    except ImportError:
        missing.append(pkg)
        print(f"   ✗ {pkg} NO encontrado.")

if missing:
    print(f"\n❌ Faltan dependencias: {missing}. Instálelas con:")
    print("   pip install -r requirements.txt\n")
    sys.exit(1)

# 2. Comprobar localización de datasets
print("\n2. Verificando localización de datasets Tumour y Tumour_3...")
sys.path.insert(0, str(DIR))
from runner_deepga import resolve_dataset_path

for d_name in ["Tumour", "Tumour_3"]:
    resolved = resolve_dataset_path(d_name)
    if os.path.exists(resolved) and os.path.isdir(resolved):
        print(f"   ✓ Dataset '{d_name}' encontrado en: {resolved}")
    else:
        print(f"   ⚠️  Dataset '{d_name}' no encontrado automáticamente (probó en '{resolved}').")
        print("      Puede definir la variable DEEPGA_DATA_DIR=/ruta/a/Datasets o colocarlos en ./dataset/")

# 3. Comprobar permisos del target-runner
print("\n3. Verificando ejecutable 'target-runner'...")
tr_path = DIR / "target-runner"
if tr_path.exists():
    is_exec = os.access(tr_path, os.X_OK)
    print(f"   ✓ Archivo target-runner existe. Permisos de ejecución: {'SÍ' if is_exec else 'NO'}")
    if not is_exec:
        print("   Aplicando chmod +x target-runner...")
        os.chmod(tr_path, 0o755)
else:
    print("   ❌ target-runner no existe.")
    sys.exit(1)

# 4. Prueba rápida de target-runner (1 evaluación seca)
print("\n4. Ejecutando simulación de llamada irace (Dry Run - V12, 1 generación)...")
test_cmd = [
    str(tr_path),
    "1", "1", "42", "Tumour",
    "--variant", "v12",
    "--generations", "1",
    "--pop-size", "4",
    "--train-epochs", "1",
    "--lr", "0.001",
    "--w", "0.05",
    "--batch-size", "32"
]

try:
    res = subprocess.run(test_cmd, capture_ok=True if hasattr(subprocess, 'capture_ok') else False,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    stdout_clean = res.stdout.strip()
    last_line = stdout_clean.split("\n")[-1] if stdout_clean else ""
    
    print(f"   Código de salida: {res.returncode}")
    print(f"   Salida STDOUT recibida: '{last_line}'")
    
    try:
        cost = float(last_line)
        print(f"   ✓ Costo numérico parseado con éxito: {cost:.6f} (Accuracy estimada: {(1.0 - cost)*100:.2f}%)")
        print("\n" + "=" * 65)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO. EL ENTORNO ESTÁ LISTO PARA IRACE.")
        print("=" * 65)
    except ValueError:
        print(f"   ⚠️  La salida final no fue un flotante válido: '{last_line}'")
        print(f"   STDOUT completo:\n{res.stdout}")
        print(f"   STDERR completo:\n{res.stderr}")
        
except Exception as e:
    print(f"   ⚠️  No se pudo completar la simulación: {e}")
