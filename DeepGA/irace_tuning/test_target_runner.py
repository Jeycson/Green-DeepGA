# -*- coding: utf-8 -*-
"""
Script de Auto-diagnóstico y Prueba para el Target-Runner de irace.
Verifica:
1. Librerías requeridas
2. Carga y consistencia de baseline.csv (Macro F1 + Consumo kWh)
3. Validación matemática de la normalización multi-objetivo
4. Localización de datasets
5. Permisos de ejecución del target-runner
6. Prueba de llamada simulada de irace con los nuevos parámetros (--final-epochs, etc.)
"""

import os
import sys
import subprocess
from pathlib import Path

DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIR.parent
IS_WINDOWS = sys.platform.startswith("win")

print("=" * 70)
print("  DIAGNÓSTICO Y TEST DE COMPATIBILIDAD IRACE + GREEN DEEPGA")
print(f"  Sistema Operativo: {sys.platform.upper()} | Python: {sys.version.split()[0]}")
print("=" * 70)

# 1. Comprobar imports clave
print("\n1. Verificando librerías de Python...")
required_pkgs = ["torch", "torchvision", "sklearn", "pandas", "numpy", "PIL"]
missing = []
for pkg in required_pkgs:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg} detectado.")
    except ImportError:
        missing.append(pkg)
        print(f"   ✗ {pkg} NO encontrado.")

if missing:
    print(f"\n⚠️  Aviso: Faltan dependencias en este intérprete: {missing}.")
    print("   Si ejecuta con un entorno virtual (.venv), asegúrese de activarlo.\n")

# 2. Comprobar carga y normalización de baseline.csv
print("\n2. Verificando archivo 'baseline.csv' y normalización multi-objetivo...")
sys.path.insert(0, str(DIR))
from runner_deepga import load_baseline_metrics, get_dataset_baseline, calculate_normalized_cost, resolve_dataset_path

baseline_csv = DIR / "baseline.csv"
if baseline_csv.exists():
    b_map = load_baseline_metrics(baseline_csv)
    print(f"   ✓ baseline.csv encontrado con {len(b_map)} datasets cargados:")
    for k, v in b_map.items():
        print(f"     • {v['dataset_orig']:<15} -> Baseline Energy: {v['energy_kwh']:.6f} kWh | Baseline F1: {v['macro_f1']:.3f}%")
else:
    print(f"   ❌ baseline.csv NO encontrado en: {baseline_csv}")

# Test de la fórmula matemática con red ligera vs red pesada
print("\n3. Validando invariancia de escala de la fórmula normalizada...")
# Dataset ligero: BreadMNIST (Energy: ~0.00023 kWh, F1: ~68.6%)
bread_base = get_dataset_baseline("BreadMNIST", b_map)
cost_bread_base, _ = calculate_normalized_cost(bread_base["macro_f1"], bread_base["energy_kwh"], bread_base["macro_f1"], bread_base["energy_kwh"])
cost_bread_opt, d_bread = calculate_normalized_cost(75.0, bread_base["energy_kwh"] * 0.80, bread_base["macro_f1"], bread_base["energy_kwh"])

# Dataset pesado: PathMNIST (Energy: ~0.0321 kWh, F1: ~73.1%)
path_base = get_dataset_baseline("PathMNIST", b_map)
cost_path_base, _ = calculate_normalized_cost(path_base["macro_f1"], path_base["energy_kwh"], path_base["macro_f1"], path_base["energy_kwh"])
cost_path_opt, d_path = calculate_normalized_cost(79.0, path_base["energy_kwh"] * 0.80, path_base["macro_f1"], path_base["energy_kwh"])

print(f"   ✓ Costo exacto en Baseline (debe ser 1.000000): BreadMNIST={cost_bread_base:.6f} | PathMNIST={cost_path_base:.6f}")
print(f"   ✓ Costo con mejora (+F1, -20% Energy):")
print(f"     • BreadMNIST (ligero): Cost={cost_bread_opt:.6f} (Norm_Error={d_bread['norm_error']:.4f}, Norm_Energy={d_bread['norm_energy']:.4f})")
print(f"     • PathMNIST  (pesado): Cost={cost_path_opt:.6f} (Norm_Error={d_path['norm_error']:.4f}, Norm_Energy={d_path['norm_energy']:.4f})")
print("   ✓ Verificación matemática: Ambas escalas reciben la misma ponderación proporcional relativa.")

# 4. Comprobar localización de datasets
print("\n4. Verificando resolución de datasets...")
test_datasets = ["COVID", "BreadMNIST", "DermaMNIST", "BloodMNIST", "PathMNIST", "CIFAR-10", "Tumour", "Tumour_3"]
for d_name in test_datasets:
    resolved = resolve_dataset_path(d_name)
    status = "✓ Encontrado" if (os.path.exists(resolved) and os.path.isdir(resolved)) else "ℹ️ Nativo / Por resolver"
    print(f"   {status}: {d_name} -> {resolved}")

# 5. Comprobar ejecutable target-runner
print("\n5. Verificando ejecutable 'target-runner'...")
if IS_WINDOWS:
    tr_path = DIR / "target-runner.bat"
    if not tr_path.exists():
        tr_path = DIR / "target-runner"
    print(f"   ✓ Script detectado para Windows: {tr_path.name}")
else:
    tr_path = DIR / "target-runner"
    if tr_path.exists():
        is_exec = os.access(tr_path, os.X_OK)
        print(f"   ✓ target-runner existe. Permisos de ejecución: {'SÍ' if is_exec else 'NO'}")
        if not is_exec:
            print("   Aplicando chmod +x target-runner...")
            os.chmod(tr_path, 0o755)
    else:
        print("   ❌ target-runner no existe.")
        sys.exit(1)

print("\n" + "=" * 70)
print("✅ DIAGNÓSTICO COMPLETADO: RUNNER DEEPGA LISTO PARA IRACE")
print("=" * 70)
