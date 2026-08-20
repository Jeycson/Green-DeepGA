# -*- coding: utf-8 -*-
"""
Script de Verificación y Diagnóstico de Experimentos:
Analiza el directorio de checkpoints y reportes para identificar con precisión
cuál experimento falló o faltó por completar, y ofrece el comando exacto para re-ejecutarlo.

Uso:
    python verificar_experimentos.py
    python verificar_experimentos.py --start-seed 104 --end-seed 109 --start-exec 166
"""

import os
import sys
import glob
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Verificador de estado de experimentos")
    parser.add_argument("--start-seed", type=int, default=104, help="Semilla inicial (default: 104)")
    parser.add_argument("--end-seed", type=int, default=109, help="Semilla final (default: 109)")
    parser.add_argument("--start-exec", type=int, default=166, help="Execution inicial (default: 166)")
    parser.add_argument("--variants", nargs="+", default=["v1", "v10", "v11", "v12"], help="Variantes evaluadas")
    parser.add_argument("--data-root", type=str, default="./Datasets/Covid", help="Ruta al dataset")
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 3], help="Canales de entrada")
    parser.add_argument("--pop-size", type=int, default=12, help="Población")
    parser.add_argument("--generations", type=int, default=5, help="Generaciones")
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/", help="Directorio de checkpoints")
    parser.add_argument("--rerun", action="store_true", help="Ejecuta automáticamente los experimentos faltantes")
    return parser.parse_args()


def check_experiment_status(chck_dir, variant, seed, execution):
    """
    Verifica si una corrida específica completó exitosamente buscando:
    1. Checkpoint .pth del modelo ganador (best_model_{variant}_exec_{execution}.pth)
    2. Archivo de reporte individual o archivo solo valores
    """
    variant_clean = variant.lower()
    
    # 1. Buscar checkpoint del modelo
    model_pattern = os.path.join(chck_dir, f"best_model_{variant_clean}_exec_{execution}.pth")
    model_exists = os.path.exists(model_pattern)
    
    # 2. Buscar reporte individual o archivo solo valores
    values_pattern = glob.glob(os.path.join(chck_dir, f"exp_*_seed{seed}_*_values.txt"))
    # O buscar en el archivo de texto acumulativo si está registrado
    summary_path = os.path.join(chck_dir, "experiments_summary_values.txt")
    in_summary = False
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # Buscar coincidencia de variante y semilla
            for line in content.splitlines():
                parts = line.split("\t")
                if len(parts) >= 3:
                    if str(seed) == parts[2].strip() and variant_clean in parts[1].lower():
                        in_summary = True
                        break
                        
    is_complete = model_exists or (len(values_pattern) > 0 and in_summary)
    return {
        "complete": is_complete,
        "model_file": model_exists,
        "in_summary": in_summary
    }


def main():
    args = parse_args()
    
    print("\n" + "=" * 78)
    print("           DIAGNÓSTICO Y VERIFICACIÓN DE EXPERIMENTOS DEEPGA")
    print("=" * 78)
    print(f"📁 Directorio analizado: {os.path.abspath(args.chck_dir)}")
    print(f"🌱 Rango de Semillas:   {args.start_seed} a {args.end_seed}")
    print(f"🔢 Execution Inicial:   {args.start_exec}")
    print(f"🧬 Variantes:           {', '.join(args.variants)}")
    print("-" * 78)
    
    if not os.path.exists(args.chck_dir):
        print(f"❌ El directorio '{args.chck_dir}' no existe aún.")
        return

    current_exec = args.start_exec
    run_idx = 0
    total_runs = (args.end_seed - args.start_seed + 1) * len(args.variants)
    
    completed_runs = []
    missing_runs = []
    
    print(f"{'#':<4} | {'Variante':<10} | {'Semilla':<8} | {'Execution':<10} | {'Estado':<18} | {'Detalle'}")
    print("-" * 78)
    
    for seed in range(args.start_seed, args.end_seed + 1):
        for variant in args.variants:
            run_idx += 1
            st = check_experiment_status(args.chck_dir, variant, seed, current_exec)
            
            cmd = (
                f"python ejemplo_local.py --execution {current_exec} --variant {variant} "
                f"--seed {seed} --data-root {args.data_root} --pop-size {args.pop_size} "
                f"--generations {args.generations} --in-channels {args.in_channels}"
            )
            
            info = {
                "run_idx": run_idx,
                "variant": variant,
                "seed": seed,
                "execution": current_exec,
                "cmd": cmd
            }
            
            if st["complete"]:
                completed_runs.append(info)
                status_str = "✅ COMPLETADO"
                detail_str = f"best_model_{variant}_exec_{current_exec}.pth OK"
            else:
                missing_runs.append(info)
                status_str = "❌ FALLÓ / FALTANTE"
                detail_str = "No se encontró checkpoint ni reporte"
                
            print(f"{run_idx:<4} | {variant.upper():<10} | {seed:<8} | {current_exec:<10} | {status_str:<18} | {detail_str}")
            current_exec += 1

    print("=" * 78)
    print(f"📊 RESUMEN: {len(completed_runs)} exitosos / {len(missing_runs)} fallidos (Total: {total_runs})")
    print("=" * 78)
    
    if missing_runs:
        print("\n" + "!" * 78)
        print("  ⚠️ EXPERIMENTO(S) QUE FALLARON O NO SE COMPLETARON:")
        print("!" * 78)
        for m in missing_runs:
            print(f"\n❌ Corrida #{m['run_idx']}: Variante {m['variant'].upper()} | Semilla {m['seed']} | Execution {m['execution']}")
            print(f"   ▶ Comando exacto para volver a correr solo este:")
            print(f"   {m['cmd']}")
        print("\n" + "!" * 78)
        
        if args.rerun:
            print("\n🚀 Re-ejecutando experimentos faltantes automáticamente...")
            for m in missing_runs:
                print(f"\nEjecutando: {m['cmd']}")
                subprocess.run(m['cmd'], shell=True)
    else:
        print("\n🎉 ¡Excelente! Todos los experimentos planificados están completos al 100%.")


if __name__ == "__main__":
    main()
