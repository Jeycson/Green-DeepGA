# -*- coding: utf-8 -*-
"""
Script de Automatización de Experimentos Green DeepGA:
- Semillas: Itera de uno en uno desde start_seed (103) hasta end_seed (110).
- Un ciclo: Evalúa consecutivamente las variantes v1, v10, v11 y v12 con la misma semilla.
- Execution: Contador incremental independiente que avanza de 1 en 1 en cada variante (165, 166, 167, ...).
- Dataset: ./Datasets/Covid con in_channels=1.

Uso rápido:
    python ejecutar_experimentos.py
    
Uso personalizado (CLI):
    python ejecutar_experimentos.py --start-seed 103 --end-seed 110 --start-exec 165 --data-root ./Datasets/Covid
"""

import sys
import subprocess
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Automatización de experimentos Green DeepGA")
    parser.add_argument("--start-seed", type=int, default=104,
                        help="Semilla inicial del primer ciclo (default: 104)")
    parser.add_argument("--end-seed", type=int, default=109,
                        help="Semilla final del último ciclo (default: 109)")
    parser.add_argument("--start-exec", type=int, default=166,
                        help="Número de execution inicial (default: 166)")
    parser.add_argument("--variants", nargs="+", default=["v1", "v10", "v11", "v12"],
                        help="Variantes a evaluar en cada ciclo (default: v1 v10 v11 v12)")
    parser.add_argument("--data-root", type=str, default="./Datasets/Covid",
                        help="Ruta al dataset (default: ./Datasets/Covid)")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Resolución de las imágenes (default: 28 para MNIST/MedMNIST, 64 para otros)")
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 3],
                        help="Canales de entrada: 1 para Grayscale/Covid/BreastMNIST, 3 para RGB (default: 1)")
    parser.add_argument("--pop-size", type=int, default=12,
                        help="Tamaño de la población (default: 12)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Número de generaciones (default: 5)")
    parser.add_argument("--train-epochs", type=int, default=2,
                        help="Épocas por individuo en el GA (default: 2)")
    parser.add_argument("--final-epochs", type=int, default=10,
                        help="Épocas de entrenamiento final del ganador (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--w", type=float, default=0.3,
                        help="Peso w de penalización de parámetros (default: 0.3)")
    parser.add_argument("--t-size", type=int, default=3,
                        help="Tamaño de torneo de selección (default: 3)")
    parser.add_argument("--cr", type=float, default=0.7,
                        help="Probabilidad de cruce / crossover rate (default: 0.7)")
    parser.add_argument("--mr", type=float, default=0.5,
                        help="Probabilidad de mutación base (default: 0.5)")
    parser.add_argument("--mr-min", type=float, default=0.10,
                        help="Tasa mínima de mutación adaptativa (default: 0.10)")
    parser.add_argument("--mr-max", type=float, default=0.85,
                        help="Tasa máxima de mutación adaptativa (default: 0.85)")
    parser.add_argument("--pool-candidates-factor", type=int, default=5,
                        help="Factor de candidatos a predecir por el subrogado (default: 5)")
    parser.add_argument("--kappa", type=float, default=0.10,
                        help="Parámetro kappa de exploración UCB (default: 0.10)")
    parser.add_argument("--rho", type=float, default=0.10,
                        help="Tasa de evaporación de feromonas rho en V10/V11 (default: 0.10)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ponderación alpha de feromonas en V10/V11 (default: 1.0)")
    parser.add_argument("--top-k-ratio", type=float, default=0.20,
                        help="Ratio élite para depósito de feromonas (default: 0.20)")
    parser.add_argument("--n-islands", type=int, default=3,
                        help="Número de islas evolutivas (default: 3)")
    parser.add_argument("--migration-interval", type=int, default=12,
                        help="Intervalo de generaciones para migración (default: 12)")
    parser.add_argument("--migration-size", type=int, default=1,
                        help="Cantidad de individuos que migran por isla (default: 1)")
    parser.add_argument("--target-diversity", type=float, default=0.25,
                        help="Diversidad estructural objetivo intra-isla (default: 0.25)")
    parser.add_argument("--stagnation-limit", type=int, default=4,
                        help="Límite de generaciones sin mejora antes de anti-estancamiento (default: 4)")
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/",
                        help="Directorio de checkpoints (default: ./checkpoints/)")
    parser.add_argument("--country-iso", type=str, default="MEX",
                        help="Código ISO del país (default: MEX)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Auto-detección de tamaño de imagen: 28 para MedMNIST/MNIST o 64 por defecto
    if args.img_size is not None:
        effective_img_size = args.img_size
    elif "mnist" in str(args.data_root).lower():
        effective_img_size = 28
    else:
        effective_img_size = 64
    
    total_seeds = args.end_seed - args.start_seed + 1
    total_runs = total_seeds * len(args.variants)
    
    print("\n" + "=" * 76, flush=True)
    print("         AUTOMATIZACIÓN DE EXPERIMENTOS GREEN DEEPGA", flush=True)
    print("=" * 76, flush=True)
    print(f"📌 Rango de Semillas:        {args.start_seed} a {args.end_seed} ({total_seeds} ciclos)", flush=True)
    print(f"📌 Execution Inicial:        {args.start_exec} (incrementa de 1 en 1 por variante)", flush=True)
    print(f"📌 Variantes por Ciclo:      {', '.join(args.variants)}", flush=True)
    print(f"📌 Dataset / Canales:        {args.data_root} | {args.in_channels} canal(es)", flush=True)
    print(f"📌 Resolución de Imágenes:   {effective_img_size}x{effective_img_size}", flush=True)
    print(f"📌 Población / Generaciones: {args.pop_size} individuos | {args.generations} generaciones", flush=True)
    print(f"📌 Épocas (GA / Final):      {args.train_epochs} épocas | {args.final_epochs} épocas", flush=True)
    print(f"📌 Directorio Checkpoints:   {args.chck_dir}", flush=True)
    print(f"📌 Total de Corridas:        {total_runs} experimentos", flush=True)
    print(f"📌 Intérprete Python:        {sys.executable}", flush=True)
    print("=" * 76 + "\n", flush=True)
    
    current_exec = args.start_exec
    run_idx = 0
    successful_runs = 0
    failed_runs = 0
    failed_experiments = []
    start_time_all = time.time()
    
    for seed in range(args.start_seed, args.end_seed + 1):
        print("\n" + "#" * 76, flush=True)
        print(f"   [CICLO DE EXPERIMENTACIÓN] SEMILLA ACTUAL = {seed} (Ciclo {seed - args.start_seed + 1}/{total_seeds})", flush=True)
        print("#" * 76, flush=True)
        
        for variant in args.variants:
            run_idx += 1
            print("\n" + "-" * 76, flush=True)
            print(f"▶ [Corrida {run_idx}/{total_runs}] Variante: {variant.upper()} | Semilla: {seed} | Execution: {current_exec}", flush=True)
            print("-" * 76, flush=True)
            
            cmd = [
                sys.executable, "ejemplo_local.py",
                "--execution", str(current_exec),
                "--variant", variant,
                "--seed", str(seed),
                "--data-root", args.data_root,
                "--img-size", str(effective_img_size),
                "--pop-size", str(args.pop_size),
                "--generations", str(args.generations),
                "--in-channels", str(args.in_channels),
                "--train-epochs", str(args.train_epochs),
                "--final-epochs", str(args.final_epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--w", str(args.w),
                "--t-size", str(args.t_size),
                "--cr", str(args.cr),
                "--mr", str(args.mr),
                "--mr-min", str(args.mr_min),
                "--mr-max", str(args.mr_max),
                "--pool-candidates-factor", str(args.pool_candidates_factor),
                "--kappa", str(args.kappa),
                "--rho", str(args.rho),
                "--alpha", str(args.alpha),
                "--top-k-ratio", str(args.top_k_ratio),
                "--n-islands", str(args.n_islands),
                "--migration-interval", str(args.migration_interval),
                "--migration-size", str(args.migration_size),
                "--target-diversity", str(args.target_diversity),
                "--stagnation-limit", str(args.stagnation_limit),
                "--chck-dir", args.chck_dir,
                "--country-iso", args.country_iso
            ]
            
            ret = subprocess.run(cmd)
            
            if ret.returncode == 0:
                successful_runs += 1
                print(f"\n✅ [OK] Variante {variant.upper()} finalizada con éxito (Semilla: {seed}, Exec: {current_exec}).", flush=True)
            else:
                failed_runs += 1
                cmd_str = " ".join([f'"{c}"' if " " in str(c) else str(c) for c in cmd])
                failed_item = {
                    "run_idx": run_idx,
                    "variant": variant,
                    "seed": seed,
                    "execution": current_exec,
                    "returncode": ret.returncode,
                    "cmd_str": cmd_str
                }
                failed_experiments.append(failed_item)
                print(f"\n❌ [ERROR] Falló la corrida de {variant.upper()} con Semilla {seed} y Exec {current_exec} (Código: {ret.returncode}).", flush=True)
            
            current_exec += 1

    elapsed_total = time.time() - start_time_all
    print("\n" + "=" * 76, flush=True)
    print("                    RESUMEN DE LA AUTOMATIZACIÓN", flush=True)
    print("=" * 76, flush=True)
    print(f"  Total de Corridas Realizadas: {run_idx}", flush=True)
    print(f"  Corridas Exitosas:            {successful_runs}", flush=True)
    print(f"  Corridas Fallidas:            {failed_runs}", flush=True)
    print(f"  Tiempo Total Transcurrido:    {elapsed_total / 60.0:.2f} minutos", flush=True)
    print(f"  Última Ejecución Utilizada:   {current_exec - 1}", flush=True)

    if failed_experiments:
        print("\n" + "!" * 76, flush=True)
        print("  ⚠️ DETALLE DE LOS EXPERIMENTOS QUE FALLARON:", flush=True)
        print("!" * 76, flush=True)
        for item in failed_experiments:
            print(f"  ❌ Corrida #{item['run_idx']}: Variante {item['variant'].upper()} | Semilla: {item['seed']} | Execution: {item['execution']} (Código error: {item['returncode']})", flush=True)
            print(f"     Comando para re-ejecutar solo este:", flush=True)
            print(f"     python ejemplo_local.py --execution {item['execution']} --variant {item['variant']} --seed {item['seed']} --data-root {args.data_root} --pop-size {args.pop_size} --generations {args.generations} --in-channels {args.in_channels}\n", flush=True)
        print("!" * 76, flush=True)

    print(f"\n  Archivos generados en: {args.chck_dir}", flush=True)
    print("    - Reportes con texto:             exp_*.txt y reporte_*.txt", flush=True)
    print("    - Reportes SOLO VALORES (Excel):  exp_*_values.txt y exp_*_values.csv", flush=True)
    print("    - Resumen acumulativo TSV:        experiments_summary_values.txt", flush=True)
    print("    - Resumen acumulativo Excel CSV:  experiments_summary_values.csv", flush=True)
    print("=" * 76 + "\n", flush=True)


if __name__ == "__main__":
    main()
