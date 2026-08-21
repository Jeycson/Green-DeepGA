"""                                                                                                          
Automatización de Experimentos para ResNet-18:                                                               
- Itera sobre semillas desde --start-seed hasta --end-seed.                                                  
- Incrementa execution consecutivamente.                                                                     
- Registra métricas normalizadas en las mismas tablas acumulativas que DeepGA.                               
"""                                                                                                          
                                                                                                                
import sys                                                                                                   
import subprocess                                                                                            
import argparse                                                                                              
import time                                                                                                  
                                                                                                                
                                                                                                                
def parse_args():                                                                                            
    parser = argparse.ArgumentParser(description="Automatización de experimentos ResNet-18")                 
    parser.add_argument("--start-seed", type=int, default=104,                                               
                        help="Semilla inicial (default: 104)")                                               
    parser.add_argument("--end-seed", type=int, default=109,                                                 
                        help="Semilla final (default: 109)")                                                 
    parser.add_argument("--start-exec", type=int, default=200,                                               
                        help="Número de execution inicial (default: 200)")                                   
    parser.add_argument("--data-root", type=str, default="./Datasets/Covid",                                 
                        help="Ruta al dataset (default: ./Datasets/Covid)")                                  
    parser.add_argument("--img-size", type=int, default=None,                                                
                        help="Resolución de imágenes (default: 28 para MNIST/MedMNIST, 64 otros)")           
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 3],                                
                        help="Canales de entrada: 1 para Covid/Grayscale, 3 para RGB")                       
    parser.add_argument("--epochs", type=int, default=10,                                                    
                        help="Épocas de entrenamiento por corrida (default: 10)")                            
    parser.add_argument("--batch-size", type=int, default=32,                                                
                        help="Batch size (default: 32)")                                                     
    parser.add_argument("--lr", type=float, default=1e-4,                                                    
                        help="Learning rate (default: 1e-4)")                                                
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/",                                    
                        help="Directorio de checkpoints y reportes (default: ./checkpoints/)")               
    parser.add_argument("--country-iso", type=str, default="MEX",                                            
                        help="Código ISO del país para huella de carbono")                                   
    return parser.parse_args()                                                                               
                                                                                                                
                                                                                                                
def main():                                                                                                  
    args = parse_args()                                                                                      
                                                                                                                
    # Detección automática de resolución si no se especifica                                                 
    if args.img_size is not None:                                                                            
        effective_img_size = args.img_size                                                                   
    elif "mnist" in str(args.data_root).lower():                                                             
        effective_img_size = 28                                                                              
    else:                                                                                                    
        effective_img_size = 64                                                                              
                                                                                                                
    total_runs = args.end_seed - args.start_seed + 1                                                         
                                                                                                                
    print("\n" + "=" * 76, flush=True)                                                                       
    print("         AUTOMATIZACIÓN DE EXPERIMENTOS BASELINE - RESNET-18", flush=True)                        
    print("=" * 76, flush=True)                                                                              
    print(f"📌 Rango de Semillas:        {args.start_seed} a {args.end_seed} ({total_runs} corridas)",       
flush=True)                                                                                                    
    print(f"📌 Execution Inicial:        {args.start_exec}", flush=True)                                     
    print(f"📌 Dataset / Canales:        {args.data_root} | {args.in_channels} canal(es)", flush=True)       
    print(f"📌 Resolución:               {effective_img_size}x{effective_img_size}", flush=True)             
    print(f"📌 Épocas por Corrida:       {args.epochs}", flush=True)                                         
    print(f"📌 Checkpoints / Reportes:   {args.chck_dir}", flush=True)                                       
    print("=" * 76 + "\n", flush=True)                                                                       
                                                                                                                
    current_exec = args.start_exec                                                                           
    run_idx = 0                                                                                              
    successful_runs = 0                                                                                      
    failed_runs = 0                                                                                          
    start_time_all = time.time()                                                                             
                                                                                                                
    for seed in range(args.start_seed, args.end_seed + 1):                                                   
        run_idx += 1                                                                                         
        print("\n" + "#" * 76, flush=True)                                                                   
        print(f"▶ [Corrida {run_idx}/{total_runs}] ResNet-18 | Semilla: {seed} | Execution: {current_exec}", 
flush=True)                                                                                                    
        print("#" * 76, flush=True)                                                                          
                                                                                                                
        cmd = [                                                                                              
            sys.executable, "ejemplo_resnet18.py",                                                           
            "--execution", str(current_exec),                                                                
            "--seed", str(seed),                                                                             
            "--data-root", args.data_root,                                                                   
            "--img-size", str(effective_img_size),                                                           
            "--in-channels", str(args.in_channels),                                                          
            "--epochs", str(args.epochs),                                                                    
            "--batch-size", str(args.batch_size),                                                            
            "--lr", str(args.lr),                                                                            
            "--chck-dir", args.chck_dir,                                                                     
            "--country-iso", args.country_iso                                                                
        ]                                                                                                    
                                                                                                                
        ret = subprocess.run(cmd)                                                                            
                                                                                                                
        if ret.returncode == 0:                                                                              
            successful_runs += 1                                                                             
            print(f"\n✅ [OK] ResNet-18 finalizada con éxito (Semilla: {seed}, Exec: {current_exec}).",      
flush=True)                                                                                                    
        else:                                                                                                
            failed_runs += 1                                                                                 
            print(f"\n❌ [ERROR] Falló ResNet-18 con Semilla {seed} (Código: {ret.returncode}).", flush=True)
                                                                                                                
        current_exec += 1                                                                                    
                                                                                                                
    elapsed_total = time.time() - start_time_all                                                             
    print("\n" + "=" * 76, flush=True)                                                                       
    print("                    RESUMEN DE LA AUTOMATIZACIÓN", flush=True)                                    
    print("=" * 76, flush=True)                                                                              
    print(f"  Total de Corridas:         {run_idx}", flush=True)                                             
    print(f"  Corridas Exitosas:         {successful_runs}", flush=True)                                     
    print(f"  Corridas Fallidas:         {failed_runs}", flush=True)                                         
    print(f"  Tiempo Total:              {elapsed_total / 60.0:.2f} minutos", flush=True)                    
    print(f"  Archivos generados en:     {args.chck_dir}", flush=True)                                       
    print("    - Resumen acumulativo TSV: experiments_summary_values.txt", flush=True)                       
    print("    - Resumen acumulativo CSV: experiments_summary_values.csv", flush=True)                       
    print("=" * 76 + "\n", flush=True)                                                                       
                                                                                                                
                                                                                                                
if __name__ == "__main__":                                                                                   
    main()             