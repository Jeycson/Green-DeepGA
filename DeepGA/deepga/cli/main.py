# -*- coding: utf-8 -*-
"""
CLI de DeepGA (Interfaz por Línea de Comandos).
Permite ejecutar y monitorear la optimización evolutiva V10 directamente desde la terminal,
así como lanzar la interfaz gráfica web.

Por defecto utiliza la configuración óptima calibrada por irace (best_configuration_4.json).
"""

import sys
import os
import argparse
import subprocess
from deepga import __version__
from deepga.config import DeepGAV10Config


def cmd_v10_info(args):
    """Muestra detalles técnicos y parámetros óptimos de DeepGA V10."""
    print("=" * 68)
    print("🧠 DeepGA V10 (ACO-Enhanced Pheromone-Guided Evolution)")
    print("   [Configuración Óptima por Defecto: irace best_configuration_4]")
    print("=" * 68)
    print("""
Características principales de la Versión 10:
  1. Matriz de Feromonas Arquitectónicas (ACO-NAS Pheromone Matrix):
     Registra las frecuencias de transiciones y tipos de capas exitosas.
  2. Depósito y Evaporación Élite:
     Los mejores individuos refuerzan los caminos de feromona con tasa rho.
  3. Mutación Guiada por Probabilidad (Softmax/Boltzmann sobre Feromonas).
  4. Modelo Subrogado Asistido:
     Random Forest + Upper Confidence Bound (UCB) para pre-filtrar candidatos.
  5. Cruce Topológico Emparejado V7 (Graph-Based Coherent Crossover).
  6. Mutación Adaptativa Individual (Srinivas & Patnaik).
    """)
    default_cfg = DeepGAV10Config.optimal()
    print("Parámetros Óptimos Calibrados (irace):")
    for k, v in default_cfg.to_dict().items():
        if k != "expert_mode":
            print(f"  --{k.replace('_', '-')}: {v}")
    print("=" * 68)


def cmd_v10_run(args):
    """Ejecuta una búsqueda evolutiva con DeepGA V10 usando configuración óptima."""
    from deepga import DeepGASearch

    # Cargar configuración base (óptima por defecto) o desde JSON
    if args.config and os.path.exists(args.config):
        print(f"📄 Cargando configuración desde '{args.config}'...")
        cfg = DeepGAV10Config.load_json(args.config)
    else:
        cfg = DeepGAV10Config.optimal()

    # Verificar si el usuario intenta alterar parámetros sin modo experto
    overrides = {
        "generations": args.generations,
        "pop_size": args.pop_size,
        "train_epochs": args.train_epochs,
        "final_epochs": args.final_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr
    }
    has_overrides = any(v is not None for v in overrides.values())

    if has_overrides:
        if not args.expert:
            print("\n⚠️  [AVISO]: Has especificado hiperparámetros manuales (--pop-size, --generations, etc.).")
            print("   DeepGA V10 está calibrado para máxima convergencia con la configuración óptima de irace.")
            print("   Para aplicar estas modificaciones conscientemente, añade la bandera '--expert'.")
            print("   Ejecutando con la configuración óptima por defecto...\n")
        else:
            cfg.expert_mode = True
            for k, v in overrides.items():
                if v is not None:
                    setattr(cfg, k, v)
            print("🔧 [Modo Experto Activo]: Hiperparámetros modificados manualmente.")

    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    print(f"🚀 Configuración en uso: Pop={cfg.pop_size}, Gens={cfg.generations}, LR={cfg.learning_rate}, Device={cfg.device}")
    searcher = DeepGASearch(config=cfg)

    # Iniciar búsqueda con dataset personalizado o CIFAR-10
    searcher.fit(dataset=args.dataset)

    # Guardar modelo final si se especificó salida
    model_save_path = os.path.join(cfg.output_dir, "best_model_v10.pt")
    searcher.save(model_save_path)
    print(f"\n🎉 ¡Proceso finalizado con éxito! Resultados en '{cfg.output_dir}'.")


def cmd_ui(args):
    """Lanza la interfaz gráfica interactiva con Streamlit."""
    ui_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui", "app.py"))
    if not os.path.exists(ui_script):
        print(f"❌ Error: No se encontró el script de la UI en '{ui_script}'.")
        sys.exit(1)

    port = args.port or 8501
    print(f"🌐 Iniciando servidor DeepGA Web UI en http://localhost:{port} ...")
    cmd = [sys.executable, "-m", "streamlit", "run", ui_script, "--server.port", str(port)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Servidor UI detenido.")
    except Exception as e:
        print(f"❌ Error al iniciar Streamlit: {e}")
        print("Asegúrate de tener streamlit instalado (`pip install streamlit`).")


def main():
    parser = argparse.ArgumentParser(
        prog="deepga",
        description=f"DeepGA CLI v{__version__} - Optimización de Redes Neuronales y Algoritmos Evolutivos"
    )
    parser.add_argument("-v", "--version", action="version", version=f"DeepGA v{__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # Comando 'info'
    p_info = subparsers.add_parser("info", help="Ver detalles y parámetros óptimos de DeepGA V10")
    p_info.set_defaults(func=cmd_v10_info)

    # Comando 'run'
    p_run = subparsers.add_parser("run", help="Ejecutar búsqueda evolutiva DeepGA V10")
    p_run.add_argument("--dataset", type=str, default=None, help="Ruta a la carpeta del dataset o 'cifar10'")
    p_run.add_argument("--config", type=str, default=None, help="Ruta a un archivo config.json")
    p_run.add_argument("--device", type=str, default=None, help="Dispositivo ('cuda', 'cpu' o 'auto')")
    p_run.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    p_run.add_argument("--output-dir", type=str, default=None, help="Directorio para guardar resultados")

    # Parámetros avanzados / protegidos
    expert_group = p_run.add_argument_group("Opciones Avanzadas (Requiere --expert para aplicar)")
    expert_group.add_argument("--expert", action="store_true", help="Habilita la modificación de la configuración óptima")
    expert_group.add_argument("--generations", "-g", type=int, default=None, help="Número de generaciones (óptimo: 32)")
    expert_group.add_argument("--pop-size", "-p", type=int, default=None, help="Tamaño de la población (óptimo: 17)")
    expert_group.add_argument("--train-epochs", "-e", type=int, default=None, help="Épocas de evaluación rápida (óptimo: 2)")
    expert_group.add_argument("--final-epochs", type=int, default=None, help="Épocas de re-entrenamiento final (óptimo: 9)")
    expert_group.add_argument("--batch-size", "-b", type=int, default=None, help="Tamaño del batch")
    expert_group.add_argument("--lr", type=float, default=None, help="Learning rate (óptimo: 0.0009)")
    p_run.set_defaults(func=cmd_v10_run)

    # Comando 'ui'
    p_ui = subparsers.add_parser("ui", help="Iniciar la interfaz gráfica Web (Streamlit)")
    p_ui.add_argument("--port", type=int, default=8501, help="Puerto para el servidor web (defecto: 8501)")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
