# -*- coding: utf-8 -*-
"""
CLI de DeepGA (Interfaz por Línea de Comandos).
Permite ejecutar y monitorear la optimización evolutiva V10 directamente desde la terminal,
así como lanzar la interfaz gráfica web.
"""

import sys
import os
import argparse
import subprocess
from deepga import __version__
from deepga.config import DeepGAV10Config


def cmd_v10_info(args):
    """Muestra detalles técnicos y parámetros de DeepGA V10."""
    print("=" * 65)
    print("🧠 DeepGA V10 (ACO-Enhanced Pheromone-Guided Evolution)")
    print("=" * 65)
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
    default_cfg = DeepGAV10Config()
    print("Parámetros por defecto:")
    for k, v in default_cfg.to_dict().items():
        print(f"  --{k.replace('_', '-')}: {v}")
    print("=" * 65)


def cmd_v10_run(args):
    """Ejecuta una búsqueda evolutiva con DeepGA V10."""
    from deepga import DeepGASearch

    # Cargar configuración base o crear una nueva
    if args.config and os.path.exists(args.config):
        print(f"📄 Cargando configuración desde '{args.config}'...")
        cfg = DeepGAV10Config.load_json(args.config)
    else:
        cfg = DeepGAV10Config()

    # Sobrescribir con banderas de la CLI si se proporcionaron
    if args.generations is not None:
        cfg.generations = args.generations
    if args.pop_size is not None:
        cfg.pop_size = args.pop_size
    if args.train_epochs is not None:
        cfg.train_epochs = args.train_epochs
    if args.final_epochs is not None:
        cfg.final_epochs = args.final_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    print(f"🔧 Configuración lista: Pop={cfg.pop_size}, Gens={cfg.generations}, Device={cfg.device}")
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
    p_info = subparsers.add_parser("info", help="Ver detalles y parámetros de DeepGA V10")
    p_info.set_defaults(func=cmd_v10_info)

    # Comando 'run'
    p_run = subparsers.add_parser("run", help="Ejecutar búsqueda evolutiva DeepGA V10")
    p_run.add_argument("--dataset", type=str, default=None, help="Ruta a la carpeta del dataset o 'cifar10'")
    p_run.add_argument("--config", type=str, default=None, help="Ruta a un archivo config.json")
    p_run.add_argument("--generations", "-g", type=int, default=None, help="Número de generaciones")
    p_run.add_argument("--pop-size", "-p", type=int, default=None, help="Tamaño de la población")
    p_run.add_argument("--train-epochs", "-e", type=int, default=None, help="Épocas de evaluación rápida")
    p_run.add_argument("--final-epochs", type=int, default=None, help="Épocas de re-entrenamiento final")
    p_run.add_argument("--batch-size", "-b", type=int, default=None, help="Tamaño del batch")
    p_run.add_argument("--lr", type=float, default=None, help="Learning rate")
    p_run.add_argument("--device", type=str, default=None, help="Dispositivo ('cuda', 'cpu' o 'auto')")
    p_run.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    p_run.add_argument("--output-dir", type=str, default=None, help="Directorio para guardar resultados")
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
