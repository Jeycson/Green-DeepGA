# -*- coding: utf-8 -*-
"""
Script para Re-entrenar un Modelo Ganador Erolucionado Previamente Guardado:
----------------------------------------------------------------------------
Permite cargar un archivo .pth (o .pkl) generado por DeepGA (ej. best_model_v10_exec_43.pth)
y entrenarlo completamente por las épocas deseadas (ej. 10, 20, 30 épocas),
sin tener que volver a ejecutar todo el algoritmo genético.

Uso rápido:
    python entrenar_modelo_guardado.py --model-path ./checkpoints/best_model_v10_exec_43.pth --epochs 10

Uso con dataset personalizado o 2-split:
    python entrenar_modelo_guardado.py --model-path ./checkpoints/best_model_v10_exec_43.pth --epochs 15 --batch-size 64 --lr 1e-4 --use-2split
"""

import os
import argparse
import timeit

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from model_utils import load_saved_model, evaluate_model, generate_confusion_matrix, save_best_model
    from dataset_loader import load_dataset_auto, load_dataset_2split
    from DistributedTraining import training
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento completo de un modelo DeepGA guardado (.pth / .pkl)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Ruta al archivo .pth o .pkl del modelo guardado (ej. ./checkpoints/best_model_v10_exec_43.pth)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Número de épocas de entrenamiento (por defecto: 10)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño del lote / batch size (por defecto: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (por defecto: 1e-4)")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Ruta al dataset (ej. ./data para CIFAR-10 o ./dataset/covid)")
    parser.add_argument("--use-2split", action="store_true", default=False,
                        help="Usa partición de 2 conjuntos (Train y Val) con más datos para entrenamiento")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fracción para validación en modo 2-split (por defecto: 0.15)")
    parser.add_argument("--no-preload-gpu", action="store_true", default=False,
                        help="Desactiva precarga completa en VRAM")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70, flush=True)
    print("      RE-ENTRENAMIENTO DE MODELO DEEPGA GUARDADO (.PTH)", flush=True)
    print("=" * 70, flush=True)
    print(f"📌 Dispositivo:            {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU)"), flush=True)
    print(f"📌 Archivo de Modelo:      {os.path.abspath(args.model_path)}", flush=True)
    print(f"📌 Épocas a Entrenar:      {args.epochs} épocas", flush=True)
    print(f"📌 Learning Rate (LR):     {args.lr}", flush=True)
    print(f"📌 Dataset:                {os.path.abspath(args.data_root)}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # 1. Cargar el modelo reconstruido a partir del genoma
    model, checkpoint = load_saved_model(args.model_path, device=device)
    genome = checkpoint["genome"]
    in_channels = checkpoint.get("in_channels", 3)
    out_size = checkpoint.get("out_size", 32)
    n_classes = checkpoint.get("n_classes", 10)
    variant = checkpoint.get("variant", "v10")
    execution = checkpoint.get("execution", 1)

    print(f"✓ Modelo {variant.upper()} reconstruido exitosamente a partir del genoma guardado.")
    print(f"  - Capas Convolucionales: {genome.n_conv} | Capas Densas (FC): {genome.n_full}")
    print(f"  - Resolución de entrada: {out_size}x{out_size} px | Canales: {in_channels} | Clases: {n_classes}\n")

    # 2. Cargar DataLoaders
    if args.use_2split:
        train_dl, val_dl, in_c, img_s, n_c, class_names = load_dataset_2split(
            data_root=args.data_root,
            img_size=out_size,
            in_channels=in_channels,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            preload_gpu=not args.no_preload_gpu,
            device=device
        )
        test_dl = None
    else:
        train_dl, val_dl, test_dl, in_c, img_s, n_c, class_names = load_dataset_auto(
            data_root=args.data_root,
            img_size=out_size,
            in_channels=in_channels,
            batch_size=args.batch_size,
            preload_gpu=not args.no_preload_gpu,
            device=device
        )

    # Si hay conjunto de test independiente, validamos sobre test_dl; si es 2-split, sobre val_dl
    eval_dl = test_dl if test_dl is not None else val_dl
    eval_name = "Test Independiente (10,000 imágenes)" if test_dl is not None else "Validación (2-Split)"

    # 3. Entrenar el modelo
    loss_func = nn.NLLLoss()
    acc_list = []
    print(f"🚀 Iniciando entrenamiento por {args.epochs} épocas...")
    print(f"   (Evaluación por época sobre: {eval_name})\n", flush=True)

    start_t = timeit.default_timer()
    fit_val, final_acc, pars, trained_model = training(
        '1', device, model, args.epochs, loss_func, train_dl, eval_dl, args.lr, 0.0, 2e6, acc_list
    )
    elapsed_s = timeit.default_timer() - start_t

    print("\n" + "=" * 70, flush=True)
    print("              RESULTADOS DEL ENTRENAMIENTO COMPLETO", flush=True)
    print("=" * 70, flush=True)
    print(f"🎯 Precisión Final ({eval_name}): {final_acc:.2f}%")
    print(f"🧠 Parámetros de la Red:           {pars:,}")
    print(f"⏱️ Tiempo de Entrenamiento:         {elapsed_s:.2f} s ({elapsed_s/60.0:.2f} min)")
    print("=" * 70 + "\n", flush=True)

    # 4. Guardar el modelo con los nuevos pesos entrenados
    chck_dir = os.path.dirname(os.path.abspath(args.model_path))
    trained_model_path = os.path.join(chck_dir, f"trained_model_{variant}_exec_{execution}_{args.epochs}epochs.pth")

    bestind = [genome, fit_val, final_acc, pars]
    save_best_model(
        variant=f"{variant}_trained_{args.epochs}epochs",
        execution=execution,
        bestind=bestind,
        in_channels=in_channels,
        out_size=out_size,
        n_classes=n_classes,
        chck_dir=chck_dir,
        trained_model=trained_model
    )
    print(f"💾 Modelo con pesos entrenados guardado en: {trained_model_path}", flush=True)

    # 5. Generar Matriz de Confusión
    cm_path = os.path.join(chck_dir, f"confusion_matrix_trained_{variant}_exec_{execution}.png")
    generate_confusion_matrix(
        model_or_path=trained_model,
        dataloader=eval_dl,
        class_names=class_names,
        device=device,
        title=f"Matriz de Confusión ({eval_name}) - {variant.upper()}",
        save_fig_path=cm_path
    )
    print(f"📊 Matriz de Confusión guardada en: {os.path.abspath(cm_path)}", flush=True)
    print("\n✨ Proceso de re-entrenamiento finalizado con éxito.")


if __name__ == "__main__":
    main()
