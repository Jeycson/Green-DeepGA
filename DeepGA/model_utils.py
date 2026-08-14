# -*- coding: utf-8 -*-
"""
Módulo de utilidades para:
1. Guardar, exportar y descargar el modelo ganador de cada versión de DeepGA.
2. Cargar modelos guardados para inferencia con imágenes propias.
3. Generar matrices de confusión y reportes de métricas detallados.
4. Empaquetar y descargar todos los modelos en un archivo ZIP.
"""

import os
import shutil
import zipfile
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from Decoding import decoding, CNN


def is_colab() -> bool:
    """Verifica si el código se está ejecutando dentro de un entorno Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def download_file(file_path: str) -> bool:
    """
    Descarga automáticamente un archivo si se ejecuta en Google Colab.
    Si se ejecuta de forma local, muestra la ruta absoluta del archivo generado.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ El archivo '{file_path}' no existe para descargar.")
        return False

    abs_path = os.path.abspath(file_path)
    if is_colab():
        try:
            from google.colab import files
            print(f"⬇️ Iniciando descarga en navegador (Colab): {os.path.basename(file_path)}...")
            files.download(file_path)
            return True
        except Exception as e:
            print(f"ℹ️ Archivo listo en Colab ({abs_path}). (Nota: {e})")
            return False
    else:
        print(f"ℹ️ Archivo guardado localmente en: {abs_path}")
        return True


def save_best_model(
    variant: str,
    execution: int,
    bestind: list,
    in_channels: int = 3,
    out_size: int = 32,
    n_classes: int = 10,
    chck_dir: str = "./checkpoints/",
    trained_model: nn.Module = None,
    cnn_metrics: dict = None,
    auto_download: bool = False,
    data_root: str = None
) -> str:
    """
    Guarda la configuración (genoma), metadatos y pesos (si están disponibles)
    del mejor modelo de una variante de DeepGA en formato .pth y .pkl.
    """
    os.makedirs(chck_dir, exist_ok=True)
    var_str = str(variant).lower()
    model_filename = f"best_model_{var_str}_exec_{execution}.pth"
    model_path = os.path.join(chck_dir, model_filename)

    state_dict = None
    if trained_model is not None:
        state_dict = copy.deepcopy(trained_model.state_dict())

    checkpoint_data = {
        "variant": var_str,
        "execution": execution,
        "genome": bestind[0],
        "fitness": float(bestind[1]) if len(bestind) > 1 else None,
        "accuracy": float(bestind[2]) if len(bestind) > 2 else None,
        "params": int(bestind[3]) if len(bestind) > 3 else None,
        "in_channels": in_channels,
        "out_size": out_size,
        "n_classes": n_classes,
        "data_root": data_root,
        "cnn_metrics": cnn_metrics or {},
        "state_dict": state_dict
    }

    # Guardar archivo estándar PyTorch
    torch.save(checkpoint_data, model_path)

    # Guardar también archivo pkl de respaldo con el modelo completo si está entrenado
    pkl_filename = f"best_model_{var_str}_exec_{execution}.pkl"
    pkl_path = os.path.join(chck_dir, pkl_filename)
    with open(pkl_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"💾 [DeepGA {var_str.upper()}] Modelo ganador guardado con éxito en:")
    print(f"   -> PyTorch Checkpoint: {os.path.abspath(model_path)}")

    if auto_download:
        download_file(model_path)

    return model_path


def load_saved_model(model_path: str, device: torch.device = None):
    """
    Carga un modelo exportado (.pth o .pkl), reconstruye su arquitectura exacta
    a partir del genoma guardado y carga sus pesos.
    
    Retorna:
        model (torch.nn.Module): Modelo reconstruido en modo eval()
        checkpoint_data (dict): Diccionario con genoma, métricas y metadatos
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo de modelo: {model_path}")

    # Carga compatible con .pth y .pkl
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
    else:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception:
            checkpoint = torch.load(model_path, map_location=device)

    genome = checkpoint["genome"]
    in_channels = checkpoint.get("in_channels", 3)
    out_size = checkpoint.get("out_size", 32)
    n_classes = checkpoint.get("n_classes", 10)

    # Reconstruir la red convolucional
    network = decoding(genome, in_channels, out_size, n_classes)
    model = CNN(genome, network[0], network[1], network[2])

    if checkpoint.get("state_dict") is not None:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"✅ Pesos cargados correctamente para variante {checkpoint.get('variant', '').upper()}.")
    else:
        print(f"⚠️ El checkpoint contiene la arquitectura pero no pesos preentrenados finales.")

    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_model(
    model,
    dataloader,
    device: torch.device = None
):
    """Evalúa un modelo sobre un DataLoader."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, (list, tuple)):
                xb, yb = data[0].to(device), data[1].to(device)
            else:
                xb, yb = data["image"].to(device), data["label"].to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    return 100 * correct / total


def predict_image(
    model_or_path,
    image_path: str,
    class_names: list = None,
    device: torch.device = None,
    in_channels: int = 3,
    out_size: int = 32,
    custom_transform: transforms.Compose = None
):
    """
    Realiza inferencia sobre una imagen propia usando un modelo o ruta de checkpoint.
    
    Retorna:
        predicted_class (str/int): Clase predicha
        confidence (float): Probabilidad porcentual (0-100%)
        probabilities (numpy.ndarray): Vector de probabilidades para todas las clases
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model_or_path, str):
        model, chk = load_saved_model(model_or_path, device=device)
        in_channels = chk.get("in_channels", in_channels)
        out_size = chk.get("out_size", out_size)
    else:
        model = model_or_path
        model.to(device)
        model.eval()

    if custom_transform is None:
        transform_steps = [
            transforms.Resize((out_size, out_size)),
            transforms.ToTensor()
        ]
        if in_channels == 3:
            transform_steps.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        custom_transform = transforms.Compose(transform_steps)

    img = Image.open(image_path)
    if in_channels == 3:
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    img_tensor = custom_transform(img).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.exp(output) if torch.any(output < 0) else torch.softmax(output, dim=1)
        probs_np = probs.squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_idx] * 100.0)

    pred_label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else pred_idx
    return pred_label, confidence, probs_np


def generate_confusion_matrix(
    model_or_path,
    dataloader,
    class_names: list = None,
    device: torch.device = None,
    title: str = "Matriz de Confusión - DeepGA",
    save_fig_path: str = None,
    auto_download_plot: bool = False
):
    """
    Calcula y grafica la matriz de confusión sobre un DataLoader de prueba.
    Genera el reporte de clasificación (Precision, Recall, F1-Score).
    
    Retorna:
        cm (numpy.ndarray): Matriz de confusión
        report_dict (dict): Reporte de clasificación estructurado
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model_or_path, str):
        model, _ = load_saved_model(model_or_path, device=device)
    else:
        model = model_or_path
        model.to(device)
        model.eval()

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, (list, tuple)):
                xb, yb = data[0], data[1]
            elif isinstance(data, dict):
                xb, yb = data["image"], data["label"]
            else:
                continue

            xb = xb.to(device, dtype=torch.float32)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            if torch.is_tensor(yb):
                y_true.extend(yb.cpu().numpy())
            else:
                y_true.extend(yb)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if class_names is None:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [f"Clase {c}" for c in unique_classes]

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    # Graficar
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel("Predicción del Modelo", fontsize=12, fontweight="bold")
    plt.ylabel("Etiqueta Real", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig_path)), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300)
        print(f"📊 Gráfica de matriz de confusión guardada en: {save_fig_path}")
        if auto_download_plot:
            download_file(save_fig_path)

    plt.show()

    print("\n" + "=" * 65)
    print(f"     REPORTE DE CLASIFICACIÓN ({title})")
    print("=" * 65)
    print(report_text)

    return cm, report


def download_all_models_zip(
    chck_dir: str = "./checkpoints/",
    zip_name: str = "deepga_best_models.zip",
    auto_download: bool = True
) -> str:
    """
    Empaqueta todos los modelos .pth y .pkl encontrados en chck_dir en un archivo .ZIP
    y permite su descarga automática.
    """
    if not os.path.exists(chck_dir):
        print(f"⚠️ El directorio {chck_dir} no existe.")
        return None

    zip_path = os.path.join(chck_dir, zip_name) if not zip_name.startswith("/") else zip_name
    files_to_zip = [
        f for f in os.listdir(chck_dir)
        if (f.endswith(".pth") or f.endswith(".pkl") or f.endswith(".png") or f.endswith(".csv")) and f != zip_name
    ]

    if not files_to_zip:
        print(f"⚠️ No se encontraron archivos de modelos (.pth / .pkl) en {chck_dir}.")
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            full_path = os.path.join(chck_dir, file)
            zipf.write(full_path, arcname=file)

    print(f"📦 Todos los modelos y reportes han sido empaquetados en: {os.path.abspath(zip_path)} ({len(files_to_zip)} archivos)")

    if auto_download:
        download_file(zip_path)

    return zip_path


def plot_pareto_frontier(
    pareto_front: list,
    all_evaluated_history: list = None,
    title: str = "Frente de Pareto MO-DeepGA (Precisión vs Huella de Carbono)",
    save_fig_path: str = None,
    auto_download_plot: bool = False
):
    """
    Grafica el Frente de Pareto 2D (Precisión vs Huella de Carbono gCO2eq).
    Resalta los 3 modelos representativos de la frontera:
    - 🏆 Modelo de Máxima Precisión (Best Accuracy)
    - 🌿 Modelo Más Verde / Menor Huella de Carbono (Greenest / Ultra-Low Carbon)
    - ⚖️ Modelo Equilibrado (Knee Point / Compromiso Óptimo)
    """
    if len(pareto_front) == 0:
        print("⚠️ No hay individuos en el Frente de Pareto para graficar.")
        return None

    plt.figure(figsize=(10, 6.5))

    # 1. Graficar todas las arquitecturas exploradas (si están disponibles)
    if all_evaluated_history and len(all_evaluated_history) > 0:
        hist_acc = [item[1] for item in all_evaluated_history]
        hist_carb = [item[2] for item in all_evaluated_history]
        plt.scatter(hist_carb, hist_acc, c="#94a3b8", alpha=0.45, s=35, label="Arquitecturas Evaluadas (Espacio de Búsqueda)")

    # 2. Ordenar Frente de Pareto por huella de carbono ascendente
    pf_sorted = sorted(pareto_front, key=lambda x: x[2])
    pf_carb = [p[2] for p in pf_sorted]
    pf_acc = [p[1] for p in pf_sorted]

    # Conectar el frente con línea punteada
    plt.plot(pf_carb, pf_acc, color="#059669", linestyle="--", linewidth=2.0, alpha=0.8, label="Frontera de Pareto (F1)")
    plt.scatter(pf_carb, pf_acc, color="#10b981", s=70, edgecolors="#047857", linewidth=1.5, zorder=3)

    # 3. Identificar puntos clave
    best_acc_ind = max(pareto_front, key=lambda x: x[1])
    greenest_ind = min(pareto_front, key=lambda x: x[2])

    # Knee Point
    all_accs = [p[1] for p in pareto_front]
    all_carbs = [p[2] for p in pareto_front]
    min_a, max_a = min(all_accs), max(all_accs)
    min_c, max_c = min(all_carbs), max(all_carbs)
    range_a = max(1e-5, max_a - min_a)
    range_c = max(1e-5, max_c - min_c)

    knee_ind = pareto_front[0]
    min_dist = float('inf')
    for ind in pareto_front:
        norm_a = (max_a - ind[1]) / range_a
        norm_c = (ind[2] - min_c) / range_c
        dist = np.sqrt(norm_a**2 + norm_c**2)
        if dist < min_dist:
            min_dist = dist
            knee_ind = ind

    # 4. Destacar los 3 modelos clave con marcadores especiales y anotaciones
    # Máxima Precisión
    plt.scatter([best_acc_ind[2]], [best_acc_ind[1]], color="#f59e0b", s=180, marker="*", edgecolors="#b45309", linewidth=2, zorder=5, label=f"🏆 Máx. Precisión ({best_acc_ind[1]:.2f}%, {best_acc_ind[2]:.4f}g)")
    plt.annotate(f"🏆 Máx. Precisión\n{best_acc_ind[1]:.2f}% | {best_acc_ind[2]:.3f} gCO2",
                 (best_acc_ind[2], best_acc_ind[1]),
                 textcoords="offset points", xytext=(10, 10),
                 fontsize=9, fontweight="bold", color="#92400e",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#f59e0b", lw=1))

    # Más Verde
    plt.scatter([greenest_ind[2]], [greenest_ind[1]], color="#16a34a", s=140, marker="P", edgecolors="#14532d", linewidth=2, zorder=5, label=f"🌿 Más Verde ({greenest_ind[1]:.2f}%, {greenest_ind[2]:.4f}g)")
    plt.annotate(f"🌿 Más Verde\n{greenest_ind[1]:.2f}% | {greenest_ind[2]:.3f} gCO2",
                 (greenest_ind[2], greenest_ind[1]),
                 textcoords="offset points", xytext=(-20, -25),
                 fontsize=9, fontweight="bold", color="#14532d",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#16a34a", lw=1))

    # Knee Point
    plt.scatter([knee_ind[2]], [knee_ind[1]], color="#0ea5e9", s=140, marker="D", edgecolors="#0369a1", linewidth=2, zorder=5, label=f"⚖️ Equilibrado / Knee ({knee_ind[1]:.2f}%, {knee_ind[2]:.4f}g)")
    plt.annotate(f"⚖️ Compromiso Knee\n{knee_ind[1]:.2f}% | {knee_ind[2]:.3f} gCO2",
                 (knee_ind[2], knee_ind[1]),
                 textcoords="offset points", xytext=(10, -20),
                 fontsize=9, fontweight="bold", color="#0369a1",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e0f2fe", ec="#0ea5e9", lw=1))

    plt.xlabel("Huella de Carbono por Entrenamiento (gCO2eq)  [↓ Minimizar]", fontsize=11, fontweight="bold")
    plt.ylabel("Precisión en Validación (%)  [↑ Maximizar]", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig_path)), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300)
        print(f"📊 Gráfico del Frente de Pareto guardado en: {os.path.abspath(save_fig_path)}")
        if auto_download_plot:
            download_file(save_fig_path)

    plt.show()
    return save_fig_path

