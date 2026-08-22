# -*- coding: utf-8 -*-
"""
Módulo de utilidades para:
1. Guardar, exportar y descargar el modelo ganador de cada versión de DeepGA.
2. Cargar modelos guardados para inferencia con imágenes propias.
3. Generar matrices de confusión y reportes de métricas detallados.
4. Empaquetar y descargar todos los modelos en un archivo ZIP.
"""

import os
import time
import shutil
import zipfile
import pickle
import copy

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from Decoding import decoding, CNN
    DECODING_AVAILABLE = True
except ImportError:
    DECODING_AVAILABLE = False


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
    
    Ajusta automáticamente la resolución (out_size) e in_channels si detecta
    discrepancias dimensionales con los pesos entrenados en state_dict.
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
    state_dict = checkpoint.get("state_dict")

    # Detección infalible de canales desde state_dict
    if state_dict is not None and "extraction.0.weight" in state_dict:
        in_channels = state_dict["extraction.0.weight"].shape[1]

    # Reconstruir la red convolucional base
    network = decoding(genome, in_channels, out_size, n_classes)

    # Reconciliación automática de resolución (out_size) si state_dict tiene dimensión distinta
    if state_dict is not None and "classifier.0.weight" in state_dict:
        target_in_features = state_dict["classifier.0.weight"].shape[1]
        current_in_features = network[1][0].in_features
        if current_in_features != target_in_features:
            matched = False
            for test_res in [32, 64, 28, 128, 256, 16, 48, 96, 224, 40, 56]:
                try:
                    test_net = decoding(genome, in_channels, test_res, n_classes)
                    if test_net[1][0].in_features == target_in_features:
                        out_size = test_res
                        network = test_net
                        matched = True
                        break
                except Exception:
                    continue

            if not matched:
                n_neurons = state_dict["classifier.0.weight"].shape[0]
                network[1][0] = nn.Linear(target_in_features, n_neurons)

    model = CNN(genome, network[0], network[1], network[2])

    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(f"✅ Pesos cargados correctamente para variante {checkpoint.get('variant', '').upper()} ({in_channels} canales, entrada {out_size}x{out_size}).")
    else:
        print(f"⚠️ El checkpoint contiene la arquitectura pero no pesos preentrenados finales.")

    model.to(device)
    model.eval()

    checkpoint["out_size"] = out_size
    checkpoint["in_channels"] = in_channels

    return model, checkpoint


def _adapt_input_tensor(xb: torch.Tensor, model: nn.Module, expected_size: int = None) -> torch.Tensor:
    """Adapta canales y resolución de xb para que coincida exactamente con lo esperado por la CNN."""
    # 1. Adaptar canales detectando la primera capa convolucional del modelo
    req_channels = None
    if hasattr(model, 'conv1') and hasattr(model.conv1, 'in_channels'):
        req_channels = model.conv1.in_channels
    elif hasattr(model, 'features') and len(model.features) > 0:
        first_mod = model.features[0]
        if isinstance(first_mod, (list, nn.Sequential, tuple)) and len(first_mod) > 0:
            first_mod = first_mod[0]
        if hasattr(first_mod, 'in_channels'):
            req_channels = first_mod.in_channels
    elif hasattr(model, 'extraction') and len(model.extraction) > 0:
        first_mod = model.extraction[0]
        if hasattr(first_mod, 'in_channels'):
            req_channels = first_mod.in_channels

    # Si no se detectó por atributos conocidos, buscar el primer Conv2d del modelo
    if req_channels is None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                req_channels = m.in_channels
                break

    if req_channels is None:
        req_channels = xb.shape[1]

    if xb.shape[1] == 3 and req_channels == 1:
        xb = xb.mean(dim=1, keepdim=True)
    elif xb.shape[1] == 1 and req_channels == 3:
        xb = xb.repeat(1, 3, 1, 1)

    # 2. Adaptar resolución espacial si es necesario
    if expected_size is not None and (xb.shape[2] != expected_size or xb.shape[3] != expected_size):
        xb = nn.functional.interpolate(xb, size=(expected_size, expected_size), mode='bilinear', align_corners=False)

    return xb


def evaluate_model(
    model,
    dataloader,
    device: torch.device = None
):
    """Evalúa un modelo sobre un DataLoader con adaptación dimensional automática."""
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

            xb = _adapt_input_tensor(xb, model)
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
            xb = _adapt_input_tensor(xb, model)
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


def calculate_cnn_metrics(bestind: list, in_channels: int, out_size: int, n_classes: int):
    """
    Calcula las variables estructurales y computacionales de la CNN generada:
    - Parámetros totales y entrenables
    - Tamaño estimado en memoria (MB)
    - Estimación de FLOPs
    - Número de capas convolucionales, densas y skip-connections
    """
    genome = bestind[0]
    network = decoding(genome, in_channels, out_size, n_classes)
    model = CNN(genome, network[0], network[1], network[2])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4.0) / (1024.0 * 1024.0)

    # Estimación de FLOPs/MACs
    macs_est = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            macs_est += m.out_channels * m.in_channels * kh * kw * 16 * 16
        elif isinstance(m, nn.Linear):
            macs_est += m.in_features * m.out_features
    flops_est = macs_est * 2

    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    fc_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    skip_count = sum(1 for bit in genome.second_level if bit == 1) if hasattr(genome, "second_level") else 0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 4),
        "estimated_flops": flops_est,
        "conv_layers": conv_count,
        "fc_layers": fc_count,
        "skip_connections": skip_count,
        "model": model
    }


def compute_classification_metrics(model, dataloader, device: torch.device = None):
    """
    Evalúa un modelo sobre un DataLoader y calcula:
    - Accuracy (%)
    - Precision Macro (%)
    - Recall Macro (%)
    - F1-Score Macro (%)
    """
    if dataloader is None or model is None:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, (list, tuple)):
                xb, yb = data[0], data[1]
            elif isinstance(data, dict):
                xb, yb = data["image"], data["label"]
            else:
                continue

            xb = xb.to(device, dtype=torch.float32)
            xb = _adapt_input_tensor(xb, model)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)

            y_pred.extend(preds.cpu().numpy())
            if torch.is_tensor(yb):
                y_true.extend(yb.cpu().numpy())
            else:
                y_true.extend(yb)

    if len(y_true) == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(np.mean(y_true == y_pred) * 100.0)

    try:
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec = float(prec * 100.0)
        rec = float(rec * 100.0)
        f1 = float(f1 * 100.0)
    except Exception:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precs, recs, f1s = [], [], []
        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f = float((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0
            precs.append(p)
            recs.append(r)
            f1s.append(f)
        prec = float(np.mean(precs) * 100.0) if precs else acc
        rec = float(np.mean(recs) * 100.0) if recs else acc
        f1 = float(np.mean(f1s) * 100.0) if f1s else acc

    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2)
    }


def format_experiment_row(data: dict) -> dict:
    """
    Formatea la fila de experimento con los 20 campos especificados:
    Dataset    Método    Seed    Gen    Pop    Mig.    Epoch    Time    Energy    CO₂    Fitness    Val Acc    Test Acc    Precision    Recall    F1    Params    Memory    FLOPs    Evaluations
    """
    dataset = str(data.get("dataset", "CIFAR-10")).strip()
    method = str(data.get("method", "DeepGA")).strip()
    seed = str(data.get("seed", 1)).strip()
    gen = str(data.get("gen", data.get("generations", 1))).strip()
    pop = str(data.get("pop", data.get("population", 12))).strip()
    mig = str(data.get("mig", data.get("migration", "N/A"))).strip()
    epoch = str(data.get("epoch", data.get("train_epochs", 2))).strip()

    # Time (s)
    t_val = data.get("time", data.get("execution_time_seconds", 0.0))
    time_str = f"{float(t_val):.2f}" if isinstance(t_val, (int, float)) else str(t_val)

    # Energy (kWh)
    e_val = data.get("energy", data.get("energy_consumed_kwh", 0.0))
    energy_str = f"{float(e_val):.6f}" if isinstance(e_val, (int, float)) else str(e_val)

    # CO2 (g)
    c_val = data.get("co2", data.get("carbon_emissions_g_co2", 0.0))
    co2_str = f"{float(c_val):.4f}" if isinstance(c_val, (int, float)) else str(c_val)

    # Fitness
    f_val = data.get("fitness", data.get("best_fitness", 0.0))
    fit_str = f"{float(f_val):.4f}" if isinstance(f_val, (int, float)) else str(f_val)

    # Val Acc (%)
    va_val = data.get("val_acc", data.get("best_val_accuracy", 0.0))
    if isinstance(va_val, (int, float)):
        va_float = float(va_val) * 100.0 if float(va_val) <= 1.0 and float(va_val) > 0.0 else float(va_val)
        val_acc_str = f"{va_float:.2f}"
    else:
        val_acc_str = str(va_val)

    # Test Acc (%)
    ta_val = data.get("test_acc", data.get("final_test_accuracy", None))
    if ta_val is not None and str(ta_val).lower() not in ["none", "n/a", ""]:
        ta_float = float(ta_val) * 100.0 if float(ta_val) <= 1.0 and float(ta_val) > 0.0 else float(ta_val)
        test_acc_str = f"{ta_float:.2f}"
    else:
        test_acc_str = "N/A"

    # Precision (%)
    p_val = data.get("precision", None)
    if p_val is not None and str(p_val).lower() not in ["none", "n/a", ""]:
        p_float = float(p_val) * 100.0 if float(p_val) <= 1.0 and float(p_val) > 0.0 else float(p_val)
        prec_str = f"{p_float:.2f}"
    else:
        prec_str = "N/A"

    # Recall (%)
    r_val = data.get("recall", None)
    if r_val is not None and str(r_val).lower() not in ["none", "n/a", ""]:
        r_float = float(r_val) * 100.0 if float(r_val) <= 1.0 and float(r_val) > 0.0 else float(r_val)
        rec_str = f"{r_float:.2f}"
    else:
        rec_str = "N/A"

    # F1 (%)
    f1_val = data.get("f1", None)
    if f1_val is not None and str(f1_val).lower() not in ["none", "n/a", ""]:
        f1_float = float(f1_val) * 100.0 if float(f1_val) <= 1.0 and float(f1_val) > 0.0 else float(f1_val)
        f1_str = f"{f1_float:.2f}"
    else:
        f1_str = "N/A"

    # Params
    par_val = data.get("params", data.get("best_total_params", 0))
    params_str = f"{int(par_val):,}" if isinstance(par_val, (int, float)) else str(par_val)
    params_raw = str(int(par_val)) if isinstance(par_val, (int, float)) else str(par_val)

    # Memory (MB)
    mem_val = data.get("memory", data.get("best_model_size_mb", 0.0))
    mem_str = f"{float(mem_val):.2f}" if isinstance(mem_val, (int, float)) else str(mem_val)

    # FLOPs
    fl_val = data.get("flops", data.get("best_estimated_flops", 0))
    flops_str = f"{int(fl_val):,}" if isinstance(fl_val, (int, float)) else str(fl_val)
    flops_raw = str(int(fl_val)) if isinstance(fl_val, (int, float)) else str(fl_val)

    # Evaluations
    ev_val = data.get("evaluations", data.get("evals", 0))
    evals_str = str(int(ev_val)) if isinstance(ev_val, (int, float)) else str(ev_val)

    cols = [
        ("Dataset", dataset, 12, "<"),
        ("Método", method, 14, "<"),
        ("Seed", seed, 6, ">"),
        ("Gen", gen, 5, ">"),
        ("Pop", pop, 5, ">"),
        ("Mig.", mig, 8, "^"),
        ("Epoch", epoch, 6, ">"),
        ("Time", time_str, 9, ">"),
        ("Energy", energy_str, 11, ">"),
        ("CO₂", co2_str, 9, ">"),
        ("Fitness", fit_str, 8, ">"),
        ("Val Acc", val_acc_str, 9, ">"),
        ("Test Acc", test_acc_str, 9, ">"),
        ("Precision", prec_str, 10, ">"),
        ("Recall", rec_str, 8, ">"),
        ("F1", f1_str, 8, ">"),
        ("Params", params_str, 12, ">"),
        ("Memory", mem_str, 8, ">"),
        ("FLOPs", flops_str, 13, ">"),
        ("Evaluations", evals_str, 11, ">")
    ]

    header_parts = []
    row_parts = []
    tsv_headers = []
    tsv_values = []
    raw_dict = {}

    for name, val, width, align in cols:
        tsv_headers.append(name)
        tsv_values.append(val)
        raw_dict[name] = val
        if align == "<":
            header_parts.append(f"{name:<{width}}")
            row_parts.append(f"{val:<{width}}")
        elif align == ">":
            header_parts.append(f"{name:>{width}}")
            row_parts.append(f"{val:>{width}}")
        else:
            header_parts.append(f"{name:^{width}}")
            row_parts.append(f"{val:^{width}}")

    table_header = "  ".join(header_parts)
    table_row = "  ".join(row_parts)
    sep_line = "-" * len(table_header)
    box_line = "=" * len(table_header)

    tsv_header_str = "\t".join(tsv_headers)
    tsv_row_str = "\t".join(tsv_values)

    # Formatos limpios SOLO VALORES (sin comas en miles, ideal para copiar a Excel / CSV)
    raw_headers = [
        "Dataset", "Método", "Seed", "Gen", "Pop", "Mig.", "Epoch",
        "Time", "Energy", "CO₂", "Fitness", "Val Acc", "Test Acc",
        "Precision", "Recall", "F1", "Params", "Memory", "FLOPs", "Evaluations"
    ]
    raw_values = [
        dataset, method, seed, gen, pop, mig, epoch,
        time_str, energy_str, co2_str, fit_str, val_acc_str, test_acc_str,
        prec_str, rec_str, f1_str, params_raw, mem_str, flops_raw, evals_str
    ]
    raw_tsv_header_str = "\t".join(raw_headers)
    raw_tsv_row_str = "\t".join(raw_values)

    raw_csv_headers = []
    for h in raw_headers:
        raw_csv_headers.append(f'"{h}"' if (',' in h or '"' in h) else h)
    raw_csv_values = []
    for v in raw_values:
        v_str = str(v)
        if ',' in v_str or '"' in v_str or '\n' in v_str:
            raw_csv_values.append('"' + v_str.replace('"', '""') + '"')
        else:
            raw_csv_values.append(v_str)

    raw_csv_header_str = ",".join(raw_csv_headers)
    raw_csv_row_str = ",".join(raw_csv_values)

    return {
        "table_header": table_header,
        "table_row": table_row,
        "sep_line": sep_line,
        "box_line": box_line,
        "tsv_header": tsv_header_str,
        "tsv_row": tsv_row_str,
        "raw_tsv_header": raw_tsv_header_str,
        "raw_tsv_row": raw_tsv_row_str,
        "raw_csv_header": raw_csv_header_str,
        "raw_csv_row": raw_csv_row_str,
        "raw_headers": raw_headers,
        "raw_values": raw_values,
        "values_dict": raw_dict
    }


def save_experiment_record(
    data: dict,
    chck_dir: str = "./checkpoints/",
    custom_filename: str = None,
    print_console: bool = True
) -> dict:
    """
    Guarda y muestra la información del experimento según los 20 campos especificados:
    Dataset    Método    Seed    Gen    Pop    Mig.    Epoch    Time    Energy    CO₂    Fitness    Val Acc    Test Acc    Precision    Recall    F1    Params    Memory    FLOPs    Evaluations

    Garantiza:
    1. Archivo 1: Reporte individual con formato visual y detalle explicativo.
    2. Archivo 2: Archivo individual SOLO VALORES (sin texto decorativo), perfecto para copiar y pegar en Excel en 1 paso.
    3. Archivos maestros acumulativos (experiments_summary.txt, experiments_summary_values.txt y experiments_summary_values.csv).
    4. Impresión por consola clara con enlaces a todos los archivos generados.
    """
    os.makedirs(chck_dir, exist_ok=True)
    formatted = format_experiment_row(data)

    method_clean = str(data.get("method", "DeepGA")).replace(" ", "_")
    dataset_clean = str(data.get("dataset", "Dataset")).replace(" ", "_").replace("/", "_")
    seed_clean = str(data.get("seed", 1))
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 1. Archivo individual detallado con formato
    if custom_filename is None:
        base_name = f"exp_{method_clean}_{dataset_clean}_seed{seed_clean}_{timestamp}.txt"
        base_values_name = f"exp_{method_clean}_{dataset_clean}_seed{seed_clean}_{timestamp}_values.txt"
        base_csv_name = f"exp_{method_clean}_{dataset_clean}_seed{seed_clean}_{timestamp}_values.csv"
    else:
        base_name = custom_filename
        name_root, ext = os.path.splitext(custom_filename)
        base_values_name = f"{name_root}_values{ext}"
        base_csv_name = f"{name_root}_values.csv"

    individual_path = os.path.join(chck_dir, base_name)
    individual_values_path = os.path.join(chck_dir, base_values_name)
    individual_csv_path = os.path.join(chck_dir, base_csv_name)

    counter = 1
    while os.path.exists(individual_path):
        name_root, ext = os.path.splitext(base_name)
        individual_path = os.path.join(chck_dir, f"{name_root}_{counter}{ext}")
        individual_values_path = os.path.join(chck_dir, f"{name_root}_{counter}_values{ext}")
        individual_csv_path = os.path.join(chck_dir, f"{name_root}_{counter}_values.csv")
        counter += 1

    # Contenido del reporte detallado (Archivo 1)
    indiv_lines = [
        formatted["box_line"],
        "                     REPORTE DE EXPERIMENTO - GREEN DEEPGA",
        formatted["box_line"],
        f"Fecha y Hora:  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset:       {data.get('dataset', 'N/A')}",
        f"Método:        {data.get('method', 'N/A')}",
        f"Semilla:       {data.get('seed', 'N/A')}",
        formatted["sep_line"],
        "TABLA DE RESULTADOS DE EXPERIMENTACIÓN:",
        formatted["sep_line"],
        formatted["table_header"],
        formatted["sep_line"],
        formatted["table_row"],
        formatted["box_line"],
        "",
        "# Formato Tab-Separated Values (TSV) para fácil exportación:",
        formatted["tsv_header"],
        formatted["tsv_row"],
        "",
        formatted["sep_line"],
        "DETALLE DE MÉTRICAS OBTENIDAS:",
        formatted["sep_line"],
    ]

    for k, v in formatted["values_dict"].items():
        indiv_lines.append(f"  • {k:<15}: {v}")

    indiv_lines.append(formatted["box_line"])

    with open(individual_path, "w", encoding="utf-8") as f:
        f.write("\n".join(indiv_lines) + "\n")

    # 2. Contenido del archivo SOLO VALORES (Archivo 2 - TSV limpio sin encabezados de texto)
    # Permite copiar todo el archivo (Ctrl+A, Ctrl+C) y pegarlo directamente en Excel (Ctrl+V) en una sola fila limpia.
    with open(individual_values_path, "w", encoding="utf-8") as f:
        f.write(formatted["raw_tsv_row"] + "\n")

    # Guardar también versión CSV individual (encabezado + valores) para abrir directamente
    with open(individual_csv_path, "w", encoding="utf-8") as f:
        f.write(formatted["raw_csv_header"] + "\n")
        f.write(formatted["raw_csv_row"] + "\n")

    # 3. Guardar / Añadir a los archivos maestros acumulativos
    summary_path = os.path.join(chck_dir, "experiments_summary.txt")
    write_header_sum = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header_sum:
            f.write(formatted["tsv_header"] + "\n")
        f.write(formatted["tsv_row"] + "\n")

    summary_values_path = os.path.join(chck_dir, "experiments_summary_values.txt")
    write_header_val = not os.path.exists(summary_values_path) or os.path.getsize(summary_values_path) == 0
    with open(summary_values_path, "a", encoding="utf-8") as f:
        if write_header_val:
            f.write(formatted["raw_tsv_header"] + "\n")
        f.write(formatted["raw_tsv_row"] + "\n")

    summary_csv_path = os.path.join(chck_dir, "experiments_summary_values.csv")
    write_header_csv = not os.path.exists(summary_csv_path) or os.path.getsize(summary_csv_path) == 0
    with open(summary_csv_path, "a", encoding="utf-8") as f:
        if write_header_csv:
            f.write(formatted["raw_csv_header"] + "\n")
        f.write(formatted["raw_csv_row"] + "\n")

    # 4. Imprimir por consola
    if print_console:
        print("\n" + formatted["box_line"], flush=True)
        print("                     RESUMEN DE EXPERIMENTACIÓN (GREEN DEEPGA)", flush=True)
        print(formatted["box_line"], flush=True)
        print(formatted["table_header"], flush=True)
        print(formatted["sep_line"], flush=True)
        print(formatted["table_row"], flush=True)
        print(formatted["box_line"], flush=True)
        print(f"📄 Reporte detallado (.txt):               {os.path.abspath(individual_path)}", flush=True)
        print(f"📋 Archivo SOLO VALORES para Excel (.txt):    {os.path.abspath(individual_values_path)}", flush=True)
        print(f"📊 Resumen acumulativo TSV (.txt):         {os.path.abspath(summary_values_path)}", flush=True)
        print(f"📊 Resumen acumulativo Excel (.csv):       {os.path.abspath(summary_csv_path)}", flush=True)
        print(formatted["box_line"] + "\n", flush=True)

    return {
        "individual_file": individual_path,
        "individual_values_file": individual_values_path,
        "individual_csv_file": individual_csv_path,
        "summary_file": summary_path,
        "summary_values_file": summary_values_path,
        "summary_csv_file": summary_csv_path,
        "formatted_row": formatted["table_row"],
        "tsv_row": formatted["tsv_row"],
        "raw_tsv_row": formatted["raw_tsv_row"],
        "raw_csv_row": formatted["raw_csv_row"]
    }


