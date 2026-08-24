# -*- coding: utf-8 -*-
"""
Runner Individual de ResNet-18 para DeepGA (100% Autónomo / Self-contained):
1. Datasets compatibles: MedMNIST (organcmnist, breastmnist, etc.), CIFAR-10, Covid, Tumour, etc.
2. Normalización y métricas de 20 columnas idénticas a DeepGA (TSV / CSV).
3. Medición de huella de carbono y energía (CodeCarbon o fallback analítico).
4. Matriz de confusión y checkpoints (.pth).
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, datasets, transforms
from PIL import Image

# Import opcional de sklearn
try:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Tracker de carbono opcional
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Matplotlib y Seaborn para matriz de confusión
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


def _convert_to_rgb(img):
    """Convierte imagen PIL a RGB."""
    return img.convert('RGB') if hasattr(img, 'convert') else img


def adapt_input_channels(xb: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Adapta canales del tensor de entrada según lo que espera la red convolucional."""
    if xb.shape[1] == 3 and target_channels == 1:
        return xb.mean(dim=1, keepdim=True)
    elif xb.shape[1] == 1 and target_channels == 3:
        return xb.repeat(1, 3, 1, 1)
    return xb


def build_resnet18(in_channels: int, n_classes: int, pretrained: bool = False):
    """Construye y adapta ResNet-18 a los canales y clases del dataset."""
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # 1. Adaptar primera capa convolucional si no es RGB (3 canales)
    if in_channels != 3:
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

    # 2. Adaptar clasificador final a n_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model


def calculate_model_complexity(model: nn.Module, in_channels: int, img_size: int):
    """Calcula parámetros totales, entrenables, memoria en MB y FLOPs estimados."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4.0) / (1024.0 * 1024.0)

    # Estimación de FLOPs mediante hook dinámico
    flops_total = 0
    def conv_hook(module, input_val, output_val):
        nonlocal flops_total
        batch_size, out_c, out_h, out_w = output_val.shape
        in_c = module.in_channels
        k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        flops_total += 2 * batch_size * in_c * out_c * k_h * k_w * out_h * out_w

    def linear_hook(module, input_val, output_val):
        nonlocal flops_total
        batch_size = input_val[0].shape[0]
        flops_total += 2 * batch_size * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.zeros(1, in_channels, img_size, img_size)
    with torch.no_grad():
        try:
            model.eval()(dummy_input)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 4),
        "estimated_flops": flops_total
    }


class MedMNISTDatasetWrapper:
    """Wrapper para aplicar transformaciones a datasets de MedMNIST."""
    def __init__(self, medmnist_dataset, transform=None):
        self.dataset = medmnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img, mode='L')
            elif img.ndim == 3 and img.shape[2] == 1:
                img = Image.fromarray(img.squeeze(2), mode='L')
            elif img.ndim == 3 and img.shape[2] == 3:
                img = Image.fromarray(img, mode='RGB')
            else:
                img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if isinstance(target, np.ndarray):
            target = target.item() if target.size == 1 else target[0]
        elif isinstance(target, (list, tuple)):
            target = target[0]

        return img, int(target)


def get_medmnist_loaders(
    dataset_flag: str = "organcmnist",
    data_dir: str = "./data",
    img_size: int = 28,
    in_channels: int = 1,
    batch_size: int = 32,
    download: bool = True
):
    """Carga datasets de MedMNIST de forma automática."""
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise ImportError(
            "\n❌ El paquete 'medmnist' no está instalado en este entorno de Python.\n"
            "Instálalo ejecutando: pip install medmnist\n"
        )

    clean_str = str(dataset_flag).lower().replace("medmnist/", "").replace("medmnist:", "").strip()
    selected_flag = None
    if clean_str in INFO:
        selected_flag = clean_str
    else:
        for k in INFO:
            if k in clean_str:
                selected_flag = k
                break

    if selected_flag is None:
        raise ValueError(
            f"Dataset MedMNIST '{dataset_flag}' no reconocido.\n"
            f"Opciones disponibles: {list(INFO.keys())}"
        )

    info = INFO[selected_flag]
    DataClass = getattr(medmnist, info['python_class'])

    labels_dict = info.get('label', {})
    if labels_dict:
        class_names = [labels_dict[str(i)] for i in range(len(labels_dict))]
        n_classes = len(labels_dict)
    else:
        n_classes = info.get('n_classes', 2)
        class_names = [f"clase_{i}" for i in range(n_classes)]

    dataset_channels = info.get('n_channels', in_channels)
    effective_channels = in_channels if in_channels is not None else dataset_channels

    if effective_channels == 1:
        train_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        eval_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        eval_transforms = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    os.makedirs(data_dir, exist_ok=True)
    raw_train = DataClass(split='train', transform=None, download=download, root=data_dir)
    raw_val = DataClass(split='val', transform=None, download=download, root=data_dir)
    raw_test = DataClass(split='test', transform=None, download=download, root=data_dir)

    train_ds = MedMNISTDatasetWrapper(raw_train, transform=train_transforms)
    val_ds = MedMNISTDatasetWrapper(raw_val, transform=eval_transforms)
    test_ds = MedMNISTDatasetWrapper(raw_test, transform=eval_transforms)

    print(f"\n📂 [MedMNIST] Cargado '{info['python_class']}' ({selected_flag}):", flush=True)
    print(f"   🏷️  Clases detectadas ({n_classes}): {class_names}", flush=True)
    print(f"   📐 Dimensiones: {img_size}x{img_size} | Canales: {effective_channels}", flush=True)
    print(f"   📊 Distribución -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,} imágenes", flush=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, effective_channels, img_size, n_classes, class_names


def get_cifar10_loaders(data_dir: str = "./data", batch_size: int = 32, img_size: int = 32):
    """Carga CIFAR-10 oficial."""
    os.makedirs(data_dir, exist_ok=True)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    n_train = int(len(full_train) * 0.8)
    n_val = len(full_train) - n_train
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    class_names = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, 3, img_size, 10, class_names


def get_custom_imagefolder_loaders(
    data_root: str,
    img_size: int = 64,
    in_channels: int = 3,
    batch_size: int = 32
):
    """Carga dataset personalizado desde carpetas (ImageFolder)."""
    if in_channels == 1:
        t_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        t_eval = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        t_train = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        t_eval = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=data_root, transform=t_eval)
    class_names = dataset.classes
    n_classes = len(class_names)
    total_len = len(dataset)

    n_train = int(total_len * 0.70)
    n_val = int(total_len * 0.15)
    n_test = total_len - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, in_channels, img_size, n_classes, class_names


def load_dataset_auto(
    data_root: str = "organcmnist",
    img_size: int = 28,
    in_channels: int = 1,
    batch_size: int = 32
):
    """Selector automático de dataset según ruta o nombre."""
    clean = str(data_root).lower().strip()

    medmnist_keys = [
        "breastmnist", "pneumoniamnist", "chestmnist", "dermamnist",
        "octmnist", "pathmnist", "bloodmnist", "tissuemnist",
        "organamnist", "organcmnist", "organsmnist", "retinamnist", "synapsemnist"
    ]

    if any(k in clean for k in medmnist_keys) or clean.startswith("medmnist"):
        return get_medmnist_loaders(
            dataset_flag=data_root,
            img_size=img_size,
            in_channels=in_channels,
            batch_size=batch_size
        )
    elif "cifar" in clean:
        return get_cifar10_loaders(batch_size=batch_size, img_size=img_size or 32)
    elif os.path.exists(data_root) and os.path.isdir(data_root):
        return get_custom_imagefolder_loaders(
            data_root=data_root,
            img_size=img_size or 64,
            in_channels=in_channels,
            batch_size=batch_size
        )
    else:
        raise ValueError(
            f"No se pudo cargar el dataset '{data_root}'.\n"
            f"Verifica que la carpeta exista o que sea un dataset de MedMNIST / CIFAR-10 válido."
        )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena 1 época y retorna (loss_promedio, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    expected_channels = model.conv1.in_channels if hasattr(model, 'conv1') else 3

    for data in dataloader:
        if isinstance(data, (list, tuple)):
            inputs, labels = data[0], data[1]
        elif isinstance(data, dict):
            inputs, labels = data["image"], data["label"]
        else:
            continue

        inputs = inputs.to(device, dtype=torch.float32)
        inputs = adapt_input_channels(inputs, expected_channels)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return epoch_loss, epoch_acc


def evaluate_accuracy(model, dataloader, device):
    """Calcula accuracy de validación en porcentaje."""
    model.eval()
    correct = 0
    total = 0
    expected_channels = model.conv1.in_channels if hasattr(model, 'conv1') else 3

    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, (list, tuple)):
                inputs, labels = data[0], data[1]
            elif isinstance(data, dict):
                inputs, labels = data["image"], data["label"]
            else:
                continue

            inputs = inputs.to(device, dtype=torch.float32)
            inputs = adapt_input_channels(inputs, expected_channels)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

    return (correct / max(1, total)) * 100.0 if total > 0 else 0.0


def compute_classification_metrics(model, dataloader, device=None):
    """Calcula Accuracy, Precision, Recall y Macro F1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    expected_channels = model.conv1.in_channels if hasattr(model, 'conv1') else 3

    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, (list, tuple)):
                inputs, labels = data[0], data[1]
            elif isinstance(data, dict):
                inputs, labels = data["image"], data["label"]
            else:
                continue

            inputs = inputs.to(device, dtype=torch.float32)
            inputs = adapt_input_channels(inputs, expected_channels)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_pred.extend(preds.cpu().numpy())
            if torch.is_tensor(labels):
                y_true.extend(labels.cpu().numpy())
            else:
                y_true.extend(labels)

    if len(y_true) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred) * 100.0)

    if SKLEARN_AVAILABLE:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec = float(prec * 100.0)
        rec = float(rec * 100.0)
        f1 = float(f1 * 100.0)
    else:
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


def generate_confusion_matrix(model_or_path, dataloader, class_names, title="Matriz de Confusión", save_fig_path=None):
    """Genera y guarda la matriz de confusión."""
    if not PLT_AVAILABLE or dataloader is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_or_path.to(device)
    model.eval()

    y_true = []
    y_pred = []
    expected_channels = model.conv1.in_channels if hasattr(model, 'conv1') else 3

    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, (list, tuple)):
                inputs, labels = data[0], data[1]
            elif isinstance(data, dict):
                inputs, labels = data["image"], data["label"]
            else:
                continue

            inputs = inputs.to(device, dtype=torch.float32)
            inputs = adapt_input_channels(inputs, expected_channels)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_pred.extend(preds.cpu().numpy())
            if torch.is_tensor(labels):
                y_true.extend(labels.cpu().numpy())
            else:
                y_true.extend(labels)

    if len(y_true) == 0:
        return

    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred)
    else:
        n = len(class_names)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < n and 0 <= p < n:
                cm[t, p] += 1

    plt.figure(figsize=(8, 6))
    try:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    except Exception:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300)
        print(f"📊 Matriz de confusión guardada en: {save_fig_path}")
    plt.close()


def save_experiment_record(exp_data: dict, chck_dir: str = "./checkpoints/", print_console: bool = True):
    """Guarda fila de métricas en formato TSV y CSV estándar de 20 columnas."""
    os.makedirs(chck_dir, exist_ok=True)
    headers = [
        "Dataset", "Método", "Seed", "Gen", "Pop", "Mig.", "Epoch",
        "Time (s)", "Energy (kWh)", "CO₂ (g)", "Fitness",
        "Val Acc (%)", "Test Acc (%)", "Precision (%)", "Recall (%)", "F1 (%)",
        "Params", "Memory (MB)", "FLOPs", "Evaluations"
    ]

    row_values = [
        str(exp_data.get("dataset", "N/A")),
        str(exp_data.get("method", "ResNet18")),
        str(exp_data.get("seed", 1)),
        str(exp_data.get("gen", "N/A")),
        str(exp_data.get("pop", "N/A")),
        str(exp_data.get("mig", "N/A")),
        str(exp_data.get("epoch", 10)),
        f"{float(exp_data.get('time', 0.0)):.2f}",
        f"{float(exp_data.get('energy', 0.0)):.6f}",
        f"{float(exp_data.get('co2', 0.0)):.4f}",
        f"{float(exp_data.get('fitness', 0.0)):.4f}",
        f"{float(exp_data.get('val_acc', 0.0)):.2f}",
        f"{float(exp_data.get('test_acc', 0.0)):.2f}",
        f"{float(exp_data.get('precision', 0.0)):.2f}",
        f"{float(exp_data.get('recall', 0.0)):.2f}",
        f"{float(exp_data.get('f1', 0.0)):.2f}",
        f"{int(exp_data.get('params', 0)):,}",
        f"{float(exp_data.get('memory', 0.0)):.4f}",
        f"{int(exp_data.get('flops', 0)):,}",
        str(exp_data.get("evaluations", 1))
    ]

    # Guardar en TSV y CSV
    tsv_path = os.path.join(chck_dir, "metricas_acumuladas.tsv")
    csv_path = os.path.join(chck_dir, "resultados_experimentos.csv")

    for path, sep in [(tsv_path, "\t"), (csv_path, ",")]:
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(sep.join(headers) + "\n")
            f.write(sep.join(row_values) + "\n")

    if print_console:
        print("\n" + "=" * 76)
        print("                TABLA RESUMEN DEL EXPERIMENTO (RESNET-18)")
        print("=" * 76)
        for h, v in zip(headers, row_values):
            print(f"  • {h:<22}: {v}")
        print("=" * 76 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecución autónoma de ResNet-18 para Green-DeepGA")
    parser.add_argument("--execution", "--exec", type=int, default=1,
                        help="ID numérico de la ejecución")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla aleatoria (por defecto igual a execution)")
    parser.add_argument("--data-root", "--data_root", "--dataset", type=str, default="organcmnist",
                        help="Ruta al dataset o nombre de MedMNIST / CIFAR (ej. organcmnist, BreadMNIST, Covid, CIFAR-10)")
    parser.add_argument("--img-size", "--img_size", type=int, default=28,
                        help="Resolución cuadrada de imágenes (ej. 28 para MedMNIST, 32 CIFAR, 64/128 otros)")
    parser.add_argument("--in-channels", "--in-channel", "--in_channels", "--in_channel", type=int, default=1, choices=[1, 3],
                        help="Canales de entrada: 1 para escala de grises/MedMNIST, 3 para RGB")
    parser.add_argument("--epochs", "--final-epochs", "--final-epoch", "--final_epochs", "--final_epoch", type=int, default=10,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32,
                        help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (Learning Rate)")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Usar pesos preentrenados de ImageNet")
    parser.add_argument("--chck-dir", "--chck_dir", type=str, default="./checkpoints/",
                        help="Directorio de salida para checkpoints y reportes")
    parser.add_argument("--country-iso", "--country_iso", type=str, default="MEX",
                        help="Código ISO del país para huella de carbono")
    parser.add_argument("--device", type=str, default=None,
                        help="Dispositivo ('cuda', 'cpu' o None para auto-detección)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else args.execution

    # Reproducibilidad estricta
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dispositivo
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.chck_dir, exist_ok=True)

    # 1. Carga automática del Dataset
    print(f"\n📦 Cargando dataset '{args.data_root}'...")
    train_dl, val_dl, test_dl, in_channels, img_size, n_classes, class_names = load_dataset_auto(
        data_root=args.data_root,
        img_size=args.img_size,
        in_channels=args.in_channels,
        batch_size=args.batch_size
    )

    dataset_name = os.path.basename(os.path.normpath(args.data_root))
    if not dataset_name or dataset_name == ".":
        dataset_name = str(args.data_root)

    # 2. Inicializar ResNet-18
    model = build_resnet18(in_channels=in_channels, n_classes=n_classes, pretrained=args.pretrained)
    model = model.to(device)

    complexity = calculate_model_complexity(model, in_channels, img_size)

    # 3. Iniciar rastreo de Huella de Carbono y Tiempo
    tracker = None
    if CODECARBON_AVAILABLE:
        try:
            tracker = OfflineEmissionsTracker(
                country_iso_code=args.country_iso,
                output_dir=args.chck_dir,
                log_level="error"
            )
            tracker.start()
        except Exception:
            tracker = None

    start_time = time.perf_counter()

    # 4. Entrenamiento y validación por época
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_weights = None

    print(f"\n🚀 Iniciando entrenamiento de ResNet-18 ({args.epochs} épocas, LR: {args.lr})...")
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        v_acc = evaluate_accuracy(model, val_dl, device)

        if v_acc > best_val_acc or best_weights is None:
            best_val_acc = v_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"   [Época {epoch:02d}/{args.epochs:02d}] Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% (Mejor: {best_val_acc:.2f}%)")

    # Restaurar mejores pesos obtenidos
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # 5. Detener tiempos y calcular huella
    elapsed_seconds = time.perf_counter() - start_time
    emissions_g_co2 = 0.0
    energy_kwh = 0.0

    if tracker is not None:
        try:
            emissions_kg = tracker.stop()
            if emissions_kg is not None:
                emissions_g_co2 = float(emissions_kg * 1000.0)
            if hasattr(tracker, "_total_energy"):
                energy_kwh = float(getattr(tracker._total_energy, "kWh", 0.0))
        except Exception:
            pass
    else:
        energy_kwh = 0.150 * (elapsed_seconds / 3600.0)
        emissions_g_co2 = energy_kwh * 430.0

    # 6. Evaluación exhaustiva en Test Set independiente
    print("\n📊 Evaluando modelo sobre el Test Set independiente...")
    target_dl = test_dl if test_dl is not None else val_dl
    cls_metrics = compute_classification_metrics(model, target_dl, device=device)

    test_acc = cls_metrics["accuracy"]
    prec_val = cls_metrics["precision"]
    rec_val = cls_metrics["recall"]
    f1_val = cls_metrics["f1"]

    # Guardar Checkpoint .pth de ResNet-18
    saved_model_path = os.path.join(args.chck_dir, f"best_model_resnet18_exec_{args.execution}.pth")
    torch.save({
        "variant": "resnet18",
        "execution": args.execution,
        "seed": seed,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "in_channels": in_channels,
        "img_size": img_size,
        "n_classes": n_classes,
        "state_dict": model.state_dict(),
        "class_names": class_names
    }, saved_model_path)

    # 7. Generar Matriz de Confusión
    cm_path = os.path.join(args.chck_dir, f"matriz_confusion_resnet18_exec_{args.execution}.png")
    generate_confusion_matrix(
        model_or_path=model,
        dataloader=target_dl,
        class_names=class_names,
        title=f"Matriz de Confusión - ResNet-18 (Exec {args.execution})",
        save_fig_path=cm_path
    )

    # 8. Guardar Fila de Experimento en Formato Estándar de 20 Columnas
    exp_data = {
        "dataset": dataset_name,
        "method": "ResNet18",
        "seed": seed,
        "gen": "N/A",
        "pop": "N/A",
        "mig": "N/A",
        "epoch": args.epochs,
        "time": elapsed_seconds,
        "energy": energy_kwh,
        "co2": emissions_g_co2,
        "fitness": best_val_acc,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "precision": prec_val,
        "recall": rec_val,
        "f1": f1_val,
        "params": complexity["total_params"],
        "memory": complexity["model_size_mb"],
        "flops": complexity["estimated_flops"],
        "evaluations": 1
    }

    save_experiment_record(exp_data, chck_dir=args.chck_dir, print_console=True)
    print("✨ Ejecución de ResNet-18 finalizada exitosamente.")


if __name__ == "__main__":
    main()
