# -*- coding: utf-8 -*-
"""
Módulo de Carga y Preprocesamiento de Datasets para DeepGA.
Soporta:
1. Datasets personalizados organizados en carpetas (ImageFolder):
   Ejemplo:
   dataset/
   └── covid/
       ├── covid/
       ├── neumonia/
       └── normal/
2. Partición Estratificada (Train / Val / Test) exacta para máximo balance y accuracy.
3. Data Augmentation avanzado en Train (Flip, Rotación, Contraste) para imágenes médicas/generales.
4. Soporte para imágenes RGB (3 canales) o Escala de Grises (1 canal).
5. Precarga opcional en VRAM (FastGPULoader) para entrenamiento ultra-rápido.
6. Fallback automático a CIFAR-10 si se especifica.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FastGPUDatasetWrapper:
    """Wrapper para retornar la cantidad total de muestras en dataset_dl.dataset."""
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples


class FastGPULoader:
    """
    Iterador ultra-rápido que almacena y realiza el batching / shuffling
    directamente en la VRAM de la GPU, eliminando transferencias CPU-GPU por época.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64, shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.x)
        self.dataset = FastGPUDatasetWrapper(self.num_samples)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples, device=self.x.device)
        else:
            indices = torch.arange(self.num_samples, device=self.x.device)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_idx = indices[start_idx:end_idx]
            yield self.x[batch_idx], self.y[batch_idx]


def get_custom_imagefolder_loaders(
    data_dir: str,
    img_size: int = 64,
    in_channels: int = 1,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 2,
    preload_gpu: bool = False,
    device: torch.device = None,
    random_state: int = 42
):
    """
    Carga un dataset organizado en subcarpetas mediante torchvision.datasets.ImageFolder.
    
    Aplica:
    - Partición Estratificada exacta (Train/Val/Test) para balancear clases equitativamente.
    - Data Augmentation especializado en Train para maximizar accuracy y evitar overfitting.
    - Normalización y redimensionado consistente para GPU.
    
    Retorna:
    - train_dl, val_dl, test_dl, in_channels, out_size, n_classes, class_names
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directorio de dataset no encontrado: '{data_dir}'")

    # 1. Definir transformaciones por canal
    if in_channels == 1:
        # Escala de grises (1 canal)
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
        # RGB (3 canales)
        train_transforms = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_transforms = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 2. Cargar estructura de clases
    raw_dataset = datasets.ImageFolder(root=data_dir)
    class_names = raw_dataset.classes
    n_classes = len(class_names)
    targets = np.array(raw_dataset.targets)
    num_total = len(targets)

    if num_total == 0:
        raise ValueError(f"No se encontraron imágenes en el directorio '{data_dir}'. Asegúrate de que las subcarpetas contengan archivos de imagen válidos.")

    # 3. Partición Estratificada
    indices = np.arange(num_total)
    if SKLEARN_AVAILABLE and len(np.unique(targets)) > 1:
        # División Train vs Temp (Val + Test)
        train_idx, temp_idx, _, temp_targets = train_test_split(
            indices,
            targets,
            test_size=(1.0 - train_ratio),
            stratify=targets,
            random_state=random_state
        )

        # División Val vs Test
        val_relative = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_relative),
            stratify=temp_targets,
            random_state=random_state
        )
    else:
        # Fallback estratificado manual
        np.random.seed(random_state)
        train_idx, val_idx, test_idx = [], [], []
        for c in range(n_classes):
            c_indices = indices[targets == c]
            np.random.shuffle(c_indices)
            n_c = len(c_indices)
            n_train = int(n_c * train_ratio)
            n_val = int(n_c * val_ratio)
            train_idx.extend(c_indices[:n_train])
            val_idx.extend(c_indices[n_train:n_train + n_val])
            test_idx.extend(c_indices[n_train + n_val:])
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)

    # 4. Crear datasets con sus respectivas transformaciones
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    eval_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(eval_dataset, val_idx)
    test_ds = Subset(eval_dataset, test_idx)

    print(f"\n📂 [Custom Dataset] Cargado desde: '{os.path.abspath(data_dir)}'", flush=True)
    print(f"   🏷️  Clases detectadas ({n_classes}): {class_names}", flush=True)
    print(f"   📐 Dimensiones: {img_size}x{img_size} | Canales: {in_channels}", flush=True)
    print(f"   📊 Distribución Estratificada -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,} imágenes", flush=True)

    # Conteo por clase en cada partición para verificar balance
    for c_idx, c_name in enumerate(class_names):
        c_train = sum(1 for i in train_idx if targets[i] == c_idx)
        c_val = sum(1 for i in val_idx if targets[i] == c_idx)
        c_test = sum(1 for i in test_idx if targets[i] == c_idx)
        print(f"      - {c_name:<15}: Train={c_train:<4} | Val={c_val:<4} | Test={c_test:<4} (Total: {c_train+c_val+c_test})", flush=True)

    # 5. Precarga en VRAM o DataLoaders estándar
    if preload_gpu and device.type == "cuda":
        print(f"   🚀 Precargando dataset completo en VRAM de la GPU ({device})...", flush=True)
        loader_train_all = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        loader_val_all = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        loader_test_all = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

        x_train, y_train = next(iter(loader_train_all))
        x_val, y_val = next(iter(loader_val_all))
        x_test, y_test = next(iter(loader_test_all))

        train_dl = FastGPULoader(x_train.to(device, dtype=torch.float32), y_train.to(device, dtype=torch.long), batch_size=batch_size, shuffle=True)
        val_dl = FastGPULoader(x_val.to(device, dtype=torch.float32), y_val.to(device, dtype=torch.long), batch_size=batch_size, shuffle=False)
        test_dl = FastGPULoader(x_test.to(device, dtype=torch.float32), y_test.to(device, dtype=torch.long), batch_size=batch_size, shuffle=False)
        print(f"   ✓ Dataset ({len(train_ds)+len(val_ds)+len(test_ds)} imágenes) precargado en GPU VRAM.", flush=True)
    else:
        use_pin = torch.cuda.is_available() and (device.type == "cuda")
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    return train_dl, val_dl, test_dl, in_channels, img_size, n_classes, class_names


def load_dataset_auto(
    data_root: str = "./data",
    img_size: int = 64,
    in_channels: int = 3,
    batch_size: int = 32,
    preload_gpu: bool = True,
    device: torch.device = None,
    num_workers: int = 2
):
    """
    Función de detección automática:
    - Si `data_root` contiene subcarpetas con imágenes (como covid, neumonia, normal o tumores),
      utiliza get_custom_imagefolder_loaders.
    - De lo contrario, carga CIFAR-10 con fallback automático.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verificar si es una carpeta con subdirectorios de clases
    is_custom_folder = False
    if os.path.exists(data_root) and os.path.isdir(data_root):
        subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")]
        if len(subdirs) >= 2:
            is_custom_folder = True

    if is_custom_folder:
        return get_custom_imagefolder_loaders(
            data_dir=data_root,
            img_size=img_size,
            in_channels=in_channels,
            batch_size=batch_size,
            preload_gpu=preload_gpu,
            device=device,
            num_workers=num_workers
        )
    else:
        # Cargar CIFAR-10
        from torchvision import datasets, transforms
        cifar_classes = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        try:
            full_train = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
            test_ds = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
        except Exception:
            target_root = data_root if not data_root.startswith("/content") else "./data"
            os.makedirs(target_root, exist_ok=True)
            print(f"Descargando CIFAR-10 en '{target_root}'...", flush=True)
            full_train = datasets.CIFAR10(root=target_root, train=True, download=True, transform=transform_train)
            test_ds = datasets.CIFAR10(root=target_root, train=False, download=True, transform=transform_test)

        val_size = int(len(full_train) * 0.10)
        train_size = len(full_train) - val_size
        train_ds, val_ds = random_split(full_train, [train_size, val_size])

        if preload_gpu and device.type == "cuda":
            loader_train_all = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
            loader_val_all = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
            loader_test_all = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

            x_train, y_train = next(iter(loader_train_all))
            x_val, y_val = next(iter(loader_val_all))
            x_test, y_test = next(iter(loader_test_all))

            train_dl = FastGPULoader(x_train.to(device, dtype=torch.float32), y_train.to(device, dtype=torch.long), batch_size=batch_size, shuffle=True)
            val_dl = FastGPULoader(x_val.to(device, dtype=torch.float32), y_val.to(device, dtype=torch.long), batch_size=batch_size, shuffle=False)
            test_dl = FastGPULoader(x_test.to(device, dtype=torch.float32), y_test.to(device, dtype=torch.long), batch_size=batch_size, shuffle=False)
        else:
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_dl, val_dl, test_dl, 3, 32, 10, cifar_classes
