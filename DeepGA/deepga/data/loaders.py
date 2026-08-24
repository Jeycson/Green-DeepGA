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
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _convert_to_rgb(img):
    """Función de nivel superior serializable con pickle para PyTorch DataLoader en Windows."""
    return img.convert('RGB')


def _identity(x):
    """Función identidad serializable con pickle."""
    return x


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

    if num_workers > 0 and (os.name == 'nt' or sys.platform.startswith('win')):
        num_workers = 0

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
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_transforms = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
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


def get_presplit_imagefolder_loaders(
    data_dir: str,
    train_subdir: str = "train",
    test_subdir: str = "test",
    img_size: int = 64,
    in_channels: int = 1,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    num_workers: int = 2,
    preload_gpu: bool = False,
    device: torch.device = None,
    random_state: int = 42
):
    """
    Carga un dataset que YA viene pre-dividido en dos carpetas (train/ y test/),
    cada una con las mismas subcarpetas de clases dentro.

    Ejemplo de estructura esperada:
    dataset/
    ├── train/
    │   ├── clase1/
    │   ├── clase2/
    │   └── clase3/
    └── test/
        ├── clase1/
        ├── clase2/
        └── clase3/

    Como el dataset no trae partición de validación, esta función separa
    val_ratio de forma ESTRATIFICADA a partir de train/, y usa test/ tal cual.

    Retorna:
    - train_dl, val_dl, test_dl, in_channels, out_size, n_classes, class_names
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_workers > 0 and (os.name == 'nt' or sys.platform.startswith('win')):
        num_workers = 0

    train_dir = os.path.join(data_dir, train_subdir)
    test_dir = os.path.join(data_dir, test_subdir)

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"No se encontró la carpeta de entrenamiento: '{train_dir}'")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"No se encontró la carpeta de prueba: '{test_dir}'")

    # 1. Definir transformaciones por canal (idénticas a la versión original)
    if in_channels == 1:
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
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_transforms = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 2. Cargar estructura de clases de train y test por separado
    raw_train = datasets.ImageFolder(root=train_dir)
    raw_test = datasets.ImageFolder(root=test_dir)

    if raw_train.classes != raw_test.classes:
        raise ValueError(
            f"Las clases de train y test no coinciden.\n"
            f"  train: {raw_train.classes}\n"
            f"  test:  {raw_test.classes}\n"
            f"Asegúrate de que ambas carpetas tengan las mismas subcarpetas de clase (mismos nombres, mismo orden)."
        )

    class_names = raw_train.classes
    n_classes = len(class_names)
    train_targets = np.array(raw_train.targets)
    num_train_total = len(train_targets)

    if num_train_total == 0:
        raise ValueError(f"No se encontraron imágenes en '{train_dir}'.")
    if len(raw_test) == 0:
        raise ValueError(f"No se encontraron imágenes en '{test_dir}'.")

    # 3. Partición estratificada de train/ -> train_final + val
    indices = np.arange(num_train_total)
    if SKLEARN_AVAILABLE and len(np.unique(train_targets)) > 1:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=train_targets,
            random_state=random_state
        )
    else:
        np.random.seed(random_state)
        train_idx, val_idx = [], []
        for c in range(n_classes):
            c_indices = indices[train_targets == c]
            np.random.shuffle(c_indices)
            n_c = len(c_indices)
            n_val = int(n_c * val_ratio)
            val_idx.extend(c_indices[:n_val])
            train_idx.extend(c_indices[n_val:])
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

    # 4. Crear datasets con sus respectivas transformaciones
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    eval_dataset = datasets.ImageFolder(root=train_dir, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transforms)

    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(eval_dataset, val_idx)
    test_ds = test_dataset  # ya viene completo y separado

    print(f"\n📂 [Pre-split Dataset] Train: '{os.path.abspath(train_dir)}' | Test: '{os.path.abspath(test_dir)}'", flush=True)
    print(f"   🏷️  Clases detectadas ({n_classes}): {class_names}", flush=True)
    print(f"   📐 Dimensiones: {img_size}x{img_size} | Canales: {in_channels}", flush=True)
    print(f"   📊 Distribución -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,} imágenes", flush=True)

    test_targets = np.array(test_dataset.targets)
    for c_idx, c_name in enumerate(class_names):
        c_train = sum(1 for i in train_idx if train_targets[i] == c_idx)
        c_val = sum(1 for i in val_idx if train_targets[i] == c_idx)
        c_test = int((test_targets == c_idx).sum())
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


class MedMNISTDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper para adaptar datasets de MedMNIST al formato de clasificación de DeepGA."""
    def __init__(self, raw_ds, transform=None):
        self.raw_ds = raw_ds
        self.transform = transform

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, idx):
        img, target = self.raw_ds[idx]
        if self.transform is not None:
            img = self.transform(img)
        if isinstance(target, np.ndarray):
            target = int(target.squeeze())
        elif isinstance(target, torch.Tensor):
            target = int(target.squeeze().item())
        else:
            target = int(target)
        return img, target


def get_medmnist_loaders(
    dataset_flag: str = "breastmnist",
    data_dir: str = "./data",
    img_size: int = 64,
    in_channels: int = 1,
    batch_size: int = 32,
    num_workers: int = 2,
    preload_gpu: bool = False,
    device: torch.device = None,
    download: bool = True
):
    """
    Carga datasets de MedMNIST (como BreastMNIST, PneumoniaMNIST, DermaMNIST, etc.)
    con soporte para particiones oficiales (Train/Val/Test), Data Augmentation y GPU VRAM.
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise ImportError(
            "El paquete 'medmnist' no está instalado en este entorno Python.\n"
            "Instálalo ejecutando: pip install medmnist"
        )

    clean_str = dataset_flag.lower().replace("medmnist/", "").replace("medmnist:", "").strip()
    base_name = os.path.basename(clean_str)

    selected_flag = None
    if clean_str in INFO:
        selected_flag = clean_str
    elif base_name in INFO:
        selected_flag = base_name
    else:
        for k in INFO:
            if k in clean_str:
                selected_flag = k
                break

    if selected_flag is None:
        raise ValueError(
            f"Dataset MedMNIST '{dataset_flag}' no reconocido.\n"
            f"Datasets disponibles en MedMNIST: {list(INFO.keys())}"
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
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        eval_transforms = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    download_dir = data_dir if (os.path.exists(data_dir) and os.path.isdir(data_dir)) else "./data"
    os.makedirs(download_dir, exist_ok=True)

    raw_train = DataClass(split='train', transform=None, download=download, root=download_dir)
    raw_val = DataClass(split='val', transform=None, download=download, root=download_dir)
    raw_test = DataClass(split='test', transform=None, download=download, root=download_dir)

    train_ds = MedMNISTDatasetWrapper(raw_train, transform=train_transforms)
    val_ds = MedMNISTDatasetWrapper(raw_val, transform=eval_transforms)
    test_ds = MedMNISTDatasetWrapper(raw_test, transform=eval_transforms)

    print(f"\n📂 [MedMNIST] Cargado '{info['python_class']}' ({selected_flag}):", flush=True)
    print(f"   🏷️  Clases detectadas ({n_classes}): {class_names}", flush=True)
    print(f"   📐 Dimensiones: {img_size}x{img_size} | Canales: {effective_channels}", flush=True)
    print(f"   📊 Distribución oficial -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,} imágenes", flush=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_workers > 0 and (os.name == 'nt' or sys.platform.startswith('win')):
        num_workers = 0

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
        print(f"   ✓ MedMNIST ({len(train_ds)+len(val_ds)+len(test_ds)} imágenes) precargado en GPU VRAM.", flush=True)
    else:
        use_pin = torch.cuda.is_available() and (device.type == "cuda")
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    return train_dl, val_dl, test_dl, effective_channels, img_size, n_classes, class_names


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
    - Si `data_root` es un dataset de MedMNIST (ej. breastmnist, pneumoniamnist, etc.),
      utiliza get_medmnist_loaders automáticamente.
    - Si `data_root` contiene subcarpetas con imágenes (como covid, neumonia, normal o tumores),
      utiliza get_custom_imagefolder_loaders.
    - De lo contrario, carga CIFAR-10 con fallback automático.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img_size >= 128 and preload_gpu:
        print(f"ℹ️  [Memory Optimizer] Resolución alta detectada ({img_size}x{img_size}). Se utiliza DataLoader con Pinned Memory para proteger la VRAM de la GPU.", flush=True)
        preload_gpu = False

    # 1. Detección de MedMNIST por nombre o prefijo
    medmnist_candidates = [
        "breastmnist", "pneumoniamnist", "chestmnist", "dermamnist",
        "octmnist", "pathmnist", "bloodmnist", "tissuemnist",
        "organamnist", "organcmnist", "organsmnist", "retinamnist", "synapsemnist"
    ]
    data_root_clean = str(data_root).lower().strip()
    is_medmnist = (
        data_root_clean.startswith("medmnist") or 
        any(cand in data_root_clean for cand in medmnist_candidates)
    ) and not (os.path.exists(data_root) and os.path.isdir(data_root) and len([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")]) >= 2)

    if is_medmnist:
        return get_medmnist_loaders(
            dataset_flag=data_root,
            data_dir="./data",
            img_size=img_size,
            in_channels=in_channels,
            batch_size=batch_size,
            preload_gpu=preload_gpu,
            device=device,
            num_workers=num_workers
        )

    # 2. Verificar si es una carpeta con subdirectorios de clases
    is_custom_folder = False
    is_presplit_folder = False
    if os.path.exists(data_root) and os.path.isdir(data_root):
        subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")]
        # Caso: dataset ya dividido en train/ y test/ (o train/ y val/)
        if "train" in subdirs and ("test" in subdirs or "val" in subdirs):
            is_presplit_folder = True
        elif len(subdirs) >= 2:
            is_custom_folder = True

    if is_presplit_folder:
        test_name = "test" if "test" in os.listdir(data_root) else "val"
        return get_presplit_imagefolder_loaders(
            data_dir=data_root,
            train_subdir="train",
            test_subdir=test_name,
            img_size=img_size,
            in_channels=in_channels,
            batch_size=batch_size,
            preload_gpu=preload_gpu,
            device=device,
            num_workers=num_workers
        )
    elif is_custom_folder:
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

        if in_channels == 1:
            transform_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            transform_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            tf_train_list = [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            tf_test_list = [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            if img_size != 32:
                tf_train_list.insert(0, transforms.Resize((img_size, img_size)))
                tf_test_list.insert(0, transforms.Resize((img_size, img_size)))
            transform_train = transforms.Compose(tf_train_list)
            transform_test = transforms.Compose(tf_test_list)

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

        return train_dl, val_dl, test_dl, in_channels, img_size, 10, cifar_classes


def get_custom_imagefolder_2split_loaders(
    data_dir: str,
    img_size: int = 64,
    in_channels: int = 1,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    num_workers: int = 2,
    preload_gpu: bool = False,
    device: torch.device = None,
    random_state: int = 42
):
    """
    Carga un dataset en subcarpetas y lo divide ESTRICTAMENTE en 2 conjuntos:
    - Entrenamiento (1 - val_ratio, ej. 85%): Con Data Augmentation.
    - Validación (val_ratio, ej. 15%): Para evaluar fitness/accuracy durante la neuroevolución.
    
    Permite maximizar el número de imágenes de entrenamiento disponibles sin reservar un conjunto de test.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directorio de dataset no encontrado: '{data_dir}'")

    if in_channels == 1:
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
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_transforms = transforms.Compose([
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    raw_dataset = datasets.ImageFolder(root=data_dir)
    class_names = raw_dataset.classes
    n_classes = len(class_names)
    targets = np.array(raw_dataset.targets)
    num_total = len(targets)

    if num_total == 0:
        raise ValueError(f"No se encontraron imágenes en '{data_dir}'.")

    # Partición estratificada en 2 (Train y Val)
    indices = np.arange(num_total)
    if SKLEARN_AVAILABLE and len(np.unique(targets)) > 1:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=targets, random_state=random_state
        )
    else:
        np.random.seed(random_state)
        train_idx, val_idx = [], []
        for c in range(n_classes):
            c_indices = indices[targets == c]
            np.random.shuffle(c_indices)
            n_c = len(c_indices)
            n_val = max(1, int(n_c * val_ratio))
            val_idx.extend(c_indices[:n_val])
            train_idx.extend(c_indices[n_val:])
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

    train_base = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    eval_base = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(eval_base, val_idx)

    train_pct = round((len(train_ds) / num_total) * 100, 1)
    val_pct = round((len(val_ds) / num_total) * 100, 1)

    print(f"\n📂 [Partición 2-Split: Train + Val] Directorio: '{os.path.abspath(data_dir)}'", flush=True)
    print(f"   🏷️  Clases detectadas ({n_classes}): {class_names}", flush=True)
    print(f"   📐 Dimensiones: {img_size}x{img_size} | Canales: {in_channels}", flush=True)
    print(f"   📊 Distribución -> Train: {len(train_ds):,} ({train_pct}%) | Val: {len(val_ds):,} ({val_pct}%) | Total: {num_total:,} imágenes", flush=True)

    for c_idx, c_name in enumerate(class_names):
        c_train = sum(1 for i in train_idx if targets[i] == c_idx)
        c_val = sum(1 for i in val_idx if targets[i] == c_idx)
        print(f"      - {c_name:<15}: Train={c_train:<4} | Val={c_val:<4} (Total: {c_train+c_val})", flush=True)

    if preload_gpu and device.type == "cuda":
        print(f"   🚀 Precargando dataset 2-Split en VRAM de la GPU ({device})...", flush=True)
        loader_train_all = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        loader_val_all = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        x_train, y_train = next(iter(loader_train_all))
        x_val, y_val = next(iter(loader_val_all))

        train_dl = FastGPULoader(x_train.to(device, dtype=torch.float32), y_train.to(device, dtype=torch.long), batch_size=batch_size, shuffle=True)
        val_dl = FastGPULoader(x_val.to(device, dtype=torch.float32), y_val.to(device, dtype=torch.long), batch_size=batch_size, shuffle=False)
        print(f"   ✓ Dataset ({num_total} imágenes) precargado en GPU VRAM.", flush=True)
    else:
        use_pin = torch.cuda.is_available() and (device.type == "cuda")
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    return train_dl, val_dl, in_channels, img_size, n_classes, class_names


def load_dataset_2split(
    data_root: str = "./data",
    img_size: int = 64,
    in_channels: int = 3,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    preload_gpu: bool = True,
    device: torch.device = None,
    num_workers: int = 2
):
    """
    Carga cualquier dataset dividiéndolo ESTRICTAMENTE en 2 conjuntos (Entrenamiento y Validación):
    - Si data_root contiene subcarpetas con imágenes, usa get_custom_imagefolder_2split_loaders.
    - Si es CIFAR-10, utiliza los 50,000 de entrenamiento como Train y los 10,000 oficiales como Val
      (o partición con val_ratio), otorgando el máximo volumen posible para entrenamiento.
    
    Retorna:
    - train_dl, val_dl, in_channels, out_size, n_classes, class_names
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img_size >= 128 and preload_gpu:
        print(f"ℹ️  [Memory Optimizer] Resolución alta detectada ({img_size}x{img_size}). Se utiliza DataLoader 2-Split con Pinned Memory para proteger la VRAM de la GPU.", flush=True)
        preload_gpu = False

    is_custom_folder = False
    is_presplit_folder = False
    if os.path.exists(data_root) and os.path.isdir(data_root):
        subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")]
        if "train" in subdirs and ("test" in subdirs or "val" in subdirs):
            is_presplit_folder = True
        elif len(subdirs) >= 2:
            is_custom_folder = True

    if is_presplit_folder:
        val_dir_name = "val" if "val" in os.listdir(data_root) else "test"
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, val_dir_name)

        if in_channels == 1:
            train_tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            val_tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            train_tf = transforms.Compose([
                transforms.Lambda(_convert_to_rgb),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            val_tf = transforms.Compose([
                transforms.Lambda(_convert_to_rgb),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        train_ds = datasets.ImageFolder(root=train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(root=val_dir, transform=val_tf)
        class_names = train_ds.classes
        n_classes = len(class_names)

        print(f"\n📂 [Pre-Split 2-Way] Train: '{train_dir}' ({len(train_ds):,} imgs) | Val: '{val_dir}' ({len(val_ds):,} imgs)", flush=True)

        if preload_gpu and device.type == "cuda":
            l_tr = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
            l_val = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
            x_tr, y_tr = next(iter(l_tr))
            x_val, y_val = next(iter(l_val))
            train_dl = FastGPULoader(x_tr.to(device, dtype=torch.float32), y_tr.to(device, dtype=torch.long), batch_size=batch_size, shuffle=True)
            val_dl = FastGPULoader(x_val.to(device, dtype=torch.float32), y_val.to(device, dtype=torch.long), batch_size=batch_size, shuffle=False)
        else:
            use_pin = torch.cuda.is_available() and (device.type == "cuda")
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

        return train_dl, val_dl, in_channels, img_size, n_classes, class_names

    elif is_custom_folder:
        return get_custom_imagefolder_2split_loaders(
            data_dir=data_root,
            img_size=img_size,
            in_channels=in_channels,
            batch_size=batch_size,
            val_ratio=val_ratio,
            preload_gpu=preload_gpu,
            device=device,
            num_workers=num_workers
        )
    else:
        # CIFAR-10 con asignación completa de 50,000 para Train y 10,000 para Val
        cifar_classes = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

        if in_channels == 1:
            transform_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            transform_val = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            tf_train_list = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4) if img_size == 32 else transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            tf_val_list = [transforms.Resize((img_size, img_size)) if img_size != 32 else transforms.Lambda(_identity), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            transform_train = transforms.Compose(tf_train_list)
            transform_val = transforms.Compose(tf_val_list)

        target_root = data_root if not data_root.startswith("/content") else "./data"
        os.makedirs(target_root, exist_ok=True)

        try:
            train_ds = datasets.CIFAR10(root=target_root, train=True, download=False, transform=transform_train)
            val_ds = datasets.CIFAR10(root=target_root, train=False, download=False, transform=transform_val)
        except Exception:
            print(f"Descargando CIFAR-10 en '{target_root}'...", flush=True)
            train_ds = datasets.CIFAR10(root=target_root, train=True, download=True, transform=transform_train)
            val_ds = datasets.CIFAR10(root=target_root, train=False, download=True, transform=transform_val)

        print(f"\n📂 [CIFAR-10 2-Split] Train: {len(train_ds):,} imágenes | Val: {len(val_ds):,} imágenes (Total: {len(train_ds)+len(val_ds):,}) | Canales: {in_channels}", flush=True)

        if preload_gpu and device.type == "cuda":
            loader_train_all = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
            loader_val_all = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
            x_train, y_train = next(iter(loader_train_all))
            x_val, y_val = next(iter(loader_val_all))
            train_dl = FastGPULoader(x_train.to(device, dtype=torch.float32), y_train.to(device, dtype=torch.long), batch_size=batch_size, shuffle=True)
            val_dl = FastGPULoader(x_val.to(device, dtype=torch.float32), y_val.to(device, dtype=torch.long), batch_size=batch_size, shuffle=False)
        else:
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_dl, val_dl, in_channels, img_size, 10, cifar_classes
