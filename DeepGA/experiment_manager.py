# -*- coding: utf-8 -*-
"""
Experiment Manager para DeepGA y MODeepGA.
Llama directamente a las funciones originales del proyecto sin modificar su lógica,
agregando medición de huella de carbono, tiempos y métricas de las CNNs generadas.
"""

import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pandas as pd

# Importación directa de la función original
from variants import deepGA, green_DeepGA_v2, green_DeepGA_v3, green_DeepGA_v4
from Decoding import decoding, CNN

# Tracker de carbono opcional (CodeCarbon con fallback analítico)
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


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
        # Número de lotes (batches) por época
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


def get_cifar10_loaders(
    batch_size: int = 64,
    val_split: float = 0.1,
    data_root: str = "/content/drive/MyDrive/CIFAR-10",
    preload_gpu: bool = True,
    device: torch.device = None
):
    """
    Carga y prepara los DataLoaders de CIFAR-10.
    Si preload_gpu=True, precarga el dataset 100% en la VRAM de la GPU con FastGPULoader.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        fallback_root = "./data"
        full_train = datasets.CIFAR10(root=fallback_root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=fallback_root, train=False, download=True, transform=transform_test)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    if preload_gpu:
        print(f"Precargando dataset CIFAR-10 completo en VRAM del dispositivo ({device})...")
        loader_train_all = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        loader_val_all = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        x_train, y_train = next(iter(loader_train_all))
        x_val, y_val = next(iter(loader_val_all))

        x_train_gpu = x_train.to(device, dtype=torch.float32)
        y_train_gpu = y_train.to(device, dtype=torch.long)
        x_val_gpu = x_val.to(device, dtype=torch.float32)
        y_val_gpu = y_val.to(device, dtype=torch.long)

        train_dl = FastGPULoader(x_train_gpu, y_train_gpu, batch_size=batch_size, shuffle=True)
        val_dl = FastGPULoader(x_val_gpu, y_val_gpu, batch_size=batch_size, shuffle=False)
        print("Dataset precargado en VRAM exitosamente.")
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    return train_dl, val_dl, 3, 32, 10  # in_channels, out_size, n_classes


def calculate_cnn_metrics(bestind: list, in_channels: int, out_size: int, n_classes: int):
    """
    Calcula las variables propias de la red neuronal convolucional generada:
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


class ExperimentManager:
    """
    Gestor para ejecutar variantes de DeepGA/MODeepGA y registrar:
    - Huella de carbono (gCO2eq, kWh)
    - Tiempo de ejecución
    - Variables de las redes convolucionales generadas
    """

    def __init__(self, country_iso_code: str = "MEX", track_carbon: bool = True):
        self.country_iso_code = country_iso_code
        self.track_carbon = track_carbon

    def run_deepga(
        self,
        execution: int = 1,
        variant: str = "v3",  # "v1", "v2", "v3", o "v4"
        memoryC: bool = True,
        train_epochs: int = 5,
        population_size: int = 10,  # N
        generations: int = 5,       # T
        lr: float = 1e-4,
        cr: float = 0.7,
        mr: float = 0.5,
        t_size: int = 5,
        w: float = 0.3,
        min_conv: int = 2,
        max_conv: int = 5,
        min_full: int = 1,
        max_full: int = 4,
        max_params: int = 2000000,
        batch_size: int = 64,
        num_workers: int = 2,
        preload_gpu: bool = True,
        data_root: str = "/content/drive/MyDrive/CIFAR-10",
        chck_dir: str = "./checkpoints/",
        device: torch.device = None
    ):
        """
        Ejecuta la variante seleccionada de DeepGA (v1, v2, v3, o v4) sobre CIFAR-10
        midiendo huella de carbono, tiempos y métricas de la CNN.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Cargar CIFAR-10
        print(f"Cargando dataset CIFAR-10 (batch_size={batch_size}) en dispositivo: {device}...")
        train_dl, val_dl, in_channels, out_size, n_classes = get_cifar10_loaders(
            batch_size=batch_size,
            data_root=data_root,
            preload_gpu=preload_gpu,
            device=device
        )
        loss_func = nn.NLLLoss()

        # 2. Iniciar Medición de Carbono y Tiempo
        tracker = None
        if self.track_carbon and CODECARBON_AVAILABLE:
            try:
                os.makedirs("./carbon_logs", exist_ok=True)
                tracker = OfflineEmissionsTracker(
                    country_iso_code=self.country_iso_code,
                    output_dir="./carbon_logs",
                    log_level="error",
                    save_to_file=True
                )
                tracker.start()
            except Exception:
                tracker = None

        start_time = time.perf_counter()

        # 3. Llamar a la variante deseada
        print("\n" + "=" * 50)
        print(f"Iniciando ejecución de DeepGA (Variante: {variant.upper()})...")
        print("=" * 50)
        
        common_args = dict(
            execution=execution,
            memoryC=memoryC,
            train_epochs=train_epochs,
            train_dl=train_dl,
            val_dl=val_dl,
            lr=lr,
            min_conv=min_conv,
            max_conv=max_conv,
            min_full=min_full,
            max_full=max_full,
            max_params=max_params,
            cr=cr,
            mr=mr,
            N=population_size,
            T=generations,
            t_size=t_size,
            w=w,
            device=device,
            chck_dir=chck_dir,
            n_channels=in_channels,
            n_classes=n_classes,
            out_size=out_size,
            loss_func=loss_func
        )

        if variant.lower() == "v4":
            results_df, final_pop, bestind = green_DeepGA_v4(
                **common_args
            )
        elif variant.lower() == "v3":
            results_df, final_pop, bestind = green_DeepGA_v3(
                **common_args,
                num_workers=num_workers
            )
        elif variant.lower() == "v2":
            results_df, final_pop, bestind = green_DeepGA_v2(
                **common_args,
                num_workers=num_workers
            )
        else:
            results_df, final_pop, bestind = deepGA(
                **common_args
            )

        # 4. Detener tiempos y huella de carbono
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
            # Fallback analítico aproximado (TDP 150W)
            power_kw = 0.150
            energy_kwh = power_kw * (elapsed_seconds / 3600.0)
            emissions_g_co2 = energy_kwh * 430.0  # ~430 gCO2eq/kWh promedio

        # 5. Extraer métricas de la CNN del mejor individuo
        cnn_metrics = calculate_cnn_metrics(bestind, in_channels, out_size, n_classes)

        # 6. Consolidar reporte de resultados
        metrics_summary = {
            "execution_time_seconds": round(elapsed_seconds, 2),
            "execution_time_minutes": round(elapsed_seconds / 60.0, 2),
            "carbon_emissions_g_co2": round(emissions_g_co2, 4),
            "energy_consumed_kwh": round(energy_kwh, 6),
            "best_accuracy": bestind[2],
            "best_fitness": bestind[1],
            "best_total_params": cnn_metrics["total_params"],
            "best_trainable_params": cnn_metrics["trainable_params"],
            "best_model_size_mb": cnn_metrics["model_size_mb"],
            "best_estimated_flops": cnn_metrics["estimated_flops"],
            "conv_layers_count": cnn_metrics["conv_layers"],
            "fc_layers_count": cnn_metrics["fc_layers"],
            "skip_connections_count": cnn_metrics["skip_connections"],
            "history_dataframe": results_df,
            "best_individual_raw": bestind,
            "final_population_raw": final_pop
        }

        print("\n" + "=" * 55)
        print("           RESUMEN DE MÉTRICAS DEL EXPERIMENTO")
        print("=" * 55)
        print(f" Tiempo Total de Ejecución:  {metrics_summary['execution_time_seconds']:.2f} s ({metrics_summary['execution_time_minutes']:.2f} min)")
        print(f" Huella de Carbono:          {metrics_summary['carbon_emissions_g_co2']:.4f} gCO2eq")
        print(f" Consumo de Energía:         {metrics_summary['energy_consumed_kwh']:.6f} kWh")
        print("-" * 55)
        print(" VARIABLES DE LA MEJOR CNN GENERADA:")
        print(f"   - Accuracy:                {metrics_summary['best_accuracy']:.2f}%")
        print(f"   - Fitness:                 {metrics_summary['best_fitness']:.4f}")
        print(f"   - Parámetros Totales:      {metrics_summary['best_total_params']:,}")
        print(f"   - Parámetros Entrenables:  {metrics_summary['best_trainable_params']:,}")
        print(f"   - FLOPs Estimados:         {metrics_summary['best_estimated_flops']:,}")
        print(f"   - Tamaño en Memoria:       {metrics_summary['best_model_size_mb']:.3f} MB")
        print(f"   - Capas Convolucionales:   {metrics_summary['conv_layers_count']}")
        print(f"   - Capas Densas (FC):       {metrics_summary['fc_layers_count']}")
        print(f"   - Conexiones Residuales:   {metrics_summary['skip_connections_count']}")
        print("=" * 55)

        return metrics_summary

