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
from variants import (
    deepGA, green_DeepGA_v2, green_DeepGA_v3, green_DeepGA_v4,
    green_DeepGA_v5, green_DeepGA_v6, green_DeepGA_v7, green_DeepGA_v8, green_DeepGA_v9, green_DeepGA_v10, green_DeepGA_v11,
    final_evaluation
)
from Decoding import decoding, CNN
from model_utils import (
    save_best_model,
    load_saved_model,
    evaluate_model as util_evaluate_model,
    predict_image as util_predict_image,
    generate_confusion_matrix as util_generate_confusion_matrix,
    download_file,
    download_all_models_zip
)
from dataset_loader import load_dataset_auto, get_custom_imagefolder_loaders


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
    data_root: str = "./data",
    preload_gpu: bool = True,
    device: torch.device = None
):
    """
    Carga y prepara los DataLoaders de CIFAR-10 en 3 particiones independientes:
    1. train_dl: Conjunto de entrenamiento (45,000 imágenes si val_split=0.1)
    2. val_dl: Conjunto de validación (5,000 imágenes si val_split=0.1, para guiar la evolución genética)
    3. test_dl: Conjunto de prueba independiente oficial de CIFAR-10 (10,000 imágenes, para evaluación final y matriz de confusión)
    
    Si preload_gpu=True, precarga los 3 conjuntos 100% en la VRAM de la GPU con FastGPULoader.
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
        target_root = data_root if not data_root.startswith("/content") else "./data"
        os.makedirs(target_root, exist_ok=True)
        print(f"Dataset CIFAR-10 no encontrado en '{data_root}'. Descargando automáticamente en '{target_root}'...")
        full_train = datasets.CIFAR10(root=target_root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=target_root, train=False, download=True, transform=transform_test)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    if preload_gpu:
        print(f"Precargando dataset CIFAR-10 (Train: {train_size}, Val: {val_size}, Test: {len(test_ds)}) en VRAM del dispositivo ({device})...")
        loader_train_all = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        loader_val_all = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        loader_test_all = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

        x_train, y_train = next(iter(loader_train_all))
        x_val, y_val = next(iter(loader_val_all))
        x_test, y_test = next(iter(loader_test_all))

        x_train_gpu = x_train.to(device, dtype=torch.float32)
        y_train_gpu = y_train.to(device, dtype=torch.long)
        x_val_gpu = x_val.to(device, dtype=torch.float32)
        y_val_gpu = y_val.to(device, dtype=torch.long)
        x_test_gpu = x_test.to(device, dtype=torch.float32)
        y_test_gpu = y_test.to(device, dtype=torch.long)

        train_dl = FastGPULoader(x_train_gpu, y_train_gpu, batch_size=batch_size, shuffle=True)
        val_dl = FastGPULoader(x_val_gpu, y_val_gpu, batch_size=batch_size, shuffle=False)
        test_dl = FastGPULoader(x_test_gpu, y_test_gpu, batch_size=batch_size, shuffle=False)
        print("Dataset completo (Train: 45k, Val: 5k, Test: 10k) precargado en VRAM exitosamente.")
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    return train_dl, val_dl, test_dl, 3, 32, 10  # in_channels, out_size, n_classes


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


def save_experiment_report_txt(metrics_summary: dict, chck_dir: str = "./checkpoints/", custom_filename: str = None) -> str:
    """
    Genera y guarda un reporte exhaustivo de texto plano (.txt) con todos los resultados,
    tiempos, huella de carbono, métricas de la CNN ganadora, subrogado/feromonas e historial generacional.
    """
    os.makedirs(chck_dir, exist_ok=True)
    variant = str(metrics_summary.get("variant", "UNKNOWN")).lower()
    execution = metrics_summary.get("execution", 1)

    if custom_filename is None:
        filename = f"reporte_experimento_{variant}_exec_{execution}.txt"
    else:
        filename = custom_filename

    filepath = os.path.join(chck_dir, filename)

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 75)
    lines.append(f"          REPORTE DE EXPERIMENTO - GREEN DEEPGA ({variant.upper()})")
    lines.append("=" * 75)
    lines.append(f"Fecha y Hora:             {now_str}")
    lines.append(f"Variante Ejecutada:       {metrics_summary.get('variant', 'N/A')}")
    lines.append(f"Ejecución ID:             {execution}")
    lines.append(f"Dispositivo de Hardware:  {metrics_summary.get('hardware_device', 'GPU / CPU')}")
    lines.append("-" * 75)
    lines.append("1. TIEMPO Y EFICIENCIA ENERGÉTICA (GREEN AI)")
    lines.append("-" * 75)
    lines.append(f"Tiempo Total de Ejecución: {metrics_summary.get('execution_time_seconds', 0):.2f} s ({metrics_summary.get('execution_time_minutes', 0):.2f} min)")
    lines.append(f"Huella de Carbono:         {metrics_summary.get('carbon_emissions_g_co2', 0):.4f} gCO2eq")
    lines.append(f"Consumo Energético:        {metrics_summary.get('energy_consumed_kwh', 0):.6f} kWh")
    lines.append("-" * 75)
    lines.append("2. RENDIMIENTO DE LA MEJOR ARQUITECTURA (CNN GANADORA)")
    lines.append("-" * 75)
    lines.append(f"Fitness Óptimo:            {metrics_summary.get('best_fitness', 0):.4f}")
    lines.append(f"Precisión Validación (GA): {metrics_summary.get('best_val_accuracy', 0):.2f}% (Validación durante búsqueda evolutiva)")
    if metrics_summary.get('final_test_accuracy') is not None:
        lines.append(f"Precisión Test Set (10k):  {metrics_summary.get('final_test_accuracy', 0):.2f}% (Test set independiente de CIFAR-10)")
    lines.append(f"Parámetros Totales:        {metrics_summary.get('best_total_params', 0):,}")
    lines.append(f"Parámetros Entrenables:    {metrics_summary.get('best_trainable_params', 0):,}")
    lines.append(f"FLOPs Estimados:           {metrics_summary.get('best_estimated_flops', 0):,}")
    lines.append(f"Tamaño en Memoria:         {metrics_summary.get('best_model_size_mb', 0):.3f} MB")
    lines.append(f"Capas Convolucionales:     {metrics_summary.get('conv_layers_count', 0)}")
    lines.append(f"Capas Densas (FC):         {metrics_summary.get('fc_layers_count', 0)}")
    lines.append(f"Conexiones Residuales:     {metrics_summary.get('skip_connections_count', 0)}")
    if metrics_summary.get('saved_model_path'):
        lines.append(f"Ruta Checkpoint .pth:      {os.path.abspath(metrics_summary['saved_model_path'])}")

    # Detalle de capas de la mejor CNN
    bestind = metrics_summary.get("best_individual_raw")
    if bestind and len(bestind) > 0 and hasattr(bestind[0], 'first_level'):
        genome = bestind[0]
        lines.append("-" * 75)
        lines.append("3. DETALLE ESTRUCTURAL CAPA POR CAPA")
        lines.append("-" * 75)
        lines.append("Capas Convolucionales:")
        for idx in range(genome.n_conv):
            if idx < len(genome.first_level):
                l = genome.first_level[idx]
                lines.append(f"  [Conv {idx+1}] Filtros: {l.get('nfilters', 'N/A')}, Kernel: {l.get('fsize', 'N/A')}x{l.get('fsize', 'N/A')}, Pool: {l.get('pool', 'off')}, PoolSize: {l.get('psize', 'N/A')}")
        lines.append("Capas Fully Connected:")
        for f_idx in range(genome.n_full):
            r_idx = genome.n_conv + f_idx
            if r_idx < len(genome.first_level):
                l = genome.first_level[r_idx]
                lines.append(f"  [FC {f_idx+1}] Neuronas: {l.get('neurons', 'N/A')}")
        if hasattr(genome, 'second_level') and genome.second_level:
            lines.append(f"Conexiones Residuales (Bits de Skip): {genome.second_level}")

    # Métricas de búsqueda asistida (subrogado, feromonas y modelo de islas V6 / V8 / V9 / V10 / V11)
    surr = metrics_summary.get("surrogate_metrics")
    if surr:
        lines.append("-" * 75)
        lines.append(f"4. MÉTRICAS DE BÚSQUEDA ASISTIDA ({variant.upper()})")
        lines.append("-" * 75)
        lines.append(f"Candidatos Explorados en CPU:  {surr.get('total_cpu_screened', 0)}")
        lines.append(f"Evaluaciones en GPU:           {surr.get('total_gpu_evaluations', 0)}")
        lines.append(f"Factor de Exploración:         {surr.get('exploration_multiplier', 0)}x más arquitecturas con 0 coste GPU")
        lines.append(f"Error MAE Predicción Fitness:  {surr.get('mean_absolute_error_fit', 0):.4f}")
        lines.append(f"Muestras del Subrogado:        {surr.get('surrogate_training_samples', 0)}")

        # Métricas de Islas (V11)
        if "n_islands" in surr:
            lines.append(f"\nConfiguración del Modelo de Islas (Island Model V11):")
            lines.append(f"  - Número de Islas:           {surr.get('n_islands')}")
            lines.append(f"  - Población por Isla:        {surr.get('pop_per_island')} individuos (Total Población: {surr.get('total_population')})")
            lines.append(f"  - Frecuencia de Migración:   Cada {surr.get('migration_interval')} generaciones")
            lines.append(f"  - Tasa de Migración:         {surr.get('migration_size')} individuos por isla (Topología Anillo)")
            lines.append(f"  - Migraciones Realizadas:    {surr.get('total_migrations_performed')}")
            lines.append(f"  - Diversidad Inter-Islas:    {surr.get('final_inter_island_diversity', 0.0):.4f}")
            lines.append(f"  - Aislamiento de Feromonas:  ESTRICTO (Sin contaminación cruzada entre matrices)")

            islands_sum = surr.get("islands_summary", [])
            if islands_sum:
                lines.append("\nResumen por Cada Isla y Nichos de Feromonas:")
                for isl in islands_sum:
                    lines.append(f"  [Isla {isl['island_id']}] Mejor Fit: {isl['best_fitness']:.4f} | Acc: {isl['best_accuracy']:.2f}% | Params: {isl['best_params']:,} | Conv={isl['conv_layers']}, FC={isl['fc_layers']} | Div Intra-Isla: {isl['intra_island_diversity']:.4f}")
                    pm = isl.get("pheromone_motifs", {})
                    if pm:
                        lines.append(f"    ↳ Feromonas Favorecen: Conv={pm.get('favored_conv_count')}, FC={pm.get('favored_fc_count')}, Skips Activos={len(pm.get('reinforced_skip_connections', []))}")

        p_motifs = surr.get("pheromone_motifs")
        if p_motifs:
            lines.append("\nMotivos Arquitectónicos Favorecidos por Rastro de Feromonas (ACO):")
            lines.append(f"  - Número de Capas Favorecido: Conv={p_motifs.get('favored_conv_count')}, FC={p_motifs.get('favored_fc_count')}")
            for lm in p_motifs.get('layer_motifs', []):
                lines.append(f"  - Capa {lm['layer']}: Filtros={lm['favored_filters']}, Kernel={lm['favored_kernel']}x{lm['favored_kernel']}, Pool={lm['favored_pool']}")
            sk_fav = p_motifs.get('reinforced_skip_connections', [])
            if sk_fav:
                lines.append(f"  - Conexiones Skip Reforzadas (>60% probabilidad): {sk_fav}")

    # Métricas de Poda (V5)
    prun = metrics_summary.get("pruning_metrics")
    if prun:
        lines.append("-" * 75)
        lines.append("4. ANÁLISIS DE PRECISIÓN DE PODA (PRUNING PRECISION V5)")
        lines.append("-" * 75)
        lines.append(f"Modelos Podados:               {prun.get('total_pruned_events', 0)} ({prun.get('pruning_rate_pct', 0)}% de candidatos)")
        lines.append(f"Precisión de Poda:             {prun.get('pruning_precision_pct', 0):.2f}% (Podas Correctas)")
        lines.append(f"Falsas Podas (Modelos Perdidos):{prun.get('good_models_lost_pct', 0):.2f}%")
        lines.append(f"Fitness Promedio Podados:      {prun.get('mean_pruned_fitness', 0):.4f}")
        lines.append(f"Accuracy Promedio Podados:     {prun.get('mean_pruned_accuracy', 0):.2f}%")

    # Historial generacional
    df = metrics_summary.get("history_dataframe")
    if df is not None and not df.empty:
        lines.append("-" * 75)
        lines.append("5. HISTORIAL GENERACIONAL DE LA EVOLUCIÓN")
        lines.append("-" * 75)
        lines.append(df.to_string(index=True))

    lines.append("=" * 75)
    lines.append("                         FIN DEL REPORTE")
    lines.append("=" * 75)

    report_text = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    return filepath


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
        variant: str = "v11",  # "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", o "v11"
        memoryC: bool = True,
        train_epochs: int = 5,
        population_size: int = 12, # N (distribuido entre islas si es V11)
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
        evaluate_pruned_models: bool = True,
        pool_candidates_factor: int = 5,
        kappa: float = 0.1,
        mr_min: float = 0.10,
        mr_max: float = 0.85,
        rho: float = 0.10,
        alpha: float = 1.2,
        top_k_ratio: float = 0.35,
        n_islands: int = 3,
        migration_interval: int = 10,
        migration_size: int = 2,
        data_root: str = "./data",
        img_size: int = 64,
        in_channels: int = 3,
        chck_dir: str = "./checkpoints/",
        device: torch.device = None,
        save_best_model_file: bool = True,
        save_txt_report: bool = True,
        train_final_model: bool = False,
        final_train_epochs: int = 30,
        auto_download: bool = False
    ):
        """
        Ejecuta la variante seleccionada de DeepGA (v1 .. v11) sobre CIFAR-10 o cualquier dataset personalizado
        organizado en carpetas (ej. dataset/covid/{covid, neumonia, normal} o dataset/tumores/{...}).
        Mide huella de carbono, tiempos, métricas de la CNN, precisión de poda (en V5), subrogado (en V6/V8/V9/V10/V11),
        mutación adaptativa (en V9/V10/V11), rastro de feromonas ACO (en V10/V11) y modelo de islas aisladas (en V11).
        
        Parámetros de Dataset:
        - data_root (str): Ruta al dataset (ej. "./data" para CIFAR-10 o "./dataset/covid" para dataset personalizado).
        - img_size (int): Resolución de imagen (por defecto: 64 para datasets personalizados o 32 para CIFAR-10).
        - in_channels (int): Canales de entrada (3 para RGB, 1 para escala de grises).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Cargar Dataset (CIFAR-10 o Dataset Personalizado con partición estratificada)
        train_dl, val_dl, test_dl, in_channels, out_size, n_classes, class_names = load_dataset_auto(
            data_root=data_root,
            img_size=img_size,
            in_channels=in_channels,
            batch_size=batch_size,
            preload_gpu=preload_gpu,
            device=device,
            num_workers=num_workers
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
        print("\n" + "=" * 50, flush=True)
        print(f"Iniciando ejecución de DeepGA (Variante: {variant.upper()})...", flush=True)
        print("=" * 50, flush=True)
        
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

        pruned_stats = None
        surrogate_stats = None

        if variant.lower() == "v11":
            results_df, final_pop, bestind, surrogate_stats = green_DeepGA_v11(
                **common_args,
                n_islands=n_islands,
                migration_interval=migration_interval,
                migration_size=migration_size,
                pool_candidates_factor=pool_candidates_factor,
                kappa=kappa,
                mr_min=mr_min,
                mr_max=mr_max,
                rho=rho,
                alpha=alpha,
                top_k_ratio=top_k_ratio
            )
        elif variant.lower() == "v10":
            results_df, final_pop, bestind, surrogate_stats = green_DeepGA_v10(
                **common_args,
                pool_candidates_factor=pool_candidates_factor,
                kappa=kappa,
                mr_min=mr_min,
                mr_max=mr_max,
                rho=rho,
                alpha=alpha,
                top_k_ratio=top_k_ratio
            )
        elif variant.lower() == "v9":
            results_df, final_pop, bestind, surrogate_stats = green_DeepGA_v9(
                **common_args,
                pool_candidates_factor=pool_candidates_factor,
                kappa=kappa,
                mr_min=mr_min,
                mr_max=mr_max
            )
        elif variant.lower() == "v8":
            results_df, final_pop, bestind, surrogate_stats = green_DeepGA_v8(
                **common_args,
                pool_candidates_factor=pool_candidates_factor,
                kappa=kappa
            )
        elif variant.lower() == "v7":
            results_df, final_pop, bestind = green_DeepGA_v7(
                **common_args
            )
        elif variant.lower() == "v6":
            results_df, final_pop, bestind, surrogate_stats = green_DeepGA_v6(
                **common_args,
                pool_candidates_factor=pool_candidates_factor,
                kappa=kappa
            )
        elif variant.lower() == "v5":
            results_df, final_pop, bestind, pruned_stats = green_DeepGA_v5(
                **common_args,
                evaluate_pruned_models=evaluate_pruned_models
            )
        elif variant.lower() == "v4":
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

        # 6. Opcional: Entrenamiento final completo del modelo ganador
        trained_final_model = None
        final_test_acc = None
        if train_final_model and final_train_epochs > 0:
            print(f"\n🚀 Entrenando completamente el modelo ganador por {final_train_epochs} épocas...")
            print(f"   (Validación durante re-entrenamiento sobre el conjunto de test independiente: 10,000 imágenes)")
            trained_final_model = final_evaluation(
                execution=execution,
                bestind=bestind,
                train_dl=train_dl,
                val_dl=test_dl,  # Evaluación sobre el conjunto de prueba independiente oficial (10,000 imágenes)
                lr=lr,
                max_params=max_params,
                w=w,
                device=device,
                train_epochs=final_train_epochs,
                loss_func=loss_func,
                chck_dir=chck_dir,
                n_channels=in_channels,
                n_classes=n_classes,
                out_size=out_size,
                variant=variant.lower(),
                auto_download=auto_download
            )

        # Si el modelo final está entrenado, calcular precisión en test set independiente
        if trained_final_model is not None:
            try:
                final_test_acc = util_evaluate_model(trained_final_model, test_dl, device=device)
            except Exception:
                final_test_acc = None

        # 7. Guardar y/o descargar automáticamente el mejor modelo
        saved_model_path = None
        if save_best_model_file:
            saved_model_path = save_best_model(
                variant=variant.lower(),
                execution=execution,
                bestind=bestind,
                in_channels=in_channels,
                out_size=out_size,
                n_classes=n_classes,
                chck_dir=chck_dir,
                trained_model=trained_final_model,
                cnn_metrics=cnn_metrics,
                auto_download=auto_download
            )

        # 8. Consolidar reporte de resultados
        device_str = str(device)
        if torch.cuda.is_available() and device.type == "cuda":
            device_str += f" ({torch.cuda.get_device_name(0)})"

        metrics_summary = {
            "variant": variant.upper(),
            "execution": execution,
            "hardware_device": device_str,
            "execution_time_seconds": round(elapsed_seconds, 2),
            "execution_time_minutes": round(elapsed_seconds / 60.0, 2),
            "carbon_emissions_g_co2": round(emissions_g_co2, 4),
            "energy_consumed_kwh": round(energy_kwh, 6),
            "best_accuracy": bestind[2],
            "best_val_accuracy": bestind[2],
            "final_test_accuracy": final_test_acc,
            "best_fitness": bestind[1],
            "best_total_params": cnn_metrics["total_params"],
            "best_trainable_params": cnn_metrics["trainable_params"],
            "best_model_size_mb": cnn_metrics["model_size_mb"],
            "best_estimated_flops": cnn_metrics["estimated_flops"],
            "conv_layers_count": cnn_metrics["conv_layers"],
            "fc_layers_count": cnn_metrics["fc_layers"],
            "skip_connections_count": cnn_metrics["skip_connections"],
            "saved_model_path": saved_model_path,
            "history_dataframe": results_df,
            "best_individual_raw": bestind,
            "final_population_raw": final_pop,
            "final_trained_model": trained_final_model,
            "val_dataloader": val_dl,
            "test_dataloader": test_dl,
            "class_names": class_names,
            "txt_report_path": None
        }

        if pruned_stats is not None:
            metrics_summary["pruning_metrics"] = {
                "total_candidates_checked": pruned_stats.get("total_candidates_checked", 0),
                "total_pruned_events": pruned_stats.get("total_pruned_events", 0),
                "pruning_rate_pct": pruned_stats.get("pruning_rate_pct", 0.0),
                "unique_pruned_count": pruned_stats.get("unique_pruned_count", 0),
                "correct_prunings": pruned_stats.get("correct_prunings", 0),
                "false_prunings": pruned_stats.get("false_prunings", 0),
                "pruning_precision_pct": pruned_stats.get("pruning_precision_pct", 0.0),
                "good_models_lost_pct": pruned_stats.get("good_models_lost_pct", 0.0),
                "mean_pruned_fitness": pruned_stats.get("mean_pruned_fitness", 0.0),
                "mean_pruned_accuracy": pruned_stats.get("mean_pruned_accuracy", 0.0),
                "pruned_evaluation_dataframe": pruned_stats.get("pruned_evaluation_df", pd.DataFrame()),
                "pruned_records_raw": pruned_stats.get("pruned_records_raw", [])
            }

        if surrogate_stats is not None:
            metrics_summary["surrogate_metrics"] = surrogate_stats

        # 9. Guardar Reporte de Resultados en archivo TXT
        txt_report_path = None
        if save_txt_report:
            try:
                txt_report_path = save_experiment_report_txt(metrics_summary, chck_dir=chck_dir)
                metrics_summary["txt_report_path"] = txt_report_path
            except Exception as e:
                print(f"Nota al generar reporte .txt: {e}", flush=True)

        print("\n" + "=" * 55, flush=True)
        print("           RESUMEN DE MÉTRICAS DEL EXPERIMENTO", flush=True)
        print("=" * 55, flush=True)
        print(f" Variante:                   {metrics_summary['variant']}", flush=True)
        print(f" Dispositivo:                {metrics_summary['hardware_device']}", flush=True)
        print(f" Tiempo Total de Ejecución:  {metrics_summary['execution_time_seconds']:.2f} s ({metrics_summary['execution_time_minutes']:.2f} min)", flush=True)
        print(f" Huella de Carbono:          {metrics_summary['carbon_emissions_g_co2']:.4f} gCO2eq", flush=True)
        print(f" Consumo de Energía:         {metrics_summary['energy_consumed_kwh']:.6f} kWh", flush=True)
        if saved_model_path:
            print(f" Checkpoint del Modelo:      {saved_model_path}", flush=True)
        if txt_report_path:
            print(f" Reporte TXT Generado:       {os.path.abspath(txt_report_path)}", flush=True)
        print("-" * 55, flush=True)
        print(" VARIABLES DE LA MEJOR CNN GENERADA:", flush=True)
        print(f"   - Val Accuracy (GA 5k):    {metrics_summary['best_val_accuracy']:.2f}%", flush=True)
        if metrics_summary.get('final_test_accuracy') is not None:
            print(f"   - Test Accuracy (Test 10k):{metrics_summary['final_test_accuracy']:.2f}%", flush=True)
        print(f"   - Fitness:                 {metrics_summary['best_fitness']:.4f}", flush=True)
        print(f"   - Parámetros Totales:      {metrics_summary['best_total_params']:,}", flush=True)
        print(f"   - Parámetros Entrenables:  {metrics_summary['best_trainable_params']:,}", flush=True)
        print(f"   - FLOPs Estimados:         {metrics_summary['best_estimated_flops']:,}", flush=True)
        print(f"   - Tamaño en Memoria:       {metrics_summary['best_model_size_mb']:.3f} MB", flush=True)
        print(f"   - Capas Convolucionales:   {metrics_summary['conv_layers_count']}", flush=True)
        print(f"   - Capas Densas (FC):       {metrics_summary['fc_layers_count']}", flush=True)
        print(f"   - Conexiones Residuales:   {metrics_summary['skip_connections_count']}", flush=True)
        
        if pruned_stats is not None:
            p_met = metrics_summary["pruning_metrics"]
            print("-" * 55, flush=True)
            print(" ANÁLISIS DE PRECISIÓN DE PODA (PRUNING PRECISION):", flush=True)
            print(f"   - Modelos Podados:         {p_met['total_pruned_events']} ({p_met['pruning_rate_pct']}% de candidatos)", flush=True)
            print(f"   - Precisión de Poda:       {p_met['pruning_precision_pct']:.2f}% (Podas Correctas)", flush=True)
            print(f"   - Falsas Podas (Perdidos): {p_met['good_models_lost_pct']:.2f}%", flush=True)
            print(f"   - Fitness Prom. Podados:   {p_met['mean_pruned_fitness']:.4f}", flush=True)
            print(f"   - Accuracy Prom. Podados:  {p_met['mean_pruned_accuracy']:.2f}%", flush=True)

        if surrogate_stats is not None:
            print("-" * 55, flush=True)
            print(f" MÉTRICAS DE BÚSQUEDA ASISTIDA ({metrics_summary['variant']}):", flush=True)
            print(f"   - Candidatos en CPU:       {surrogate_stats['total_cpu_screened']}", flush=True)
            print(f"   - Evaluaciones en GPU:     {surrogate_stats['total_gpu_evaluations']}", flush=True)
            print(f"   - Factor de Exploración:   {surrogate_stats['exploration_multiplier']}x más búsqueda con 0 coste GPU", flush=True)
            print(f"   - Error MAE Predicción:    {surrogate_stats['mean_absolute_error_fit']:.4f}", flush=True)
            print(f"   - Muestras del Subrogado:  {surrogate_stats['surrogate_training_samples']}", flush=True)
            if 'n_islands' in surrogate_stats:
                print(f"   - Modelo de Islas (V11):   {surrogate_stats['n_islands']} Islas | {surrogate_stats['total_migrations_performed']} Migraciones realizadas | Div Inter-Islas: {surrogate_stats.get('final_inter_island_diversity', 0.0):.4f}", flush=True)
            if 'pheromone_motifs' in surrogate_stats:
                pm = surrogate_stats['pheromone_motifs']
                print(f"   - Feromonas Favorecen:     Conv={pm.get('favored_conv_count')}, FC={pm.get('favored_fc_count')}, Skips={len(pm.get('reinforced_skip_connections', []))}", flush=True)

        print("=" * 55, flush=True)

        return metrics_summary

    def save_report_txt(self, metrics_summary: dict, chck_dir: str = "./checkpoints/", filename: str = None) -> str:
        """Guarda un reporte de texto exhaustivo con los resultados del experimento."""
        return save_experiment_report_txt(metrics_summary, chck_dir=chck_dir, custom_filename=filename)

    def load_model(self, model_path: str, device: torch.device = None):
        """Carga y reconstruye un modelo guardado a partir de su ruta .pth o .pkl."""
        return load_saved_model(model_path, device=device)

    def evaluate_model(self, model_or_path, dataloader, device: torch.device = None):
        """Evalúa la precisión de un modelo sobre un DataLoader (ej. test_dataloader con 10,000 imágenes)."""
        return util_evaluate_model(model_or_path, dataloader, device=device)

    def predict_image(self, model_or_path, image_path: str, class_names: list = None, device: torch.device = None):
        """Realiza inferencia sobre una imagen propia."""
        return util_predict_image(model_or_path, image_path, class_names=class_names, device=device)

    def generate_confusion_matrix(
        self,
        model_or_path,
        dataloader=None,
        class_names: list = None,
        device: torch.device = None,
        title: str = "Matriz de Confusión - DeepGA",
        save_fig_path: str = None,
        auto_download_plot: bool = False
    ):
        """Calcula y grafica la matriz de confusión sobre un DataLoader."""
        return util_generate_confusion_matrix(
            model_or_path=model_or_path,
            dataloader=dataloader,
            class_names=class_names,
            device=device,
            title=title,
            save_fig_path=save_fig_path,
            auto_download_plot=auto_download_plot
        )

    def download_model(self, variant: str = "v9", execution: int = 1, chck_dir: str = "./checkpoints/"):
        """Descarga el archivo .pth del modelo ganador de la variante seleccionada."""
        model_file = os.path.join(chck_dir, f"best_model_{variant.lower()}_exec_{execution}.pth")
        return download_file(model_file)

    def download_all_models(self, chck_dir: str = "./checkpoints/", zip_name: str = "deepga_best_models.zip"):
        """Empaqueta todos los modelos guardados en un ZIP y los descarga."""
        return download_all_models_zip(chck_dir=chck_dir, zip_name=zip_name, auto_download=True)

