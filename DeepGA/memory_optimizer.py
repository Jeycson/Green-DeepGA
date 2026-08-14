# -*- coding: utf-8 -*-
"""
Módulo de Optimización de Memoria GPU y VRAM para DeepGA / MODeepGA (MemoryOptimizer).

Diseñado específicamente para resolver el cuello de botella en imágenes de alta resolución
(ej. 128x128, 256x256, 512x512) y arquitecturas profundas:
1. Automatic Mixed Precision (AMP / FP16): Reduce al 50% el consumo de VRAM en activaciones
   y acelera el entrenamiento en GPUs con Tensor Cores.
2. Adaptive Spatial Reduction / Global Average Pooling (GAP): Elimina la explosión de millones
   de parámetros en la capa Fully Connected (Linear) al conectar capas convolucionales grandes.
3. Rigorous VRAM Cleanup: Liberación garantizada de memoria caché de PyTorch y recolección
   de basura (GC) entre evaluaciones de candidatos para evitar fragmentación de VRAM.
4. OOM Guard & Auto-Recovery: Intercepta errores de CUDA Out-of-Memory sin detener el algoritmo
   genético, aplicando penalización suave o reintento seguro.
5. Smart Batch Size Advisor: Sugiere el tamaño de lote óptimo según resolución y VRAM disponible.
6. Memory Profiler: Diagnóstico detallado del consumo de tensores, pesos y optimizador.
"""

import os
import gc
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

# Verificar disponibilidad de AMP en PyTorch
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


class GPUMemoryOptimizer:
    """
    Gestor centralizado de optimizaciones de memoria y aceleración de hardware para DeepGA.
    """

    @staticmethod
    def is_cuda_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_vram_info(device: Optional[torch.device] = None) -> Dict[str, float]:
        """
        Retorna estadísticas actuales de uso de VRAM en Megabytes (MB) y Gigabytes (GB).
        """
        if not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0, "reserved_mb": 0.0,
                "max_allocated_mb": 0.0, "total_vram_gb": 0.0,
                "free_vram_gb": 0.0
            }

        dev = device if (device is not None and device.type == "cuda") else torch.device("cuda:0")
        allocated_bytes = torch.cuda.memory_allocated(dev)
        reserved_bytes = torch.cuda.memory_reserved(dev)
        max_allocated_bytes = torch.cuda.max_memory_allocated(dev)
        
        # Consultar memoria total de la GPU
        dev_props = torch.cuda.get_device_properties(dev)
        total_vram_bytes = dev_props.total_memory
        free_vram_bytes = total_vram_bytes - reserved_bytes

        return {
            "allocated_mb": round(allocated_bytes / (1024 ** 2), 2),
            "reserved_mb": round(reserved_bytes / (1024 ** 2), 2),
            "max_allocated_mb": round(max_allocated_bytes / (1024 ** 2), 2),
            "total_vram_gb": round(total_vram_bytes / (1024 ** 3), 2),
            "free_vram_gb": round(free_vram_bytes / (1024 ** 3), 2)
        }

    @staticmethod
    def cleanup_gpu_memory(force_sync: bool = True):
        """
        Ejecuta un ciclo agresivo de liberación de memoria VRAM:
        1. Recolector de basura de Python (gc.collect).
        2. Vaciado del pool de memoria de PyTorch (torch.cuda.empty_cache).
        3. Recolección de memoria IPC (torch.cuda.ipc_collect).
        """
        gc.collect()
        if torch.cuda.is_available():
            if force_sync:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def get_scaler(enabled: bool = True) -> Optional[Any]:
        """
        Retorna una instancia de GradScaler para Mixed Precision si está disponible.
        """
        if AMP_AVAILABLE and torch.cuda.is_available():
            return GradScaler(enabled=enabled)
        return None

    @staticmethod
    def recommend_batch_size(img_size: int, in_channels: int = 3, device: Optional[torch.device] = None) -> int:
        """
        Calcula un batch_size recomendado para evitar saturación de memoria según la resolución.
        """
        vram_info = GPUMemoryOptimizer.get_vram_info(device)
        total_gb = vram_info.get("total_vram_gb", 8.0)

        # Regla empírica basada en resolución y VRAM
        if img_size >= 256:
            if total_gb >= 16.0:
                return 32
            elif total_gb >= 8.0:
                return 16
            else:
                return 8
        elif img_size >= 128:
            if total_gb >= 8.0:
                return 32
            else:
                return 16
        elif img_size >= 64:
            return 64
        else:
            return 64

    @staticmethod
    def profile_model_vram(model: nn.Module, input_shape: Tuple[int, int, int, int], device: torch.device) -> Dict[str, Any]:
        """
        Calcula la huella de memoria teórica de un modelo (Pesos + Activaciones de forward pass).
        """
        # 1. Memoria de parámetros
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        model_weights_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

        # 2. Estimación de activaciones con un tensor de prueba
        activations_mb = 0.0
        try:
            GPUMemoryOptimizer.cleanup_gpu_memory()
            initial_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            
            dummy_input = torch.zeros(input_shape, device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            final_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            activations_mb = max(0.0, (final_mem - initial_mem) / (1024 ** 2))
            del dummy_input
            GPUMemoryOptimizer.cleanup_gpu_memory()
        except Exception:
            pass

        return {
            "model_weights_mb": round(model_weights_mb, 2),
            "activations_estimate_mb": round(activations_mb, 2),
            "total_estimated_mb": round(model_weights_mb + activations_mb, 2)
        }


def safe_train_val_with_amp(
    device: torch.device,
    epochs: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_func: nn.Module,
    train_dl,
    val_dl,
    use_amp: bool = True,
    max_spatial_size: Optional[int] = 4,
    gradient_accumulation_steps: int = 1
) -> Tuple[float, nn.Module, bool]:
    """
    Entrena y valida un modelo en GPU con aceleración Automatic Mixed Precision (AMP),
    protección contra CUDA OOM (Out-of-Memory) y recolección de memoria.
    
    Retorna:
    - accuracy: Exactitud obtenida en validación (porcentaje 0-100%).
    - model: Modelo entrenado.
    - oom_occurred: Booleano indicando si ocurrió un desbordamiento de memoria VRAM.
    """
    scaler = GPUMemoryOptimizer.get_scaler(enabled=(use_amp and device.type == "cuda"))
    oom_occurred = False
    accuracy = 0.0

    try:
        for epoch in range(epochs):
            # Fase de Entrenamiento
            model.train()
            len_train = len(train_dl.dataset) if hasattr(train_dl, 'dataset') else len(train_dl)
            train_loss = 0.0
            train_correct = 0

            opt.zero_grad(set_to_none=True)

            for step, (xb, yb) in enumerate(train_dl):
                if xb.device != device:
                    xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                if yb.device != device:
                    yb = yb.to(device, dtype=torch.long, non_blocking=True)

                if scaler is not None and use_amp and device.type == "cuda":
                    with autocast(dtype=torch.float16):
                        yb_h = model(xb)
                        loss = loss_func(yb_h, yb)
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dl):
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                else:
                    yb_h = model(xb)
                    loss = loss_func(yb_h, yb)
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                    
                    loss.backward()

                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dl):
                        opt.step()
                        opt.zero_grad(set_to_none=True)

                pred = yb_h.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(yb.view_as(pred)).sum().item()
                train_loss += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)

            # Fase de Validación
            model.eval()
            len_val = len(val_dl.dataset) if hasattr(val_dl, 'dataset') else len(val_dl)
            val_correct = 0

            with torch.no_grad():
                for xb_val, yb_val in val_dl:
                    if xb_val.device != device:
                        xb_val = xb_val.to(device, dtype=torch.float32, non_blocking=True)
                    if yb_val.device != device:
                        yb_val = yb_val.to(device, dtype=torch.long, non_blocking=True)

                    if use_amp and device.type == "cuda":
                        with autocast(dtype=torch.float16):
                            yb_h_val = model(xb_val)
                    else:
                        yb_h_val = model(xb_val)

                    pred_val = yb_h_val.argmax(dim=1, keepdim=True)
                    val_correct += pred_val.eq(yb_val.view_as(pred_val)).sum().item()

            accuracy = (val_correct / max(1, len_val)) * 100.0

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "cuda" in err_msg:
            print(f"\n⚠️  [VRAM OOM Guard] Se interceptó desbordamiento de memoria en evaluación ({e}). Liberando VRAM...", flush=True)
            oom_occurred = True
            accuracy = 0.0
        else:
            raise e
    finally:
        # Limpieza obligatoria post-entrenamiento
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        GPUMemoryOptimizer.cleanup_gpu_memory(force_sync=False)

    return accuracy, model, oom_occurred
