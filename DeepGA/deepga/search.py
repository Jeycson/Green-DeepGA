# -*- coding: utf-8 -*-
"""
Clase principal de alto nivel DeepGASearch (Centrada en DeepGA V10).
Permite a cualquier usuario configurar la búsqueda mediante código,
inyectar datasets personalizados (rutas o DataLoaders de PyTorch)
y obtener el modelo óptimo en pocas líneas.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Tuple
from torch.utils.data import DataLoader
import pandas as pd

from deepga.config import DeepGAV10Config, DeepGAConfig
from deepga.data.loaders import load_dataset_auto
from deepga.core.decoding import decoding, CNN
from deepga.core.evaluator import BaseEvaluator, CNNEvaluator, CustomEvaluator
from variants.DeepGA_V10 import green_DeepGA_v10, final_evaluation


class DeepGASearch:
    """
    Optimizador evolutivo de alto nivel DeepGA V10.

    Ejemplo de uso rápido:
    >>> from deepga import DeepGASearch, DeepGAConfig
    >>> config = DeepGAConfig(pop_size=8, generations=3, train_epochs=1)
    >>> searcher = DeepGASearch(config=config)
    >>> searcher.fit(dataset="./mis_imagenes")
    >>> model = searcher.get_best_model()
    """

    def __init__(self, config: Optional[DeepGAV10Config] = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = DeepGAV10Config(**kwargs)

        # Determinar dispositivo de cómputo
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Variables de estado tras la búsqueda
        self.best_individual = None
        self.best_model: Optional[nn.Module] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.surrogate_stats: Optional[Dict[str, Any]] = None
        self.train_dl: Optional[DataLoader] = None
        self.val_dl: Optional[DataLoader] = None
        self.test_dl: Optional[DataLoader] = None
        self.class_names = []

    def fit(
        self,
        dataset: Optional[Union[str, DataLoader]] = None,
        train_dl: Optional[DataLoader] = None,
        val_dl: Optional[DataLoader] = None,
        test_dl: Optional[DataLoader] = None,
        evaluator: Optional[BaseEvaluator] = None,
        loss_func: Optional[nn.Module] = None
    ) -> "DeepGASearch":
        """
        Ejecuta el proceso evolutivo de DeepGA V10.

        Args:
            dataset: Ruta a una carpeta con imágenes (ImageFolder) o nombre ("cifar10").
            train_dl: DataLoader personalizado de entrenamiento.
            val_dl: DataLoader personalizado de validación (usado para calcular fitness).
            test_dl: DataLoader personalizado de prueba (para evaluación final).
            evaluator: Evaluador opcional que implemente BaseEvaluator para tareas personalizadas.
            loss_func: Función de pérdida de PyTorch (por defecto CrossEntropyLoss).

        Returns:
            self (instancia entrenada de DeepGASearch)
        """
        # 1. Resolver y preparar los datos
        if train_dl is not None and val_dl is not None:
            self.train_dl = train_dl
            self.val_dl = val_dl
            self.test_dl = test_dl or val_dl
        elif isinstance(dataset, str):
            print(f"📦 Cargando y procesando dataset desde: '{dataset}'...")
            t_dl, v_dl, te_dl, in_ch, img_sz, n_cls, classes = load_dataset_auto(
                data_root=dataset,
                img_size=self.config.image_size,
                in_channels=self.config.n_channels,
                batch_size=self.config.batch_size,
                preload_gpu=self.config.preload_gpu,
                device=self.device
            )
            self.train_dl = t_dl
            self.val_dl = v_dl
            self.test_dl = te_dl
            self.config.n_channels = in_ch
            self.config.image_size = img_sz
            self.config.n_classes = n_cls
            self.class_names = classes
            print(f"✅ Dataset cargado: {n_cls} clases, {in_ch} canales, {img_sz}x{img_sz} resolución.")
        elif dataset is None and self.train_dl is None:
            # Fallback a CIFAR-10
            print("ℹ️  No se proporcionó dataset. Cargando CIFAR-10 por defecto...")
            t_dl, v_dl, te_dl, in_ch, img_sz, n_cls, classes = load_dataset_auto(
                data_root="./data",
                img_size=self.config.image_size,
                in_channels=self.config.n_channels,
                batch_size=self.config.batch_size,
                preload_gpu=self.config.preload_gpu,
                device=self.device
            )
            self.train_dl = t_dl
            self.val_dl = v_dl
            self.test_dl = te_dl
            self.config.n_channels = in_ch
            self.config.image_size = img_sz
            self.config.n_classes = n_cls
            self.class_names = classes
        else:
            raise ValueError("Debe proporcionar 'dataset' como ruta o tuplas de DataLoaders (train_dl, val_dl).")

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()

        print("\n" + "=" * 60)
        print(f"🚀 INICIANDO BÚSQUEDA EVOLUTIVA DeepGA V10 (ACO-Enhanced)")
        print(f"   Población: {self.config.pop_size} | Generaciones: {self.config.generations}")
        print(f"   Dispositivo: {self.device} | Épocas búsqueda: {self.config.train_epochs}")
        print("=" * 60 + "\n")

        # 2. Ejecutar Algoritmo Genético V10
        results_df, final_pop, bestind, surrogate_stats = green_DeepGA_v10(
            execution=self.config.execution,
            memoryC=self.config.memory_cleanup,
            train_epochs=self.config.train_epochs,
            train_dl=self.train_dl,
            val_dl=self.val_dl,
            lr=self.config.learning_rate,
            min_conv=self.config.min_conv,
            max_conv=self.config.max_conv,
            min_full=self.config.min_full,
            max_full=self.config.max_full,
            max_params=self.config.max_params,
            cr=self.config.crossover_rate,
            mr=self.config.mutation_rate,
            N=self.config.pop_size,
            T=self.config.generations,
            t_size=self.config.tournament_size,
            w=self.config.weight_params,
            device=self.device,
            chck_dir=self.config.checkpoint_dir,
            n_channels=self.config.n_channels,
            n_classes=self.config.n_classes,
            out_size=self.config.image_size,
            loss_func=loss_func,
            pool_candidates_factor=self.config.pool_candidates_factor,
            kappa=self.config.kappa,
            mr_min=self.config.mr_min,
            mr_max=self.config.mr_max,
            rho=self.config.rho,
            alpha=self.config.alpha,
            top_k_ratio=self.config.top_k_ratio,
            test_dl=self.test_dl,
            dataset_name=self.config.dataset_name,
            seed=self.config.seed,
            save_txt=self.config.save_txt
        )

        self.best_individual = bestind
        self.results_df = results_df
        self.surrogate_stats = surrogate_stats

        # 3. Entrenamiento y decodificación final del mejor individuo
        print("\n" + "=" * 60)
        print("🏆 Búsqueda completada. Re-entrenando la mejor arquitectura encontrada...")
        print("=" * 60)

        # Instanciar el modelo de PyTorch para el mejor individuo
        network = decoding(
            bestind[0],
            self.config.n_channels,
            self.config.image_size,
            self.config.n_classes
        )
        self.best_model = CNN(bestind[0], network[0], network[1], network[2]).to(self.device)

        if self.config.final_epochs > 0 and self.test_dl is not None:
            final_acc, final_pars, confusion = final_evaluation(
                execution=self.config.execution,
                bestind=bestind,
                train_dl=self.train_dl,
                val_dl=self.test_dl,
                lr=self.config.learning_rate,
                num_epochs=self.config.final_epochs,
                loss_func=loss_func,
                device=self.device,
                n_channels=self.config.n_channels,
                n_classes=self.config.n_classes,
                out_size=self.config.image_size
            )
            print(f"🎯 Accuracy en Test del mejor modelo: {final_acc:.4f} | Parámetros: {final_pars:,}")

        return self

    def get_best_model(self) -> Optional[nn.Module]:
        """Retorna el modelo PyTorch instanciado de la mejor arquitectura."""
        return self.best_model

    def get_best_genome(self) -> Any:
        """Retorna el genoma (Encoding) del mejor individuo."""
        if self.best_individual is not None:
            return self.best_individual[0]
        return None

    def summary(self) -> pd.DataFrame:
        """Retorna el historial de evolución como un DataFrame de Pandas."""
        if self.results_df is not None:
            return self.results_df
        return pd.DataFrame()

    def save(self, output_path: str):
        """Guarda los pesos del mejor modelo entrenado y su genoma."""
        if self.best_model is None:
            raise RuntimeError("No hay un modelo entrenado para guardar. Ejecuta fit() primero.")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save({
            "genome": self.best_individual,
            "model_state_dict": self.best_model.state_dict(),
            "config": self.config.to_dict()
        }, output_path)
        print(f"💾 Modelo guardado exitosamente en '{output_path}'.")


# Alias para claridad
DeepGAV10Search = DeepGASearch
