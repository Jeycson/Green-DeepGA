# -*- coding: utf-8 -*-
"""
Clase principal de alto nivel DeepGASearch (Centrada en DeepGA V10).
Permite a cualquier usuario configurar la búsqueda mediante código,
inyectar datasets personalizados (rutas o DataLoaders de PyTorch)
y cambiar de modelo a arquitecturas personalizadas o evaluadores externos.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Tuple, Callable
from torch.utils.data import DataLoader
import pandas as pd

from deepga.config import DeepGAV10Config, DeepGAConfig
from deepga.data.loaders import load_dataset_auto
from deepga.core.decoding import decoding, CNN
from deepga.core.evaluator import BaseEvaluator, CNNEvaluator, ModelEvaluator, CustomEvaluator
from variants.DeepGA_V10 import green_DeepGA_v10, final_evaluation


class DeepGASearch:
    """
    Optimizador evolutivo de alto nivel DeepGA V10.

    Ejemplo 1 (CNN con dataset propio):
    >>> from deepga import DeepGASearch, DeepGAConfig
    >>> searcher = DeepGASearch(config=DeepGAConfig.optimal())
    >>> searcher.fit(dataset="./mis_imagenes")
    >>> model = searcher.get_best_model()

    Ejemplo 2 (Modelo personalizado no-CNN):
    >>> def mi_mlp_builder(genome):
    ...     return MiRedMLP(hidden=genome.n_full * 64)
    >>> searcher.fit(train_dl=loader_train, val_dl=loader_val, model_builder=mi_mlp_builder)
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
        self.evaluator: Optional[BaseEvaluator] = None

    def fit(
        self,
        dataset: Optional[Union[str, DataLoader]] = None,
        train_dl: Optional[DataLoader] = None,
        val_dl: Optional[DataLoader] = None,
        test_dl: Optional[DataLoader] = None,
        model_builder: Optional[Callable[[Any], nn.Module]] = None,
        evaluator: Optional[BaseEvaluator] = None,
        loss_func: Optional[nn.Module] = None
    ) -> "DeepGASearch":
        """
        Ejecuta el proceso evolutivo de DeepGA V10.

        Args:
            dataset: Ruta a una carpeta con imágenes (ImageFolder) o nombre ("cifar10").
            train_dl: DataLoader personalizado de entrenamiento.
            val_dl: DataLoader personalizado de validación (usado para fitness).
            test_dl: DataLoader personalizado de prueba (para evaluación final).
            model_builder: Función constructora personalizada que mapea un individuo a un nn.Module.
                           Permite cambiar de modelo a MLPs, Transformers o arquitecturas propias.
            evaluator: Evaluador que implemente BaseEvaluator para tareas de optimización arbitrarias.
            loss_func: Función de pérdida de PyTorch (por defecto CrossEntropyLoss).

        Returns:
            self (instancia entrenada de DeepGASearch)
        """
        # 1. Resolver DataLoaders si se proporcionan o cargar dataset
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
        elif evaluator is not None and self.train_dl is None:
            # Caso de optimización pura con evaluador externo (sin DataLoaders de PyTorch)
            self.train_dl = DataLoader([0])
            self.val_dl = DataLoader([0])
            self.test_dl = self.val_dl
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

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()

        # 2. Configurar Evaluador personalizado si se especificó model_builder o evaluator
        if model_builder is not None:
            print("🧠 Configurando ModelEvaluator con arquitectura de red personalizada...")
            self.evaluator = ModelEvaluator(
                model_builder=model_builder,
                train_dl=self.train_dl,
                val_dl=self.val_dl,
                device=self.device,
                num_epochs=self.config.train_epochs,
                lr=self.config.learning_rate,
                w=self.config.weight_params,
                loss_func=loss_func,
                memory_cleanup=self.config.memory_cleanup
            )
        elif evaluator is not None:
            self.evaluator = evaluator

        print("\n" + "=" * 65)
        print(f"🚀 INICIANDO BÚSQUEDA EVOLUTIVA DeepGA V10 (ACO-Enhanced)")
        print(f"   Población: {self.config.pop_size} | Generaciones: {self.config.generations}")
        print(f"   Dispositivo: {self.device} | Épocas búsqueda: {self.config.train_epochs}")
        if self.evaluator is not None:
            print(f"   Evaluador Activo: {self.evaluator.__class__.__name__}")
        print("=" * 65 + "\n")

        # 3. Ejecutar Algoritmo Genético V10
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
            save_txt=self.config.save_txt,
            evaluator=self.evaluator
        )

        self.best_individual = bestind
        self.results_df = results_df
        self.surrogate_stats = surrogate_stats

        # 4. Decodificación y re-entrenamiento del mejor modelo
        print("\n" + "=" * 65)
        print("🏆 Búsqueda completada. Recuperando la mejor arquitectura encontrada...")
        print("=" * 65)

        if self.evaluator is not None and hasattr(self.evaluator, 'build_model'):
            self.best_model = self.evaluator.build_model(bestind[0])
        elif hasattr(bestind[0], 'first_level'):
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
            "model_state_dict": self.best_model.state_dict() if hasattr(self.best_model, 'state_dict') else None,
            "config": self.config.to_dict()
        }, output_path)
        print(f"💾 Modelo guardado exitosamente en '{output_path}'.")


# Alias para claridad
DeepGAV10Search = DeepGASearch
