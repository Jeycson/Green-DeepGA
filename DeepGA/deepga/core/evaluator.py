# -*- coding: utf-8 -*-
"""
Módulo de Abstracción de Evaluadores para DeepGA.
Permite desacoplar el motor genético (V10) del tipo específico de modelo o problema,
haciendo posible optimizar tanto CNNs de PyTorch como funciones u otros modelos personalizados.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepga.core.decoding import decoding, CNN
from deepga.training.engine import training


class BaseEvaluator(ABC):
    """
    Clase base abstracta para cualquier evaluador de fitness en DeepGA.
    Cualquier problema a optimizar debe implementar el método `evaluate`.
    """

    @abstractmethod
    def evaluate(self, individual: Any) -> Tuple[float, float, int]:
        """
        Evalúa un individuo generado por el algoritmo genético.

        Returns:
            Tuple con:
                - fit (float): Valor de fitness a maximizar (entre 0 y 1 o mayor).
                - metric (float): Métrica de desempeño principal (ej. accuracy).
                - complexity (int): Complejidad del modelo (ej. número de parámetros).
        """
        pass


class CNNEvaluator(BaseEvaluator):
    """
    Evaluador especializado para arquitecturas CNN estándar en PyTorch.
    Decodifica el genoma a módulos PyTorch y entrena durante `num_epochs` para medir accuracy.
    """

    def __init__(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        n_channels: int = 3,
        out_size: int = 32,
        n_classes: int = 10,
        device: Optional[torch.device] = None,
        num_epochs: int = 2,
        lr: float = 0.001,
        w: float = 0.1,
        max_params: int = 1_500_000,
        loss_func: Optional[Any] = None,
        memory_cleanup: bool = True
    ):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.n_channels = n_channels
        self.out_size = out_size
        self.n_classes = n_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.lr = lr
        self.w = w
        self.max_params = max_params
        self.loss_func = loss_func or torch.nn.CrossEntropyLoss()
        self.memory_cleanup = memory_cleanup

    def evaluate(self, individual: Any) -> Tuple[float, float, int]:
        network = decoding(individual, self.n_channels, self.out_size, self.n_classes)
        cnn = CNN(individual, network[0], network[1], network[2])
        acc_list = []
        
        fit, acc, pars, _ = training(
            '1',
            self.device,
            cnn,
            self.num_epochs,
            self.loss_func,
            self.train_dl,
            self.val_dl,
            self.lr,
            self.w,
            self.max_params,
            acc_list
        )

        if self.memory_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return fit, acc, pars


class ModelEvaluator(BaseEvaluator):
    """
    Evaluador para arquitecturas PyTorch personalizadas NO-CNN (ej. MLPs, Transformers, Autoencoders).
    El desarrollador inyecta su propia función constructora:
        model_builder(individual) -> nn.Module
    """

    def __init__(
        self,
        model_builder: Callable[[Any], nn.Module],
        train_dl: DataLoader,
        val_dl: DataLoader,
        device: Optional[torch.device] = None,
        num_epochs: int = 2,
        lr: float = 0.001,
        w: float = 0.10,
        loss_func: Optional[nn.Module] = None,
        memory_cleanup: bool = True
    ):
        self.model_builder = model_builder
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.lr = lr
        self.w = w
        self.loss_func = loss_func or nn.CrossEntropyLoss()
        self.memory_cleanup = memory_cleanup

    def build_model(self, individual: Any) -> nn.Module:
        """Construye la instancia del modelo PyTorch a partir del individuo."""
        return self.model_builder(individual).to(self.device)

    def evaluate(self, individual: Any) -> Tuple[float, float, int]:
        model = self.build_model(individual)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Entrenamiento rápido por num_epochs
        model.train()
        for _ in range(self.num_epochs):
            for xb, yb in self.train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = self.loss_func(pred, yb)
                loss.backward()
                optimizer.step()

        # Evaluación en validación
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in self.val_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                preds = pred.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = (correct / total * 100.0) if total > 0 else 0.0

        # Función de fitness con penalización de parámetros
        param_penalty = (num_params / 1_500_000.0) if num_params > 0 else 0.0
        fit = (acc / 100.0) - (self.w * param_penalty)

        if self.memory_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(fit), float(acc), int(num_params)


class CustomEvaluator(BaseEvaluator):
    """
    Evaluador genérico para optimizar cualquier función o modelo no-PyTorch (Scikit-Learn, XGBoost, funciones matemáticas).
    Permite a los usuarios inyectar una función personalizada `eval_fn(individual) -> float | tuple`.
    """

    def __init__(self, eval_fn: Callable[[Any], Any]):
        self.eval_fn = eval_fn

    def evaluate(self, individual: Any) -> Tuple[float, float, int]:
        res = self.eval_fn(individual)
        if isinstance(res, (int, float)):
            return float(res), float(res), 0
        elif isinstance(res, (tuple, list)):
            fit = float(res[0])
            metric = float(res[1]) if len(res) > 1 else fit
            complexity = int(res[2]) if len(res) > 2 else 0
            return fit, metric, complexity
        else:
            raise ValueError(f"El retorno de eval_fn debe ser un float o tuple, recibido: {type(res)}")
