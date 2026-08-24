# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:31:05 2020
Modified 2026: Soporte de Automatic Mixed Precision (AMP), OOM Guard y optimización de memoria VRAM.

@author: user
"""

import gc
import torch
from torch import nn
from torch import optim

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


#Helper function to compute de loss on a batch
def loss_batch(loss_func, xb, yb, yb_h, opt = None, scaler = None):
  #Obtain the loss
  loss = loss_func(yb_h, yb)
  #Obtain peformance metric
  metric_b = metrics_batch(yb, yb_h)
  if opt is not None:
    if scaler is not None and AMP_AVAILABLE:
      scaler.scale(loss).backward()
      scaler.step(opt)
      scaler.update()
      opt.zero_grad(set_to_none=True)
    else:
      loss.backward()
      opt.step()
      opt.zero_grad(set_to_none=True)

  return loss.item(), metric_b

#Helper function to compute the accuracy per mini_batch
def metrics_batch(target, output):
  #Obtain output class
  pred = output.argmax(dim=1, keepdim = True)
  #Compare output class with target class
  corrects = pred.eq(target.view_as(pred)).sum().item()

  return corrects

#Helper function to compute the loss and metric values for a dataset
def loss_epoch(device, model, loss_func, dataset_dl, opt = None, scaler = None, use_amp = True):
  loss = 0.0
  metric = 0.0
  len_data = len(dataset_dl.dataset) if hasattr(dataset_dl, 'dataset') else len(dataset_dl)
  is_cuda = (device.type == "cuda") if hasattr(device, 'type') else (str(device) == "cuda")

  for i, data in enumerate(dataset_dl, 0):
    xb, yb = data
    if xb.device != device:
      xb = xb.to(device, dtype=torch.float32, non_blocking=True)
    elif xb.dtype != torch.float32:
      xb = xb.type(torch.float32)

    if yb.device != device:
      yb = yb.to(device, dtype=torch.long, non_blocking=True)
    elif yb.dtype != torch.long:
      yb = yb.long()

    #Obtain model output con aceleración AMP si está disponible
    if use_amp and is_cuda and AMP_AVAILABLE:
      with autocast(dtype=torch.float16):
        yb_h = model(xb)
        loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt, scaler=scaler)
    else:
      yb_h = model(xb)
      loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt, scaler=None)

    loss += loss_b
    if metric_b is not None:
      metric += metric_b

  loss /= max(1, len_data)
  metric /= max(1, len_data)
    
  return loss, metric

#Define the training function
def train_val(device, epochs, model, opt, loss_func, train_dl, test_dl, use_amp = True):
  is_cuda = (device.type == "cuda") if hasattr(device, 'type') else (str(device) == "cuda")
  scaler = GradScaler(enabled=(use_amp and is_cuda)) if (AMP_AVAILABLE and is_cuda) else None

  accuracy = 0.0
  try:
    for epoch in range(epochs):
      model.train()
      train_loss, train_metric = loss_epoch(device, model, loss_func, train_dl, opt, scaler=scaler, use_amp=use_amp)

      model.eval()
      with torch.no_grad():
        val_loss, val_metric = loss_epoch(device, model, loss_func, test_dl, opt=None, scaler=None, use_amp=use_amp)
      accuracy = val_metric
  except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    if "out of memory" in str(e).lower():
      print(f"⚠️  [OOM Warning] Memoria GPU agotada en entrenamiento. Liberando caché...", flush=True)
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      accuracy = 0.0
    else:
      raise e

  return accuracy, model

def training(num, device, model, n_epochs, loss_func, train_dl, test_dl, lr, w, max_params, acc_list, use_amp = True):
    #Number of parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.to(device)

    #Optimizer
    opt = optim.Adam(model.parameters(), lr = lr)

    #Obtaining training accuracy
    accuracy, modelTrainded = train_val(device, n_epochs, model, opt, loss_func, train_dl, test_dl, use_amp=use_amp)

    #Fitness function based on accuracy and No. of parameters
    f = (1 - w)*accuracy + w*((max_params - params)/max_params)

    # Limpieza de memoria
    if opt is not None:
      opt.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    return f, accuracy, params, modelTrainded

        
        
        
    
    