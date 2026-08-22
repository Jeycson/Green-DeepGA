"""                                                                                                          
Runner Individual de ResNet-18 con soporte para:                                                             
1. Datasets personalizados (Covid, BreastMNIST, CIFAR-10, etc.)                                              
2. Normalización y métricas idénticas a DeepGA (20 columnas TSV/CSV)                                         
3. Medición de huella de carbono y energía (CodeCarbon)                                                      
4. Matriz de confusión y checkpoints de pesos (.pth)                                                         
"""                                                                                                          
                                                                                                                 
import os                                                                                                    
import sys                                                                                                   
import time                                                                                                  
import argparse                                                                                              
import numpy as np                                                                                           
import torch                                                                                                 
import torch.nn as nn                                                                                        
from torchvision import models                                                                               
                                                                                                                
# Importar utilidades y loaders existentes en el proyecto                                                    
from dataset_loader import load_dataset_auto                                                                 
from model_utils import (                                                                                    
    compute_classification_metrics,                                                                          
    save_experiment_record,                                                                                  
    generate_confusion_matrix                                                                                
)                                                                                                            
                                                                                                                
# Tracker de carbono opcional
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecución de ResNet-18 con métricas normalizadas DeepGA")
    parser.add_argument("--execution", type=int, default=1,
                        help="ID numérico de la ejecución")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla aleatoria (por defecto igual a execution)")
    parser.add_argument("--data-root", type=str, default="./Datasets/Covid",
                        help="Ruta al dataset (ej. ./Datasets/Covid o ./data)")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Resolución de las imágenes (ej. 28, 64, 128, 224)")
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 3],
                        help="Canales de entrada: 1 para escala de grises/Covid, 3 para RGB")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (Learning Rate)")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Usar pesos preentrenados de ImageNet (default: False, entrenamiento desde cero)")
    parser.add_argument("--chck-dir", type=str, default="./checkpoints/",
                        help="Directorio de salida para checkpoints y reportes")
    parser.add_argument("--country-iso", type=str, default="MEX",
                        help="Código ISO del país para huella de carbono")
    parser.add_argument("--device", type=str, default=None,
                        help="Dispositivo ('cuda', 'cpu' o None para auto-detección)")
    return parser.parse_args()


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


def adapt_input_channels(xb: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Adapta canales del tensor de entrada según lo que espera la red convolucional."""
    if xb.shape[1] == 3 and target_channels == 1:
        return xb.mean(dim=1, keepdim=True)
    elif xb.shape[1] == 1 and target_channels == 3:
        return xb.repeat(1, 3, 1, 1)
    return xb


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

    # Inferencia simulada de 1 muestra
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


def train_epoch(model, dataloader, criterion, optimizer, device):
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
    print(f"\n📦 Cargando dataset desde '{args.data_root}'...")                                              
    train_dl, val_dl, test_dl, in_channels, img_size, n_classes, class_names = load_dataset_auto(            
        data_root=args.data_root,                                                                            
        img_size=args.img_size,                                                                              
        in_channels=args.in_channels,                                                                        
        batch_size=args.batch_size,                                                                          
        preload_gpu=True,                                                                                    
        device=device                                                                                        
    )                                                                                                        
                                                                                                                
    dataset_name = os.path.basename(os.path.normpath(args.data_root))                                        
    if not dataset_name or dataset_name == ".":                                                              
        dataset_name = "Dataset"                                                                             
                                                                                                                
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
        # Fallback analítico aproximado (TDP ~150W)                                                          
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
        save_fig_path=cm_path,                                                                               
        auto_download_plot=False                                                                             
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
                                                                                                                
    record_res = save_experiment_record(exp_data, chck_dir=args.chck_dir, print_console=True)                
    print("✨ Ejecución de ResNet-18 finalizada exitosamente.")                                              
                                                                                                                
                                                                                                                
if __name__ == "__main__":                                                                                   
    main()      