# Getting started

## Requirements

- Python 3.9 or newer is recommended. The historical code was written for Python 3.7, but the dependency constraints in `requirements.txt` are lower bounds rather than a fully pinned environment.
- PyTorch and torchvision compatible with the selected Python version and, when applicable, CUDA driver. Install the PyTorch build appropriate for the target machine before installing the remaining requirements.
- A CUDA GPU is strongly recommended for evolution. CPU execution is supported for a smoke test but can be impractically slow.
- R plus the `irace` package are required only for `irace_tuning/`.

Install the core environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install MedMNIST only when using one of its datasets:

```bash
python -m pip install medmnist
```

Check the runtime after installation:

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
python ejemplo_local.py --help
```

## First execution

Use a deliberately small V12 run first. `--no-preload-gpu` avoids storing the complete dataset in VRAM and is the safer initial choice.

```bash
python ejemplo_local.py \
  --variant v12 --execution 1 --seed 42 \
  --pop-size 6 --generations 1 --train-epochs 1 --final-epochs 1 \
  --data-root ./data --img-size 32 --in-channels 3 \
  --chck-dir ./checkpoints/smoke --no-preload-gpu
```

When `./data` is not an ImageFolder dataset, the loader treats it as CIFAR-10 and downloads CIFAR-10 if absent. Network access is therefore needed for a first CIFAR-10 run.

Expected outputs include a saved checkpoint/model, a plain-text report, metrics summaries, and a confusion-matrix image under the chosen checkpoint directory. Exact names are documented in [Reference](reference.md#artifacts).

## Importing from Python

New code should import from the canonical package layout rather than from the
legacy root-level module names:

```python
from deepga.experiment.manager import ExperimentManager
from deepga.data.loaders import load_dataset_auto
from deepga.core.encoding import Encoding
```

The old imports remain as compatibility bridges for existing notebooks and
saved research scripts, but should not be used for new work.

## Common execution modes

Single-objective architecture evolution:

```bash
python ejemplo_local.py --variant v12 --execution 10 --seed 10 \
  --pop-size 12 --generations 10 --train-epochs 3 --final-epochs 20 \
  --data-root /path/to/dataset --img-size 64 --in-channels 1 \
  --chck-dir ./checkpoints/experiment_10
```

Multi-objective evolution (accuracy versus emission estimate):

```bash
python ejemplo_mo.py --variant mo_v11 --execution 10 \
  --pop-size 12 --generations 10 --train-epochs 3 \
  --data-root /path/to/dataset --img-size 64 --in-channels 1 \
  --chck-dir ./checkpoints/mo_10 --no-preload-gpu
```

Re-train a stored winner:

```bash
python entrenar_modelo_guardado.py \
  --model-path ./checkpoints/experiment_10/best_model_v12_exec_10.pth \
  --epochs 30 --data-root /path/to/dataset --batch-size 32 --lr 1e-4
```

## Operational advice

- Start with `preload_gpu=False`/`--no-preload-gpu`; enable preload only after confirming that the full train, validation, and test tensors fit in VRAM.
- For images of 128 pixels or larger, the automatic loader disables GPU preload to reduce memory pressure.
- Do not overwrite a prior experiment directory. Use one output directory per run so that checkpoints and appended summaries remain attributable.
- The example flag `--use-amp` is enabled by default. It is useful on CUDA but should be validated for numerical stability on a new environment.
