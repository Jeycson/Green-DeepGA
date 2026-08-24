# Datasets

`dataset_loader.load_dataset_auto` chooses a loader from `data_root`:

| Input | Detection rule | Resulting split |
| --- | --- | --- |
| MedMNIST identifier, e.g. `breastmnist` | Name matches a supported MedMNIST flag and is not a class-folder dataset | Official train / validation / test |
| Pre-split ImageFolder | Root contains `train/` plus `test/` or `val/` | Training split plus a validation subset from training; supplied `test`/`val` is held out |
| Class-folder ImageFolder | Root has two or more class subdirectories | Stratified 70% / 15% / 15% train / validation / test |
| Other or empty path | Fallback | CIFAR-10: 90% / 10% split of official train plus official test |

## ImageFolder layouts

Unsplit folders:

```text
dataset/
├── healthy/
│   ├── image_001.png
│   └── ...
├── pneumonia/
└── covid/
```

Pre-split folders:

```text
dataset/
├── train/
│   ├── healthy/
│   ├── pneumonia/
│   └── covid/
└── test/                         # `val/` may be used instead
    ├── healthy/
    ├── pneumonia/
    └── covid/
```

The same class directories must exist in each supplied split. Class labels are derived alphabetically from folder names; record the resulting class order with every experiment.

## Transform and split behavior

- `in_channels=1`: grayscale conversion, resize, train-time horizontal flip and ±10° rotation, then normalization using mean/std `0.5`.
- `in_channels=3`: RGB conversion, resize, the same flip/rotation, mild brightness/contrast jitter, and ImageNet mean/std normalization.
- The class-folder split uses `random_state=42` internally. This is independent of the evolutionary `seed` supplied to `ExperimentManager`.
- Stratification uses scikit-learn when available; otherwise the implementation falls back to a per-class random split.

For strict reproducibility, materialize and version-control the train/validation/test file lists rather than relying on runtime splitting. For clinical datasets, split by patient/study before importing the data into DeepGA to avoid leakage; the repository's image-level split cannot enforce patient grouping.

## Two-split mode

`ExperimentManager.run_deepga(..., use_2split=True, val_ratio=0.15)` creates only train and validation sets. This is useful during tuning, but it does not provide an independent test set. Do not report validation performance from that mode as final generalization performance.

## MedMNIST

Install `medmnist` separately, then pass a flag such as `breastmnist`, `pneumoniamnist`, or `pathmnist` as `data_root`. The loader downloads the official splits under `./data` when needed. It reports the canonical class names provided by MedMNIST.

## VRAM preload

With `preload_gpu=True` and CUDA available, all splits are converted to tensors and stored in GPU memory through `FastGPULoader`. This is fast but its memory cost is approximately the combined image tensors plus labels, before model and optimizer allocations. Disable it for large datasets, high resolution, or limited VRAM.
