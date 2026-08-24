# Reference

## Main Python API

```python
from experiment_manager import ExperimentManager

manager = ExperimentManager(country_iso_code="MEX", track_carbon=True)
result = manager.run_deepga(
    variant="v12", execution=1, seed=42,
    population_size=12, generations=10, train_epochs=3,
    data_root="/path/to/dataset", img_size=64, in_channels=1,
    chck_dir="./checkpoints/run_001", preload_gpu=False,
    train_final_model=False,
)
```

`run_deepga` accepts the core GA settings (`cr`, `mr`, `t_size`, `w`, bounds, and `max_params`), surrogate/pheromone settings, island settings, data-loading settings, and output controls. See its signature and docstring in `deepga/experiment/manager.py` for the authoritative full parameter list.

The returned dictionary contains execution metadata, validation/test metrics, CNN metrics, paths, data loaders, final population, and, when relevant, `surrogate_stats`, `mo_stats`, and Pareto information. Its exact contents vary by variant, so consumers should use `.get()` for optional keys.

Convenience methods include `load_model`, `evaluate_model`, `predict_image`, `generate_confusion_matrix`, `generate_pareto_front_plot`, `train_saved_model`, and `download_all_models`.

## Command-line scripts

| Script | Purpose |
| --- | --- |
| `ejemplo_local.py` | Single-objective run using the manager |
| `ejemplo_mo.py` | Multi-objective run and Pareto plot |
| `ejemplo_train_val.py` | Train/validation-focused example |
| `ejemplo_resnet18.py` | ResNet-18 baseline run |
| `ejecutar_experimentos.py` | Batch DeepGA orchestrator |
| `ejecutar_experimentos_resnet18.py` | Batch ResNet-18 orchestrator |
| `verificar_experimentos.py` | Inspect batch/output state |
| `entrenar_modelo_guardado.py` | Re-train a stored `.pth` or `.pkl` model |

Use `python <script> --help` to inspect the actual options in the checked-out version.

## Artifacts

The exact set depends on options and variant. Typical files under `chck_dir` include:

| Artifact | Meaning |
| --- | --- |
| `best_model_<variant>_exec_<n>.pth` | PyTorch checkpoint with architecture metadata and possibly trained weights |
| `best_model_<variant>_exec_<n>.pkl` | Pickled companion checkpoint/genome data |
| `checkpoint_...pkl` | Evolution state used for variant-specific resume behavior |
| `reporte_experimento_...txt` | Human-readable summary |
| `experiments_summary.*` / individual records | Appended machine-readable and human-readable summaries from `model_utils` |
| `matriz_confusion_...png` | Test/target-loader confusion matrix |
| `pareto_front_...png` | Multi-objective frontier visualization |

Pickle files are unsafe to open from untrusted sources. Load only artifacts created by a trusted DeepGA environment.

## Compatibility and legacy code

The former root-level imports (`experiment_manager`, `dataset_loader`, `Decoding`, `Operators_V12`, and related modules) remain as thin compatibility bridges. They are retained so older notebooks and variant code keep working, but all new imports must use the `deepga.*` namespace.

The `variants/` directory remains at the repository root because it is the experiment lineage and its names are used by saved checkpoints and tuning scripts. It is the next safe migration boundary once an automated regression suite exists.

## Legacy comparison code

The `Wang*` modules implement the Wang encoding comparison used by the earlier study. Their top-level execution scripts depend on legacy modules and paths; they should be modernized and tested separately before use in a new study.
