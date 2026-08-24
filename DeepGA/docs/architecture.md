# Architecture

## Execution flow

```text
CLI example / batch script
          │
          ▼
deepga.experiment.manager.ExperimentManager.run_deepga
          ├── deepga.data: train, validation, test loaders + metadata
          ├── variants/: evolutionary search
          │       ├── deepga.core: genome and decoding
          │       ├── deepga.evolution: selection, crossover, mutation
          │       └── deepga.training: short candidate training/evaluation
          └── deepga.utils: model, metrics, reports, plots, CSV summaries
```

## Genome and decoding

`deepga.core.encoding.Encoding` represents an architecture using convolutional and fully connected blocks plus a second-level binary encoding for convolutional connectivity. `deepga.core.decoding.decoding` translates that representation into layer specifications, and `deepga.core.decoding.CNN` constructs the executable PyTorch module.

The search bounds default to 2–5 convolutional blocks and 1–4 fully connected blocks. `max_params` rejects or penalizes oversized models. The exact neural building blocks and connectivity semantics should be treated as the source code definition in `deepga/core/encoding.py` and `deepga/core/decoding.py` when publishing results.

## Training and objectives

Every candidate is trained for the configured `train_epochs` and assessed on the validation loader. The single-objective lineage optimizes an aggregate of accuracy and parameter efficiency; `w` controls the complexity penalty. Higher `w` puts more pressure on small models and can reduce accuracy substantially.

Multi-objective variants maintain non-dominated solutions over validation accuracy and estimated carbon emissions. They use non-dominated sorting, crowding distance, hypervolume tracking, and report a highest-accuracy, lowest-carbon, and knee-point solution.

Carbon measurement is handled by `ExperimentManager`: CodeCarbon's offline tracker is used when installed and initialized successfully. Otherwise, the manager estimates energy with a fixed 150 W assumption and converts it using 430 gCO2eq/kWh. This fallback is only a coarse comparative proxy, not a hardware measurement.

## Variant lineage

| Variant | Main additions visible in the implementation |
| --- | --- |
| `v1` | Baseline DeepGA |
| `v2`–`v5` | Early green/search-efficiency extensions, including pruning checks in V5 |
| `v6` | Surrogate-assisted candidate evaluation |
| `v7` | Graph-aware crossover/mutation operators |
| `v8` | Surrogate-assisted V7 lineage |
| `v9` | Adaptive mutation |
| `v10` | Random-forest surrogate plus pheromone-guided (ACO) evolution |
| `v11` | V10-style search with multi-island evolution and migration |
| `v12` | Pure multi-island V12 search, diversity-aware mutation, crowding migration, anti-stagnation, AMP/memory safeguards |
| `mo_v9` | NSGA-II-style multi-objective search with dual surrogate |
| `mo_v10` | Multi-objective pheromone-guided search |
| `mo_v11` | Multi-objective islands, per-island pheromones, migration |

The variants are research alternatives, not interchangeable implementations. Compare only runs with matching dataset protocol, seeds, budgets, and training schedule.

## Legacy modules

The top-level `DeepGA.py`, `MODeepGA.py`, `WangDeepGA.py`, `MOWang.py`, `Wang*`, `DataReader.py`, and compatibility bridges such as `Operators*.py` preserve earlier paths. Several import a missing `Training.py` or contain machine-specific image paths, so they are not the supported entry points for a fresh environment. Use `deepga.experiment.manager.ExperimentManager` and `variants/` for new work.
