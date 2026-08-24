# Experiment guide

## Recommended study protocol

1. Create a fixed, versioned dataset split. Keep a held-out test set untouched.
2. Run a small smoke experiment to validate loaders, class mapping, CUDA, and checkpoint writing.
3. Pre-register the comparison: variants, population, generations, per-candidate epochs, final epochs, objective weights, and number of seeds.
4. Run each configuration with the same independent seed set. Use a distinct `execution` ID and output directory for every run.
5. Select an architecture on validation performance only. Re-train it as specified, then evaluate the held-out test set once.
6. Aggregate the per-run CSV/report artifacts; report spread, not only the best run. Preserve model/genome files for representative results.

## Parameters that materially change a comparison

| Parameter | Meaning | Notes |
| --- | --- | --- |
| `population_size` (`N`) | Candidate population | Island modes round to `max(4, N // n_islands) * n_islands`; record effective N. |
| `generations` (`T`) | Evolutionary budget | More generations also create more candidate evaluations. |
| `train_epochs` | Short training per candidate | Fitness quality versus cost trade-off. |
| `final_train_epochs` | Winner re-training | Must be fixed across compared methods. |
| `w` | Single-objective complexity penalty | Do not compare with different values without stating it. |
| `cr`, `mr`, `t_size` | Core GA dynamics | Record all three. |
| `img_size`, `in_channels` | Input configuration | Affect both accuracy and architecture size. |
| island/migration settings | V11/V12/MO-V11 topology | Change effective population and exploration. |

## Batch execution

`ejecutar_experimentos.py` invokes `ejemplo_local.py` across seed and variant combinations. First inspect its options:

```bash
python ejecutar_experimentos.py --help
```

Example small batch:

```bash
python ejecutar_experimentos.py \
  --start-seed 1 --end-seed 3 --start-exec 1 \
  --variants v10 v11 v12 \
  --data-root /path/to/dataset --img-size 64 --in-channels 1 \
  --pop-size 12 --generations 10 --train-epochs 3 --final-epochs 20 \
  --chck-dir ./checkpoints/baseline
```

`ejecutar_experimentos_resnet18.py` and `ejemplo_resnet18.py` provide a non-evolutionary ResNet-18 baseline using the same dataset/metrics framework. Use this baseline when assessing whether NAS offers value over a conventional architecture.

## Multi-objective interpretation

The multi-objective modes do not yield one universally best model. Archive the complete Pareto front, its hypervolume history, reference point, carbon tracking mode, and hardware. The reported knee point is a project-defined compromise, not a clinical or deployment decision rule.

## Hyperparameter tuning

The `irace_tuning/` workflow tunes V10–V12. It invokes the manager through a target runner and uses a normalized cost based on macro F1 and energy. Follow its dedicated [README](../irace_tuning/README.md), but treat `baseline.csv`, `best_configuration.json`, and `irace.Rdata` as historical artifacts unless you can reproduce their dataset and environment.
