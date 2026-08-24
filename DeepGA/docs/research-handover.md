# Research handover

## Current baseline and ownership boundary

Use `deepga.experiment.manager.ExperimentManager` plus `variants/` as the maintained execution path. For a new single-objective study, start with `v12`; for a green multi-objective study, start with `mo_v11`. Older variants are valuable ablations, not default production baselines.

The codebase combines original DeepGA research code with later Green AI, surrogate, ACO, island, and multi-objective extensions. Its behavior is defined by the current commit, not by the historical README alone.

## First actions for the next researcher

1. Create a clean environment and complete the small V12 smoke run.
2. Validate the Windows DataLoader worker configuration described in [Reproducibility](reproducibility.md) if that platform is in scope.
3. Create a locked environment (`pip freeze` or Conda export) after validating the target CUDA stack.
4. Prepare a versioned dataset manifest with patient-level splitting if data are clinical.
5. Run a fixed-seed baseline suite: ResNet-18, V10, V11, V12, and optionally MO-V11, all under an identical budget.
6. Archive outputs outside the repository's ignored patterns and attach the command, commit, environment, and dataset manifest to each run.
7. Import new functionality from `deepga.*`; do not add new logic to root-level compatibility bridges.

## Research backlog

These are high-value, bounded next steps rather than claims of completed work:

- Add automated smoke tests using a tiny synthetic ImageFolder and CPU-safe settings, covering decoding, loaders, saving/loading, and one generation.
- Add a lockfile and CI matrix for supported Python/PyTorch versions.
- Replace implicit runtime splits with persisted split manifests and expose a dataset seed through the public CLI.
- Separate selection validation from final test evaluation in final training.
- Standardize checkpoints and result schemas across all variants.
- Measure energy at process/GPU level under a documented protocol and report uncertainty instead of relying on the fallback estimate.
- Modernize or retire legacy `Training.py`-dependent and Wang entry points.
- Make variant-specific algorithm descriptions and ablation tables part of the next paper/release.

## Decision log template

For every new branch of research, create a short decision record containing:

```text
Date / owner:
Question:
Commit and environment:
Dataset and split manifest:
Methods and fixed budget:
Seeds:
Primary metric and selection rule:
Carbon measurement mode/boundary:
Outcome and artifact location:
Follow-up:
```

This small record prevents the most common handover failure: results that cannot be attributed to a code version, data split, or experimental decision.
