# Reproducibility and result integrity

## Minimum run record

For each execution, store the following with the generated artifacts:

- Git commit hash and any uncommitted diff.
- Python version, PyTorch/torchvision versions, CUDA runtime/driver, GPU model, operating system, and whether AMP and GPU preload were used.
- Dataset source/version, license/access conditions, class order, transforms, exact split manifest, image size, and channel count.
- Variant name and every search/training setting passed to `run_deepga`.
- Evolutionary seed, execution ID, and all independent runs included in an aggregate.
- Carbon tracker mode (`CodeCarbon` or analytical fallback), country ISO code, and measurement boundaries.

The manager saves useful run artifacts but cannot infer all of this external context. A practical pattern is to save the command line and a copy of `git status --short` next to each output directory before a batch starts.

## Randomness

`ejemplo_local.py` seeds PyTorch from `--seed` (or `--execution`). The manager passes the seed to variants, but the code also uses Python `random`, NumPy, DataLoader behavior, CUDA kernels, augmentation, and dataset splitting. Exact bitwise replication is not guaranteed. If strict repeatability is required, seed Python and NumPy in the launcher, configure deterministic PyTorch/CUDA behavior, and use materialized split manifests.

## Evaluation discipline

- Candidate fitness is validation performance after limited training; it is not test performance.
- The convenience option `train_final_model=True` currently validates the final re-training against the test loader when one exists. For an unbiased scientific evaluation, avoid using that test loader for epoch selection or early stopping; supply a separate final-validation split and evaluate the test set only once afterward.
- Report macro metrics for imbalanced medical datasets, not accuracy alone.
- Carbon figures from the fallback are estimates based on a constant power and grid factor; do not label them as measured emissions.

## Known handover risks

1. `requirements.txt` specifies lower bounds, not locked versions. Create a tested lockfile/environment export before a long study.
2. `medmnist` is optional at runtime and is intentionally not in the core requirements; install and record it when used.
3. The repository's `.gitignore` excludes `*.txt`, `*.png`, model files, and logs. Important generated reports may therefore be untracked unless copied to a results archive or force-added deliberately.
4. Historical top-level drivers import a missing `Training.py`; they are legacy paths. The supported current route is `ExperimentManager`.
5. The canonical loader (`deepga.data.loaders`) imports `sys` for its Windows worker guard. Validate DataLoader multiprocessing on the target Windows/PyTorch combination before launching a long run.

## Validation checklist before publishing

- [ ] Smoke run completed from a clean environment.
- [ ] Dataset classes and split counts inspected in the run log.
- [ ] Same budget and seeds used for each compared method.
- [ ] Test set was not used during candidate selection or final epoch selection.
- [ ] Artifacts, environment record, and commit are archived.
- [ ] Aggregate statistics and per-seed values are reported.
