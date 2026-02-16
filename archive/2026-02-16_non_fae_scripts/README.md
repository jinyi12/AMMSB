# Legacy (non-FAE) scripts archive (2026-02-16)

This folder contains legacy training/evaluation scripts that are **not part of the active Functional Autoencoder (FAE) workflow**.

## What moved here

- `scripts/*.py` entrypoints for MMSFM core training, latent-flow/MSBM experiments, and geodesic/NeuralODE autoencoder training.
- `scripts/archive/` (older scratch/experimental scripts).
- `scripts/images/images_*.py` (image-progression trainer/eval/plot scripts).

The only pieces intentionally kept under the active `scripts/` tree are:

- `scripts/fae/` (FAE pipelines and helpers)
- `scripts/utils.py` and `scripts/wandb_compat.py` (shared utilities)
- `scripts/images/field_visualization.py` (shared visualization utilities used by FAE evaluation)

## How to run legacy scripts

Run them by path, e.g.:

```bash
python archive/2026-02-16_non_fae_scripts/scripts/main.py --help
python archive/2026-02-16_non_fae_scripts/scripts/latent_msbm_main.py --help
python archive/2026-02-16_non_fae_scripts/scripts/images/images_main.py --help
```

