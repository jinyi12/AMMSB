# Experiment Catalog

## MMSFM Core (PyTorch)

| Experiment | Script | Description |
|-----------|--------|-------------|
| Synthetic/Single-cell | `archive/2026-02-16_non_fae_scripts/scripts/main.py` | Multi-marginal flow matching with splines |
| Image progression | `archive/2026-02-16_non_fae_scripts/scripts/images/images_main.py` | Class progression on CIFAR-10/Imagenette |
| Latent MSBM | `archive/2026-02-16_non_fae_scripts/scripts/latent_msbm_main.py` | Latent-space Schrödinger Bridge Matching |

## FAE Extensions (JAX/Flax)

| Experiment | Script | Description |
|-----------|--------|-------------|
| Standard FAE | `scripts/fae/fae_naive/train_attention.py` | Unified training (phases A-C), spectral bias mitigation |
| Spectral diagnostic (FAE) | `scripts/fae/fae_naive/analyze_dataset_spectral.py` | PSD-based frequency support summary + MFN `B0` recommendation (see `docs/experiments/mfn_fixed_frequencies.md`) |
| SIREN vs MFN note | `docs/experiments/siren_vs_mfn_multiscale.md` | Mathematical comparison + suggested next steps for multiscale data |
| Combinatorial matrix memory | `docs/experiments/combinatorial_matrix_memory.md` | Coverage audit + run-organization and sweep-planning protocol |
| Combinatorial run registry | `docs/experiments/combinatorial_run_registry.csv` | Live matrix planning sheet with status, best runs, and track labels |
| Run manifest template | `docs/experiments/run_manifest_template.json` | Sidecar schema template for per-run metadata and eval protocol |
| Diffusion denoiser | `scripts/fae/fae_naive/train_attention_denoiser.py` | Diffusion-based denoiser decoder |
| Drifting denoiser | `scripts/fae/fae_naive/train_attention_drifting_denoiser.py` | Drifting denoiser variant |
| Latent MSBM (FAE) | `scripts/fae/fae_naive/train_latent_msbm.py` | MSBM in FAE latent space |

## Decoder Variants

| Decoder | Module | Notes |
|---------|--------|-------|
| Standard MLP | (built-in FAE) | Default decoder |
| Wire2D | `wire2d_decoder.py` | Wire implicit neural representation |
| Residual MFN | *(archived)* | Fixed or learned frequency banks — see `archive/2026-02-22_mfn_archive/` |
| Diffusion denoiser (`denoiser`) | `diffusion_denoiser_decoder.py` | Scaled denoiser backbone |
| Diffusion denoiser (`denoiser_standard`) | `diffusion_denoiser_decoder.py` | Standard-decoder-style MLP backbone |

## Loss Functions

| Loss | Module | Notes |
|------|--------|-------|
| MSE | (built-in FAE) | Standard reconstruction |
| H1 semi-norm | `spectral_losses.py` | Gradient-based, requires full grid |
| Fourier-weighted | `spectral_losses.py` | Frequency-reweighted MSE |
| High-pass residual | `spectral_losses.py` | Targets high-frequency content |
