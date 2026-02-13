# Experiment Catalog

## MMSFM Core (PyTorch)

| Experiment | Script | Description |
|-----------|--------|-------------|
| Synthetic/Single-cell | `scripts/main.py` | Multi-marginal flow matching with splines |
| Image progression | `scripts/images/images_main.py` | Class progression on CIFAR-10/Imagenette |
| Latent MSBM | `scripts/latent_msbm_main.py` | Latent-space Schrödinger Bridge Matching |

## FAE Extensions (JAX/Flax)

| Experiment | Script | Description |
|-----------|--------|-------------|
| Standard FAE | `scripts/fae/fae_naive/train_attention.py` | Unified training (phases A-C), spectral bias mitigation |
| Diffusion denoiser | `scripts/fae/fae_naive/train_attention_denoiser.py` | Diffusion-based denoiser decoder |
| Drifting denoiser | `scripts/fae/fae_naive/train_attention_drifting_denoiser.py` | Drifting denoiser variant |
| Latent MSBM (FAE) | `scripts/fae/fae_naive/train_latent_msbm.py` | MSBM in FAE latent space |

## Decoder Variants

| Decoder | Module | Notes |
|---------|--------|-------|
| Standard MLP | (built-in FAE) | Default decoder |
| RFF readout | `fourier_enhanced_decoder.py` | Random Fourier Features output layer |
| Enhanced RFF | `enhanced_rff_decoder.py` | Multi-band RFF (Phase C) |
| MLP decoder | `mmlp_decoder.py` | Custom MLP variant |
| Wire2D | `wire2d_decoder.py` | Wire implicit neural representation |
| Diffusion denoiser | `diffusion_denoiser_decoder.py` | Score-based denoising |
| Diffusion locality | `diffusion_locality_denoiser_decoder.py` | Locality-aware denoising |

## Loss Functions

| Loss | Module | Notes |
|------|--------|-------|
| MSE | (built-in FAE) | Standard reconstruction |
| H1 semi-norm | `spectral_losses.py` | Gradient-based, requires full grid |
| Fourier-weighted | `spectral_losses.py` | Frequency-reweighted MSE |
| High-pass residual | `spectral_losses.py` | Targets high-frequency content |
