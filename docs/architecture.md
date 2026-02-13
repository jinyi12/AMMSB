# Architecture

## Module Map

```
MMSFM/
├── mmsfm/                          # Core Python package (PyTorch)
│   ├── models/models.py            # Network architectures (UNet, MLP)
│   ├── multimarginal_cfm.py        # Multi-marginal flow matching with splines
│   ├── multimarginal_otsampler.py  # Ordered multi-marginal optimal transport
│   ├── latent_flow/                # Latent-space flow matching
│   ├── latent_msbm/                # Latent-space MSBM
│   ├── geodesic_ae.py              # Geodesic autoencoder
│   ├── ode_diffeo_ae.py            # ODE diffeomorphism autoencoder
│   └── training/                   # Training utilities
│
├── functional_autoencoders/        # [SUBMODULE — do not modify]
│   └── src/functional_autoencoders/
│       ├── losses/                 # Loss functions (contains _call_autoencoder_fn)
│       ├── train/                  # Trainer, metrics base classes
│       ├── encoders/               # Encoder architectures
│       ├── decoders/               # Decoder architectures
│       ├── domains/                # Domain definitions
│       └── positional_encodings/   # Positional encoding utilities
│
├── MSBM/                           # [SUBMODULE — separate repo]
│
├── scripts/fae/                    # FAE experiment scripts (JAX/Flax)
│   ├── multiscale_dataset_naive.py # Multi-scale Gaussian-filtered dataset
│   ├── wandb_trainer.py            # W&B-integrated trainer
│   └── fae_naive/                  # Experiment implementations
│       ├── train_attention.py      # Main training entry point (unified phases A-C)
│       ├── train_attention_components.py  # Shared training utilities
│       ├── train_attention_denoiser.py    # Diffusion denoiser variant
│       ├── train_attention_drifting_denoiser.py  # Drifting denoiser variant
│       ├── spectral_metrics.py     # Frequency-binned error tracking
│       ├── spectral_losses.py      # H1, Fourier-weighted, high-pass losses
│       ├── decoder_builders.py     # Decoder factory (standard/rff/enhanced)
│       ├── fourier_enhanced_decoder.py   # RFF readout decoder
│       ├── single_scale_dataset.py # Single-scale dataset loader
│       └── [other decoder variants]
│
├── scripts/                        # Main MMSFM training scripts (PyTorch)
│   ├── main.py                     # Core training for synthetic/single-cell
│   └── images/                     # Image dataset training
│
└── tests/                          # Test suite (pytest)
```

## Data Flow (FAE Pipeline)

```
Dataset (multiscale_dataset_naive.py)
  → batch: {u_enc, x_enc, u_dec, x_dec}
    → Encoder (functional_autoencoders)
      → latents (z)
        → Decoder (standard | rff_output | enhanced_rff)
          → reconstruction (û)
            → Loss (MSE | H1 | Fourier-weighted)
              → optimizer step
```

## Dependency Rules

- `scripts/fae/` depends on `functional_autoencoders` (submodule)
- `scripts/fae/fae_naive/` depends on `scripts/fae/` (dataset, trainer)
- `mmsfm/` is independent of FAE code (separate PyTorch stack)
- `functional_autoencoders/` is read-only — never modify in this repo
