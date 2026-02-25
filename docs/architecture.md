# Architecture

## Module Map

```
MMSFM/
├── mmsfm/                          # Core Python package (PyTorch)
│   ├── models/models.py            # Network architectures (TimeFiLMMLP, ResNet, etc.)
│   ├── multimarginal_cfm.py        # Multi-marginal flow matching with splines
│   ├── multimarginal_otsampler.py  # Ordered multi-marginal optimal transport
│   ├── multimarginal_pairedsampler.py  # Paired marginal sampler
│   ├── flow_objectives.py          # Flow matching objectives (velocity prediction)
│   ├── flow_ode_trainer.py         # Neural ODE-based flow training
│   ├── psi_provider.py             # Precomputed dense diffusion-map embedding sampler
│   ├── data_utils.py               # PCA utilities, marginal splitting
│   ├── noise_schedules.py          # Exponential/mini-flow schedules for latent flow
│   ├── wandb_compat.py             # Optional W&B wrapper
│   ├── latent_msbm/                # Latent-space Multi-marginal Schrödinger Bridge Matching
│   │   ├── agent.py                # Main MSBM training agent
│   │   ├── coupling.py             # Hybrid coupling sampler
│   │   ├── policy.py               # MSBM policies (AugmentedMLP, FiLM)
│   │   ├── sde.py                  # Bridge SDE definitions
│   │   ├── noise_schedule.py       # Diffusion schedules
│   │   └── utils.py                # Freeze/activate policy, EMA scope
│   └── training/
│       └── ema.py                  # Exponential Moving Average wrapper
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
│       ├── train_attention_flow.py # Shared train/eval flow used by entry scripts
│       ├── train_attention_components.py  # Shared training utilities
│       ├── train_attention_denoiser.py    # Diffusion denoiser variant
│       ├── train_attention_drifting_denoiser.py  # Drifting denoiser variant
│       ├── train_latent_msbm.py    # Latent MSBM training entry point
│       ├── decoder_builders.py     # Decoder factory (standard/wire2d/denoiser)
│       ├── sobolev_losses.py       # H¹ reconstruction loss
│       ├── ntk_losses.py           # NTK-scaled reconstruction loss
│       ├── single_scale_dataset.py # Single-scale dataset loader
│       └── [other decoder/loss variants]
│
├── scripts/fae/tran_evaluation/    # Tran-aligned evaluation pipeline
│   ├── evaluate.py                 # Main orchestrator (7-phase evaluation)
│   ├── generate.py                 # Backward SDE generation
│   ├── core.py                     # FilterLadder, data loaders
│   ├── conditioning.py             # Macroscale constraint verification
│   ├── first_order.py              # One-point statistics, W₁ distance
│   ├── second_order.py             # Correlation analysis
│   ├── spectral.py                 # Power spectral density diagnostics
│   ├── diversity.py                # Mode-collapse detection
│   └── report.py                   # Visualization & reporting
│
├── notebooks/                      # Active analysis notebooks
│   ├── fae_latent_msbm_latent_viz.py
│   ├── visualize_field_trajectories.py
│   ├── visualize_full_trajectories.py
│   └── tran_inclusions_dataset_viz.py
│
├── tests/                          # Test suite (pytest)
│
├── docs/                           # Documentation
│   ├── architecture.md             # This file
│   ├── conventions.md              # FAE framework patterns
│   ├── experiments/index.md        # Experiment catalog
│   ├── plans/                      # Execution plans
│   └── references/                 # Paper summaries
│
├── archive/                        # Dated archives of legacy code
│
└── spacetime-geometry/             # Spacetime geometry experiments (separate repo)
```

## Data Flow (FAE Pipeline)

```
Dataset (multiscale_dataset_naive.py)
  → batch: {u_enc, x_enc, u_dec, x_dec}
    → Encoder (functional_autoencoders)
      → latents (z)
        → Decoder (standard | wire2d | denoiser | denoiser_standard)
          → reconstruction (û)
            → Loss (MSE | H¹ | NTK-scaled)
              → optimizer step
```

## Dependency Rules

- `scripts/fae/` depends on `functional_autoencoders` (submodule)
- `scripts/fae/fae_naive/` depends on `scripts/fae/` (dataset, trainer)
- `mmsfm/` is independent of FAE code (separate PyTorch stack)
- `functional_autoencoders/` is read-only — never modify in this repo
