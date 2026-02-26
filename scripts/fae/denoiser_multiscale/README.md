# Multiscale Denoiser Experiments

This folder contains the **multiscale** (σ={1.0, 2.0, 4.0, 8.0}) versions of the three-way denoiser decoder ablation.

## Motivation

- **Single-scale experiments** ([../run_deterministic_film.sh](../run_deterministic_film.sh), [../run_denoiser_film_heek.sh](../run_denoiser_film_heek.sh), [../run_film_prior.sh](../run_film_prior.sh)) use σ=1.0 only to cleanly isolate architectural differences
- **Multiscale experiments** (this folder) use σ={1.0, 2.0, 4.0, 8.0} (99% spectral coverage) to validate that findings hold at realistic scale with 8× gradient imbalance

## Experiments

| Script | ID | Decoder | Prior | Beta | Purpose |
|--------|-----|---------|-------|------|---------|
| `run_1M_deterministic_film_multiscale.sh` | 1M | Deterministic FiLM | ✗ | 1e-4 | Baseline w/ L2 reg |
| `run_2M_denoiser_film_heek_multiscale.sh` | 2M | Denoising FiLM (Heek) | ✓ | 0 | Full Heek protocol |
| `run_3M_film_prior_multiscale.sh` | 3M | Deterministic FiLM | ✓ | 0 | Structured prior, no denoising |
| `run_3M_beta_film_prior_multiscale_beta.sh` | 3M-beta | Deterministic FiLM | ✓ | 1e-4 | Prior + L2 reg (ablation) |

## Key Comparisons

- **1M vs 2M:** Does denoising + prior beat L2 reg at multiscale?
- **2M vs 3M:** Does iterative denoising help given same prior + multiscale?
- **1M vs 3M:** Does structured prior beat L2 reg for deterministic + multiscale?
- **3M vs 3M-beta:** Is L2 reg helpful alongside structured prior?

## Running

Sequential launch (recommended):

```bash
# Launch from repo root
bash scripts/fae/denoiser_multiscale/run_1M_deterministic_film_multiscale.sh
# Wait ~2 hours, then:
bash scripts/fae/denoiser_multiscale/run_2M_denoiser_film_heek_multiscale.sh
# Wait ~2 hours, then:
bash scripts/fae/denoiser_multiscale/run_3M_film_prior_multiscale.sh
# (Optional) Wait ~2 hours, then:
bash scripts/fae/denoiser_multiscale/run_3M_beta_film_prior_multiscale_beta.sh
```

## Shared Configuration

All experiments use:
- Encoder: Dual-stream bottleneck, σ=1.0, MLP(3×256)
- Decoder: FiLM backbone, 3×256 features, **multiscale σ={1.0, 2.0, 4.0, 8.0}**
- Prior (2M/3M/3M-beta): 3-layer residual MLP, hidden_dim=256, logsnr_max=5.0
- Optimizer: Muon, lr=1e-3
- Training: 50k steps, masking strategy=random, encoder_point_ratio=0.8→0.1
- W&B project: `fae-denoiser-comparison`

## Expected Timeline

| Exp | Duration | Cumulative |
|-----|----------|------------|
| 1M  | ~2.0 hrs | +2:00 |
| 2M  | ~2.5 hrs | +4:30 |
| 3M  | ~2.0 hrs | +6:30 |
| 3M-beta | ~2.0 hrs | +8:30 |

**Total sequential:** ~8.5 hours
