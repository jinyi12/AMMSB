# MMSFM — Agent Instructions

Multi-Marginal Stochastic Flow Matching for high-dimensional snapshot data. This repo contains the ICML 2025 paper implementation plus experimental extensions (FAE-based latent flow, spectral bias mitigation, diffusion denoisers).

## Architecture

| Module | Role | Notes |
|--------|------|-------|
| `mmsfm/` | Core package | Flow matching, OT sampler, models, latent flow/MSBM |
| `functional_autoencoders/` | FAE framework (submodule) | **Do not modify directly** — treat as external dependency |
| `MSBM/` | Schrödinger Bridge Matching (submodule) | Separate git repo |
| `scripts/fae/` | FAE training scripts | Dataset loaders, W&B trainer |
| `scripts/fae/fae_naive/` | Experiment implementations | Decoders, losses, metrics, training entry points |
| `tests/` | Test suite | Run with `pytest tests/` |

See [docs/architecture.md](docs/architecture.md) for detailed module map and data flow.

## Key Conventions

See [docs/conventions.md](docs/conventions.md) for full details. Critical rules:

1. **Loss function signature**: `def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec) -> (loss, batch_stats)`
2. **Encoder/decoder calls**: Use `_call_autoencoder_fn` from `functional_autoencoders.losses` — never `autoencoder.encode_apply()`
3. **Latent regularization**: `jnp.mean(beta * jnp.sum(latents**2, axis=-1))` — sum over latent dim, average over batch
4. **RFF pipeline**: `B ~ N(0, σ²I)`, transform `γ(v) = [cos(2πBv), sin(2πBv)]`, no scaling
5. **Metrics**: Must inherit `functional_autoencoders.train.metrics.Metric`; override `__call__` for dict-valued metrics

## Environment

```bash
conda activate mmsfmvenv   # or ./mmsfmvenv
# FAE scripts use JAX — set backend:
JAX_PLATFORM_NAME=cpu python ...   # for CPU
```

## Running

```bash
# Main MMSFM training (PyTorch)
./runner.sh

# FAE experiments (JAX)
python scripts/fae/fae_naive/train_attention.py --help

# Validation
bash scripts/check.sh

# Tests
pytest tests/
```

## Docs Map

- [docs/architecture.md](docs/architecture.md) — Module map and dependency directions
- [docs/conventions.md](docs/conventions.md) — FAE framework patterns and pitfalls
- [docs/experiments/index.md](docs/experiments/index.md) — Experiment catalog
- [docs/plans/](docs/plans/) — Active and completed execution plans

## Common Pitfalls

- FFT-based losses require full grid (`n_points == resolution²`). Check before reshape.
- Point splitting (`encoder_point_ratio`) means encoder/decoder get different point counts.
- `_print_metrics` formats with `:.3E` — override for dict-valued metrics.
- `beta * jnp.mean(latents**2)` is `latent_dim` times weaker than the correct form.
