# FAE Framework Conventions

## Loss Function Signature

The FAE framework expects loss functions with this exact signature:

```python
def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec) -> (loss, batch_stats)
```

- `key` is the 2nd parameter (not 4th)
- Batch fields are unpacked into keyword arguments
- Returns a tuple `(loss_value, updated_batch_stats)`

## Encoder/Decoder Calls

Always use `_call_autoencoder_fn`:

```python
from functional_autoencoders.losses import _call_autoencoder_fn

latents, encoder_updates = _call_autoencoder_fn(
    params=params, batch_stats=batch_stats, fn=autoencoder.encoder.apply,
    u=u_enc, x=x_enc, name="encoder", dropout_key=dropout_key)
```

**Never** use `autoencoder.encode_apply()` — it does not exist.

## Latent Regularization

The correct form sums over latent dim, then averages over batch:

```python
reg = jnp.mean(beta * jnp.sum(latents**2, axis=-1))
```

Using `beta * jnp.mean(latents**2)` is `latent_dim` times weaker. Always match the original.

## Metric Interface

- Base class: `functional_autoencoders.train.metrics.Metric`
- Must inherit (not just implement the interface)
- `__call__(self, state, key, test_dataloader)` iterates batches, calls `call_batched`
- Default accumulation: `metric_value += self.call_batched(...)` — requires float returns
- For dict-valued metrics: override `__call__` to handle dict accumulation
- `_print_metrics` in base trainer formats with `:.3E` — override for dict metrics

## RFF Pipeline

All Random Fourier Features follow this standardized procedure:

- **Sampling**: `B ~ N(0, σ²I)` — no `1/√feature_dim` normalization
- **Transform**: `γ(v) = [cos(2π B v), sin(2π B v)]` — no output scaling
- Adjust σ via `--rff-sigma` (smaller σ for larger feature_dim)

## Point Splitting

Single-scale dataset uses `encoder_point_ratio` to split points between encoder and decoder. FFT-based spectral metrics and gradient-based losses (H1, Fourier-weighted) require full grid:

```python
assert n_points == resolution**2, "FFT reshape requires full grid"
```

Always check before attempting FFT reshape.

## Time-Scale Mapping (Latent MSBM)

Time-scale consistency is critical in latent MSBM evaluation. Maintain explicit `t_dists ↔ zt` mapping as implemented in `archive/2026-02-16_non_fae_scripts/scripts/latent_msbm_eval.py`. The `t_dists` represents physical time points while `zt` represents indices — they must remain synchronized across forward/backward rollouts.

## Data Utilities

- `data/multimarginal_generation.py` — Multi-marginal OT data generation
- `data/transform_utils.py` — Data transformation utilities (used by `scripts/fae/tran_evaluation/core.py`)
- `data/datagen.py` — Main dataset preprocessing script
