# Plan: HierarchicalFourierEncoder — CNO + FANO Hybrid Encoder for FAE

## Context

The project has standalone PyTorch implementations of CNO (`CNO2d/CNO2d.py`) and FANO (`TNO/transformer_custom.py`, `TNO/FANO_pytorch.py`) neural operators with zero FAE integration. The user proposed a hybrid encoder that chains CNO (local hierarchical downsampling) → FANO (global Fourier attention) → mean pool → latent vector. This must be ported to **JAX/Flax** and integrated as a proper `functional_autoencoders.encoders.Encoder` subclass.

The `fae_naive/` folder is bloated (~18 files). New reusable encoder modules go in `mmsfm/encoders/` to keep experiment scripts separate from architecture code.

**Training-script constraint (revision)**: Do **not** add an `--encoder-type` switch to `scripts/fae/fae_naive/train_attention.py`. Instead, add a *separate* training entrypoint for the new encoder that preserves the same end-to-end training flow (dataset → build → trainer.fit → eval → viz → artifact saving) while keeping `train_attention.py` focused on the existing pooling-encoder baselines.

## Architecture Summary

```
u: [B, n_points, d_out] + x: [n_points, 2]
  → reshape to [B, 64, 64, 1]
  → Stage 1: Lift + (ResNet + CNOBlock↓) × N_layers + Bottleneck ResNet
     → [B, 8, 8, 64]  (example: 3 downsamplings, channel_mult=16)
  → Stage 2: FourierAttentionBlock × K layers (spectral conv mixing, no patch attention)
     → [B, 8, 8, 64]
  → Stage 3: spatial mean pool → MLP → Dense(latent_dim)
     → [B, latent_dim]
```

**Key simplification**: With `num_patches=1` (whole compressed domain), FANO's inter-patch attention degenerates. We use only the spectral convolution layers (which already provide non-local Fourier-domain mixing) as the global mixing stage, dropping the trivial softmax attention.

## New Files

### 1. `mmsfm/encoders/__init__.py`
- Package init, exports `HierarchicalFourierEncoder`

### 2. `mmsfm/encoders/cno_blocks.py`
Port of CNO2d building blocks to JAX/Flax (NHWC layout throughout):

| Class | Port of | Notes |
|-------|---------|-------|
| `CNOActivation` | `CNO_LReLu` | `jax.image.resize` bicubic up 2x → `nn.leaky_relu` → resize down |
| `CNOBlock` | `CNOBlock` | Conv3x3 → LayerNorm → CNOActivation (default LayerNorm, avoids batch_stats) |
| `LiftProjectBlock` | `LiftProjectBlock` | CNOBlock(no norm) → Conv3x3 |
| `ResidualBlock` | `ResidualBlock` | Conv → Norm → CNOActivation → Conv → Norm + skip |
| `ResNet` | `ResNet` | Stacks N ResidualBlock instances |

### 3. `mmsfm/encoders/spectral_attention.py`
Port of FANO spectral convolution and attention to JAX/Flax:

| Class | Port of | Notes |
|-------|---------|-------|
| `SpectralConv2d` | `SpectralConv2d_Attention` | Complex weights as `real + 1j*imag` params (follows `FNOLayer` pattern). `jnp.fft.rfft2` / `irfft2`. |
| `FourierMixingLayer` | Simplified `TransformerEncoderLayer_Conv` | SpectralConv (global mixing) → residual → LayerNorm → FFN → residual → LayerNorm. No degenerate patch attention. |
| `FourierMixingBlock` | `TransformerEncoder_Operator` | Stacks N `FourierMixingLayer` instances |

### 4. `mmsfm/encoders/hierarchical_fourier_encoder.py`
Main encoder class composing all three stages:

```python
class HierarchicalFourierEncoder(Encoder):
    latent_dim: int = 64
    input_size: int = 64
    d_out: int = 1
    cno_n_layers: int = 3
    cno_channel_multiplier: int = 16
    cno_n_res_blocks: int = 4
    cno_n_res_blocks_neck: int = 4
    fano_n_layers: int = 2
    fano_nhead: int = 4
    fano_modes: tuple | None = None  # auto: compressed_size//2 + 1
    fano_dim_feedforward: int = 128
    mlp_features: tuple = (256, 128)
    is_variational: bool = False
```

Encoder-only CNO (no decoder path, no skip connections) — analogous to using a ResNet backbone for classification.

### 5. `tests/test_hierarchical_fourier_encoder.py`
- Shape test: output is `[B, latent_dim]`
- Gradient test: `jax.grad` through all stages produces no NaN
- Integration test: wraps in `Autoencoder(encoder=..., decoder=...)`

### 6. `scripts/fae/fae_naive/train_attention_flow.py`
- Extract the *training flow* from `train_attention.py` into a reusable library function (same behavior, less duplication).
- The flow stays “train_attention-shaped” (same steps, logging, evaluation hooks), but becomes callable as:
  - `run_training(args, *, build_autoencoder_fn, architecture_name, tags=...)`
- Keeps scripting simple: new training scripts should only define CLI args and a `build_autoencoder_fn`.

### 7. `scripts/fae/fae_naive/train_hierarchical_fourier.py`
- New dedicated training script for `HierarchicalFourierEncoder`.
- Retains the flow of `train_attention.py` by calling `train_attention_flow.run_training(...)`.
- Simplified CLI:
  - Drop pooling-specific args (`--pooling-type`, `--n-heads`, `--n-queries`, etc.).
  - Drop encoder RFF args if the new encoder does not use them.
  - Keep dataset/training/loss/wandb args consistent with `train_attention.py` to minimize cognitive overhead.

## Modified Files

### 8. `scripts/fae/fae_naive/train_attention_components.py`
- Add a small, explicit constructor helper for the new encoder *without* branching inside the baseline builder:
  - `build_hierarchical_fourier_autoencoder(key, *, latent_dim, decoder_type, decoder_features, ...)`
- Keep the existing `build_autoencoder(...)` (PoolingEncoder + pooling_fn) unchanged to avoid CLI branching and keep baseline behavior stable.

### 9. `scripts/fae/fae_naive/train_attention.py`
- Keep as the baseline attention-pooling trainer (no `--encoder-type`).
- Refactor to call `train_attention_flow.run_training(...)` to reduce script size and keep the flow consistent across training entrypoints.

### 10. `docs/architecture.md`
- Add `mmsfm/encoders/` to module map

## Porting Notes

| PyTorch | JAX/Flax |
|---------|----------|
| `nn.Conv2d(in, out, 3, padding=1)` NCHW | `nn.Conv(out, (3,3), padding='SAME')` NHWC |
| `nn.BatchNorm2d` | `nn.LayerNorm()` (default — avoids batch_stats complexity) |
| `F.interpolate(bicubic, antialias)` | `jax.image.resize(method='bicubic')` |
| `torch.fft.rfft2/irfft2` | `jnp.fft.rfft2/irfft2` |
| `nn.Parameter(cfloat)` | Two `self.param()` for real/imag, combine as `r + 1j*i` |
| `torch.einsum` | `jnp.einsum` |

## Dimension Flow (64×64, 3 CNO layers, channel_mult=16)

| Stage | Shape | Channels |
|-------|-------|----------|
| Input reshaped | `[B, 64, 64, 1]` | 1 |
| Lift | `[B, 64, 64, 8]` | 8 |
| ResNet+Down 1 | `[B, 32, 32, 16]` | 16 |
| ResNet+Down 2 | `[B, 16, 16, 32]` | 32 |
| ResNet+Down 3 | `[B, 8, 8, 64]` | 64 |
| Bottleneck | `[B, 8, 8, 64]` | 64 |
| FANO layers | `[B, 8, 8, 64]` | 64 |
| Mean pool | `[B, 64]` | — |
| MLP + Dense | `[B, latent_dim]` | — |

## Verification

1. `python -c "from mmsfm.encoders import HierarchicalFourierEncoder"` — import succeeds
2. `pytest tests/test_hierarchical_fourier_encoder.py` — shape + gradient tests pass
3. `python scripts/fae/fae_naive/train_hierarchical_fourier.py --max-steps 10 ...` — sanity run completes
4. `bash scripts/check.sh` — ruff + imports + pytest all pass

## Implementation Order

1. `mmsfm/encoders/__init__.py`
2. `mmsfm/encoders/cno_blocks.py`
3. `mmsfm/encoders/spectral_attention.py`
4. `mmsfm/encoders/hierarchical_fourier_encoder.py`
5. `tests/test_hierarchical_fourier_encoder.py`
6. Add `scripts/fae/fae_naive/train_attention_flow.py`
7. Add `build_hierarchical_fourier_autoencoder(...)` to `train_attention_components.py`
8. Add `scripts/fae/fae_naive/train_hierarchical_fourier.py`
9. Refactor `scripts/fae/fae_naive/train_attention.py` to call the shared flow (no behavioral changes)
10. Update `docs/architecture.md`
