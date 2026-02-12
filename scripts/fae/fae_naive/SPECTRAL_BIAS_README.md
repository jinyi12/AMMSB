# NTK-Guided Spectral Bias Mitigation for FAE

This directory contains a comprehensive implementation for combating spectral bias in Functional Autoencoders (FAE) when reconstructing multiscale microstructure data with sharp inclusion boundaries.

## Theoretical Foundation

### The Problem

Your multiscale Gaussian-filtered microstructure data exhibits two challenges:

1. **Spectral bias in neural networks** (Wang et al., NTK theory):
   - Training error in frequency mode *i* decays as *e^(-λᵢt)*
   - Networks have small eigenvalues λᵢ for high frequencies
   - High-frequency features converge extremely slowly

2. **Multi-scale training imbalance**:
   - Later time scales (large *t*) have exponentially suppressed high frequencies
   - Loss dominated by low-frequency content from smooth scales
   - Early scales (sharp boundaries) provide weak gradient signal

### The Solution

From FAE NTK decomposition:

```
Ker_FAE((u,x), (u',x')) = Ker_dec((z,x), (z',x')) + Ker_enc((u,x), (u',x'))
```

**Key insight**: High-frequency *spatial* reconstruction is governed by the decoder kernel **Ker_dec**.

Three-phase approach:

- **Phase A**: Identify the bottleneck (ablations, diagnostics)
- **Phase B**: Fix the loss (emphasize high frequencies)
- **Phase C**: Fix the architecture (reshape Ker_dec spectrum)

## Implementation Overview

### New Modules

```
scripts/fae/fae_naive/
├── train_attention.py              # Main training script (all phases, unified)
├── train_attention_spectral.py     # Deprecated compatibility wrapper
├── spectral_metrics.py             # Frequency-domain error tracking
├── spectral_losses.py              # Enhanced loss functions (H¹, Fourier-weighted, etc.)
├── fourier_enhanced_decoder.py     # RFF readout decoder (Phase C)
├── single_scale_dataset.py         # Single-scale dataset loader (Phase A)
└── decoder_builders.py             # Decoder builder utilities
```

Note: `train_attention_spectral.py` now forwards to `train_attention.py` and is kept only for backwards compatibility.

### Key Features

1. **Modular design**: Each phase can be toggled independently via CLI arguments
2. **Full WandB integration**: All metrics, visualizations, and spectral diagnostics logged
3. **Frequency-domain analysis**: Track error by frequency band to detect spectral bias
4. **NTK-guided architecture choices**: RFF placement follows IFeF-PINN theory

## Usage

### Phase A: Ablations (Identify the Bottleneck)

#### Experiment 1: Single-scale training (first filtered field)

**Hypothesis**: Multi-scale mixing causes blur by dominating loss with smooth scales.

```bash
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_a_single_scale \
    --training-mode single_scale \
    --single-scale-index 1 \
    --track-spectral-metrics \
    --spectral-n-bins 15 \
    --latent-dim 32 \
    --beta 1e-4 \
    --max-steps 50000 \
    --wandb-project fae-spectral \
    --wandb-name "phase_a_single_scale_t1"
```

**What to check**:
- Does blur disappear or persist?
- In WandB: `eval/spectral/high_freq_error_ratio` (should decrease if learning high freqs)
- Spectral metrics plot: is high-frequency error flatlining?

#### Experiment 2: Vary latent capacity and regularization

**Hypothesis**: β or d_z too constraining, encoder loses fine details.

```bash
# Low regularization, high capacity
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_a_capacity \
    --training-mode single_scale \
    --single-scale-index 1 \
    --latent-dim 64 \
    --beta 1e-5 \
    --wandb-name "phase_a_high_capacity"
```

**What to check**:
- Does increasing d_z / decreasing β improve sharp feature reconstruction?
- Track `final/test_rel_mse` and visual reconstructions

### Phase B: Enhanced Losses (Force High-Frequency Learning)

#### Experiment 3: H¹ Sobolev loss (gradient matching)

**Motivation**: Bunker et al. explicitly support Sobolev space losses. H¹ directly penalizes edge blur.

```bash
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_b_h1_loss \
    --training-mode multi_scale \
    --loss-type h1 \
    --lambda-grad 0.5 \
    --track-spectral-metrics \
    --wandb-name "phase_b_h1_lambda0.5"
```

**What to check**:
- Reconstruction of inclusion boundaries (sharp edges)
- `eval/spectral/high_freq_mse` (should decrease faster than baseline)

#### Experiment 4: Fourier-weighted loss

**Motivation**: Directly weight high frequencies more in the loss.

```bash
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_b_fourier_weighted \
    --training-mode multi_scale \
    --loss-type fourier_weighted \
    --freq-weight-power 1.0 \
    --wandb-name "phase_b_fourier_p1"
```

**Hyperparameter sweep**: Try `freq-weight-power` in [0.5, 1.0, 2.0]

#### Experiment 5: Combined loss (H¹ + Fourier weighting)

```bash
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_b_combined \
    --training-mode multi_scale \
    --loss-type combined \
    --lambda-grad 0.3 \
    --freq-weight-power 0.5 \
    --wandb-name "phase_b_combined"
```

### Phase C: Enhanced Architecture (Reshape Ker_dec)

#### Experiment 6: Enhanced RFF decoder (penultimate layer)

**Motivation**: IFeF-PINN shows RFF at penultimate layer "upgrades dot-product kernel → stationary kernel", injecting high-frequency basis functions where they most directly affect spatial reconstruction.

```bash
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_c_enhanced_rff \
    --training-mode multi_scale \
    --decoder-type rff_output \
    --rff-dim 512 \
    --track-spectral-metrics \
    --wandb-name "phase_c_rff512_penultimate"
```

**Hyperparameter sweep**:
- RFF dimension: [256, 512, 1024]
- Location: [penultimate, coordinates, both] (penultimate recommended per NTK theory)

#### Experiment 7: Multi-band RFF (multiscale frequency coverage)

**Motivation**: Your data are multiscale (Gaussian filtered at multiple σ_t). Multi-band RFF provides kernel support across multiple frequency scales.

```bash
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_c_multiband_rff \
    --training-mode multi_scale \
    --decoder-type rff_output \
    --rff-dim 256 \
    --rff-multiscale-sigmas "0.5,1.0,2.0,4.0" \
    --wandb-name "phase_c_multiband_4scales"
```

**What to check**:
- Reconstruction quality across ALL time scales (not just early or late)
- Compare `final/train_times_avg_mse` and `final/held_out_avg_mse`

### Phase C + Phase B: Combined Best Practices

**Recommendation**: After identifying best options from Phases A/B/C individually, combine them.

```bash
python scripts/fae/fae_naive/train_attention_spectral.py \
    --data-path /path/to/data.npz \
    --output-dir outputs/phase_combined_best \
    --training-mode multi_scale \
    --loss-type h1 \
    --lambda-grad 0.5 \
    --decoder-type rff_output \
    --rff-dim 512 \
    --rff-multiscale-sigmas "0.5,1.0,2.0,4.0" \
    --latent-dim 64 \
    --beta 5e-5 \
    --pooling-type augmented_residual \
    --n-heads 8 \
    --track-spectral-metrics \
    --save-best-model \
    --wandb-name "combined_best"
```

## Interpreting Results

### Frequency-Domain Metrics

WandB logs these key indicators:

1. **`eval/spectral/high_freq_error_ratio`**:
   - Ratio of high-freq error to total error
   - High ratio → spectral bias (high freqs not learned)
   - Should **decrease** over training if mitigation works

2. **`eval/spectral/low_freq_mse`** vs **`eval/spectral/high_freq_mse`**:
   - Compare convergence rates
   - If high_freq_mse flatlines early → spectral bias

3. **Spectral metrics plot** (`plots/spectral_metrics`):
   - Shows MSE vs frequency |k| over training
   - Look for: does error decrease uniformly across frequencies?
   - Or does high-|k| error flatline?

### Reconstruction Quality

1. **Visual inspection**: `reconstructions/*` in WandB
   - Are inclusion boundaries sharp or blurred?
   - Are small inclusions preserved or merged?

2. **Per-time metrics** (multi-scale mode):
   - `final/train_time_mse_t*`: Early (small t) should have low error
   - `final/held_out_mse_t*`: Generalization to unseen times

3. **Relative error**:
   - `final/test_rel_mse`: Normalized by signal energy
   - Better metric than absolute MSE for multiscale data

## Theory References

### Neural Tangent Kernel (NTK)

- **Jacot et al. (2018)**: NTK definition and infinite-width limit
- **Wang et al. (2021)**: "Understanding Spectral Bias..." - explicitly connects eigenvalue spectrum to frequency learning speed
- **Wang et al. (2022)**: IFeF-PINN - RFF at penultimate layer for spectral bias mitigation

### Functional Autoencoders

- **Bunker et al. (2023)**: FAE formulation, Sobolev space losses, function space compatibility
- **PointNet / DeepSets**: Permutation-invariant aggregation via pooling

### Multi-scale Theory

- **Gaussian filtering**: û_t(k) = exp(-σ²k²/2) û_0(k)
- **Multiscale kernels**: Concatenate RFF with multiple σ for broad frequency coverage

## Troubleshooting

### Spectral tracking fails

**Error**: `ValueError: Cannot interpret shape ... as resolution`

**Solution**: Ensure dataset provides regular grid with `resolution` field in .npz file.

### Multi-band RFF causes NaN

**Symptom**: Training loss becomes NaN with multi-band RFF.

**Solution**:
- Check sigma range (too large σ can cause numerical issues)
- Reduce learning rate
- Use gradient clipping: add `--grad-clip 1.0` (would need to add this arg)

### Single-scale training doesn't improve

**Interpretation**: Blur is intrinsic to architecture/loss, NOT multi-scale mixing.

**Action**: Move directly to Phase B/C (loss and architecture fixes).

### H¹ loss doesn't help

**Possible reasons**:
- λ_grad too small (try increasing)
- Finite difference gradient approximation is noisy (check resolution is high enough)
- Boundary conditions (periodic BC assumed in implementation)

## Next Steps

1. **Run Phase A ablations** to definitively identify bottleneck
2. **Track spectral metrics** - use them to guide which phase to focus on
3. **Iterate Phase B/C** based on frequency-domain diagnostics
4. **Combine best approaches** once individual components are validated
5. **Consider adaptive strategies**:
   - Curriculum learning (train on sharp scales first, then add smooth)
   - Dynamic loss weighting (increase λ_grad over training)
   - Scale-specific decoders (separate decoder per time range)

## Files Created

- `train_attention.py`: Unified training script (baseline + spectral options)
- `train_attention_spectral.py`: Backwards-compatible wrapper to `train_attention.py`
- `spectral_metrics.py`: Frequency analysis tools (250 lines)
- `spectral_losses.py`: Enhanced loss functions (350 lines)
- `fourier_enhanced_decoder.py`: RFF readout decoder (Phase C)
- `single_scale_dataset.py`: Single-scale data loader (230 lines)
- `decoder_builders.py`: Decoder builder utilities

Total: ~2000 lines of clean, modular, NTK-theory-guided code.

## WandB Project Organization

Recommended tagging scheme:

- Tags: `[training_mode, loss_type, decoder_type, pooling_type]`
- Example: `["single_scale", "h1", "rff_output", "augmented_residual"]`

This enables filtering WandB runs by:
- Phase A: filter by `single_scale` tag
- Phase B: filter by loss types (`h1`, `fourier_weighted`, etc.)
- Phase C: filter by decoder types (`rff_output`)

## Key Design Decisions

1. **Why penultimate-layer RFF over coordinate RFF?**
   - NTK theory: Ker_dec governs spatial frequency learning
   - IFeF: RFF on learned features "upgrades implicit kernel"
   - Direct reshaping of decoder eigen-spectrum

2. **Why H¹ over higher Sobolev spaces?**
   - H¹ captures first-order edge information (sufficient for piecewise smooth fields)
   - H² requires second derivatives (noisier to approximate, may not be needed)
   - Bunker et al. explicitly mention H^s as available option

3. **Why multi-band RFF?**
   - Your data are explicitly multiscale (σ_t varies)
   - Single σ may not cover full frequency range
   - Multiple bands ≈ multiscale kernel (rich frequency coverage)

4. **Why track spectral metrics?**
   - Direct diagnostic for spectral bias (Wang et al.)
   - Informs which frequencies are learning slowly
   - Validates that mitigation strategies work as intended

## Contact / Questions

For questions about this implementation or NTK theory:
1. Check WandB runs for similar configurations
2. Review spectral metrics plots to diagnose issues
3. Consult papers: Wang et al. 2021 (spectral bias), Bunker et al. 2023 (FAE), Wang et al. 2022 (IFeF-PINN)
