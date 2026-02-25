#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Denoiser decoder — Scaled backbone (ScaledDenoiserDecoder), velocity loss
# ===================================================================
# Fair comparison to run_standard_dual_stream_bottleneck.sh:
#   - SAME encoder: dual_stream_bottleneck, latent_dim=256, n_freqs=64,
#     encoder_mlp_layers=3, encoder_mlp_dim=256, RFF sigma=1.0
#   - SAME optimizer: muon, lr=1e-3, max_steps=50000
#   - SAME masking: random, same point ratios
#   - SAME beta: 1e-4
#
# Decoder differences (by design):
#   - ScaledDenoiserDecoder with scaling=1.0 (2.2M params).
#     This is the wider backbone used in the original implementation.
#     Higher capacity may capture complex noise→clean mappings but
#     costs ~4x more FLOPs per forward pass vs the FiLM variant.
#   - 100 diffusion steps, 32-step eval (same as FiLM run).
#   - Velocity loss.
#
# Compute notes:
#   - Training per step: ~2-3x baseline (larger decoder)
#   - Eval: same 32-step ODE as FiLM run, but each step is ~4x heavier
#   - --eval-interval=50 to keep total eval budget similar
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_denoiser_scaled_velocity \
  --pooling-type dual_stream_bottleneck \
  --decoder-type denoiser \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --denoiser-scaling 1.0 \
  --denoiser-diffusion-steps 100 \
  --denoiser-eval-sample-steps 32 \
  --denoiser-beta-schedule cosine \
  --denoiser-time-sampling logit_normal \
  --denoiser-velocity-loss-weight 1.0 \
  --denoiser-x0-loss-weight 0.0 \
  --masking-strategy random \
  --eval-masking-strategy same \
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
  --beta 1e-4 \
  --save-best-model \
  --wandb-project fae-denoiser-comparison \
  --optimizer muon \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 50 \
  --eval-n-batches 5 \
  --vis-interval 10 > fae_denoiser_scaled_velocity.log 2>&1 &
