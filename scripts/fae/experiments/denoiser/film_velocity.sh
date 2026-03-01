#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Denoiser decoder — FiLM (StandardDenoiserDecoder), velocity loss
# ===================================================================
# Fair comparison to run_standard_dual_stream_bottleneck.sh:
#   - SAME encoder: dual_stream_bottleneck, latent_dim=256, n_freqs=64,
#     encoder_mlp_layers=3, encoder_mlp_dim=256, RFF sigma=1.0
#   - SAME optimizer: muon, lr=1e-3, max_steps=50000
#   - SAME masking: random, same point ratios
#   - SAME beta: 1e-4
#
# Decoder differences (by design):
#   - StandardDenoiserDecoder with FiLM conditioning (603K params vs 198K)
#     Larger decoder is justified: the denoiser must learn the full
#     noise→clean mapping, not just z→u pointwise.
#   - 100 diffusion steps (cosine schedule) — good reconstruction at
#     reasonable eval cost.  32-step eval keeps validation fast.
#   - Velocity loss (rectified flow v-prediction) — default objective.
#
# Compute notes:
#   - Training cost per step: ~1.2x baseline (noise/time sampling overhead)
#   - Eval cost: ~32x per sample (multi-step ODE), but --eval-interval=25
#     and --eval-n-batches=5 keep wall-clock reasonable.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_denoiser_film_velocity \
  --pooling-type dual_stream_bottleneck \
  --decoder-type denoiser_standard \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
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
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_denoiser_film_velocity.log 2>&1 &
