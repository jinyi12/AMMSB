#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Denoiser + Muon + NTK + Prior — Full stack with iterative decoder
# ===================================================================
# Combines all three mechanisms with Heek et al. denoiser decoder:
#
#   Type I  — Muon spectral gradient descent (balanced gradient directions)
#   Type II — NTK trace-equalisation on decoder velocity loss
#   Type III — Latent diffusion prior (Heek et al. 2026 UL §3.1-3.2)
#
# Tests whether NTK scaling can fix the denoiser's poor spectral fidelity
# at late times (t7) by rebalancing the per-time NTK eigenvalue gap.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_denoiser_muon_ntk_prior \
  --pooling-type dual_stream_bottleneck \
  --decoder-type denoiser_standard \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0,2.0,4.0,8.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
  --denoiser-diffusion-steps 100 \
  --denoiser-eval-sample-steps 32 \
  --denoiser-beta-schedule cosine \
  --denoiser-time-sampling logsnr_uniform \
  --denoiser-logsnr-max 5.0 \
  --denoiser-velocity-loss-weight 1.0 \
  --denoiser-x0-loss-weight 0.0 \
  --masking-strategy random \
  --eval-masking-strategy same \
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
  --beta 0.0 \
  --loss-type ntk_scaled \
  --ntk-scale-norm 10.0 \
  --ntk-epsilon 1e-8 \
  --ntk-n-loss-terms 5 \
  --use-prior \
  --prior-hidden-dim 256 \
  --prior-n-layers 3 \
  --prior-time-emb-dim 32 \
  --prior-logsnr-max 5.0 \
  --prior-loss-weight 1.0 \
  --decoder-loss-factor 1.3 \
  --save-best-model \
  --wandb-project fae-denoiser-comparison \
  --optimizer muon \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_denoiser_muon_ntk_prior.log 2>&1 &
