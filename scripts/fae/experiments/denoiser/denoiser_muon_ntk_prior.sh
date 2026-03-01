#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Denoiser D3 — Muon + NTK + Prior (full stack, iterative decoder)
# ===================================================================
# Full-stack gradient correction with iterative denoiser decoder.
# Tests whether NTK reweighting can fix the denoiser's poor late-time
# spectral fidelity by rebalancing per-time learning signals.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

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
  --ntk-estimate-total-trace \
  --ntk-total-trace-ema-decay 0.99 \
  --ntk-epsilon 1e-8 \
  --ntk-calibration-interval 100 \
  --ntk-cv-threshold 0.2 \
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
