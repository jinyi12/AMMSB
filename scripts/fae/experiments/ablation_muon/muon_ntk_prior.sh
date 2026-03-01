#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Pareto M4 — Muon + NTK + Prior (full stack, spectral optimizer)
# ===================================================================
# All three mechanisms with Muon optimizer. Full Pareto comparator
# against Adam+NTK+Prior (proposed method).
#
# If Muon's spectral preconditioning and NTK's per-time trace-
# equalization are redundant, this should perform similarly to
# Adam+NTK+Prior. If they're truly complementary, this should be
# the Pareto-optimal point (best reconstruction AND latent quality).
#
# Decoder:  DeterministicFiLMDecoder (single-shot, no diffusion).
# Loss:     NTK-scaled MSE + unweighted prior ELBO.
# Prior:    3-layer residual MLP on 256-d latent.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_muon_ntk_prior \
  --pooling-type dual_stream_bottleneck \
  --decoder-type film \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0,2.0,4.0,8.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
  --masking-strategy random \
  --eval-masking-strategy same \
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
  --beta 0.0 \
  --loss-type ntk_scaled \
  --ntk-epsilon 1e-8 \
  --ntk-estimate-total-trace \
  --ntk-total-trace-ema-decay 0.0 \
  --ntk-calibration-interval 100 \
  --ntk-cv-threshold 0.2 \
  --ntk-calibration-pilot-samples 8 \
  --use-prior \
  --prior-hidden-dim 256 \
  --prior-n-layers 3 \
  --prior-time-emb-dim 32 \
  --prior-logsnr-max 5.0 \
  --prior-loss-weight 1.0 \
  --save-best-model \
  --wandb-project fae-film-optimizer-loss-ablation \
  --optimizer muon \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_muon_ntk_prior.log 2>&1 &
