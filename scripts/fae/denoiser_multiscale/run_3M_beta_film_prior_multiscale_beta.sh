#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 3M-beta — Deterministic FiLM + prior + L2 reg, Multiscale
# ===================================================================
# Beta ablation variant of Experiment 3M.
#
# Fair-comparison ablation for the three-way multiscale decoder ablation:
#
#   Exp 1M         FiLM backbone, no prior, MSE + L2 reg, multiscale
#   Exp 2M         FiLM diffusion, prior, sigmoid-weighted velocity, multiscale
#   Exp 3M         FiLM backbone, prior, MSE (deterministic), multiscale
#   Exp 3M-beta (this)  FiLM backbone, prior, MSE + L2 reg, multiscale
#
# Multiscale configuration: σ = {1.0, 2.0, 4.0, 8.0} (99% spectral coverage)
#
# Isolates: Does L2 regularization help deterministic decoder when combined
# with structured prior? Tests whether dual regularization (prior + L2) is
# beneficial or redundant.
#
# Key difference from Exp 3M:
#   --beta 1e-4  (adds L2 latent regularization alongside structured prior)
#
# Expected outcome: L2 reg likely redundant given structured prior already
# constrains latent space. This experiment validates that hypothesis.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_prior_multiscale_beta \
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
  --beta 1e-4 \
  --use-prior \
  --prior-hidden-dim 256 \
  --prior-n-layers 3 \
  --prior-time-emb-dim 32 \
  --prior-logsnr-max 5.0 \
  --prior-loss-weight 1.0 \
  --save-best-model \
  --wandb-project fae-denoiser-comparison \
  --optimizer muon \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_film_prior_multiscale_beta.log 2>&1 &
