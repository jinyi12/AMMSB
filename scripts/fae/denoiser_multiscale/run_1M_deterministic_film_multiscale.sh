#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 1M — Deterministic FiLM decoder, L2 reg, Multiscale
# ===================================================================
# Multiscale version of Experiment 1 (single-scale baseline).
#
# Fair-comparison baseline for the three-way multiscale decoder ablation:
#
#   Exp 1M (this)  FiLM backbone, no prior, MSE + L2 reg, multiscale
#   Exp 2M         FiLM diffusion, prior, sigmoid-weighted velocity, multiscale
#   Exp 3M         FiLM backbone, prior, MSE (deterministic), multiscale
#   Exp 3M-beta    FiLM backbone, prior, MSE + L2 reg, multiscale
#
# Multiscale configuration: σ = {1.0, 2.0, 4.0, 8.0} (99% spectral coverage)
# Validates that architectural benefits from single-scale experiments hold
# at realistic multiscale (8× max gradient ratio).
#
# Isolates: Baseline FiLM performance with L2 regularization at multiscale.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_deterministic_film_multiscale \
  --pooling-type dual_stream_bottleneck \
  --decoder-type film \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0,2.0,4.0,8.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
  --loss-type l2 \
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
  --vis-interval 5 > fae_deterministic_film_multiscale.log 2>&1 &
