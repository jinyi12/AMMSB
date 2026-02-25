#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 1 — Deterministic FiLM decoder, L2 regularisation
# ===================================================================
# Fair-comparison baseline for the three-way decoder ablation:
#
#   Exp 1 (this)  FiLM backbone, no prior, MSE + L2 reg
#   Exp 2         FiLM diffusion, prior, sigmoid-weighted velocity
#   Exp 3         FiLM backbone, prior, MSE (deterministic)
#
# Isolates: backbone architecture improvement over NonlinearDecoder.
# Encoder/optimizer/masking identical across all three experiments.
# Uses plain MSE (--loss-type l2) to match Exp 3's base loss.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_deterministic_film \
  --pooling-type dual_stream_bottleneck \
  --decoder-type film \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0 \
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
  --vis-interval 5 > fae_deterministic_film.log 2>&1 &
