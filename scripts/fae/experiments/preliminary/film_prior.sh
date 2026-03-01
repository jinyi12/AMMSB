#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 3 — Deterministic FiLM decoder + structured prior
# ===================================================================
# Fair-comparison ablation for the three-way decoder ablation:
#
#   Exp 1         FiLM backbone, no prior, MSE + L2 reg
#   Exp 2         FiLM diffusion, prior, sigmoid-weighted velocity
#   Exp 3 (this)  FiLM backbone, prior, MSE (deterministic)
#
# Isolates two questions when compared with the others:
#   vs Exp 1: Does a structured latent prior help the deterministic decoder?
#   vs Exp 2: Is iterative denoising beneficial given the same prior?
#
# Architecture: DeterministicFiLMDecoder — identical FiLM + LayerNorm +
# residual backbone as StandardDenoiserDecoder, but no noisy_field input,
# no time embedding, and single-shot reconstruction (no ODE/SDE sampling).
#
# Training: same prior (logsnr_max=5, hidden_dim=256, 3-layer residual MLP)
# as Exp 2. Decoder loss is plain MSE on clean field.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_prior \
  --pooling-type dual_stream_bottleneck \
  --decoder-type film \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
  --masking-strategy random \
  --eval-masking-strategy same \
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
  --beta 0.0 \
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
  --vis-interval 5 > fae_film_prior.log 2>&1 &
