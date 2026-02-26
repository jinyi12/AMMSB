#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Muon + NTK + Prior — Deterministic FiLM decoder (full stack)
# ===================================================================
# Combines all three complementary gradient/regularisation mechanisms:
#
#   Type I  — Muon spectral gradient descent (balanced gradient directions)
#   Type II — NTK trace-equalisation (balanced convergence across times)
#   Type III — Latent diffusion prior (structured latent for downstream MSBM)
#
# This is the full-stack ablation: deterministic FiLM backbone with prior
# regularisation and NTK-scaled reconstruction loss.
#
# Decoder: DeterministicFiLMDecoder (single-shot, no diffusion).
# Prior:   3-layer residual MLP on 256-d latent (Heek et al. 2026 UL §3.1).
# Loss:    NTK-scaled MSE + unweighted prior ELBO.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../.."

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
  --ntk-scale-norm 10.0 \
  --ntk-epsilon 1e-8 \
  --ntk-n-loss-terms 5 \
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
  --vis-interval 5 > fae_film_muon_ntk_prior.log 2>&1 &
