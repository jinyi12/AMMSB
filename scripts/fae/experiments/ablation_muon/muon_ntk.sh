#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Pareto M2 — Muon + NTK (spectral optimizer + loss correction)
# ===================================================================
# Combines Muon's SVD-based gradient preconditioning (Type I: direction)
# with NTK trace-equalization (Type II: per-time convergence rate).
#
# Tests whether the two corrections are complementary or redundant.
# Data shows Muon+NTK t7 PSD = 7.63 vs Adam+NTK = 7.44, suggesting
# partial redundancy at the gradient direction level.
#
# Results: A run exists at results/fae_film_muon_ntk_99pct/run_tug7ucuw
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_muon_ntk_99pct \
  --pooling-type dual_stream_bottleneck \
  --decoder-type film \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0,2.0,4.0,8.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
  --loss-type ntk_scaled \
  --ntk-estimate-total-trace \
  --ntk-total-trace-ema-decay 0.99 \
  --ntk-epsilon 1e-8 \
  --ntk-calibration-interval 100 \
  --ntk-cv-threshold 0.2 \
  --masking-strategy random \
  --eval-masking-strategy same \
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
  --beta 1e-4 \
  --save-best-model \
  --wandb-project fae-film-optimizer-loss-ablation \
  --optimizer muon \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_muon_ntk.log 2>&1 &
