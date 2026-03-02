#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Ablation A2 — Adam + NTK (per-time loss reweighting only)
# ===================================================================
# Adds NTK trace-equalization to Adam baseline.
# Fixes Type II gradient pathology (convergence rate imbalance across
# times) by scaling per-time loss by C / Tr(K_i).
#
# Expected vs A1: late-time PSD improves (t7: ~8.4 → ~7.4), early-time
# PSD slightly degrades (fidelity redistribution, not net gain on global).
#
# Decoder:  DeterministicFiLMDecoder (single-shot).
# Loss:     NTK trace-equalized MSE.
# Prior:    None.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_adam_ntk_99pct \
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
  --ntk-epsilon 1e-8 \
  --ntk-estimate-total-trace \
  --ntk-total-trace-ema-decay 0.0 \
  --ntk-trace-update-interval 100 \
  --ntk-hutchinson-probes 1 \
  --masking-strategy random \
  --eval-masking-strategy same \
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
  --beta 1e-4 \
  --save-best-model \
  --wandb-project fae-film-optimizer-loss-ablation \
  --optimizer adam \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_adam_ntk.log 2>&1 &
