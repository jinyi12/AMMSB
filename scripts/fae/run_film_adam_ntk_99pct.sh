#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 5 (REVISED) — Deterministic FiLM decoder + NTK scaling
#   Adam optimizer + ntk_scaled loss (Wang et al. 2022 Type I fix)
#   Multiscale sigmas: {1.0, 2.0, 4.0, 8.0} — 99% information coverage
# ===================================================================
# Implements Algorithm 1 from Wang et al. (2022): "When and Why PINNs Fail"
#
# Theory: NTK trace-based scaling fixes Type I (magnitude) pathology
#   • Gradient imbalance: 1:2:4:8 ratio
#   • Type I fix: weight = Tr(K_total) / Tr(K_i) via EMA traces
#   • Type II: Directional opposition REMAINS (not fixed by NTK)
#
# Purpose:
#   Demonstrate that Type I scaling helps Adam on multiscale problem
#   Show that Type I alone is insufficient (Type II remains)
#   Intermediate step between baseline (Exp 4) and optimal (Exp 6)
#
# Expected vs Exp 4:
#   • Convergence: Better, fewer oscillations
#   • Stability: Improved trace reweighting helps
#   • MSE: ~10-20% better than Adam+L2
#   • Role: NTK approach partially addresses optimization pathology
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../.."

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
  --ntk-estimate-total-trace \
  --ntk-total-trace-ema-decay 0.99 \
  --ntk-epsilon 1e-8 \
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
  --vis-interval 5 > fae_film_adam_ntk_99pct.log 2>&1 &
