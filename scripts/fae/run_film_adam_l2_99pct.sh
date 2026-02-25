#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 4 (REVISED) — Deterministic FiLM decoder + L2 baseline
#   Adam optimizer + L2 loss (demonstrates multiscale pathology)
#   Multiscale sigmas: {1.0, 2.0, 4.0, 8.0} — 99% information coverage
# ===================================================================
# Setup:
#   • Decoder RFF frequencies: 1.0, 2.0, 4.0, 8.0 (logarithmic spacing)
#   • Gradient ratio: 1:2:4:8 (manageable 8× max imbalance)
#   • Loss: Elementary L2 (no adaptive scaling)
#
# Purpose:
#   Demonstrate Type I/II pathology even at moderate (8×) imbalance
#   Baseline to compare against NTK and Muon improvements
#
# Expected outcome:
#   • Convergence: Yes (8× imbalance ≤ Adam's limit)
#   • Stability: Good, possibly minor oscillations
#   • MSE: Moderate (~0.005-0.01)
#   • Role: Shows what Adam can do with multiscale structured problem
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_adam_l2_99pct \
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
  --wandb-project fae-film-optimizer-loss-ablation \
  --optimizer adam \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_film_adam_l2_99pct.log 2>&1 &
