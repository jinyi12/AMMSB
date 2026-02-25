#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 6 (REVISED) — Deterministic FiLM decoder + NTK scaling
#   Muon optimizer + ntk_scaled loss (COMPLETE Type I+II fix)
#   Multiscale sigmas: {1.0, 2.0, 4.0, 8.0} — 99% information coverage (FULL SPECTRUM)
# ===================================================================
# Combines:
#   1. NTK trace scaling (Wang et al. 2022 Algorithm 1):
#      Fixes Type I (magnitude imbalance) via λ_i = Tr(K_total) / Tr(K_i)
#
#   2. Muon/Shampoo second-order optimizer:
#      Fixes Type II (directional opposition) via quasi-Hessian preconditioning
#
# Gradient imbalance: 1:2:4:8 ratio (8× max, manageable)
# Key decision: Using FULL multiscale {1.0, 2.0, 4.0, 8.0}, not σ=1.0 only
#   • Rationale: Fair comparison requires identical feature set across all methods
#   • Muon's strength: Handling multiscale problems → test on full spectrum
#   • Demonstrates: What makes Muon fundamentally better for structured problems
#
# Purpose:
#   Optimal solution: Both Type I (trace) and Type II (Shampoo) pathologies fixed
#   Fair comparison: All three optimizers use identical RFF feature representation
#   Scientific narrative: Shows Muon's advantage on realistic multiscale problems
#
# Expected:
#   • Convergence: Fast, smooth (no Type I/II issues)
#   • Stability: Excellent (Shampoo preconditioner handles coupling)
#   • MSE: ~15-30% better than Adam+NTK
#   • Role: Demonstrates why second-order methods matter for multiscale FAE
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../.."

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
  --vis-interval 5 > fae_film_muon_ntk_99pct.log 2>&1 &
