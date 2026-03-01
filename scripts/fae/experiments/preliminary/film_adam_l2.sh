#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 4 — Deterministic FiLM decoder, Adam optimizer, L2 loss
# ===================================================================
# Optimizer ablation to compare with Exp 1 (Muon + L2):
#
#   Exp 1         FiLM + Muon + L2      (baseline)
#   Exp 4 (this)  FiLM + Adam + L2      (optimizer effect)
#   Exp 5         FiLM + Adam + ntk_scaled (combined optimal)
#   Exp 6         FiLM + Muon + ntk_scaled (loss effect, optional)
#
# Isolates: Adam optimizer convergence rate and final performance
# vs Muon on identical FiLM architecture with L2 loss.
#
# Hypothesis: Adam is more standard for neural nets and may converge
# faster/better than Muon on this deterministic FiLM setting.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_adam_l2 \
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
  --wandb-project fae-film-optimizer-loss-ablation \
  --optimizer adam \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_film_adam_l2.log 2>&1 &
