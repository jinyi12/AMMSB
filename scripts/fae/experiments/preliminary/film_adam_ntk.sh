#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 5 — Deterministic FiLM decoder + NTK trace balancing
#   Adam optimizer + ntk_scaled loss (Wang et al. 2022)
# ===================================================================
# Implements Algorithm 1 from Wang et al. (2022): "When and Why PINNs Fail"
#
# Theory (Wang et al. 2022, Eqs. 6.5-6.6):
#   Type I conflict: Magnitude imbalance across loss components (scales)
#   → Solution: Adaptive weights λ_i = Tr(K_total) / Tr(K_i)
#   → Via periodic exact NTK-diagonal calibration
#
# Key equation (Wang et al. Algorithm 1):
#   weight = Tr(K_total) / Tr(K_i)
# with Tr(K_total) updated on calibration steps and held fixed in between
# (no EMA smoothing).
#
# Hypothesis: Adam + NTK Type I balancing addresses magnitude imbalance
# via trace weighting, but Type II (directional opposition) remains → 
# still inferior to Muon which has second-order preconditioner.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_adam_ntk \
  --pooling-type dual_stream_bottleneck \
  --decoder-type film \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
  --loss-type ntk_scaled \
  --ntk-epsilon 1e-8 \
  --ntk-estimate-total-trace \
  --ntk-total-trace-ema-decay 0.0 \
  --ntk-calibration-interval 100 \
  --ntk-cv-threshold 0.2 \
  --ntk-calibration-pilot-samples 8 \
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
  --vis-interval 5 > fae_film_adam_ntk.log 2>&1 &
