#!/usr/bin/env bash
set -euo pipefail

# Baseline-matched dual-stream run (replaces multi_query_augmented_residual pooling)
# while keeping optimizer/loss/data settings aligned for comparable performance.

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_standard_dual_stream_bottleneck \
  --pooling-type dual_stream_bottleneck \
  --decoder-type standard \
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
  --wandb-project fae-standard-dual-stream-bottleneck-logstandardize \
  --optimizer muon \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 10 \
  --vis-interval 5 > fae_standard_dual_stream_bottleneck.log 2>&1 &
