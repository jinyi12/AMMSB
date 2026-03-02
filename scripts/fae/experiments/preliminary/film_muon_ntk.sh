#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Experiment 6 — Deterministic FiLM decoder + NTK trace balancing
#   Muon optimizer + ntk_scaled loss (Wang et al. 2022 complete)
# ===================================================================
# Implements Algorithm 1 from Wang et al. (2022) with second-order opt
#
# Theory: Combines NTK Type I balancing (trace weighting) with Type II
# resolution (Hessian preconditioner via Muon/Shampoo):
#
# (1) Wang et al. Algorithm 1 (Type I): 
#     weight = Tr(K_total) / Tr(K_i)  via periodic exact NTK-diagonal calibration
#
# (2) Muon/Shampoo (Type II):
#     Quasi-second-order preconditioner rotates gradients to avoid
#     directional opposition between competing loss terms
#
# Together: Both magnitude imbalance AND directional conflict resolved
# → This is the OPTIMAL setup per Wang et al. theory.
#
# Comparison with Exp 5 (Adam+NTK):
#   Exp 5: Type I fixed, Type II remains → suboptimal
#   Exp 6: Type I+II both fixed → optimal
#
# Expected: Exp 6 reaches best held-out MSE and trains most smoothly.
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_film_muon_ntk \
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
  --ntk-trace-update-interval 100 \
  --ntk-hutchinson-probes 1 \
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
  --vis-interval 5 > fae_film_muon_ntk.log 2>&1 &
