#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Denoiser decoder — FiLM (StandardDenoiserDecoder), x0 loss
# ===================================================================
# Ablation against film_velocity.sh to isolate the effect
# of the loss objective:
#   - x0-prediction loss ||u_pred - u_clean||² instead of velocity matching
#   - Everything else identical to the FiLM velocity run
#
# x0 loss directly penalizes reconstruction error and may converge
# faster, but velocity loss has better theoretical grounding for
# rectified flow (straight trajectories → fewer ODE steps at inference).
# ===================================================================

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"

cd "$(dirname "$0")/../../../.."

nohup "$PYTHON_BIN" scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_denoiser_film_x0 \
  --pooling-type dual_stream_bottleneck \
  --decoder-type denoiser_standard \
  --latent-dim 256 \
  --n-freqs 64 \
  --encoder-multiscale-sigmas 1.0 \
  --decoder-multiscale-sigmas 1.0 \
  --encoder-mlp-layers 3 \
  --encoder-mlp-dim 256 \
  --decoder-features 256,256,256 \
  --denoiser-diffusion-steps 100 \
  --denoiser-eval-sample-steps 32 \
  --denoiser-beta-schedule cosine \
  --denoiser-time-sampling logit_normal \
  --denoiser-velocity-loss-weight 0.0 \
  --denoiser-x0-loss-weight 1.0 \
  --masking-strategy random \
  --eval-masking-strategy same \
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
  --beta 1e-4 \
  --save-best-model \
  --wandb-project fae-denoiser-comparison \
  --optimizer muon \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-interval 25 \
  --eval-n-batches 5 \
  --vis-interval 5 > fae_denoiser_film_x0.log 2>&1 &
