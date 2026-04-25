#!/usr/bin/env bash
set -euo pipefail

# Canonical token-latent SIGReg rerun on refreshed minmax data. Transformer
# token latents are flattened before the sliced Epps-Pulley loss.

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"
ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions_minmax.npz}"
RUN_NAME="${RUN_NAME:-transformer_patch16_adamw_ntk_sigreg_balanced_l64x32}"
RESULTS_ROOT="${RESULTS_ROOT:-results/fae_transformer_patch16_adamw_ntk_sigreg_balanced_l64x32}"
LOG_PATH="${LOG_PATH:-logs/${RUN_NAME}.log}"
WANDB_PROJECT="${WANDB_PROJECT:-fae-transformer-sigreg-reruns}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-1.0}"
SIGREG_NUM_SLICES="${SIGREG_NUM_SLICES:-1024}"
SIGREG_NUM_POINTS="${SIGREG_NUM_POINTS:-17}"
SIGREG_T_MAX="${SIGREG_T_MAX:-3.0}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

CMD=(
  "$PYTHON_BIN"
  scripts/fae/train_fae_transformer_sigreg.py
  --data-path "$DATA_PATH"
  --output-dir "$RESULTS_ROOT"
  --run-name "$RUN_NAME"
  --training-mode multi_scale
  --n-freqs 64
  --fourier-sigma 1.0
  --decoder-features 128,128,128,128
  --n-heads 4
  --transformer-tokenization patches
  --transformer-emb-dim 32
  --transformer-num-latents 64
  --transformer-encoder-depth 6
  --transformer-cross-attn-depth 2
  --transformer-decoder-depth 4
  --transformer-mlp-ratio 2
  --transformer-layer-norm-eps 1e-5
  --transformer-patch-size 16
  --encoder-point-ratio 0.3
  --decoder-n-points 4096
  --masking-strategy random
  --detail-quantile 0.85
  --enc-detail-frac 0.05
  --importance-grad-weight 0.5
  --importance-power 1.0
  --eval-masking-strategy same
  --beta 0.0
  --loss-type ntk_sigreg_balanced
  --ntk-epsilon 1e-8
  --ntk-total-trace-ema-decay 0.99
  --ntk-trace-update-interval 100
  --ntk-hutchinson-probes 4
  --ntk-output-chunk-size 0
  --ntk-trace-estimator fhutch
  --optimizer adamw
  --weight-decay 1e-5
  --lr 1e-3
  --lr-warmup-steps 2000
  --lr-decay-step 2000
  --lr-decay-factor 0.9
  --max-steps 50000
  --batch-size 32
  --eval-interval 5
  --train-ratio 0.8
  --seed 42
  --n-vis-samples 4
  --vis-interval 5
  --eval-n-batches 10
  --eval-time-max-samples 128
  --eval-time-split test
  --save-best-model
  --wandb-project "$WANDB_PROJECT"
  --sigreg-weight "$SIGREG_WEIGHT"
  --sigreg-num-slices "$SIGREG_NUM_SLICES"
  --sigreg-num-points "$SIGREG_NUM_POINTS"
  --sigreg-t-max "$SIGREG_T_MAX"
)

if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo "$!" >"${LOG_PATH}.pid"
  printf 'Launched %s (pid=%s)\nLog: %s\n' "$RUN_NAME" "$(cat "${LOG_PATH}.pid")" "$LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
fi
