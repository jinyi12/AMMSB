#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
# Canonical minmax rerun — AdamW + NTK + Prior on refreshed minmax data
# ===================================================================
# Keeps the previously successful FiLM FAE architecture and the same
# ntk_prior_balanced latent diffusion-prior objective, but uses AdamW as the
# canonical optimizer on the maintained minmax archive.

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"
ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions_minmax.npz}"
RESULTS_ROOT="${RESULTS_ROOT:-results/fae_film_adamw_ntk_prior_latent128_minmax}"
RUN_NAME="${RUN_NAME:-fae_film_adamw_ntk_prior_latent128_minmax}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
NTK_OUTPUT_CHUNK_SIZE="${NTK_OUTPUT_CHUNK_SIZE:-32768}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LR="${LR:-1e-3}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_N_BATCHES="${EVAL_N_BATCHES:-5}"
VIS_INTERVAL="${VIS_INTERVAL:-5}"
# Active train-time order for this archive is:
#   t = [0.125, 0.375, 0.5, 0.75, 0.875, 1.0]
# because tran_inclusion excludes t=0 and the archive holds out indices 2 and 5.
ENCODER_POINT_RATIO_BY_TIME="${ENCODER_POINT_RATIO_BY_TIME:-0.8,0.6,0.4,0.2,0.1,0.1}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_FILE")" "$RESULTS_ROOT"

CMD=(
  "$PYTHON_BIN"
  scripts/fae/train_fae_film_prior.py
  --data-path "$DATA_PATH"
  --output-dir "$RESULTS_ROOT"
  --run-name "$RUN_NAME"
  --pooling-type dual_stream_bottleneck
  --decoder-type film
  --latent-dim 128
  --n-freqs 64
  --encoder-multiscale-sigmas 1.0,2.0,4.0
  --decoder-multiscale-sigmas 1.0,2.0,4.0
  --encoder-mlp-layers 3
  --encoder-mlp-dim 256
  --decoder-features 256,256,256
  --masking-strategy random
  --eval-masking-strategy same
  --encoder-point-ratio-by-time "$ENCODER_POINT_RATIO_BY_TIME"
  --beta 0.0
  --loss-type ntk_prior_balanced
  --ntk-epsilon 1e-8
  --ntk-total-trace-ema-decay 0.99
  --ntk-trace-update-interval 100
  --ntk-hutchinson-probes 4
  --ntk-trace-estimator fhutch
  --ntk-output-chunk-size "$NTK_OUTPUT_CHUNK_SIZE"
  --prior-hidden-dim 256
  --prior-n-layers 3
  --prior-time-emb-dim 32
  --prior-logsnr-max 8.0
  --prior-loss-weight 1.0
  --save-best-model
  --wandb-project "${WANDB_PROJECT:-fae-film-latent128-rerun}"
  --optimizer adamw
  --weight-decay "$WEIGHT_DECAY"
  --lr "$LR"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --eval-interval "$EVAL_INTERVAL"
  --eval-n-batches "$EVAL_N_BATCHES"
  --vis-interval "$VIS_INTERVAL"
)

if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  echo "$!" >"${LOG_FILE}.pid"
  printf 'Launched %s (pid=%s)\nLog: %s\n' "$RUN_NAME" "$(cat "${LOG_FILE}.pid")" "$LOG_FILE"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
