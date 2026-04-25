#!/usr/bin/env bash
set -euo pipefail

# Canonical FiLM SIGReg rerun on refreshed minmax data using AdamW and
# clean-latent sliced Epps-Pulley regularization.
#
# The fixed-weight FiLM rerun uses a much smaller SIGReg coefficient than the
# library default because the raw SIGReg statistic starts around O(10^1),
# while the desired reconstruction target is O(10^-3) to O(10^-4). The paper
# also reports limited sensitivity to the number of projections, so we use 128
# slices instead of 1024 to cut steady-state compute.

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"
ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions_minmax.npz}"
RUN_NAME="${RUN_NAME:-film_adamw_sigreg_l2_latent128_minmax_w1e4_s128}"
RESULTS_ROOT="${RESULTS_ROOT:-results/fae_film_adamw_sigreg_l2_latent128_minmax_w1e4_s128}"
LOG_PATH="${LOG_PATH:-logs/${RUN_NAME}.log}"
WANDB_PROJECT="${WANDB_PROJECT:-fae-film-latent128-rerun}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-1e-4}"
SIGREG_NUM_SLICES="${SIGREG_NUM_SLICES:-128}"
SIGREG_NUM_POINTS="${SIGREG_NUM_POINTS:-17}"
SIGREG_T_MAX="${SIGREG_T_MAX:-3.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LR="${LR:-1e-3}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_N_BATCHES="${EVAL_N_BATCHES:-5}"
VIS_INTERVAL="${VIS_INTERVAL:-5}"
BETA="${BETA:-0.0}"
ENCODER_POINT_RATIO_BY_TIME="${ENCODER_POINT_RATIO_BY_TIME:-0.8,0.6,0.4,0.2,0.1,0.1}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

CMD=(
  "$PYTHON_BIN"
  scripts/fae/train_fae_film_sigreg.py
  --data-path "$DATA_PATH"
  --output-dir "$RESULTS_ROOT"
  --run-name "$RUN_NAME"
  --training-mode multi_scale
  --decoder-type film
  --pooling-type dual_stream_bottleneck
  --optimizer adamw
  --loss-type l2
  --n-freqs 64
  --fourier-sigma 1.0
  --latent-dim 128
  --encoder-multiscale-sigmas 1.0,2.0,4.0
  --decoder-multiscale-sigmas 1.0,2.0,4.0
  --encoder-mlp-layers 3
  --encoder-mlp-dim 256
  --decoder-features 256,256,256
  --masking-strategy random
  --eval-masking-strategy same
  --encoder-point-ratio-by-time "$ENCODER_POINT_RATIO_BY_TIME"
  --beta "$BETA"
  --weight-decay "$WEIGHT_DECAY"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --train-ratio 0.8
  --eval-interval "$EVAL_INTERVAL"
  --eval-n-batches "$EVAL_N_BATCHES"
  --eval-time-max-samples 128
  --eval-time-split test
  --seed 42
  --vis-interval "$VIS_INTERVAL"
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
