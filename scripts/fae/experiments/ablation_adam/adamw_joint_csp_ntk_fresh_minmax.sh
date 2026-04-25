#!/usr/bin/env bash
set -euo pipefail

# Historical launcher name retained for continuity; fresh joint starts are now
# disabled. This surface is warm-start-only AdamW + NTK-balanced joint
# latent-CSP on the minmax archive.
#
# This launcher warm-starts the FiLM autoencoder from an explicit checkpoint,
# keeps sigma fixed, and trains the latent CSP bridge jointly using the
# shared-encoder NTK balancing surface.

PYTHON_BIN="${PYTHON_BIN:-/home/jy384/miniconda3/envs/3MASB/bin/python}"
ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions_minmax.npz}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RESULTS_ROOT="${RESULTS_ROOT:-results/fae_film_adamw_joint_csp_ntk_latent128_minmax}"
RUN_NAME="${RUN_NAME:-film_adamw_joint_csp_ntk_latent128_minmax_sigma04848_warmstart}"
LOG_PATH="${LOG_PATH:-logs/${RUN_NAME}.log}"
WANDB_PROJECT="${WANDB_PROJECT:-fae-film-latent128-rerun}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

SIGMA="${SIGMA:-0.04848}"
JOINT_CSP_LOSS_WEIGHT="${JOINT_CSP_LOSS_WEIGHT:-1.0}"
JOINT_CSP_WARMUP_STEPS="${JOINT_CSP_WARMUP_STEPS:-1000}"
JOINT_CSP_BATCH_SIZE="${JOINT_CSP_BATCH_SIZE:-96}"
JOINT_CSP_MC_MULTIPLIER="${JOINT_CSP_MC_MULTIPLIER:-4}"
JOINT_CSP_MC_CHUNK_SIZE="${JOINT_CSP_MC_CHUNK_SIZE:-8}"
JOINT_CSP_TARGET_REFRESH_INTERVAL="${JOINT_CSP_TARGET_REFRESH_INTERVAL:-250}"
JOINT_CSP_VARIANCE_FLOOR_WEIGHT="${JOINT_CSP_VARIANCE_FLOOR_WEIGHT:-1.0}"
JOINT_CSP_VARIANCE_FLOOR="${JOINT_CSP_VARIANCE_FLOOR:-1e-2}"
JOINT_CSP_VARIANCE_DIRECTIONS="${JOINT_CSP_VARIANCE_DIRECTIONS:-32}"

NTK_TRACE_UPDATE_INTERVAL="${NTK_TRACE_UPDATE_INTERVAL:-250}"
NTK_HUTCHINSON_PROBES="${NTK_HUTCHINSON_PROBES:-1}"
NTK_TRACE_ESTIMATOR="${NTK_TRACE_ESTIMATOR:-fhutch}"
NTK_OUTPUT_CHUNK_SIZE="${NTK_OUTPUT_CHUNK_SIZE:-32768}"
NTK_TOTAL_TRACE_EMA_DECAY="${NTK_TOTAL_TRACE_EMA_DECAY:-0.99}"

WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LR="${LR:-1e-3}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_N_BATCHES="${EVAL_N_BATCHES:-5}"
VIS_INTERVAL="${VIS_INTERVAL:-5}"
ENCODER_POINT_RATIO_BY_TIME="${ENCODER_POINT_RATIO_BY_TIME:-0.8,0.6,0.4,0.2,0.1,0.1}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_ROOT"

if [[ -z "$INIT_CHECKPOINT" ]]; then
  echo "ERROR: INIT_CHECKPOINT must point to a warm-start FAE checkpoint; fresh joint training is disabled." >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "ERROR: INIT_CHECKPOINT does not exist: $INIT_CHECKPOINT" >&2
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

CMD=(
  "$PYTHON_BIN"
  scripts/fae/train_fae_film_joint_csp.py
  --data-path "$DATA_PATH"
  --output-dir "$RESULTS_ROOT"
  --run-name "$RUN_NAME"
  --training-mode multi_scale
  --decoder-type film
  --pooling-type dual_stream_bottleneck
  --optimizer adamw
  --loss-type ntk_bridge_balanced
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
  --joint-csp-loss-weight "$JOINT_CSP_LOSS_WEIGHT"
  --joint-csp-warmup-steps "$JOINT_CSP_WARMUP_STEPS"
  --joint-csp-batch-size "$JOINT_CSP_BATCH_SIZE"
  --joint-csp-mc-multiplier "$JOINT_CSP_MC_MULTIPLIER"
  --joint-csp-mc-chunk-size "$JOINT_CSP_MC_CHUNK_SIZE"
  --joint-csp-target-refresh-interval "$JOINT_CSP_TARGET_REFRESH_INTERVAL"
  --joint-csp-variance-floor-weight "$JOINT_CSP_VARIANCE_FLOOR_WEIGHT"
  --joint-csp-variance-floor "$JOINT_CSP_VARIANCE_FLOOR"
  --joint-csp-variance-directions "$JOINT_CSP_VARIANCE_DIRECTIONS"
  --hidden 512 512 512
  --time-dim 128
  --drift-architecture mlp
  --init-checkpoint "$INIT_CHECKPOINT"
  --sigma "$SIGMA"
  --sigma-update-mode fixed
  --ntk-trace-update-interval "$NTK_TRACE_UPDATE_INTERVAL"
  --ntk-hutchinson-probes "$NTK_HUTCHINSON_PROBES"
  --ntk-trace-estimator "$NTK_TRACE_ESTIMATOR"
  --ntk-output-chunk-size "$NTK_OUTPUT_CHUNK_SIZE"
  --ntk-total-trace-ema-decay "$NTK_TOTAL_TRACE_EMA_DECAY"
  --dt0 0.01
  --condition-mode previous_state
  --endpoint-epsilon 1e-3
)

if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo "$!" >"${LOG_PATH}.pid"
  printf 'Launched %s (pid=%s)\nLog: %s\n' "$RUN_NAME" "$(cat "${LOG_PATH}.pid")" "$LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
fi
