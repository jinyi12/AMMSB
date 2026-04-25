#!/usr/bin/env bash
set -euo pipefail

# Serial queue for the canonical AdamW minmax comparison bundle:
#   1. beta-regularized baseline
#   2. fixed-weight SIGReg
#   3. NTK-balanced SIGReg
#
# The fixed-weight FiLM rerun keeps the raw SIGReg contribution in the same
# rough range as the desired O(10^-3) to O(10^-4) reconstruction target:
#   fixed_sigreg_weight=1e-4, num_slices=128, num_points=17, t_max=3.0
# The NTK-balanced follow-on keeps its own base prior weight:
#   ntk_sigreg_weight=1.0

ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/fae/experiments/ablation_adam/queue_adamw_beta_sigreg_ntk_minmax.sh"
QUEUE_LOG="${QUEUE_LOG:-logs/adamw_beta_sigreg_ntk_minmax_queue.log}"
WANDB_PROJECT="${WANDB_PROJECT:-fae-film-latent128-rerun}"
FIXED_SIGREG_WEIGHT="${FIXED_SIGREG_WEIGHT:-1e-4}"
NTK_SIGREG_WEIGHT="${NTK_SIGREG_WEIGHT:-1.0}"
SIGREG_NUM_SLICES="${SIGREG_NUM_SLICES:-128}"
SIGREG_NUM_POINTS="${SIGREG_NUM_POINTS:-17}"
SIGREG_T_MAX="${SIGREG_T_MAX:-3.0}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"

JOBS=(
  "scripts/fae/experiments/ablation_adam/adamw_l2_minmax.sh"
  "scripts/fae/experiments/sigreg/rerun_film_adamw_sigreg_l2_minmax.sh"
  "scripts/fae/experiments/ablation_adam/adamw_ntk_minmax.sh"
)

cd "$ROOT_DIR"
mkdir -p "$(dirname "$QUEUE_LOG")"

log_line() {
  printf '%s\n' "$1" >>"$QUEUE_LOG"
}

run_job() {
  local job="$1"
  local sigreg_weight
  local ts

  sigreg_weight="$FIXED_SIGREG_WEIGHT"
  if [[ "$job" == *"adamw_ntk_minmax.sh" ]]; then
    sigreg_weight="$NTK_SIGREG_WEIGHT"
  fi

  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  log_line "[$ts] START $job"
  env \
    WANDB_PROJECT="$WANDB_PROJECT" \
    SIGREG_WEIGHT="$sigreg_weight" \
    SIGREG_NUM_SLICES="$SIGREG_NUM_SLICES" \
    SIGREG_NUM_POINTS="$SIGREG_NUM_POINTS" \
    SIGREG_T_MAX="$SIGREG_T_MAX" \
    RUN_IN_BACKGROUND=0 \
    bash "$job" >>"$QUEUE_LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    log_line "[$ts] FAIL  $job (exit=$rc)"
    exit "$rc"
  fi
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  log_line "[$ts] DONE  $job"
}

run_queue() {
  for job in "${JOBS[@]}"; do
    run_job "$job"
  done
  log_line "[$(date '+%Y-%m-%d %H:%M:%S')] QUEUE COMPLETE"
}

if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  nohup env \
    QUEUE_LOG="$QUEUE_LOG" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    FIXED_SIGREG_WEIGHT="$FIXED_SIGREG_WEIGHT" \
    NTK_SIGREG_WEIGHT="$NTK_SIGREG_WEIGHT" \
    SIGREG_NUM_SLICES="$SIGREG_NUM_SLICES" \
    SIGREG_NUM_POINTS="$SIGREG_NUM_POINTS" \
    SIGREG_T_MAX="$SIGREG_T_MAX" \
    RUN_IN_BACKGROUND=0 \
    bash "$SCRIPT_PATH" >>"$QUEUE_LOG" 2>&1 &
  echo "$!" >"${QUEUE_LOG}.pid"
  printf 'Queued AdamW beta/SIGReg/NTK bundle (pid=%s)\nLog: %s\n' "$(cat "${QUEUE_LOG}.pid")" "$QUEUE_LOG"
else
  run_queue
fi
