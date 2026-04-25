#!/usr/bin/env bash
set -euo pipefail

# Curated paired-prior launcher for the transformer FAE run evaluated as wandb
# run `ifb1lc6i`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5}"
ENV_NAME="${ENV_NAME:-3MASB}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FAE_RESULTS_ROOT="${FAE_RESULTS_ROOT:-results/fae_${EXPERIMENT_NAME}}"
FAE_RUN_DIR="${FAE_RUN_DIR:-${FAE_RESULTS_ROOT}/${EXPERIMENT_NAME}}"
OUTPUT_BASE="${OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}_paired_prior_bridge}"
RUN_DIR="${RUN_DIR:-${OUTPUT_BASE}/main}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions_minmax.npz}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-}"

BACKGROUND="${BACKGROUND:-0}"
NOHUP_LOG="${NOHUP_LOG:-}"

TIME_DIM="${TIME_DIM:-128}"
TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-256}"
TRANSFORMER_N_LAYERS="${TRANSFORMER_N_LAYERS:-3}"
TRANSFORMER_NUM_HEADS="${TRANSFORMER_NUM_HEADS:-4}"
TRANSFORMER_MLP_RATIO="${TRANSFORMER_MLP_RATIO:-2.0}"
TRANSFORMER_TOKEN_DIM="${TRANSFORMER_TOKEN_DIM:-128}"
DELTA_V="${DELTA_V:-1.0}"
THETA_TRIM="${THETA_TRIM:-0.05}"
NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
LOG_EVERY="${LOG_EVERY:-500}"
SEED="${SEED:-0}"

resolve_fae_checkpoint() {
  if [[ -n "${FAE_CHECKPOINT}" ]]; then
    return
  fi
  if [[ -f "${FAE_RUN_DIR}/checkpoints/best_state.pkl" ]]; then
    FAE_CHECKPOINT="${FAE_RUN_DIR}/checkpoints/best_state.pkl"
    return
  fi
  if [[ -f "${FAE_RUN_DIR}/checkpoints/state.pkl" ]]; then
    FAE_CHECKPOINT="${FAE_RUN_DIR}/checkpoints/state.pkl"
    return
  fi
  echo "ERROR: could not resolve an FAE checkpoint under ${FAE_RUN_DIR}/checkpoints" >&2
  exit 1
}

activate_env() {
  if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

launch() {
  local background="$1"
  local nohup_log="$2"
  shift 2
  local -a cmd=("$@")

  if [[ "${background}" == "1" ]]; then
    mkdir -p "$(dirname "${nohup_log}")"
    nohup "${cmd[@]}" >"${nohup_log}" 2>&1 &
    local pid=$!
    echo "Started background run."
    echo "  pid: ${pid}"
    echo "  log: ${nohup_log}"
    exit 0
  fi

  exec "${cmd[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --fae_checkpoint)
      FAE_CHECKPOINT="$2"
      shift 2
      ;;
    --run_dir|--outdir)
      RUN_DIR="$2"
      shift 2
      ;;
    --delta_v)
      DELTA_V="$2"
      shift 2
      ;;
    --theta_trim)
      THETA_TRIM="$2"
      shift 2
      ;;
    --num_steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --sample_count)
      SAMPLE_COUNT="$2"
      shift 2
      ;;
    --log_every)
      LOG_EVERY="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --background)
      BACKGROUND="1"
      shift
      ;;
    --foreground)
      BACKGROUND="0"
      shift
      ;;
    --nohup-log)
      NOHUP_LOG="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,260p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      exit 1
      ;;
  esac
done

activate_env
resolve_fae_checkpoint
require_file "${DATA_PATH}" "dataset"
require_file "${FAE_CHECKPOINT}" "FAE checkpoint"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if [[ "${BACKGROUND}" == "1" && -z "${NOHUP_LOG}" ]]; then
  NOHUP_LOG="${LOG_DIR}/train_csp_paired_prior_main.nohup.log"
fi

CMD=(
  "${PYTHON_BIN}" -u scripts/csp/train_csp_paired_prior_from_fae.py
  --data_path "${DATA_PATH}"
  --fae_checkpoint "${FAE_CHECKPOINT}"
  --outdir "${RUN_DIR}"
  --drift_architecture transformer
  --time_dim "${TIME_DIM}"
  --transformer_hidden_dim "${TRANSFORMER_HIDDEN_DIM}"
  --transformer_n_layers "${TRANSFORMER_N_LAYERS}"
  --transformer_num_heads "${TRANSFORMER_NUM_HEADS}"
  --transformer_mlp_ratio "${TRANSFORMER_MLP_RATIO}"
  --transformer_token_dim "${TRANSFORMER_TOKEN_DIM}"
  --delta_v "${DELTA_V}"
  --theta_trim "${THETA_TRIM}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --sample_count "${SAMPLE_COUNT}"
  --log_every "${LOG_EVERY}"
  --seed "${SEED}"
)

echo "============================================================"
echo "  Patch-8 paired-prior CSP bridge training"
echo "  Source run       : ${FAE_RUN_DIR}"
echo "  wandb_run_id     : ifb1lc6i"
echo "  Dataset          : ${DATA_PATH}"
echo "  FAE checkpoint   : ${FAE_CHECKPOINT}"
echo "  Output run       : ${RUN_DIR}"
echo "  delta_v          : ${DELTA_V}"
echo "  theta_trim       : ${THETA_TRIM}"
echo "  Steps / batch    : ${NUM_STEPS} / ${BATCH_SIZE}"
echo "  Transformer      : hid=${TRANSFORMER_HIDDEN_DIM} layers=${TRANSFORMER_N_LAYERS} heads=${TRANSFORMER_NUM_HEADS} mlp=${TRANSFORMER_MLP_RATIO} tok=${TRANSFORMER_TOKEN_DIM}"
echo "============================================================"

launch "${BACKGROUND}" "${NOHUP_LOG}" "${CMD[@]}"
