#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch16_adamw_ntk_prior_balanced_l32x128_common.sh"
transformer_patch16_defaults

BACKGROUND="${BACKGROUND:-0}"
NOHUP_LOG="${NOHUP_LOG:-}"
RUN_DIR="${RUN_DIR:-${RUN_DIR_DEFAULT}}"
LATENTS_PATH="${LATENTS_PATH:-${LATENTS_PATH_DEFAULT}}"
SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-${TRAIN_DATA_PATH_DEFAULT}}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-${FAE_CHECKPOINT_DEFAULT}}"
CALIBRATION_JSON="${CALIBRATION_JSON:-${CALIBRATION_JSON_DEFAULT}}"
SIGMA="${SIGMA:-}"
TIME_DIM="${TIME_DIM:-128}"
DT0="${DT0:-0.01}"
LR="${LR:-1e-3}"
NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CONDITION_MODE="${CONDITION_MODE:-previous_state}"
ENDPOINT_EPSILON="${ENDPOINT_EPSILON:-1e-3}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
LOG_EVERY="${LOG_EVERY:-500}"
SEED="${SEED:-0}"
TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-256}"
TRANSFORMER_N_LAYERS="${TRANSFORMER_N_LAYERS:-2}"
TRANSFORMER_NUM_HEADS="${TRANSFORMER_NUM_HEADS:-4}"
TRANSFORMER_MLP_RATIO="${TRANSFORMER_MLP_RATIO:-2.0}"
TRANSFORMER_TOKEN_DIM="${TRANSFORMER_TOKEN_DIM:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background)
      BACKGROUND="1"
      shift
      ;;
    --foreground)
      BACKGROUND="0"
      shift
      ;;
    --run_dir|--outdir|--latents_path|--source_dataset_path|--fae_checkpoint|--calibration_json|--sigma|--time_dim|--dt0|--lr|--num_steps|--batch_size|--condition_mode|--endpoint_epsilon|--sample_count|--log_every|--seed|--transformer_hidden_dim|--transformer_n_layers|--transformer_num_heads|--transformer_mlp_ratio|--transformer_token_dim)
      case "$1" in
        --run_dir|--outdir) RUN_DIR="$2" ;;
        --latents_path) LATENTS_PATH="$2" ;;
        --source_dataset_path) SOURCE_DATASET_PATH="$2" ;;
        --fae_checkpoint) FAE_CHECKPOINT="$2" ;;
        --calibration_json) CALIBRATION_JSON="$2" ;;
        --sigma) SIGMA="$2" ;;
        --time_dim) TIME_DIM="$2" ;;
        --dt0) DT0="$2" ;;
        --lr) LR="$2" ;;
        --num_steps) NUM_STEPS="$2" ;;
        --batch_size) BATCH_SIZE="$2" ;;
        --condition_mode) CONDITION_MODE="$2" ;;
        --endpoint_epsilon) ENDPOINT_EPSILON="$2" ;;
        --sample_count) SAMPLE_COUNT="$2" ;;
        --log_every) LOG_EVERY="$2" ;;
        --seed) SEED="$2" ;;
        --transformer_hidden_dim) TRANSFORMER_HIDDEN_DIM="$2" ;;
        --transformer_n_layers) TRANSFORMER_N_LAYERS="$2" ;;
        --transformer_num_heads) TRANSFORMER_NUM_HEADS="$2" ;;
        --transformer_mlp_ratio) TRANSFORMER_MLP_RATIO="$2" ;;
        --transformer_token_dim) TRANSFORMER_TOKEN_DIM="$2" ;;
      esac
      shift 2
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
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

transformer_patch16_activate_env
transformer_patch16_require_file "${LATENTS_PATH}" "latent archive"
transformer_patch16_require_file "${FAE_CHECKPOINT}" "FAE checkpoint"
transformer_patch16_mkdirs "${RUN_DIR}" "${LOG_DIR}"

if [[ -z "${SIGMA}" ]]; then
  if [[ -f "${CALIBRATION_JSON}" ]]; then
    SIGMA="$(transformer_patch16_read_sigma "${CALIBRATION_JSON}")"
  else
    echo "ERROR: set SIGMA explicitly or run the sigma calibration wrapper first." >&2
    echo "Missing calibration JSON: ${CALIBRATION_JSON}" >&2
    exit 1
  fi
fi

if [[ -z "${TRANSFORMER_TOKEN_DIM}" ]]; then
  TRANSFORMER_TOKEN_DIM="$(transformer_patch16_infer_token_dim "${LATENTS_PATH}")"
fi

CMD=(
  "${PYTHON_BIN}" -u scripts/csp/train_csp.py
  --latents_path "${LATENTS_PATH}"
  --source_dataset_path "${SOURCE_DATASET_PATH}"
  --fae_checkpoint "${FAE_CHECKPOINT}"
  --outdir "${RUN_DIR}"
  --drift_architecture transformer
  --time_dim "${TIME_DIM}"
  --transformer_hidden_dim "${TRANSFORMER_HIDDEN_DIM}"
  --transformer_n_layers "${TRANSFORMER_N_LAYERS}"
  --transformer_num_heads "${TRANSFORMER_NUM_HEADS}"
  --transformer_mlp_ratio "${TRANSFORMER_MLP_RATIO}"
  --transformer_token_dim "${TRANSFORMER_TOKEN_DIM}"
  --sigma "${SIGMA}"
  --dt0 "${DT0}"
  --lr "${LR}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --condition_mode "${CONDITION_MODE}"
  --endpoint_epsilon "${ENDPOINT_EPSILON}"
  --sample_count "${SAMPLE_COUNT}"
  --log_every "${LOG_EVERY}"
  --seed "${SEED}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  Transformer flat CSP training"
echo "  Environment      : ${ENV_NAME}"
echo "  Latents          : ${LATENTS_PATH}"
echo "  Dataset          : ${SOURCE_DATASET_PATH}"
echo "  FAE checkpoint   : ${FAE_CHECKPOINT}"
echo "  Calibration JSON : ${CALIBRATION_JSON}"
echo "  Output run       : ${RUN_DIR}"
echo "  Sigma            : ${SIGMA}"
echo "  Condition mode   : ${CONDITION_MODE}"
echo "  Steps / batch    : ${NUM_STEPS} / ${BATCH_SIZE}"
echo "  Transformer      : hid=${TRANSFORMER_HIDDEN_DIM} layers=${TRANSFORMER_N_LAYERS} heads=${TRANSFORMER_NUM_HEADS} mlp=${TRANSFORMER_MLP_RATIO} tok=${TRANSFORMER_TOKEN_DIM}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

if [[ "${BACKGROUND}" == "1" && -z "${NOHUP_LOG}" ]]; then
  NOHUP_LOG="${LOG_DIR}/train_csp_main.nohup.log"
fi

transformer_patch16_launch "${BACKGROUND}" "${NOHUP_LOG}" "${CMD[@]}"
