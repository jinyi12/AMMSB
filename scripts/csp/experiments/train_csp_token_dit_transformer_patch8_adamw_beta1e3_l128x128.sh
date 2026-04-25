#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch8_adamw_beta1e3_l128x128_common.sh"
transformer_patch8_prior_defaults

BACKGROUND="${BACKGROUND:-0}"
NOHUP_LOG="${NOHUP_LOG:-}"
TOKEN_CONDITIONING="${TOKEN_CONDITIONING:-set_conditioned_memory}"

TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}_token_dit_${TOKEN_CONDITIONING}}"
OUTDIR="${OUTDIR:-${TOKEN_DIT_OUTPUT_BASE}/main}"
LOG_DIR_TOKEN_DIT="${LOG_DIR_TOKEN_DIT:-${TOKEN_DIT_OUTPUT_BASE}/logs}"

DATA_PATH="${DATA_PATH:-${TRAIN_DATA_PATH_DEFAULT}}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-${FAE_CHECKPOINT_DEFAULT}}"
LATENTS_PATH="${LATENTS_PATH:-${OUTDIR}/fae_token_latents.npz}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTDIR}/config/fae_token_latents_manifest.json}"
CALIBRATION_JSON="${CALIBRATION_JSON:-${TOKEN_DIT_OUTPUT_BASE}/calibration/sigma_calibration.json}"
SIGMA="${SIGMA:-}"

ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
MAX_SAMPLES_PER_TIME="${MAX_SAMPLES_PER_TIME:-}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
HELD_OUT_INDICES="${HELD_OUT_INDICES:-}"
HELD_OUT_TIMES="${HELD_OUT_TIMES:-}"
TIME_DIST_MODE="${TIME_DIST_MODE:-uniform}"
T_SCALE="${T_SCALE:-1.0}"
SKIP_ENCODE_IF_EXISTS="${SKIP_ENCODE_IF_EXISTS:-1}"

DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-256}"
DIT_N_LAYERS="${DIT_N_LAYERS:-3}"
DIT_NUM_HEADS="${DIT_NUM_HEADS:-4}"
DIT_MLP_RATIO="${DIT_MLP_RATIO:-2.0}"
DIT_TIME_EMB_DIM="${DIT_TIME_EMB_DIM:-128}"
DT0="${DT0:-0.01}"
LR="${LR:-1e-4}"
NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CONDITION_MODE="${CONDITION_MODE:-previous_state}"
ENDPOINT_EPSILON="${ENDPOINT_EPSILON:-1e-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
LOG_EVERY="${LOG_EVERY:-500}"
SEED="${SEED:-0}"
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
    --outdir|--data_path|--fae_checkpoint|--latents_path|--manifest_path|--calibration_json|--sigma|--encode_batch_size|--max_samples_per_time|--train_ratio|--held_out_indices|--held_out_times|--time_dist_mode|--t_scale|--dit_hidden_dim|--dit_n_layers|--dit_num_heads|--dit_mlp_ratio|--dit_time_emb_dim|--dt0|--lr|--num_steps|--batch_size|--condition_mode|--endpoint_epsilon|--sample_count|--log_every|--seed)
      case "$1" in
        --outdir) OUTDIR="$2" ;;
        --data_path) DATA_PATH="$2" ;;
        --fae_checkpoint) FAE_CHECKPOINT="$2" ;;
        --latents_path) LATENTS_PATH="$2" ;;
        --manifest_path) MANIFEST_PATH="$2" ;;
        --calibration_json) CALIBRATION_JSON="$2" ;;
        --sigma) SIGMA="$2" ;;
        --encode_batch_size) ENCODE_BATCH_SIZE="$2" ;;
        --max_samples_per_time) MAX_SAMPLES_PER_TIME="$2" ;;
        --train_ratio) TRAIN_RATIO="$2" ;;
        --held_out_indices) HELD_OUT_INDICES="$2" ;;
        --held_out_times) HELD_OUT_TIMES="$2" ;;
        --time_dist_mode) TIME_DIST_MODE="$2" ;;
        --t_scale) T_SCALE="$2" ;;
        --dit_hidden_dim) DIT_HIDDEN_DIM="$2" ;;
        --dit_n_layers) DIT_N_LAYERS="$2" ;;
        --dit_num_heads) DIT_NUM_HEADS="$2" ;;
        --dit_mlp_ratio) DIT_MLP_RATIO="$2" ;;
        --dit_time_emb_dim) DIT_TIME_EMB_DIM="$2" ;;
        --dt0) DT0="$2" ;;
        --lr) LR="$2" ;;
        --num_steps) NUM_STEPS="$2" ;;
        --batch_size) BATCH_SIZE="$2" ;;
        --condition_mode) CONDITION_MODE="$2" ;;
        --endpoint_epsilon) ENDPOINT_EPSILON="$2" ;;
        --sample_count) SAMPLE_COUNT="$2" ;;
        --log_every) LOG_EVERY="$2" ;;
        --seed) SEED="$2" ;;
      esac
      shift 2
      ;;
    --skip_encode_if_exists)
      SKIP_ENCODE_IF_EXISTS="1"
      shift
      ;;
    --force_reencode)
      SKIP_ENCODE_IF_EXISTS="0"
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
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

transformer_patch8_prior_activate_env
transformer_patch8_prior_require_file "${FAE_CHECKPOINT}" "FAE checkpoint"
transformer_patch8_prior_mkdirs "${OUTDIR}" "${LOG_DIR_TOKEN_DIT}" "$(dirname "${LATENTS_PATH}")" "$(dirname "${MANIFEST_PATH}")"

if [[ -z "${SIGMA}" ]]; then
  if [[ -f "${CALIBRATION_JSON}" ]]; then
    SIGMA="$(transformer_patch8_prior_read_sigma "${CALIBRATION_JSON}")"
  else
    echo "ERROR: set SIGMA explicitly or provide CALIBRATION_JSON from token-native sigma calibration." >&2
    echo "Missing calibration JSON: ${CALIBRATION_JSON}" >&2
    exit 1
  fi
fi

CMD=(
  "${PYTHON_BIN}" -u scripts/csp/train_csp_token_dit_from_fae.py
  --data_path "${DATA_PATH}"
  --fae_checkpoint "${FAE_CHECKPOINT}"
  --outdir "${OUTDIR}"
  --latents_path "${LATENTS_PATH}"
  --manifest_path "${MANIFEST_PATH}"
  --encode_batch_size "${ENCODE_BATCH_SIZE}"
  --train_ratio "${TRAIN_RATIO}"
  --time_dist_mode "${TIME_DIST_MODE}"
  --t_scale "${T_SCALE}"
  --dit_hidden_dim "${DIT_HIDDEN_DIM}"
  --dit_n_layers "${DIT_N_LAYERS}"
  --dit_num_heads "${DIT_NUM_HEADS}"
  --dit_mlp_ratio "${DIT_MLP_RATIO}"
  --dit_time_emb_dim "${DIT_TIME_EMB_DIM}"
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

if [[ "${SKIP_ENCODE_IF_EXISTS}" == "1" ]]; then
  CMD+=(--skip_encode_if_exists)
fi
if [[ -n "${MAX_SAMPLES_PER_TIME}" ]]; then
  CMD+=(--max_samples_per_time "${MAX_SAMPLES_PER_TIME}")
fi
if [[ -n "${HELD_OUT_INDICES}" ]]; then
  CMD+=(--held_out_indices "${HELD_OUT_INDICES}")
fi
if [[ -n "${HELD_OUT_TIMES}" ]]; then
  CMD+=(--held_out_times "${HELD_OUT_TIMES}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  Patch-8 token-native CSP training"
echo "  Environment      : ${ENV_NAME}"
echo "  Dataset          : ${DATA_PATH}"
echo "  FAE checkpoint   : ${FAE_CHECKPOINT}"
echo "  Outdir           : ${OUTDIR}"
echo "  Token latents    : ${LATENTS_PATH}"
echo "  Manifest         : ${MANIFEST_PATH}"
echo "  Calibration JSON : ${CALIBRATION_JSON}"
echo "  Sigma            : ${SIGMA}"
echo "  Condition mode   : ${CONDITION_MODE}"
echo "  Token bridge     : ${TOKEN_CONDITIONING}"
echo "  Encode batch     : ${ENCODE_BATCH_SIZE}"
echo "  Steps / batch    : ${NUM_STEPS} / ${BATCH_SIZE}"
echo "  DiT              : hid=${DIT_HIDDEN_DIM} layers=${DIT_N_LAYERS} heads=${DIT_NUM_HEADS} mlp=${DIT_MLP_RATIO} time=${DIT_TIME_EMB_DIM}"
echo "  Reuse latents    : ${SKIP_ENCODE_IF_EXISTS}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

if [[ "${BACKGROUND}" == "1" && -z "${NOHUP_LOG}" ]]; then
  NOHUP_LOG="${LOG_DIR_TOKEN_DIT}/train_csp_token_dit_main.nohup.log"
fi

transformer_patch8_prior_launch "${BACKGROUND}" "${NOHUP_LOG}" "${CMD[@]}"
