#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5_common.sh"
transformer_patch8_prior_defaults

RUN_TOKEN="${RUN_TOKEN:-1}"
TOKEN_PROFILE="${TOKEN_PROFILE:-publication}"
TOKEN_CONDITIONING="${TOKEN_CONDITIONING:-set_conditioned_memory}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${TRAIN_DATA_PATH_DEFAULT}}"
CORPUS_DATA_PATH="${CORPUS_DATA_PATH:-${CORPUS_DATA_PATH_DEFAULT}}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-${FAE_CHECKPOINT_DEFAULT}}"

TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}_token_dit_${TOKEN_CONDITIONING}}"
TOKEN_CSP_RUN_DIR="${TOKEN_CSP_RUN_DIR:-${TOKEN_DIT_OUTPUT_BASE}/main}"
TOKEN_LATENTS_PATH="${TOKEN_LATENTS_PATH:-${TOKEN_CSP_RUN_DIR}/fae_token_latents.npz}"
TOKEN_MANIFEST_PATH="${TOKEN_MANIFEST_PATH:-${TOKEN_CSP_RUN_DIR}/config/fae_token_latents_manifest.json}"
TOKEN_CORPUS_LATENTS_PATH="${TOKEN_CORPUS_LATENTS_PATH:-${TOKEN_CSP_RUN_DIR}/corpus_latents.npz}"
CALIBRATION_JSON="${CALIBRATION_JSON:-${TOKEN_DIT_OUTPUT_BASE}/calibration/sigma_calibration.json}"
CALIBRATION_TEXT="${CALIBRATION_TEXT:-${TOKEN_DIT_OUTPUT_BASE}/calibration/sigma_calibration.txt}"
COARSE_SPLIT="${COARSE_SPLIT:-test}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token-only)
      RUN_TOKEN="1"
      shift
      ;;
    --skip-token)
      RUN_TOKEN="0"
      shift
      ;;
    --token-profile)
      TOKEN_PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,240p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      exit 1
      ;;
  esac
done

if [[ "${RUN_TOKEN}" != "1" ]]; then
  echo "ERROR: token workflow is disabled." >&2
  exit 1
fi

transformer_patch8_prior_activate_env
transformer_patch8_prior_require_file "${TRAIN_DATA_PATH}" "training dataset"
transformer_patch8_prior_require_file "${CORPUS_DATA_PATH}" "corpus dataset"
transformer_patch8_prior_require_file "${FAE_CHECKPOINT}" "FAE checkpoint"

run_step() {
  local label="$1"
  shift
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label}"
  echo "============================================================"
  "$@"
}

run_step "Encode patch-8 token-native FAE train/test latents" \
  "${PYTHON_BIN}" -u scripts/csp/encode_fae_token_latents.py \
    --data_path "${TRAIN_DATA_PATH}" \
    --fae_checkpoint "${FAE_CHECKPOINT}" \
    --output_path "${TOKEN_LATENTS_PATH}" \
    --manifest_path "${TOKEN_MANIFEST_PATH}" \
    --encode_batch_size 64 \
    --train_ratio 0.8 \
    --time_dist_mode uniform \
    --t_scale 1.0

run_step "Calibrate sigma from patch-8 token-native latents" \
  env TOKEN_CONDITIONING="${TOKEN_CONDITIONING}" \
      LATENTS_PATH="${TOKEN_LATENTS_PATH}" \
      JSON_PATH="${CALIBRATION_JSON}" \
      TEXT_PATH="${CALIBRATION_TEXT}" \
  bash "${SCRIPT_DIR}/calibrate_sigma_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh"

run_step "Train patch-8 token-native CSP DiT bridge" \
  env TOKEN_CONDITIONING="${TOKEN_CONDITIONING}" \
      TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE}" \
      DATA_PATH="${TRAIN_DATA_PATH}" \
      FAE_CHECKPOINT="${FAE_CHECKPOINT}" \
      LATENTS_PATH="${TOKEN_LATENTS_PATH}" \
      MANIFEST_PATH="${TOKEN_MANIFEST_PATH}" \
      CALIBRATION_JSON="${CALIBRATION_JSON}" \
  bash "${SCRIPT_DIR}/train_csp_token_dit_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh" --foreground

run_step "Encode patch-8 token-native corpus latents for conditional evaluation" \
  "${PYTHON_BIN}" -u scripts/fae/tran_evaluation/encode_corpus.py \
    --corpus_path "${CORPUS_DATA_PATH}" \
    --run_dir "${TOKEN_CSP_RUN_DIR}" \
    --output_path "${TOKEN_CORPUS_LATENTS_PATH}" \
    --batch_size 64

run_step "Evaluate patch-8 token-native CSP DiT bridge" \
  env TOKEN_CONDITIONING="${TOKEN_CONDITIONING}" \
      CSP_RUN_DIR="${TOKEN_CSP_RUN_DIR}" \
      CONDITIONAL_CORPUS_LATENTS_PATH="${TOKEN_CORPUS_LATENTS_PATH}" \
      PROFILE="${TOKEN_PROFILE}" \
      COARSE_SPLIT="${COARSE_SPLIT}" \
  bash "${SCRIPT_DIR}/evaluate_csp_token_dit_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh" --foreground

echo "============================================================"
echo "Post-FAE patch-8 token-native CSP pipeline completed."
echo "  Token run      : ${TOKEN_CSP_RUN_DIR}"
echo "  Token latents  : ${TOKEN_LATENTS_PATH}"
echo "  Corpus latents : ${TOKEN_CORPUS_LATENTS_PATH}"
echo "============================================================"
