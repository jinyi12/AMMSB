#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch16_adamw_ntk_prior_balanced_l64x32_common.sh"
transformer_patch16_defaults

RUN_TOKEN="${RUN_TOKEN:-1}"
TOKEN_PROFILE="${TOKEN_PROFILE:-publication}"
TOKEN_CONDITIONING="${TOKEN_CONDITIONING:-set_conditioned_memory}"

TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}_token_dit_${TOKEN_CONDITIONING}}"
TOKEN_CSP_RUN_DIR="${TOKEN_CSP_RUN_DIR:-${TOKEN_DIT_OUTPUT_BASE}/main}"
TOKEN_LATENTS_PATH="${TOKEN_LATENTS_PATH:-${TOKEN_CSP_RUN_DIR}/fae_token_latents.npz}"
TOKEN_MANIFEST_PATH="${TOKEN_MANIFEST_PATH:-${TOKEN_CSP_RUN_DIR}/config/fae_token_latents_manifest.json}"
TOKEN_CORPUS_LATENTS_PATH="${TOKEN_CORPUS_LATENTS_PATH:-${TOKEN_CSP_RUN_DIR}/corpus_latents.npz}"

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

transformer_patch16_activate_env

run_step() {
  local label="$1"
  shift
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label}"
  echo "============================================================"
  "$@"
}

if [[ "${RUN_TOKEN}" == "1" ]]; then
  run_step "Encode token-native FAE train/test latents" \
    "${PYTHON_BIN}" -u scripts/csp/encode_fae_token_latents.py \
      --data_path "${TRAIN_DATA_PATH_DEFAULT}" \
      --fae_checkpoint "${FAE_CHECKPOINT_DEFAULT}" \
      --output_path "${TOKEN_LATENTS_PATH}" \
      --manifest_path "${TOKEN_MANIFEST_PATH}" \
      --encode_batch_size 64 \
      --train_ratio 0.8 \
      --time_dist_mode uniform \
      --t_scale 1.0

  run_step "Calibrate sigma from token-native latents" \
    env LATENTS_PATH="${TOKEN_LATENTS_PATH}" \
    bash "${SCRIPT_DIR}/calibrate_sigma_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh"

  run_step "Train token-native CSP DiT bridge" \
    env TOKEN_CONDITIONING="${TOKEN_CONDITIONING}" \
        TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE}" \
        LATENTS_PATH="${TOKEN_LATENTS_PATH}" \
        MANIFEST_PATH="${TOKEN_MANIFEST_PATH}" \
    bash "${SCRIPT_DIR}/train_csp_token_dit_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh" --foreground

  run_step "Encode token-native corpus latents for conditional evaluation" \
    "${PYTHON_BIN}" -u scripts/fae/tran_evaluation/encode_corpus.py \
      --corpus_path "${CORPUS_DATA_PATH_DEFAULT}" \
      --run_dir "${TOKEN_CSP_RUN_DIR}" \
      --output_path "${TOKEN_CORPUS_LATENTS_PATH}" \
      --batch_size 64

  run_step "Evaluate token-native CSP DiT bridge" \
    env TOKEN_CONDITIONING="${TOKEN_CONDITIONING}" \
        CSP_RUN_DIR="${TOKEN_CSP_RUN_DIR}" \
        CONDITIONAL_CORPUS_LATENTS_PATH="${TOKEN_CORPUS_LATENTS_PATH}" \
        PROFILE="${TOKEN_PROFILE}" \
    bash "${SCRIPT_DIR}/evaluate_csp_token_dit_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh" --foreground
fi

echo "============================================================"
echo "Post-FAE transformer token-native CSP pipeline completed."
echo "  Token run      : ${TOKEN_CSP_RUN_DIR}"
echo "  Token latents  : ${TOKEN_LATENTS_PATH}"
echo "  Corpus latents : ${TOKEN_CORPUS_LATENTS_PATH}"
echo "============================================================"
