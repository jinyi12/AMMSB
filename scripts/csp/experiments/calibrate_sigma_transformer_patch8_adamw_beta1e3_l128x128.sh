#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch8_adamw_beta1e3_l128x128_common.sh"
transformer_patch8_prior_defaults

TOKEN_CONDITIONING="${TOKEN_CONDITIONING:-set_conditioned_memory}"
TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}_token_dit_${TOKEN_CONDITIONING}}"
TOKEN_CSP_RUN_DIR="${TOKEN_CSP_RUN_DIR:-${TOKEN_DIT_OUTPUT_BASE}/main}"

LATENTS_PATH="${LATENTS_PATH:-${TOKEN_CSP_RUN_DIR}/fae_token_latents.npz}"
JSON_PATH="${JSON_PATH:-${TOKEN_DIT_OUTPUT_BASE}/calibration/sigma_calibration.json}"
TEXT_PATH="${TEXT_PATH:-${TOKEN_DIT_OUTPUT_BASE}/calibration/sigma_calibration.txt}"
SIGMA_CALIBRATION_METHOD="${SIGMA_CALIBRATION_METHOD:-global_mle}"
KAPPA="${KAPPA:-0.25}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
N_PROBE="${N_PROBE:-512}"
SEED="${SEED:-0}"
ZT_MODE="${ZT_MODE:-archive}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latents_path|--json_path|--text_path|--method|--kappa|--k_neighbors|--n_probe|--seed|--zt_mode)
      case "$1" in
        --latents_path) LATENTS_PATH="$2" ;;
        --json_path) JSON_PATH="$2" ;;
        --text_path) TEXT_PATH="$2" ;;
        --method) SIGMA_CALIBRATION_METHOD="$2" ;;
        --kappa) KAPPA="$2" ;;
        --k_neighbors) K_NEIGHBORS="$2" ;;
        --n_probe) N_PROBE="$2" ;;
        --seed) SEED="$2" ;;
        --zt_mode) ZT_MODE="$2" ;;
      esac
      shift 2
      ;;
    -h|--help)
      sed -n '1,220p' "$0"
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

transformer_patch8_prior_activate_env
transformer_patch8_prior_require_file "${LATENTS_PATH}" "token-native latent archive"
transformer_patch8_prior_mkdirs "$(dirname "${JSON_PATH}")" "$(dirname "${TEXT_PATH}")"

CMD=(
  "${PYTHON_BIN}" scripts/csp/calibrate_sigma.py
  --latents_path "${LATENTS_PATH}"
  --method "${SIGMA_CALIBRATION_METHOD}"
  --zt_mode "${ZT_MODE}"
  --json
)
if [[ "${SIGMA_CALIBRATION_METHOD}" == "knn_legacy" ]]; then
  CMD+=(
    --kappa "${KAPPA}"
    --k_neighbors "${K_NEIGHBORS}"
    --n_probe "${N_PROBE}"
    --seed "${SEED}"
  )
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  Patch-8 token-native CSP sigma calibration"
echo "  Environment      : ${ENV_NAME}"
echo "  Latents          : ${LATENTS_PATH}"
echo "  Output JSON      : ${JSON_PATH}"
echo "  Output text      : ${TEXT_PATH}"
echo "  method           : ${SIGMA_CALIBRATION_METHOD}"
if [[ "${SIGMA_CALIBRATION_METHOD}" == "knn_legacy" ]]; then
  echo "  kappa / k        : ${KAPPA} / ${K_NEIGHBORS}"
  echo "  n_probe / seed   : ${N_PROBE} / ${SEED}"
fi
echo "  zt_mode          : ${ZT_MODE}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

"${CMD[@]}" >"${JSON_PATH}"
REPORT="$(transformer_patch8_prior_render_calibration_report "${JSON_PATH}")"
printf '%s\n' "${REPORT}" | tee "${TEXT_PATH}"
printf 'Saved calibration JSON to %s\n' "${JSON_PATH}"
printf 'Saved calibration text to %s\n' "${TEXT_PATH}"
