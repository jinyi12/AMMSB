#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch8_adamw_beta1e3_l128x128_common.sh"
transformer_patch8_prior_defaults

BACKGROUND="${BACKGROUND:-1}"
TOKEN_CONDITIONING="${TOKEN_CONDITIONING:-set_conditioned_memory}"
TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}_token_dit_${TOKEN_CONDITIONING}}"
LOG_DIR_TOKEN_PIPELINE="${LOG_DIR_TOKEN_PIPELINE:-${TOKEN_DIT_OUTPUT_BASE}/logs}"
NOHUP_LOG="${NOHUP_LOG:-${LOG_DIR_TOKEN_PIPELINE}/post_fae_${EXPERIMENT_NAME}.nohup.log}"
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
    --nohup-log)
      NOHUP_LOG="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

transformer_patch8_prior_mkdirs "${LOG_DIR_TOKEN_PIPELINE}"

CMD=(
  bash
  "${SCRIPT_DIR}/run_post_fae_transformer_patch8_adamw_beta1e3_l128x128.sh"
)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  Post-FAE patch-8 token-native CSP queue"
echo "  Environment      : ${ENV_NAME}"
echo "  Background       : ${BACKGROUND}"
echo "  Log              : ${NOHUP_LOG}"
echo "  Experiment       : ${EXPERIMENT_NAME}"
echo "  Token bridge     : ${TOKEN_CONDITIONING}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

transformer_patch8_prior_launch "${BACKGROUND}" "${NOHUP_LOG}" "${CMD[@]}"
