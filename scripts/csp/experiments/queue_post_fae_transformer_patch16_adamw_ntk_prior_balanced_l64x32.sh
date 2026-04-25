#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch16_adamw_ntk_prior_balanced_l64x32_common.sh"
transformer_patch16_defaults

BACKGROUND="${BACKGROUND:-1}"
NOHUP_LOG="${NOHUP_LOG:-${LOG_DIR}/post_fae_transformer_patch16_adamw_ntk_prior_balanced_l64x32.nohup.log}"
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

transformer_patch16_mkdirs "${LOG_DIR}"

CMD=(
  bash
  "${SCRIPT_DIR}/run_post_fae_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  Post-FAE transformer CSP queue"
echo "  Environment      : ${ENV_NAME}"
echo "  Background       : ${BACKGROUND}"
echo "  Log              : ${NOHUP_LOG}"
echo "  Experiment       : ${EXPERIMENT_NAME}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

transformer_patch16_launch "${BACKGROUND}" "${NOHUP_LOG}" "${CMD[@]}"
