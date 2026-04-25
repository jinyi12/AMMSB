#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/fae_film_adamw_ntk_prior_latent128_minmax_common.sh"
fae_film_adamw_ntk_prior_latent128_minmax_defaults

BACKGROUND="${BACKGROUND:-1}"
NOHUP_LOG="${NOHUP_LOG:-${LOG_DIR}/full_${EXPERIMENT_NAME}.nohup.log}"
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

fae_film_adamw_ntk_prior_latent128_minmax_mkdirs "${LOG_DIR}"

CMD=(
  bash
  "${SCRIPT_DIR}/run_full_fae_film_adamw_ntk_prior_latent128_minmax.sh"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  Full FiLM minmax -> CSP queue"
echo "  Environment      : ${ENV_NAME}"
echo "  Background       : ${BACKGROUND}"
echo "  Log              : ${NOHUP_LOG}"
echo "  Experiment       : ${EXPERIMENT_NAME}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

fae_film_adamw_ntk_prior_latent128_minmax_launch "${BACKGROUND}" "${NOHUP_LOG}" "${CMD[@]}"
