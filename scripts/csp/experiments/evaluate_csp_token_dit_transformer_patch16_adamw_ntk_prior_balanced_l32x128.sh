#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transformer_patch16_adamw_ntk_prior_balanced_l32x128_common.sh"
transformer_patch16_defaults

TOKEN_CONDITIONING="${TOKEN_CONDITIONING:-set_conditioned_memory}"
TOKEN_DIT_OUTPUT_BASE="${TOKEN_DIT_OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}_token_dit_${TOKEN_CONDITIONING}}"
CSP_RUN_DIR="${CSP_RUN_DIR:-${TOKEN_DIT_OUTPUT_BASE}/main}"
CONDITIONAL_CORPUS_LATENTS_PATH="${CONDITIONAL_CORPUS_LATENTS_PATH:-${CSP_RUN_DIR}/corpus_latents.npz}"
PROFILE="${PROFILE:-publication}"
COARSE_SPLIT="${COARSE_SPLIT:-test}"

transformer_patch16_require_file "${CSP_RUN_DIR}/checkpoints/conditional_bridge_token_dit.eqx" "token-native CSP checkpoint"
transformer_patch16_require_file "${CONDITIONAL_CORPUS_LATENTS_PATH}" "conditional corpus latent archive"

export ENV_NAME="${ENV_NAME}"
export PYTHON_BIN="${PYTHON_BIN}"
export CSP_RUN_DIR
export CONDITIONAL_CORPUS_LATENTS_PATH
export PROFILE
export COARSE_SPLIT

exec bash scripts/csp/experiments/evaluate_csp_token_dit.sh "$@"
