#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_csp_token_dit_transformer_patch16_adamw_ntk_prior_balanced_l32x128.sh" "$@"
