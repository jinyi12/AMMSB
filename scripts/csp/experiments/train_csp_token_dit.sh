#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-3MASB}"
LATENTS_PATH="${LATENTS_PATH:-}"
DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions.npz}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-}"
OUTDIR="${OUTDIR:-results/csp/token_dit/manual_run}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${FAE_CHECKPOINT}" ]]; then
  CMD=(
    "$PYTHON_BIN" -u scripts/csp/train_csp_token_dit_from_fae.py
    --data_path "$DATA_PATH"
    --fae_checkpoint "$FAE_CHECKPOINT"
    --outdir "$OUTDIR"
  )
  if [[ -n "${LATENTS_PATH}" ]]; then
    CMD+=(--latents_path "$LATENTS_PATH")
  fi
  exec "${CMD[@]}" "$@"
fi

if [[ -n "${LATENTS_PATH}" ]]; then
  CMD=(
    "$PYTHON_BIN" -u scripts/csp/train_csp_token_dit.py
    --latents_path "$LATENTS_PATH"
    --source_dataset_path "$DATA_PATH"
    --outdir "$OUTDIR"
  )
  if [[ -n "${FAE_CHECKPOINT}" ]]; then
    CMD+=(--fae_checkpoint "$FAE_CHECKPOINT")
  fi
  exec "${CMD[@]}" "$@"
fi

echo "ERROR: set either FAE_CHECKPOINT for the FAE->token-archive->CSP path, or LATENTS_PATH for token-archive-only training." >&2
exit 1
