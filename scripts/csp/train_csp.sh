#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-3MASB}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/latent_msbm_muon_ntk_prior}"
SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-data/fae_tran_inclusions.npz}"
OUTDIR="${OUTDIR:-results/csp/manual_run}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -u scripts/csp/train_csp.py \
  --source_run_dir "$SOURCE_RUN_DIR" \
  --source_dataset_path "$SOURCE_DATASET_PATH" \
  --outdir "$OUTDIR" \
  "$@"
