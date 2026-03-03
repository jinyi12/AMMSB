#!/usr/bin/env bash
# Train latent MSBM from an NTK+Prior FAE checkpoint.
# Select the source autoencoder with --optimizer {adam,muon}.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer git root so this keeps working if the script is moved again.
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
cd "$REPO_ROOT"

OPTIMIZER="${OPTIMIZER:-adam}"
BACKGROUND="${BACKGROUND:-0}"
NOHUP_LOG="${NOHUP_LOG:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --optimizer)
      OPTIMIZER="$2"
      shift 2
      ;;
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
    -h|--help)
      echo "Usage: bash scripts/fae/experiments/latentmsbm/train_latent_msbm_ntk.sh [--optimizer adam|muon] [--background] [--nohup-log <path>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Use --help for usage." >&2
      exit 1
      ;;
  esac
done

case "$OPTIMIZER" in
  adam)
    DEFAULT_CKPT="results/fae_film_adam_ntk_99pct/run_2hnr5shv/checkpoints/state.pkl"
    DEFAULT_RUN_NAME="adam_ntk"
    DEFAULT_OUTDIR="latent_msbm_adam_ntk"
    ;;
  muon)
    DEFAULT_CKPT="results/fae_film_muon_ntk_99pct/run_tug7ucuw/checkpoints/state.pkl"
    DEFAULT_RUN_NAME="muon_ntk"
    DEFAULT_OUTDIR="latent_msbm_muon_ntk"
    ;;
  *)
    echo "ERROR: optimizer must be 'adam' or 'muon' (got '${OPTIMIZER}')." >&2
    exit 1
    ;;
esac

DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions.npz}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-${DEFAULT_CKPT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WANDB_MODE="${WANDB_MODE:-online}"          # online|offline|disabled
WANDB_PROJECT="${WANDB_PROJECT:-fae_latent_msbm}"
WANDB_GROUP="${WANDB_GROUP:-latent_msbm_ntk}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUTDIR="${OUTDIR:-${DEFAULT_OUTDIR}}"
WANDB_TAGS="${WANDB_TAGS:-latent_msbm,ntk,${OPTIMIZER}}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: missing dataset: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${FAE_CHECKPOINT}" ]]; then
  echo "ERROR: missing FAE checkpoint: ${FAE_CHECKPOINT}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" scripts/fae/fae_naive/train_latent_msbm.py
  --data_path "${DATA_PATH}" \
  --fae_checkpoint "${FAE_CHECKPOINT}" \
  --decode_mode standard \
  --policy_arch augmented_mlp \
  --hidden 512 512 512 \
  --time_dim 128 \
  --var_schedule exp_contract \
  --var_decay_rate 2.0 \
  --auto_var_decay \
  --auto_var \
  --auto_var_kappa 0.1 \
  --auto_var_max 1.0 \
  --var 0.0625 \
  --num_stages 50 \
  --num_itr 10000 \
  --eval_n_samples 8 \
  --eval_interval_stages 10 \
  --eval_metrics_interval_stages 10 \
  --lr 1e-4 \
  --n_decode 128 \
  --wandb_mode "${WANDB_MODE}" \
  --project "${WANDB_PROJECT}" \
  --group "${WANDB_GROUP}" \
  --run_name "${RUN_NAME}" \
  --wandb_tags "${WANDB_TAGS}" \
  --outdir "${OUTDIR}"
)

if [[ "${BACKGROUND}" == "1" ]]; then
  if [[ -z "${NOHUP_LOG}" ]]; then
    NOHUP_LOG="results/${OUTDIR}/train_latent_msbm.nohup.log"
  fi
  mkdir -p "$(dirname "${NOHUP_LOG}")"
  nohup "${CMD[@]}" >"${NOHUP_LOG}" 2>&1 &
  PID=$!
  echo "Started background run."
  echo "  pid: ${PID}"
  echo "  log: ${NOHUP_LOG}"
  echo "  optimizer: ${OPTIMIZER}"
  exit 0
fi

exec "${CMD[@]}"
