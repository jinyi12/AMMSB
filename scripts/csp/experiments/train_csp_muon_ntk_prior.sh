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
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-results/latent_msbm_muon_ntk_prior}"
LATENTS_PATH="${LATENTS_PATH:-${SOURCE_RUN_DIR}/fae_latents.npz}"
DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions.npz}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-}"
OUTPUT_BASE="${OUTPUT_BASE:-results/csp/latent128_muon_ntk_prior}"
PROFILE="${PROFILE:-main}"  # smoke | main | coarse_only
BACKGROUND="${BACKGROUND:-0}"
NOHUP_LOG="${NOHUP_LOG:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
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
    --outdir|--num_steps|--batch_size|--sample_count|--dt0|--lr|--seed|--time_dim|--endpoint_epsilon|--condition_mode|--latents_path|--data_path|--fae_checkpoint)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --hidden)
      EXTRA_ARGS+=("$1" "$2" "$3" "$4")
      shift 4
      ;;
    --nohup-log)
      NOHUP_LOG="$2"
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

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

TIME_DIM="${TIME_DIM:-128}"
DT0="${DT0:-0.01}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"
ENDPOINT_EPSILON="${ENDPOINT_EPSILON:-1e-3}"

case "$PROFILE" in
  smoke)
    OUTDIR="${OUTDIR:-${OUTPUT_BASE}/smoke}"
    NUM_STEPS="${NUM_STEPS:-10}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    SAMPLE_COUNT="${SAMPLE_COUNT:-4}"
    CONDITION_MODE="${CONDITION_MODE:-previous_state}"
    ;;
  main)
    OUTDIR="${OUTDIR:-${OUTPUT_BASE}/main}"
    NUM_STEPS="${NUM_STEPS:-10000}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    SAMPLE_COUNT="${SAMPLE_COUNT:-128}"
    CONDITION_MODE="${CONDITION_MODE:-previous_state}"
    ;;
  coarse_only)
    OUTDIR="${OUTDIR:-${OUTPUT_BASE}/coarse_only}"
    NUM_STEPS="${NUM_STEPS:-10000}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    SAMPLE_COUNT="${SAMPLE_COUNT:-128}"
    CONDITION_MODE="${CONDITION_MODE:-coarse_only}"
    ;;
  *)
    echo "ERROR: --profile must be one of: smoke, main, coarse_only." >&2
    exit 1
    ;;
esac

LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${LOG_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  "${PYTHON_BIN}" -u
)
if [[ -n "${FAE_CHECKPOINT}" ]]; then
  CMD+=(
    scripts/csp/train_csp_from_fae.py
    --data_path "${DATA_PATH}"
    --fae_checkpoint "${FAE_CHECKPOINT}"
    --outdir "${OUTDIR}"
  )
  if [[ -n "${LATENTS_PATH}" ]]; then
    CMD+=(--latents_path "${LATENTS_PATH}")
  fi
else
  if [[ ! -f "${LATENTS_PATH}" ]]; then
    echo "ERROR: missing latent archive: ${LATENTS_PATH}" >&2
    echo "Set FAE_CHECKPOINT to use the FAE->latent-archive->CSP path." >&2
    exit 1
  fi
  CMD+=(
    scripts/csp/train_csp.py
    --latents_path "${LATENTS_PATH}"
    --source_dataset_path "${DATA_PATH}"
    --outdir "${OUTDIR}"
  )
fi

CMD+=(
  --time_dim "${TIME_DIM}"
  --dt0 "${DT0}"
  --lr "${LR}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --sample_count "${SAMPLE_COUNT}"
  --condition_mode "${CONDITION_MODE}"
  --endpoint_epsilon "${ENDPOINT_EPSILON}"
  --seed "${SEED}"
  --hidden 512 512 512
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  CSP training"
echo "  Profile          : ${PROFILE}"
echo "  Environment      : ${ENV_NAME}"
echo "  Dataset          : ${DATA_PATH}"
echo "  FAE checkpoint   : ${FAE_CHECKPOINT:-<from archive / not set>}"
echo "  Latents          : ${LATENTS_PATH}"
echo "  Output           : ${OUTDIR}"
echo "  Condition mode   : ${CONDITION_MODE}"
echo "  Steps / batch    : ${NUM_STEPS} / ${BATCH_SIZE}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

if [[ "${BACKGROUND}" == "1" ]]; then
  if [[ -z "${NOHUP_LOG}" ]]; then
    NOHUP_LOG="${LOG_DIR}/train_${PROFILE}.nohup.log"
  fi
  nohup "${CMD[@]}" >"${NOHUP_LOG}" 2>&1 &
  PID=$!
  echo "Started background run."
  echo "  pid: ${PID}"
  echo "  log: ${NOHUP_LOG}"
  exit 0
fi

exec "${CMD[@]}"
