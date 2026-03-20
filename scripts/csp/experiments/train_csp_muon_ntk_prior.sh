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
SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-data/fae_tran_inclusions.npz}"
OUTPUT_BASE="${OUTPUT_BASE:-results/csp/latent128_muon_ntk_prior}"
PROFILE="${PROFILE:-main}"  # smoke | main | constant
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
    --outdir)
      OUTDIR="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --num_steps)
      NUM_STEPS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --sample_count)
      SAMPLE_COUNT="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --sigma_schedule)
      SIGMA_SCHEDULE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --sigma0)
      SIGMA0="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --decay_rate)
      DECAY_RATE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --dt0)
      DT0="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --lr)
      LR="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --seed)
      SEED="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --time_dim)
      TIME_DIM="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --sigma_reference)
      SIGMA_REFERENCE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --k_neighbors|--k-neighbors)
      K_NEIGHBORS="$2"
      EXTRA_ARGS+=("--k_neighbors" "$2")
      shift 2
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

LATENTS_PATH="${SOURCE_RUN_DIR}/fae_latents.npz"
if [[ ! -f "${LATENTS_PATH}" ]]; then
  echo "ERROR: missing latent archive: ${LATENTS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_DATASET_PATH}" ]]; then
  echo "ERROR: missing source dataset: ${SOURCE_DATASET_PATH}" >&2
  exit 1
fi

SIGMA0="${SIGMA0:-0.15}"
DT0="${DT0:-0.01}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"
TIME_DIM="${TIME_DIM:-128}"
DECAY_RATE="${DECAY_RATE:-1.8}"
SIGMA_REFERENCE="${SIGMA_REFERENCE:-fine_to_coarse}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"

case "$PROFILE" in
  smoke)
    OUTDIR="${OUTDIR:-${OUTPUT_BASE}/smoke}"
    NUM_STEPS="${NUM_STEPS:-10}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    SAMPLE_COUNT="${SAMPLE_COUNT:-4}"
    SIGMA_SCHEDULE="${SIGMA_SCHEDULE:-exp_contract}"
    ;;
  main)
    OUTDIR="${OUTDIR:-${OUTPUT_BASE}/main}"
    NUM_STEPS="${NUM_STEPS:-10000}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    SAMPLE_COUNT="${SAMPLE_COUNT:-128}"
    SIGMA_SCHEDULE="${SIGMA_SCHEDULE:-exp_contract}"
    ;;
  constant)
    OUTDIR="${OUTDIR:-${OUTPUT_BASE}/constant}"
    NUM_STEPS="${NUM_STEPS:-10000}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    SAMPLE_COUNT="${SAMPLE_COUNT:-128}"
    SIGMA_SCHEDULE="${SIGMA_SCHEDULE:-constant}"
    ;;
  *)
    echo "ERROR: --profile must be one of: smoke, main, constant." >&2
    exit 1
    ;;
esac

LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${LOG_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  "${PYTHON_BIN}" -u scripts/csp/train_csp.py
  --source_run_dir "${SOURCE_RUN_DIR}"
  --source_dataset_path "${SOURCE_DATASET_PATH}"
  --outdir "${OUTDIR}"
  --time_dim "${TIME_DIM}"
  --sigma_schedule "${SIGMA_SCHEDULE}"
  --sigma_reference "${SIGMA_REFERENCE}"
  --sigma0 "${SIGMA0}"
  --dt0 "${DT0}"
  --lr "${LR}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --sample_count "${SAMPLE_COUNT}"
  --k_neighbors "${K_NEIGHBORS}"
  --seed "${SEED}"
  --hidden 512 512 512
)

if [[ "${SIGMA_SCHEDULE}" == "exp_contract" ]]; then
  CMD+=(--decay_rate "${DECAY_RATE}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  CSP training"
echo "  Profile          : ${PROFILE}"
echo "  Environment      : ${ENV_NAME}"
echo "  Source run       : ${SOURCE_RUN_DIR}"
echo "  Source dataset   : ${SOURCE_DATASET_PATH}"
echo "  Output           : ${OUTDIR}"
echo "  Sigma schedule   : ${SIGMA_SCHEDULE}"
echo "  Sigma reference  : ${SIGMA_REFERENCE}"
echo "  K neighbors      : ${K_NEIGHBORS}"
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
