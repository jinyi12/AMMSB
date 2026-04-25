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
CSP_RUN_DIR="${CSP_RUN_DIR:-results/csp/latent128_muon_ntk_prior/main}"
PROFILE="${PROFILE:-publication}"  # smoke | publication
RESOURCE_PROFILE="${RESOURCE_PROFILE:-shared_safe}"
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
    --output_dir)
      OUTPUT_DIR="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --n_realizations)
      N_REALIZATIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --n_gt_neighbors)
      N_GT_NEIGHBORS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_rollout_realizations)
      CONDITIONAL_REALIZATIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_rollout_n_test_samples)
      CONDITIONAL_TEST_SAMPLES="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_rollout_k_neighbors)
      CONDITIONAL_K_NEIGHBORS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --fae_checkpoint)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --sample_idx)
      SAMPLE_IDX="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --seed)
      SEED="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_split)
      COARSE_SPLIT="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_selection)
      COARSE_SELECTION="$2"
      EXTRA_ARGS+=("$1" "$2")
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

if [[ ! -f "${CSP_RUN_DIR}/checkpoints/conditional_bridge.eqx" && ! -f "${CSP_RUN_DIR}/checkpoints/csp_drift.eqx" ]]; then
  echo "ERROR: missing CSP checkpoint under ${CSP_RUN_DIR}" >&2
  exit 1
fi

case "$PROFILE" in
  smoke)
    N_REALIZATIONS="${N_REALIZATIONS:-32}"
    N_GT_NEIGHBORS="${N_GT_NEIGHBORS:-32}"
    CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-8}"
    CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-8}"
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-16}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CSP_RUN_DIR}/eval/smoke_n32}"
    ;;
  publication)
    N_REALIZATIONS="${N_REALIZATIONS:-512}"
    N_GT_NEIGHBORS="${N_GT_NEIGHBORS:-512}"
    CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-200}"
    CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-50}"
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-16}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CSP_RUN_DIR}/eval/n512}"
    ;;
  *)
    echo "ERROR: --profile must be one of: smoke, publication." >&2
    exit 1
    ;;
esac

SAMPLE_IDX="${SAMPLE_IDX:-0}"
COARSE_SPLIT="${COARSE_SPLIT:-train}"
COARSE_SELECTION="${COARSE_SELECTION:-random}"
SEED="${SEED:-0}"
LOG_DIR="${CSP_RUN_DIR}/eval/logs"
mkdir -p "${LOG_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  "${PYTHON_BIN}" -u scripts/csp/evaluate_csp.py
  --run_dir "${CSP_RUN_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --n_realizations "${N_REALIZATIONS}"
  --n_gt_neighbors "${N_GT_NEIGHBORS}"
  --sample_idx "${SAMPLE_IDX}"
  --seed "${SEED}"
  --coarse_split "${COARSE_SPLIT}"
  --coarse_selection "${COARSE_SELECTION}"
  --resource_profile "${RESOURCE_PROFILE}"
  --conditional_rollout_realizations "${CONDITIONAL_REALIZATIONS}"
  --conditional_rollout_n_test_samples "${CONDITIONAL_TEST_SAMPLES}"
  --conditional_rollout_k_neighbors "${CONDITIONAL_K_NEIGHBORS}"
)

if [[ -n "${CPU_THREADS:-}" ]]; then
  CMD+=(--cpu_threads "${CPU_THREADS}")
fi
if [[ -n "${CPU_CORES:-}" ]]; then
  CMD+=(--cpu_cores "${CPU_CORES}")
fi
if [[ -n "${MEMORY_BUDGET_GB:-}" ]]; then
  CMD+=(--memory_budget_gb "${MEMORY_BUDGET_GB}")
fi
if [[ -n "${CONDITION_CHUNK_SIZE:-}" ]]; then
  CMD+=(--condition_chunk_size "${CONDITION_CHUNK_SIZE}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "  CSP evaluation"
echo "  Profile          : ${PROFILE}"
echo "  Environment      : ${ENV_NAME}"
echo "  CSP run          : ${CSP_RUN_DIR}"
echo "  Output           : ${OUTPUT_DIR}"
echo "  Realizations     : ${N_REALIZATIONS}"
echo "  GT neighbors     : ${N_GT_NEIGHBORS}"
echo "  Conditional rollout n : ${CONDITIONAL_REALIZATIONS}"
echo "  Conditional rollout m : ${CONDITIONAL_TEST_SAMPLES}"
echo "  Conditional rollout k : ${CONDITIONAL_K_NEIGHBORS}"
echo "  Resource profile  : ${RESOURCE_PROFILE}"
echo "  Coarse split     : ${COARSE_SPLIT}"
echo "  Coarse selection : ${COARSE_SELECTION}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args       : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

if [[ "${BACKGROUND}" == "1" ]]; then
  if [[ -z "${NOHUP_LOG}" ]]; then
    NOHUP_LOG="${LOG_DIR}/evaluate_${PROFILE}.nohup.log"
  fi
  nohup "${CMD[@]}" >"${NOHUP_LOG}" 2>&1 &
  PID=$!
  echo "Started background evaluation."
  echo "  pid: ${PID}"
  echo "  log: ${NOHUP_LOG}"
  exit 0
fi

exec "${CMD[@]}"
