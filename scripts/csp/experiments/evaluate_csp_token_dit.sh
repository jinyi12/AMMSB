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
CSP_RUN_DIR="${CSP_RUN_DIR:-results/csp/token_dit/manual_run}"
PROFILE="${PROFILE:-publication}"  # smoke | light | publication
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
    --conditional_rollout_n_plot_conditions)
      CONDITIONAL_N_PLOT_CONDITIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --skip_conditional_rollout_reports)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --coarse_eval_mode)
      COARSE_EVAL_MODE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_eval_conditions)
      COARSE_EVAL_CONDITIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_eval_realizations)
      COARSE_EVAL_REALIZATIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditioned_global_conditions)
      CONDITIONED_GLOBAL_CONDITIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditioned_global_realizations)
      CONDITIONED_GLOBAL_REALIZATIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --decode_batch_size)
      DECODE_BATCH_SIZE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_decode_batch_size)
      COARSE_DECODE_BATCH_SIZE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --cache_sampling_device)
      CACHE_SAMPLING_DEVICE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --cache_decode_device)
      CACHE_DECODE_DEVICE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --cache_decode_point_batch_size)
      CACHE_DECODE_POINT_BATCH_SIZE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_sampling_device)
      COARSE_SAMPLING_DEVICE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_decode_device)
      COARSE_DECODE_DEVICE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --coarse_decode_point_batch_size)
      COARSE_DECODE_POINT_BATCH_SIZE="$2"
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
      sed -n '1,260p' "$0"
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

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${CSP_RUN_DIR}/checkpoints/conditional_bridge_token_dit.eqx" ]]; then
  echo "ERROR: missing token-native CSP checkpoint under ${CSP_RUN_DIR}" >&2
  exit 1
fi

case "$PROFILE" in
  smoke)
    N_REALIZATIONS="${N_REALIZATIONS:-32}"
    N_GT_NEIGHBORS="${N_GT_NEIGHBORS:-32}"
    CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-8}"
    CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-8}"
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-16}"
    CONDITIONAL_N_PLOT_CONDITIONS="${CONDITIONAL_N_PLOT_CONDITIONS:-0}"
    COARSE_EVAL_MODE="${COARSE_EVAL_MODE:-global}"
    COARSE_EVAL_CONDITIONS="${COARSE_EVAL_CONDITIONS:-8}"
    COARSE_EVAL_REALIZATIONS="${COARSE_EVAL_REALIZATIONS:-8}"
    CONDITIONED_GLOBAL_CONDITIONS="${CONDITIONED_GLOBAL_CONDITIONS:-8}"
    CONDITIONED_GLOBAL_REALIZATIONS="${CONDITIONED_GLOBAL_REALIZATIONS:-8}"
    DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"
    COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-128}"
    CACHE_SAMPLING_DEVICE="${CACHE_SAMPLING_DEVICE:-auto}"
    CACHE_DECODE_DEVICE="${CACHE_DECODE_DEVICE:-auto}"
    COARSE_SAMPLING_DEVICE="${COARSE_SAMPLING_DEVICE:-auto}"
    COARSE_DECODE_DEVICE="${COARSE_DECODE_DEVICE:-auto}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CSP_RUN_DIR}/eval/smoke_n32}"
    ;;
  light)
    N_REALIZATIONS="${N_REALIZATIONS:-128}"
    N_GT_NEIGHBORS="${N_GT_NEIGHBORS:-128}"
    CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-64}"
    CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-16}"
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-16}"
    CONDITIONAL_N_PLOT_CONDITIONS="${CONDITIONAL_N_PLOT_CONDITIONS:-0}"
    COARSE_EVAL_MODE="${COARSE_EVAL_MODE:-global}"
    COARSE_EVAL_CONDITIONS="${COARSE_EVAL_CONDITIONS:-8}"
    COARSE_EVAL_REALIZATIONS="${COARSE_EVAL_REALIZATIONS:-16}"
    CONDITIONED_GLOBAL_CONDITIONS="${CONDITIONED_GLOBAL_CONDITIONS:-8}"
    CONDITIONED_GLOBAL_REALIZATIONS="${CONDITIONED_GLOBAL_REALIZATIONS:-16}"
    DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"
    COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-128}"
    CACHE_SAMPLING_DEVICE="${CACHE_SAMPLING_DEVICE:-auto}"
    CACHE_DECODE_DEVICE="${CACHE_DECODE_DEVICE:-auto}"
    COARSE_SAMPLING_DEVICE="${COARSE_SAMPLING_DEVICE:-auto}"
    COARSE_DECODE_DEVICE="${COARSE_DECODE_DEVICE:-auto}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CSP_RUN_DIR}/eval/light_n128}"
    ;;
  publication)
    N_REALIZATIONS="${N_REALIZATIONS:-512}"
    N_GT_NEIGHBORS="${N_GT_NEIGHBORS:-512}"
    CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-200}"
    CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-50}"
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-16}"
    CONDITIONAL_N_PLOT_CONDITIONS="${CONDITIONAL_N_PLOT_CONDITIONS:-5}"
    COARSE_EVAL_MODE="${COARSE_EVAL_MODE:-both}"
    COARSE_EVAL_CONDITIONS="${COARSE_EVAL_CONDITIONS:-16}"
    COARSE_EVAL_REALIZATIONS="${COARSE_EVAL_REALIZATIONS:-32}"
    CONDITIONED_GLOBAL_CONDITIONS="${CONDITIONED_GLOBAL_CONDITIONS:-16}"
    CONDITIONED_GLOBAL_REALIZATIONS="${CONDITIONED_GLOBAL_REALIZATIONS:-32}"
    DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-16}"
    COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-8}"
    CACHE_SAMPLING_DEVICE="${CACHE_SAMPLING_DEVICE:-auto}"
    CACHE_DECODE_DEVICE="${CACHE_DECODE_DEVICE:-auto}"
    COARSE_SAMPLING_DEVICE="${COARSE_SAMPLING_DEVICE:-auto}"
    COARSE_DECODE_DEVICE="${COARSE_DECODE_DEVICE:-auto}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CSP_RUN_DIR}/eval/n512}"
    ;;
  *)
    echo "ERROR: --profile must be one of: smoke, light, publication." >&2
    exit 1
    ;;
esac

SAMPLE_IDX="${SAMPLE_IDX:-0}"
COARSE_SPLIT="${COARSE_SPLIT:-train}"
COARSE_SELECTION="${COARSE_SELECTION:-random}"
SEED="${SEED:-0}"
LOG_DIR="${CSP_RUN_DIR}/eval/logs"
mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" -u scripts/csp/evaluate_csp_token_dit.py
  --run_dir "${CSP_RUN_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --n_realizations "${N_REALIZATIONS}"
  --n_gt_neighbors "${N_GT_NEIGHBORS}"
  --sample_idx "${SAMPLE_IDX}"
  --seed "${SEED}"
  --coarse_split "${COARSE_SPLIT}"
  --coarse_selection "${COARSE_SELECTION}"
  --decode_batch_size "${DECODE_BATCH_SIZE}"
  --coarse_eval_mode "${COARSE_EVAL_MODE}"
  --coarse_eval_conditions "${COARSE_EVAL_CONDITIONS}"
  --coarse_eval_realizations "${COARSE_EVAL_REALIZATIONS}"
  --conditioned_global_conditions "${CONDITIONED_GLOBAL_CONDITIONS}"
  --conditioned_global_realizations "${CONDITIONED_GLOBAL_REALIZATIONS}"
  --coarse_decode_batch_size "${COARSE_DECODE_BATCH_SIZE}"
  --cache_sampling_device "${CACHE_SAMPLING_DEVICE}"
  --cache_decode_device "${CACHE_DECODE_DEVICE}"
  --coarse_sampling_device "${COARSE_SAMPLING_DEVICE}"
  --coarse_decode_device "${COARSE_DECODE_DEVICE}"
  --resource_profile "${RESOURCE_PROFILE}"
  --conditional_rollout_realizations "${CONDITIONAL_REALIZATIONS}"
  --conditional_rollout_n_test_samples "${CONDITIONAL_TEST_SAMPLES}"
  --conditional_rollout_k_neighbors "${CONDITIONAL_K_NEIGHBORS}"
  --conditional_rollout_n_plot_conditions "${CONDITIONAL_N_PLOT_CONDITIONS}"
)

if [[ -n "${CACHE_DECODE_POINT_BATCH_SIZE:-}" ]]; then
  CMD+=(--cache_decode_point_batch_size "${CACHE_DECODE_POINT_BATCH_SIZE}")
fi
if [[ -n "${COARSE_DECODE_POINT_BATCH_SIZE:-}" ]]; then
  CMD+=(--coarse_decode_point_batch_size "${COARSE_DECODE_POINT_BATCH_SIZE}")
fi
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
echo "  Token-native CSP evaluation"
echo "  Profile              : ${PROFILE}"
echo "  Environment          : ${ENV_NAME}"
echo "  CSP run              : ${CSP_RUN_DIR}"
echo "  Output               : ${OUTPUT_DIR}"
echo "  Realizations         : ${N_REALIZATIONS}"
echo "  GT neighbors         : ${N_GT_NEIGHBORS}"
echo "  Coarse mode          : ${COARSE_EVAL_MODE}"
echo "  Coarse conds         : ${COARSE_EVAL_CONDITIONS}"
echo "  Coarse realizations  : ${COARSE_EVAL_REALIZATIONS}"
echo "  Global conds         : ${CONDITIONED_GLOBAL_CONDITIONS}"
echo "  Global realizations  : ${CONDITIONED_GLOBAL_REALIZATIONS}"
echo "  Cache devices        : sample=${CACHE_SAMPLING_DEVICE} decode=${CACHE_DECODE_DEVICE}"
echo "  Coarse devices       : sample=${COARSE_SAMPLING_DEVICE} decode=${COARSE_DECODE_DEVICE}"
echo "  Resource profile     : ${RESOURCE_PROFILE}"
echo "  Conditional rollout n : ${CONDITIONAL_REALIZATIONS}"
echo "  Conditional rollout m : ${CONDITIONAL_TEST_SAMPLES}"
echo "  Conditional rollout k : ${CONDITIONAL_K_NEIGHBORS}"
echo "  Conditional rollout p : ${CONDITIONAL_N_PLOT_CONDITIONS}"
echo "  Coarse split         : ${COARSE_SPLIT}"
echo "  Coarse selection     : ${COARSE_SELECTION}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args           : ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

if [[ "${BACKGROUND}" == "1" ]]; then
  if [[ -z "${NOHUP_LOG}" ]]; then
    NOHUP_LOG="${LOG_DIR}/evaluate_token_${PROFILE}.nohup.log"
  fi
  nohup "${CMD[@]}" >"${NOHUP_LOG}" 2>&1 &
  PID=$!
  echo "Started background evaluation."
  echo "  pid: ${PID}"
  echo "  log: ${NOHUP_LOG}"
  exit 0
fi

exec "${CMD[@]}"
