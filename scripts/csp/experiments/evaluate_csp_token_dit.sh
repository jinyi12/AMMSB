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
BACKGROUND="${BACKGROUND:-0}"
NOHUP_LOG="${NOHUP_LOG:-}"
CONDITIONAL_CORPUS_LATENTS_PATH="${CONDITIONAL_CORPUS_LATENTS_PATH:-data/corpus_latents_ntk_prior.npz}"
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
    --conditional_realizations)
      CONDITIONAL_REALIZATIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_n_test_samples)
      CONDITIONAL_TEST_SAMPLES="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_k_neighbors)
      CONDITIONAL_K_NEIGHBORS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_max_corpus_samples)
      CONDITIONAL_MAX_CORPUS_SAMPLES="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_n_plot_conditions)
      CONDITIONAL_N_PLOT_CONDITIONS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_plot_value_budget)
      CONDITIONAL_PLOT_VALUE_BUDGET="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --conditional_corpus_latents_path)
      CONDITIONAL_CORPUS_LATENTS_PATH="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
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
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-32}"
    CONDITIONAL_MAX_CORPUS_SAMPLES="${CONDITIONAL_MAX_CORPUS_SAMPLES:-512}"
    CONDITIONAL_N_PLOT_CONDITIONS="${CONDITIONAL_N_PLOT_CONDITIONS:-0}"
    CONDITIONAL_PLOT_VALUE_BUDGET="${CONDITIONAL_PLOT_VALUE_BUDGET:-2000}"
    COARSE_EVAL_MODE="${COARSE_EVAL_MODE:-global}"
    COARSE_EVAL_CONDITIONS="${COARSE_EVAL_CONDITIONS:-8}"
    COARSE_EVAL_REALIZATIONS="${COARSE_EVAL_REALIZATIONS:-8}"
    CONDITIONED_GLOBAL_CONDITIONS="${CONDITIONED_GLOBAL_CONDITIONS:-8}"
    CONDITIONED_GLOBAL_REALIZATIONS="${CONDITIONED_GLOBAL_REALIZATIONS:-8}"
    DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"
    COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-128}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CSP_RUN_DIR}/eval/smoke_n32}"
    ;;
  light)
    N_REALIZATIONS="${N_REALIZATIONS:-128}"
    N_GT_NEIGHBORS="${N_GT_NEIGHBORS:-128}"
    CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-64}"
    CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-16}"
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-64}"
    CONDITIONAL_MAX_CORPUS_SAMPLES="${CONDITIONAL_MAX_CORPUS_SAMPLES:-2000}"
    CONDITIONAL_N_PLOT_CONDITIONS="${CONDITIONAL_N_PLOT_CONDITIONS:-0}"
    CONDITIONAL_PLOT_VALUE_BUDGET="${CONDITIONAL_PLOT_VALUE_BUDGET:-4000}"
    COARSE_EVAL_MODE="${COARSE_EVAL_MODE:-global}"
    COARSE_EVAL_CONDITIONS="${COARSE_EVAL_CONDITIONS:-8}"
    COARSE_EVAL_REALIZATIONS="${COARSE_EVAL_REALIZATIONS:-16}"
    CONDITIONED_GLOBAL_CONDITIONS="${CONDITIONED_GLOBAL_CONDITIONS:-8}"
    CONDITIONED_GLOBAL_REALIZATIONS="${CONDITIONED_GLOBAL_REALIZATIONS:-16}"
    DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"
    COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-128}"
    OUTPUT_DIR="${OUTPUT_DIR:-${CSP_RUN_DIR}/eval/light_n128}"
    ;;
  publication)
    N_REALIZATIONS="${N_REALIZATIONS:-512}"
    N_GT_NEIGHBORS="${N_GT_NEIGHBORS:-512}"
    CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-200}"
    CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-50}"
    CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-200}"
    CONDITIONAL_MAX_CORPUS_SAMPLES="${CONDITIONAL_MAX_CORPUS_SAMPLES:-}"
    CONDITIONAL_N_PLOT_CONDITIONS="${CONDITIONAL_N_PLOT_CONDITIONS:-5}"
    CONDITIONAL_PLOT_VALUE_BUDGET="${CONDITIONAL_PLOT_VALUE_BUDGET:-20000}"
    COARSE_EVAL_MODE="${COARSE_EVAL_MODE:-both}"
    COARSE_EVAL_CONDITIONS="${COARSE_EVAL_CONDITIONS:-16}"
    COARSE_EVAL_REALIZATIONS="${COARSE_EVAL_REALIZATIONS:-32}"
    CONDITIONED_GLOBAL_CONDITIONS="${CONDITIONED_GLOBAL_CONDITIONS:-16}"
    CONDITIONED_GLOBAL_REALIZATIONS="${CONDITIONED_GLOBAL_REALIZATIONS:-32}"
    DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"
    COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-256}"
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
PYTHON_BIN="${PYTHON_BIN:-python}"

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
  --conditional_corpus_latents_path "${CONDITIONAL_CORPUS_LATENTS_PATH}"
  --conditional_realizations "${CONDITIONAL_REALIZATIONS}"
  --conditional_n_test_samples "${CONDITIONAL_TEST_SAMPLES}"
  --conditional_k_neighbors "${CONDITIONAL_K_NEIGHBORS}"
  --conditional_n_plot_conditions "${CONDITIONAL_N_PLOT_CONDITIONS}"
  --conditional_plot_value_budget "${CONDITIONAL_PLOT_VALUE_BUDGET}"
)

if [[ -n "${CONDITIONAL_MAX_CORPUS_SAMPLES}" ]]; then
  CMD+=(--conditional_max_corpus_samples "${CONDITIONAL_MAX_CORPUS_SAMPLES}")
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
echo "  Cond realizations    : ${CONDITIONAL_REALIZATIONS}"
echo "  Cond test samples    : ${CONDITIONAL_TEST_SAMPLES}"
echo "  Cond k-neighbors     : ${CONDITIONAL_K_NEIGHBORS}"
echo "  Cond corpus cap      : ${CONDITIONAL_MAX_CORPUS_SAMPLES:-full}"
echo "  Cond plot conditions : ${CONDITIONAL_N_PLOT_CONDITIONS}"
echo "  Cond corpus          : ${CONDITIONAL_CORPUS_LATENTS_PATH}"
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
