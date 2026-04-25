#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/fae_film_adamw_ntk_prior_latent128_minmax_common.sh"
fae_film_adamw_ntk_prior_latent128_minmax_defaults

RUN_FAE_TRAIN="${RUN_FAE_TRAIN:-1}"
RUN_LATENT_ENCODE="${RUN_LATENT_ENCODE:-1}"
RUN_SIGMA_CALIBRATION="${RUN_SIGMA_CALIBRATION:-1}"
RUN_CSP_TRAIN="${RUN_CSP_TRAIN:-1}"
RUN_CORPUS_ENCODE="${RUN_CORPUS_ENCODE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${TRAIN_DATA_PATH_DEFAULT}}"
CORPUS_DATA_PATH="${CORPUS_DATA_PATH:-${CORPUS_DATA_PATH_DEFAULT}}"
FAE_CHECKPOINT="${FAE_CHECKPOINT:-}"
CSP_RUN_DIR="${CSP_RUN_DIR:-${RUN_DIR_DEFAULT}}"
LATENTS_PATH="${LATENTS_PATH:-${LATENTS_PATH_DEFAULT}}"
LATENTS_MANIFEST_PATH="${LATENTS_MANIFEST_PATH:-${LATENTS_MANIFEST_PATH_DEFAULT}}"
CORPUS_LATENTS_PATH="${CORPUS_LATENTS_PATH:-${CORPUS_LATENTS_PATH_DEFAULT}}"
CONDITIONAL_CORPUS_LATENTS_PATH="${CONDITIONAL_CORPUS_LATENTS_PATH:-${CONDITIONAL_CORPUS_LATENTS_PATH_DEFAULT}}"
CALIBRATION_JSON="${CALIBRATION_JSON:-${CALIBRATION_JSON_DEFAULT}}"
CALIBRATION_TEXT="${CALIBRATION_TEXT:-${CALIBRATION_TEXT_DEFAULT}}"

FAE_LOG_FILE="${FAE_LOG_FILE:-${LOG_DIR}/train_fae.log}"
FAE_MAX_STEPS="${FAE_MAX_STEPS:-50000}"
FAE_BATCH_SIZE="${FAE_BATCH_SIZE:-32}"
FAE_EVAL_INTERVAL="${FAE_EVAL_INTERVAL:-25}"
FAE_EVAL_N_BATCHES="${FAE_EVAL_N_BATCHES:-5}"
FAE_VIS_INTERVAL="${FAE_VIS_INTERVAL:-5}"
FAE_LR="${FAE_LR:-1e-3}"
FAE_WEIGHT_DECAY="${FAE_WEIGHT_DECAY:-1e-5}"
FAE_CHECKPOINT_PREFERENCE="${FAE_CHECKPOINT_PREFERENCE:-best_then_state}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
CORPUS_BATCH_SIZE="${CORPUS_BATCH_SIZE:-64}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
ARCHIVE_ZT_MODE="${ARCHIVE_ZT_MODE:-retained_times}"
TIME_DIST_MODE="${TIME_DIST_MODE:-zt}"
T_SCALE="${T_SCALE:-1.0}"

SIGMA_CALIBRATION_METHOD="${SIGMA_CALIBRATION_METHOD:-global_mle}"
KAPPA="${KAPPA:-0.25}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"
N_PROBE="${N_PROBE:-512}"
SIGMA_ZT_MODE="${SIGMA_ZT_MODE:-archive}"
SIGMA="${SIGMA:-}"

TIME_DIM="${TIME_DIM:-128}"
DT0="${DT0:-0.01}"
LR="${LR:-1e-4}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CONDITION_MODE="${CONDITION_MODE:-previous_state}"
ENDPOINT_EPSILON="${ENDPOINT_EPSILON:-0.01}"
SAMPLE_COUNT="${SAMPLE_COUNT:-128}"
LOG_EVERY="${LOG_EVERY:-0}"
SEED="${SEED:-0}"
HIDDEN_DIMS="${HIDDEN_DIMS:-512 512 512}"

EVAL_PROFILE="${EVAL_PROFILE:-publication}"  # smoke | publication
SAMPLE_IDX="${SAMPLE_IDX:-0}"
COARSE_SPLIT="${COARSE_SPLIT:-train}"
COARSE_SELECTION="${COARSE_SELECTION:-random}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-64}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fae-train)
      RUN_FAE_TRAIN="0"
      shift
      ;;
    --skip-latent-encode)
      RUN_LATENT_ENCODE="0"
      shift
      ;;
    --skip-sigma-calibration)
      RUN_SIGMA_CALIBRATION="0"
      shift
      ;;
    --skip-csp-train)
      RUN_CSP_TRAIN="0"
      shift
      ;;
    --skip-corpus-encode)
      RUN_CORPUS_ENCODE="0"
      shift
      ;;
    --skip-eval)
      RUN_EVAL="0"
      shift
      ;;
    --eval-profile)
      EVAL_PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,260p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      exit 1
      ;;
  esac
done

fae_film_adamw_ntk_prior_latent128_minmax_activate_env
fae_film_adamw_ntk_prior_latent128_minmax_require_file "${TRAIN_WRAPPER}" "FAE training wrapper"
fae_film_adamw_ntk_prior_latent128_minmax_require_file "${TRAIN_DATA_PATH}" "training dataset"
if [[ "${RUN_CORPUS_ENCODE}" == "1" ]]; then
  fae_film_adamw_ntk_prior_latent128_minmax_require_file "${CORPUS_DATA_PATH}" "large corpus dataset"
fi
fae_film_adamw_ntk_prior_latent128_minmax_mkdirs \
  "${LOG_DIR}" \
  "${CSP_RUN_DIR}" \
  "$(dirname "${LATENTS_MANIFEST_PATH}")" \
  "$(dirname "${CORPUS_LATENTS_PATH}")" \
  "$(dirname "${CONDITIONAL_CORPUS_LATENTS_PATH}")" \
  "${CALIBRATION_DIR}"

read -r -a HIDDEN_ARRAY <<< "${HIDDEN_DIMS}"

run_step() {
  local label="$1"
  shift
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label}"
  echo "============================================================"
  "$@"
}

resolve_fae_checkpoint() {
  if [[ -n "${FAE_CHECKPOINT}" ]]; then
    printf '%s\n' "${FAE_CHECKPOINT}"
    return 0
  fi
  fae_film_adamw_ntk_prior_latent128_minmax_resolve_checkpoint \
    "${FAE_RUN_DIR}" \
    "${FAE_CHECKPOINT_PREFERENCE}"
}

if [[ "${RUN_FAE_TRAIN}" == "1" ]]; then
  run_step "Train FiLM FAE on corpus/minmax archive" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      DATA_PATH="${TRAIN_DATA_PATH}" \
      RESULTS_ROOT="${FAE_RESULTS_ROOT}" \
      RUN_NAME="${EXPERIMENT_NAME}" \
      LOG_FILE="${FAE_LOG_FILE}" \
      MAX_STEPS="${FAE_MAX_STEPS}" \
      BATCH_SIZE="${FAE_BATCH_SIZE}" \
      EVAL_INTERVAL="${FAE_EVAL_INTERVAL}" \
      EVAL_N_BATCHES="${FAE_EVAL_N_BATCHES}" \
      VIS_INTERVAL="${FAE_VIS_INTERVAL}" \
      LR="${FAE_LR}" \
      WEIGHT_DECAY="${FAE_WEIGHT_DECAY}" \
      RUN_IN_BACKGROUND="0" \
      bash "${TRAIN_WRAPPER}"
fi

FAE_CHECKPOINT="$(resolve_fae_checkpoint)"
fae_film_adamw_ntk_prior_latent128_minmax_require_file "${FAE_CHECKPOINT}" "FAE checkpoint"

if [[ "${RUN_LATENT_ENCODE}" == "1" ]]; then
  run_step "Encode FiLM train/test latents for flat CSP" \
    "${PYTHON_BIN}" -u scripts/csp/encode_fae_latents.py \
      --data_path "${TRAIN_DATA_PATH}" \
      --fae_checkpoint "${FAE_CHECKPOINT}" \
      --output_path "${LATENTS_PATH}" \
      --manifest_path "${LATENTS_MANIFEST_PATH}" \
      --encode_batch_size "${ENCODE_BATCH_SIZE}" \
      --train_ratio "${TRAIN_RATIO}" \
      --time_dist_mode "${TIME_DIST_MODE}" \
      --zt_mode "${ARCHIVE_ZT_MODE}" \
      --t_scale "${T_SCALE}"
fi

fae_film_adamw_ntk_prior_latent128_minmax_require_file "${LATENTS_PATH}" "flat latent archive"

if [[ "${RUN_SIGMA_CALIBRATION}" == "1" ]]; then
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calibrate vector-latent CSP sigma"
  echo "============================================================"
  CALIBRATION_CMD=(
    "${PYTHON_BIN}" scripts/csp/calibrate_sigma.py
    --latents_path "${LATENTS_PATH}"
    --method "${SIGMA_CALIBRATION_METHOD}"
    --zt_mode "${SIGMA_ZT_MODE}"
    --json
  )
  if [[ "${SIGMA_CALIBRATION_METHOD}" == "knn_legacy" ]]; then
    CALIBRATION_CMD+=(
      --kappa "${KAPPA}"
      --k_neighbors "${K_NEIGHBORS}"
      --n_probe "${N_PROBE}"
      --seed "${SEED}"
    )
  fi
  "${CALIBRATION_CMD[@]}" >"${CALIBRATION_JSON}"

  REPORT="$(fae_film_adamw_ntk_prior_latent128_minmax_render_calibration_report "${CALIBRATION_JSON}")"
  printf '%s\n' "${REPORT}" | tee "${CALIBRATION_TEXT}"
  printf 'Saved calibration JSON to %s\n' "${CALIBRATION_JSON}"
  printf 'Saved calibration text to %s\n' "${CALIBRATION_TEXT}"
  if [[ -z "${SIGMA}" ]]; then
    SIGMA="$(fae_film_adamw_ntk_prior_latent128_minmax_read_sigma "${CALIBRATION_JSON}")"
  fi
elif [[ -z "${SIGMA}" ]]; then
  fae_film_adamw_ntk_prior_latent128_minmax_require_file "${CALIBRATION_JSON}" "sigma calibration JSON"
  SIGMA="$(fae_film_adamw_ntk_prior_latent128_minmax_read_sigma "${CALIBRATION_JSON}")"
fi

if [[ "${RUN_CSP_TRAIN}" == "1" ]]; then
  if [[ -z "${SIGMA}" ]]; then
    echo "ERROR: SIGMA is empty. Run calibration or set SIGMA explicitly." >&2
    exit 1
  fi
  run_step "Train flat vector CSP bridge" \
    "${PYTHON_BIN}" -u scripts/csp/train_csp_from_fae.py \
      --data_path "${TRAIN_DATA_PATH}" \
      --fae_checkpoint "${FAE_CHECKPOINT}" \
      --outdir "${CSP_RUN_DIR}" \
      --latents_path "${LATENTS_PATH}" \
      --manifest_path "${LATENTS_MANIFEST_PATH}" \
      --encode_batch_size "${ENCODE_BATCH_SIZE}" \
      --train_ratio "${TRAIN_RATIO}" \
      --time_dist_mode "${TIME_DIST_MODE}" \
      --zt_mode "${ARCHIVE_ZT_MODE}" \
      --t_scale "${T_SCALE}" \
      --skip_encode_if_exists \
      --sigma "${SIGMA}" \
      --dt0 "${DT0}" \
      --lr "${LR}" \
      --num_steps "${NUM_STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --condition_mode "${CONDITION_MODE}" \
      --endpoint_epsilon "${ENDPOINT_EPSILON}" \
      --sample_count "${SAMPLE_COUNT}" \
      --log_every "${LOG_EVERY}" \
      --seed "${SEED}" \
      --hidden "${HIDDEN_ARRAY[@]}"
fi

if [[ "${RUN_CORPUS_ENCODE}" == "1" ]]; then
  run_step "Encode large minmax corpus latents for conditional evaluation" \
    "${PYTHON_BIN}" -u scripts/fae/tran_evaluation/encode_corpus.py \
      --corpus_path "${CORPUS_DATA_PATH}" \
      --run_dir "${CSP_RUN_DIR}" \
      --output_path "${CORPUS_LATENTS_PATH}" \
      --batch_size "${CORPUS_BATCH_SIZE}"
  CONDITIONAL_CORPUS_LATENTS_PATH="${CORPUS_LATENTS_PATH}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  fae_film_adamw_ntk_prior_latent128_minmax_require_file "${CONDITIONAL_CORPUS_LATENTS_PATH}" "conditional latent archive"
  case "${EVAL_PROFILE}" in
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
      COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-128}"
      LATENT_TRAJECTORY_COUNT="${LATENT_TRAJECTORY_COUNT:-32}"
      LATENT_TRAJECTORY_REFERENCE_BUDGET="${LATENT_TRAJECTORY_REFERENCE_BUDGET:-512}"
      EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${CSP_RUN_DIR}/eval/smoke_n32}"
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
      COARSE_DECODE_BATCH_SIZE="${COARSE_DECODE_BATCH_SIZE:-256}"
      LATENT_TRAJECTORY_COUNT="${LATENT_TRAJECTORY_COUNT:-64}"
      LATENT_TRAJECTORY_REFERENCE_BUDGET="${LATENT_TRAJECTORY_REFERENCE_BUDGET:-2000}"
      EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${CSP_RUN_DIR}/eval/n512}"
      ;;
    *)
      echo "ERROR: --eval-profile must be one of: smoke, publication." >&2
      exit 1
      ;;
  esac

  run_step "Evaluate flat vector CSP bridge" \
    "${PYTHON_BIN}" -u scripts/csp/evaluate_csp.py \
      --run_dir "${CSP_RUN_DIR}" \
      --output_dir "${EVAL_OUTPUT_DIR}" \
      --n_realizations "${N_REALIZATIONS}" \
      --n_gt_neighbors "${N_GT_NEIGHBORS}" \
      --sample_idx "${SAMPLE_IDX}" \
      --seed "${SEED}" \
      --coarse_split "${COARSE_SPLIT}" \
      --coarse_selection "${COARSE_SELECTION}" \
      --decode_batch_size "${DECODE_BATCH_SIZE}" \
      --coarse_eval_mode "${COARSE_EVAL_MODE}" \
      --coarse_eval_conditions "${COARSE_EVAL_CONDITIONS}" \
      --coarse_eval_realizations "${COARSE_EVAL_REALIZATIONS}" \
      --conditioned_global_conditions "${CONDITIONED_GLOBAL_CONDITIONS}" \
      --conditioned_global_realizations "${CONDITIONED_GLOBAL_REALIZATIONS}" \
      --coarse_decode_batch_size "${COARSE_DECODE_BATCH_SIZE}" \
      --latent_trajectory_count "${LATENT_TRAJECTORY_COUNT}" \
      --latent_trajectory_reference_budget "${LATENT_TRAJECTORY_REFERENCE_BUDGET}" \
      --conditional_rollout_realizations "${CONDITIONAL_REALIZATIONS}" \
      --conditional_rollout_n_test_samples "${CONDITIONAL_TEST_SAMPLES}" \
      --conditional_rollout_k_neighbors "${CONDITIONAL_K_NEIGHBORS}" \
      --conditional_rollout_n_plot_conditions "${CONDITIONAL_N_PLOT_CONDITIONS}" \
      --fae_checkpoint "${FAE_CHECKPOINT}"
fi

echo "============================================================"
echo "Full FiLM minmax -> CSP pipeline completed."
echo "  FAE run        : ${FAE_RUN_DIR}"
echo "  FAE checkpoint : ${FAE_CHECKPOINT}"
echo "  CSP run        : ${CSP_RUN_DIR}"
echo "  Latents        : ${LATENTS_PATH}"
echo "  Calibration    : ${CALIBRATION_JSON}"
echo "  Conditional    : ${CONDITIONAL_CORPUS_LATENTS_PATH}"
if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "  Eval output    : ${EVAL_OUTPUT_DIR}"
fi
echo "============================================================"
