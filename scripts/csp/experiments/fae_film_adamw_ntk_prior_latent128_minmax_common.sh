#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file from a launcher script; do not execute it directly." >&2
  exit 1
fi

fae_film_adamw_ntk_prior_latent128_minmax_defaults() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if REPO_ROOT="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)"; then
    :
  else
    REPO_ROOT="$(cd "$script_dir/../../.." && pwd)"
  fi
  cd "$REPO_ROOT"

  EXPERIMENT_NAME="${EXPERIMENT_NAME:-fae_film_adamw_ntk_prior_latent128_minmax_corpus}"
  ENV_NAME="${ENV_NAME:-3MASB}"
  PYTHON_BIN="${PYTHON_BIN:-python}"

  TRAIN_WRAPPER="${TRAIN_WRAPPER:-scripts/fae/experiments/ablation_adam/adamw_ntk_prior_minmax.sh}"
  FAE_RESULTS_ROOT="${FAE_RESULTS_ROOT:-results/${EXPERIMENT_NAME}}"
  FAE_RUN_DIR="${FAE_RUN_DIR:-${FAE_RESULTS_ROOT}/${EXPERIMENT_NAME}}"

  OUTPUT_BASE="${OUTPUT_BASE:-results/csp/${EXPERIMENT_NAME}}"
  RUN_DIR_DEFAULT="${RUN_DIR_DEFAULT:-${OUTPUT_BASE}/main}"
  LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs}"
  CALIBRATION_DIR="${CALIBRATION_DIR:-${OUTPUT_BASE}/calibration}"

  TRAIN_DATA_PATH_DEFAULT="${TRAIN_DATA_PATH_DEFAULT:-data/fae_tran_inclusions_minmax_corpus.npz}"
  CORPUS_DATA_PATH_DEFAULT="${CORPUS_DATA_PATH_DEFAULT:-data/fae_tran_inclusions_minmax_corpus.npz}"

  LATENTS_PATH_DEFAULT="${LATENTS_PATH_DEFAULT:-${RUN_DIR_DEFAULT}/fae_latents.npz}"
  LATENTS_MANIFEST_PATH_DEFAULT="${LATENTS_MANIFEST_PATH_DEFAULT:-${RUN_DIR_DEFAULT}/config/fae_latents_manifest.json}"
  CORPUS_LATENTS_PATH_DEFAULT="${CORPUS_LATENTS_PATH_DEFAULT:-${RUN_DIR_DEFAULT}/corpus_latents.npz}"
  CONDITIONAL_CORPUS_LATENTS_PATH_DEFAULT="${CONDITIONAL_CORPUS_LATENTS_PATH_DEFAULT:-${LATENTS_PATH_DEFAULT}}"
  CALIBRATION_JSON_DEFAULT="${CALIBRATION_JSON_DEFAULT:-${CALIBRATION_DIR}/sigma_calibration.json}"
  CALIBRATION_TEXT_DEFAULT="${CALIBRATION_TEXT_DEFAULT:-${CALIBRATION_DIR}/sigma_calibration.txt}"
}

fae_film_adamw_ntk_prior_latent128_minmax_activate_env() {
  if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
  fi
}

fae_film_adamw_ntk_prior_latent128_minmax_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

fae_film_adamw_ntk_prior_latent128_minmax_mkdirs() {
  mkdir -p "$@"
}

fae_film_adamw_ntk_prior_latent128_minmax_launch() {
  local background="$1"
  local nohup_log="$2"
  shift 2
  local -a cmd=("$@")

  if [[ "$background" == "1" ]]; then
    mkdir -p "$(dirname "$nohup_log")"
    nohup "${cmd[@]}" >"${nohup_log}" 2>&1 &
    local pid=$!
    echo "Started background run."
    echo "  pid: ${pid}"
    echo "  log: ${nohup_log}"
    exit 0
  fi

  exec "${cmd[@]}"
}

fae_film_adamw_ntk_prior_latent128_minmax_resolve_checkpoint() {
  local run_dir="$1"
  local preference="${2:-best_then_state}"
  local best_path="${run_dir}/checkpoints/best_state.pkl"
  local state_path="${run_dir}/checkpoints/state.pkl"

  case "${preference}" in
    best)
      if [[ -f "${best_path}" ]]; then
        printf '%s\n' "${best_path}"
        return 0
      fi
      echo "ERROR: best checkpoint requested but not found: ${best_path}" >&2
      exit 1
      ;;
    state)
      if [[ -f "${state_path}" ]]; then
        printf '%s\n' "${state_path}"
        return 0
      fi
      echo "ERROR: final checkpoint requested but not found: ${state_path}" >&2
      exit 1
      ;;
    best_then_state)
      if [[ -f "${best_path}" ]]; then
        printf '%s\n' "${best_path}"
        return 0
      fi
      if [[ -f "${state_path}" ]]; then
        printf '%s\n' "${state_path}"
        return 0
      fi
      ;;
    state_then_best)
      if [[ -f "${state_path}" ]]; then
        printf '%s\n' "${state_path}"
        return 0
      fi
      if [[ -f "${best_path}" ]]; then
        printf '%s\n' "${best_path}"
        return 0
      fi
      ;;
    *)
      echo "ERROR: unknown checkpoint preference '${preference}'." >&2
      exit 1
      ;;
  esac

  echo "ERROR: no FAE checkpoint found under ${run_dir}/checkpoints" >&2
  exit 1
}

fae_film_adamw_ntk_prior_latent128_minmax_read_sigma() {
  local json_path="$1"
  "${PYTHON_BIN}" - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
value = payload.get("recommended_constant_sigma")
if value is None:
    raise SystemExit("recommended_constant_sigma missing from calibration JSON")
print(f"{float(value):.12g}")
PY
}

fae_film_adamw_ntk_prior_latent128_minmax_render_calibration_report() {
  local json_path="$1"
  "${PYTHON_BIN}" - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
method = payload.get("method", "knn_legacy")

lines = [
    "============================================================",
    "FiLM minmax CSP sigma calibration",
    f"  latents_path              : {payload['latents_path']}",
    f"  archive_format            : {payload['archive_format']}",
    f"  method                    : {method}",
    f"  zt_mode                   : {payload['zt_mode']}",
    f"  uniform_intervals         : {payload['uniform_intervals']}",
    f"  interval_lengths          : {payload['interval_lengths']}",
    f"  recommended_constant_sigma: {payload['recommended_constant_sigma']:.6g}",
    f"  recommendation_source     : {payload.get('recommended_constant_sigma_source', 'legacy')}",
    f"  global_sigma_sq_mle       : {payload.get('global_sigma_sq_mle')}",
    f"  global_sigma_mle          : {payload.get('global_sigma_mle')}",
    f"  global_mle_sqsum_over_dt  : {payload.get('global_mle_standardized_squared_l2_sum')}",
    f"  global_mle_sample_weight  : {payload.get('global_mle_sample_weight')}",
    f"  global_sigma_denominator  : {payload.get('global_sigma_sq_mle_denominator')}",
    f"  pooled_squared_l2_sum     : {payload.get('pooled_squared_l2_sum')}",
    f"  pooled_interval_weight    : {payload.get('pooled_interval_weight')}",
    f"  pooled_sigma_denominator  : {payload.get('pooled_sigma_sq_denominator')}",
    f"  midpoint_std(constant)    : {payload['midpoint_std_by_constant_sigma']}",
    f"  delta_rms                 : {payload['delta_rms']}",
    f"  sigma_mle_by_interval     : {payload.get('sigma_mle_by_interval', payload.get('sigma_by_delta'))}",
    f"  sigma_sq_mle_by_interval  : {payload.get('sigma_sq_mle_by_interval')}",
    f"  reference_sigma_by_delta  : {payload['reference_constant_sigma_by_delta']:.6g}",
    "============================================================",
]
if method == "knn_legacy":
    lines.insert(8, f"  kappa                     : {payload.get('kappa')}")
    lines.insert(9, f"  conditional_residual_rms  : {payload['conditional_residual_rms']}")
    lines.insert(10, f"  sigma_by_conditional      : {payload['sigma_by_conditional']}")
print("\n".join(lines))
PY
}
