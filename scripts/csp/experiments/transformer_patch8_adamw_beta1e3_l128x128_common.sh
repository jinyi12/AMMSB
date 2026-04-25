#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file from a launcher script; do not execute it directly." >&2
  exit 1
fi

transformer_patch8_prior_defaults() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if REPO_ROOT="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)"; then
    :
  else
    REPO_ROOT="$(cd "$script_dir/../../.." && pwd)"
  fi
  cd "$REPO_ROOT"

  EXPERIMENT_NAME="${EXPERIMENT_NAME:-transformer_patch8_adamw_beta1e3_l128x128}"
  ENV_NAME="${ENV_NAME:-3MASB}"
  PYTHON_BIN="${PYTHON_BIN:-python}"

  FAE_RESULTS_ROOT="${FAE_RESULTS_ROOT:-results/fae_${EXPERIMENT_NAME}}"
  FAE_RUN_DIR="${FAE_RUN_DIR:-${FAE_RESULTS_ROOT}/${EXPERIMENT_NAME}}"
  FAE_CHECKPOINT_DEFAULT="${FAE_CHECKPOINT_DEFAULT:-${FAE_RUN_DIR}/checkpoints/best_state.pkl}"

  TRAIN_DATA_PATH_DEFAULT="${TRAIN_DATA_PATH_DEFAULT:-data/fae_tran_inclusions_minmax.npz}"
  CORPUS_DATA_PATH_DEFAULT="${CORPUS_DATA_PATH_DEFAULT:-data/fae_tran_inclusions_minmax_corpus.npz}"
}

transformer_patch8_prior_activate_env() {
  if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
  fi
}

transformer_patch8_prior_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

transformer_patch8_prior_mkdirs() {
  mkdir -p "$@"
}

transformer_patch8_prior_launch() {
  local background="$1"
  local nohup_log="$2"
  shift 2
  local -a cmd=("$@")

  if [[ "$background" == "1" ]]; then
    mkdir -p "$(dirname "$nohup_log")"
    setsid "${cmd[@]}" < /dev/null >"${nohup_log}" 2>&1 &
    local pid=$!
    echo "Started background run."
    echo "  pid: ${pid}"
    echo "  log: ${nohup_log}"
    exit 0
  fi

  exec "${cmd[@]}"
}

transformer_patch8_prior_read_sigma() {
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

transformer_patch8_prior_render_calibration_report() {
  local json_path="$1"
  "${PYTHON_BIN}" - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
method = payload.get("method", "global_mle")

lines = [
    "============================================================",
    "Patch-8 token-native CSP sigma calibration",
    f"  latents_path              : {payload['latents_path']}",
    f"  archive_format            : {payload.get('archive_format', 'flat')}",
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
