#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-3MASB}"
OUT_DIR="${OUT_DIR:-results/publication}"
MAX_STEPS="${MAX_STEPS:-50000}"
MSBM_RUN_DIR="${MSBM_RUN_DIR:-/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior}"
INCLUDE_MSBM_FIG17="1"
REEVALUATE_MISSING="0"
SKIP_PSD_COMPUTE="0"
RECOMPUTE_PSD="0"

FAE_ROOTS=(
  "/data1/jy384/research/MMSFM/results/fae_film_muon_l2_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_muon_ntk_99pct_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_muon_prior_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_muon_ntk_prior_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_l2_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_ntk_99pct_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_prior_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_ntk_prior_latent128"
)

PSD_RUN_DIRS=(
  "results/fae_film_adam_l2_latent128/run_g0ysv6bb"
  "results/fae_film_adam_ntk_99pct_latent128/run_sphluzvp"
  "results/fae_film_adam_prior_latent128/run_mgn5f93n"
  "results/fae_film_adam_ntk_prior_latent128/run_uae85cd8"
  "results/fae_film_muon_l2_latent128/run_4cyupstm"
  "results/fae_film_muon_ntk_99pct_latent128/run_ea5yckkq"
  "results/fae_film_muon_prior_latent128/run_xn2xd51y"
  "results/fae_film_muon_ntk_prior_latent128/run_vq1adonq"
)

PSD_LABELS=(
  'Adam + $\ell_2$'
  'Adam + NTK'
  'Adam + $\ell_2$ + Prior'
  'Adam + NTK + Prior'
  'Muon + $\ell_2$'
  'Muon + NTK'
  'Muon + $\ell_2$ + Prior'
  'Muon + NTK + Prior'
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_name)
      ENV_NAME="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --msbm_run_dir)
      MSBM_RUN_DIR="$2"
      INCLUDE_MSBM_FIG17="1"
      shift 2
      ;;
    --no_msbm_fig17)
      INCLUDE_MSBM_FIG17="0"
      shift
      ;;
    --reevaluate_missing)
      REEVALUATE_MISSING="1"
      shift
      ;;
    --skip_psd_compute)
      SKIP_PSD_COMPUTE="1"
      shift
      ;;
    --recompute_psd)
      RECOMPUTE_PSD="1"
      shift
      ;;
    -h|--help)
      sed -n '1,220p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR"

run_in_env() {
  conda run -n "$ENV_NAME" "$@"
}

CMD=(
  python scripts/fae/fae_publication_figures.py
  --out-dir "$OUT_DIR"
  --max-steps "$MAX_STEPS"
)
if [[ "$INCLUDE_MSBM_FIG17" == "1" ]]; then
  CMD+=(--msbm-run-dir "$MSBM_RUN_DIR")
fi

echo "============================================"
echo "FAE publication figures"
echo "============================================"
echo "Environment                : $ENV_NAME"
echo "Repo root                  : $REPO_ROOT"
echo "Output dir                 : $OUT_DIR"
echo "Max steps                  : $MAX_STEPS"
echo "Reevaluate missing evals   : $REEVALUATE_MISSING"
echo "Skip PSD compute           : $SKIP_PSD_COMPUTE"
echo "Recompute PSD              : $RECOMPUTE_PSD"
if [[ "$INCLUDE_MSBM_FIG17" == "1" ]]; then
  echo "MSBM fig17 run             : $MSBM_RUN_DIR"
else
  echo "MSBM fig17 run             : disabled"
fi

if [[ "$REEVALUATE_MISSING" == "1" ]]; then
  REEVAL_CMD=(
    python scripts/fae/reevaluate_runs.py
    --missing_only
  )
  for root in "${FAE_ROOTS[@]}"; do
    REEVAL_CMD+=(--run_dir "$root")
  done
  run_in_env "${REEVAL_CMD[@]}"
fi

PSD_NPZ="$OUT_DIR/psd_data.npz"
PSD_METRICS_JSON="$OUT_DIR/psd_latent_metrics.json"
if [[ "$SKIP_PSD_COMPUTE" != "1" ]]; then
  if [[ "$RECOMPUTE_PSD" == "1" || ! -f "$PSD_NPZ" ]]; then
    echo
    echo "Generating PSD data for publication figures..."
    PSD_CMD=(
      python scripts/fae/compute_psd.py
      --run-dirs
    )
    for run_dir in "${PSD_RUN_DIRS[@]}"; do
      PSD_CMD+=("$run_dir")
    done
    PSD_CMD+=(--labels)
    for label in "${PSD_LABELS[@]}"; do
      PSD_CMD+=("$label")
    done
    PSD_CMD+=(
      --out "$PSD_NPZ"
      --metrics-out "$PSD_METRICS_JSON"
    )
    run_in_env "${PSD_CMD[@]}"
  else
    echo "PSD data already exists     : $PSD_NPZ"
  fi
fi

run_in_env "${CMD[@]}"

echo
echo "============================================"
echo "FAE publication figures ready"
echo "============================================"
echo "Output dir                 : $OUT_DIR"
