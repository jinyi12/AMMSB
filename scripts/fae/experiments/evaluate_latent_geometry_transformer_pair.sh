#!/usr/bin/env bash
# Canonical latent-geometry comparison for the maintained transformer pair.
#
# Runtime default pair:
#   selection is owned by scripts/fae/tran_evaluation/latent_geometry_model_selection.py
#   docs/experiments/transformer_pair_geometry_registry.csv is provenance only
#
# Output: results/latent_geometry_transformer_pair/
#
# Usage:
#   bash scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh
#   sbatch scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh
#
# Maintained pair:
#   baseline  = transformer_patch8_adamw_beta1e3_l128x128
#   treatment = transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5
#
# Output contract:
#   latent_geom_pair_summary.json/.csv
#   latent_geom_pair_delta_table.md/.csv
#   latent_geom_pair_metric_*.png/.pdf
#   latent_geom_pair_time_delta_*.png/.pdf
#
#SBATCH --job-name=latgeom_tr_pair
#SBATCH --output=logs/slurm_%j_latgeom_tr_pair.out
#SBATCH --error=logs/slurm_%j_latgeom_tr_pair.err
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jy384@duke.edu

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,200p' "$0"
  exit 0
fi

BUDGET="${BUDGET:-standard}"          # light | standard | thorough
SPLIT="${SPLIT:-test}"                # train | test | all
MAX_SAMPLES="${MAX_SAMPLES:-128}"     # encoded fields per time marginal
ESTIMATOR="${ESTIMATOR:-fhutch}"      # fhutch | hutchpp
SEED="${SEED:-42}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget)
      BUDGET="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --estimator)
      ESTIMATOR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --force_recompute)
      FORCE_RECOMPUTE="1"
      shift
      ;;
    -h|--help)
      sed -n '1,200p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$REPO_ROOT"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "3MASB" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate 3MASB
fi

mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-results/latent_geometry_transformer_pair}"

FORCE_FLAG=""
[[ "$FORCE_RECOMPUTE" == "1" ]] && FORCE_FLAG="--force_recompute"

echo "============================================================"
echo "  Latent-geometry evaluation  [canonical transformer pair]"
echo "  Output   : $OUTPUT_DIR"
echo "  Budget   : $BUDGET  |  Split: $SPLIT  |  Estimator: $ESTIMATOR"
echo "  Max samples/time: $MAX_SAMPLES  |  Seed: $SEED"
echo "  Force recompute : $FORCE_RECOMPUTE"
echo "============================================================"

python scripts/fae/tran_evaluation/compare_latent_geometry_models.py \
  --output_dir "$OUTPUT_DIR" \
  --latent_geom_budget "$BUDGET" \
  --latent_geom_split "$SPLIT" \
  --latent_geom_max_samples_per_time "$MAX_SAMPLES" \
  --latent_geom_trace_estimator "$ESTIMATOR" \
  --seed "$SEED" \
  $FORCE_FLAG

echo ""
echo "Done. Results in: $OUTPUT_DIR"
