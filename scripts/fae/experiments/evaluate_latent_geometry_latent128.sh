#!/usr/bin/env bash
# Latent-geometry evaluation for the latent_dim=128 ablation runs.
#
# Registry: docs/experiments/latent128_ablation_registry.csv
# Output:   results/latent_geometry_latent128_ablation/
#
# Usage:
#   # Interactive / local:
#   bash scripts/fae/experiments/evaluate_latent_geometry_latent128.sh
#
#   # SLURM (submit via sbatch):
#   sbatch scripts/fae/experiments/evaluate_latent_geometry_latent128.sh
#
#SBATCH --job-name=latgeom_l128
#SBATCH --output=logs/slurm_%j_latgeom_l128.out
#SBATCH --error=logs/slurm_%j_latgeom_l128.err
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jy384@duke.edu

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,160p' "$0"
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
      sed -n '1,160p' "$0"
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

# ── Conda env ──────────────────────────────────────────────────────────────
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "3MASB" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate 3MASB
fi

mkdir -p logs

# ── Config ─────────────────────────────────────────────────────────────────
REGISTRY="docs/experiments/latent128_ablation_registry.csv"
OUTPUT_DIR="results/latent_geometry_latent128_ablation"
TRACKS="latent128_ablation"
PAPER_TRACK="latent128"

FORCE_FLAG=""
[[ "$FORCE_RECOMPUTE" == "1" ]] && FORCE_FLAG="--force_recompute"

echo "============================================================"
echo "  Latent-geometry evaluation  [latent_dim=128 ablation]"
echo "  Registry : $REGISTRY"
echo "  Output   : $OUTPUT_DIR"
echo "  Budget   : $BUDGET  |  Split: $SPLIT  |  Estimator: $ESTIMATOR"
echo "  Max samples/time: $MAX_SAMPLES  |  Seed: $SEED"
echo "  Force recompute : $FORCE_RECOMPUTE"
echo "============================================================"

python scripts/fae/tran_evaluation/compare_latent_geometry_models.py \
  --registry_csv          "$REGISTRY" \
  --output_dir            "$OUTPUT_DIR" \
  --tracks                "$TRACKS" \
  --paper_track           "$PAPER_TRACK" \
  --effect_baseline_scope "$TRACKS" \
  --latent_geom_budget    "$BUDGET" \
  --latent_geom_split     "$SPLIT" \
  --latent_geom_max_samples_per_time "$MAX_SAMPLES" \
  --latent_geom_trace_estimator      "$ESTIMATOR" \
  --seed                  "$SEED" \
  $FORCE_FLAG

echo ""
echo "Done. Results in: $OUTPUT_DIR"
