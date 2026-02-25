#!/usr/bin/env bash
# =============================================================================
# Full post-training evaluation pipeline for latent MSBM runs.
#
# Steps:
#   1) Generate full trajectories (latent + decoded fields)
#   2) Plot latent manifold + knot trajectories
#   3) Plot full trajectory visualizations
#   4) Run Tran-aligned evaluation (includes report.py figure generation)
#
# Usage:
#   bash scripts/fae/run_latent_msbm_evaluation.sh \
#     --run_dir results/latent_msbm_drifting_denoiser_vonly_new3
#
# Optional:
#   --python /path/to/python
#   --n_samples 32
#   --n_realizations 200
#   --sample_idx 0
#   --nogpu
#   --no_use_ema
# =============================================================================
set -euo pipefail

RUN_DIR="/data1/jy384/research/MMSFM/results/latent_msbm_drifting_denoiser_vonly_new2"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_SAMPLES="32"
N_REALIZATIONS="200"
SAMPLE_IDX="0"
USE_EMA="1"
NOGPU="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --n_samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --n_realizations)
      N_REALIZATIONS="$2"
      shift 2
      ;;
    --sample_idx)
      SAMPLE_IDX="$2"
      shift 2
      ;;
    --no_use_ema)
      USE_EMA="0"
      shift
      ;;
    --use_ema)
      USE_EMA="1"
      shift
      ;;
    --nogpu)
      NOGPU="1"
      shift
      ;;
    -h|--help)
      sed -n '1,45p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "Error: --run_dir is required."
  echo "Example: bash scripts/fae/run_latent_msbm_evaluation.sh --run_dir results/latent_msbm_drifting_denoiser_vonly_new3"
  exit 1
fi

# Resolve repo root from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Error: run_dir not found: $RUN_DIR"
  exit 1
fi

if [[ ! -f "$RUN_DIR/args.txt" ]]; then
  echo "Error: missing $RUN_DIR/args.txt"
  exit 1
fi

if [[ ! -f "$RUN_DIR/fae_latents.npz" ]]; then
  echo "Warning: $RUN_DIR/fae_latents.npz not found."
  echo "The latent visualization step may fail unless train_latent_msbm outputs are present."
fi

echo "============================================"
echo "  Latent MSBM Evaluation Pipeline"
echo "============================================"
echo "Repo root       : $REPO_ROOT"
echo "Run dir         : $RUN_DIR"
echo "Python          : $PYTHON_BIN"
echo "n_samples       : $N_SAMPLES"
echo "n_realizations  : $N_REALIZATIONS"
echo "sample_idx      : $SAMPLE_IDX"
echo "use_ema         : $USE_EMA"
echo "nogpu           : $NOGPU"

EMA_FLAG="--use_ema"
if [[ "$USE_EMA" == "0" ]]; then
  EMA_FLAG="--no_use_ema"
fi

GPU_FLAG=""
if [[ "$NOGPU" == "1" ]]; then
  GPU_FLAG="--nogpu"
fi

echo ""
echo "--- Step 1: Generate full trajectories ---"
"$PYTHON_BIN" scripts/fae/generate_full_trajectories.py \
  --run_dir "$RUN_DIR" \
  --n_samples "$N_SAMPLES" \
  --direction both \
  --save_decoded \
  --output_name full_trajectories.npz \
  $EMA_FLAG \
  $GPU_FLAG

echo ""
echo "--- Step 2: Latent manifold + knot trajectories ---"
(
  cd notebooks
  MSBM_DIR="$REPO_ROOT/$RUN_DIR" "$PYTHON_BIN" fae_latent_msbm_latent_viz.py
)

echo ""
echo "--- Step 3: Full trajectory visualizations ---"
(
  cd notebooks
  TRAJ_PATH="$REPO_ROOT/$RUN_DIR/full_trajectories.npz" "$PYTHON_BIN" visualize_full_trajectories.py
)

echo ""
echo "--- Step 4: Tran-aligned evaluation + report figures ---"
"$PYTHON_BIN" scripts/fae/tran_evaluation/evaluate.py \
  --run_dir "$RUN_DIR" \
  --n_realizations "$N_REALIZATIONS" \
  --sample_idx "$SAMPLE_IDX" \
  $EMA_FLAG \
  $GPU_FLAG

echo ""
echo "============================================"
echo "Evaluation complete"
echo "Outputs:"
echo "  - $RUN_DIR/full_trajectories.npz"
echo "  - $RUN_DIR/eval/latent_viz/"
echo "  - $RUN_DIR/full_traj_viz/"
echo "  - $RUN_DIR/tran_evaluation/"
echo "============================================"
