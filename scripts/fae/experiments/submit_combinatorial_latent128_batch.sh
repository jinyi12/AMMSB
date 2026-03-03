#!/usr/bin/env bash
set -euo pipefail

# Submit combinatorial latent-128 reruns on SLURM:
#   - Adam/Muon + L2
#   - Adam/Muon + NTK
#   - Adam/Muon + NTK + Prior
#
# Fixed experiment controls (no overrides):
#   latent_dim=128
#   encoder/decoder width=256
#   batch_size=32 for all runs
#   NTK trace estimator=fhutch
#   NTK output chunking disabled (chunk_size=0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=0
FORCE_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes)
      FORCE_YES=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dry-run] [--yes]"
      exit 1
      ;;
  esac
done

LATENT_DIM=128
WIDTH_DIM=256
PRIOR_HIDDEN_DIM=256
BATCH_SIZE=32
RESULTS_ROOT="${RESULTS_ROOT:-/work/jy384/AMMSB_results}"
WANDB_PROJECT="${WANDB_PROJECT:-fae-film-latent128-rerun}"
SWEEP_ID="${SWEEP_ID:-latent128_fhutch_nochunk_2026q1}"

COMMON_HEADER=$(cat <<'EOF'
#!/bin/bash
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jy384@duke.edu

set -euo pipefail

module load CUDA/12.8
source /hpc/dctrl/jy384/miniconda3/etc/profile.d/conda.sh
conda activate 3MASB

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

cd /hpc/dctrl/jy384/AMMSB
mkdir -p logs
EOF
)

COMMON_FLAGS=(
  --data-path data/fae_tran_inclusions.npz
  --pooling-type dual_stream_bottleneck
  --decoder-type film
  --latent-dim "$LATENT_DIM"
  --n-freqs 64
  --encoder-multiscale-sigmas 1.0,2.0,4.0,8.0
  --decoder-multiscale-sigmas 1.0,2.0,4.0,8.0
  --encoder-mlp-layers 3
  --encoder-mlp-dim "$WIDTH_DIM"
  --decoder-features "${WIDTH_DIM},${WIDTH_DIM},${WIDTH_DIM}"
  --masking-strategy random
  --eval-masking-strategy same
  --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1
  --save-best-model
  --wandb-project "$WANDB_PROJECT"
  --lr 1e-3
  --max-steps 50000
  --batch-size "$BATCH_SIZE"
  --eval-interval 25
  --eval-n-batches 5
  --vis-interval 5
)

NTK_FLAGS=(
  --loss-type ntk_scaled
  --ntk-estimate-total-trace
  --ntk-total-trace-ema-decay 0.0
  --ntk-epsilon 1e-8
  --ntk-trace-update-interval 100
  --ntk-hutchinson-probes 1
  --ntk-trace-estimator fhutch
  --ntk-output-chunk-size 0
)

submit_job() {
  local job_name="$1"
  local trainer="$2"
  shift 2
  local -a extra_flags=("$@")

  local run_name="fae_${job_name}"
  local output_dir="${RESULTS_ROOT}/${run_name}"
  local slurm_script
  slurm_script="$(mktemp)"

  {
    echo "$COMMON_HEADER"
    echo "python scripts/fae/fae_naive/${trainer} \\"
    printf '  --output-dir "%s" \\\n' "$output_dir"
    printf '  --run-name "%s" \\\n' "$run_name"
    for flag in "${COMMON_FLAGS[@]}"; do
      printf '  %s \\\n' "$flag"
    done
    for flag in "${extra_flags[@]}"; do
      printf '  %s \\\n' "$flag"
    done
    echo ""
  } > "$slurm_script"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] sbatch --job-name=${job_name} $slurm_script"
    echo "          run=${run_name}"
  else
    echo "[sbatch] ${job_name}"
    sbatch --job-name="${job_name}" "$slurm_script"
  fi

  rm -f "$slurm_script"
}

echo "Combinatorial latent-128 SLURM batch"
echo "Sweep ID: $SWEEP_ID"
echo "Latent dim: $LATENT_DIM"
echo "MLP/prior dim: $WIDTH_DIM"
echo "Batch size: $BATCH_SIZE (all runs)"
echo "NTK estimator: fhutch"
echo "NTK output chunking: disabled (0)"
echo "Results root: $RESULTS_ROOT"
echo

echo "Jobs:"
echo "  - film_adam_l2_l128_w256"
echo "  - film_muon_l2_l128_w256"
echo "  - film_adam_ntk_l128_w256_fhutch"
echo "  - film_muon_ntk_l128_w256_fhutch"
echo "  - film_adam_ntk_prior_l128_w256_fhutch"
echo "  - film_muon_ntk_prior_l128_w256_fhutch"
echo

if [[ "$DRY_RUN" -eq 0 && "$FORCE_YES" -ne 1 ]]; then
  read -r -p "Submit all 6 jobs with sbatch? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

submit_job "film_adam_l2_l128_w256" "train_attention.py" \
  --optimizer adam \
  --loss-type l2 \
  --beta 1e-4

submit_job "film_muon_l2_l128_w256" "train_attention.py" \
  --optimizer muon \
  --loss-type l2 \
  --beta 1e-4

submit_job "film_adam_ntk_l128_w256_fhutch" "train_attention.py" \
  --optimizer adam \
  --beta 1e-4 \
  "${NTK_FLAGS[@]}"

submit_job "film_muon_ntk_l128_w256_fhutch" "train_attention.py" \
  --optimizer muon \
  --beta 1e-4 \
  "${NTK_FLAGS[@]}"

submit_job "film_adam_ntk_prior_l128_w256_fhutch" "train_attention_denoiser.py" \
  --optimizer adam \
  --beta 0.0 \
  "${NTK_FLAGS[@]}" \
  --use-prior \
  --prior-hidden-dim "$PRIOR_HIDDEN_DIM" \
  --prior-n-layers 3 \
  --prior-time-emb-dim 32 \
  --prior-logsnr-max 5.0 \
  --prior-loss-weight 1.0

submit_job "film_muon_ntk_prior_l128_w256_fhutch" "train_attention_denoiser.py" \
  --optimizer muon \
  --beta 0.0 \
  "${NTK_FLAGS[@]}" \
  --use-prior \
  --prior-hidden-dim "$PRIOR_HIDDEN_DIM" \
  --prior-n-layers 3 \
  --prior-time-emb-dim 32 \
  --prior-logsnr-max 5.0 \
  --prior-loss-weight 1.0

echo
echo "Done. Monitor with: squeue -u jy384"
