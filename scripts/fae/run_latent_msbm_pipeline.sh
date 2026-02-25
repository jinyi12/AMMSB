#!/usr/bin/env bash
# =============================================================================
# Full pipeline: FAE latent MSBM training + evaluation
#
# Supports both standard MLP and denoiser decoders via --decode_mode auto.
#
# Usage:
#   bash scripts/fae/run_latent_msbm_pipeline.sh \
#       --fae_checkpoint <path> \
#       --outdir <name> \
#       [--decode_mode one_step|multistep|auto] \
#       [extra train_latent_msbm args...]
#
# Example (drifting denoiser):
#   bash scripts/fae/run_latent_msbm_pipeline.sh \
#       --fae_checkpoint results/fae_drifting_denoiser_augmented_residual_msfourier/run_smf8jqws/checkpoints/best_state.pkl \
#       --outdir latent_msbm_drifting_denoiser \
#       --policy_arch augmented_mlp --hidden 512 512 512 --time_dim 128 \
#       --var_schedule exp_contract --var_decay_rate 2.0 --var 0.0625 \
#       --num_stages 50 --num_itr 10000 \
#       --wandb online --project fae_latent_msbm
#
# Example (standard MLP decoder):
#   bash scripts/fae/run_latent_msbm_pipeline.sh \
#       --fae_checkpoint results/fae__naive_augmented_residual/run_3ivys7jf/checkpoints/best_state.pkl \
#       --outdir latent_msbm_standard_mlp \
#       --policy_arch augmented_mlp --hidden 512 512 512 --time_dim 128 \
#       --var_schedule exp_contract --var_decay_rate 2.0 --var 0.0625 \
#       --num_stages 50 --num_itr 10000
# =============================================================================
set -euo pipefail

DATA_PATH="${DATA_PATH:-data/fae_tran_inclusions.npz}"

echo "============================================"
echo "  FAE Latent MSBM Pipeline"
echo "============================================"
echo "Data: ${DATA_PATH}"
echo "Args: $@"
echo ""

# Step 1: Train latent MSBM
echo "--- Step 1: Training latent MSBM ---"
python scripts/fae/fae_naive/train_latent_msbm.py \
    --data_path "${DATA_PATH}" \
    --eval_n_samples 16 \
    --n_decode 128 \
    "$@"

# Discover the output directory from the most recent results
# (user should have passed --outdir, so we find it)
OUTDIR_FLAG=""
for i in "$@"; do
    if [[ "$prev" == "--outdir" ]]; then
        OUTDIR_FLAG="$i"
    fi
    prev="$i"
done

if [[ -z "${OUTDIR_FLAG}" ]]; then
    # Find most recent results dir
    RUN_DIR=$(ls -td results/*/ | head -1)
else
    RUN_DIR="results/${OUTDIR_FLAG}"
fi

echo ""
echo "Run directory: ${RUN_DIR}"

# Step 2: Generate full trajectories
echo ""
echo "--- Step 2: Generating full trajectories ---"
python scripts/fae/generate_full_trajectories.py \
    --run_dir "${RUN_DIR}" \
    --n_samples 16 \
    --direction both \
    --save_decoded \
    --n_realizations 200 \
    --realization_sample_idx 0

# Step 3: Run Tran-aligned evaluation
echo ""
echo "--- Step 3: Running Tran-aligned evaluation ---"
python scripts/fae/tran_evaluation/evaluate.py \
    --run_dir "${RUN_DIR}" \
    --n_realizations 200 \
    --sample_idx 0

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results in: ${RUN_DIR}"
echo "============================================"
echo ""
echo "Output structure:"
echo "  ${RUN_DIR}/"
echo "    ├── args.txt                      # Training config"
echo "    ├── fae_latents.npz               # Encoded latent marginals"
echo "    ├── latent_msbm_policy_*.pth      # Trained MSBM policies"
echo "    ├── msbm_decoded_samples.npz      # Decoded sample trajectories"
echo "    ├── full_trajectories.npz         # Full SDE trajectories"
echo "    ├── eval/                          # Stage-wise eval snapshots"
echo "    └── tran_evaluation/              # Full statistical evaluation"
echo "        ├── summary.txt"
echo "        ├── metrics.json"
echo "        ├── curves.npz"
echo "        └── *.png"
