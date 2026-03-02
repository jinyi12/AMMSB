#!/bin/bash
# Sequential sweep of latent noise scales for geometric regularisation analysis.
# Bjerregaard et al. (2025): z' = z + N(0, σ²I) before decoding.
#
# Usage:
#   nohup bash scripts/fae/run_latent_noise_sweep.sh > latent_noise_sweep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../../.."

BASE_OUTPUT="results/latent_noise_sweep"
DATA_PATH="data/fae_tran_inclusions.npz"
# NOISE_LEVELS=(0.1 0.25 0.5 0.75 1.0)
NOISE_LEVELS=(0.01 0.025 0.05 0.075)

for SIGMA in "${NOISE_LEVELS[@]}"; do
    echo "========================================"
    echo "  Training with latent_noise_scale=$SIGMA"
    echo "  $(date)"
    echo "========================================"

    RUN_NAME="sigma_${SIGMA}"
    # Pass BASE_OUTPUT as --output-dir and RUN_NAME as --run-name so that
    # setup_output_directory creates: BASE_OUTPUT/sigma_X.X/checkpoints/
    # (avoids the double-nesting BASE_OUTPUT/sigma_X.X/sigma_X.X/).

    python scripts/fae/fae_naive/train_attention.py \
        --data-path "$DATA_PATH" \
        --output-dir "$BASE_OUTPUT" \
        --run-name "$RUN_NAME" \
        --pooling-type multi_query_augmented_residual \
        --n-queries 8 \
        --decoder-type standard \
        --latent-dim 128 \
        --n-freqs 128 \
        --encoder-multiscale-sigmas 1.0,4.0,8.0,16.0 \
        --decoder-multiscale-sigmas 1.0,4.0,8.0,16.0 \
        --encoder-mlp-layers 3 \
        --encoder-mlp-dim 128 \
        --decoder-features 256,256,256 \
        --loss-type ntk_scaled \
        --ntk-epsilon 1e-8 \
        --ntk-estimate-total-trace \
        --ntk-total-trace-ema-decay 0.0 \
        --ntk-trace-update-interval 100 \
        --ntk-hutchinson-probes 1 \
        --masking-strategy random \
        --eval-masking-strategy same \
        --encoder-point-ratio-by-time 0.8,0.8,0.7,0.6,0.4,0.3,0.2,0.1 \
        --beta 1e-4 \
        --latent-noise-scale "$SIGMA" \
        --save-best-model \
        --wandb-project fae-latent-noise-sweep \
        --wandb-name "sigma_${SIGMA}" \
        --optimizer muon \
        --lr 1e-3 \
        --max-steps 50000 \
        --eval-interval 10 \
        --vis-interval 10

    echo "  Completed sigma=$SIGMA at $(date)"
    echo ""
done

echo "========================================"
echo "  All sweep runs complete  $(date)"
echo "========================================"
