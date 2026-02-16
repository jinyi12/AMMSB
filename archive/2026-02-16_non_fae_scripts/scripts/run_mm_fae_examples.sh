#!/bin/bash
# Example training scripts for both encoder types

# Example 1: Order-agnostic spatiotemporal multiscale encoder (original)
echo "Running order-agnostic encoder..."
python scripts/mm_fae_experiment.py \
    --encoder-type spatiotemporal_multiscale \
    --latent-dim 256 \
    --encoder-scales 1,2,4,8 \
    --point-mlp-features 128,128,128 \
    --rho-features 128,128,128 \
    --patch-feature-dim 64 \
    --decoder-features 128,128,128 \
    --time-fourier-features 32 \
    --space-fourier-features 64 \
    --fourier-sigma 3.0 \
    --beta 1e-4 \
    --lr 1e-3 \
    --max-steps 1000 \
    --batch-size 32 \
    --n-dec 1024 \
    --vis-samples 3 \
    --output-dir results/mm_fae_agnostic

# Example 2: Order-preserving spatiotemporal encoder (new with GRU)
echo "Running order-preserving encoder..."
python scripts/mm_fae_experiment.py \
    --encoder-type spatiotemporal_order_preserving \
    --latent-dim 256 \
    --encoder-scales 1,2,4,8 \
    --point-mlp-features 128,128,128 \
    --rho-features 128,128,128 \
    --patch-feature-dim 64 \
    --temporal-hidden-dim 256 \
    --decoder-features 128,128,128 \
    --time-fourier-features 32 \
    --space-fourier-features 64 \
    --fourier-sigma 3.0 \
    --beta 1e-4 \
    --lr 1e-3 \
    --max-steps 1000 \
    --batch-size 32 \
    --n-dec 1024 \
    --vis-samples 3 \
    --output-dir results/mm_fae_order_preserving

# Example 3: Quick test with fewer steps
echo "Running quick test..."
python scripts/mm_fae_experiment.py \
    --encoder-type spatiotemporal_order_preserving \
    --latent-dim 128 \
    --encoder-scales 1,4 \
    --max-steps 100 \
    --batch-size 16 \
    --n-dec 512 \
    --vis-samples 2 \
    --output-dir results/mm_fae_quicktest
