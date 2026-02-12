#!/usr/bin/env bash
# Train FAE on GPU
set -euo pipefail

# train FAE with tran_inclusion data on GPU
python scripts/fae/train_fae.py \
  --data-path data/fae_tran_inclusions.npz \
  --output-dir results/fae_tran_inclusions_gpu \
  --data-generator tran_inclusion \
  --resolution 128 \
  --n-samples 5000 \
  --L-domain 6.0 \
  --scale-mode log_standardize \
  --tran-D-large 1.0 \
  --tran-H-meso-list "1D,1.25D,1.5D,2D,2.5D,3D" \
  --tran-inclusion-value 1000 \
  --tran-vol-frac-large 0.2 \
  --tran-vol-frac-small 0.1 \
  --latent-dim 64 \
  --n-freqs 128 \
  --fourier-sigma 1.0 \
  --decoder-features "256,256,256,256" \
  --encoder-mlp-dim 256 \
  --encoder-mlp-layers 4 \
  --encoder-point-ratio 0.3 \
  --beta 1e-4 \
  --lr 1e-3 \
  --lr-decay-step 50000 \
  --lr-decay-factor 0.5 \
  --max-steps 50000 \
  --eval-interval 1000 \
  --batch-size 32 \
  --train-ratio 0.8 \
  --seed 42 \
  --n-vis-samples 4 \
  --wandb-project fae-multiscale \
  --wandb-name tran_inclusions_gpu \