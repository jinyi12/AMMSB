#!/usr/bin/env bash
# End-to-end pipeline: data generation + FAE training.
#
# Usage:
#   bash scripts/fae/run_fae_pipeline.sh          # full run
#   SMOKE=1 bash scripts/fae/run_fae_pipeline.sh  # quick smoke test

set -euo pipefail
cd "$(dirname "$0")/../.."   # project root

# ── Defaults (override via environment variables) ────────────────────
DATA_GENERATOR="${DATA_GENERATOR:-grf}"
N_SAMPLES="${N_SAMPLES:-2000}"
RESOLUTION="${RESOLUTION:-32}"
N_CONSTRAINTS="${N_CONSTRAINTS:-7}"
HELD_OUT="${HELD_OUT:-2,4}"
SCALE_MODE="${SCALE_MODE:-log_standardize}"

LATENT_DIM="${LATENT_DIM:-32}"
N_FREQS="${N_FREQS:-64}"
FOURIER_SIGMA="${FOURIER_SIGMA:-1.0}"
DECODER_FEATURES="${DECODER_FEATURES:-128,128,128,128}"
ENCODER_POINT_RATIO="${ENCODER_POINT_RATIO:-0.3}"
BETA="${BETA:-1e-4}"

LR="${LR:-1e-3}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

# ── Smoke-test overrides ─────────────────────────────────────────────
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "=== SMOKE TEST MODE ==="
    N_SAMPLES=100
    MAX_STEPS=500
    BATCH_SIZE=16
    EVAL_INTERVAL=5
else
    EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

# ── Paths ────────────────────────────────────────────────────────────
DATA_PATH="data/fae_${DATA_GENERATOR}.npz"
OUTPUT_DIR="results/fae_${DATA_GENERATOR}"

echo "============================================="
echo " FAE Pipeline: ${DATA_GENERATOR}"
echo "============================================="

# ── Step 1: Generate data ────────────────────────────────────────────
echo ""
echo ">>> Step 1/2: Generating data -> ${DATA_PATH}"
python data/generate_fae_data.py \
    --output_path "${DATA_PATH}" \
    --data_generator "${DATA_GENERATOR}" \
    --n_samples "${N_SAMPLES}" \
    --resolution "${RESOLUTION}" \
    --n_constraints "${N_CONSTRAINTS}" \
    --scale_mode "${SCALE_MODE}" \
    --held_out_indices "${HELD_OUT}"

# ── Step 2: Train FAE ───────────────────────────────────────────────
echo ""
echo ">>> Step 2/2: Training FAE -> ${OUTPUT_DIR}"
python scripts/fae/train_fae.py \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --latent-dim "${LATENT_DIM}" \
    --n-freqs "${N_FREQS}" \
    --fourier-sigma "${FOURIER_SIGMA}" \
    --decoder-features "${DECODER_FEATURES}" \
    --encoder-point-ratio "${ENCODER_POINT_RATIO}" \
    --beta "${BETA}" \
    --lr "${LR}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --seed "${SEED}"

echo ""
echo "============================================="
echo " Pipeline complete."
echo " Results in: ${OUTPUT_DIR}"
echo "============================================="
