#!/usr/bin/env bash
# =============================================================================
# Small prototype pipeline for:
#   - denoiser-decoder FAE (JAX/Flax)
#   - latent MSBM dynamics (Torch)
#   - spacetime-geometry decode-energy diagnostics (JAX)
#
# This is intended to be a lightweight end-to-end setup for experimenting with
# potential path biasing in latent dynamics using decoder spacetime FR energy.
#
# Usage:
#   bash scripts/fae/run_spacetime_bias_prototype.sh
#   SMOKE=1 bash scripts/fae/run_spacetime_bias_prototype.sh
#
# Notes:
# - Defaults use a small GRF dataset (`resolution=16`).
# - WandB is disabled by default for speed/portability.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../../../.."   # project root

# ── Defaults (override via environment variables) ────────────────────────────
DATA_GENERATOR="${DATA_GENERATOR:-grf}"
N_SAMPLES="${N_SAMPLES:-800}"
RESOLUTION="${RESOLUTION:-16}"
N_CONSTRAINTS="${N_CONSTRAINTS:-5}"
SCALE_MODE="${SCALE_MODE:-log_standardize}"
HELD_OUT="${HELD_OUT:-}"               # keep all marginals by default

FAE_OUTPUT_DIR="${FAE_OUTPUT_DIR:-results/fae_denoiser_proto_${DATA_GENERATOR}${RESOLUTION}}"
FAE_RUN_NAME="${FAE_RUN_NAME:-proto_$(date +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-data/fae_${DATA_GENERATOR}${RESOLUTION}.npz}"

# Denoiser FAE (keep it small)
LATENT_DIM="${LATENT_DIM:-16}"
N_FREQS="${N_FREQS:-48}"
FOURIER_SIGMA="${FOURIER_SIGMA:-1.0}"
DECODER_FEATURES="${DECODER_FEATURES:-96,96,96}"
ENCODER_POINT_RATIO="${ENCODER_POINT_RATIO:-0.3}"
ENCODER_N_POINTS="${ENCODER_N_POINTS:-64}"
DECODER_N_POINTS="${DECODER_N_POINTS:-128}"

DENOISER_DIFFUSION_STEPS="${DENOISER_DIFFUSION_STEPS:-128}"
DENOISER_EVAL_SAMPLE_STEPS="${DENOISER_EVAL_SAMPLE_STEPS:-32}"
DENOISER_BETA_SCHEDULE="${DENOISER_BETA_SCHEDULE:-cosine}"

LR="${LR:-1e-3}"
MAX_STEPS="${MAX_STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
SEED="${SEED:-0}"

# Latent MSBM (small)
MSBM_OUTDIR="${MSBM_OUTDIR:-latent_msbm_proto_${DATA_GENERATOR}${RESOLUTION}_${FAE_RUN_NAME}}"
MSBM_NUM_STAGES="${MSBM_NUM_STAGES:-20}"
MSBM_NUM_ITR="${MSBM_NUM_ITR:-4000}"
MSBM_VAR="${MSBM_VAR:-0.25}"
MSBM_POLICY_ARCH="${MSBM_POLICY_ARCH:-augmented_mlp}"
MSBM_HIDDEN="${MSBM_HIDDEN:-256 256 256}"
MSBM_TIME_DIM="${MSBM_TIME_DIM:-64}"
MSBM_WANDB_MODE="${MSBM_WANDB_MODE:-disabled}"

# Decode-energy eval
ENERGY_NUM_STEPS="${ENERGY_NUM_STEPS:-16}"
ENERGY_N_LATENTS="${ENERGY_N_LATENTS:-64}"

# Optional non-heuristic path-biasing run (Phase 2 style)
RUN_BIASING="${RUN_BIASING:-0}"
GENERATE_CORPUS="${GENERATE_CORPUS:-1}"
CORPUS_N_SAMPLES="${CORPUS_N_SAMPLES:-5000}"
CORPUS_BATCH_SIZE="${CORPUS_BATCH_SIZE:-500}"
CORPUS_PATH="${CORPUS_PATH:-data/fae_${DATA_GENERATOR}${RESOLUTION}_corpus.npz}"
CORPUS_LATENTS_PATH="${CORPUS_LATENTS_PATH:-data/fae_${DATA_GENERATOR}${RESOLUTION}_corpus_latents.npz}"

BIAS_OUTDIR="${BIAS_OUTDIR:-path_biasing_smc_nonheuristic_frmeas}"
BIAS_N_PARTICLES="${BIAS_N_PARTICLES:-64}"
BIAS_K_NEIGHBORS="${BIAS_K_NEIGHBORS:-200}"
BIAS_SIGMA_COND="${BIAS_SIGMA_COND:-1.0}"
BIAS_BETA_EMP="${BIAS_BETA_EMP:-1.0}"
BIAS_LAMBDA_FR="${BIAS_LAMBDA_FR:-1.0}"
BIAS_KAPPA_FR_MEAS="${BIAS_KAPPA_FR_MEAS:-0.5}"
BIAS_FR_MEAS_TBAR="${BIAS_FR_MEAS_TBAR:-0.5}"
BIAS_FR_MEAS_FILTER="${BIAS_FR_MEAS_FILTER:-gaussian}"   # identity|gaussian|tran
BIAS_FR_MEAS_GAUSSIAN_SIGMA="${BIAS_FR_MEAS_GAUSSIAN_SIGMA:-0.8}"
BIAS_FR_MEAS_TRAN_L_DOMAIN="${BIAS_FR_MEAS_TRAN_L_DOMAIN:-6.0}"
BIAS_FR_MEAS_EMBED_MODE="${BIAS_FR_MEAS_EMBED_MODE:-deterministic}"
BIAS_RESAMPLE_ESS_FRAC="${BIAS_RESAMPLE_ESS_FRAC:-0.6}"
BIAS_N_VIZ="${BIAS_N_VIZ:-0}"
BIAS_MAX_PARTICLES_EVAL="${BIAS_MAX_PARTICLES_EVAL:-64}"
BIAS_KAPPA_FR_MEAS_SCHEDULE="${BIAS_KAPPA_FR_MEAS_SCHEDULE:-}"
BIAS_FR_MEAS_GAUSSIAN_SIGMA_SCHEDULE="${BIAS_FR_MEAS_GAUSSIAN_SIGMA_SCHEDULE:-}"
BIAS_FR_MEAS_TRAN_H_SCHEDULE="${BIAS_FR_MEAS_TRAN_H_SCHEDULE:-}"

# ── Smoke-test overrides ─────────────────────────────────────────────────────
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "=== SMOKE TEST MODE ==="
    N_SAMPLES=120
    MAX_STEPS=800
    BATCH_SIZE=16
    MSBM_NUM_STAGES=6
    MSBM_NUM_ITR=600
    ENERGY_N_LATENTS=16
    CORPUS_N_SAMPLES=1000
    CORPUS_BATCH_SIZE=250
    BIAS_N_PARTICLES=24
    BIAS_MAX_PARTICLES_EVAL=24
fi

echo "============================================"
echo "  Spacetime Bias Prototype"
echo "============================================"
echo "Data:   ${DATA_PATH} (gen=${DATA_GENERATOR}, res=${RESOLUTION}, n=${N_SAMPLES}, T=${N_CONSTRAINTS})"
echo "FAE:    ${FAE_OUTPUT_DIR}/${FAE_RUN_NAME}"
echo "MSBM:   results/${MSBM_OUTDIR}"
if [[ "${RUN_BIASING}" == "1" ]]; then
    echo "Bias:   enabled (outdir=${BIAS_OUTDIR})"
else
    echo "Bias:   disabled (set RUN_BIASING=1 to enable)"
fi
echo ""

# ── Step 1: Generate data ────────────────────────────────────────────────────
echo "--- Step 1: Generating data ---"
python data/generate_fae_data.py \
    --output_path "${DATA_PATH}" \
    --data_generator "${DATA_GENERATOR}" \
    --n_samples "${N_SAMPLES}" \
    --resolution "${RESOLUTION}" \
    --n_constraints "${N_CONSTRAINTS}" \
    --scale_mode "${SCALE_MODE}" \
    --held_out_indices "${HELD_OUT}"

# ── Step 2: Train denoiser FAE ───────────────────────────────────────────────
echo ""
echo "--- Step 2: Training denoiser FAE ---"
python scripts/fae/fae_naive/train_attention_denoiser.py \
    --data-path "${DATA_PATH}" \
    --output-dir "${FAE_OUTPUT_DIR}" \
    --run-name "${FAE_RUN_NAME}" \
    --training-mode multi_scale \
    --latent-dim "${LATENT_DIM}" \
    --n-freqs "${N_FREQS}" \
    --fourier-sigma "${FOURIER_SIGMA}" \
    --decoder-features "${DECODER_FEATURES}" \
    --decoder-type denoiser_standard \
    --denoiser-time-emb-dim 32 \
    --denoiser-scaling 1.0 \
    --denoiser-diffusion-steps "${DENOISER_DIFFUSION_STEPS}" \
    --denoiser-beta-schedule "${DENOISER_BETA_SCHEDULE}" \
    --denoiser-eval-sample-steps "${DENOISER_EVAL_SAMPLE_STEPS}" \
    --encoder-point-ratio "${ENCODER_POINT_RATIO}" \
    --encoder-n-points "${ENCODER_N_POINTS}" \
    --decoder-n-points "${DECODER_N_POINTS}" \
    --lr "${LR}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --seed "${SEED}" \
    --save-best-model \
    --wandb-disabled

FAE_CKPT="${FAE_OUTPUT_DIR}/${FAE_RUN_NAME}/checkpoints/best_state.pkl"
if [[ ! -f "${FAE_CKPT}" ]]; then
    echo "ERROR: Expected checkpoint not found at: ${FAE_CKPT}" >&2
    exit 1
fi

# ── Step 3: Train latent MSBM ────────────────────────────────────────────────
echo ""
echo "--- Step 3: Training latent MSBM ---"
python scripts/fae/fae_naive/train_latent_msbm.py \
    --data_path "${DATA_PATH}" \
    --fae_checkpoint "${FAE_CKPT}" \
    --outdir "${MSBM_OUTDIR}" \
    --policy_arch "${MSBM_POLICY_ARCH}" \
    --hidden ${MSBM_HIDDEN} \
    --time_dim "${MSBM_TIME_DIM}" \
    --var "${MSBM_VAR}" \
    --var_schedule constant \
    --num_stages "${MSBM_NUM_STAGES}" \
    --num_itr "${MSBM_NUM_ITR}" \
    --eval_interval_stages 2 \
    --eval_n_samples 8 \
    --n_decode 16 \
    --decode_mode auto \
    --wandb_mode "${MSBM_WANDB_MODE}"

# ── Step 4: Compute decode spacetime energy by marginal ──────────────────────
echo ""
echo "--- Step 4: Decoder spacetime energy diagnostics ---"
python scripts/fae/eval_spacetime_decode_energy.py \
    --data_path "${DATA_PATH}" \
    --fae_checkpoint "${FAE_CKPT}" \
    --outdir "results/${MSBM_OUTDIR}/spacetime_energy" \
    --n_latents "${ENERGY_N_LATENTS}" \
    --denoiser_num_steps "${ENERGY_NUM_STEPS}" \
    --sampler ode \
    --num_probes 1 \
    --probe rademacher \
    --seed "${SEED}"

if [[ "${RUN_BIASING}" == "1" ]]; then
    RUN_DIR="results/${MSBM_OUTDIR}"
    echo ""
    echo "--- Step 5: Prepare corpus + latent encoding for conditional references ---"
    if [[ "${GENERATE_CORPUS}" == "1" && "${DATA_GENERATOR}" == "tran_inclusion" ]]; then
        python data/generate_large_corpus.py \
            --reference_dataset "${DATA_PATH}" \
            --n_samples "${CORPUS_N_SAMPLES}" \
            --batch_size "${CORPUS_BATCH_SIZE}" \
            --output_path "${CORPUS_PATH}"
    else
        if [[ ! -f "${CORPUS_PATH}" ]]; then
            echo "Corpus file not found at ${CORPUS_PATH}; falling back to training dataset ${DATA_PATH}"
            CORPUS_PATH="${DATA_PATH}"
        fi
    fi

    python scripts/fae/tran_evaluation/encode_corpus.py \
        --corpus_path "${CORPUS_PATH}" \
        --run_dir "${RUN_DIR}" \
        --output_path "${CORPUS_LATENTS_PATH}" \
        --batch_size "${BATCH_SIZE}"

    echo ""
    echo "--- Step 6: Non-heuristic backward SMC with FR measurement likelihood ---"
    BIAS_CMD=(
        python scripts/fae/tran_evaluation/bias_backward_sampling_smc_mala.py
        --run_dir "${RUN_DIR}"
        --corpus_latents_path "${CORPUS_LATENTS_PATH}"
        --outdir "${BIAS_OUTDIR}"
        --n_particles "${BIAS_N_PARTICLES}"
        --k_neighbors "${BIAS_K_NEIGHBORS}"
        --sigma_cond "${BIAS_SIGMA_COND}"
        --beta_emp "${BIAS_BETA_EMP}"
        --lambda_fr "${BIAS_LAMBDA_FR}"
        --kappa_fr_meas "${BIAS_KAPPA_FR_MEAS}"
        --fr_meas_tbar "${BIAS_FR_MEAS_TBAR}"
        --fr_meas_embed_mode "${BIAS_FR_MEAS_EMBED_MODE}"
        --fr_meas_filter "${BIAS_FR_MEAS_FILTER}"
        --fr_meas_gaussian_sigma "${BIAS_FR_MEAS_GAUSSIAN_SIGMA}"
        --fr_meas_tran_L_domain "${BIAS_FR_MEAS_TRAN_L_DOMAIN}"
        --resample_ess_frac "${BIAS_RESAMPLE_ESS_FRAC}"
        --n_viz "${BIAS_N_VIZ}"
        --seed "${SEED}"
    )
    if [[ -n "${BIAS_KAPPA_FR_MEAS_SCHEDULE}" ]]; then
        BIAS_CMD+=(--kappa_fr_meas_schedule "${BIAS_KAPPA_FR_MEAS_SCHEDULE}")
    fi
    if [[ -n "${BIAS_FR_MEAS_GAUSSIAN_SIGMA_SCHEDULE}" ]]; then
        BIAS_CMD+=(--fr_meas_gaussian_sigma_schedule "${BIAS_FR_MEAS_GAUSSIAN_SIGMA_SCHEDULE}")
    fi
    if [[ -n "${BIAS_FR_MEAS_TRAN_H_SCHEDULE}" ]]; then
        BIAS_CMD+=(--fr_meas_tran_H_schedule "${BIAS_FR_MEAS_TRAN_H_SCHEDULE}")
    fi
    "${BIAS_CMD[@]}"

    echo ""
    echo "--- Step 7: Evaluate biased vs baseline conditional quality ---"
    BIASED_NPZ="${RUN_DIR}/${BIAS_OUTDIR}/biased_backward_sampling.npz"
    python scripts/fae/tran_evaluation/evaluate_biased_conditional.py \
        --run_dir "${RUN_DIR}" \
        --biased_npz "${BIASED_NPZ}" \
        --corpus_path "${CORPUS_PATH}" \
        --corpus_latents_path "${CORPUS_LATENTS_PATH}" \
        --max_particles "${BIAS_MAX_PARTICLES_EVAL}" \
        --seed "${SEED}"
fi

echo ""
echo "============================================"
echo "  Prototype complete."
echo "  FAE checkpoint: ${FAE_CKPT}"
echo "  MSBM run dir:   results/${MSBM_OUTDIR}"
echo "  Energy plots:   results/${MSBM_OUTDIR}/spacetime_energy"
if [[ "${RUN_BIASING}" == "1" ]]; then
    echo "  Bias outputs:   results/${MSBM_OUTDIR}/${BIAS_OUTDIR}"
fi
echo "============================================"
