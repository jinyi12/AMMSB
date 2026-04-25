#!/usr/bin/env bash
# Historical latent128 + latent-MSBM compatibility bundle.
#
# Maintained scope in the active tree:
#   1. latent-space PCA comparison for the latent128 FiLM eight-run set
#   2. latent-MSBM compatibility trajectory generation
#   3. latent / decoded trajectory visualizations written next to the MSBM run
#   4. unconditional Tran-aligned compatibility evaluation
#   5. optional post-filtered unconditional / conditional diagnostics
#
# Removed from the active tree:
#   - latent128 cross-model latent-geometry comparison
#   - latent-MSBM conditional evaluation wrapper
#   - latent-MSBM conditional projection wrapper
#
# Current manuscript-facing figures now live in:
#   - docs/manuscript_figure_map.md
#   - docs/experiments/transformer_pair_geometry.md
#   - docs/csp_conditional_evaluation_core.md
set -euo pipefail

ENV_NAME="${ENV_NAME:-3MASB}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/publication_latent128_and_msbm}"
MSBM_RUN_DIR="${MSBM_RUN_DIR:-/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior}"
LATENT_SPACE_SPLIT="${LATENT_SPACE_SPLIT:-${LATENT_GEOM_SPLIT:-test}}"
TRAJECTORY_SAMPLES="${TRAJECTORY_SAMPLES:-32}"
EVAL_REALIZATIONS="${EVAL_REALIZATIONS:-200}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
DRIFT_CLIP_NORM="${DRIFT_CLIP_NORM:-}"
REUSE_EXISTING="1"
USE_EMA="1"
REFRESH_PLOTS_ONLY="0"
RUN_LATENT_SPACE="1"
RUN_TRAJECTORY_EVAL_ONLY="0"
RUN_POSTFILTERED_EVAL="0"
RUN_POSTFILTERED_EVAL_ONLY="0"
CONDITIONAL_CORPUS_PATH="${CONDITIONAL_CORPUS_PATH:-data/fae_tran_inclusions_corpus.npz}"
CONDITIONAL_CORPUS_LATENTS_PATH="${CONDITIONAL_CORPUS_LATENTS_PATH:-data/corpus_latents_ntk_prior.npz}"
CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-200}"
CONDITIONAL_MIN_SPACING_PIXELS="${CONDITIONAL_MIN_SPACING_PIXELS:-4}"

IGNORED_DEPRECATED_FLAGS=()
REMOVED_FEATURE_REQUESTS=()

FAE_ROOTS=(
  "/data1/jy384/research/MMSFM/results/fae_film_muon_l2_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_muon_ntk_99pct_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_muon_prior_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_muon_ntk_prior_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_l2_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_ntk_99pct_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_prior_latent128"
  "/data1/jy384/research/MMSFM/results/fae_film_adam_ntk_prior_latent128"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_name)
      ENV_NAME="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --msbm_run_dir)
      MSBM_RUN_DIR="$2"
      shift 2
      ;;
    --latent_space_split|--latent_geom_split)
      LATENT_SPACE_SPLIT="$2"
      shift 2
      ;;
    --trajectory_samples)
      TRAJECTORY_SAMPLES="$2"
      shift 2
      ;;
    --eval_realizations)
      EVAL_REALIZATIONS="$2"
      shift 2
      ;;
    --sample_idx)
      SAMPLE_IDX="$2"
      shift 2
      ;;
    --drift_clip_norm)
      DRIFT_CLIP_NORM="$2"
      shift 2
      ;;
    --no_drift_clip)
      DRIFT_CLIP_NORM=""
      shift
      ;;
    --force_recompute)
      REUSE_EXISTING="0"
      shift
      ;;
    --no_reuse_existing)
      REUSE_EXISTING="0"
      shift
      ;;
    --no_use_ema)
      USE_EMA="0"
      shift
      ;;
    --refresh_plots_only)
      REFRESH_PLOTS_ONLY="1"
      shift
      ;;
    --skip_latent_space)
      RUN_LATENT_SPACE="0"
      shift
      ;;
    --trajectory_eval_only)
      RUN_TRAJECTORY_EVAL_ONLY="1"
      RUN_LATENT_SPACE="0"
      RUN_POSTFILTERED_EVAL="0"
      RUN_POSTFILTERED_EVAL_ONLY="0"
      shift
      ;;
    --run_postfiltered_eval)
      RUN_POSTFILTERED_EVAL="1"
      shift
      ;;
    --postfiltered_eval_only)
      RUN_POSTFILTERED_EVAL="1"
      RUN_POSTFILTERED_EVAL_ONLY="1"
      RUN_TRAJECTORY_EVAL_ONLY="0"
      RUN_LATENT_SPACE="0"
      shift
      ;;
    --conditional_corpus_path)
      CONDITIONAL_CORPUS_PATH="$2"
      shift 2
      ;;
    --knn_reference_corpus_latents_path)
      CONDITIONAL_CORPUS_LATENTS_PATH="$2"
      shift 2
      ;;
    --knn_reference_k_neighbors)
      CONDITIONAL_K_NEIGHBORS="$2"
      shift 2
      ;;
    --conditional_min_spacing_pixels)
      CONDITIONAL_MIN_SPACING_PIXELS="$2"
      shift 2
      ;;
    --msbm_only)
      RUN_LATENT_SPACE="0"
      shift
      ;;
    --skip_latent_geometry|--skip_eval_latent_geometry|--latent_geom_budget|--latent_geom_trace_estimator|--no_check_existing_latent_geometry|--no_umap_manifold)
      IGNORED_DEPRECATED_FLAGS+=("$1")
      if [[ "$1" == "--latent_geom_budget" || "$1" == "--latent_geom_trace_estimator" ]]; then
        shift 2
      else
        shift
      fi
      ;;
    --run_conditional_eval|--skip_conditional_projection_plots|--conditional_projection_plot_conditions|--conditional_projection_pair_indices|--conditional_test_samples|--knn_reference_realizations|--conditional_pdf_values_per_sample)
      REMOVED_FEATURE_REQUESTS+=("$1")
      if [[ "$1" == "--conditional_projection_plot_conditions" || "$1" == "--conditional_projection_pair_indices" || "$1" == "--conditional_test_samples" || "$1" == "--knn_reference_realizations" || "$1" == "--conditional_pdf_values_per_sample" ]]; then
        shift 2
      else
        shift
      fi
      ;;
    -h|--help)
      sed -n '1,220p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_ROOT"

run_in_env() {
  conda run -n "$ENV_NAME" "$@"
}

have_files() {
  local path
  for path in "$@"; do
    [[ -f "$path" ]] || return 1
  done
  return 0
}

stamp_matches() {
  local stamp_path="$1"
  shift
  local expected
  [[ -f "$stamp_path" ]] || return 1
  for expected in "$@"; do
    grep -Fxq "$expected" "$stamp_path" || return 1
  done
  return 0
}

write_stamp() {
  local stamp_path="$1"
  shift
  printf '%s\n' "$@" > "$stamp_path"
}

resolve_run_dir() {
  local root="$1"
  run_in_env python -c '
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve()
if not root.exists():
    raise SystemExit(f"Run root does not exist: {root}")

def has_checkpoint(path: Path) -> bool:
    return any((path / "checkpoints" / name).exists() for name in ("best_state.pkl", "state.pkl"))

if (root / "args.json").exists() and has_checkpoint(root):
    print(root)
    raise SystemExit(0)

candidates = []
for child in sorted(root.iterdir()):
    if not child.is_dir():
        continue
    if not (child / "args.json").exists() or not has_checkpoint(child):
        continue
    eval_file = child / "eval_results.json"
    score = (
        1 if eval_file.exists() else 0,
        eval_file.stat().st_mtime if eval_file.exists() else (child / "args.json").stat().st_mtime,
    )
    candidates.append((score, child))

if not candidates:
    raise SystemExit(f"No completed run directory found under {root}")

candidates.sort(key=lambda item: item[0], reverse=True)
print(candidates[0][1].resolve())
' "$root"
}

if [[ ${#REMOVED_FEATURE_REQUESTS[@]} -gt 0 ]]; then
  echo "This historical compatibility bundle no longer supports the requested removed surfaces: ${REMOVED_FEATURE_REQUESTS[*]}" >&2
  echo "Use the active CSP evaluation stack instead:" >&2
  echo "  - scripts/csp/evaluate_csp_conditional_rollout.py" >&2
  echo "  - scripts/csp/evaluate_csp.py" >&2
  echo "  - scripts/csp/plot_latent_trajectories.py" >&2
  exit 1
fi

echo "============================================"
echo "Historical latent128 + latent-MSBM bundle"
echo "============================================"
echo "Environment                : $ENV_NAME"
echo "Repo root                  : $REPO_ROOT"
echo "Output root                : $OUTPUT_ROOT"
echo "Latent MSBM run            : $MSBM_RUN_DIR"
echo "Latent-space split         : $LATENT_SPACE_SPLIT"
echo "Reuse existing outputs     : $REUSE_EXISTING"
echo "Use EMA checkpoints        : $USE_EMA"
echo "Drift clip norm            : ${DRIFT_CLIP_NORM:-none}"
echo "Refresh plots only         : $REFRESH_PLOTS_ONLY"
echo "Run latent space           : $RUN_LATENT_SPACE"
echo "Trajectory eval only       : $RUN_TRAJECTORY_EVAL_ONLY"
echo "Run postfiltered eval      : $RUN_POSTFILTERED_EVAL"
echo "Postfiltered eval only     : $RUN_POSTFILTERED_EVAL_ONLY"
echo "Trajectory samples         : $TRAJECTORY_SAMPLES"
echo "Evaluation realizations    : $EVAL_REALIZATIONS"
echo "Sample index               : $SAMPLE_IDX"
if [[ "$RUN_POSTFILTERED_EVAL" == "1" ]]; then
  echo "Conditional corpus         : $CONDITIONAL_CORPUS_PATH"
  echo "Conditional corpus latents : $CONDITIONAL_CORPUS_LATENTS_PATH"
  echo "Conditional k-neighbors    : $CONDITIONAL_K_NEIGHBORS"
fi
if [[ ${#IGNORED_DEPRECATED_FLAGS[@]} -gt 0 ]]; then
  echo "Ignored deprecated flags   : ${IGNORED_DEPRECATED_FLAGS[*]}"
fi

RESOLVED_FAE_RUNS=()
if [[ "$RUN_LATENT_SPACE" == "1" ]]; then
  for root in "${FAE_ROOTS[@]}"; do
    resolved="$(resolve_run_dir "$root")"
    RESOLVED_FAE_RUNS+=("$resolved")
    echo "Resolved FAE run           : $root -> $resolved"
  done
fi

LATENT_SPACE_OUT="$OUTPUT_ROOT/latent_space"
MSBM_EVAL_OUT_BASE="$OUTPUT_ROOT/latent_msbm/tran_evaluation"
MSBM_EVAL_OUT_TRAJECTORY="$OUTPUT_ROOT/latent_msbm/tran_evaluation_trajectory_unconditional"
if [[ "$RUN_TRAJECTORY_EVAL_ONLY" == "1" ]]; then
  MSBM_EVAL_OUT="$MSBM_EVAL_OUT_TRAJECTORY"
else
  MSBM_EVAL_OUT="$MSBM_EVAL_OUT_BASE"
fi
MSBM_POSTFILTERED_UNCONDITIONAL_OUT="$OUTPUT_ROOT/latent_msbm/postfiltered_unconditional"
MSBM_POSTFILTERED_CONDITIONAL_OUT="$OUTPUT_ROOT/latent_msbm/postfiltered_conditional"
TRAJ_FILE="$MSBM_RUN_DIR/publication_full_trajectories.npz"
TRAJ_STAMP="$MSBM_RUN_DIR/publication_full_trajectories.meta"
EVAL_STAMP="$MSBM_EVAL_OUT/evaluation.meta"
POSTFILTERED_UNCONDITIONAL_STAMP="$MSBM_POSTFILTERED_UNCONDITIONAL_OUT/postfiltered.meta"
POSTFILTERED_CONDITIONAL_STAMP="$MSBM_POSTFILTERED_CONDITIONAL_OUT/postfiltered.meta"
EVAL_DATA_CACHE="$MSBM_EVAL_OUT/generated_realizations.npz"

mkdir -p \
  "$LATENT_SPACE_OUT" \
  "$MSBM_EVAL_OUT" \
  "$MSBM_POSTFILTERED_UNCONDITIONAL_OUT" \
  "$MSBM_POSTFILTERED_CONDITIONAL_OUT"

if [[ "$RUN_LATENT_SPACE" == "1" ]]; then
  LATENT_SPACE_CMD=(
    python scripts/fae/tran_evaluation/visualize_autoencoder_latent_space.py
    --output_dir "$LATENT_SPACE_OUT"
    --split "$LATENT_SPACE_SPLIT"
  )
  for run_dir in "${RESOLVED_FAE_RUNS[@]}"; do
    LATENT_SPACE_CMD+=(--run_dir "$run_dir")
  done
fi

echo
echo "--- Step 1: Per-run latent PCA scatter plots ---"
if [[ "$RUN_TRAJECTORY_EVAL_ONLY" == "1" || "$RUN_POSTFILTERED_EVAL_ONLY" == "1" ]]; then
  echo "Skipping latent-space comparison step"
elif [[ "$RUN_LATENT_SPACE" != "1" ]]; then
  echo "Skipping latent-space comparison step"
elif [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
  "$LATENT_SPACE_OUT/latent_space_viz_summary.json" \
  "$LATENT_SPACE_OUT/latent_pca2_scatter_grid.png"; then
  echo "Reusing existing latent-space outputs in $LATENT_SPACE_OUT"
else
  run_in_env "${LATENT_SPACE_CMD[@]}"
fi

echo
echo "--- Step 2: Generate publication trajectory bundle ---"
TRAJ_STAMP_LINES=(
  "run_dir=$MSBM_RUN_DIR"
  "n_samples=$TRAJECTORY_SAMPLES"
  "use_ema=$USE_EMA"
  "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
)
if [[ "$RUN_POSTFILTERED_EVAL_ONLY" == "1" ]]; then
  echo "Skipping publication trajectory bundle step"
elif [[ "$REUSE_EXISTING" == "1" ]] && have_files "$TRAJ_FILE" && stamp_matches "$TRAJ_STAMP" "${TRAJ_STAMP_LINES[@]}"; then
  echo "Reusing existing trajectory bundle $TRAJ_FILE"
else
  TRAJ_CMD=(
    python scripts/fae/generate_full_trajectories.py
    --run_dir "$MSBM_RUN_DIR"
    --n_samples "$TRAJECTORY_SAMPLES"
    --direction both
    --save_decoded
    --output_name "$(basename "$TRAJ_FILE")"
  )
  if [[ "$USE_EMA" == "0" ]]; then
    TRAJ_CMD+=(--no_use_ema)
  fi
  if [[ -n "$DRIFT_CLIP_NORM" ]]; then
    TRAJ_CMD+=(--drift_clip_norm "$DRIFT_CLIP_NORM")
  fi
  run_in_env "${TRAJ_CMD[@]}"
  write_stamp "$TRAJ_STAMP" "${TRAJ_STAMP_LINES[@]}"
fi

echo
echo "--- Step 3: Visualize latent and decoded trajectories ---"
if [[ "$RUN_POSTFILTERED_EVAL_ONLY" == "1" ]]; then
  echo "Skipping publication trajectory visualization step"
elif [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
  "$MSBM_RUN_DIR/eval/latent_viz/latent_manifold_2d.png" \
  "$MSBM_RUN_DIR/field_viz/backward_vs_reference.png" \
  && stamp_matches "$TRAJ_STAMP" "${TRAJ_STAMP_LINES[@]}"; then
  echo "Reusing existing trajectory visualizations under $MSBM_RUN_DIR"
else
  run_in_env env MSBM_DIR="$MSBM_RUN_DIR" python notebooks/fae_latent_msbm_latent_viz.py
  run_in_env env TRAJ_PATH="$TRAJ_FILE" python notebooks/visualize_full_trajectories.py
  run_in_env env TRAJ_PATH="$TRAJ_FILE" python notebooks/visualize_field_trajectories.py
fi

echo
echo "--- Step 4: Tran-aligned MSBM evaluation figures and tables ---"
if [[ "$RUN_POSTFILTERED_EVAL_ONLY" == "1" ]]; then
  echo "Skipping Tran-aligned evaluation step"
elif [[ "$RUN_TRAJECTORY_EVAL_ONLY" == "1" ]]; then
  EVAL_STAMP_LINES=(
    "run_dir=$MSBM_RUN_DIR"
    "n_realizations=$EVAL_REALIZATIONS"
    "trajectory_only=1"
    "trajectory_seed_mode=marginal"
    "use_ema=$USE_EMA"
    "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
  )
  if [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
    "$MSBM_EVAL_OUT/metrics.json" \
    "$MSBM_EVAL_OUT/trajectory_summary.txt" \
    "$MSBM_EVAL_OUT/fig13_trajectory_pdfs.png" \
    "$MSBM_EVAL_OUT/fig14_trajectory_correlation.png" \
    && stamp_matches "$EVAL_STAMP" "${EVAL_STAMP_LINES[@]}"; then
    echo "Reusing unconditional trajectory evaluation outputs in $MSBM_EVAL_OUT"
  else
    EVAL_CMD=(
      python scripts/fae/tran_evaluation/evaluate.py
      --run_dir "$MSBM_RUN_DIR"
      --output_dir "$MSBM_EVAL_OUT"
      --generated_data_file "$EVAL_DATA_CACHE"
      --n_realizations "$EVAL_REALIZATIONS"
      --trajectory_only
      --trajectory_seed_mode marginal
      --no_latent_geometry
    )
    if [[ "$REFRESH_PLOTS_ONLY" == "1" ]]; then
      EVAL_CMD+=(--reuse_generated_data)
    fi
    if [[ "$USE_EMA" == "0" ]]; then
      EVAL_CMD+=(--no_use_ema)
    fi
    if [[ -n "$DRIFT_CLIP_NORM" ]]; then
      EVAL_CMD+=(--drift_clip_norm "$DRIFT_CLIP_NORM")
    fi
    run_in_env "${EVAL_CMD[@]}"
    write_stamp "$EVAL_STAMP" "${EVAL_STAMP_LINES[@]}"
  fi
else
  EVAL_STAMP_LINES=(
    "run_dir=$MSBM_RUN_DIR"
    "n_realizations=$EVAL_REALIZATIONS"
    "sample_idx=$SAMPLE_IDX"
    "use_ema=$USE_EMA"
    "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
    "latent_geometry=0"
  )
  if [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
    "$MSBM_EVAL_OUT/metrics.json" \
    "$MSBM_EVAL_OUT/trajectory_summary.txt" \
    "$MSBM_EVAL_OUT/fig12_trajectory_fields.png" \
    && stamp_matches "$EVAL_STAMP" "${EVAL_STAMP_LINES[@]}"; then
    echo "Reusing existing Tran-aligned evaluation outputs in $MSBM_EVAL_OUT"
  else
    EVAL_CMD=(
      python scripts/fae/tran_evaluation/evaluate.py
      --run_dir "$MSBM_RUN_DIR"
      --output_dir "$MSBM_EVAL_OUT"
      --generated_data_file "$EVAL_DATA_CACHE"
      --n_realizations "$EVAL_REALIZATIONS"
      --sample_idx "$SAMPLE_IDX"
      --no_latent_geometry
    )
    if [[ "$REFRESH_PLOTS_ONLY" == "1" ]]; then
      EVAL_CMD+=(--reuse_generated_data)
    fi
    if [[ "$USE_EMA" == "0" ]]; then
      EVAL_CMD+=(--no_use_ema)
    fi
    if [[ -n "$DRIFT_CLIP_NORM" ]]; then
      EVAL_CMD+=(--drift_clip_norm "$DRIFT_CLIP_NORM")
    fi
    run_in_env "${EVAL_CMD[@]}"
    write_stamp "$EVAL_STAMP" "${EVAL_STAMP_LINES[@]}"
  fi
fi

if [[ "$RUN_POSTFILTERED_EVAL" == "1" ]]; then
  echo
  echo "--- Step 5: Post-filtered conditional and unconditional consistency ---"

  POSTFILTERED_UNCONDITIONAL_STAMP_LINES=(
    "run_dir=$MSBM_RUN_DIR"
    "output_dir=$MSBM_POSTFILTERED_UNCONDITIONAL_OUT"
    "mode=unconditional"
    "n_realizations=$EVAL_REALIZATIONS"
    "n_gt_neighbors=$EVAL_REALIZATIONS"
    "use_ema=$USE_EMA"
    "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
    "min_spacing_pixels=$CONDITIONAL_MIN_SPACING_PIXELS"
  )
  if [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
    "$MSBM_POSTFILTERED_UNCONDITIONAL_OUT/metrics.json" \
    "$MSBM_POSTFILTERED_UNCONDITIONAL_OUT/summary.txt" \
    "$MSBM_POSTFILTERED_UNCONDITIONAL_OUT/fig_postfiltered_unconditional_pdfs.png" \
    "$MSBM_POSTFILTERED_UNCONDITIONAL_OUT/fig_postfiltered_unconditional_correlation.png" \
    && stamp_matches "$POSTFILTERED_UNCONDITIONAL_STAMP" "${POSTFILTERED_UNCONDITIONAL_STAMP_LINES[@]}"; then
    echo "Reusing unconditional post-filtered outputs in $MSBM_POSTFILTERED_UNCONDITIONAL_OUT"
  else
    POSTFILTERED_UNCONDITIONAL_CMD=(
      python scripts/fae/tran_evaluation/evaluate_postfiltered_consistency.py
      --run_dir "$MSBM_RUN_DIR"
      --output_dir "$MSBM_POSTFILTERED_UNCONDITIONAL_OUT"
      --mode unconditional
      --n_realizations "$EVAL_REALIZATIONS"
      --n_gt_neighbors "$EVAL_REALIZATIONS"
      --min_spacing_pixels "$CONDITIONAL_MIN_SPACING_PIXELS"
    )
    if [[ "$USE_EMA" == "0" ]]; then
      POSTFILTERED_UNCONDITIONAL_CMD+=(--no_use_ema)
    fi
    if [[ -n "$DRIFT_CLIP_NORM" ]]; then
      POSTFILTERED_UNCONDITIONAL_CMD+=(--drift_clip_norm "$DRIFT_CLIP_NORM")
    fi
    run_in_env "${POSTFILTERED_UNCONDITIONAL_CMD[@]}"
    write_stamp "$POSTFILTERED_UNCONDITIONAL_STAMP" "${POSTFILTERED_UNCONDITIONAL_STAMP_LINES[@]}"
  fi

  POSTFILTERED_CONDITIONAL_STAMP_LINES=(
    "run_dir=$MSBM_RUN_DIR"
    "output_dir=$MSBM_POSTFILTERED_CONDITIONAL_OUT"
    "mode=conditional"
    "sample_idx=$SAMPLE_IDX"
    "n_realizations=$EVAL_REALIZATIONS"
    "corpus_path=$CONDITIONAL_CORPUS_PATH"
    "corpus_latents_path=$CONDITIONAL_CORPUS_LATENTS_PATH"
    "k_neighbors=$CONDITIONAL_K_NEIGHBORS"
    "use_ema=$USE_EMA"
    "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
    "min_spacing_pixels=$CONDITIONAL_MIN_SPACING_PIXELS"
  )
  if [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
    "$MSBM_POSTFILTERED_CONDITIONAL_OUT/metrics.json" \
    "$MSBM_POSTFILTERED_CONDITIONAL_OUT/summary.txt" \
    "$MSBM_POSTFILTERED_CONDITIONAL_OUT/fig_postfiltered_conditional_pdfs.png" \
    "$MSBM_POSTFILTERED_CONDITIONAL_OUT/fig_postfiltered_conditional_correlation.png" \
    && stamp_matches "$POSTFILTERED_CONDITIONAL_STAMP" "${POSTFILTERED_CONDITIONAL_STAMP_LINES[@]}"; then
    echo "Reusing conditional post-filtered outputs in $MSBM_POSTFILTERED_CONDITIONAL_OUT"
  else
    POSTFILTERED_CONDITIONAL_CMD=(
      python scripts/fae/tran_evaluation/evaluate_postfiltered_consistency.py
      --run_dir "$MSBM_RUN_DIR"
      --output_dir "$MSBM_POSTFILTERED_CONDITIONAL_OUT"
      --mode conditional
      --sample_idx "$SAMPLE_IDX"
      --n_realizations "$EVAL_REALIZATIONS"
      --corpus_path "$CONDITIONAL_CORPUS_PATH"
      --corpus_latents_path "$CONDITIONAL_CORPUS_LATENTS_PATH"
      --k_neighbors "$CONDITIONAL_K_NEIGHBORS"
      --min_spacing_pixels "$CONDITIONAL_MIN_SPACING_PIXELS"
    )
    if [[ "$USE_EMA" == "0" ]]; then
      POSTFILTERED_CONDITIONAL_CMD+=(--no_use_ema)
    fi
    if [[ -n "$DRIFT_CLIP_NORM" ]]; then
      POSTFILTERED_CONDITIONAL_CMD+=(--drift_clip_norm "$DRIFT_CLIP_NORM")
    fi
    run_in_env "${POSTFILTERED_CONDITIONAL_CMD[@]}"
    write_stamp "$POSTFILTERED_CONDITIONAL_STAMP" "${POSTFILTERED_CONDITIONAL_STAMP_LINES[@]}"
  fi
fi

echo
echo "============================================"
echo "Historical compatibility outputs ready"
echo "============================================"
echo "Latent PCA plots             : $LATENT_SPACE_OUT"
echo "Trajectory bundle            : $TRAJ_FILE"
echo "Latent trajectory viz        : $MSBM_RUN_DIR/eval/latent_viz"
echo "Full trajectory viz          : $MSBM_RUN_DIR/full_traj_viz"
echo "Field trajectory viz         : $MSBM_RUN_DIR/field_viz"
echo "Tran evaluation plots/tables : $MSBM_EVAL_OUT"
if [[ "$RUN_POSTFILTERED_EVAL" == "1" ]]; then
  echo "Postfiltered unconditional   : $MSBM_POSTFILTERED_UNCONDITIONAL_OUT"
  echo "Postfiltered conditional     : $MSBM_POSTFILTERED_CONDITIONAL_OUT"
fi
