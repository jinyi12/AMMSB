#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-3MASB}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/publication_latent128_and_msbm}"
MSBM_RUN_DIR="${MSBM_RUN_DIR:-/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior}"
LATENT_GEOM_BUDGET="${LATENT_GEOM_BUDGET:-thorough}"
LATENT_GEOM_TRACE_ESTIMATOR="${LATENT_GEOM_TRACE_ESTIMATOR:-fhutch}"
LATENT_GEOM_SPLIT="${LATENT_GEOM_SPLIT:-test}"
TRAJECTORY_SAMPLES="${TRAJECTORY_SAMPLES:-32}"
EVAL_REALIZATIONS="${EVAL_REALIZATIONS:-200}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
DRIFT_CLIP_NORM="${DRIFT_CLIP_NORM:-}"
FORCE_RECOMPUTE="0"
CHECK_EXISTING_LATENT_GEOM="1"
REUSE_EXISTING="1"
USE_EMA="1"
REFRESH_PLOTS_ONLY="0"
USE_UMAP_MANIFOLD="1"
RUN_LATENT_GEOMETRY="1"
RUN_LATENT_SPACE="1"
RUN_EVAL_LATENT_GEOMETRY="1"
RUN_CONDITIONAL_EVAL="0"
RUN_TRAJECTORY_EVAL_ONLY="0"
RUN_POSTFILTERED_EVAL="0"
RUN_POSTFILTERED_EVAL_ONLY="0"
CONDITIONAL_CORPUS_PATH="${CONDITIONAL_CORPUS_PATH:-data/fae_tran_inclusions_corpus.npz}"
CONDITIONAL_CORPUS_LATENTS_PATH="${CONDITIONAL_CORPUS_LATENTS_PATH:-data/corpus_latents_ntk_prior.npz}"
CONDITIONAL_K_NEIGHBORS="${CONDITIONAL_K_NEIGHBORS:-200}"
CONDITIONAL_TEST_SAMPLES="${CONDITIONAL_TEST_SAMPLES:-50}"
CONDITIONAL_REALIZATIONS="${CONDITIONAL_REALIZATIONS:-200}"
CONDITIONAL_PDF_VALUES_PER_SAMPLE="${CONDITIONAL_PDF_VALUES_PER_SAMPLE:-4000}"
CONDITIONAL_MIN_SPACING_PIXELS="${CONDITIONAL_MIN_SPACING_PIXELS:-4}"
CONDITIONAL_PROJECTION_PLOTS="1"
CONDITIONAL_PROJECTION_PLOT_CONDITIONS="${CONDITIONAL_PROJECTION_PLOT_CONDITIONS:-6}"
CONDITIONAL_PROJECTION_PAIR_INDICES="${CONDITIONAL_PROJECTION_PAIR_INDICES:-}"

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
    --latent_geom_budget)
      LATENT_GEOM_BUDGET="$2"
      shift 2
      ;;
    --latent_geom_trace_estimator)
      LATENT_GEOM_TRACE_ESTIMATOR="$2"
      shift 2
      ;;
    --latent_geom_split)
      LATENT_GEOM_SPLIT="$2"
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
      FORCE_RECOMPUTE="1"
      shift
      ;;
    --no_check_existing_latent_geometry)
      CHECK_EXISTING_LATENT_GEOM="0"
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
    --no_umap_manifold)
      USE_UMAP_MANIFOLD="0"
      shift
      ;;
    --skip_latent_geometry)
      RUN_LATENT_GEOMETRY="0"
      shift
      ;;
    --skip_latent_space)
      RUN_LATENT_SPACE="0"
      shift
      ;;
    --skip_eval_latent_geometry)
      RUN_EVAL_LATENT_GEOMETRY="0"
      shift
      ;;
    --run_conditional_eval)
      RUN_CONDITIONAL_EVAL="1"
      shift
      ;;
    --skip_conditional_projection_plots)
      CONDITIONAL_PROJECTION_PLOTS="0"
      shift
      ;;
    --conditional_projection_plot_conditions)
      CONDITIONAL_PROJECTION_PLOT_CONDITIONS="$2"
      shift 2
      ;;
    --conditional_projection_pair_indices)
      CONDITIONAL_PROJECTION_PAIR_INDICES="$2"
      shift 2
      ;;
    --trajectory_eval_only)
      RUN_TRAJECTORY_EVAL_ONLY="1"
      RUN_LATENT_GEOMETRY="0"
      RUN_LATENT_SPACE="0"
      RUN_EVAL_LATENT_GEOMETRY="0"
      RUN_CONDITIONAL_EVAL="0"
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
      RUN_LATENT_GEOMETRY="0"
      RUN_LATENT_SPACE="0"
      RUN_EVAL_LATENT_GEOMETRY="0"
      RUN_CONDITIONAL_EVAL="0"
      shift
      ;;
    --conditional_corpus_path)
      CONDITIONAL_CORPUS_PATH="$2"
      shift 2
      ;;
    --conditional_corpus_latents_path)
      CONDITIONAL_CORPUS_LATENTS_PATH="$2"
      shift 2
      ;;
    --conditional_k_neighbors)
      CONDITIONAL_K_NEIGHBORS="$2"
      shift 2
      ;;
    --conditional_test_samples)
      CONDITIONAL_TEST_SAMPLES="$2"
      shift 2
      ;;
    --conditional_realizations)
      CONDITIONAL_REALIZATIONS="$2"
      shift 2
      ;;
    --conditional_pdf_values_per_sample)
      CONDITIONAL_PDF_VALUES_PER_SAMPLE="$2"
      shift 2
      ;;
    --conditional_min_spacing_pixels)
      CONDITIONAL_MIN_SPACING_PIXELS="$2"
      shift 2
      ;;
    --msbm_only)
      RUN_LATENT_GEOMETRY="0"
      RUN_LATENT_SPACE="0"
      RUN_EVAL_LATENT_GEOMETRY="0"
      shift
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
for child in sorted(root.glob("run_*")):
    if not (child / "args.json").exists() or not has_checkpoint(child):
        continue
    eval_file = child / "eval_results.json"
    score = (1 if eval_file.exists() else 0, eval_file.stat().st_mtime if eval_file.exists() else (child / "args.json").stat().st_mtime)
    candidates.append((score, child))

if not candidates:
    raise SystemExit(f"No completed run_* directory found under {root}")

candidates.sort(key=lambda item: item[0], reverse=True)
print(candidates[0][1].resolve())
' "$root"
}

echo "============================================"
echo "Publication pipeline"
echo "============================================"
echo "Environment                : $ENV_NAME"
echo "Repo root                  : $REPO_ROOT"
echo "Output root                : $OUTPUT_ROOT"
echo "Latent MSBM run            : $MSBM_RUN_DIR"
echo "Latent geometry budget     : $LATENT_GEOM_BUDGET"
echo "Latent geometry estimator  : $LATENT_GEOM_TRACE_ESTIMATOR"
echo "Check existing geom data   : $CHECK_EXISTING_LATENT_GEOM"
echo "Reuse existing outputs     : $REUSE_EXISTING"
echo "Use EMA checkpoints        : $USE_EMA"
echo "Drift clip norm            : ${DRIFT_CLIP_NORM:-none}"
echo "Use UMAP companion         : $USE_UMAP_MANIFOLD"
echo "Refresh plots only         : $REFRESH_PLOTS_ONLY"
echo "Run latent geometry        : $RUN_LATENT_GEOMETRY"
echo "Run latent space           : $RUN_LATENT_SPACE"
echo "Run eval latent geometry   : $RUN_EVAL_LATENT_GEOMETRY"
echo "Run conditional eval       : $RUN_CONDITIONAL_EVAL"
echo "Trajectory eval only       : $RUN_TRAJECTORY_EVAL_ONLY"
echo "Run postfiltered eval      : $RUN_POSTFILTERED_EVAL"
echo "Postfiltered eval only     : $RUN_POSTFILTERED_EVAL_ONLY"
echo "Trajectory samples         : $TRAJECTORY_SAMPLES"
echo "Evaluation realizations    : $EVAL_REALIZATIONS"
echo "Sample index               : $SAMPLE_IDX"
if [[ "$RUN_CONDITIONAL_EVAL" == "1" || "$RUN_POSTFILTERED_EVAL" == "1" ]]; then
  echo "Conditional corpus         : $CONDITIONAL_CORPUS_PATH"
  echo "Conditional corpus latents : $CONDITIONAL_CORPUS_LATENTS_PATH"
  echo "Conditional k-neighbors    : $CONDITIONAL_K_NEIGHBORS"
fi
if [[ "$RUN_CONDITIONAL_EVAL" == "1" ]]; then
  echo "Conditional test samples   : $CONDITIONAL_TEST_SAMPLES"
  echo "Conditional realizations   : $CONDITIONAL_REALIZATIONS"
  echo "Conditional projections    : $CONDITIONAL_PROJECTION_PLOTS"
  if [[ "$CONDITIONAL_PROJECTION_PLOTS" == "1" ]]; then
    echo "Projection plot count      : $CONDITIONAL_PROJECTION_PLOT_CONDITIONS"
  fi
fi

RESOLVED_FAE_RUNS=()
if [[ "$RUN_LATENT_GEOMETRY" == "1" || "$RUN_LATENT_SPACE" == "1" ]]; then
  for root in "${FAE_ROOTS[@]}"; do
    resolved="$(resolve_run_dir "$root")"
    RESOLVED_FAE_RUNS+=("$resolved")
    echo "Resolved FAE run           : $root -> $resolved"
  done
fi

GEOM_OUT="$OUTPUT_ROOT/latent_geometry"
LATENT_SPACE_OUT="$OUTPUT_ROOT/latent_space"
MSBM_EVAL_OUT_BASE="$OUTPUT_ROOT/latent_msbm/tran_evaluation"
MSBM_EVAL_OUT_TRAJECTORY="$OUTPUT_ROOT/latent_msbm/tran_evaluation_trajectory_unconditional"
if [[ "$RUN_TRAJECTORY_EVAL_ONLY" == "1" ]]; then
  MSBM_EVAL_OUT="$MSBM_EVAL_OUT_TRAJECTORY"
else
MSBM_EVAL_OUT="$MSBM_EVAL_OUT_BASE"
fi
MSBM_CONDITIONAL_OUT="$OUTPUT_ROOT/latent_msbm/conditional"
MSBM_CONDITIONAL_PROJECTION_OUT="$MSBM_CONDITIONAL_OUT/projections"
MSBM_POSTFILTERED_UNCONDITIONAL_OUT="$OUTPUT_ROOT/latent_msbm/postfiltered_unconditional"
MSBM_POSTFILTERED_CONDITIONAL_OUT="$OUTPUT_ROOT/latent_msbm/postfiltered_conditional"
TRAJ_FILE="$MSBM_RUN_DIR/publication_full_trajectories.npz"
TRAJ_STAMP="$MSBM_RUN_DIR/publication_full_trajectories.meta"
EVAL_STAMP="$MSBM_EVAL_OUT/evaluation.meta"
CONDITIONAL_STAMP="$MSBM_CONDITIONAL_OUT/conditional.meta"
CONDITIONAL_PROJECTION_STAMP="$MSBM_CONDITIONAL_PROJECTION_OUT/projection.meta"
POSTFILTERED_UNCONDITIONAL_STAMP="$MSBM_POSTFILTERED_UNCONDITIONAL_OUT/postfiltered.meta"
POSTFILTERED_CONDITIONAL_STAMP="$MSBM_POSTFILTERED_CONDITIONAL_OUT/postfiltered.meta"
EVAL_DATA_CACHE="$MSBM_EVAL_OUT/generated_realizations.npz"

mkdir -p \
  "$GEOM_OUT" \
  "$LATENT_SPACE_OUT" \
  "$MSBM_EVAL_OUT" \
  "$MSBM_CONDITIONAL_OUT" \
  "$MSBM_CONDITIONAL_PROJECTION_OUT" \
  "$MSBM_POSTFILTERED_UNCONDITIONAL_OUT" \
  "$MSBM_POSTFILTERED_CONDITIONAL_OUT"

if [[ "$RUN_LATENT_GEOMETRY" == "1" ]]; then
  GEOM_CMD=(
    python scripts/fae/tran_evaluation/compare_latent_geometry_models.py
    --output_dir "$GEOM_OUT"
    --effect_baseline_scope manual
    --effect_scale_scope multi_1248
    --latent_geom_budget "$LATENT_GEOM_BUDGET"
    --latent_geom_trace_estimator "$LATENT_GEOM_TRACE_ESTIMATOR"
    --latent_geom_split "$LATENT_GEOM_SPLIT"
  )
  if [[ "$FORCE_RECOMPUTE" == "1" ]]; then
    GEOM_CMD+=(--force_recompute)
  fi
  if [[ "$CHECK_EXISTING_LATENT_GEOM" == "0" ]]; then
    GEOM_CMD+=(--no_check_existing_metrics)
  fi
  for run_dir in "${RESOLVED_FAE_RUNS[@]}"; do
    if [[ -z "$run_dir" ]]; then
      echo "Failed to resolve a concrete FAE run directory." >&2
      exit 1
    fi
    GEOM_CMD+=(--run_dir "$run_dir")
  done
fi

if [[ "$RUN_LATENT_SPACE" == "1" ]]; then
  LATENT_SPACE_CMD=(
    python scripts/fae/tran_evaluation/visualize_autoencoder_latent_space.py
    --output_dir "$LATENT_SPACE_OUT"
    --split "$LATENT_GEOM_SPLIT"
  )
  for run_dir in "${RESOLVED_FAE_RUNS[@]}"; do
    LATENT_SPACE_CMD+=(--run_dir "$run_dir")
  done
fi

echo
echo "--- Step 1: Latent-geometry comparison plots and tables ---"
if [[ "$RUN_TRAJECTORY_EVAL_ONLY" == "1" || "$RUN_POSTFILTERED_EVAL_ONLY" == "1" ]]; then
  echo "Skipping latent-geometry comparison step"
elif [[ "$RUN_LATENT_GEOMETRY" != "1" ]]; then
  echo "Skipping latent-geometry comparison step"
elif [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
    "$GEOM_OUT/latent_geom_model_summary.json" \
    "$GEOM_OUT/latent_geom_model_metric_trace_g.png" \
    "$GEOM_OUT/latent_geom_l2_ntk_prior_chain_trace_g.png"; then
  echo "Reusing existing latent-geometry comparison outputs in $GEOM_OUT"
else
  run_in_env "${GEOM_CMD[@]}"
fi

echo
echo "--- Step 2: Per-run latent PCA scatter plots ---"
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
echo "--- Step 3: Generate publication trajectory bundle ---"
TRAJ_STAMP_LINES=(
  "run_dir=$MSBM_RUN_DIR"
  "n_samples=$TRAJECTORY_SAMPLES"
  "use_ema=$USE_EMA"
  "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
)
if [[ "$RUN_TRAJECTORY_EVAL_ONLY" == "1" || "$RUN_POSTFILTERED_EVAL_ONLY" == "1" ]]; then
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
echo "--- Step 4: Visualize latent and decoded trajectories ---"
if [[ "$RUN_TRAJECTORY_EVAL_ONLY" == "1" || "$RUN_POSTFILTERED_EVAL_ONLY" == "1" ]]; then
  echo "Skipping publication trajectory visualization step"
elif [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
  "$MSBM_RUN_DIR/eval/latent_viz/latent_manifold_2d.png" \
  "$MSBM_RUN_DIR/full_traj_viz/latent_manifold_only_2d.png" \
  "$MSBM_RUN_DIR/full_traj_viz/latent_manifold_backward_publication.png" \
  "$MSBM_RUN_DIR/field_viz/backward_vs_reference.png" \
  && stamp_matches "$TRAJ_STAMP" "${TRAJ_STAMP_LINES[@]}"; then
  echo "Reusing existing trajectory visualizations under $MSBM_RUN_DIR"
else
  run_in_env env MSBM_DIR="$MSBM_RUN_DIR" python notebooks/fae_latent_msbm_latent_viz.py
  run_in_env env TRAJ_PATH="$TRAJ_FILE" python notebooks/visualize_full_trajectories.py
  MANIFOLD_CMD=(
    python scripts/fae/tran_evaluation/visualize_latent_msbm_manifold.py
    --traj_path "$TRAJ_FILE"
    --output_dir "$MSBM_RUN_DIR/full_traj_viz"
  )
  if [[ "$USE_UMAP_MANIFOLD" == "0" ]]; then
    MANIFOLD_CMD+=(--no_umap)
  fi
  run_in_env "${MANIFOLD_CMD[@]}"
  run_in_env env TRAJ_PATH="$TRAJ_FILE" python notebooks/visualize_field_trajectories.py
fi

echo
echo "--- Step 5: Tran-aligned MSBM evaluation figures and tables ---"
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
    "eval_latent_geometry=$RUN_EVAL_LATENT_GEOMETRY"
  )
  if [[ "$RUN_EVAL_LATENT_GEOMETRY" == "1" ]]; then
    EVAL_STAMP_LINES+=(
      "latent_geom_budget=$LATENT_GEOM_BUDGET"
      "latent_geom_trace_estimator=$LATENT_GEOM_TRACE_ESTIMATOR"
    )
  fi
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
    )
    if [[ "$RUN_EVAL_LATENT_GEOMETRY" == "1" ]]; then
      EVAL_CMD+=(
        --latent_geom_budget "$LATENT_GEOM_BUDGET"
        --latent_geom_trace_estimator "$LATENT_GEOM_TRACE_ESTIMATOR"
      )
    else
      EVAL_CMD+=(--no_latent_geometry)
    fi
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

if [[ "$RUN_CONDITIONAL_EVAL" == "1" ]]; then
  echo
  echo "--- Step 6: Conditional backward evaluation figures and tables ---"
  CONDITIONAL_STAMP_LINES=(
    "run_dir=$MSBM_RUN_DIR"
    "output_dir=$MSBM_CONDITIONAL_OUT"
    "corpus_path=$CONDITIONAL_CORPUS_PATH"
    "corpus_latents_path=$CONDITIONAL_CORPUS_LATENTS_PATH"
    "k_neighbors=$CONDITIONAL_K_NEIGHBORS"
    "n_test_samples=$CONDITIONAL_TEST_SAMPLES"
    "n_realizations=$CONDITIONAL_REALIZATIONS"
    "use_ema=$USE_EMA"
    "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
    "pdf_values_per_sample=$CONDITIONAL_PDF_VALUES_PER_SAMPLE"
    "min_spacing_pixels=$CONDITIONAL_MIN_SPACING_PIXELS"
  )
  if [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
    "$MSBM_CONDITIONAL_OUT/conditional_metrics.json" \
    "$MSBM_CONDITIONAL_OUT/conditional_summary.txt" \
    "$MSBM_CONDITIONAL_OUT/fig_conditional_pdfs.png" \
    && stamp_matches "$CONDITIONAL_STAMP" "${CONDITIONAL_STAMP_LINES[@]}"; then
    echo "Reusing existing conditional evaluation outputs in $MSBM_CONDITIONAL_OUT"
  else
    CONDITIONAL_CMD=(
      python scripts/fae/tran_evaluation/evaluate_conditional.py
      --run_dir "$MSBM_RUN_DIR"
      --output_dir "$MSBM_CONDITIONAL_OUT"
      --corpus_path "$CONDITIONAL_CORPUS_PATH"
      --corpus_latents_path "$CONDITIONAL_CORPUS_LATENTS_PATH"
      --k_neighbors "$CONDITIONAL_K_NEIGHBORS"
      --n_test_samples "$CONDITIONAL_TEST_SAMPLES"
      --n_realizations "$CONDITIONAL_REALIZATIONS"
      --pdf_values_per_sample "$CONDITIONAL_PDF_VALUES_PER_SAMPLE"
      --min_spacing_pixels "$CONDITIONAL_MIN_SPACING_PIXELS"
    )
    if [[ "$USE_EMA" == "0" ]]; then
      CONDITIONAL_CMD+=(--no_use_ema)
    fi
    if [[ -n "$DRIFT_CLIP_NORM" ]]; then
      CONDITIONAL_CMD+=(--drift_clip_norm "$DRIFT_CLIP_NORM")
    fi
    run_in_env "${CONDITIONAL_CMD[@]}"
    write_stamp "$CONDITIONAL_STAMP" "${CONDITIONAL_STAMP_LINES[@]}"
  fi

  if [[ "$CONDITIONAL_PROJECTION_PLOTS" == "1" ]]; then
    CONDITIONAL_PROJECTION_STAMP_LINES=(
      "run_dir=$MSBM_RUN_DIR"
      "output_dir=$MSBM_CONDITIONAL_PROJECTION_OUT"
      "corpus_latents_path=$CONDITIONAL_CORPUS_LATENTS_PATH"
      "k_neighbors=$CONDITIONAL_K_NEIGHBORS"
      "n_test_samples=$CONDITIONAL_TEST_SAMPLES"
      "n_realizations=$CONDITIONAL_REALIZATIONS"
      "n_plot_conditions=$CONDITIONAL_PROJECTION_PLOT_CONDITIONS"
      "pair_indices=${CONDITIONAL_PROJECTION_PAIR_INDICES:-all}"
      "use_ema=$USE_EMA"
      "drift_clip_norm=${DRIFT_CLIP_NORM:-none}"
    )
    if [[ "$REUSE_EXISTING" == "1" && "$REFRESH_PLOTS_ONLY" == "0" ]] && have_files \
      "$MSBM_CONDITIONAL_PROJECTION_OUT/projection_summary.json" \
      && stamp_matches "$CONDITIONAL_PROJECTION_STAMP" "${CONDITIONAL_PROJECTION_STAMP_LINES[@]}"; then
      echo "Reusing conditional projection publication outputs in $MSBM_CONDITIONAL_PROJECTION_OUT"
    else
      CONDITIONAL_PROJECTION_CMD=(
        python scripts/fae/tran_evaluation/visualize_conditional_latent_projections.py
        --run_dir "$MSBM_RUN_DIR"
        --output_dir "$MSBM_CONDITIONAL_PROJECTION_OUT"
        --corpus_latents_path "$CONDITIONAL_CORPUS_LATENTS_PATH"
        --k_neighbors "$CONDITIONAL_K_NEIGHBORS"
        --n_test_samples "$CONDITIONAL_TEST_SAMPLES"
        --n_realizations "$CONDITIONAL_REALIZATIONS"
        --n_plot_conditions "$CONDITIONAL_PROJECTION_PLOT_CONDITIONS"
      )
      if [[ -n "$CONDITIONAL_PROJECTION_PAIR_INDICES" ]]; then
        CONDITIONAL_PROJECTION_CMD+=(--pair_indices "$CONDITIONAL_PROJECTION_PAIR_INDICES")
      fi
      if [[ "$USE_EMA" == "0" ]]; then
        CONDITIONAL_PROJECTION_CMD+=(--no_use_ema)
      fi
      if [[ -n "$DRIFT_CLIP_NORM" ]]; then
        CONDITIONAL_PROJECTION_CMD+=(--drift_clip_norm "$DRIFT_CLIP_NORM")
      fi
      run_in_env "${CONDITIONAL_PROJECTION_CMD[@]}"
      write_stamp "$CONDITIONAL_PROJECTION_STAMP" "${CONDITIONAL_PROJECTION_STAMP_LINES[@]}"
    fi
  fi
fi

if [[ "$RUN_POSTFILTERED_EVAL" == "1" ]]; then
  echo
  echo "--- Step 7: Post-filtered conditional and unconditional consistency ---"

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
echo "Publication outputs ready"
echo "============================================"
echo "Latent geometry plots/tables : $GEOM_OUT"
echo "Latent PCA plots             : $LATENT_SPACE_OUT"
echo "Trajectory bundle            : $TRAJ_FILE"
echo "Latent trajectory viz        : $MSBM_RUN_DIR/eval/latent_viz"
echo "Full trajectory viz          : $MSBM_RUN_DIR/full_traj_viz"
echo "Field trajectory viz         : $MSBM_RUN_DIR/field_viz"
echo "Tran evaluation plots/tables : $MSBM_EVAL_OUT"
if [[ "$RUN_CONDITIONAL_EVAL" == "1" ]]; then
  echo "Conditional evaluation       : $MSBM_CONDITIONAL_OUT"
  if [[ "$CONDITIONAL_PROJECTION_PLOTS" == "1" ]]; then
    echo "Conditional projections      : $MSBM_CONDITIONAL_PROJECTION_OUT"
  fi
fi
if [[ "$RUN_POSTFILTERED_EVAL" == "1" ]]; then
  echo "Postfiltered unconditional   : $MSBM_POSTFILTERED_UNCONDITIONAL_OUT"
  echo "Postfiltered conditional     : $MSBM_POSTFILTERED_CONDITIONAL_OUT"
fi
