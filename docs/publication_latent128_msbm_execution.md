# Publication Execution: latent128 FAE + latent MSBM

This document is the concrete execution guide for the publication-quality plotting and tabulation workflow covering:

- the eight latent-128 FiLM FAE ablation runs
- the latent MSBM run at `/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior`

The workflow is implemented in:

- `scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh`

The runtime environment for this workflow is `3MASB`.

## 1. Scope

The script executes four publication-facing tasks end to end:

1. Resolves the completed FAE run leaf directories under the eight requested latent-128 experiment roots.
2. Generates cross-model latent-geometry plots and metric tables.
3. Generates and visualizes full latent / decoded MSBM trajectories.
4. Runs the Tran-aligned MSBM evaluation to generate publication figures and tabulated trajectory metrics.

All field-image plots now use `cmaps.bamako` from `import colormaps as cmaps`.

## 2. Runs covered

FAE roots resolved by the script:

- `/data1/jy384/research/MMSFM/results/fae_film_muon_l2_latent128`
- `/data1/jy384/research/MMSFM/results/fae_film_muon_ntk_99pct_latent128`
- `/data1/jy384/research/MMSFM/results/fae_film_muon_prior_latent128`
- `/data1/jy384/research/MMSFM/results/fae_film_muon_ntk_prior_latent128`
- `/data1/jy384/research/MMSFM/results/fae_film_adam_l2_latent128`
- `/data1/jy384/research/MMSFM/results/fae_film_adam_ntk_99pct_latent128`
- `/data1/jy384/research/MMSFM/results/fae_film_adam_prior_latent128`
- `/data1/jy384/research/MMSFM/results/fae_film_adam_ntk_prior_latent128`

Latent MSBM run:

- `/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior`

## 3. Command

```bash
cd /data1/jy384/research/MMSFM
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh
```

Useful overrides:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --output_root results/publication_latent128_and_msbm \
  --trajectory_samples 32 \
  --eval_realizations 200 \
  --sample_idx 0 \
  --latent_geom_budget thorough \
  --latent_geom_trace_estimator fhutch \
  --force_recompute
```

By default the latent-geometry comparison step checks for existing
`latent_geometry_metrics.json` artifacts for each FAE run and reuses them.

By default the MSBM trajectory and evaluation steps:

- use EMA checkpoints when available
- apply mild evaluation-time drift clipping with `--drift_clip_norm 1.0`
- reuse existing outputs only when a small metadata stamp matches the current settings

To match the original `MSBM` sampling path more closely, disable drift clipping:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --no_drift_clip
```

To disable EMA reuse for sampling/evaluation:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --no_use_ema
```

To keep only the PCA publication manifold plots and skip the optional UMAP companion:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --no_umap_manifold
```

If you only changed plotting code or colormaps and want fresh figures without
recomputing trajectories, use:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --refresh_plots_only
```

This reruns the figure-producing steps while reusing saved latent-geometry metrics,
the saved trajectory bundle, and cached evaluation realizations when present.

If you only want the latent MSBM trajectory bundle, trajectory visualizations,
and Tran-aligned evaluation, and do not want the FAE latent-geometry / latent-space
steps to run at all, use:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --msbm_only
```

For the drift-unclipped trajectory fix plus Tran evaluation refresh:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --msbm_only \
  --no_drift_clip \
  --refresh_plots_only \
  --no_umap_manifold
```

To force a fresh latent-geometry recomputation:

```bash
conda run -n 3MASB \
  bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --force_recompute \
  --no_check_existing_latent_geometry
```

## 4. Outputs

Primary publication bundle under the script output root:

- `<output_root>/latent_geometry/`
  - `latent_geom_model_metric_*.png/.pdf`
  - `latent_geom_l2_ntk_prior_chain_*.png/.pdf`
  - `latent_geom_model_flag_*.png/.pdf`
  - `latent_geom_model_summary.json`
  - `latent_geom_model_summary.csv`
  - `latent_geom_ntk_effect_table.md/.csv`
  - `latent_geom_prior_effect_table.md/.csv`
- `<output_root>/latent_space/`
  - per-run latent PCA scatter plots
  - combined scatter grid
- `<output_root>/latent_msbm/tran_evaluation/`
  - `fig12_trajectory_fields`
  - `fig13_trajectory_pdfs`
  - `fig14_trajectory_correlation`
  - `fig15_trajectory_psd`
  - `fig16_trajectory_qq`
  - `fig17_trajectory_correlation_superposed`
  - `metrics.json`
  - `summary.txt`
  - `trajectory_summary.txt`
  - `curves.npz`

Trajectory-generation and visualization artifacts written next to the latent MSBM run:

- `/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior/publication_full_trajectories.npz`
- `/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior/eval/latent_viz/`
- `/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior/full_traj_viz/`
  - `latent_manifold_backward_publication.{png,pdf}`
  - `latent_manifold_forward_publication.{png,pdf}`
  - `latent_manifold_mean_paths_publication.{png,pdf}`
  - `latent_manifold_backward_umap_publication.{png,pdf}` when `umap-learn` is available
  - `latent_manifold_summary.json`
- `/data1/jy384/research/MMSFM/results/latent_msbm_muon_ntk_prior/field_viz/`

## 5. Notes

- The latent-geometry comparison script is run in explicit `--run_dir` mode, so it does not depend on the registry CSV for this publication subset.
- The effect tables are computed on the shared multiscale tag `multi_1248`.
- Existing per-run latent-geometry metrics are reused automatically when available.
- Existing MSBM trajectory and evaluation outputs are reused automatically when their saved settings match the current `run_dir`, sample counts, EMA setting, and drift clip norm.
- `--refresh_plots_only` is the right mode for cmap/style changes because it bypasses output-file reuse for plot steps while keeping saved data caches.
- The wrapper default is `--drift_clip_norm 1.0` for publication-time sampling only. Training is unchanged.
- If a root contains multiple `run_*` children, the pipeline selects the completed run that has checkpoints and prefers one with `eval_results.json`.
- Step 4 includes dedicated publication-style latent-manifold overlays that follow the existing notebook visual language. PCA is the primary view for path interpretation; UMAP is only a qualitative companion for local neighborhood structure and should not be used to argue global distance or path-length fidelity.
