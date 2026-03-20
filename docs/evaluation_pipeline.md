# Evaluation Pipeline: Latent MSBM Generative Model

This document covers the four evaluation scripts for the latent MSBM generative model, their phases, output files, and figure inventory from `report.py`.

## 1. Script Overview

| Script | Purpose |
|--------|---------|
| `evaluate.py` | Full Tran-aligned evaluation (8 phases + latent geometry) |
| `report.py` | All figure-generation functions called by `evaluate.py` |
| `evaluate_conditional.py` | Conditional W1/W2 between empirical conditionals |
| `evaluate_conditional_diagnostic.py` | Separated kernel vs decoder metrics (R²_Wz, Ẽ_coarse, R²_Wf, AE health) |
| `compare_latent_geometry_models.py` | Cross-run latent-geometry aggregation, paired effect tables, and publication plots |
| `encode_corpus.py` | Corpus latent encoding for conditional evaluation and projection workflows |

---

## 2. `evaluate.py` — Main Evaluation Orchestrator

### Modes

| Mode | Trigger | Description |
|------|---------|-------------|
| Single-command | `--run_dir` | Auto-discovers dataset, generates backward SDE realizations, runs all phases |
| Legacy | `--trajectory_file + --dataset_file` | Uses pre-generated trajectory npz |
| Latent-geometry-only | `--latent_geom_fae_run_dir` | Skips generation; evaluates FAE geometry only |

### CLI

```bash
# Standard (single-command)
python scripts/fae/tran_evaluation/evaluate.py \
    --run_dir results/2026-02-01T23-00-12-38 \
    --n_realizations 200 \
    --sample_idx 0 \
    --drift_clip_norm 1.0

# Latent geometry only (no MSBM)
python scripts/fae/tran_evaluation/evaluate.py \
    --latent_geom_fae_run_dir results/fae_run_xyz \
    --output_dir results/fae_run_xyz/latent_geometry_eval

# Key options
--decode_mode auto|standard|one_step|multistep
--denoiser_steps_schedule "exp:500:0.5:8"   # adaptive per-knot step counts
--trajectory_only                           # skip Phases 1-6; keep knot-time trajectory diagnostics only
--trajectory_seed_mode fixed|marginal       # reuse --sample_idx or draw coarse seeds from the train marginal
--latent_geom_budget light|standard|thorough
--latent_geom_trace_estimator fhutch|hutchpp
--latent_geom_n_slq_probes 8
--latent_geom_n_lanczos_steps 24
--drift_clip_norm 1.0                         # optional eval-time stabilization
--no_latent_geometry                          # skip Phase 7b
--no_plot                                     # skip all figures
```

### Phase Structure

| Phase | Description | Key outputs |
|-------|-------------|-------------|
| 0 | Load GT, build filter ladder, map time indices | `eval_H_schedule`, `eval_gt_fields` |
| 1 | Conditioning consistency | `cond["mean"]`, pass/fail flag |
| 2 | Detail field decomposition | `gen_details`, `obs_details` (6 bands) |
| 3 | First-order statistics (PDFs, W1) | W1, W1_normalized per band |
| 4 | Second-order statistics (R(τ), J) | J_normalized, correlation lengths, isotropy |
| 5 | PSD diagnostics | PSD mismatch, characteristic wavelength |
| 6 | Diversity / mode-collapse | Mean pairwise dist, CV, diversity ratio |
| 7 | Backward SDE trajectory evaluation | Per-knot W1, J, PSD across all MSBM marginals |
| 7b | Latent geometry robustness | Pullback trace, effective rank, `rho_vol`, near-null mass, Hessian curvature |
| 8 | Reporting and visualization | All figures, `metrics.json`, `summary.txt` |

**Time index mapping (critical):**
MSBM excludes time index 0 (raw piecewise-constant microscale) and held-out indices.
`time_indices = [1, 3, 4, 6, 7]` for tran_inclusion with held_out={2, 5}.
The evaluation ladder starts at `H=0` (identity filter = the generated field at its native scale),
then coarser bands, to avoid double-convolution artefacts.

### Output Files

All written to `<run_dir>/tran_evaluation/` (or `--output_dir`):

| File | Description |
|------|-------------|
| `summary.txt` | Human-readable summary table |
| `metrics.json` | Full scalar metrics (all phases) |
| `curves.npz` | Numpy arrays: correlation curves, PSD curves, trajectory knots |
| `trajectory_summary.txt` | Per-knot W1/J/PSD summary table |
| `latent_geometry_metrics.json` | Phase 7b geometry results |
| `fig_*.png / fig_*.pdf` | All figures (see §4) |

---

## 3. `report.py` — Visualization Functions

All functions save both PNG (150 dpi) and PDF (vector) via `_save_fig`.

Publication field-image panels use `cmaps.bamako` (`import colormaps as cmaps`) rather than the legacy `viridis` default.

### Style Constants

```python
C_OBS   = EASTERN_HUES[7]   # steel blue — observed/GT
C_GEN   = EASTERN_HUES[4]   # red — generated
C_FILL  = EASTERN_HUES[0]   # gold — shading / fill / warning
C_OK    = EASTERN_HUES[2]   # deep green — pass flag
C_RISK  = EASTERN_HUES[4]   # red — failure flag
C_ACCENT = EASTERN_HUES[3]  # light teal — accent curves
```

Layout: `FIG_WIDTH = 7.0"`, `N_COLS = 3`.

### Figure Inventory

#### Main evaluation figures

| Function | Output name | Description |
|----------|-------------|-------------|
| `plot_conditioning` | `fig_conditioning` | Top: coarse GT + re-filtered gen fields. Bottom: difference maps (shared color scale). Pass if mean relative error < 5%. |
| `plot_conditioning_errors` | `fig_conditioning_errors` | Histogram of per-realization E^coarse values |
| `plot_sample_realizations` | `fig_sample_realizations` | Field image grid: GT (top row) vs K generated samples |
| `plot_detail_bands` | `fig_detail_bands` | Field images of observed vs generated detail bands d_l = u_{H_l} − u_{H_{l+1}} |
| `plot_pdfs` | `fig_pdfs` | Histogram-based PDF curves (obs vs gen) per detail band |
| `plot_qq` | `fig_qq` | Quantile–quantile plots per detail band |
| `plot_directional_correlation` | `fig_directional_correlation` | R(τ) in e₁ and e₂ directions per band (obs vs gen ± std) |
| `plot_J_bars` | `fig_J_bars` | Bar chart of Tran J mismatch statistic per band |
| `plot_psd` | `fig_psd` | Radial PSD curves per detail band (obs vs gen) |
| `plot_diversity` | `fig_diversity` | Pairwise distance histogram and diversity ratio |
| `plot_direct_field_pdfs` | `fig_direct_field_pdfs` | PDFs of unfiltered fields at each eval scale |
| `plot_direct_field_correlation` | `fig_direct_field_correlation` | R(τ) of unfiltered fields at each eval scale |
| `plot_conditional_pdfs` | `fig_conditional_pdfs` | Per-scale-pair conditional field PDFs (from `evaluate_conditional.py`) |

#### Backward SDE trajectory figures (Phase 7)

| Function | Output name | Description |
|----------|-------------|-------------|
| `plot_trajectory_fields` | `fig_trajectory_fields` | Field image panels at each MSBM knot (GT row + multiple gen rows, configurable via `n_gen_show`, default 4) |
| `plot_trajectory_pdfs` | `fig_trajectory_pdfs` | PDFs at each trajectory knot (obs vs gen) |
| `plot_trajectory_correlation` | `fig_trajectory_correlation` | R(τ) at each knot, separate panel per knot |
| `plot_trajectory_correlation_superposed` | `fig_trajectory_correlation_superposed` | All knot correlation curves superposed on one axis |
| `plot_trajectory_psd` | `fig_trajectory_psd` | Radial PSD at each knot (obs vs gen) |
| `plot_trajectory_qq` | `fig_trajectory_qq` | QQ plots at each trajectory knot |

Trajectory generation and standalone visualization are paired with this evaluation stage via:

- `scripts/fae/generate_full_trajectories.py`
- `notebooks/fae_latent_msbm_latent_viz.py`
- `notebooks/visualize_full_trajectories.py`
- `notebooks/visualize_field_trajectories.py`

For the publication bundle that combines the latent-128 FAE ablations with the latent MSBM run, see [docs/publication_latent128_msbm_execution.md](publication_latent128_msbm_execution.md).

#### Latent geometry figures

Removed from `evaluate.py` and `report.py` — superseded by `compare_latent_geometry_models.py`:

- `latent_geom_effective_rank`, `latent_geom_rho_vol` — covered by geometry comparison bar/chain plots
- `latent_geom_hessian` (curvature proxy) — covered by geometry comparison
- `latent_geom_flag_collapse`, `latent_geom_flag_folding`, `near_null_mass` — not required as standalone plots

The per-run `latent_geometry_metrics.json` is still written by Phase 7b; use `compare_latent_geometry_models.py` for cross-model visualization. For metric definitions see [docs/latent_geometry_formulation.md](latent_geometry_formulation.md).

#### Summary tables (printed to stdout and saved as txt)

| Function | Description |
|----------|-------------|
| `print_summary_table` | Conditioning pass, W1/J/PSD per band, diversity |
| `print_trajectory_summary_table` | Per-knot W1/J/PSD across all MSBM marginals |

---

## 4. `evaluate_conditional.py` — Latent Conditional OT + ECMMD

For each consecutive scale pair `(s, s-1)` in the MSBM trajectory:
1. Select `n_test_samples` test latents at scale `s`.
2. Find `k_neighbors` in corpus latent space at scale `s` (Gaussian kernel weights, median bandwidth).
3. Reference conditional = corpus fields at finer scale `s-1` for the k neighbors.
4. Generated conditional = backward SDE from test latent, decoded to physical fields.
5. Compute W1 and W2 (exact OT via POT; fallback: sliced Wasserstein with 100 projections).
6. Compute latent ECMMD on consecutive condition neighborhoods.
7. Optionally bootstrap ECMMD against a local empirical conditional null (`--ecmmd_bootstrap_reps`) for a finite-sample goodness-of-fit p-value.

Pair labels are written in physical H-space, not sparse dataset indices, e.g.
`pair_H1p5_to_H1`, so the first modeled marginal is `H=1` and the last modeled
marginal is the coarsest learned scale `H=6` on the default Tran ladder.

### CLI

```bash
python scripts/fae/tran_evaluation/evaluate_conditional.py \
    --run_dir results/latent_msbm_run \
    --corpus_latents_path data/corpus_latents.npz \
    --k_neighbors 200 \
    --n_test_samples 50 \
    --n_realizations 200 \
    --ecmmd_bootstrap_reps 64
```

### Outputs (in `<run_dir>/tran_evaluation/conditional/`)

| File | Description |
|------|-------------|
| `conditional_latent_metrics.json` | Latent W1/W2 summaries, null-baseline OT skill, ECMMD stats, bootstrap ECMMD p-values when enabled |
| `conditional_latent_results.npz` | Per-sample latent W1/W2 arrays and ECMMD summaries |
| `conditional_latent_summary.txt` | Human-readable table |

### Plot style (`fig_conditional_pdfs`)

`plot_conditional_pdfs` in `report.py` applies full publication style:
- `format_for_paper()` — serif fonts, EasternHues color cycle, cividis cmap
- Subplot titles use H-band labels: `$H = X.XX \to H = Y.YY$` (from `zt` in `fae_latents.npz`), matching `_H_val_label` in `fae_publication_figures.py` — `%.2f` for non-integers, `%g` for integers, spaces around `=`
- Legend: `"Reference"` (steel blue, `C_OBS`) and `"Generated"` (red, `C_GEN`), matching `report.py` conventions
- x-axis: `"Volume fraction, $u$"`; y-axis: `"Density"`

### Regenerating the figure

**Plot only** (requires a prior run that saved `conditional_pdf_data.npz`):

```bash
python scripts/fae/tran_evaluation/evaluate_conditional.py \
  --run_dir results/latent_msbm_muon_ntk_prior \
  --output_dir results/publication_latent128_and_msbm/latent_msbm/conditional \
  --replot_only
```

**Full rerun** (recomputes W1/W2 and refreshes `conditional_pdf_data.npz`):

```bash
# Direct (publication defaults)
conda run -n 3MASB python scripts/fae/tran_evaluation/evaluate_conditional.py \
  --run_dir results/latent_msbm_muon_ntk_prior \
  --output_dir results/publication_latent128_and_msbm/latent_msbm/conditional \
  --corpus_path data/fae_tran_inclusions_corpus.npz \
  --corpus_latents_path data/corpus_latents_ntk_prior.npz \
  --k_neighbors 200 \
  --n_test_samples 50 \
  --n_realizations 200 \
  --pdf_values_per_sample 4000 \
  --min_spacing_pixels 4 \
  --drift_clip_norm 1.0

# Via pipeline (delete stamp to bypass caching, skip other steps)
rm -f results/publication_latent128_and_msbm/latent_msbm/conditional/conditional.meta
bash scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh \
  --run_conditional_eval \
  --skip_latent_geometry \
  --skip_latent_space \
  --skip_eval_latent_geometry
```

---

## 5. `evaluate_conditional_diagnostic.py` — Kernel vs Decoder Separation

Implements three refined metrics that separate SDE kernel quality from decoder quality, plus an AE health check.

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R²_{W,z} | `1 − E[W₂²(ν̂_gen, ν_ref)] / E[W₂²(ν₀, ν_ref)]` | Latent conditional skill. Positive = SDE kernel better than marginal null. |
| Ẽ_coarse | `E[‖C(u_gen) − x_s‖²] / E[‖C(u_null) − x_s‖²]` | Conditioning consistency normalized by null baseline. Should be < 1.0. |
| R²_{W,f} | `1 − E[W₂²(f♯μ̂, f♯μ_ref)] / E[W₂²(f♯μ₀, f♯μ_ref)]` | Perceptual/texture skill in PSD feature space. If R²_Wz > 0 but R²_Wf < 0, decoder is the bottleneck. |
| AE health | Pixel RMSE, RMSE/pixel, PSD log-spectral error | Per-scale autoencoder reconstruction quality (independent of SDE). |

Feature map `f(u) = [mean, log_PSD_1, ..., log_PSD_B]` uses `B=16` radial PSD bins (log-spaced).

### CLI

```bash
python scripts/fae/tran_evaluation/evaluate_conditional_diagnostic.py \
    --run_dir results/latent_msbm_muon_ntk_prior \
    --corpus_path data/fae_tran_inclusions_corpus.npz \
    --corpus_latents_path data/corpus_latents_ntk_prior.npz \
    --k_neighbors 200 \
    --n_test_samples 50 \
    --n_realizations 200
```

### Outputs (in `<run_dir>/tran_evaluation/conditional_diagnostic/`)

| File | Description |
|------|-------------|
| `diagnostic_metrics.json` | R²_Wz, Ẽ_coarse, R²_Wf per scale pair + AE health per scale |
| `diagnostic_summary.txt` | Human-readable table with interpretation flags |

**Interpretation table** (printed to stdout and saved):

```
Pair                  | R²_Wz    | Ẽ_coarse   | R²_Wf    | Interpretation
pair_H6_to_H3         | +0.3412  |     0.4821  | +0.2104  | kernel OK, cond OK, texture OK
pair_H3_to_H2         | +0.1830  |     0.8940  | -0.0522  | kernel OK, cond OK, decoder suspect
```

---

## 6. Companion Entry Points

These scripts stay outside the main three evaluation CLIs, but they are part of
the maintained Tran-evaluation workflow:

| Script | Purpose |
|--------|---------|
| `generate.py` | Generate backward MSBM realizations and cache trajectory bundles |
| `evaluate_postfiltered_consistency.py` | Re-filter generated fields and compare them to unconditional or local-conditional references |
| `visualize_latent_msbm_manifold.py` | PCA/UMAP latent-manifold views for full latent trajectories |
| `visualize_conditional_latent_projections.py` | Publication-grade latent conditional projections and ambient-field panels |

These entrypoints are exercised by `make smoke-tran-eval` together with the
main evaluation CLIs.

---

## 7. Dependency Graph

```
evaluate.py
  ├── generate.py              (backward SDE sampling)
  ├── core.py                  (FilterLadder, GT loading, time index mapping)
  ├── conditioning.py          (Phase 1)
  ├── detail_fields.py         (Phase 2)
  ├── first_order.py           (Phase 3, also used in Phase 7)
  ├── second_order.py          (Phase 4, also used in Phase 7)
  ├── spectral.py              (Phase 5, also used in Phase 7)
  ├── diversity.py             (Phase 6)
  ├── latent_geometry.py       (Phase 7b: Hutchinson estimators for pullback metric)
  └── report.py                (Phase 8: all figures)

evaluate_conditional.py
  ├── conditional_support.py   (KNN weighting, H-label helpers, ECMMD support)
  ├── conditional_metrics.py   (ECMMD statistics + bootstrap calibration)
  ├── latent_msbm_runtime.py   (run reconstruction, checkpoints, interval sampling)
  └── report.py                (plot_conditional_pdfs)

evaluate_conditional_diagnostic.py
  ├── conditional_support.py
  └── latent_msbm_runtime.py

compare_latent_geometry_models.py
  ├── latent_geometry_model_summary.py
  └── latent_geometry_model_plots.py
```

## 8. Prerequisites

| Input | Source | Used by |
|-------|--------|---------|
| `<run_dir>/args.txt` | Training script | `evaluate.py`, `evaluate_conditional.py`, `evaluate_conditional_diagnostic.py` |
| `<run_dir>/fae_latents.npz` | `train_latent_msbm.py` | `evaluate_conditional.py`, `evaluate_conditional_diagnostic.py` |
| `<run_dir>/latent_msbm_policy_*.pth` | Training | Conditional scripts |
| Dataset `.npz` | `data/` | All scripts |
| Corpus `.npz` + corpus latents `.npz` | `encode_corpus.py` | Conditional scripts |
| FAE checkpoint | `fae_run/checkpoints/` | Phase 7b, conditional scripts |
