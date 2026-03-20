# Publication Figures: Architecture and Conventions

This document describes the two-script architecture for generating publication-quality figures, the figure inventory, styling conventions, and how to run each script.

## 1. Two-Script Architecture

Publication figures are split across two complementary scripts:

| Script | Scope |
|--------|-------|
| `scripts/fae/fae_publication_figures.py` | Training curves, reconstruction quality, field panels, PSD spectra |
| `scripts/fae/tran_evaluation/compare_latent_geometry_models.py` | All latent-geometry diagnostics (pullback metrics, risk flags, effect tables) |

For shell entrypoints, use:

- `scripts/fae/experiments/pipeline/publication_fae_figures.sh`
- `scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh`

The current FAE publication registry is the latent128 FiLM eight-run set:

- `fae_film_muon_l2_latent128`
- `fae_film_muon_ntk_99pct_latent128`
- `fae_film_muon_prior_latent128`
- `fae_film_muon_ntk_prior_latent128`
- `fae_film_adam_l2_latent128`
- `fae_film_adam_ntk_99pct_latent128`
- `fae_film_adam_prior_latent128`
- `fae_film_adam_ntk_prior_latent128`

The split is intentional: reconstruction/training figures operate on per-run model outputs, while geometry diagnostics aggregate across multiple runs and require the evaluation artifacts produced by `scripts/fae/tran_evaluation/latent_geometry.py`.

## 2. Figure Inventory

### 2.1 `fae_publication_figures.py`

All figures use the 8 latent-128 experiments (2 optimizers × 4 loss types) defined in
`docs/experiments/latent128_ablation_registry.csv`.

| Figure | File(s) | Description |
|--------|---------|-------------|
| fig1 | `fig1_training_convergence.{png,pdf}` | Training loss — 4 panels (L2 / NTK / Prior / NTK+Prior), Adam vs Muon per panel |
| fig2 | `fig2_performance_bars.{png,pdf}` | Test relative MSE bar chart, all 8 methods; Muon bars hatched `//` |
| fig3 | `fig3_observation_time.{png,pdf}` | Relative MSE vs filter width H (training marginals) |
| fig4 | `fig4_held_out.{png,pdf}` | Mean relative MSE across held-out H-bands (interpolation score, same style as fig2) |
| fig7 | `fig7_summary_table.{png,pdf}` + `table_summary.tex` | Summary statistics table |
| fig8 | `fig8_sample{N}_{train,held_out}_fields.{png,pdf}` | GT vs reconstructed field panels (physical scale, Adam+NTK+Prior) |
| fig9 | `fig9_psd_spectral_per_time.{png,pdf}` + `fig9b_psd_mismatch_per_time.{png,pdf}` | PSD spectra per H-band |
| fig10 | `fig10_sample{N}_all_times.{png,pdf}` + `fig10_sample{N}_t7_zoom.{png,pdf}` | Side-by-side field comparison: Adam+L2 / Adam+NTK / Muon+NTK+Prior |
| fig11 | `fig11_generated_fields.{png,pdf}` | MSBM-generated fields: GT + backward SDE realizations per knot. Requires `--msbm-run-dir`. |

### 2.2 `compare_latent_geometry_models.py`

| Output | Description |
|--------|-------------|
| `latent_geom_model_metric_*.pdf` | Per-metric bar charts: trace, effective rank, `rho_vol`, curvature proxy |
| `latent_geom_l2_ntk_prior_chain_*.pdf` | NTK/prior chain plots (L2 → NTK → NTK+Prior) |
| `latent_geom_ntk_effect_table.{md,csv}` | NTK effect table (relative improvement) |
| `latent_geom_prior_effect_table.{md,csv}` | Prior effect table (relative improvement) |

### 2.3 Removed Figures (Superseded or No Data)

The following figures were removed from `fae_publication_figures.py`:

| Figure | Reason |
|--------|--------|
| fig5 (scale comparison) | Compared single-scale σ=1 vs multiscale σ={1,2,4,8}; no single-scale latent-128 run exists |
| fig6 (architecture comparison) | Superseded by the full 8-method comparison in fig2/fig3/fig4 |
| fig9 two-scale training (old) | Removed — redundant with fig1 training curves |
| fig11 denoiser (old) | Removed — denoiser runs not part of latent-128 ablation |
| fig13 (latent regularization) | Superseded by effective-rank and near-null mass plots in `compare_latent_geometry_models.py` |
| fig14 (per-marginal latent) | Superseded by time-resolved geometry metrics |
| fig15 (inter-marginal distance) | Superseded by geometry comparison + effect tables |

For metric definitions see [docs/latent_geometry_formulation.md](latent_geometry_formulation.md) and [docs/latent_geometry_plotting.md](latent_geometry_plotting.md).

## 3. Styling Conventions

### 3.1 EasternHues Palette

All plots use the `EasternHues` palette from `scripts/images/field_visualization.py`:

```python
EASTERN_HUES = ["#D9AB42", "#A35E47", "#0F4C3A", "#78C2C4",
                "#C73E3A", "#563F2E", "#B47157", "#2B5F75"]
#                  [0]         [1]        [2]        [3]
#                  [4]         [5]        [6]        [7]
```

Index assignments in `fae_publication_figures.py` (one color per optimizer×loss combination):

| Index | Hex | Method |
|-------|-----|--------|
| `[0]` | `#D9AB42` | Adam + NTK |
| `[1]` | `#A35E47` | Adam + Prior |
| `[2]` | `#0F4C3A` | Muon + L2 |
| `[3]` | `#78C2C4` | Muon + NTK |
| `[4]` | `#C73E3A` | Adam + L2 (`C_GEN`) |
| `[5]` | `#563F2E` | Adam + NTK + Prior |
| `[6]` | `#B47157` | Muon + Prior (`C_FILL`) |
| `[7]` | `#2B5F75` | Muon + NTK + Prior (`C_OBS`) |

Optimizer encoding (shared across line plots and bar charts):

- **Line style**: solid (`-`) = Adam, dashed (`--`) = Muon
- **Bar hatch**: none = Adam, `//` = Muon

Index assignments in `compare_latent_geometry_models.py` follow [docs/latent_geometry_plotting.md](latent_geometry_plotting.md):

- `C_OBS = EASTERN_HUES[7]` (steel blue)
- `C_GEN = EASTERN_HUES[4]` (red)
- Bar hatch: none = Adam, `//` = Muon
- Line style: solid = Adam, dashed = Muon

### 3.3 Field Colormap

Publication field-image panels use `cividis` (perceptually uniform, colour-blind safe).
This is set via `CMAP_FIELD = "cividis"` in `fae_publication_figures.py`.

Sequential continuous colormaps (latent scatter plots, time-ordered visualizations)
also use `cividis`. The `cmaps.bamako` option is available but not used for these plots.

### 3.2 No Titles or Annotations

All publication plots omit:

- `ax.set_title(descriptive_string)` — no descriptive subplot titles
- `fig.suptitle(...)` — no figure-level titles
- `fig.text(...)` — no caveat or architecture notes in the figure

Metadata belongs in captions, not in the figure. The only allowed `set_title` calls are data-label titles in field/PSD subplots (H-band labels; see §4).

## 4. H-Band Labels and Held-Out Detection

### 4.1 Convention

Marginal time indices are labelled using the physical filter scale H:

```
$H=0.25$        # training marginal
$H=0.50$ (HO)  # held-out marginal
```

`H` is the Tran filter bandwidth parameter — this is preferred over `$t_x$` indexing because H has direct physical meaning. `(HO)` marks held-out marginals not seen during training.

### 4.2 Field Panels (fig8, fig10, fig11)

```python
ho_set = set(held_out_indices)
# for tran-style datasets also add {0} (raw microscale always excluded)

def _col_label(tidx: int, t_val: float, is_ho: bool) -> str:
    ho_tag = " (HO)" if is_ho else ""
    return _H_val_label(t_val) + ho_tag

def _H_val_label(H_val: float) -> str:
    """H value → LaTeX label (no trailing zeros for integers)."""
    if H_val == int(H_val):
        return "$H = %g$" % H_val
    return "$H = %.2f$" % H_val
```

### 4.3 PSD Spectra (fig9)

PSD data is stored under keys like `t1_train`, `t3_held_out`. Since actual H values are not available in `psd_data.npz`, PSD uses index-based labels as a fallback:

```python
def _psd_time_label(tk: str) -> str:
    parts = str(tk).split("_", 1)
    tidx_str = parts[0]          # e.g. "t1"
    split_raw = parts[1] if len(parts) > 1 else ""
    ho_tag = " (HO)" if "held_out" in split_raw else ""
    return f"$H_{{{tidx_str[1:]}}}${ho_tag}"
```

## 5. Physical-Scale Display (Inverse Transform)

Field panels (fig8, fig10) display fields in physical units by applying the inverse of the dataset normalization transform before plotting.

```python
from data.transform_utils import load_transform_info, apply_inverse_transform

transform_info = load_transform_info(data)   # data is the loaded npz dict

# per field (shape: [n_points]):
phys = apply_inverse_transform(field[None, :], transform_info)[0]
```

`load_transform_info` reads `transform_type`, `mean`, `std`, `min`, `max`, and `epsilon` from the npz metadata. `apply_inverse_transform` reverses `log_standardize` (unstandardize then `exp(x) - epsilon`) or `minmax` as appropriate. Colorbars on field imshow panels are therefore in physical (not log-standardized) units.

PSD spectra in fig9 are computed from the stored `psd_data.npz` which already contains band-averaged spectra; no additional inverse transform is applied there.

## 6. Running the Scripts

### 6.1 `fae_publication_figures.py`

```bash
conda activate 3MASB
python scripts/fae/fae_publication_figures.py \
    --run_dir  /path/to/wandb/run \
    --data_dir /path/to/dataset \
    --out_dir  figures/ \
    --msbm-run-dir /path/to/latent_msbm_run   # optional, for fig17
```

Prerequisites:
- A completed FAE training run (W&B artifact or local checkpoint directory)
- The dataset `.npz` file (used for transform metadata and marginal keys)
- `psd_data.npz` in the run directory (produced by `compute_psd.py`) for fig12
- A latent MSBM run directory (for fig17 only; skipped if `--msbm-run-dir` not provided)

Preferred wrapper:

- [docs/publication_fae_figures_execution.md](publication_fae_figures_execution.md)
- `scripts/fae/experiments/pipeline/publication_fae_figures.sh`

### 6.2 `compare_latent_geometry_models.py`

```bash
conda activate 3MASB
python scripts/fae/tran_evaluation/compare_latent_geometry_models.py \
    --registry_csv docs/experiments/combinatorial_run_registry.csv \
    --output_dir  results/latent_geometry_model_comparison
```

Prerequisites:
- Completed FAE run directories with `args.json` and `checkpoints/`
- Or explicit `--run_dir` arguments when plotting a fixed publication subset

When explicit runs are supplied, `compare_latent_geometry_models.py` can reuse existing
`latent_geometry_metrics.json` artifacts before falling back to recomputation.

For the latent-128 publication subset plus the latent MSBM trajectory workflow, use:

- [docs/publication_latent128_msbm_execution.md](publication_latent128_msbm_execution.md)
- `scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh`

That wrapper defaults to EMA sampling/evaluation and `--drift_clip_norm 1.0` for the
latent-MSBM publication trajectory stages, and reuses matching existing outputs.
For cmap/style-only changes, rerun it with `--refresh_plots_only`.

For metric definitions and estimation details see:
- [docs/latent_geometry_formulation.md](latent_geometry_formulation.md)
- [docs/latent_geometry_plotting.md](latent_geometry_plotting.md)
