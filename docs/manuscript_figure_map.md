# Manuscript Figure Map

This is the maintained map for the current manuscript-facing figure surface.
Use it instead of the historical latent128 publication bundle docs when deciding
which scripts own the figures and tables that still matter in the active tree.

## Figure Priorities

1. latent geometry metrics
2. latent manifold visualization comparison
3. conditional evaluations:
   global coarse consistency
   latent-manifold conditional generated trajectories
   latent trajectory projection visualization
   one-point and two-point statistics in both generation and recoarsening modes

## Active Ownership

| Manuscript item | Primary surface | Primary artifacts | Notes |
| --- | --- | --- | --- |
| Latent geometry metrics | `scripts/fae/tran_evaluation/compare_latent_geometry_models.py` | `latent_geom_pair_summary.json/.csv`, `latent_geom_pair_delta_table.md/.csv` | Use the canonical transformer pair only. For a fixed baseline/treatment pair, the manuscript-primary output is quantitative tables, not a standalone figure. Pairwise plots are diagnostic or appendix-level only. |
| Latent manifold visualization comparison | `scripts/fae/tran_evaluation/visualize_autoencoder_latent_space.py` | `latent_pca2_scatter.png/.pdf`, `latent_pca2_scatter_grid.png/.pdf`, `latent_space_viz_summary.json` | This is the maintained FAE-side qualitative comparison surface. |
| Global coarse consistency figures | `scripts/csp/evaluate_csp.py`, `scripts/csp/evaluate_csp_token_dit.py`, `scripts/fae/tran_evaluation/evaluate.py` | `generated_consistency/fig1_coarse_consistency_breakdown`, `fig1b_coarse_consistency_conditions`, `fig1c_*`, `fig1d_*` | These figures come from the decoded Tran phase invoked by the CSP wrappers. |
| Latent-manifold conditional generated trajectories | `scripts/csp/evaluate_csp_conditional_rollout.py`, `scripts/csp/evaluate_csp_token_dit_conditional_rollout.py`, `scripts/csp/plot_latent_trajectories.py` | `fig_conditional_rollout_latent_trajectories.png/.pdf` | This is the primary maintained manuscript conditional trajectory panel. |
| Latent trajectory projection visualization | `scripts/csp/plot_latent_trajectories.py` through the CSP wrappers or `conditional_rollout` stages | `fig_latent_trajectory_projection.png/.pdf`, `latent_trajectory_projection_summary.json` | Use this for shared latent projection views of generated and reference trajectories. |
| One-point and two-point conditional figures | `scripts/csp/evaluate_csp_conditional_rollout.py` and `scripts/csp/conditional_eval/rollout_reports.py` | `fig_conditional_rollout_field_pdfs_<role>.png/.pdf`, `fig_conditional_rollout_field_corr_<role>.png/.pdf` | These are the maintained decoded-field manuscript figures for generation mode. |
| Recoarsened one-point and two-point conditional figures | `scripts/csp/evaluate_csp_conditional_rollout.py` and `scripts/csp/conditional_eval/rollout_reports.py` | `fig_conditional_rollout_recoarsened_field_pdfs_<role>.png/.pdf`, `fig_conditional_rollout_recoarsened_field_corr_<role>.png/.pdf` | Use these when the manuscript needs recoarsening-mode conditional fidelity rather than generation-mode conditional fidelity. |

## Recommended Commands

Canonical geometry metrics:

```bash
bash scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh
```

Latent manifold comparison:

```bash
python scripts/fae/tran_evaluation/visualize_autoencoder_latent_space.py \
  --run_dir <fae_run_dir_a> \
  --run_dir <fae_run_dir_b> \
  --output_dir results/latent_space_autoencoder_viz
```

Primary CSP manuscript conditional evaluation:

```bash
python scripts/csp/evaluate_csp.py \
  --run_dir results/csp/<run_dir>
```

Explicit staged conditional-rollout rerun:

```bash
python scripts/csp/build_conditional_rollout_latent_cache.py \
  --run_dir results/csp/<run_dir> \
  --output_dir results/csp/<run_dir>/eval/conditional_rollout

python scripts/csp/build_conditional_rollout_decoded_cache.py \
  --run_dir results/csp/<run_dir> \
  --output_dir results/csp/<run_dir>/eval/conditional_rollout

python scripts/csp/evaluate_csp_conditional_rollout.py \
  --run_dir results/csp/<run_dir> \
  --output_dir results/csp/<run_dir>/eval/conditional_rollout \
  --phases latent_metrics,field_metrics,reports
```

## Historical Boundary

The following are historical or compatibility-only and should not be treated as
the active manuscript figure surface:

- `scripts/fae/experiments/pipeline/publication_latent128_and_msbm.sh`
- `docs/publication_figures.md`
- `docs/publication_latent128_msbm_execution.md`
- latent128 chain/effect-table geometry outputs such as `latent_geom_l2_ntk_*`
- removed latent-MSBM conditional entrypoints such as `evaluate_conditional.py`
