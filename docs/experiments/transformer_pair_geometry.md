# Transformer Pair Geometry Evaluation

This is the maintained latent-geometry comparison workflow.

It evaluates exactly two transformer-token FAE runs:

- baseline: `transformer_patch8_adamw_beta1e3_l128x128`
- treatment: `transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5`

The runtime default pair is owned by:

- `scripts/fae/tran_evaluation/latent_geometry_model_selection.py`

The maintained pair registry is retained as provenance only:

- `docs/experiments/transformer_pair_geometry_registry.csv`

The maintained wrapper is:

- `scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh`

The underlying comparison entrypoint is:

- `scripts/fae/tran_evaluation/compare_latent_geometry_models.py`

## Scope

This workflow compares deterministic decoder-manifold geometry only. It does
not estimate latent-prior geometry, bridge geometry, or CSP conditional-law
geometry.

The evaluator is transformer-token compatible. Transformer latents are encoded
through the maintained flattened downstream surface and restored to token
memory before direct decoder Jacobian and Hessian probes.

## Command

```bash
cd /data1/jy384/research/MMSFM
conda run -n 3MASB \
  bash scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh
```

Useful overrides:

```bash
conda run -n 3MASB \
  BUDGET=light MAX_SAMPLES=32 \
  bash scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh
```

Expert override with explicit leaf run directories:

```bash
python scripts/fae/tran_evaluation/compare_latent_geometry_models.py \
  --run_dir results/fae_transformer_patch8_adamw_beta1e3_l128x128/transformer_patch8_adamw_beta1e3_l128x128 \
  --run_dir results/fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5/transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5
```

## Outputs

The maintained output contract is pairwise-first:

- `latent_geom_pair_summary.json`
- `latent_geom_pair_summary.csv`
- `latent_geom_pair_delta_table.md`
- `latent_geom_pair_delta_table.csv`
- `latent_geom_pair_metric_*.png/.pdf`
- `latent_geom_pair_time_delta_*.png/.pdf`
- `per_run/*/latent_geometry_metrics.json`

For the current manuscript, the primary deliverables are the quantitative pair
summary and delta tables. The pairwise plots remain useful as diagnostics or
appendix material, but a standalone latent-geometry figure is no longer
required for the fixed two-run comparison.

## Historical Note

The latent128 FiLM publication geometry workflow is historical only and is not
part of the active experiment surface.
