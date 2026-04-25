# CSP Conditional Evaluation Core

This document is the maintained map for the active conditional-evaluation
surface. Use it together with [runbooks/csp.md](runbooks/csp.md),
[runbooks/tran-evaluation.md](runbooks/tran-evaluation.md), and
[tran_coarse_consistency.md](tran_coarse_consistency.md). For the April 2026
token-native rollout GPU-memory diagnosis and the forward-sampling solver fix,
see [conditional_rollout_solver_memory_analysis.md](conditional_rollout_solver_memory_analysis.md).
For the maintained mathematical definition of the local-conditional correlation
figures and the worked anisotropy diagnosis of
`fig_conditional_rollout_field_corr_best`, see
[conditional_rollout_correlation_curves.md](conditional_rollout_correlation_curves.md).

## Scope

The maintained conditional-evaluation core keeps only the scientific operations
that are still used in current CSP evaluation and report generation:

1. global rollout latent cache construction
2. rollout decoded cache construction
3. token-native `knn_reference`
4. shared and conditional latent trajectory projections
5. conditional rollout latent trajectory panels
6. coarse-consistency evaluation for legacy `coarse_consistency_m50_r32`-style
   runs and current CSP wrappers

Vector CSP conditional evaluation remains compatibility-only. Token-native CSP
is the primary maintained implementation.

## Maintained Entry Points

- [scripts/csp/build_conditional_rollout_latent_cache.py](../scripts/csp/build_conditional_rollout_latent_cache.py)
  builds the authoritative global rollout latent cache
- [scripts/csp/build_conditional_rollout_decoded_cache.py](../scripts/csp/build_conditional_rollout_decoded_cache.py)
  decodes from an existing latent rollout store and finalizes the decoded cache
- [scripts/csp/evaluate_csp_token_dit_conditional_rollout.py](../scripts/csp/evaluate_csp_token_dit_conditional_rollout.py)
  runs rollout metrics, reporting, and optional compatibility export
- [scripts/csp/evaluate_csp_token_dit_knn_reference.py](../scripts/csp/evaluate_csp_token_dit_knn_reference.py)
  runs token-native pairwise `knn_reference`
- [scripts/fae/tran_evaluation/evaluate_generated_consistency.py](../scripts/fae/tran_evaluation/evaluate_generated_consistency.py)
  runs standalone coarse consistency for generated caches
- [scripts/fae/tran_evaluation/coarse_consistency_eval.py](../scripts/fae/tran_evaluation/coarse_consistency_eval.py)
  owns the legacy coarse-consistency artifact contract used by prior
  `coarse_consistency_m50_r32`-style runs

Compatibility wrappers that remain but do not own scientific logic:

- [scripts/csp/evaluate_csp.py](../scripts/csp/evaluate_csp.py)
- [scripts/csp/evaluate_csp_token_dit.py](../scripts/csp/evaluate_csp_token_dit.py)
- [scripts/csp/evaluate_csp_knn_reference.py](../scripts/csp/evaluate_csp_knn_reference.py)
- [scripts/csp/evaluate_csp_conditional_rollout.py](../scripts/csp/evaluate_csp_conditional_rollout.py)

## Ownership

### Rollout caches and evaluation

- [scripts/csp/conditional_rollout_runtime.py](../scripts/csp/conditional_rollout_runtime.py)
  owns CLI parsing and seam-preserving wrappers only
- [scripts/csp/conditional_eval/rollout_stage_runtime.py](../scripts/csp/conditional_eval/rollout_stage_runtime.py)
  owns latent-cache, decoded-cache, and evaluation stage execution
- [scripts/csp/conditional_eval/rollout_context.py](../scripts/csp/conditional_eval/rollout_context.py)
  owns rollout context assembly, dataset/run contracts, and manifest construction
- [scripts/csp/conditional_eval/rollout_condition_mode.py](../scripts/csp/conditional_eval/rollout_condition_mode.py)
  owns the maintained rollout mode contract: `chatterjee_knn` by default and
  `exact_query` for compatibility-only reuse
- [scripts/csp/conditional_eval/rollout_execution.py](../scripts/csp/conditional_eval/rollout_execution.py)
  owns rollout execution policy, device selection, and latent-trajectory report entry
- [scripts/csp/conditional_eval/rollout_reference_contract.py](../scripts/csp/conditional_eval/rollout_reference_contract.py)
  owns reference-cache fingerprint checks and active-condition slicing
- [scripts/csp/conditional_eval/rollout_reference_cache.py](../scripts/csp/conditional_eval/rollout_reference_cache.py)
  owns the coarse-root `H=6` Chatterjee Gaussian kNN support used across
  downstream rollout targets
- [scripts/csp/conditional_eval/rollout_assignment_cache.py](../scripts/csp/conditional_eval/rollout_assignment_cache.py)
  owns deterministic sampled neighborhood assignments for the data-side and
  model-side local empirical conditionals
- [scripts/csp/conditional_eval/rollout_assignment_contract.py](../scripts/csp/conditional_eval/rollout_assignment_contract.py)
  owns assignment-cache fingerprint checks and active-condition slicing
- [scripts/csp/conditional_eval/rollout_neighborhood_cache.py](../scripts/csp/conditional_eval/rollout_neighborhood_cache.py)
  owns the maintained neighborhood-conditioned rollout latent and decoded cache
  stores
- [scripts/csp/conditional_eval/rollout_generated_cache.py](../scripts/csp/conditional_eval/rollout_generated_cache.py)
  owns latent and decoded rollout store loading for both the maintained
  neighborhood-conditioned path and the exact-query compatibility path
- [scripts/csp/conditional_eval/rollout_metrics.py](../scripts/csp/conditional_eval/rollout_metrics.py)
  owns rollout latent response-adherence metrics, coarse-root conditional
  diversity metrics, and decoded field metrics
- [scripts/fae/tran_evaluation/conditional_diversity.py](../scripts/fae/tran_evaluation/conditional_diversity.py)
  owns frozen-latent Conditional-RKE, Conditional-Vendi, Information-Vendi,
  and conditioning-state bootstrap for rollout evaluation
- [scripts/csp/conditional_eval/rollout_reports.py](../scripts/csp/conditional_eval/rollout_reports.py)
  owns rollout summaries, result payloads, PDF/correlation figures, and report
  tables

### Token-native `knn_reference`

- [scripts/csp/token_conditional_eval_runtime.py](../scripts/csp/token_conditional_eval_runtime.py)
  owns stage orchestration only
- [scripts/csp/token_conditional_phases.py](../scripts/csp/token_conditional_phases.py)
  owns token-native pairwise sampling, reference payload construction, W2, and
  ECMMD support
- [scripts/csp/conditional_eval/token_reference_context.py](../scripts/csp/conditional_eval/token_reference_context.py)
  owns corpus-latent validation, runtime reconstruction, pair metadata, and
  result-store/header setup
- [scripts/csp/conditional_eval/token_field_metrics.py](../scripts/csp/conditional_eval/token_field_metrics.py)
  owns token-latent decode and decoded field metrics for pairwise reports
- [scripts/csp/conditional_eval/pairwise_artifacts.py](../scripts/csp/conditional_eval/pairwise_artifacts.py)
  owns saved artifact reuse and export writing

### Visualization core

- [scripts/csp/plot_latent_trajectories.py](../scripts/csp/plot_latent_trajectories.py)
  owns the shared latent projection figure plus conditional and rollout
  trajectory panels
- [scripts/csp/conditional_eval/pairwise_plots.py](../scripts/csp/conditional_eval/pairwise_plots.py)
  owns conditional one-point latent PDF plots
- [scripts/csp/conditional_pdf_plots.py](../scripts/csp/conditional_pdf_plots.py)
  is now a thin compatibility re-export for the maintained pairwise plot module

### Coarse consistency

- [scripts/fae/tran_evaluation/coarse_consistency.py](../scripts/fae/tran_evaluation/coarse_consistency.py)
  owns the total/bias/spread estimators
- [scripts/fae/tran_evaluation/coarse_consistency_runtime.py](../scripts/fae/tran_evaluation/coarse_consistency_runtime.py)
  owns runtime-backed interval/global sampling and cache construction
- [scripts/fae/tran_evaluation/coarse_consistency_artifacts.py](../scripts/fae/tran_evaluation/coarse_consistency_artifacts.py)
  owns the legacy summary, metrics, arrays, and manifest artifact family
- [scripts/fae/tran_evaluation/report.py](../scripts/fae/tran_evaluation/report.py)
  owns the coarse-consistency figures, including the qualitative panels used by
  prior `coarse_consistency_m50_r32`-style runs

## Stage Boundaries

### `conditional_rollout`

- `build_conditional_rollout_latent_cache.py`
  writes only the latent rollout store and manifest
- `build_conditional_rollout_decoded_cache.py`
  requires an existing latent rollout store and writes only the decoded rollout
  store; the legacy compatibility export remains exact-query only
- `evaluate_csp*_conditional_rollout.py --phases latent_metrics`
  reads only the latent store and writes latent response-adherence `W2` plus
  coarse-root conditional diversity in the frozen FAE latent space
- `--phases field_metrics,reports`
  reads only the decoded store
- `--phases compat_export`
  is the only stage that writes the legacy `conditioned_global.npz` export

The maintained rollout default is now `--rollout_condition_mode chatterjee_knn`.
In that mode:

- neighborhoods are determined only on the coarsest modeled state `H=6`
- the support uses the Chatterjee Gaussian kNN rule
- a saved assignment cache fixes the local empirical data-side and model-side
  draws for every query row
- rollout metrics and field figures compare sampled neighborhood conditionals on
  both sides

`--rollout_condition_mode exact_query` remains only for compatibility reuse,
legacy export, and current coarse-consistency consumers. Its decoded-field
second-order path no longer uses the neighborhood-conditional stationary-style
correlation contract: exact-query now computes a single-field ergodic
pair-correlation surrogate on the observed conditional target, applies the same
field-level functional to each generated sample through a pooled
ensemble-covariance estimator, and reports observed moving-block-bootstrap
uncertainty bands plus generated sample-index bootstrap bands. The maintained
`chatterjee_knn` path now uses the same pooled ensemble estimator on both the
reference and generated neighborhood conditionals.

### `knn_reference`

- `reference_cache`
  writes reusable pairwise reference/generated chunks only
- `latent_metrics`
  reuses the saved pair payloads and computes latent W2 outputs
- `field_metrics,reports`
  reuses the saved pair payloads, decodes fields, writes one-point PDFs,
  field metrics, tables, and ECMMD figures

### Coarse consistency

The maintained coarse-consistency operations are:

- `conditioned_interval`
- `conditioned_interval_metadata`
- `conditioned_global_return`
- `cache_global_return`
- `path_self_consistency`

These stay active because they are still required for both current CSP wrappers
and previously evaluated `coarse_consistency_m50_r32`-style outputs.

## Artifact Families

### `conditional_rollout`

- `conditional_rollout_manifest.json`
- `conditional_rollout_metrics.json`
- `conditional_rollout_results.npz`
- `conditional_rollout_summary.txt`
- `conditional_rollout_reference_cache.npz`
- `conditional_rollout_reference_cache_manifest.json`
- `conditional_rollout_assignment_cache.npz`
- `conditional_rollout_assignment_cache_manifest.json`
- `cache/conditioned_global_latents.cache/`
- `cache/conditioned_global.cache/`
- `cache/conditioned_global.npz` only when `compat_export` is requested in
  `exact_query` mode
- latent trajectory projection figures and rollout field figures under the same
  output directory

`conditional_rollout_metrics.json` now records the coarse root as the
`conditioning coarse state` and each finer rollout destination as a `response
scale`. The latent-metrics payload adds `headline_response_label`,
`conditional_diversity_config`, response-wise `conditional_diversity`, and a
parallel `response_specs` contract. `conditional_rollout_results.npz` now adds
per-response `latent_conditional_rke_<response_label>`,
`latent_conditional_vendi_<response_label>`, and
`latent_information_vendi_<response_label>` arrays. These latent diversity
outputs remain separate from decoded-field fidelity and coarse-consistency
artifacts.

Exact-query `field_metrics,reports` now add provenance fields
`correlation_estimator=exact_query_single_field_obs_generated_ensemble_paircorr_bootstrap_v2`,
`observed_band_method=moving_block_bootstrap_percentile`,
`generated_band_method=sample_index_bootstrap_percentile`, and
`line_block_length_rule=line_summary_e_folding`. Their two-point figure stems
are `fig_conditional_rollout_field_paircorr_<role>` and
`fig_conditional_rollout_recoarsened_field_paircorr_<role>`.
`chatterjee_knn` now adds
`correlation_estimator=chatterjee_knn_reference_generated_ensemble_paircorr_bootstrap_v1`,
`observed_band_method=sample_index_bootstrap_percentile`, and
`generated_band_method=sample_index_bootstrap_percentile` while keeping the
existing `field_corr` figure family. For the estimator formula, the role of the
assignment cache on both sides, and how to interpret directional splitting in
one selected panel, see
[conditional_rollout_correlation_curves.md](conditional_rollout_correlation_curves.md).

### `knn_reference`

- `knn_reference_manifest.json`
- `knn_reference_metrics.json`
- `knn_reference_results.npz`
- `knn_reference_summary.txt`
- ECMMD dashboards
- conditional one-point PDF figures
- field tables and representative field figures

### Coarse consistency

- `generated_consistency_summary.txt`
- `generated_consistency_metrics.json`
- `generated_consistency_arrays.npz`
- `generated_consistency_manifest.json`
- `fig1_coarse_consistency_breakdown`
- `fig1b_coarse_consistency_conditions`
- `fig1c_coarse_consistency_global_qualitative`
- `fig1d_coarse_consistency_<pair>_qualitative`
- cache stores under `coarse_consistency/cache/`

## Removed Standalone Scripts

The following deprecated standalone Tran scripts were removed from the active
surface and from smoke targets:

- `scripts/fae/tran_evaluation/evaluate_conditional.py`
- `scripts/fae/tran_evaluation/evaluate_conditional_diagnostic.py`
- `scripts/fae/tran_evaluation/evaluate_coarse_consistency.py`
- `scripts/fae/tran_evaluation/visualize_conditional_latent_projections.py`
- `scripts/fae/tran_evaluation/visualize_latent_msbm_manifold.py`

Their retained logic now lives in the maintained modules above. Historical
workflow notes should point to this document and the active runbooks instead of
reintroducing those entrypoints.
