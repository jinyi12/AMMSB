from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from csp import integrate_interval, sample_conditional_batch
from csp.paired_prior_bridge import sample_paired_prior_conditional_batch
from data.transform_utils import apply_inverse_transform
from scripts.csp.conditional_eval.condition_set import (
    build_condition_set,
    condition_set_from_metadata_arrays,
    condition_set_to_metadata_arrays,
    ensure_condition_set_matches,
)
from scripts.csp.conditional_eval.field_metrics import (
    evaluate_pair_field_metrics,
    infer_pixel_size_from_grid,
    sample_reference_latent_draws,
)
from scripts.csp.conditional_eval.pairwise_artifacts import (
    completed_conditional_phases,
    load_existing_conditional_eval_exports,
    write_conditional_eval_artifacts,
)
from scripts.csp.conditional_eval.pairwise_runtime_support import (
    deferred_ecmmd_metrics,
    deferred_pairwise_reasons,
    deferred_w_metrics,
    selected_field_metric_rows,
)
from scripts.csp.conditional_eval_phases import (
    CONDITIONAL_PHASE_FIELD_METRICS,
    CONDITIONAL_PHASE_ECMMD,
    CONDITIONAL_PHASE_SAMPLE,
    CONDITIONAL_PHASE_W2,
    resolve_requested_conditional_phases,
)
from scripts.csp.conditional_eval.seed_policy import (
    build_seed_policy,
    seed_policy_from_metadata_arrays,
    seed_policy_to_metadata_arrays,
)
from scripts.csp.conditional_sample_cache import (
    build_conditional_sample_cache_manifest,
    conditional_sample_cache_matches,
    has_conditional_pair_sample,
    load_conditional_pair_sample,
    load_conditional_pair_sample_chunk,
    load_conditional_sample_metadata,
    load_conditional_sample_metadata_chunk,
    prepare_conditional_sample_cache,
    iter_conditional_sample_realization_spans,
    write_conditional_pair_sample,
    write_conditional_sample_metadata,
)
from scripts.csp.token_conditional_phases import load_saved_pair_sample_payload
from scripts.csp.conditional_eval.report_tables import build_field_metric_table_text
from scripts.fae.tran_evaluation.conditional_metrics import (
    metric_summary,
    parse_positive_int_list_arg,
)
from scripts.fae.tran_evaluation.conditional_support import (
    AdaptiveReferenceConfig,
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    DEFAULT_CONDITIONAL_EVAL_MODE,
    build_full_H_schedule,
    build_local_reference_samples,
    build_local_reference_spec,
    build_uniform_sampling_specs_from_neighbors,
    fit_whitened_pca_metric,
    make_pair_label,
    minimal_adaptive_ess_target,
    pack_reference_support_arrays,
    sample_weighted_rows,
    sampling_spec_ess,
    sampling_spec_eig_rse,
    sampling_spec_indices,
    sampling_spec_weights,
    sampling_spec_mean_rse,
    sampling_spec_radius,
    summarize_reference_sampling_specs,
    transform_condition_vectors,
    validate_conditional_eval_mode,
    wasserstein1_wasserstein2_latents,
)


@eqx.filter_jit
def _sample_interval_batch(
    drift_net: Any,
    z_start_batch: jax.Array,
    tau_start: jax.Array,
    tau_end: jax.Array,
    dt0: jax.Array,
    keys: jax.Array,
    sigma_fn: Any,
) -> jax.Array:
    return jax.vmap(
        lambda y0, key: integrate_interval(
            drift_net,
            y0,
            tau_start,
            tau_end,
            dt0,
            key,
            sigma_fn,
        )
    )(z_start_batch, keys)


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / "eval" / "knn_reference"


def resolve_requested_phases(args: argparse.Namespace) -> list[str]:
    return resolve_requested_conditional_phases(
        phases_arg=getattr(args, "phases", None),
        skip_ecmmd=bool(getattr(args, "skip_ecmmd", False)),
    )


def _resolve_conditional_eval_mode(args: argparse.Namespace) -> str:
    return validate_conditional_eval_mode(
        getattr(args, "conditional_eval_mode", DEFAULT_CONDITIONAL_EVAL_MODE)
    )


def _build_adaptive_reference_config(args: argparse.Namespace) -> AdaptiveReferenceConfig:
    adaptive_ess_min = getattr(args, "adaptive_ess_min", None)
    if adaptive_ess_min is not None and int(adaptive_ess_min) <= 0:
        raise ValueError("--adaptive_ess_min must be positive when provided.")
    return AdaptiveReferenceConfig(
        metric_dim_cap=int(getattr(args, "adaptive_metric_dim_cap", 24)),
        bootstrap_reps=int(getattr(args, "adaptive_reference_bootstrap_reps", 64)),
        ess_min=int(adaptive_ess_min) if adaptive_ess_min is not None else 32,
    )


def _resolve_pair_adaptive_config(
    adaptive_config: AdaptiveReferenceConfig,
    *,
    n_conditions: int,
    adaptive_ess_min_override: int | None,
) -> AdaptiveReferenceConfig:
    ess_min = (
        int(adaptive_ess_min_override)
        if adaptive_ess_min_override is not None
        else minimal_adaptive_ess_target(int(n_conditions))
    )
    return replace(adaptive_config, ess_min=int(ess_min))


def sample_csp_conditionals(
    drift_net: Any,
    coarse_conditions: np.ndarray,
    *,
    tau_start: float | None = None,
    tau_end: float | None = None,
    zt: np.ndarray | None = None,
    dt0: float,
    sigma_fn: Any,
    n_realizations: int,
    seed: int,
    model_type: str = "legacy_unconditional",
    condition_mode: str | None = None,
    global_conditions: np.ndarray | None = None,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    delta_v: float | None = None,
    theta_feature_clip: float | None = None,
    sampling_max_batch_size: int | None = None,
    sample_conditional_batch_fn=sample_conditional_batch,
    sample_paired_prior_conditional_batch_fn=sample_paired_prior_conditional_batch,
) -> np.ndarray:
    coarse_np = np.asarray(coarse_conditions, dtype=np.float32)
    if coarse_np.ndim != 2:
        raise ValueError(f"coarse_conditions must have shape (n_conditions, latent_dim), got {coarse_np.shape}.")
    if int(n_realizations) <= 0:
        raise ValueError("n_realizations must be positive.")

    n_conditions = int(coarse_np.shape[0])
    n_realizations_int = int(n_realizations)
    generated = np.empty((n_conditions, n_realizations_int, coarse_np.shape[1]), dtype=np.float32)
    if str(model_type) == "paired_prior_bridge":
        if zt is None:
            raise ValueError("zt is required for paired-prior conditional interval sampling.")
        if delta_v is None:
            raise ValueError("delta_v is required for paired-prior conditional interval sampling.")
    global_np = None if global_conditions is None else np.asarray(global_conditions, dtype=np.float32)
    if global_np is not None and global_np.shape != coarse_np.shape:
        raise ValueError(
            "global_conditions must match coarse_conditions in shape before realization expansion, "
            f"got {global_np.shape} and {coarse_np.shape}."
        )

    for realization_start, realization_stop in iter_conditional_sample_realization_spans(
        n_conditions=n_conditions,
        n_realizations=n_realizations_int,
        sampling_max_batch_size=sampling_max_batch_size,
    ):
        realization_count = int(realization_stop - realization_start)
        repeated = np.repeat(coarse_np, realization_count, axis=0)
        chunk_seed = int(seed + realization_start)

        if str(model_type) == "paired_prior_bridge":
            traj = sample_paired_prior_conditional_batch_fn(
                drift_net,
                jnp.asarray(repeated, dtype=jnp.float32),
                jnp.asarray(np.asarray(zt, dtype=np.float32), dtype=jnp.float32),
                float(delta_v),
                float(dt0),
                jax.random.PRNGKey(chunk_seed),
                condition_num_intervals=condition_num_intervals,
                interval_offset=int(interval_offset),
                theta_feature_clip=float(theta_feature_clip if theta_feature_clip is not None else 1e-4),
            )
            generated[:, realization_start:realization_stop] = np.asarray(traj[:, 0, :], dtype=np.float32).reshape(
                n_conditions,
                realization_count,
                coarse_np.shape[1],
            )
            continue

        if condition_mode is None:
            if tau_start is None or tau_end is None:
                raise ValueError("tau_start and tau_end are required for legacy unconditional interval sampling.")
            keys = jax.random.split(jax.random.PRNGKey(chunk_seed), repeated.shape[0])
            generated[:, realization_start:realization_stop] = np.asarray(
                _sample_interval_batch(
                    drift_net,
                    jnp.asarray(repeated, dtype=jnp.float32),
                    jnp.asarray(float(tau_start), dtype=jnp.float32),
                    jnp.asarray(float(tau_end), dtype=jnp.float32),
                    jnp.asarray(float(dt0), dtype=jnp.float32),
                    keys,
                    sigma_fn,
                ),
                dtype=np.float32,
            ).reshape(n_conditions, realization_count, coarse_np.shape[1])
            continue

        if zt is None:
            raise ValueError("zt is required for sequential conditional interval sampling.")
        repeated_global = repeated if global_np is None else np.repeat(global_np, realization_count, axis=0)
        traj = sample_conditional_batch_fn(
            drift_net,
            jnp.asarray(repeated, dtype=jnp.float32),
            jnp.asarray(np.asarray(zt, dtype=np.float32), dtype=jnp.float32),
            sigma_fn,
            float(dt0),
            jax.random.PRNGKey(chunk_seed),
            condition_mode=str(condition_mode),
            global_condition_batch=jnp.asarray(repeated_global, dtype=jnp.float32),
            condition_num_intervals=condition_num_intervals,
            interval_offset=int(interval_offset),
        )
        generated[:, realization_start:realization_stop] = np.asarray(traj[:, 0, :], dtype=np.float32).reshape(
            n_conditions,
            realization_count,
            coarse_np.shape[1],
        )

    return generated


def _evaluate_scale_pair(
    *,
    pair_idx: int,
    latent_test: np.ndarray,
    corpus_z_coarse: np.ndarray,
    corpus_z_fine: np.ndarray,
    test_global_conditions: np.ndarray | None,
    corpus_global_conditions: np.ndarray | None,
    test_sample_indices: np.ndarray,
    corpus_eval_indices: np.ndarray,
    drift_net: Any,
    zt: np.ndarray,
    tau_knots: np.ndarray,
    condition_mode: str | None,
    condition_num_intervals: int | None,
    dt0: float,
    sigma_fn: Any,
    model_type: str,
    delta_v: float | None,
    theta_feature_clip: float | None,
    conditional_eval_mode: str,
    adaptive_config: AdaptiveReferenceConfig,
    adaptive_ess_min_override: int | None,
    k_neighbors: int,
    n_realizations: int,
    ecmmd_k_values: list[int],
    ecmmd_bootstrap_reps: int,
    n_plot_conditions: int,
    plot_value_budget: int,
    base_seed: int,
    rng: np.random.Generator,
    compute_w_metrics: bool = True,
    compute_ecmmd_metrics: bool = True,
    sample_conditional_batch_fn=sample_conditional_batch,
    sample_paired_prior_conditional_batch_fn=sample_paired_prior_conditional_batch,
    compute_chatterjee_local_scores_fn=None,
    compute_ecmmd_metrics_fn=None,
    add_bootstrap_ecmmd_calibration_fn=None,
) -> dict[str, object]:
    del n_plot_conditions, plot_value_budget
    pair_adaptive_config = _resolve_pair_adaptive_config(
        adaptive_config,
        n_conditions=int(corpus_z_coarse.shape[0]),
        adaptive_ess_min_override=adaptive_ess_min_override,
    )
    test_conditions = np.asarray(latent_test[pair_idx + 1, test_sample_indices], dtype=np.float32)
    condition_metric = None
    corpus_z_coarse_metric = None
    if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE:
        condition_metric = fit_whitened_pca_metric(
            corpus_z_coarse,
            variance_retained=float(pair_adaptive_config.variance_retained),
            dim_cap=int(pair_adaptive_config.metric_dim_cap),
        )
        corpus_z_coarse_metric = transform_condition_vectors(corpus_z_coarse, condition_metric)
    pair_zt = np.asarray(zt[pair_idx : pair_idx + 2], dtype=np.float32)
    interval_offset = int(len(zt) - 2 - pair_idx)
    test_generated = sample_csp_conditionals(
        drift_net,
        test_conditions,
        zt=pair_zt,
        tau_start=float(tau_knots[pair_idx + 1]),
        tau_end=float(tau_knots[pair_idx]),
        dt0=float(dt0),
        sigma_fn=sigma_fn,
        n_realizations=n_realizations,
        seed=base_seed,
        model_type=model_type,
        condition_mode=condition_mode,
        global_conditions=test_global_conditions,
        condition_num_intervals=condition_num_intervals,
        interval_offset=interval_offset,
        delta_v=delta_v,
        theta_feature_clip=theta_feature_clip,
        sample_conditional_batch_fn=sample_conditional_batch_fn,
        sample_paired_prior_conditional_batch_fn=sample_paired_prior_conditional_batch_fn,
    )

    w_sampling_specs: list[dict[str, object]] = []
    latent_w1_values: list[float] = []
    latent_w2_values: list[float] = []
    latent_w1_null_values: list[float] = []
    latent_w2_null_values: list[float] = []
    for sample_offset, z_test_coarse in enumerate(test_conditions):
        sampling_spec = build_local_reference_spec(
            query=z_test_coarse,
            corpus_conditions=corpus_z_coarse,
            corpus_targets=corpus_z_fine,
            conditional_eval_mode=conditional_eval_mode,
            k_neighbors=k_neighbors,
            condition_metric=condition_metric,
            corpus_conditions_transformed=corpus_z_coarse_metric,
            adaptive_config=pair_adaptive_config,
            rng=rng,
        )
        w_sampling_specs.append(sampling_spec)
        if not compute_w_metrics:
            continue
        support_idx = sampling_spec_indices(sampling_spec)
        support_weights = sampling_spec_weights(sampling_spec)
        ref_latents = corpus_z_fine[support_idx]
        z_gen_np = test_generated[sample_offset]

        latent_w1, latent_w2 = wasserstein1_wasserstein2_latents(
            z_gen_np,
            ref_latents,
            weights_a=None,
            weights_b=np.asarray(support_weights, dtype=np.float64),
        )
        latent_w1_values.append(latent_w1)
        latent_w2_values.append(latent_w2)

        n_null = int(min(n_realizations, corpus_z_fine.shape[0]))
        null_idx = rng.choice(corpus_z_fine.shape[0], size=n_null, replace=False)
        z_null = corpus_z_fine[null_idx]
        latent_w1_null, latent_w2_null = wasserstein1_wasserstein2_latents(
            z_null,
            ref_latents,
            weights_a=None,
            weights_b=np.asarray(support_weights, dtype=np.float64),
        )
        latent_w1_null_values.append(latent_w1_null)
        latent_w2_null_values.append(latent_w2_null)

        if (sample_offset + 1) % 10 == 0 or sample_offset == 0:
            print(
                f"  Test condition {sample_offset + 1}/{len(test_sample_indices)}: "
                f"latent W1={latent_w1:.4f}  latent W2={latent_w2:.4f}",
                flush=True,
            )
    (
        _w_reference_support,
        w_reference_weights,
        w_reference_indices,
        w_reference_counts,
    ) = pack_reference_support_arrays(corpus_z_fine, w_sampling_specs)

    ecmmd_conditions = corpus_z_coarse[corpus_eval_indices].astype(np.float32)
    ecmmd_generated = sample_csp_conditionals(
        drift_net,
        ecmmd_conditions,
        zt=pair_zt,
        tau_start=float(tau_knots[pair_idx + 1]),
        tau_end=float(tau_knots[pair_idx]),
        dt0=float(dt0),
        sigma_fn=sigma_fn,
        n_realizations=n_realizations,
        seed=base_seed + 100_000,
        model_type=model_type,
        condition_mode=condition_mode,
        global_conditions=corpus_global_conditions,
        condition_num_intervals=condition_num_intervals,
        interval_offset=interval_offset,
        delta_v=delta_v,
        theta_feature_clip=theta_feature_clip,
        sample_conditional_batch_fn=sample_conditional_batch_fn,
        sample_paired_prior_conditional_batch_fn=sample_paired_prior_conditional_batch_fn,
    )
    ecmmd_observed_reference = corpus_z_fine[corpus_eval_indices].astype(np.float32)
    visualization_k_requested = (
        int(ecmmd_k_values[0])
        if ecmmd_k_values
        else int(max(1, min(int(k_neighbors), int(max(1, ecmmd_conditions.shape[0] - 1)))))
    )
    graph_condition_vectors = (
        transform_condition_vectors(ecmmd_conditions, condition_metric)
        if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE and condition_metric is not None
        else None
    )

    if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE:
        if compute_chatterjee_local_scores_fn is None:
            raise ValueError("Chatterjee ECMMD construction requires compute_chatterjee_local_scores_fn.")
        chatterjee_payload = compute_chatterjee_local_scores_fn(
            ecmmd_conditions,
            ecmmd_observed_reference,
            ecmmd_generated,
            visualization_k_requested,
        )
        ecmmd_sampling_specs = build_uniform_sampling_specs_from_neighbors(
            np.asarray(chatterjee_payload["neighbor_indices"], dtype=np.int64),
            neighbor_radii=np.asarray(chatterjee_payload["neighbor_radii"], dtype=np.float64),
            mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
        )
        ecmmd_reference = np.stack(
            [
                sample_weighted_rows(
                    ecmmd_observed_reference,
                    sampling_spec_indices(spec),
                    sampling_spec_weights(spec),
                    int(n_realizations),
                    np.random.default_rng(int(base_seed) + 200_000 + int(row)),
                )
                for row, spec in enumerate(ecmmd_sampling_specs)
            ],
            axis=0,
        ).astype(np.float32)
        reference_support, reference_weights, reference_indices, reference_counts = pack_reference_support_arrays(
            ecmmd_observed_reference,
            ecmmd_sampling_specs,
        )
        adaptive_radii = np.asarray([sampling_spec_radius(spec) for spec in ecmmd_sampling_specs], dtype=np.float32)
        reference_ess = np.asarray([sampling_spec_ess(spec) for spec in ecmmd_sampling_specs], dtype=np.float32)
        reference_mean_rse = np.asarray(
            [sampling_spec_mean_rse(spec) for spec in ecmmd_sampling_specs],
            dtype=np.float32,
        )
        reference_eig_rse = np.stack(
            [sampling_spec_eig_rse(spec) for spec in ecmmd_sampling_specs],
            axis=0,
        ).astype(np.float32)
        local_scores = np.asarray(chatterjee_payload["derandomized_scores"], dtype=np.float32)
        neighborhood_indices = np.asarray(chatterjee_payload["neighbor_indices"], dtype=np.int64)
        neighborhood_radii = np.asarray(chatterjee_payload["neighbor_radii"], dtype=np.float32)
        neighborhood_distances = np.asarray(chatterjee_payload["neighbor_distances"], dtype=np.float32)
        ecmmd_real_samples = ecmmd_observed_reference
        ecmmd_reference_weights = None
    else:
        ecmmd_reference, ecmmd_sampling_specs = build_local_reference_samples(
            conditions=ecmmd_conditions,
            corpus_conditions=corpus_z_coarse,
            corpus_targets=corpus_z_fine,
            corpus_condition_indices=corpus_eval_indices,
            k_neighbors=k_neighbors,
            n_realizations=n_realizations,
            rng=rng,
            conditional_eval_mode=conditional_eval_mode,
            condition_metric=condition_metric,
            corpus_conditions_transformed=corpus_z_coarse_metric,
            adaptive_config=pair_adaptive_config,
        )
        reference_support, reference_weights, reference_indices, reference_counts = pack_reference_support_arrays(
            corpus_z_fine,
            ecmmd_sampling_specs,
        )
        adaptive_radii = np.asarray([sampling_spec_radius(spec) for spec in ecmmd_sampling_specs], dtype=np.float32)
        reference_ess = np.asarray([sampling_spec_ess(spec) for spec in ecmmd_sampling_specs], dtype=np.float32)
        reference_mean_rse = np.asarray(
            [sampling_spec_mean_rse(spec) for spec in ecmmd_sampling_specs],
            dtype=np.float32,
        )
        reference_eig_rse = np.stack(
            [sampling_spec_eig_rse(spec) for spec in ecmmd_sampling_specs],
            axis=0,
        ).astype(np.float32)
        local_scores = np.full((ecmmd_conditions.shape[0],), np.nan, dtype=np.float32)
        neighborhood_indices = np.full((ecmmd_conditions.shape[0], 1), -1, dtype=np.int64)
        neighborhood_radii = adaptive_radii.astype(np.float32)
        neighborhood_distances = np.full((ecmmd_conditions.shape[0], 1), np.nan, dtype=np.float32)
        ecmmd_real_samples = reference_support
        ecmmd_reference_weights = reference_weights

    latent_ecmmd: dict[str, object] | None = None
    if compute_ecmmd_metrics:
        if compute_ecmmd_metrics_fn is None or add_bootstrap_ecmmd_calibration_fn is None:
            raise ValueError("ECMMD computation requires injected metric and calibration functions.")
        latent_ecmmd = compute_ecmmd_metrics_fn(
            ecmmd_conditions,
            ecmmd_real_samples,
            ecmmd_generated,
            ecmmd_k_values,
            reference_weights=ecmmd_reference_weights,
            condition_graph_mode=conditional_eval_mode,
            graph_condition_vectors=graph_condition_vectors,
            adaptive_radii=adaptive_radii if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE else None,
        )
        latent_ecmmd = add_bootstrap_ecmmd_calibration_fn(
            latent_ecmmd,
            conditions=ecmmd_conditions,
            reference_samples=ecmmd_real_samples,
            corpus_targets=ecmmd_observed_reference if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE else corpus_z_fine,
            sampling_specs=ecmmd_sampling_specs,
            k_values=ecmmd_k_values,
            n_bootstrap=ecmmd_bootstrap_reps,
            rng=rng,
            reference_weights=ecmmd_reference_weights,
            condition_graph_mode=conditional_eval_mode,
            graph_condition_vectors=graph_condition_vectors,
            adaptive_radii=adaptive_radii if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE else None,
            generated_samples=ecmmd_generated,
        )
        latent_ecmmd["reference_mode"] = conditional_eval_mode
        latent_ecmmd["reference_diagnostics"] = summarize_reference_sampling_specs(ecmmd_sampling_specs)
        latent_ecmmd["visualization_k_requested"] = int(visualization_k_requested)
        if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE:
            latent_ecmmd["visualization_k_effective"] = int(chatterjee_payload["k_effective"])
        if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE and condition_metric is not None:
            latent_ecmmd["condition_metric"] = {
                "retained_dim": int(condition_metric.retained_dim),
                "explained_variance": float(condition_metric.explained_variance),
            }

    return {
        "latent_w1_values": (
            None if not compute_w_metrics else np.asarray(latent_w1_values, dtype=np.float64)
        ),
        "latent_w2_values": (
            None if not compute_w_metrics else np.asarray(latent_w2_values, dtype=np.float64)
        ),
        "latent_w1_null_values": (
            None if not compute_w_metrics else np.asarray(latent_w1_null_values, dtype=np.float64)
        ),
        "latent_w2_null_values": (
            None if not compute_w_metrics else np.asarray(latent_w2_null_values, dtype=np.float64)
        ),
        "latent_w2_conditions": test_conditions.astype(np.float32),
        "latent_w2_generated": test_generated.astype(np.float32),
        "latent_w2_reference_support_indices": w_reference_indices.astype(np.int64),
        "latent_w2_reference_support_weights": w_reference_weights.astype(np.float32),
        "latent_w2_reference_support_counts": w_reference_counts.astype(np.int64),
        "latent_ecmmd_conditions": ecmmd_conditions.astype(np.float32),
        "latent_ecmmd_reference": ecmmd_reference.astype(np.float32),
        "latent_ecmmd_observed_reference": ecmmd_observed_reference.astype(np.float32),
        "latent_ecmmd_generated": ecmmd_generated.astype(np.float32),
        "latent_ecmmd": latent_ecmmd,
        "latent_ecmmd_local_scores": local_scores.astype(np.float32),
        "latent_ecmmd_neighbor_indices": neighborhood_indices.astype(np.int64),
        "latent_ecmmd_neighbor_radii": neighborhood_radii.astype(np.float32),
        "latent_ecmmd_neighbor_distances": neighborhood_distances.astype(np.float32),
        "latent_ecmmd_reference_support_indices": reference_indices.astype(np.int64),
        "latent_ecmmd_reference_support_weights": reference_weights.astype(np.float32),
        "latent_ecmmd_reference_support_counts": reference_counts.astype(np.int64),
        "latent_ecmmd_reference_radius": adaptive_radii.astype(np.float32),
        "latent_ecmmd_reference_ess": reference_ess.astype(np.float32),
        "latent_ecmmd_reference_mean_rse": reference_mean_rse.astype(np.float32),
        "latent_ecmmd_reference_eig_rse": reference_eig_rse.astype(np.float32),
        "adaptive_ess_min": int(pair_adaptive_config.ess_min),
    }


def _build_sample_cache_manifest(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
    runtime,
    latents_path: Path,
    corpus_latents_path: Path,
    time_indices: np.ndarray,
    zt: np.ndarray,
    tau_knots: np.ndarray,
    conditional_eval_mode: str,
    adaptive_config: AdaptiveReferenceConfig,
    full_h_schedule: list[float],
    ecmmd_k_values: list[int],
) -> dict[str, Any]:
    return build_conditional_sample_cache_manifest(
        fingerprint={
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "model_type": str(runtime.model_type),
            "condition_mode": str(runtime.condition_mode if runtime.model_type in {"conditional_bridge", "paired_prior_bridge"} else None),
            "source_run_dir": (
                str(getattr(runtime.source, "source_run_dir"))
                if getattr(runtime.source, "source_run_dir", None) is not None
                else None
            ),
            "dataset_path": (
                str(getattr(runtime.source, "dataset_path"))
                if getattr(runtime.source, "dataset_path", None) is not None
                else None
            ),
            "source_latents_path": str(latents_path),
            "corpus_latents_path": str(corpus_latents_path),
            "conditional_eval_mode": str(conditional_eval_mode),
            "adaptive_metric_dim_cap": int(adaptive_config.metric_dim_cap),
            "adaptive_ess_min": (
                int(args.adaptive_ess_min) if getattr(args, "adaptive_ess_min", None) is not None else None
            ),
            "n_test_samples": int(args.n_test_samples),
            "n_realizations": int(args.n_realizations),
            "k_neighbors": int(args.k_neighbors),
            "seed": int(args.seed),
            "time_indices": np.asarray(time_indices, dtype=np.int64).tolist(),
            "zt": np.asarray(zt, dtype=np.float32).tolist(),
            "tau_knots": np.asarray(tau_knots, dtype=np.float32).tolist(),
            "full_H_schedule": list(map(float, full_h_schedule)),
            "ecmmd_k_values_requested": list(map(int, ecmmd_k_values)),
        }
    )


def _load_sample_metadata(
    *,
    output_dir: Path,
    requested_phases: list[str],
    sample_cache_manifest: dict[str, Any],
):
    sample_cache = None
    sample_metadata: dict[str, np.ndarray] | None = None
    if CONDITIONAL_PHASE_SAMPLE in requested_phases:
        sample_cache = prepare_conditional_sample_cache(
            output_dir=output_dir,
            manifest=sample_cache_manifest,
        )
        if sample_cache.has_chunk("metadata"):
            sample_metadata = load_conditional_sample_metadata_chunk(sample_cache)
    elif conditional_sample_cache_matches(
        output_dir=output_dir,
        manifest=sample_cache_manifest,
        require_complete=True,
    ):
        sample_metadata = load_conditional_sample_metadata(
            output_dir=output_dir,
            manifest=sample_cache_manifest,
        )
    return sample_cache, sample_metadata


def _print_evaluation_header(
    *,
    run_dir: Path,
    output_dir: Path,
    runtime,
    condition_mode: str | None,
    conditional_eval_mode: str,
    requested_phases: list[str],
    latents_path: Path,
    corpus_latents_path: Path,
    n_test_samples: int,
    n_ecmmd_conditions: int,
    args: argparse.Namespace,
) -> None:
    print("============================================================", flush=True)
    print("CSP kNN reference evaluation", flush=True)
    print(f"  run_dir            : {run_dir}", flush=True)
    print(f"  output_dir         : {output_dir}", flush=True)
    print(f"  model_type         : {runtime.model_type}", flush=True)
    print(f"  condition_mode     : {condition_mode}", flush=True)
    print(f"  conditional_eval_mode : {conditional_eval_mode}", flush=True)
    print(f"  stages             : {', '.join(requested_phases) if requested_phases else 'none'}", flush=True)
    print(f"  source_latents     : {latents_path}", flush=True)
    print(f"  corpus_latents     : {corpus_latents_path}", flush=True)
    print(f"  n_test_samples     : {n_test_samples}", flush=True)
    print(f"  n_ecmmd_conditions : {n_ecmmd_conditions}", flush=True)
    print(f"  n_realizations     : {args.n_realizations}", flush=True)
    print(f"  k_neighbors        : {args.k_neighbors}", flush=True)
    print(f"  skip_reports       : {bool(getattr(args, 'skip_ecmmd', False))}", flush=True)
    print(f"  n_plot_conditions  : {max(0, min(args.n_plot_conditions, n_test_samples))}", flush=True)
    print(f"  plot_value_budget  : {args.plot_value_budget}", flush=True)
    print("============================================================", flush=True)


def _resolve_saved_indices(
    *,
    saved: dict[str, np.ndarray] | None,
    key: str,
    n_available: int,
) -> np.ndarray | None:
    if saved is None or key not in saved:
        return None
    indices = np.asarray(saved[key], dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError(f"Saved {key} is empty. Re-run with --phases reference_cache.")
    if np.any(indices < 0) or np.any(indices >= n_available):
        raise ValueError(f"Saved {key} are out of range: {indices.tolist()}.")
    return indices


def _compute_latent_w_metrics_from_payload(
    *,
    pair_label: str,
    pair_sample_payload: dict[str, object],
    corpus_z_fine: np.ndarray,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generated = np.asarray(pair_sample_payload["latent_w2_generated"], dtype=np.float32)
    support_indices = np.asarray(pair_sample_payload["latent_w2_reference_support_indices"], dtype=np.int64)
    support_weights = np.asarray(pair_sample_payload["latent_w2_reference_support_weights"], dtype=np.float64)
    support_counts = np.asarray(pair_sample_payload["latent_w2_reference_support_counts"], dtype=np.int64)
    rng = np.random.default_rng(int(base_seed))

    latent_w1_values: list[float] = []
    latent_w2_values: list[float] = []
    latent_w1_null_values: list[float] = []
    latent_w2_null_values: list[float] = []
    for sample_offset in range(int(generated.shape[0])):
        take = int(support_counts[sample_offset]) if support_counts.size > 0 else int(np.sum(support_indices[sample_offset] >= 0))
        if take <= 0:
            raise ValueError(f"No reference support rows saved for {pair_label} sample_offset={sample_offset}.")
        ref_latents = np.asarray(
            corpus_z_fine[np.asarray(support_indices[sample_offset, :take], dtype=np.int64)],
            dtype=np.float32,
        )
        ref_weights = np.asarray(support_weights[sample_offset, :take], dtype=np.float64)
        z_gen_np = np.asarray(generated[sample_offset], dtype=np.float32)
        latent_w1, latent_w2 = wasserstein1_wasserstein2_latents(
            z_gen_np,
            ref_latents,
            weights_a=None,
            weights_b=ref_weights,
        )
        latent_w1_values.append(latent_w1)
        latent_w2_values.append(latent_w2)

        n_null = int(min(generated.shape[1], corpus_z_fine.shape[0]))
        null_idx = rng.choice(corpus_z_fine.shape[0], size=n_null, replace=False)
        z_null = np.asarray(corpus_z_fine[null_idx], dtype=np.float32)
        latent_w1_null, latent_w2_null = wasserstein1_wasserstein2_latents(
            z_null,
            ref_latents,
            weights_a=None,
            weights_b=ref_weights,
        )
        latent_w1_null_values.append(latent_w1_null)
        latent_w2_null_values.append(latent_w2_null)

        if (sample_offset + 1) % 10 == 0 or sample_offset == 0:
            print(
                f"  Test condition {sample_offset + 1}/{generated.shape[0]}: "
                f"latent W1={latent_w1:.4f}  latent W2={latent_w2:.4f}",
                flush=True,
            )

    return (
        np.asarray(latent_w1_values, dtype=np.float64),
        np.asarray(latent_w2_values, dtype=np.float64),
        np.asarray(latent_w1_null_values, dtype=np.float64),
        np.asarray(latent_w2_null_values, dtype=np.float64),
    )


def _decode_vector_latents_to_fields(
    latents: np.ndarray,
    *,
    decode_context,
    decode_batch_size: int = 64,
) -> np.ndarray:
    z = np.asarray(latents, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Vector decode expects shape (N, K), got {z.shape}.")
    parts: list[np.ndarray] = []
    batch_size = max(1, int(decode_batch_size))
    for start in range(0, int(z.shape[0]), batch_size):
        stop = min(start + batch_size, int(z.shape[0]))
        z_batch = z[start:stop]
        x_batch = np.broadcast_to(
            np.asarray(decode_context.grid_coords, dtype=np.float32)[None, ...],
            (z_batch.shape[0], *np.asarray(decode_context.grid_coords).shape),
        )
        decoded = np.asarray(decode_context.decode_fn(z_batch, x_batch), dtype=np.float32)
        if decoded.ndim == 3:
            decoded = decoded.squeeze(-1)
        clip_bounds = getattr(decode_context, "clip_bounds", None)
        if clip_bounds is not None:
            decoded = np.clip(decoded, float(clip_bounds[0]), float(clip_bounds[1]))
        parts.append(decoded)
    decoded_fields = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float32)
    return np.asarray(
        apply_inverse_transform(
            decoded_fields,
            getattr(decode_context, "transform_info"),
        ),
        dtype=np.float32,
    )


def _compute_field_metrics_from_payload(
    *,
    args: argparse.Namespace,
    pair_label: str,
    pair_metadata: dict[str, object],
    pair_sample_payload: dict[str, object],
    corpus_z_fine: np.ndarray,
    test_sample_indices: np.ndarray,
    reference_sampling_seed: int,
    representative_seed: int,
    output_dir: Path,
    decode_context,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_latents = np.asarray(pair_sample_payload["latent_ecmmd_generated"], dtype=np.float32)
    support_indices = np.asarray(pair_sample_payload["latent_ecmmd_reference_support_indices"], dtype=np.int64)
    support_weights = np.asarray(pair_sample_payload["latent_ecmmd_reference_support_weights"], dtype=np.float64)
    support_counts = np.asarray(pair_sample_payload["latent_ecmmd_reference_support_counts"], dtype=np.int64)
    reference_latents = sample_reference_latent_draws(
        values=np.asarray(corpus_z_fine, dtype=np.float32),
        support_indices=support_indices,
        support_weights=support_weights,
        support_counts=support_counts,
        n_draws=int(generated_latents.shape[1]),
        base_seed=int(reference_sampling_seed),
    )
    generated_fields = _decode_vector_latents_to_fields(
        generated_latents.reshape(-1, generated_latents.shape[-1]),
        decode_context=decode_context,
    ).reshape(generated_latents.shape[0], generated_latents.shape[1], -1)
    reference_fields = _decode_vector_latents_to_fields(
        reference_latents.reshape(-1, reference_latents.shape[-1]),
        decode_context=decode_context,
    ).reshape(reference_latents.shape[0], reference_latents.shape[1], -1)
    field_metrics, figure_manifest = evaluate_pair_field_metrics(
        pair_label=pair_label,
        pair_display_label=str(pair_metadata["display_label"]),
        pair_h_value=float(pair_metadata["H_fine"]),
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        conditions=np.asarray(pair_sample_payload["latent_ecmmd_conditions"], dtype=np.float32),
        reference_fields=reference_fields,
        generated_fields=generated_fields,
        resolution=int(decode_context.resolution),
        pixel_size=float(
            infer_pixel_size_from_grid(
                grid_coords=np.asarray(decode_context.grid_coords, dtype=np.float32),
                resolution=int(decode_context.resolution),
            )
        ),
        min_spacing_pixels=4,
        representative_seed=int(representative_seed),
        n_plot_conditions=int(max(0, getattr(args, "n_plot_conditions", 0))),
        output_dir=output_dir,
    )
    return field_metrics, figure_manifest


def _sampling_specs_from_saved_support(
    *,
    pair_sample_payload: dict[str, object],
    conditional_eval_mode: str,
    metric_retained_dim: int = 0,
    metric_explained_variance: float = 1.0,
) -> list[dict[str, object]]:
    support_indices = np.asarray(pair_sample_payload["latent_ecmmd_reference_support_indices"], dtype=np.int64)
    support_weights = np.asarray(pair_sample_payload["latent_ecmmd_reference_support_weights"], dtype=np.float64)
    support_counts = np.asarray(pair_sample_payload["latent_ecmmd_reference_support_counts"], dtype=np.int64)
    support_radius = np.asarray(pair_sample_payload["latent_ecmmd_reference_radius"], dtype=np.float64)
    support_ess = np.asarray(pair_sample_payload["latent_ecmmd_reference_ess"], dtype=np.float64)
    support_mean_rse = np.asarray(pair_sample_payload["latent_ecmmd_reference_mean_rse"], dtype=np.float64)
    support_eig_rse = np.asarray(pair_sample_payload["latent_ecmmd_reference_eig_rse"], dtype=np.float64)

    specs: list[dict[str, object]] = []
    for row in range(int(support_indices.shape[0])):
        take = int(support_counts[row]) if support_counts.size > 0 else int(np.sum(support_indices[row] >= 0))
        candidate_indices = np.asarray(support_indices[row, :take], dtype=np.int64)
        candidate_weights = np.asarray(support_weights[row, :take], dtype=np.float64)
        specs.append(
            {
                "mode": str(conditional_eval_mode),
                "candidate_indices": candidate_indices,
                "candidate_weights": candidate_weights,
                "radius": float(support_radius[row]),
                "ess": float(support_ess[row]),
                "support_size": int(candidate_indices.shape[0]),
                "metric_retained_dim": int(metric_retained_dim),
                "metric_explained_variance": float(metric_explained_variance),
                "mean_rse": float(support_mean_rse[row]),
                "eig_rse": np.asarray(support_eig_rse[row], dtype=np.float64),
                "passed_stability": True,
            }
        )
    return specs


def _compute_latent_ecmmd_from_payload(
    *,
    pair_sample_payload: dict[str, object],
    conditional_eval_mode: str,
    ecmmd_k_values: list[int],
    ecmmd_bootstrap_reps: int,
    pair_adaptive_config: AdaptiveReferenceConfig,
    corpus_z_coarse_flat: np.ndarray,
    corpus_z_fine_flat: np.ndarray,
    base_seed: int,
    compute_chatterjee_local_scores_fn,
    compute_ecmmd_metrics_fn,
    add_bootstrap_ecmmd_calibration_fn,
) -> dict[str, object]:
    conditions = np.asarray(pair_sample_payload["latent_ecmmd_conditions"], dtype=np.float32)
    observed_reference = np.asarray(pair_sample_payload["latent_ecmmd_observed_reference"], dtype=np.float32)
    generated = np.asarray(pair_sample_payload["latent_ecmmd_generated"], dtype=np.float32)
    rng = np.random.default_rng(int(base_seed))

    if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE:
        visualization_k_requested = (
            int(ecmmd_k_values[0])
            if ecmmd_k_values
            else int(max(1, min(int(observed_reference.shape[0] - 1), int(observed_reference.shape[0] - 1))))
        )
        chatterjee_payload = compute_chatterjee_local_scores_fn(
            conditions,
            observed_reference,
            generated,
            visualization_k_requested,
        )
        sampling_specs = build_uniform_sampling_specs_from_neighbors(
            np.asarray(chatterjee_payload["neighbor_indices"], dtype=np.int64),
            neighbor_radii=np.asarray(chatterjee_payload["neighbor_radii"], dtype=np.float64),
            mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
        )
        _reference_support, reference_weights, reference_indices, reference_counts = pack_reference_support_arrays(
            observed_reference,
            sampling_specs,
        )
        reference_radius = np.asarray([sampling_spec_radius(spec) for spec in sampling_specs], dtype=np.float32)
        reference_ess = np.asarray([sampling_spec_ess(spec) for spec in sampling_specs], dtype=np.float32)
        reference_mean_rse = np.asarray(
            [sampling_spec_mean_rse(spec) for spec in sampling_specs],
            dtype=np.float32,
        )
        reference_eig_rse = np.stack(
            [sampling_spec_eig_rse(spec) for spec in sampling_specs],
            axis=0,
        ).astype(np.float32)
        local_scores = np.asarray(chatterjee_payload["derandomized_scores"], dtype=np.float32)
        neighborhood_indices = np.asarray(chatterjee_payload["neighbor_indices"], dtype=np.int64)
        neighborhood_radii = np.asarray(chatterjee_payload["neighbor_radii"], dtype=np.float32)
        neighborhood_distances = np.asarray(chatterjee_payload["neighbor_distances"], dtype=np.float32)
        reference_samples = observed_reference
        reference_sample_weights = None
        graph_condition_vectors = None
        adaptive_radii = None
        corpus_targets = observed_reference
        latent_ecmmd = compute_ecmmd_metrics_fn(
            conditions,
            reference_samples,
            generated,
            ecmmd_k_values,
            condition_graph_mode=conditional_eval_mode,
        )
    else:
        condition_metric = fit_whitened_pca_metric(
            corpus_z_coarse_flat,
            variance_retained=float(pair_adaptive_config.variance_retained),
            dim_cap=int(pair_adaptive_config.metric_dim_cap),
        )
        graph_condition_vectors = transform_condition_vectors(conditions, condition_metric)
        sampling_specs = _sampling_specs_from_saved_support(
            pair_sample_payload=pair_sample_payload,
            conditional_eval_mode=conditional_eval_mode,
            metric_retained_dim=int(condition_metric.retained_dim),
            metric_explained_variance=float(condition_metric.explained_variance),
        )
        reference_samples, reference_weights, reference_indices, reference_counts = pack_reference_support_arrays(
            corpus_z_fine_flat,
            sampling_specs,
        )
        reference_radius = np.asarray([sampling_spec_radius(spec) for spec in sampling_specs], dtype=np.float32)
        reference_ess = np.asarray([sampling_spec_ess(spec) for spec in sampling_specs], dtype=np.float32)
        reference_mean_rse = np.asarray(
            [sampling_spec_mean_rse(spec) for spec in sampling_specs],
            dtype=np.float32,
        )
        reference_eig_rse = np.stack(
            [sampling_spec_eig_rse(spec) for spec in sampling_specs],
            axis=0,
        ).astype(np.float32)
        local_scores = np.asarray(pair_sample_payload["latent_ecmmd_local_scores"], dtype=np.float32)
        neighborhood_indices = np.asarray(pair_sample_payload["latent_ecmmd_neighbor_indices"], dtype=np.int64)
        neighborhood_radii = np.asarray(pair_sample_payload["latent_ecmmd_neighbor_radii"], dtype=np.float32)
        neighborhood_distances = np.asarray(pair_sample_payload["latent_ecmmd_neighbor_distances"], dtype=np.float32)
        reference_sample_weights = reference_weights
        adaptive_radii = reference_radius
        corpus_targets = corpus_z_fine_flat
        latent_ecmmd = compute_ecmmd_metrics_fn(
            conditions,
            reference_samples,
            generated,
            ecmmd_k_values,
            reference_weights=reference_sample_weights,
            condition_graph_mode=conditional_eval_mode,
            graph_condition_vectors=graph_condition_vectors,
            adaptive_radii=adaptive_radii,
        )
        latent_ecmmd["condition_metric"] = {
            "retained_dim": int(condition_metric.retained_dim),
            "explained_variance": float(condition_metric.explained_variance),
        }

    latent_ecmmd = add_bootstrap_ecmmd_calibration_fn(
        latent_ecmmd,
        conditions=conditions,
        reference_samples=reference_samples,
        corpus_targets=corpus_targets,
        sampling_specs=sampling_specs,
        k_values=ecmmd_k_values,
        n_bootstrap=ecmmd_bootstrap_reps,
        rng=rng,
        reference_weights=reference_sample_weights,
        condition_graph_mode=conditional_eval_mode,
        graph_condition_vectors=graph_condition_vectors,
        adaptive_radii=adaptive_radii,
        generated_samples=generated,
    )
    latent_ecmmd["reference_mode"] = conditional_eval_mode
    latent_ecmmd["reference_diagnostics"] = summarize_reference_sampling_specs(sampling_specs)
    if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE:
        latent_ecmmd["visualization_k_requested"] = int(visualization_k_requested)
        latent_ecmmd["visualization_k_effective"] = int(chatterjee_payload["k_effective"])

    return {
        "latent_ecmmd": latent_ecmmd,
        "latent_ecmmd_local_scores": local_scores,
        "latent_ecmmd_neighbor_indices": neighborhood_indices,
        "latent_ecmmd_neighbor_radii": neighborhood_radii,
        "latent_ecmmd_neighbor_distances": neighborhood_distances,
        "latent_ecmmd_reference_support_indices": reference_indices.astype(np.int64),
        "latent_ecmmd_reference_support_weights": reference_weights.astype(np.float32),
        "latent_ecmmd_reference_support_counts": reference_counts.astype(np.int64),
        "latent_ecmmd_reference_radius": reference_radius,
        "latent_ecmmd_reference_ess": reference_ess,
        "latent_ecmmd_reference_mean_rse": reference_mean_rse,
        "latent_ecmmd_reference_eig_rse": reference_eig_rse,
    }


def _build_summary_text(
    *,
    args: argparse.Namespace,
    conditional_eval_mode: str,
    adaptive_config: AdaptiveReferenceConfig,
    test_sample_indices: np.ndarray,
    corpus_eval_indices: np.ndarray,
    n_corpus: int,
    pair_labels: list[str],
    metrics: dict[str, object],
) -> str:
    lines = [
        "CSP kNN Reference Evaluation",
        "=" * 50,
        f"conditional_eval_mode: {conditional_eval_mode}",
        f"k_neighbors: {args.k_neighbors}",
        f"n_test_samples: {len(test_sample_indices)}",
        f"n_ecmmd_conditions: {len(corpus_eval_indices)}",
        f"n_realizations: {args.n_realizations}",
        f"n_corpus: {n_corpus}",
        f"ecmmd_k_values_requested: {parse_positive_int_list_arg(args.ecmmd_k_values)}",
        f"ecmmd_bootstrap_reps: {args.ecmmd_bootstrap_reps}",
        f"adaptive_metric_dim_cap: {adaptive_config.metric_dim_cap}",
        f"adaptive_reference_bootstrap_reps: {adaptive_config.bootstrap_reps}",
        (
            f"adaptive_ess_min: {int(args.adaptive_ess_min)}"
            if getattr(args, "adaptive_ess_min", None) is not None
            else "adaptive_ess_min: auto=min(32, max(8, floor(sqrt(n_interval))))"
        ),
        "",
    ]

    scale_pairs = metrics["scale_pairs"]
    for pair_label in pair_labels:
        pair_metrics = scale_pairs[pair_label]
        pair_meta = pair_metrics["pair_metadata"]
        w1 = pair_metrics["latent_w1"]
        w2 = pair_metrics["latent_w2"]
        lines.append(
            f"{pair_label}: {pair_meta['display_label']} "
            f"(modeled marginal {pair_meta['modeled_marginal_coarse_order']}/{pair_meta['modeled_n_marginals']} "
            f"-> {pair_meta['modeled_marginal_fine_order']}/{pair_meta['modeled_n_marginals']}, "
            f"dataset idx {pair_meta['tidx_coarse']} -> {pair_meta['tidx_fine']}, "
            f"adaptive_ess_min={pair_metrics['adaptive_ess_min']})"
        )
        if bool(w1.get("deferred")) or bool(w2.get("deferred")):
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent W1/W2 deferred; reusable reference cache saved"
            )
        else:
            w1_null = pair_metrics["latent_w1_null"]
            w2_null = pair_metrics["latent_w2_null"]
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent W1 = {w1['mean']:.4f} +/- {w1['std']:.4f} "
                f"(median={w1['median']:.4f}, range=[{w1['min']:.4f}, {w1['max']:.4f}])"
            )
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent W2 = {w2['mean']:.4f} +/- {w2['std']:.4f} "
                f"(median={w2['median']:.4f}, range=[{w2['min']:.4f}, {w2['max']:.4f}])"
            )
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent W1 null = {w1_null['mean']:.4f} +/- {w1_null['std']:.4f} "
                f"(skill={pair_metrics['latent_w1_skill_vs_null']:+.4f})"
            )
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent W2 null = {w2_null['mean']:.4f} +/- {w2_null['std']:.4f} "
                f"(skill={pair_metrics['latent_w2_skill_vs_null']:+.4f})"
            )
        ecmmd_metrics = pair_metrics["latent_ecmmd"]
        if bool(ecmmd_metrics.get("deferred")):
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent ECMMD deferred; reusable reference cache saved"
            )
        elif "skipped_reason" in ecmmd_metrics:
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent ECMMD skipped: {ecmmd_metrics['skipped_reason']}"
            )
        elif "bandwidth" in ecmmd_metrics:
            lines.append(f"{'':>{len(pair_label) + 2}}latent ECMMD bandwidth = {ecmmd_metrics['bandwidth']:.4f}")
        else:
            if str(ecmmd_metrics.get("graph_mode")) == DEFAULT_CONDITIONAL_EVAL_MODE and "adaptive_radius" in ecmmd_metrics:
                adaptive = ecmmd_metrics["adaptive_radius"]
                single = adaptive["single_draw"]
                multi = adaptive["derandomized"]
                ci_suffix = ""
                if "bootstrap_ci_lower" in multi and "bootstrap_ci_upper" in multi:
                    ci_suffix = f", CI=[{multi['bootstrap_ci_lower']:.4e}, {multi['bootstrap_ci_upper']:.4e}]"
                boot_suffix = f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
                lines.append(
                    f"{'':>{len(pair_label) + 2}}latent ECMMD adaptive: "
                    f"single={single['score']:.4e}; D_n={multi['score']:.4e}{ci_suffix}{boot_suffix}"
                )
            else:
                for k_key, k_metrics in ecmmd_metrics.get("k_values", {}).items():
                    single = k_metrics["single_draw"]
                    multi = k_metrics["derandomized"]
                    single_boot = (
                        f", p_boot={single['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in single else ""
                    )
                    multi_boot = (
                        f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
                    )
                    lines.append(
                        f"{'':>{len(pair_label) + 2}}latent ECMMD K={k_metrics['k_effective']} "
                        f"(req={k_key}): single={single['score']:.4e}, z={single['z_score']:.3f}, p={single['p_value']:.3g}{single_boot}; "
                        f"D_n={multi['score']:.4e}, z={multi['z_score']:.3f}, p={multi['p_value']:.3g}{multi_boot}"
                    )
        field_metrics = pair_metrics.get("field_metrics")
        if isinstance(field_metrics, dict) and field_metrics:
            summary = field_metrics["summary"]
            lines.append(
                f"{'':>{len(pair_label) + 2}}field metrics: "
                f"W1={summary['mean_w1_normalized']:.4f}, "
                f"J={summary['mean_J_normalized']:.4f}, "
                f"corr_len={summary['mean_corr_length_relative_error']:.4f}"
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def run_csp_conditional_evaluation(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    load_csp_sampling_runtime_fn,
    load_corpus_latents_fn,
    load_fae_decode_context_fn,
    sample_conditional_batch_fn=sample_conditional_batch,
    sample_paired_prior_conditional_batch_fn=sample_paired_prior_conditional_batch,
    plot_conditioned_ecmmd_dashboard_fn,
    compute_chatterjee_local_scores_fn,
    compute_ecmmd_metrics_fn,
    add_bootstrap_ecmmd_calibration_fn,
) -> None:
    requested_phases = resolve_requested_phases(args)
    deferred_w_reason, deferred_ecmmd_reason = deferred_pairwise_reasons(
        phases_arg=getattr(args, "phases", None),
        skip_ecmmd=bool(getattr(args, "skip_ecmmd", False)),
        latent_metrics_hint="latent_metrics",
    )
    conditional_eval_mode = _resolve_conditional_eval_mode(args)
    adaptive_config = _build_adaptive_reference_config(args)
    ecmmd_k_values = parse_positive_int_list_arg(args.ecmmd_k_values)
    ecmmd_k_values_raw = str(getattr(args, "ecmmd_k_values", "")).strip()
    if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE and not ecmmd_k_values:
        raise ValueError("--ecmmd_k_values must contain at least one positive integer.")
    if (
        conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE
        and ecmmd_k_values
        and ecmmd_k_values_raw not in {"", "20", "10,20,30"}
    ):
        warnings.warn(
            "adaptive_radius ignores --ecmmd_k_values and uses an adaptive radius graph instead.",
            stacklevel=2,
        )

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(run_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_csp_sampling_runtime_fn(
        run_dir,
        latents_override=args.latents_path,
        fae_checkpoint_override=args.fae_checkpoint,
    )
    latents_path = runtime.source.latents_path
    latent_test = runtime.archive.latent_test
    zt = runtime.archive.zt
    time_indices = runtime.archive.time_indices
    tau_knots = runtime.tau_knots
    t_count, n_test, latent_dim = latent_test.shape
    if n_test <= 0:
        raise ValueError("No test latents available for CSP conditional evaluation.")

    corpus_latents_path = Path(args.corpus_latents_path).expanduser()
    if not corpus_latents_path.is_absolute():
        corpus_latents_path = (repo_root / corpus_latents_path).resolve()
    corpus_latents_by_tidx, n_corpus = load_corpus_latents_fn(corpus_latents_path, time_indices)
    condition_mode = (
        runtime.condition_mode
        if runtime.model_type in {"conditional_bridge", "paired_prior_bridge"}
        else None
    )
    condition_num_intervals = int(t_count - 1) if condition_mode is not None else None
    full_H_schedule = build_full_H_schedule(args.H_meso_list, args.H_macro)
    sample_cache_manifest = _build_sample_cache_manifest(
        args=args,
        run_dir=run_dir,
        output_dir=output_dir,
        runtime=runtime,
        latents_path=latents_path,
        corpus_latents_path=corpus_latents_path,
        time_indices=time_indices,
        zt=zt,
        tau_knots=tau_knots,
        conditional_eval_mode=conditional_eval_mode,
        adaptive_config=adaptive_config,
        full_h_schedule=full_H_schedule,
        ecmmd_k_values=ecmmd_k_values,
    )
    existing_metrics, existing_manifest, existing_results = load_existing_conditional_eval_exports(output_dir)
    sample_cache, sample_metadata = _load_sample_metadata(
        output_dir=output_dir,
        requested_phases=requested_phases,
        sample_cache_manifest=sample_cache_manifest,
    )
    _print_evaluation_header(
        run_dir=run_dir,
        output_dir=output_dir,
        runtime=runtime,
        condition_mode=condition_mode,
        conditional_eval_mode=conditional_eval_mode,
        requested_phases=requested_phases,
        latents_path=latents_path,
        corpus_latents_path=corpus_latents_path,
        n_test_samples=int(min(args.n_test_samples, n_test)),
        n_ecmmd_conditions=int(min(args.n_test_samples, n_corpus)),
        args=args,
    )

    seed_policy = build_seed_policy(int(args.seed))
    np.random.seed(int(seed_policy["generation_seed"]))
    selection_rng = np.random.default_rng(int(seed_policy["condition_selection_seed"]))
    reference_rng = np.random.default_rng(int(seed_policy["reference_sampling_seed"]))
    pair_labels_metadata = [
        make_pair_label(
            tidx_coarse=int(time_indices[pair_idx + 1]),
            tidx_fine=int(time_indices[pair_idx]),
            full_H_schedule=full_H_schedule,
        )[0]
        for pair_idx in range(t_count - 1)
    ]
    saved_condition_set = None
    if sample_metadata is not None and "condition_set_id" in sample_metadata:
        saved_condition_set = condition_set_from_metadata_arrays(sample_metadata)
        ensure_condition_set_matches(saved_condition_set, expected_time_indices=time_indices, n_test=n_test)
    test_sample_indices = (
        np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64)
        if saved_condition_set is not None
        else _resolve_saved_indices(
            saved=existing_results,
            key="test_sample_indices",
            n_available=n_test,
        )
    )
    if test_sample_indices is None:
        test_sample_indices = selection_rng.choice(n_test, size=min(args.n_test_samples, n_test), replace=False)
        test_sample_indices.sort()
        test_sample_indices = test_sample_indices.astype(np.int64)
    condition_set = (
        saved_condition_set
        if saved_condition_set is not None
        else build_condition_set(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=time_indices,
            pair_labels=pair_labels_metadata,
        )
    )

    corpus_eval_indices = _resolve_saved_indices(
        saved=sample_metadata,
        key="corpus_eval_indices",
        n_available=n_corpus,
    )
    if corpus_eval_indices is None:
        corpus_eval_indices = _resolve_saved_indices(
            saved=existing_results,
            key="corpus_eval_indices",
            n_available=n_corpus,
        )
    if corpus_eval_indices is None:
        corpus_eval_indices = reference_rng.choice(n_corpus, size=min(args.n_test_samples, n_corpus), replace=False)
        corpus_eval_indices.sort()
        corpus_eval_indices = corpus_eval_indices.astype(np.int64)

    if sample_cache is not None and not sample_cache.has_chunk("metadata"):
        write_conditional_sample_metadata(
            sample_cache,
            metadata={
                **condition_set_to_metadata_arrays(condition_set),
                **seed_policy_to_metadata_arrays(seed_policy),
                "corpus_eval_indices": corpus_eval_indices.astype(np.int64),
            },
        )
    global_test_conditions = (
        np.asarray(latent_test[-1, test_sample_indices], dtype=np.float32)
        if condition_mode is not None and condition_mode != "previous_state"
        else None
    )
    corpus_global_latents = (
        np.asarray(corpus_latents_by_tidx[int(time_indices[-1])], dtype=np.float32)
        if condition_mode is not None and condition_mode != "previous_state"
        else None
    )
    corpus_global_conditions = (
        np.asarray(corpus_global_latents[corpus_eval_indices], dtype=np.float32)
        if corpus_global_latents is not None
        else None
    )
    decode_context = None
    if CONDITIONAL_PHASE_FIELD_METRICS in requested_phases:
        dataset_path = getattr(runtime.source, "dataset_path", None)
        fae_checkpoint_path = getattr(runtime.source, "fae_checkpoint_path", None)
        if dataset_path is None or fae_checkpoint_path is None:
            raise FileNotFoundError(
                "field_metrics requires a resolved dataset path and FAE checkpoint in the CSP run contract."
            )
        decode_context = load_fae_decode_context_fn(
            dataset_path=Path(dataset_path),
            fae_checkpoint_path=Path(fae_checkpoint_path),
            decode_mode="standard",
        )

    results_latent_w1_all: dict[str, np.ndarray] = {}
    results_latent_w2_all: dict[str, np.ndarray] = {}
    results_latent_w1_null_all: dict[str, np.ndarray] = {}
    results_latent_w2_null_all: dict[str, np.ndarray] = {}
    results_latent_w2_conditions_all: dict[str, np.ndarray] = {}
    results_latent_w2_generated_all: dict[str, np.ndarray] = {}
    results_latent_w2_support_indices_all: dict[str, np.ndarray] = {}
    results_latent_w2_support_weights_all: dict[str, np.ndarray] = {}
    results_latent_w2_support_counts_all: dict[str, np.ndarray] = {}
    results_ecmmd_latent_all: dict[str, dict[str, object]] = {}
    results_ecmmd_conditions_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_all: dict[str, np.ndarray] = {}
    results_ecmmd_observed_reference_all: dict[str, np.ndarray] = {}
    results_ecmmd_generated_all: dict[str, np.ndarray] = {}
    results_ecmmd_neighbor_indices_all: dict[str, np.ndarray] = {}
    results_ecmmd_neighbor_radii_all: dict[str, np.ndarray] = {}
    results_ecmmd_neighbor_distances_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_support_indices_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_support_weights_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_support_counts_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_radius_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_ess_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_mean_rse_all: dict[str, np.ndarray] = {}
    results_ecmmd_reference_eig_rse_all: dict[str, np.ndarray] = {}
    results_ecmmd_local_scores_all: dict[str, np.ndarray] = {}
    results_adaptive_ess_min_all: dict[str, int] = {}
    results_ecmmd_selected_rows_all: dict[str, np.ndarray] = {}
    results_ecmmd_selected_roles_all: dict[str, list[str]] = {}
    results_field_metrics_all: dict[str, dict[str, Any]] = {}
    pair_metadata_all: dict[str, dict[str, object]] = {}
    conditional_ecmmd_figures: dict[str, dict[str, object]] = {}
    field_metrics_figures: dict[str, dict[str, object]] = {}
    pair_labels: list[str] = []

    for pair_idx in range(t_count - 1):
        tidx_fine = int(time_indices[pair_idx])
        tidx_coarse = int(time_indices[pair_idx + 1])
        pair_label, h_coarse, h_fine, display_label = make_pair_label(
            tidx_coarse=tidx_coarse,
            tidx_fine=tidx_fine,
            full_H_schedule=full_H_schedule,
        )
        pair_labels.append(pair_label)
        pair_metadata_all[pair_label] = {
            "tidx_coarse": tidx_coarse,
            "tidx_fine": tidx_fine,
            "H_coarse": float(h_coarse),
            "H_fine": float(h_fine),
            "display_label": display_label,
            "modeled_marginal_coarse_order": int(pair_idx + 2),
            "modeled_marginal_fine_order": int(pair_idx + 1),
            "modeled_n_marginals": int(t_count),
        }

        print(f"\n{'=' * 60}", flush=True)
        print(
            f"Scale pair: {display_label}  "
            f"(modeled marginal {pair_idx + 2}/{t_count} -> {pair_idx + 1}/{t_count})",
            flush=True,
        )
        print(
            f"  dataset idx {tidx_coarse} -> {tidx_fine}  "
            f"(zt[{pair_idx + 1}]={zt[pair_idx + 1]:.4f} -> zt[{pair_idx}]={zt[pair_idx]:.4f})",
            flush=True,
        )
        pair_adaptive_config = _resolve_pair_adaptive_config(
            adaptive_config,
            n_conditions=int(corpus_latents_by_tidx[tidx_coarse].shape[0]),
            adaptive_ess_min_override=getattr(args, "adaptive_ess_min", None),
        )
        print(f"  adaptive_ess_min={pair_adaptive_config.ess_min}", flush=True)
        print(f"{'=' * 60}", flush=True)

        if CONDITIONAL_PHASE_SAMPLE in requested_phases:
            if has_conditional_pair_sample(sample_cache, pair_label=pair_label):
                pair_sample_payload = load_conditional_pair_sample_chunk(sample_cache, pair_label=pair_label)
            else:
                pair_sample_payload = _evaluate_scale_pair(
                    pair_idx=pair_idx,
                    latent_test=latent_test,
                    corpus_z_coarse=corpus_latents_by_tidx[tidx_coarse],
                    corpus_z_fine=corpus_latents_by_tidx[tidx_fine],
                    test_global_conditions=global_test_conditions,
                    corpus_global_conditions=corpus_global_conditions,
                    test_sample_indices=test_sample_indices,
                    corpus_eval_indices=corpus_eval_indices,
                    drift_net=runtime.model,
                    zt=zt,
                    tau_knots=tau_knots,
                    condition_mode=condition_mode,
                    condition_num_intervals=condition_num_intervals,
                    dt0=float(runtime.dt0),
                    sigma_fn=runtime.sigma_fn,
                    model_type=str(runtime.model_type),
                    delta_v=getattr(runtime, "delta_v", None),
                    theta_feature_clip=getattr(runtime, "theta_feature_clip", None),
                    conditional_eval_mode=conditional_eval_mode,
                    adaptive_config=adaptive_config,
                    adaptive_ess_min_override=getattr(args, "adaptive_ess_min", None),
                    k_neighbors=args.k_neighbors,
                    n_realizations=args.n_realizations,
                    ecmmd_k_values=ecmmd_k_values,
                    ecmmd_bootstrap_reps=args.ecmmd_bootstrap_reps,
                    n_plot_conditions=args.n_plot_conditions,
                    plot_value_budget=args.plot_value_budget,
                    base_seed=int(seed_policy["generation_seed"]) + pair_idx * 10_000,
                    rng=np.random.default_rng(int(seed_policy["reference_sampling_seed"]) + pair_idx * 10_000),
                    compute_w_metrics=False,
                    compute_ecmmd_metrics=False,
                    sample_conditional_batch_fn=sample_conditional_batch_fn,
                    sample_paired_prior_conditional_batch_fn=sample_paired_prior_conditional_batch_fn,
                )
                write_conditional_pair_sample(
                    sample_cache,
                    pair_label=pair_label,
                    pair_sample_payload=pair_sample_payload,
                    adaptive_ess_min=int(pair_sample_payload["adaptive_ess_min"]),
                )
        elif conditional_sample_cache_matches(
            output_dir=output_dir,
            manifest=sample_cache_manifest,
            require_complete=True,
        ):
            pair_sample_payload = load_conditional_pair_sample(
                output_dir=output_dir,
                pair_label=pair_label,
                manifest=sample_cache_manifest,
            )
        elif existing_results is not None:
            pair_sample_payload = load_saved_pair_sample_payload(existing_results, pair_label=pair_label)
            for optional_key in (
                "latent_w2_conditions",
                "latent_w2_generated",
                "latent_w2_reference_support_indices",
                "latent_w2_reference_support_weights",
                "latent_w2_reference_support_counts",
            ):
                result_key = f"{optional_key}_{pair_label}"
                if result_key in existing_results:
                    pair_sample_payload[optional_key] = np.asarray(existing_results[result_key])
            pair_sample_payload["adaptive_ess_min"] = int(
                np.asarray(existing_results.get(f"adaptive_ess_min_{pair_label}", pair_adaptive_config.ess_min)).item()
            )
        else:
            raise FileNotFoundError(
                "Missing reusable kNN reference cache. Run the reference-cache stage first: --phases reference_cache."
            )

        results_latent_w2_conditions_all[pair_label] = np.asarray(pair_sample_payload["latent_w2_conditions"], dtype=np.float32)
        results_latent_w2_generated_all[pair_label] = np.asarray(pair_sample_payload["latent_w2_generated"], dtype=np.float32)
        results_latent_w2_support_indices_all[pair_label] = np.asarray(
            pair_sample_payload["latent_w2_reference_support_indices"],
            dtype=np.int64,
        )
        results_latent_w2_support_weights_all[pair_label] = np.asarray(
            pair_sample_payload["latent_w2_reference_support_weights"],
            dtype=np.float32,
        )
        results_latent_w2_support_counts_all[pair_label] = np.asarray(
            pair_sample_payload["latent_w2_reference_support_counts"],
            dtype=np.int64,
        )
        results_ecmmd_conditions_all[pair_label] = np.asarray(pair_sample_payload["latent_ecmmd_conditions"], dtype=np.float32)
        results_ecmmd_reference_all[pair_label] = np.asarray(pair_sample_payload["latent_ecmmd_reference"], dtype=np.float32)
        results_ecmmd_observed_reference_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_observed_reference"],
            dtype=np.float32,
        )
        results_ecmmd_generated_all[pair_label] = np.asarray(pair_sample_payload["latent_ecmmd_generated"], dtype=np.float32)
        results_ecmmd_neighbor_indices_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_neighbor_indices"],
            dtype=np.int64,
        )
        results_ecmmd_neighbor_radii_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_neighbor_radii"],
            dtype=np.float32,
        )
        results_ecmmd_neighbor_distances_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_neighbor_distances"],
            dtype=np.float32,
        )
        results_ecmmd_reference_support_indices_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_reference_support_indices"],
            dtype=np.int64,
        )
        results_ecmmd_reference_support_weights_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_reference_support_weights"],
            dtype=np.float32,
        )
        results_ecmmd_reference_support_counts_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_reference_support_counts"],
            dtype=np.int64,
        )
        results_ecmmd_reference_radius_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_reference_radius"],
            dtype=np.float32,
        )
        results_ecmmd_reference_ess_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_reference_ess"],
            dtype=np.float32,
        )
        results_ecmmd_reference_mean_rse_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_reference_mean_rse"],
            dtype=np.float32,
        )
        results_ecmmd_reference_eig_rse_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_reference_eig_rse"],
            dtype=np.float32,
        )
        results_ecmmd_local_scores_all[pair_label] = np.asarray(
            pair_sample_payload["latent_ecmmd_local_scores"],
            dtype=np.float32,
        )
        results_adaptive_ess_min_all[pair_label] = int(pair_sample_payload["adaptive_ess_min"])

        if CONDITIONAL_PHASE_W2 in requested_phases:
            if all(key in pair_sample_payload for key in (
                "latent_w2_generated",
                "latent_w2_reference_support_indices",
                "latent_w2_reference_support_weights",
                "latent_w2_reference_support_counts",
            )):
                latent_w1_arr, latent_w2_arr, latent_w1_null_arr, latent_w2_null_arr = _compute_latent_w_metrics_from_payload(
                    pair_label=pair_label,
                    pair_sample_payload=pair_sample_payload,
                    corpus_z_fine=np.asarray(corpus_latents_by_tidx[tidx_fine], dtype=np.float32),
                    base_seed=int(seed_policy["reference_sampling_seed"]) + pair_idx * 10_000,
                )
            elif existing_results is not None and all(
                key in existing_results
                for key in (
                    f"latent_w1_{pair_label}",
                    f"latent_w2_{pair_label}",
                    f"latent_w1_null_{pair_label}",
                    f"latent_w2_null_{pair_label}",
                )
            ):
                latent_w1_arr = np.asarray(existing_results[f"latent_w1_{pair_label}"], dtype=np.float64)
                latent_w2_arr = np.asarray(existing_results[f"latent_w2_{pair_label}"], dtype=np.float64)
                latent_w1_null_arr = np.asarray(existing_results[f"latent_w1_null_{pair_label}"], dtype=np.float64)
                latent_w2_null_arr = np.asarray(existing_results[f"latent_w2_null_{pair_label}"], dtype=np.float64)
            else:
                raise FileNotFoundError(
                    f"Missing saved W1/W2 sample support for {pair_label}. Re-run with --phases reference_cache."
                )
            results_latent_w1_all[pair_label] = latent_w1_arr
            results_latent_w2_all[pair_label] = latent_w2_arr
            results_latent_w1_null_all[pair_label] = latent_w1_null_arr
            results_latent_w2_null_all[pair_label] = latent_w2_null_arr
        if CONDITIONAL_PHASE_ECMMD in requested_phases:
            ecmmd_result = _compute_latent_ecmmd_from_payload(
                pair_sample_payload=pair_sample_payload,
                conditional_eval_mode=conditional_eval_mode,
                ecmmd_k_values=ecmmd_k_values,
                ecmmd_bootstrap_reps=args.ecmmd_bootstrap_reps,
                pair_adaptive_config=pair_adaptive_config,
                corpus_z_coarse_flat=np.asarray(corpus_latents_by_tidx[tidx_coarse], dtype=np.float32),
                corpus_z_fine_flat=np.asarray(corpus_latents_by_tidx[tidx_fine], dtype=np.float32),
                base_seed=int(seed_policy["bootstrap_seed"]) + pair_idx * 10_000,
                compute_chatterjee_local_scores_fn=compute_chatterjee_local_scores_fn,
                compute_ecmmd_metrics_fn=compute_ecmmd_metrics_fn,
                add_bootstrap_ecmmd_calibration_fn=add_bootstrap_ecmmd_calibration_fn,
            )
            latent_ecmmd = dict(ecmmd_result["latent_ecmmd"])
            results_ecmmd_latent_all[pair_label] = latent_ecmmd
            results_ecmmd_local_scores_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_local_scores"],
                dtype=np.float32,
            )
            results_ecmmd_neighbor_indices_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_neighbor_indices"],
                dtype=np.int64,
            )
            results_ecmmd_neighbor_radii_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_neighbor_radii"],
                dtype=np.float32,
            )
            results_ecmmd_neighbor_distances_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_neighbor_distances"],
                dtype=np.float32,
            )
            results_ecmmd_reference_support_indices_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_reference_support_indices"],
                dtype=np.int64,
            )
            results_ecmmd_reference_support_weights_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_reference_support_weights"],
                dtype=np.float32,
            )
            results_ecmmd_reference_support_counts_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_reference_support_counts"],
                dtype=np.int64,
            )
            results_ecmmd_reference_radius_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_reference_radius"],
                dtype=np.float32,
            )
            results_ecmmd_reference_ess_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_reference_ess"],
                dtype=np.float32,
            )
            results_ecmmd_reference_mean_rse_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_reference_mean_rse"],
                dtype=np.float32,
            )
            results_ecmmd_reference_eig_rse_all[pair_label] = np.asarray(
                ecmmd_result["latent_ecmmd_reference_eig_rse"],
                dtype=np.float32,
            )
            if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE:
                ecmmd_vis = plot_conditioned_ecmmd_dashboard_fn(
                    pair_label=pair_label,
                    display_label=display_label,
                    conditions=results_ecmmd_conditions_all[pair_label],
                    observed_reference=results_ecmmd_observed_reference_all[pair_label],
                    generated_samples=results_ecmmd_generated_all[pair_label],
                    local_scores=results_ecmmd_local_scores_all[pair_label],
                    neighborhood_indices=results_ecmmd_neighbor_indices_all[pair_label],
                    neighborhood_radii=results_ecmmd_neighbor_radii_all[pair_label],
                    latent_ecmmd=latent_ecmmd,
                    output_stem=output_dir / f"fig_conditional_ecmmd_{pair_label}",
                    n_plot_conditions=int(args.n_plot_conditions),
                    seed=int(seed_policy["representative_selection_seed"] + pair_idx * 10_000),
                    condition_indices=np.asarray(corpus_eval_indices, dtype=np.int64),
                )
                results_ecmmd_selected_rows_all[pair_label] = np.asarray(
                    ecmmd_vis["selected_condition_rows"],
                    dtype=np.int64,
                )
                results_ecmmd_selected_roles_all[pair_label] = list(ecmmd_vis["selected_condition_roles"])
                conditional_ecmmd_figures[pair_label] = {
                    "overview": ecmmd_vis["overview_figure"],
                    "detail": ecmmd_vis["detail_figure"],
                    "selected_condition_rows": results_ecmmd_selected_rows_all[pair_label].astype(int).tolist(),
                    "selected_condition_roles": list(results_ecmmd_selected_roles_all[pair_label]),
                }
                print(
                    f"  Saved conditional ECMMD figures for {pair_label}: "
                    f"{ecmmd_vis['overview_figure'].get('png', '')}",
                    flush=True,
                )
            else:
                results_ecmmd_selected_rows_all[pair_label] = np.asarray([], dtype=np.int64)
                results_ecmmd_selected_roles_all[pair_label] = []
                conditional_ecmmd_figures[pair_label] = {
                    "skipped_reason": "publication figures are only generated for chatterjee_knn mode",
                }
        elif (
            existing_metrics is not None
            and isinstance(existing_metrics.get("scale_pairs"), dict)
            and pair_label in existing_metrics["scale_pairs"]
        ):
            results_ecmmd_latent_all[pair_label] = dict(existing_metrics["scale_pairs"][pair_label].get("latent_ecmmd", {}))
            results_ecmmd_selected_rows_all[pair_label] = (
                np.asarray(existing_results[f"latent_ecmmd_selected_rows_{pair_label}"], dtype=np.int64)
                if existing_results is not None and f"latent_ecmmd_selected_rows_{pair_label}" in existing_results
                else np.asarray([], dtype=np.int64)
            )
            results_ecmmd_selected_roles_all[pair_label] = (
                [str(value) for value in np.asarray(existing_results[f"latent_ecmmd_selected_roles_{pair_label}"]).tolist()]
                if existing_results is not None and f"latent_ecmmd_selected_roles_{pair_label}" in existing_results
                else []
            )
            if isinstance(existing_manifest, dict):
                reports_figures = existing_manifest.get("reports_figures", {})
                if not isinstance(reports_figures, dict):
                    reports_figures = existing_manifest.get("conditional_ecmmd_figures", {})
                conditional_ecmmd_figures[pair_label] = dict(reports_figures.get(pair_label, {}))
            else:
                conditional_ecmmd_figures[pair_label] = {}
        else:
            results_ecmmd_latent_all[pair_label] = deferred_ecmmd_metrics(deferred_ecmmd_reason)
            results_ecmmd_selected_rows_all[pair_label] = np.asarray([], dtype=np.int64)
            results_ecmmd_selected_roles_all[pair_label] = []
            conditional_ecmmd_figures[pair_label] = {
                "skipped_reason": deferred_ecmmd_reason,
                "reuse_ready": True,
            }

        if CONDITIONAL_PHASE_FIELD_METRICS in requested_phases:
            if decode_context is None:
                raise RuntimeError("field_metrics requested without a decode context.")
            field_metrics_payload, field_figure_manifest = _compute_field_metrics_from_payload(
                args=args,
                pair_label=pair_label,
                pair_metadata=pair_metadata_all[pair_label],
                pair_sample_payload=pair_sample_payload,
                corpus_z_fine=np.asarray(corpus_latents_by_tidx[tidx_fine], dtype=np.float32),
                test_sample_indices=test_sample_indices,
                reference_sampling_seed=int(seed_policy["reference_sampling_seed"]) + pair_idx * 10_000,
                representative_seed=int(seed_policy["representative_selection_seed"]) + pair_idx * 10_000,
                output_dir=output_dir,
                decode_context=decode_context,
            )
            results_field_metrics_all[pair_label] = field_metrics_payload
            field_metrics_figures[pair_label] = field_figure_manifest
        elif (
            existing_metrics is not None
            and isinstance(existing_metrics.get("scale_pairs"), dict)
            and pair_label in existing_metrics["scale_pairs"]
            and isinstance(existing_metrics["scale_pairs"][pair_label].get("field_metrics"), dict)
            and existing_metrics["scale_pairs"][pair_label].get("field_metrics")
        ):
            results_field_metrics_all[pair_label] = dict(existing_metrics["scale_pairs"][pair_label]["field_metrics"])
            if isinstance(existing_manifest, dict):
                existing_field_figures = existing_manifest.get("field_metrics_figures", {})
                if isinstance(existing_field_figures, dict):
                    field_metrics_figures[pair_label] = dict(existing_field_figures.get(pair_label, {}))

        if pair_label in results_latent_w2_all and pair_label in results_latent_w2_null_all:
            mean_w1_null = float(results_latent_w1_null_all[pair_label].mean())
            mean_w2_null = float(results_latent_w2_null_all[pair_label].mean())
            w1_skill = (
                1.0 - float(results_latent_w1_all[pair_label].mean()) / mean_w1_null
                if mean_w1_null > 0.0 else float("nan")
            )
            w2_skill = (
                1.0 - float(results_latent_w2_all[pair_label].mean()) / mean_w2_null
                if mean_w2_null > 0.0 else float("nan")
            )
            print(
                f"  Summary: latent W1 mean={results_latent_w1_all[pair_label].mean():.4f}, "
                f"latent W2 mean={results_latent_w2_all[pair_label].mean():.4f}, "
                f"W1 skill={w1_skill:+.4f}, W2 skill={w2_skill:+.4f}",
                flush=True,
            )
        else:
            print("  Latent W1/W2 deferred; reusable conditional sample cache saved.", flush=True)
        latent_ecmmd = results_ecmmd_latent_all[pair_label]
        if bool(latent_ecmmd.get("deferred")):
            print("  ECMMD deferred; reusable conditional sample cache saved.", flush=True)
        elif str(latent_ecmmd.get("graph_mode")) == DEFAULT_CONDITIONAL_EVAL_MODE and "adaptive_radius" in latent_ecmmd:
            adaptive = latent_ecmmd["adaptive_radius"]
            multi = adaptive["derandomized"]
            ci_suffix = ""
            if "bootstrap_ci_lower" in multi and "bootstrap_ci_upper" in multi:
                ci_suffix = f", CI=[{multi['bootstrap_ci_lower']:.4e}, {multi['bootstrap_ci_upper']:.4e}]"
            boot_suffix = f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
            print(
                f"  Adaptive ECMMD: single={adaptive['single_draw']['score']:.4e}; "
                f"D_n={multi['score']:.4e}{ci_suffix}{boot_suffix}",
                flush=True,
            )
        if pair_label in results_field_metrics_all:
            field_summary = results_field_metrics_all[pair_label]["summary"]
            print(
                "  Field summary: "
                f"W1={field_summary['mean_w1_normalized']:.4f}, "
                f"J={field_summary['mean_J_normalized']:.4f}, "
                f"corr_len={field_summary['mean_corr_length_relative_error']:.4f}",
                flush=True,
            )

    metrics: dict[str, object] = {
        "model_family": "csp",
        "model_type": runtime.model_type,
        "condition_mode": condition_mode,
        "conditional_eval_mode": conditional_eval_mode,
        "requested_stages": requested_phases,
        "source_latents_path": str(latents_path),
        "corpus_latents_path": str(corpus_latents_path),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "skip_ecmmd": bool(getattr(args, "skip_ecmmd", False)),
        "adaptive_metric_dim_cap": int(adaptive_config.metric_dim_cap),
        "adaptive_reference_bootstrap_reps": int(adaptive_config.bootstrap_reps),
        "adaptive_ess_min": (
            int(args.adaptive_ess_min) if getattr(args, "adaptive_ess_min", None) is not None else None
        ),
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(corpus_eval_indices)),
        "n_realizations": int(args.n_realizations),
        "n_corpus": int(n_corpus),
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "time_indices": time_indices.tolist(),
        "zt": zt.astype(float).tolist(),
        "tau_knots": tau_knots.astype(float).tolist(),
        "full_H_schedule": list(map(float, full_H_schedule)),
        "sample_cache_ready": True,
        "scale_pairs": {},
    }
    npz_dict: dict[str, object] = {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "corpus_eval_indices": corpus_eval_indices.astype(np.int64),
        "time_indices": time_indices.astype(np.int64),
        "zt": zt.astype(np.float32),
        "tau_knots": tau_knots.astype(np.float32),
        "ecmmd_k_values_requested": np.asarray(ecmmd_k_values, dtype=np.int64),
        "skip_ecmmd": np.asarray(bool(getattr(args, "skip_ecmmd", False)), dtype=np.bool_),
        "pair_labels": np.asarray(pair_labels, dtype=object),
        "condition_set_id": np.asarray(str(condition_set["condition_set_id"])),
    }

    for pair_label in pair_labels:
        w_metrics_ready = (
            pair_label in results_latent_w1_all
            and pair_label in results_latent_w2_all
            and pair_label in results_latent_w1_null_all
            and pair_label in results_latent_w2_null_all
        )
        field_metrics_payload = results_field_metrics_all.get(pair_label)
        pair_metrics = {
            "pair_metadata": pair_metadata_all[pair_label],
            "adaptive_ess_min": int(results_adaptive_ess_min_all[pair_label]),
            "latent_w1": (
                metric_summary(results_latent_w1_all[pair_label])
                if w_metrics_ready
                else deferred_w_metrics(deferred_w_reason)
            ),
            "latent_w2": (
                metric_summary(results_latent_w2_all[pair_label])
                if w_metrics_ready
                else deferred_w_metrics(deferred_w_reason)
            ),
            "latent_w1_null": (
                metric_summary(results_latent_w1_null_all[pair_label])
                if w_metrics_ready
                else deferred_w_metrics(deferred_w_reason)
            ),
            "latent_w2_null": (
                metric_summary(results_latent_w2_null_all[pair_label])
                if w_metrics_ready
                else deferred_w_metrics(deferred_w_reason)
            ),
            "latent_w1_skill_vs_null": (
                (
                    1.0 - float(results_latent_w1_all[pair_label].mean()) / float(results_latent_w1_null_all[pair_label].mean())
                )
                if w_metrics_ready and float(results_latent_w1_null_all[pair_label].mean()) > 0.0
                else None
            ),
            "latent_w2_skill_vs_null": (
                (
                    1.0 - float(results_latent_w2_all[pair_label].mean()) / float(results_latent_w2_null_all[pair_label].mean())
                )
                if w_metrics_ready and float(results_latent_w2_null_all[pair_label].mean()) > 0.0
                else None
            ),
            "latent_ecmmd": results_ecmmd_latent_all[pair_label],
        }
        if field_metrics_payload is not None:
            pair_metrics["field_metrics"] = field_metrics_payload
        metrics["scale_pairs"][pair_label] = pair_metrics

        if w_metrics_ready:
            npz_dict[f"latent_w1_{pair_label}"] = results_latent_w1_all[pair_label].astype(np.float32)
            npz_dict[f"latent_w2_{pair_label}"] = results_latent_w2_all[pair_label].astype(np.float32)
            npz_dict[f"latent_w1_null_{pair_label}"] = results_latent_w1_null_all[pair_label].astype(np.float32)
            npz_dict[f"latent_w2_null_{pair_label}"] = results_latent_w2_null_all[pair_label].astype(np.float32)
            npz_dict[f"latent_w1_skill_vs_null_{pair_label}"] = np.float32(pair_metrics["latent_w1_skill_vs_null"])
            npz_dict[f"latent_w2_skill_vs_null_{pair_label}"] = np.float32(pair_metrics["latent_w2_skill_vs_null"])
        if field_metrics_payload is not None:
            per_condition_rows = list(field_metrics_payload.get("per_condition", []))
            npz_dict[f"field_w1_normalized_{pair_label}"] = np.asarray(
                [row["w1_normalized"] for row in per_condition_rows],
                dtype=np.float32,
            )
            npz_dict[f"field_J_normalized_{pair_label}"] = np.asarray(
                [row["J_normalized"] for row in per_condition_rows],
                dtype=np.float32,
            )
            npz_dict[f"field_corr_length_relative_error_{pair_label}"] = np.asarray(
                [row["corr_length_relative_error"] for row in per_condition_rows],
                dtype=np.float32,
            )
            npz_dict[f"field_selected_rows_{pair_label}"] = np.asarray(
                field_metrics_payload.get("selected_condition_rows", []),
                dtype=np.int64,
            )
            npz_dict[f"field_selected_roles_{pair_label}"] = np.asarray(
                field_metrics_payload.get("selected_condition_roles", []),
                dtype=np.str_,
            )
        npz_dict[f"latent_w2_conditions_{pair_label}"] = results_latent_w2_conditions_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w2_generated_{pair_label}"] = results_latent_w2_generated_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w2_reference_support_indices_{pair_label}"] = results_latent_w2_support_indices_all[pair_label].astype(np.int64)
        npz_dict[f"latent_w2_reference_support_weights_{pair_label}"] = results_latent_w2_support_weights_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w2_reference_support_counts_{pair_label}"] = results_latent_w2_support_counts_all[pair_label].astype(np.int64)
        npz_dict[f"latent_ecmmd_conditions_{pair_label}"] = np.asarray(
            results_ecmmd_conditions_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_reference_{pair_label}"] = np.asarray(
            results_ecmmd_reference_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_observed_reference_{pair_label}"] = np.asarray(
            results_ecmmd_observed_reference_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_generated_{pair_label}"] = np.asarray(
            results_ecmmd_generated_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_neighbor_indices_{pair_label}"] = np.asarray(
            results_ecmmd_neighbor_indices_all[pair_label],
            dtype=np.int64,
        )
        npz_dict[f"latent_ecmmd_neighbor_radii_{pair_label}"] = np.asarray(
            results_ecmmd_neighbor_radii_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_neighbor_distances_{pair_label}"] = np.asarray(
            results_ecmmd_neighbor_distances_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_reference_support_indices_{pair_label}"] = np.asarray(
            results_ecmmd_reference_support_indices_all[pair_label],
            dtype=np.int64,
        )
        npz_dict[f"latent_ecmmd_reference_support_weights_{pair_label}"] = np.asarray(
            results_ecmmd_reference_support_weights_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_reference_support_counts_{pair_label}"] = np.asarray(
            results_ecmmd_reference_support_counts_all[pair_label],
            dtype=np.int64,
        )
        npz_dict[f"latent_ecmmd_reference_radius_{pair_label}"] = np.asarray(
            results_ecmmd_reference_radius_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_reference_ess_{pair_label}"] = np.asarray(
            results_ecmmd_reference_ess_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_reference_mean_rse_{pair_label}"] = np.asarray(
            results_ecmmd_reference_mean_rse_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_reference_eig_rse_{pair_label}"] = np.asarray(
            results_ecmmd_reference_eig_rse_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_local_scores_{pair_label}"] = np.asarray(
            results_ecmmd_local_scores_all[pair_label],
            dtype=np.float32,
        )
        npz_dict[f"latent_ecmmd_selected_rows_{pair_label}"] = np.asarray(
            results_ecmmd_selected_rows_all[pair_label],
            dtype=np.int64,
        )
        npz_dict[f"latent_ecmmd_selected_roles_{pair_label}"] = np.asarray(
            results_ecmmd_selected_roles_all[pair_label],
            dtype=np.str_,
        )

        pair_ecmmd = results_ecmmd_latent_all[pair_label]
        if "bandwidth" in pair_ecmmd:
            npz_dict[f"latent_ecmmd_bandwidth_{pair_label}"] = np.float32(pair_ecmmd["bandwidth"])
        if str(pair_ecmmd.get("graph_mode")) == DEFAULT_CONDITIONAL_EVAL_MODE and "adaptive_radius" in pair_ecmmd:
            adaptive = pair_ecmmd["adaptive_radius"]
            single = adaptive["single_draw"]
            multi = adaptive["derandomized"]
            npz_dict[f"latent_ecmmd_adaptive_single_score_{pair_label}"] = np.float32(single["score"])
            npz_dict[f"latent_ecmmd_adaptive_derand_score_{pair_label}"] = np.float32(multi["score"])
            if "bootstrap_p_value" in single:
                npz_dict[f"latent_ecmmd_adaptive_single_boot_p_{pair_label}"] = np.float32(single["bootstrap_p_value"])
                npz_dict[f"latent_ecmmd_adaptive_single_boot_z_{pair_label}"] = np.float32(single["bootstrap_z_score"])
            if "bootstrap_p_value" in multi:
                npz_dict[f"latent_ecmmd_adaptive_derand_boot_p_{pair_label}"] = np.float32(multi["bootstrap_p_value"])
                npz_dict[f"latent_ecmmd_adaptive_derand_boot_z_{pair_label}"] = np.float32(multi["bootstrap_z_score"])
            if "bootstrap_ci_lower" in multi and "bootstrap_ci_upper" in multi:
                npz_dict[f"latent_ecmmd_adaptive_derand_ci_lower_{pair_label}"] = np.float32(
                    multi["bootstrap_ci_lower"]
                )
                npz_dict[f"latent_ecmmd_adaptive_derand_ci_upper_{pair_label}"] = np.float32(
                    multi["bootstrap_ci_upper"]
                )
        for k_key, k_metrics in pair_ecmmd.get("k_values", {}).items():
            suffix = f"{pair_label}_k{k_key}"
            single = k_metrics["single_draw"]
            multi = k_metrics["derandomized"]
            npz_dict[f"latent_ecmmd_single_score_{suffix}"] = np.float32(single["score"])
            npz_dict[f"latent_ecmmd_single_z_{suffix}"] = np.float32(single["z_score"])
            npz_dict[f"latent_ecmmd_single_p_{suffix}"] = np.float32(single["p_value"])
            if "bootstrap_p_value" in single:
                npz_dict[f"latent_ecmmd_single_boot_p_{suffix}"] = np.float32(single["bootstrap_p_value"])
                npz_dict[f"latent_ecmmd_single_boot_z_{suffix}"] = np.float32(single["bootstrap_z_score"])
            npz_dict[f"latent_ecmmd_derand_score_{suffix}"] = np.float32(multi["score"])
            npz_dict[f"latent_ecmmd_derand_z_{suffix}"] = np.float32(multi["z_score"])
            npz_dict[f"latent_ecmmd_derand_p_{suffix}"] = np.float32(multi["p_value"])
            if "bootstrap_p_value" in multi:
                npz_dict[f"latent_ecmmd_derand_boot_p_{suffix}"] = np.float32(multi["bootstrap_p_value"])
                npz_dict[f"latent_ecmmd_derand_boot_z_{suffix}"] = np.float32(multi["bootstrap_z_score"])

    if sample_cache is not None:
        sample_cache.mark_complete(
            status_updates={
                "n_pairs": int(len(pair_labels)),
                "n_test_samples": int(len(test_sample_indices)),
                "n_ecmmd_conditions": int(len(corpus_eval_indices)),
            }
        )

    if CONDITIONAL_PHASE_ECMMD in requested_phases:
        for pair_label in pair_labels:
            pair_metrics = metrics["scale_pairs"].get(pair_label, {})
            field_metrics_payload = pair_metrics.get("field_metrics")
            if not isinstance(field_metrics_payload, dict) or not field_metrics_payload:
                continue
            table_rows = selected_field_metric_rows(field_metrics_payload)
            if not table_rows:
                continue
            table_path = output_dir / f"field_metrics_table_{pair_label}.txt"
            table_path.write_text(
                build_field_metric_table_text(
                    pair_label=pair_label,
                    pair_display_label=str(pair_metrics["pair_metadata"]["display_label"]),
                    per_condition_rows=table_rows,
                )
            )
            field_metrics_figures.setdefault(pair_label, {})["table"] = str(table_path)

    summary_text = _build_summary_text(
        args=args,
        conditional_eval_mode=conditional_eval_mode,
        adaptive_config=adaptive_config,
        test_sample_indices=test_sample_indices,
        corpus_eval_indices=corpus_eval_indices,
        n_corpus=n_corpus,
        pair_labels=pair_labels,
        metrics=metrics,
    )
    print(f"\n{summary_text}", flush=True)

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "model_type": runtime.model_type,
        "condition_mode": condition_mode,
        "conditional_eval_mode": conditional_eval_mode,
        "requested_stages": requested_phases,
        "completed_stages": completed_conditional_phases(
            pair_labels=pair_labels,
            latent_w2_all=results_latent_w2_all,
            ecmmd_latent_all=results_ecmmd_latent_all,
            field_metrics_ready=all(label in results_field_metrics_all for label in pair_labels),
        ),
        "source_latents_path": str(latents_path),
        "corpus_latents_path": str(corpus_latents_path),
        "condition_set_id": str(condition_set["condition_set_id"]),
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(corpus_eval_indices)),
        "n_realizations": int(args.n_realizations),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "skip_ecmmd": bool(getattr(args, "skip_ecmmd", False)),
        "adaptive_metric_dim_cap": int(adaptive_config.metric_dim_cap),
        "adaptive_reference_bootstrap_reps": int(adaptive_config.bootstrap_reps),
        "adaptive_ess_min": (
            int(args.adaptive_ess_min) if getattr(args, "adaptive_ess_min", None) is not None else None
        ),
        "seed": int(args.seed),
        "n_plot_conditions": int(max(0, args.n_plot_conditions)),
        "plot_value_budget": int(args.plot_value_budget),
        "sample_cache_ready": True,
        "field_metrics_figures": field_metrics_figures,
        "reports_figures": conditional_ecmmd_figures,
    }
    write_conditional_eval_artifacts(
        output_dir,
        metrics=metrics,
        npz_payload=npz_dict,
        summary_text=summary_text,
        manifest=manifest,
    )
    print(f"\nAll CSP knn_reference results saved to {output_dir}/", flush=True)
