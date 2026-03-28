from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Allow `--nogpu` to force JAX onto CPU before importing JAX.
if "--nogpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.csp.conditional_ecmmd_plots import plot_conditioned_ecmmd_dashboard
from csp import integrate_interval, sample_conditional_batch
from scripts.csp.run_context import load_corpus_latents, load_csp_sampling_runtime
from scripts.fae.tran_evaluation.conditional_metrics import (
    add_bootstrap_ecmmd_calibration,
    compute_chatterjee_local_scores,
    compute_ecmmd_metrics,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSP latent conditional evaluation via latent W1/W2 and latent ECMMD.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <run_dir>/eval/conditional/latent.",
    )
    parser.add_argument(
        "--corpus_latents_path",
        type=str,
        default="data/corpus_latents_ntk_prior.npz",
        help="Corpus latent codes npz with aligned latents_<time_idx> arrays.",
    )
    parser.add_argument("--latents_path", type=str, default=None, help="Optional source latent archive override.")
    parser.add_argument(
        "--fae_checkpoint",
        type=str,
        default=None,
        help="Optional FAE checkpoint override for run-contract reconstruction.",
    )
    parser.add_argument(
        "--conditional_eval_mode",
        type=str,
        default=CHATTERJEE_CONDITIONAL_EVAL_MODE,
        choices=[CHATTERJEE_CONDITIONAL_EVAL_MODE, DEFAULT_CONDITIONAL_EVAL_MODE],
        help="Use chatterjee_knn as the canonical conditional-law evaluation; adaptive_radius remains available for legacy comparison.",
    )
    parser.add_argument("--k_neighbors", type=int, default=200)
    parser.add_argument("--n_test_samples", type=int, default=50)
    parser.add_argument("--n_realizations", type=int, default=200)
    parser.add_argument("--n_plot_conditions", type=int, default=5)
    parser.add_argument(
        "--plot_value_budget",
        type=int,
        default=20_000,
        help="Maximum pooled scalar coordinate values retained per plotted condition.",
    )
    parser.add_argument("--ecmmd_k_values", type=str, default="20")
    parser.add_argument("--ecmmd_bootstrap_reps", type=int, default=64)
    parser.add_argument("--adaptive_metric_dim_cap", type=int, default=24)
    parser.add_argument("--adaptive_reference_bootstrap_reps", type=int, default=64)
    parser.add_argument("--adaptive_ess_min", type=int, default=32)
    parser.add_argument(
        "--H_meso_list",
        type=str,
        default="1.0,1.25,1.5,2.0,2.5,3.0",
        help="Comma-separated mesoscale H values used for human-readable pair labels.",
    )
    parser.add_argument("--H_macro", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")
    return parser.parse_args()


def _default_output_dir(run_dir: Path) -> Path:
    return run_dir / "eval" / "conditional" / "latent"


def _resolve_conditional_eval_mode(args: argparse.Namespace) -> str:
    return validate_conditional_eval_mode(
        getattr(args, "conditional_eval_mode", CHATTERJEE_CONDITIONAL_EVAL_MODE)
    )


def _build_adaptive_reference_config(args: argparse.Namespace) -> AdaptiveReferenceConfig:
    return AdaptiveReferenceConfig(
        metric_dim_cap=int(getattr(args, "adaptive_metric_dim_cap", 24)),
        bootstrap_reps=int(getattr(args, "adaptive_reference_bootstrap_reps", 64)),
        ess_min=int(getattr(args, "adaptive_ess_min", 32)),
    )


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
    condition_mode: str | None = None,
    global_conditions: np.ndarray | None = None,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
) -> np.ndarray:
    coarse_np = np.asarray(coarse_conditions, dtype=np.float32)
    if coarse_np.ndim != 2:
        raise ValueError(f"coarse_conditions must have shape (n_conditions, latent_dim), got {coarse_np.shape}.")
    if int(n_realizations) <= 0:
        raise ValueError("n_realizations must be positive.")

    repeated = np.repeat(coarse_np, int(n_realizations), axis=0)
    if condition_mode is None:
        if tau_start is None or tau_end is None:
            raise ValueError("tau_start and tau_end are required for legacy unconditional interval sampling.")
        keys = jax.random.split(jax.random.PRNGKey(int(seed)), repeated.shape[0])
        generated = _sample_interval_batch(
            drift_net,
            jnp.asarray(repeated, dtype=jnp.float32),
            jnp.asarray(float(tau_start), dtype=jnp.float32),
            jnp.asarray(float(tau_end), dtype=jnp.float32),
            jnp.asarray(float(dt0), dtype=jnp.float32),
            keys,
            sigma_fn,
        )
        return np.asarray(generated, dtype=np.float32).reshape(coarse_np.shape[0], int(n_realizations), coarse_np.shape[1])

    if zt is None:
        raise ValueError("zt is required for sequential conditional interval sampling.")
    repeated_global = repeated if global_conditions is None else np.repeat(
        np.asarray(global_conditions, dtype=np.float32),
        int(n_realizations),
        axis=0,
    )
    if repeated_global.shape != repeated.shape:
        raise ValueError(
            "global_conditions must match coarse_conditions in shape before realization expansion, "
            f"got {np.asarray(global_conditions).shape if global_conditions is not None else None} "
            f"and {coarse_np.shape}."
        )
    traj = sample_conditional_batch(
        drift_net,
        jnp.asarray(repeated, dtype=jnp.float32),
        jnp.asarray(np.asarray(zt, dtype=np.float32), dtype=jnp.float32),
        sigma_fn,
        float(dt0),
        jax.random.PRNGKey(int(seed)),
        condition_mode=str(condition_mode),
        global_condition_batch=jnp.asarray(repeated_global, dtype=jnp.float32),
        condition_num_intervals=condition_num_intervals,
        interval_offset=int(interval_offset),
    )
    generated = np.asarray(traj[:, 0, :], dtype=np.float32)
    return generated.reshape(coarse_np.shape[0], int(n_realizations), coarse_np.shape[1])


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
    conditional_eval_mode: str,
    adaptive_config: AdaptiveReferenceConfig,
    k_neighbors: int,
    n_realizations: int,
    ecmmd_k_values: list[int],
    ecmmd_bootstrap_reps: int,
    n_plot_conditions: int,
    plot_value_budget: int,
    base_seed: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    del n_plot_conditions, plot_value_budget
    test_conditions = np.asarray(latent_test[pair_idx + 1, test_sample_indices], dtype=np.float32)
    condition_metric = None
    corpus_z_coarse_metric = None
    if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE:
        condition_metric = fit_whitened_pca_metric(
            corpus_z_coarse,
            variance_retained=float(adaptive_config.variance_retained),
            dim_cap=int(adaptive_config.metric_dim_cap),
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
        condition_mode=condition_mode,
        global_conditions=test_global_conditions,
        condition_num_intervals=condition_num_intervals,
        interval_offset=interval_offset,
    )

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
            adaptive_config=adaptive_config,
            rng=rng,
        )
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
        condition_mode=condition_mode,
        global_conditions=corpus_global_conditions,
        condition_num_intervals=condition_num_intervals,
        interval_offset=interval_offset,
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
        chatterjee_payload = compute_chatterjee_local_scores(
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
            adaptive_config=adaptive_config,
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

    latent_ecmmd = compute_ecmmd_metrics(
        ecmmd_conditions,
        ecmmd_real_samples,
        ecmmd_generated,
        ecmmd_k_values,
        reference_weights=ecmmd_reference_weights,
        condition_graph_mode=conditional_eval_mode,
        graph_condition_vectors=graph_condition_vectors,
        adaptive_radii=adaptive_radii if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE else None,
    )
    latent_ecmmd = add_bootstrap_ecmmd_calibration(
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
        "latent_w1_values": np.asarray(latent_w1_values, dtype=np.float64),
        "latent_w2_values": np.asarray(latent_w2_values, dtype=np.float64),
        "latent_w1_null_values": np.asarray(latent_w1_null_values, dtype=np.float64),
        "latent_w2_null_values": np.asarray(latent_w2_null_values, dtype=np.float64),
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
        "CSP Latent Conditional Evaluation",
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
        f"adaptive_ess_min: {adaptive_config.ess_min}",
        "",
    ]

    scale_pairs = metrics["scale_pairs"]
    for pair_label in pair_labels:
        pair_metrics = scale_pairs[pair_label]
        pair_meta = pair_metrics["pair_metadata"]
        w1 = pair_metrics["latent_w1"]
        w2 = pair_metrics["latent_w2"]
        w1_null = pair_metrics["latent_w1_null"]
        w2_null = pair_metrics["latent_w2_null"]
        lines.append(
            f"{pair_label}: {pair_meta['display_label']} "
            f"(modeled marginal {pair_meta['modeled_marginal_coarse_order']}/{pair_meta['modeled_n_marginals']} "
            f"-> {pair_meta['modeled_marginal_fine_order']}/{pair_meta['modeled_n_marginals']}, "
            f"dataset idx {pair_meta['tidx_coarse']} -> {pair_meta['tidx_fine']})"
        )
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
        if "bandwidth" in ecmmd_metrics:
            lines.append(f"{'':>{len(pair_label) + 2}}latent ECMMD bandwidth = {ecmmd_metrics['bandwidth']:.4f}")
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
                single_boot = f", p_boot={single['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in single else ""
                multi_boot = f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
                lines.append(
                    f"{'':>{len(pair_label) + 2}}latent ECMMD K={k_metrics['k_effective']} "
                    f"(req={k_key}): single={single['score']:.4e}, z={single['z_score']:.3f}, p={single['p_value']:.3g}{single_boot}; "
                    f"D_n={multi['score']:.4e}, z={multi['z_score']:.3f}, p={multi['p_value']:.3g}{multi_boot}"
                )
        lines.append("")

    return "\n".join(lines).rstrip()


def main() -> None:
    args = _parse_args()
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
        else _default_output_dir(run_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_csp_sampling_runtime(
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
        corpus_latents_path = (_REPO_ROOT / corpus_latents_path).resolve()
    corpus_latents_by_tidx, n_corpus = load_corpus_latents(corpus_latents_path, time_indices)
    condition_mode = runtime.condition_mode if runtime.model_type == "conditional_bridge" else None
    condition_num_intervals = int(t_count - 1) if condition_mode is not None else None

    print("============================================================", flush=True)
    print("CSP latent conditional evaluation", flush=True)
    print(f"  run_dir            : {run_dir}", flush=True)
    print(f"  output_dir         : {output_dir}", flush=True)
    print(f"  model_type         : {runtime.model_type}", flush=True)
    print(f"  condition_mode     : {condition_mode}", flush=True)
    print(f"  conditional_eval_mode : {conditional_eval_mode}", flush=True)
    print(f"  source_latents     : {latents_path}", flush=True)
    print(f"  corpus_latents     : {corpus_latents_path}", flush=True)
    print(f"  n_test_samples     : {min(args.n_test_samples, n_test)}", flush=True)
    print(f"  n_ecmmd_conditions : {min(args.n_test_samples, n_corpus)}", flush=True)
    print(f"  n_realizations     : {args.n_realizations}", flush=True)
    print(f"  k_neighbors        : {args.k_neighbors}", flush=True)
    print(f"  n_plot_conditions  : {max(0, min(args.n_plot_conditions, min(args.n_test_samples, n_test)))}", flush=True)
    print(f"  plot_value_budget  : {args.plot_value_budget}", flush=True)
    print("============================================================", flush=True)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    test_sample_indices = rng.choice(n_test, size=min(args.n_test_samples, n_test), replace=False)
    test_sample_indices.sort()
    corpus_eval_indices = rng.choice(n_corpus, size=min(args.n_test_samples, n_corpus), replace=False)
    corpus_eval_indices.sort()
    full_H_schedule = build_full_H_schedule(args.H_meso_list, args.H_macro)
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

    results_latent_w1_all: dict[str, np.ndarray] = {}
    results_latent_w2_all: dict[str, np.ndarray] = {}
    results_latent_w1_null_all: dict[str, np.ndarray] = {}
    results_latent_w2_null_all: dict[str, np.ndarray] = {}
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
    results_ecmmd_selected_rows_all: dict[str, np.ndarray] = {}
    results_ecmmd_selected_roles_all: dict[str, list[str]] = {}
    pair_metadata_all: dict[str, dict[str, object]] = {}
    conditional_ecmmd_figures: dict[str, dict[str, object]] = {}
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
        print(f"{'=' * 60}", flush=True)

        pair_result = _evaluate_scale_pair(
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
            conditional_eval_mode=conditional_eval_mode,
            adaptive_config=adaptive_config,
            k_neighbors=args.k_neighbors,
            n_realizations=args.n_realizations,
            ecmmd_k_values=ecmmd_k_values,
            ecmmd_bootstrap_reps=args.ecmmd_bootstrap_reps,
            n_plot_conditions=args.n_plot_conditions,
            plot_value_budget=args.plot_value_budget,
            base_seed=args.seed + pair_idx * 10_000,
            rng=rng,
        )

        latent_w1_arr = pair_result["latent_w1_values"]
        latent_w2_arr = pair_result["latent_w2_values"]
        latent_w1_null_arr = pair_result["latent_w1_null_values"]
        latent_w2_null_arr = pair_result["latent_w2_null_values"]
        latent_ecmmd = pair_result["latent_ecmmd"]

        results_latent_w1_all[pair_label] = latent_w1_arr
        results_latent_w2_all[pair_label] = latent_w2_arr
        results_latent_w1_null_all[pair_label] = latent_w1_null_arr
        results_latent_w2_null_all[pair_label] = latent_w2_null_arr
        results_ecmmd_latent_all[pair_label] = latent_ecmmd
        results_ecmmd_conditions_all[pair_label] = np.asarray(pair_result["latent_ecmmd_conditions"], dtype=np.float32)
        results_ecmmd_reference_all[pair_label] = np.asarray(pair_result["latent_ecmmd_reference"], dtype=np.float32)
        results_ecmmd_observed_reference_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_observed_reference"],
            dtype=np.float32,
        )
        results_ecmmd_generated_all[pair_label] = np.asarray(pair_result["latent_ecmmd_generated"], dtype=np.float32)
        results_ecmmd_neighbor_indices_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_neighbor_indices"],
            dtype=np.int64,
        )
        results_ecmmd_neighbor_radii_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_neighbor_radii"],
            dtype=np.float32,
        )
        results_ecmmd_neighbor_distances_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_neighbor_distances"],
            dtype=np.float32,
        )
        results_ecmmd_reference_support_indices_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_reference_support_indices"], dtype=np.int64
        )
        results_ecmmd_reference_support_weights_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_reference_support_weights"], dtype=np.float32
        )
        results_ecmmd_reference_support_counts_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_reference_support_counts"], dtype=np.int64
        )
        results_ecmmd_reference_radius_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_reference_radius"], dtype=np.float32
        )
        results_ecmmd_reference_ess_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_reference_ess"], dtype=np.float32
        )
        results_ecmmd_reference_mean_rse_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_reference_mean_rse"], dtype=np.float32
        )
        results_ecmmd_reference_eig_rse_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_reference_eig_rse"], dtype=np.float32
        )
        results_ecmmd_local_scores_all[pair_label] = np.asarray(
            pair_result["latent_ecmmd_local_scores"],
            dtype=np.float32,
        )
        if conditional_eval_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE:
            ecmmd_vis = plot_conditioned_ecmmd_dashboard(
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
                seed=int(args.seed + pair_idx * 10_000),
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

        mean_w1_null = float(latent_w1_null_arr.mean())
        mean_w2_null = float(latent_w2_null_arr.mean())
        w1_skill = 1.0 - float(latent_w1_arr.mean()) / mean_w1_null if mean_w1_null > 0.0 else float("nan")
        w2_skill = 1.0 - float(latent_w2_arr.mean()) / mean_w2_null if mean_w2_null > 0.0 else float("nan")
        print(
            f"  Summary: latent W1 mean={latent_w1_arr.mean():.4f}, "
            f"latent W2 mean={latent_w2_arr.mean():.4f}, "
            f"W1 skill={w1_skill:+.4f}, W2 skill={w2_skill:+.4f}",
            flush=True,
        )
        if str(latent_ecmmd.get("graph_mode")) == DEFAULT_CONDITIONAL_EVAL_MODE and "adaptive_radius" in latent_ecmmd:
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

    metrics: dict[str, object] = {
        "model_family": "csp",
        "model_type": runtime.model_type,
        "condition_mode": condition_mode,
        "conditional_eval_mode": conditional_eval_mode,
        "source_latents_path": str(latents_path),
        "corpus_latents_path": str(corpus_latents_path),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "adaptive_metric_dim_cap": int(adaptive_config.metric_dim_cap),
        "adaptive_reference_bootstrap_reps": int(adaptive_config.bootstrap_reps),
        "adaptive_ess_min": int(adaptive_config.ess_min),
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(corpus_eval_indices)),
        "n_realizations": int(args.n_realizations),
        "n_corpus": int(n_corpus),
        "time_indices": time_indices.tolist(),
        "zt": zt.astype(float).tolist(),
        "tau_knots": tau_knots.astype(float).tolist(),
        "full_H_schedule": list(map(float, full_H_schedule)),
        "scale_pairs": {},
    }
    npz_dict: dict[str, object] = {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "corpus_eval_indices": corpus_eval_indices.astype(np.int64),
        "time_indices": time_indices.astype(np.int64),
        "zt": zt.astype(np.float32),
        "tau_knots": tau_knots.astype(np.float32),
        "ecmmd_k_values_requested": np.asarray(ecmmd_k_values, dtype=np.int64),
        "pair_labels": np.asarray(pair_labels, dtype=object),
    }

    for pair_label in pair_labels:
        mean_w1_null = float(results_latent_w1_null_all[pair_label].mean())
        mean_w2_null = float(results_latent_w2_null_all[pair_label].mean())
        pair_metrics = {
            "pair_metadata": pair_metadata_all[pair_label],
            "latent_w1": metric_summary(results_latent_w1_all[pair_label]),
            "latent_w2": metric_summary(results_latent_w2_all[pair_label]),
            "latent_w1_null": metric_summary(results_latent_w1_null_all[pair_label]),
            "latent_w2_null": metric_summary(results_latent_w2_null_all[pair_label]),
            "latent_w1_skill_vs_null": (
                1.0 - float(results_latent_w1_all[pair_label].mean()) / mean_w1_null
                if mean_w1_null > 0.0 else float("nan")
            ),
            "latent_w2_skill_vs_null": (
                1.0 - float(results_latent_w2_all[pair_label].mean()) / mean_w2_null
                if mean_w2_null > 0.0 else float("nan")
            ),
            "latent_ecmmd": results_ecmmd_latent_all[pair_label],
        }
        metrics["scale_pairs"][pair_label] = pair_metrics

        npz_dict[f"latent_w1_{pair_label}"] = results_latent_w1_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w2_{pair_label}"] = results_latent_w2_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w1_null_{pair_label}"] = results_latent_w1_null_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w2_null_{pair_label}"] = results_latent_w2_null_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w1_skill_vs_null_{pair_label}"] = np.float32(pair_metrics["latent_w1_skill_vs_null"])
        npz_dict[f"latent_w2_skill_vs_null_{pair_label}"] = np.float32(pair_metrics["latent_w2_skill_vs_null"])
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

    metrics_path = output_dir / "conditional_latent_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output_dir / "conditional_latent_results.npz", **npz_dict)

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
    (output_dir / "conditional_latent_summary.txt").write_text(summary_text + "\n")

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "model_type": runtime.model_type,
        "condition_mode": condition_mode,
        "conditional_eval_mode": conditional_eval_mode,
        "source_latents_path": str(latents_path),
        "corpus_latents_path": str(corpus_latents_path),
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(corpus_eval_indices)),
        "n_realizations": int(args.n_realizations),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "adaptive_metric_dim_cap": int(adaptive_config.metric_dim_cap),
        "adaptive_reference_bootstrap_reps": int(adaptive_config.bootstrap_reps),
        "adaptive_ess_min": int(adaptive_config.ess_min),
        "seed": int(args.seed),
        "n_plot_conditions": int(max(0, args.n_plot_conditions)),
        "plot_value_budget": int(args.plot_value_budget),
        "conditional_pdf_figures": {},
        "conditional_ecmmd_figures": conditional_ecmmd_figures,
    }
    (output_dir / "conditional_latent_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nAll CSP conditional latent results saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
