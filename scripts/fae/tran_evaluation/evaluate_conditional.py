#!/usr/bin/env python
"""Latent-space conditional evaluation for MSBM backward generation.

This script evaluates conditional generation quality only in latent space.
For each consecutive MSBM scale pair ``(s, s-1)`` it:

1. Uses coarse latent test samples ``z_s`` as query conditions.
2. Uses the aligned fine latent at each evaluation vertex together with a
   standardized condition-space Chatterjee kNN graph to measure conditional-law
   mismatch, and retains graph-induced empirical-conditionals as support
   diagnostics.
3. Generates latent samples at the finer scale ``z_{s-1}`` with the backward
   MSBM policy.
4. Computes latent conditional Wasserstein-1 and Wasserstein-2 distances.
5. Computes latent ECMMD statistics (single-draw and derandomized) with
   studentized ``z``-scores and ``p``-values.

Field-space diagnostics are handled elsewhere in the Tran evaluation pipeline
and are intentionally not duplicated here.

Usage
-----
python scripts/fae/tran_evaluation/evaluate_conditional.py \\
    --run_dir results/2026-02-01T23-00-12-38 \\
    --corpus_latents_path data/corpus_latents.npz \\
    --k_neighbors 200 \\
    --n_test_samples 50 \\
    --n_realizations 200
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    sampling_spec_mean_rse,
    sampling_spec_radius,
    summarize_reference_sampling_specs,
    transform_condition_vectors,
    validate_conditional_eval_mode,
    wasserstein1_wasserstein2_latents,
)
from scripts.fae.tran_evaluation.conditional_metrics import (
    add_bootstrap_ecmmd_calibration as _add_bootstrap_ecmmd_calibration,
    compute_chatterjee_local_scores,
    compute_ecmmd_metrics as _compute_ecmmd_metrics,
    metric_summary as _metric_summary,
    parse_positive_int_list_arg as _parse_int_list_arg,
)
from scripts.fae.tran_evaluation.latent_msbm_runtime import (
    build_latent_msbm_agent,
    load_corpus_latents,
    load_policy_checkpoints,
    load_run_latents,
    sample_backward_one_interval,
)
from scripts.utils import get_device

from mmsfm.latent_msbm import LatentMSBMAgent

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latent conditional evaluation via latent W1/W2 and latent ECMMD.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <run_dir>/tran_evaluation/conditional_latent.",
    )
    parser.add_argument(
        "--corpus_latents_path",
        type=str,
        required=True,
        help="Corpus latent codes npz with aligned latents_<time_idx> arrays.",
    )
    parser.add_argument(
        "--conditional_eval_mode",
        type=str,
        default=CHATTERJEE_CONDITIONAL_EVAL_MODE,
        choices=[CHATTERJEE_CONDITIONAL_EVAL_MODE, DEFAULT_CONDITIONAL_EVAL_MODE],
        help="Use chatterjee_knn as the canonical conditional-law evaluation; adaptive_radius remains available for legacy comparison.",
    )
    parser.add_argument("--k_neighbors", type=int, default=200)
    parser.add_argument("--n_test_samples", type=int, default=50, help="Number of test conditions for latent W1/W2.")
    parser.add_argument(
        "--n_realizations",
        type=int,
        default=200,
        help="Backward-sampled latent realizations per condition.",
    )
    parser.add_argument(
        "--ecmmd_k_values",
        type=str,
        default="20",
        help="Comma-separated K values for latent-space ECMMD.",
    )
    parser.add_argument(
        "--ecmmd_bootstrap_reps",
        type=int,
        default=64,
        help="Number of local-empirical bootstrap replicates for ECMMD goodness-of-fit calibration.",
    )
    parser.add_argument("--adaptive_metric_dim_cap", type=int, default=24)
    parser.add_argument("--adaptive_reference_bootstrap_reps", type=int, default=64)
    parser.add_argument("--adaptive_ess_min", type=int, default=32)
    parser.add_argument(
        "--H_meso_list",
        type=str,
        default="1.0,1.25,1.5,2.0,2.5,3.0",
        help="Comma-separated mesoscale H values used for human-readable pair labels.",
    )
    parser.add_argument(
        "--H_macro",
        type=float,
        default=6.0,
        help="Macroscale H value used for human-readable pair labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    parser.add_argument("--drift_clip_norm", type=float, default=None)
    return parser.parse_args()


def _resolve_conditional_eval_mode(args: argparse.Namespace) -> str:
    return validate_conditional_eval_mode(getattr(args, "conditional_eval_mode", CHATTERJEE_CONDITIONAL_EVAL_MODE))


def _build_adaptive_reference_config(args: argparse.Namespace) -> AdaptiveReferenceConfig:
    return AdaptiveReferenceConfig(
        metric_dim_cap=int(getattr(args, "adaptive_metric_dim_cap", 24)),
        bootstrap_reps=int(getattr(args, "adaptive_reference_bootstrap_reps", 64)),
        ess_min=int(getattr(args, "adaptive_ess_min", 32)),
    )


def _evaluate_scale_pair(
    *,
    pair_idx: int,
    latent_test: np.ndarray,
    corpus_z_coarse: np.ndarray,
    corpus_z_fine: np.ndarray,
    test_sample_indices: np.ndarray,
    corpus_eval_indices: np.ndarray,
    agent: LatentMSBMAgent,
    device: str,
    conditional_eval_mode: str,
    adaptive_config: AdaptiveReferenceConfig,
    k_neighbors: int,
    n_realizations: int,
    ecmmd_k_values: list[int],
    ecmmd_bootstrap_reps: int,
    base_seed: int,
    drift_clip_norm: float | None,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Evaluate one consecutive latent scale pair."""
    condition_metric = None
    corpus_z_coarse_metric = None
    if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE:
        condition_metric = fit_whitened_pca_metric(
            corpus_z_coarse,
            variance_retained=float(adaptive_config.variance_retained),
            dim_cap=int(adaptive_config.metric_dim_cap),
        )
        corpus_z_coarse_metric = transform_condition_vectors(corpus_z_coarse, condition_metric)

    latent_w1_values: list[float] = []
    latent_w2_values: list[float] = []
    latent_w1_null_values: list[float] = []
    latent_w2_null_values: list[float] = []

    for sample_offset, test_idx in enumerate(test_sample_indices):
        z_test_coarse = latent_test[pair_idx + 1, int(test_idx)]

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
        support_weights = np.asarray(sampling_spec["candidate_weights"], dtype=np.float64)
        ref_latents = corpus_z_fine[support_idx]

        z_start = torch.from_numpy(z_test_coarse[None, :]).float().to(device)
        z_gen = sample_backward_one_interval(
            agent=agent,
            policy=agent.z_b,
            z_start=z_start,
            interval_idx=pair_idx,
            n_realizations=n_realizations,
            seed=base_seed + sample_offset * 1000,
            drift_clip_norm=drift_clip_norm,
        )
        z_gen_np = z_gen.cpu().numpy().astype(np.float32)

        latent_w1, latent_w2 = wasserstein1_wasserstein2_latents(
            z_gen_np,
            ref_latents,
            weights_a=None,
            weights_b=support_weights,
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
            weights_b=support_weights,
        )
        latent_w1_null_values.append(latent_w1_null)
        latent_w2_null_values.append(latent_w2_null)

        if (sample_offset + 1) % 10 == 0 or sample_offset == 0:
            print(
                f"  Test condition {sample_offset + 1}/{len(test_sample_indices)}: "
                f"latent W1={latent_w1:.4f}  latent W2={latent_w2:.4f}"
            )

    ecmmd_conditions = corpus_z_coarse[corpus_eval_indices].astype(np.float32)
    ecmmd_generated: list[np.ndarray] = []

    for corpus_offset, corpus_idx in enumerate(corpus_eval_indices):
        z_corpus_coarse = corpus_z_coarse[int(corpus_idx)]
        z_start = torch.from_numpy(z_corpus_coarse[None, :]).float().to(device)
        z_gen = sample_backward_one_interval(
            agent=agent,
            policy=agent.z_b,
            z_start=z_start,
            interval_idx=pair_idx,
            n_realizations=n_realizations,
            seed=base_seed + 100_000 + corpus_offset * 100,
            drift_clip_norm=drift_clip_norm,
        )
        ecmmd_generated.append(z_gen.cpu().numpy().astype(np.float32))

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
            np.stack(ecmmd_generated, axis=0),
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

    latent_ecmmd = _compute_ecmmd_metrics(
        ecmmd_conditions,
        ecmmd_real_samples,
        np.stack(ecmmd_generated, axis=0),
        ecmmd_k_values,
        reference_weights=ecmmd_reference_weights,
        condition_graph_mode=conditional_eval_mode,
        graph_condition_vectors=graph_condition_vectors,
        adaptive_radii=adaptive_radii if conditional_eval_mode == DEFAULT_CONDITIONAL_EVAL_MODE else None,
    )
    latent_ecmmd = _add_bootstrap_ecmmd_calibration(
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
        generated_samples=np.stack(ecmmd_generated, axis=0),
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
        "latent_ecmmd_generated": np.stack(ecmmd_generated, axis=0).astype(np.float32),
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
    """Format a compact text summary."""
    lines = [
        "Latent Conditional Evaluation",
        "=" * 50,
        f"conditional_eval_mode: {conditional_eval_mode}",
        f"k_neighbors: {args.k_neighbors}",
        f"n_test_samples: {len(test_sample_indices)}",
        f"n_ecmmd_conditions: {len(corpus_eval_indices)}",
        f"n_realizations: {args.n_realizations}",
        f"n_corpus: {n_corpus}",
        f"ecmmd_k_values_requested: {_parse_int_list_arg(args.ecmmd_k_values)}",
        f"ecmmd_bootstrap_reps: {args.ecmmd_bootstrap_reps}",
        f"adaptive_metric_dim_cap: {adaptive_config.metric_dim_cap}",
        f"adaptive_reference_bootstrap_reps: {adaptive_config.bootstrap_reps}",
        f"adaptive_ess_min: {adaptive_config.ess_min}",
        "pair_labels use physical H values; modeled marginal 1 is the first learned scale (H=1),",
        "and the last modeled marginal is the coarsest learned scale (H=6) for the default Tran ladder.",
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
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent ECMMD bandwidth = {ecmmd_metrics['bandwidth']:.4f}"
            )
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

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    conditional_eval_mode = _resolve_conditional_eval_mode(args)
    adaptive_config = _build_adaptive_reference_config(args)
    ecmmd_k_values = _parse_int_list_arg(args.ecmmd_k_values)
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

    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "tran_evaluation" / "conditional_latent"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.nogpu)
    print(f"Device: {device}")

    train_cfg, latent_train, latent_test, zt, time_indices = load_run_latents(run_dir)
    t_count, n_test, latent_dim = latent_test.shape
    n_train = latent_train.shape[1]
    if n_test <= 0:
        raise ValueError("No test latents available for conditional evaluation.")

    print(f"MSBM: T={t_count}, n_train={n_train}, n_test={n_test}, latent_dim={latent_dim}")
    print(f"  time_indices: {time_indices.tolist()}")
    print(f"  zt: {np.round(zt, 4).tolist()}")
    print(f"  conditional_eval_mode: {conditional_eval_mode}")

    corpus_latents_by_tidx, n_corpus = load_corpus_latents(Path(args.corpus_latents_path), time_indices)
    print(f"Corpus: {n_corpus} samples per time")

    agent = build_latent_msbm_agent(
        train_cfg,
        zt,
        latent_dim,
        device,
        latent_train=latent_train,
        latent_test=latent_test,
    )
    load_policy_checkpoints(
        agent,
        run_dir,
        device,
        use_ema=args.use_ema,
        load_forward=False,
        load_backward=True,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    test_sample_indices = rng.choice(n_test, size=min(args.n_test_samples, n_test), replace=False)
    test_sample_indices.sort()
    corpus_eval_indices = rng.choice(n_corpus, size=min(args.n_test_samples, n_corpus), replace=False)
    corpus_eval_indices.sort()
    full_H_schedule = build_full_H_schedule(args.H_meso_list, args.H_macro)

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
    pair_metadata_all: dict[str, dict[str, object]] = {}
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

        print(f"\n{'=' * 60}")
        print(
            f"Scale pair: {display_label}  "
            f"(modeled marginal {pair_idx + 2}/{t_count} -> {pair_idx + 1}/{t_count})"
        )
        print(
            f"  dataset idx {tidx_coarse} -> {tidx_fine}  "
            f"(zt[{pair_idx + 1}]={zt[pair_idx + 1]:.4f} -> zt[{pair_idx}]={zt[pair_idx]:.4f})"
        )
        print(f"{'=' * 60}")

        pair_result = _evaluate_scale_pair(
            pair_idx=pair_idx,
            latent_test=latent_test,
            corpus_z_coarse=corpus_latents_by_tidx[tidx_coarse],
            corpus_z_fine=corpus_latents_by_tidx[tidx_fine],
            test_sample_indices=test_sample_indices,
            corpus_eval_indices=corpus_eval_indices,
            agent=agent,
            device=device,
            conditional_eval_mode=conditional_eval_mode,
            adaptive_config=adaptive_config,
            k_neighbors=args.k_neighbors,
            n_realizations=args.n_realizations,
            ecmmd_k_values=ecmmd_k_values,
            ecmmd_bootstrap_reps=args.ecmmd_bootstrap_reps,
            base_seed=args.seed + pair_idx * 10_000,
            drift_clip_norm=args.drift_clip_norm,
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

        print(f"\n  Summary for {pair_label}:")
        print(
            f"    latent W1 mean={latent_w1_arr.mean():.4f}, std={latent_w1_arr.std():.4f}, "
            f"median={np.median(latent_w1_arr):.4f}"
        )
        print(
            f"    latent W2 mean={latent_w2_arr.mean():.4f}, std={latent_w2_arr.std():.4f}, "
            f"median={np.median(latent_w2_arr):.4f}"
        )
        mean_w1_null = float(latent_w1_null_arr.mean())
        mean_w2_null = float(latent_w2_null_arr.mean())
        w1_skill = 1.0 - float(latent_w1_arr.mean()) / mean_w1_null if mean_w1_null > 0.0 else float("nan")
        w2_skill = 1.0 - float(latent_w2_arr.mean()) / mean_w2_null if mean_w2_null > 0.0 else float("nan")
        print(f"    latent W1 null mean={mean_w1_null:.4f}, skill_vs_null={w1_skill:+.4f}")
        print(f"    latent W2 null mean={mean_w2_null:.4f}, skill_vs_null={w2_skill:+.4f}")
        if str(latent_ecmmd.get("graph_mode")) == DEFAULT_CONDITIONAL_EVAL_MODE and "adaptive_radius" in latent_ecmmd:
            adaptive = latent_ecmmd["adaptive_radius"]
            multi = adaptive["derandomized"]
            ci_suffix = ""
            if "bootstrap_ci_lower" in multi and "bootstrap_ci_upper" in multi:
                ci_suffix = f", CI=[{multi['bootstrap_ci_lower']:.4e}, {multi['bootstrap_ci_upper']:.4e}]"
            boot_suffix = f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
            print(
                f"    adaptive ECMMD: single={adaptive['single_draw']['score']:.4e}; "
                f"D_n={multi['score']:.4e}{ci_suffix}{boot_suffix}"
            )
        elif latent_ecmmd.get("k_values"):
            print(f"    latent ECMMD bandwidth={latent_ecmmd['bandwidth']:.4f}")
            for k_key, k_metrics in latent_ecmmd["k_values"].items():
                single = k_metrics["single_draw"]
                multi = k_metrics["derandomized"]
                single_boot = ""
                multi_boot = ""
                if "bootstrap_p_value" in single:
                    single_boot = f", p_boot={single['bootstrap_p_value']:.3g}"
                if "bootstrap_p_value" in multi:
                    multi_boot = f", p_boot={multi['bootstrap_p_value']:.3g}"
                print(
                    f"      K={k_metrics['k_effective']} (requested {k_key}): "
                    f"single={single['score']:.4e}, z={single['z_score']:.3f}, p={single['p_value']:.3g}{single_boot}; "
                    f"D_n={multi['score']:.4e}, z={multi['z_score']:.3f}, p={multi['p_value']:.3g}{multi_boot}"
                )

    metrics: dict[str, object] = {
        "conditional_eval_mode": conditional_eval_mode,
        "k_neighbors": args.k_neighbors,
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
        "full_H_schedule": list(map(float, full_H_schedule)),
        "scale_pairs": {},
    }

    npz_dict: dict[str, object] = {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "corpus_eval_indices": corpus_eval_indices.astype(np.int64),
        "time_indices": time_indices.astype(np.int64),
        "zt": zt.astype(np.float32),
        "ecmmd_k_values_requested": np.asarray(ecmmd_k_values, dtype=np.int64),
        "pair_labels": np.asarray(pair_labels, dtype=object),
    }

    for pair_label in pair_labels:
        mean_w1_null = float(results_latent_w1_null_all[pair_label].mean())
        mean_w2_null = float(results_latent_w2_null_all[pair_label].mean())
        pair_metrics = {
            "pair_metadata": pair_metadata_all[pair_label],
            "latent_w1": _metric_summary(results_latent_w1_all[pair_label]),
            "latent_w2": _metric_summary(results_latent_w2_all[pair_label]),
            "latent_w1_null": _metric_summary(results_latent_w1_null_all[pair_label]),
            "latent_w2_null": _metric_summary(results_latent_w2_null_all[pair_label]),
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
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

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
    print(f"\n{summary_text}")

    summary_path = output_dir / "conditional_latent_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print(f"\nAll latent conditional results saved to {output_dir}/")


if __name__ == "__main__":
    main()
