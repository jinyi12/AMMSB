from __future__ import annotations

import numpy as np
from scripts.csp.conditional_eval_phases import (
    ALL_CONDITIONAL_PHASES,
    CONDITIONAL_PHASE_ALIASES,
    CONDITIONAL_PHASE_ECMMD,
    CONDITIONAL_PHASE_PRESETS,
    CONDITIONAL_PHASE_SAMPLE,
    CONDITIONAL_PHASE_W2,
)
from scripts.fae.tran_evaluation.conditional_support import (
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    build_local_reference_spec,
    build_uniform_sampling_specs_from_neighbors,
    normalise_weights,
    pack_reference_support_metadata,
    pack_reference_support_arrays,
    sampling_spec_ess,
    sampling_spec_eig_rse,
    sampling_spec_indices,
    sampling_spec_mean_rse,
    sampling_spec_radius,
    sampling_spec_weights,
    summarize_reference_sampling_specs,
)

DEFAULT_TOKEN_CONDITIONAL_SAMPLING_MAX_BATCH_SIZE = 512


def sample_token_csp_conditional_chunk(
    runtime,
    coarse_conditions: np.ndarray,
    *,
    zt: np.ndarray,
    realization_start: int,
    realization_stop: int,
    seed: int,
    sample_token_csp_batch_fn,
    global_conditions: np.ndarray | None = None,
    interval_offset: int = 0,
    condition_num_intervals: int | None = None,
) -> np.ndarray:
    coarse_np = np.asarray(coarse_conditions, dtype=np.float32)
    global_np = None if global_conditions is None else np.asarray(global_conditions, dtype=np.float32)
    realization_count = int(realization_stop - realization_start)
    if realization_count <= 0:
        raise ValueError("realization_stop must be greater than realization_start.")
    if coarse_np.ndim != 3:
        raise ValueError(
            "coarse_conditions must have shape (n_conditions, num_latents, token_dim), "
            f"got {coarse_np.shape}."
        )
    if global_np is not None and global_np.shape != coarse_np.shape:
        raise ValueError("global_conditions must match coarse_conditions before realization expansion.")

    repeated = np.repeat(coarse_np, realization_count, axis=0)
    repeated_global = None if global_np is None else np.repeat(global_np, realization_count, axis=0)
    traj = sample_token_csp_batch_fn(
        runtime,
        repeated,
        zt,
        seed=int(seed + realization_start),
        global_condition_batch=repeated_global,
        interval_offset=int(interval_offset),
        condition_num_intervals=condition_num_intervals,
    )
    return np.asarray(traj[:, 0, :, :], dtype=np.float32).reshape(
        int(coarse_np.shape[0]),
        realization_count,
        -1,
    )


def _sampling_spec_metadata_arrays(
    sampling_specs: list[dict[str, object]],
) -> dict[str, object]:
    reference_weights, reference_indices, reference_counts = pack_reference_support_metadata(sampling_specs)
    return {
        "support_indices": reference_indices.astype(np.int64),
        "support_weights": np.asarray(reference_weights, dtype=np.float32),
        "support_counts": reference_counts.astype(np.int64),
        "radius": np.asarray([sampling_spec_radius(spec) for spec in sampling_specs], dtype=np.float32),
        "ess": np.asarray([sampling_spec_ess(spec) for spec in sampling_specs], dtype=np.float32),
        "mean_rse": np.asarray(
            [sampling_spec_mean_rse(spec) for spec in sampling_specs],
            dtype=np.float32,
        ),
        "eig_rse": np.stack(
            [sampling_spec_eig_rse(spec) for spec in sampling_specs],
            axis=0,
        ).astype(np.float32),
        "diagnostics": summarize_reference_sampling_specs(sampling_specs),
    }


def _build_local_empirical_w2_sampling_specs(
    *,
    conditions: np.ndarray,
    observed_reference: np.ndarray,
    k_neighbors: int,
) -> list[dict[str, object]]:
    return [
        build_local_reference_spec(
            query=np.asarray(condition, dtype=np.float32),
            corpus_conditions=conditions,
            corpus_targets=observed_reference,
            exclude_index=None,
            conditional_eval_mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
            k_neighbors=k_neighbors,
        )
        for condition in np.asarray(conditions, dtype=np.float32)
    ]


def build_pair_reference_payload(
    *,
    pair_idx: int,
    latent_test_flat: np.ndarray,
    test_sample_indices: np.ndarray,
    k_neighbors: int,
    ecmmd_k_values: list[int],
    build_chatterjee_graph_payload_fn,
) -> dict[str, object]:
    test_conditions_flat = np.asarray(latent_test_flat[pair_idx + 1, test_sample_indices], dtype=np.float32)
    test_observed_fine_flat = np.asarray(latent_test_flat[pair_idx, test_sample_indices], dtype=np.float32)
    ecmmd_conditions_flat = test_conditions_flat.astype(np.float32)
    ecmmd_observed_reference = test_observed_fine_flat.astype(np.float32)
    visualization_k_requested = (
        int(ecmmd_k_values[0])
        if ecmmd_k_values
        else int(max(1, min(int(k_neighbors), int(max(1, ecmmd_conditions_flat.shape[0] - 1)))))
    )
    chatterjee_graph = build_chatterjee_graph_payload_fn(
        ecmmd_conditions_flat,
        visualization_k_requested,
    )
    neighbor_indices = np.asarray(chatterjee_graph["neighbor_indices"], dtype=np.int64)
    neighbor_radii = np.asarray(chatterjee_graph["neighbor_radii"], dtype=np.float64)
    neighbor_distances = np.asarray(chatterjee_graph["neighbor_distances"], dtype=np.float64)
    ecmmd_sampling_specs = build_uniform_sampling_specs_from_neighbors(
        neighbor_indices,
        neighbor_radii=neighbor_radii,
        mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
    )
    reference_metadata = _sampling_spec_metadata_arrays(ecmmd_sampling_specs)
    local_scores = np.full((ecmmd_conditions_flat.shape[0],), np.nan, dtype=np.float32)

    return {
        "latent_ecmmd_conditions": ecmmd_conditions_flat.astype(np.float32),
        "latent_ecmmd_observed_reference": ecmmd_observed_reference.astype(np.float32),
        "latent_ecmmd_local_scores": local_scores.astype(np.float32),
        "latent_ecmmd_neighbor_indices": neighbor_indices.astype(np.int64),
        "latent_ecmmd_neighbor_radii": neighbor_radii.astype(np.float32),
        "latent_ecmmd_neighbor_distances": neighbor_distances.astype(np.float32),
        "latent_ecmmd_reference_support_indices": np.asarray(reference_metadata["support_indices"], dtype=np.int64),
        "latent_ecmmd_reference_support_weights": np.asarray(reference_metadata["support_weights"], dtype=np.float32),
        "latent_ecmmd_reference_support_counts": np.asarray(reference_metadata["support_counts"], dtype=np.int64),
        "latent_ecmmd_reference_radius": np.asarray(reference_metadata["radius"], dtype=np.float32),
        "latent_ecmmd_reference_ess": np.asarray(reference_metadata["ess"], dtype=np.float32),
        "latent_ecmmd_reference_mean_rse": np.asarray(reference_metadata["mean_rse"], dtype=np.float32),
        "latent_ecmmd_reference_eig_rse": np.asarray(reference_metadata["eig_rse"], dtype=np.float32),
        "reference_diagnostics": reference_metadata["diagnostics"],
        "visualization_k_requested": int(visualization_k_requested),
        "visualization_k_effective": int(chatterjee_graph["k_effective"]),
    }


def reference_support_from_payload(
    source_values: np.ndarray,
    support_indices: np.ndarray,
    support_counts: np.ndarray,
) -> np.ndarray:
    source_arr = np.asarray(source_values, dtype=np.float32)
    if support_indices.ndim != 2:
        raise ValueError(f"support_indices must have shape (n_eval, max_support), got {support_indices.shape}.")
    support = np.zeros((support_indices.shape[0], support_indices.shape[1], source_arr.shape[1]), dtype=np.float32)
    for row in range(int(support_indices.shape[0])):
        take = int(support_counts[row]) if support_counts.size > 0 else int(np.sum(support_indices[row] >= 0))
        if take <= 0:
            continue
        support[row, :take] = source_arr[np.asarray(support_indices[row, :take], dtype=np.int64)]
    return support


def compute_pair_latent_w2_from_payload(
    *,
    pair_label: str,
    payload: dict[str, object],
    k_neighbors: int,
    base_seed: int,
    wasserstein2_latents_fn,
) -> tuple[np.ndarray, np.ndarray]:
    conditions = np.asarray(payload["latent_ecmmd_conditions"], dtype=np.float32)
    observed_reference = np.asarray(payload["latent_ecmmd_observed_reference"], dtype=np.float32)
    generated = np.asarray(payload["latent_ecmmd_generated"], dtype=np.float32)
    rng = np.random.default_rng(int(base_seed))
    w2_sampling_specs = _build_local_empirical_w2_sampling_specs(
        conditions=conditions,
        observed_reference=observed_reference,
        k_neighbors=k_neighbors,
    )

    latent_w2_values: list[float] = []
    latent_w2_null_values: list[float] = []
    for sample_offset, sampling_spec in enumerate(w2_sampling_specs):
        support_indices = sampling_spec_indices(sampling_spec)
        if support_indices.size <= 0:
            raise ValueError(f"No held-out reference support rows available for {pair_label} sample_offset={sample_offset}.")
        ref_latents = np.asarray(observed_reference[support_indices], dtype=np.float32)
        ref_weights = normalise_weights(
            np.asarray(sampling_spec_weights(sampling_spec), dtype=np.float64),
            int(support_indices.size),
        )
        z_gen_np = np.asarray(generated[sample_offset], dtype=np.float32)
        latent_w2 = wasserstein2_latents_fn(
            z_gen_np,
            ref_latents,
            weights_a=None,
            weights_b=ref_weights,
        )
        latent_w2_values.append(latent_w2)

        n_null = int(min(generated.shape[1], observed_reference.shape[0]))
        null_idx = rng.choice(observed_reference.shape[0], size=n_null, replace=False)
        z_null = np.asarray(observed_reference[null_idx], dtype=np.float32)
        latent_w2_null = wasserstein2_latents_fn(
            z_null,
            ref_latents,
            weights_a=None,
            weights_b=ref_weights,
        )
        latent_w2_null_values.append(latent_w2_null)

        if (sample_offset + 1) % 10 == 0 or sample_offset == 0:
            print(
                f"  Test condition {sample_offset + 1}/{generated.shape[0]}: "
                f"local empirical conditional W2={latent_w2:.4f}",
                flush=True,
            )

    return (
        np.asarray(latent_w2_values, dtype=np.float64),
        np.asarray(latent_w2_null_values, dtype=np.float64),
    )


def compute_pair_latent_ecmmd_from_payload(
    *,
    payload: dict[str, object],
    ecmmd_k_values: list[int],
    ecmmd_bootstrap_reps: int,
    base_seed: int,
    compute_chatterjee_local_scores_fn,
    compute_ecmmd_metrics_fn,
    add_bootstrap_ecmmd_calibration_fn,
) -> dict[str, object]:
    conditions = np.asarray(payload["latent_ecmmd_conditions"], dtype=np.float32)
    generated = np.asarray(payload["latent_ecmmd_generated"], dtype=np.float32)
    observed_reference = np.asarray(payload["latent_ecmmd_observed_reference"], dtype=np.float32)
    rng = np.random.default_rng(int(base_seed))

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
    ecmmd_sampling_specs = build_uniform_sampling_specs_from_neighbors(
        np.asarray(chatterjee_payload["neighbor_indices"], dtype=np.int64),
        neighbor_radii=np.asarray(chatterjee_payload["neighbor_radii"], dtype=np.float64),
        mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
    )
    _reference_support, reference_weights, reference_indices, reference_counts = pack_reference_support_arrays(
        observed_reference,
        ecmmd_sampling_specs,
    )
    reference_metadata = _sampling_spec_metadata_arrays(ecmmd_sampling_specs)
    local_scores = np.asarray(chatterjee_payload["derandomized_scores"], dtype=np.float32)
    neighborhood_indices = np.asarray(chatterjee_payload["neighbor_indices"], dtype=np.int64)
    neighborhood_radii = np.asarray(chatterjee_payload["neighbor_radii"], dtype=np.float32)
    neighborhood_distances = np.asarray(chatterjee_payload["neighbor_distances"], dtype=np.float32)
    latent_ecmmd = compute_ecmmd_metrics_fn(
        conditions,
        observed_reference,
        generated,
        ecmmd_k_values,
        condition_graph_mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
    )
    latent_ecmmd = add_bootstrap_ecmmd_calibration_fn(
        latent_ecmmd,
        conditions=conditions,
        reference_samples=observed_reference,
        corpus_targets=observed_reference,
        sampling_specs=ecmmd_sampling_specs,
        k_values=ecmmd_k_values,
        n_bootstrap=ecmmd_bootstrap_reps,
        rng=rng,
        reference_weights=None,
        condition_graph_mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
        generated_samples=generated,
    )
    latent_ecmmd["reference_mode"] = CHATTERJEE_CONDITIONAL_EVAL_MODE
    latent_ecmmd["reference_diagnostics"] = reference_metadata["diagnostics"]
    latent_ecmmd["visualization_k_requested"] = int(visualization_k_requested)
    latent_ecmmd["visualization_k_effective"] = int(chatterjee_payload["k_effective"])
    latent_ecmmd["estimand"] = "matched_pair_ecmmd"
    latent_ecmmd["estimand_label"] = "held-out matched-pair ECMMD"
    return {
        "latent_ecmmd": latent_ecmmd,
        "latent_ecmmd_local_scores": local_scores,
        "latent_ecmmd_neighbor_indices": neighborhood_indices,
        "latent_ecmmd_neighbor_radii": neighborhood_radii,
        "latent_ecmmd_neighbor_distances": neighborhood_distances,
        "latent_ecmmd_reference_support_indices": reference_indices.astype(np.int64),
        "latent_ecmmd_reference_support_weights": reference_weights.astype(np.float32),
        "latent_ecmmd_reference_support_counts": reference_counts.astype(np.int64),
        "latent_ecmmd_reference_radius": np.asarray(reference_metadata["radius"], dtype=np.float32),
        "latent_ecmmd_reference_ess": np.asarray(reference_metadata["ess"], dtype=np.float32),
        "latent_ecmmd_reference_mean_rse": np.asarray(reference_metadata["mean_rse"], dtype=np.float32),
        "latent_ecmmd_reference_eig_rse": np.asarray(reference_metadata["eig_rse"], dtype=np.float32),
    }


def load_saved_pair_sample_payload(
    existing_results: dict[str, np.ndarray] | None,
    *,
    pair_label: str,
) -> dict[str, object]:
    if existing_results is None:
        raise FileNotFoundError(
            "Missing knn_reference_results.npz. Run the reference-cache stage first: --phases reference_cache."
        )
    payload_keys = (
        "latent_ecmmd_conditions",
        "latent_ecmmd_reference",
        "latent_ecmmd_observed_reference",
        "latent_ecmmd_generated",
        "latent_ecmmd_local_scores",
        "latent_ecmmd_neighbor_indices",
        "latent_ecmmd_neighbor_radii",
        "latent_ecmmd_neighbor_distances",
        "latent_ecmmd_reference_support_indices",
        "latent_ecmmd_reference_support_weights",
        "latent_ecmmd_reference_support_counts",
        "latent_ecmmd_reference_radius",
        "latent_ecmmd_reference_ess",
        "latent_ecmmd_reference_mean_rse",
        "latent_ecmmd_reference_eig_rse",
    )
    missing = [f"{key}_{pair_label}" for key in payload_keys if f"{key}_{pair_label}" not in existing_results]
    if missing:
        raise FileNotFoundError(
            "Existing knn-reference cache payload is incomplete. Missing keys: "
            + ", ".join(missing)
            + ". Re-run with --phases reference_cache."
        )
    return {
        key: np.asarray(existing_results[f"{key}_{pair_label}"])
        for key in payload_keys
    }
