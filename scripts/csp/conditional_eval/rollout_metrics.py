from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from data.transform_utils import apply_forward_transform, load_transform_info
from mmsfm.fae.fae_latent_utils import (
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.csp.conditional_eval.field_metrics import (
    summarize_exact_query_paircorr_metrics,
    summarize_rollout_paircorr_metrics,
    summarize_field_metrics,
)
from scripts.csp.conditional_eval.representative_selection import (
    select_representative_conditions,
)
from scripts.csp.conditional_eval.rollout_generated_cache import (
    iter_generated_rollout_latent_store_chunks,
    iter_generated_rollout_store_chunks,
)
from scripts.csp.conditional_eval.rollout_targets import flatten_latent_rows
from scripts.fae.tran_evaluation.conditional_diversity import (
    DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
    LEGACY_LATENT_DIVERSITY_FEATURE_SPACE,
    PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
    RAW_FIELD_DIVERSITY_FEATURE_SPACE,
    build_local_response_diversity_metrics,
    compute_conditional_diversity_metrics,
    compute_grouped_conditional_diversity_metrics,
    extract_feature_rows,
)
from scripts.fae.tran_evaluation.conditional_support import wasserstein2_latents


def field_score(row: dict[str, Any]) -> float:
    return float(
        (
            float(row["w1_normalized"])
            + float(row.get("paircorr_J_normalized", row.get("J_normalized")))
            + float(
                row.get(
                    "paircorr_xi_relative_error",
                    row.get("corr_length_relative_error"),
                )
            )
        )
        / 3.0
    )


def _iter_rollout_latent_chunks(
    generated_cache: dict[str, Any],
) -> Iterator[tuple[int, np.ndarray]]:
    if "sampled_rollout_latents" in generated_cache:
        yield 0, np.asarray(generated_cache["sampled_rollout_latents"], dtype=np.float32)
        return
    for chunk_start, _chunk_name, latent_chunk in iter_generated_rollout_latent_store_chunks(generated_cache):
        yield int(chunk_start), np.asarray(latent_chunk["sampled_rollout_latents"], dtype=np.float32)


def _iter_rollout_field_chunks(
    generated_cache: dict[str, Any],
) -> Iterator[tuple[int, np.ndarray]]:
    if "decoded_rollout_fields" in generated_cache:
        yield 0, np.asarray(generated_cache["decoded_rollout_fields"], dtype=np.float32)
        return
    for chunk_start, _chunk_name, decoded_chunk, _rollout_latents in iter_generated_rollout_store_chunks(
        generated_cache,
        include_rollout_latents=False,
    ):
        yield int(chunk_start), np.asarray(decoded_chunk["decoded_rollout_fields"], dtype=np.float32)


def _resolve_runtime_fae_checkpoint_path(runtime) -> Path:
    metadata = getattr(runtime, "metadata", {}) or {}
    raw_path = metadata.get("fae_checkpoint_path")
    if raw_path in (None, "", "None"):
        raise ValueError(
            "Decoded-field diversity requires a resolvable frozen FAE checkpoint path in runtime.metadata."
        )
    return Path(str(raw_path)).expanduser().resolve()


def _build_frozen_field_encoder(
    *,
    dataset_path: str | Path,
    grid_coords: np.ndarray,
    fae_checkpoint_path: Path,
) -> Callable[[np.ndarray], np.ndarray]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        transform_info = load_transform_info(dataset_npz)
    checkpoint = load_fae_checkpoint(Path(fae_checkpoint_path).expanduser().resolve())
    autoencoder, fae_params, fae_batch_stats, _ = build_fae_from_checkpoint(checkpoint)
    encode_fn, _decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode="standard",
    )
    coords = np.asarray(grid_coords, dtype=np.float32)

    def _encode(field_samples: np.ndarray) -> np.ndarray:
        fields = np.asarray(field_samples, dtype=np.float32)
        if fields.ndim < 2:
            raise ValueError(f"Expected decoded field samples with shape (N, ...), got {fields.shape}.")
        flat = np.asarray(fields.reshape(fields.shape[0], -1), dtype=np.float32)
        if int(flat.shape[1]) != int(coords.shape[0]):
            raise ValueError(
                "Decoded field / grid mismatch for frozen FAE re-encoding: "
                f"{flat.shape[1]} points vs {coords.shape[0]} grid points."
            )
        transformed = np.asarray(apply_forward_transform(flat, transform_info), dtype=np.float32)
        coords_batch = np.broadcast_to(coords[None, ...], (flat.shape[0], *coords.shape))
        encoded = np.asarray(
            encode_fn(transformed[..., None], coords_batch),
            dtype=np.float32,
        )
        return np.asarray(encoded.reshape(encoded.shape[0], -1), dtype=np.float32)

    return _encode


def _compute_condition_pca(
    *,
    runtime,
    test_sample_indices: np.ndarray,
) -> np.ndarray:
    root_conditions = flatten_latent_rows(runtime.latent_test[-1, test_sample_indices, ...])
    centered = root_conditions - np.mean(root_conditions, axis=0, keepdims=True)
    if centered.shape[0] <= 1 or np.allclose(centered, 0.0):
        return np.zeros((centered.shape[0], 2), dtype=np.float64)
    u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, : min(2, s.shape[0])] * s[: min(2, s.shape[0])]
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    return np.asarray(coords[:, :2], dtype=np.float64)


def _assign_representative_roles(
    *,
    runtime,
    metrics_by_target: dict[str, dict[str, Any]],
    aggregate_scores: np.ndarray,
    test_sample_indices: np.ndarray,
    representative_seed: int,
    n_plot_conditions: int,
) -> tuple[np.ndarray, list[str]]:
    if metrics_by_target:
        aggregate_scores = np.asarray(aggregate_scores, dtype=np.float64) / float(len(metrics_by_target))
    selected_rows, selected_roles = select_representative_conditions(
        local_scores=np.asarray(aggregate_scores, dtype=np.float64),
        condition_pca=_compute_condition_pca(
            runtime=runtime,
            test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        ),
        n_show=int(n_plot_conditions),
        seed=int(representative_seed),
    )
    role_map = {
        int(row): str(role)
        for row, role in zip(selected_rows.tolist(), selected_roles, strict=False)
    }
    for metrics in metrics_by_target.values():
        for row in metrics["per_condition"]:
            row["role"] = str(role_map.get(int(row["row_index"]), ""))
    return selected_rows.astype(np.int64), [str(role) for role in selected_roles]


def _compute_rollout_latent_metrics(
    *,
    runtime,
    latent_chunks: Iterator[tuple[int, np.ndarray]],
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None,
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    rollout_condition_mode: str,
    conditional_diversity_vendi_top_k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    support_indices = np.asarray(reference_cache["reference_support_indices"], dtype=np.int64)
    support_weights = np.asarray(reference_cache["reference_support_weights"], dtype=np.float32)
    support_counts = np.asarray(reference_cache["reference_support_counts"], dtype=np.int64)
    reference_assignment_indices = None
    if assignment_cache is not None:
        reference_assignment_indices = np.asarray(
            assignment_cache["reference_assignment_indices"],
            dtype=np.int64,
        )
    per_target = {
        str(spec["label"]): np.zeros((support_indices.shape[0],), dtype=np.float32)
        for spec in target_specs
    }
    conditioning_state_latents = np.asarray(runtime.latent_test[-1, test_sample_indices, ...], dtype=np.float32)
    conditioning_state_features, conditioning_feature_pooling = extract_feature_rows(
        conditioning_state_latents,
        feature_space=LEGACY_LATENT_DIVERSITY_FEATURE_SPACE,
    )
    response_features_by_target: dict[str, np.ndarray | None] = {str(spec["label"]): None for spec in target_specs}
    response_feature_pooling_by_target: dict[str, str] = {}

    for chunk_start, rollout_latents in latent_chunks:
        for local_row in range(int(rollout_latents.shape[0])):
            row = int(chunk_start + local_row)
            for spec in target_specs:
                label = str(spec["label"])
                rollout_pos = int(spec["rollout_pos"])
                generated_response_latents = np.asarray(
                    rollout_latents[local_row, :, rollout_pos, ...],
                    dtype=np.float32,
                )
                generated_response_features, response_pooling = extract_feature_rows(
                    generated_response_latents,
                    feature_space=LEGACY_LATENT_DIVERSITY_FEATURE_SPACE,
                )
                if response_features_by_target[label] is None:
                    response_features_by_target[label] = np.zeros(
                        (
                            support_indices.shape[0],
                            generated_response_features.shape[0],
                            generated_response_features.shape[1],
                        ),
                        dtype=np.float32,
                    )
                    response_feature_pooling_by_target[label] = str(response_pooling)
                response_features_by_target[label][row] = np.asarray(
                    generated_response_features,
                    dtype=np.float32,
                )
                generated = flatten_latent_rows(rollout_latents[local_row, :, rollout_pos, ...])
                if str(rollout_condition_mode) == "chatterjee_knn":
                    if reference_assignment_indices is None:
                        raise ValueError("chatterjee_knn rollout latent metrics require an assignment cache.")
                    chosen = np.asarray(reference_assignment_indices[row], dtype=np.int64)
                    reference = flatten_latent_rows(runtime.latent_test[rollout_pos, chosen, ...])
                    per_target[label][row] = np.float32(wasserstein2_latents(generated, reference))
                else:
                    take = int(support_counts[row])
                    chosen = np.asarray(support_indices[row, :take], dtype=np.int64)
                    weights = np.asarray(support_weights[row, :take], dtype=np.float32)
                    reference = flatten_latent_rows(runtime.latent_test[rollout_pos, chosen, ...])
                    per_target[label][row] = np.float32(
                        wasserstein2_latents(
                            generated,
                            reference,
                            None,
                            weights,
                        )
                    )

    metrics_by_target: dict[str, dict[str, Any]] = {}
    results_payload: dict[str, np.ndarray] = {}
    for spec in target_specs:
        label = str(spec["label"])
        values = np.asarray(per_target[label], dtype=np.float32)
        response_features = response_features_by_target[label]
        if response_features is None:
            raise ValueError(f"Missing response features for rollout latent metric label {label}.")
        conditional_diversity, conditional_diversity_results = compute_conditional_diversity_metrics(
            conditioning_state_latents=conditioning_state_features,
            grouped_response_latents=np.asarray(response_features, dtype=np.float32),
            response_label=label,
            conditioning_state_time_index=int(runtime.time_indices[-1]),
            conditioning_scale_H=float(spec["H_condition"]),
            response_state_time_index=int(spec["time_index"]),
            response_scale_H=float(spec["H_target"]),
            test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
            vendi_top_k=int(conditional_diversity_vendi_top_k),
            conditioning_feature_pooling=str(conditioning_feature_pooling),
            response_feature_pooling=str(response_feature_pooling_by_target.get(label, "identity")),
        )
        metrics_by_target[label] = {
            "label": label,
            "display_label": str(spec["display_label"]),
            "time_index": int(spec["time_index"]),
            "H_target": float(spec["H_target"]),
            "mean_w2": float(np.mean(values)),
            "std_w2": float(np.std(values)),
            "legacy_conditional_diversity": conditional_diversity,
        }
        results_payload[f"latent_w2_{label}"] = values
        results_payload.update(conditional_diversity_results)
    return metrics_by_target, results_payload


def _compute_rollout_field_metrics(
    *,
    runtime,
    decode_resolution: int,
    pixel_size: float,
    field_chunks: Iterator[tuple[int, np.ndarray]],
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None,
    test_fields_by_tidx: dict[int, np.ndarray],
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    representative_seed: int,
    n_plot_conditions: int,
    rollout_condition_mode: str,
    generated_field_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    reference_time_index_fn: Callable[[dict[str, Any]], int] | None = None,
    results_prefix: str = "field",
    selected_rows_override: np.ndarray | None = None,
    selected_roles_override: list[str] | None = None,
    include_selection_payload: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], np.ndarray, list[str]]:
    reference_assignment_indices = None
    if assignment_cache is not None:
        reference_assignment_indices = np.asarray(
            assignment_cache["reference_assignment_indices"],
            dtype=np.int64,
        )
    metrics_rows = {
        str(spec["label"]): [None] * int(test_sample_indices.shape[0])
        for spec in target_specs
    }
    aggregate_scores = np.zeros((int(test_sample_indices.shape[0]),), dtype=np.float64)

    for chunk_start, chunk_fields in field_chunks:
        for local_row in range(int(chunk_fields.shape[0])):
            row = int(chunk_start + local_row)
            for spec in target_specs:
                label = str(spec["label"])
                rollout_pos = int(spec["rollout_pos"])
                generated_fields = np.asarray(chunk_fields[local_row, :, rollout_pos, :], dtype=np.float32)
                if generated_field_transform is not None:
                    generated_fields = np.asarray(
                        generated_field_transform(generated_fields, spec),
                        dtype=np.float32,
                    ).reshape(generated_fields.shape[0], -1)
                target_time_index = (
                    int(spec["time_index"])
                    if reference_time_index_fn is None
                    else int(reference_time_index_fn(spec))
                )
                if str(rollout_condition_mode) == "chatterjee_knn":
                    if reference_assignment_indices is None:
                        raise ValueError("chatterjee_knn rollout field metrics require an assignment cache.")
                    chosen = np.asarray(reference_assignment_indices[row], dtype=np.int64)
                else:
                    chosen = np.asarray([int(test_sample_indices[row])], dtype=np.int64)
                reference_fields = np.asarray(
                    test_fields_by_tidx[target_time_index][chosen].reshape(len(chosen), -1),
                    dtype=np.float32,
                )
                mode = str(rollout_condition_mode)
                if mode == "exact_query":
                    summary = summarize_exact_query_paircorr_metrics(
                        reference_fields=reference_fields,
                        generated_fields=generated_fields,
                        resolution=int(decode_resolution),
                        pixel_size=float(pixel_size),
                        min_spacing_pixels=4,
                    )
                    per_row = {
                        "row_index": int(row),
                        "test_sample_index": int(test_sample_indices[row]),
                        "w1_normalized": float(summary["w1"]["w1_normalised"]),
                        "paircorr_J_normalized": float(summary["paircorr_J"]["J_normalised"]),
                        "paircorr_xi_relative_error": float(summary["paircorr_xi_relative_error"]),
                        "field_score": field_score(
                            {
                                "w1_normalized": float(summary["w1"]["w1_normalised"]),
                                "paircorr_J_normalized": float(summary["paircorr_J"]["J_normalised"]),
                                "paircorr_xi_relative_error": float(summary["paircorr_xi_relative_error"]),
                            }
                        ),
                        "role": "",
                        "w1": summary["w1"],
                        "moments": summary["moments"],
                        "paircorr_J": summary["paircorr_J"],
                        "xi_obs_e1": float(summary["xi_obs_e1"]),
                        "xi_obs_e2": float(summary["xi_obs_e2"]),
                        "xi_gen_e1": float(summary["xi_gen_e1"]),
                        "xi_gen_e2": float(summary["xi_gen_e2"]),
                        "paircorr_r_max_pixels": int(summary["paircorr_r_max_pixels"]),
                    }
                elif mode == "chatterjee_knn":
                    summary = summarize_rollout_paircorr_metrics(
                        reference_fields=reference_fields,
                        generated_fields=generated_fields,
                        resolution=int(decode_resolution),
                        pixel_size=float(pixel_size),
                        min_spacing_pixels=4,
                        rollout_condition_mode=mode,
                    )
                    per_row = {
                        "row_index": int(row),
                        "test_sample_index": int(test_sample_indices[row]),
                        "w1_normalized": float(summary["w1"]["w1_normalised"]),
                        "J_normalized": float(summary["paircorr_J"]["J_normalised"]),
                        "corr_length_relative_error": float(summary["paircorr_xi_relative_error"]),
                        "field_score": field_score(
                            {
                                "w1_normalized": float(summary["w1"]["w1_normalised"]),
                                "J_normalized": float(summary["paircorr_J"]["J_normalised"]),
                                "corr_length_relative_error": float(summary["paircorr_xi_relative_error"]),
                            }
                        ),
                        "role": "",
                        "w1": summary["w1"],
                        "moments": summary["moments"],
                        "J": summary["paircorr_J"],
                        "xi_obs_e1": float(summary["xi_obs_e1"]),
                        "xi_obs_e2": float(summary["xi_obs_e2"]),
                        "xi_gen_e1": float(summary["xi_gen_e1"]),
                        "xi_gen_e2": float(summary["xi_gen_e2"]),
                        "paircorr_r_max_pixels": int(summary["paircorr_r_max_pixels"]),
                    }
                else:
                    summary = summarize_field_metrics(
                        reference_fields=reference_fields,
                        generated_fields=generated_fields,
                        resolution=int(decode_resolution),
                        pixel_size=float(pixel_size),
                        min_spacing_pixels=4,
                    )
                    per_row = {
                        "row_index": int(row),
                        "test_sample_index": int(test_sample_indices[row]),
                        "w1_normalized": float(summary["w1"]["w1_normalised"]),
                        "J_normalized": float(summary["J"]["J_normalised"]),
                        "corr_length_relative_error": float(summary["corr_length_relative_error"]),
                        "field_score": field_score(
                            {
                                "w1_normalized": float(summary["w1"]["w1_normalised"]),
                                "J_normalized": float(summary["J"]["J_normalised"]),
                                "corr_length_relative_error": float(summary["corr_length_relative_error"]),
                            }
                        ),
                        "role": "",
                        "w1": summary["w1"],
                        "moments": summary["moments"],
                        "J": summary["J"],
                        "xi_obs_e1": float(summary["xi_obs_e1"]),
                        "xi_obs_e2": float(summary["xi_obs_e2"]),
                        "xi_gen_e1": float(summary["xi_gen_e1"]),
                        "xi_gen_e2": float(summary["xi_gen_e2"]),
                    }
                aggregate_scores[row] += float(per_row["field_score"])
                metrics_rows[label][row] = per_row

    metrics_by_target: dict[str, dict[str, Any]] = {}
    results_payload: dict[str, np.ndarray] = {}
    for spec in target_specs:
        label = str(spec["label"])
        per_condition = [row for row in metrics_rows[label] if row is not None]
        mode = str(rollout_condition_mode)
        if mode == "exact_query":
            metrics_by_target[label] = {
                "label": label,
                "display_label": str(spec["display_label"]),
                "time_index": int(spec["time_index"]),
                "H_target": float(spec["H_target"]),
                "summary": {
                    "mean_w1_normalized": float(np.mean([row["w1_normalized"] for row in per_condition])),
                    "mean_paircorr_J_normalized": float(
                        np.mean([row["paircorr_J_normalized"] for row in per_condition])
                    ),
                    "mean_paircorr_xi_relative_error": float(
                        np.mean([row["paircorr_xi_relative_error"] for row in per_condition])
                    ),
                },
                "per_condition": per_condition,
            }
        else:
            metrics_by_target[label] = {
                "label": label,
                "display_label": str(spec["display_label"]),
                "time_index": int(spec["time_index"]),
                "H_target": float(spec["H_target"]),
                "summary": {
                    "mean_w1_normalized": float(np.mean([row["w1_normalized"] for row in per_condition])),
                    "mean_J_normalized": float(np.mean([row["J_normalized"] for row in per_condition])),
                    "mean_corr_length_relative_error": float(
                        np.mean([row["corr_length_relative_error"] for row in per_condition])
                    ),
                },
                "per_condition": per_condition,
            }
        results_payload[f"{results_prefix}_w1_normalized_{label}"] = np.asarray(
            [row["w1_normalized"] for row in per_condition],
            dtype=np.float32,
        )
        if mode == "exact_query":
            results_payload[f"{results_prefix}_paircorr_J_normalized_{label}"] = np.asarray(
                [row["paircorr_J_normalized"] for row in per_condition],
                dtype=np.float32,
            )
            results_payload[f"{results_prefix}_paircorr_xi_relative_error_{label}"] = np.asarray(
                [row["paircorr_xi_relative_error"] for row in per_condition],
                dtype=np.float32,
            )
        else:
            results_payload[f"{results_prefix}_J_normalized_{label}"] = np.asarray(
                [row["J_normalized"] for row in per_condition],
                dtype=np.float32,
            )
            results_payload[f"{results_prefix}_corr_length_relative_error_{label}"] = np.asarray(
                [row["corr_length_relative_error"] for row in per_condition],
                dtype=np.float32,
            )

    if selected_rows_override is None or selected_roles_override is None:
        selected_rows, selected_roles = _assign_representative_roles(
            runtime=runtime,
            metrics_by_target=metrics_by_target,
            aggregate_scores=aggregate_scores,
            test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
            representative_seed=int(representative_seed),
            n_plot_conditions=int(n_plot_conditions),
        )
    else:
        selected_rows = np.asarray(selected_rows_override, dtype=np.int64).reshape(-1)
        selected_roles = [str(role) for role in selected_roles_override]
        role_map = {
            int(row): str(role)
            for row, role in zip(selected_rows.tolist(), selected_roles, strict=False)
        }
        for metrics in metrics_by_target.values():
            for row in metrics["per_condition"]:
                row["role"] = str(role_map.get(int(row["row_index"]), ""))
    if include_selection_payload:
        results_payload["selected_condition_rows"] = np.asarray(selected_rows, dtype=np.int64)
        results_payload["selected_condition_roles"] = np.asarray(selected_roles, dtype=np.str_)
    return metrics_by_target, results_payload, selected_rows, list(selected_roles)


def _compute_rollout_field_diversity_for_target(
    *,
    runtime,
    field_chunks: Iterator[tuple[int, np.ndarray]],
    target_spec: dict[str, Any],
    test_fields_by_tidx: dict[int, np.ndarray],
    test_sample_indices: np.ndarray,
    frozen_field_encoder: Callable[[np.ndarray], np.ndarray],
    conditional_diversity_vendi_top_k: int,
    grouping_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    label = str(target_spec["label"])
    conditioning_time_index = int(target_spec["conditioning_time_index"])
    coarse_condition_fields = np.asarray(
        test_fields_by_tidx[conditioning_time_index][np.asarray(test_sample_indices, dtype=np.int64)],
        dtype=np.float32,
    ).reshape(len(test_sample_indices), -1)
    conditioning_features, conditioning_pooling = extract_feature_rows(
        coarse_condition_fields,
        feature_space=PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
        frozen_field_encoder=frozen_field_encoder,
    )

    grouped_primary_features: np.ndarray | None = None
    grouped_raw_features: np.ndarray | None = None
    response_feature_pooling = "identity"
    raw_feature_pooling = "centered_flat"

    for chunk_start, chunk_fields in field_chunks:
        for local_row in range(int(chunk_fields.shape[0])):
            row = int(chunk_start + local_row)
            generated_fields = np.asarray(
                chunk_fields[local_row, :, int(target_spec["rollout_pos"]), :],
                dtype=np.float32,
            ).reshape(int(chunk_fields.shape[1]), -1)
            primary_features, response_feature_pooling = extract_feature_rows(
                generated_fields,
                feature_space=PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
                frozen_field_encoder=frozen_field_encoder,
            )
            raw_features, raw_feature_pooling = extract_feature_rows(
                generated_fields,
                feature_space=RAW_FIELD_DIVERSITY_FEATURE_SPACE,
            )
            if grouped_primary_features is None:
                grouped_primary_features = np.zeros(
                    (
                        len(test_sample_indices),
                        primary_features.shape[0],
                        primary_features.shape[1],
                    ),
                    dtype=np.float32,
                )
            if grouped_raw_features is None:
                grouped_raw_features = np.zeros(
                    (
                        len(test_sample_indices),
                        raw_features.shape[0],
                        raw_features.shape[1],
                    ),
                    dtype=np.float32,
                )
            grouped_primary_features[row] = np.asarray(primary_features, dtype=np.float32)
            grouped_raw_features[row] = np.asarray(raw_features, dtype=np.float32)

    if grouped_primary_features is None or grouped_raw_features is None:
        raise ValueError(f"Missing generated decoded fields for rollout diversity label {label}.")

    local_metrics, local_results = build_local_response_diversity_metrics(
        grouped_primary_features,
        response_label=label,
        conditioning_state_time_index=int(target_spec["conditioning_time_index"]),
        conditioning_scale_H=float(target_spec["H_condition"]),
        response_state_time_index=int(target_spec["time_index"]),
        response_scale_H=float(target_spec["H_target"]),
        feature_space=PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
        response_feature_pooling=str(response_feature_pooling),
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        vendi_top_k=int(conditional_diversity_vendi_top_k),
        results_prefix="field",
    )
    raw_local_metrics, raw_local_results = build_local_response_diversity_metrics(
        grouped_raw_features,
        response_label=label,
        conditioning_state_time_index=int(target_spec["conditioning_time_index"]),
        conditioning_scale_H=float(target_spec["H_condition"]),
        response_state_time_index=int(target_spec["time_index"]),
        response_scale_H=float(target_spec["H_target"]),
        feature_space=RAW_FIELD_DIVERSITY_FEATURE_SPACE,
        response_feature_pooling=str(raw_feature_pooling),
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        vendi_top_k=int(conditional_diversity_vendi_top_k),
        results_prefix="field_raw",
    )
    grouped_metrics, grouped_results = compute_grouped_conditional_diversity_metrics(
        conditioning_features=conditioning_features,
        grouped_response_features=np.asarray(grouped_primary_features, dtype=np.float32),
        response_label=label,
        conditioning_state_time_index=int(target_spec["conditioning_time_index"]),
        conditioning_scale_H=float(target_spec["H_condition"]),
        response_state_time_index=int(target_spec["time_index"]),
        response_scale_H=float(target_spec["H_target"]),
        feature_space=PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
        conditioning_feature_pooling=str(conditioning_pooling),
        response_feature_pooling=str(response_feature_pooling),
        vendi_top_k=int(conditional_diversity_vendi_top_k),
        grouping_seed=int(grouping_seed),
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        results_prefix="field",
    )

    metrics = {
        "label": label,
        "display_label": str(target_spec["display_label"]),
        "time_index": int(target_spec["time_index"]),
        "H_target": float(target_spec["H_target"]),
        "summary": {
            "mean_local_rke": float(local_metrics["mean_local_rke"]),
            "mean_local_vendi": float(local_metrics["mean_local_vendi"]),
            "mean_raw_local_rke": float(raw_local_metrics["mean_local_rke"]),
            "mean_raw_local_vendi": float(raw_local_metrics["mean_local_vendi"]),
            "group_conditional_rke": float(grouped_metrics["group_conditional_rke"]),
            "group_conditional_vendi": float(grouped_metrics["group_conditional_vendi"]),
            "group_information_vendi": float(grouped_metrics["group_information_vendi"]),
            "response_vendi": float(grouped_metrics["response_vendi"]),
        },
        "local_diversity": local_metrics,
        "raw_local_diversity": raw_local_metrics,
        "grouped_global_diversity": grouped_metrics,
    }
    results = {}
    results.update(local_results)
    results.update(raw_local_results)
    results.update(grouped_results)
    return metrics, results


def _compute_rollout_field_diversity(
    *,
    runtime,
    field_chunk_factory: Callable[[], Iterator[tuple[int, np.ndarray]]],
    dataset_path: str | Path,
    grid_coords: np.ndarray,
    test_fields_by_tidx: dict[int, np.ndarray],
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    conditional_diversity_vendi_top_k: int,
    grouping_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    frozen_field_encoder = _build_frozen_field_encoder(
        dataset_path=dataset_path,
        grid_coords=np.asarray(grid_coords, dtype=np.float32),
        fae_checkpoint_path=_resolve_runtime_fae_checkpoint_path(runtime),
    )
    metrics_by_target: dict[str, dict[str, Any]] = {}
    results_payload: dict[str, np.ndarray] = {}
    for spec in target_specs:
        target_metrics, target_results = _compute_rollout_field_diversity_for_target(
            runtime=runtime,
            field_chunks=field_chunk_factory(),
            target_spec=spec,
            test_fields_by_tidx=test_fields_by_tidx,
            test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
            frozen_field_encoder=frozen_field_encoder,
            conditional_diversity_vendi_top_k=int(conditional_diversity_vendi_top_k),
            grouping_seed=int(grouping_seed),
        )
        metrics_by_target[str(spec["label"])] = target_metrics
        results_payload.update(target_results)
    return metrics_by_target, results_payload


def compute_rollout_latent_metrics(
    *,
    runtime,
    generated_rollout_latents: np.ndarray,
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None = None,
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    rollout_condition_mode: str = "exact_query",
    conditional_diversity_vendi_top_k: int = DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    return _compute_rollout_latent_metrics(
        runtime=runtime,
        latent_chunks=iter([(0, np.asarray(generated_rollout_latents, dtype=np.float32))]),
        reference_cache=reference_cache,
        assignment_cache=assignment_cache,
        target_specs=target_specs,
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        rollout_condition_mode=str(rollout_condition_mode),
        conditional_diversity_vendi_top_k=int(conditional_diversity_vendi_top_k),
    )


def compute_rollout_field_metrics(
    *,
    runtime,
    decode_resolution: int,
    pixel_size: float,
    generated_rollout_fields: np.ndarray,
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None = None,
    test_fields_by_tidx: dict[int, np.ndarray],
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    representative_seed: int,
    n_plot_conditions: int,
    rollout_condition_mode: str = "exact_query",
    generated_field_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    reference_time_index_fn: Callable[[dict[str, Any]], int] | None = None,
    results_prefix: str = "field",
    selected_rows_override: np.ndarray | None = None,
    selected_roles_override: list[str] | None = None,
    include_selection_payload: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], np.ndarray, list[str]]:
    return _compute_rollout_field_metrics(
        runtime=runtime,
        decode_resolution=int(decode_resolution),
        pixel_size=float(pixel_size),
        field_chunks=iter([(0, np.asarray(generated_rollout_fields, dtype=np.float32))]),
        reference_cache=reference_cache,
        assignment_cache=assignment_cache,
        test_fields_by_tidx=test_fields_by_tidx,
        target_specs=target_specs,
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        representative_seed=int(representative_seed),
        n_plot_conditions=int(n_plot_conditions),
        rollout_condition_mode=str(rollout_condition_mode),
        generated_field_transform=generated_field_transform,
        reference_time_index_fn=reference_time_index_fn,
        results_prefix=str(results_prefix),
        selected_rows_override=selected_rows_override,
        selected_roles_override=selected_roles_override,
        include_selection_payload=bool(include_selection_payload),
    )


def compute_rollout_field_diversity(
    *,
    runtime,
    generated_rollout_fields: np.ndarray,
    dataset_path: str | Path,
    grid_coords: np.ndarray,
    test_fields_by_tidx: dict[int, np.ndarray],
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    grouping_seed: int,
    conditional_diversity_vendi_top_k: int = DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    generated = np.asarray(generated_rollout_fields, dtype=np.float32)
    return _compute_rollout_field_diversity(
        runtime=runtime,
        field_chunk_factory=lambda: iter([(0, generated)]),
        dataset_path=dataset_path,
        grid_coords=np.asarray(grid_coords, dtype=np.float32),
        test_fields_by_tidx=test_fields_by_tidx,
        target_specs=target_specs,
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        conditional_diversity_vendi_top_k=int(conditional_diversity_vendi_top_k),
        grouping_seed=int(grouping_seed),
    )


def compute_rollout_latent_metrics_from_cache(
    *,
    runtime,
    generated_cache: dict[str, Any],
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None = None,
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    rollout_condition_mode: str = "exact_query",
    conditional_diversity_vendi_top_k: int = DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    return _compute_rollout_latent_metrics(
        runtime=runtime,
        latent_chunks=_iter_rollout_latent_chunks(generated_cache),
        reference_cache=reference_cache,
        assignment_cache=assignment_cache,
        target_specs=target_specs,
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        rollout_condition_mode=str(rollout_condition_mode),
        conditional_diversity_vendi_top_k=int(conditional_diversity_vendi_top_k),
    )


def compute_rollout_field_diversity_from_cache(
    *,
    runtime,
    generated_cache: dict[str, Any],
    dataset_path: str | Path,
    grid_coords: np.ndarray,
    test_fields_by_tidx: dict[int, np.ndarray],
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    grouping_seed: int,
    conditional_diversity_vendi_top_k: int = DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    return _compute_rollout_field_diversity(
        runtime=runtime,
        field_chunk_factory=lambda: _iter_rollout_field_chunks(generated_cache),
        dataset_path=dataset_path,
        grid_coords=np.asarray(grid_coords, dtype=np.float32),
        test_fields_by_tidx=test_fields_by_tidx,
        target_specs=target_specs,
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        conditional_diversity_vendi_top_k=int(conditional_diversity_vendi_top_k),
        grouping_seed=int(grouping_seed),
    )


def compute_rollout_field_metrics_from_cache(
    *,
    runtime,
    decode_resolution: int,
    pixel_size: float,
    generated_cache: dict[str, Any],
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None = None,
    test_fields_by_tidx: dict[int, np.ndarray],
    target_specs: list[dict[str, Any]],
    test_sample_indices: np.ndarray,
    representative_seed: int,
    n_plot_conditions: int,
    rollout_condition_mode: str = "exact_query",
    generated_field_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    reference_time_index_fn: Callable[[dict[str, Any]], int] | None = None,
    results_prefix: str = "field",
    selected_rows_override: np.ndarray | None = None,
    selected_roles_override: list[str] | None = None,
    include_selection_payload: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], np.ndarray, list[str]]:
    return _compute_rollout_field_metrics(
        runtime=runtime,
        decode_resolution=int(decode_resolution),
        pixel_size=float(pixel_size),
        field_chunks=_iter_rollout_field_chunks(generated_cache),
        reference_cache=reference_cache,
        assignment_cache=assignment_cache,
        test_fields_by_tidx=test_fields_by_tidx,
        target_specs=target_specs,
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        representative_seed=int(representative_seed),
        n_plot_conditions=int(n_plot_conditions),
        rollout_condition_mode=str(rollout_condition_mode),
        generated_field_transform=generated_field_transform,
        reference_time_index_fn=reference_time_index_fn,
        results_prefix=str(results_prefix),
        selected_rows_override=selected_rows_override,
        selected_roles_override=selected_roles_override,
        include_selection_payload=bool(include_selection_payload),
    )
