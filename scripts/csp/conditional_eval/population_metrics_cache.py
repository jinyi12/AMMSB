from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import (
    condition_set_from_metadata_arrays,
    condition_set_to_metadata_arrays,
)
from scripts.csp.conditional_eval.population_contract import (
    POPULATION_METRICS_CACHE_PHASE,
    POPULATION_OUTPUT_DIRNAME,
)
from scripts.csp.conditional_eval.population_decoded_cache import (
    iter_population_decoded_chunks,
    load_population_decoded_metadata,
    population_decoded_store_dir,
    population_decoded_store_manifest,
)
from scripts.csp.conditional_eval.rollout_recoarsening import (
    ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA,
    recoarsen_fields_to_scale,
)
from scripts.csp.conditional_eval.seed_policy import (
    seed_policy_from_metadata_arrays,
    seed_policy_to_metadata_arrays,
)
from scripts.fae.tran_evaluation.first_order import (
    decorrelation_spacing_from_curves,
    subsample_grid_indices,
)
from scripts.fae.tran_evaluation.resumable_store import (
    build_expected_store_manifest,
    load_store_manifest,
    prepare_resumable_store,
)
from scripts.fae.tran_evaluation.second_order import directional_correlation


def population_output_dir(output_dir: Path) -> Path:
    return Path(output_dir) / POPULATION_OUTPUT_DIRNAME


def population_store_dir(output_dir: Path, *, domain_key: str) -> Path:
    return population_output_dir(output_dir) / "metrics_cache" / f"{str(domain_key)}.store"


def population_store_manifest(
    *,
    runtime,
    invocation: dict[str, Any],
    domain_spec: dict[str, Any],
    seed_policy: dict[str, int],
    target_specs: list[dict[str, Any]],
    resolution: int,
    pixel_size: float,
    n_realizations: int,
    coarse_relative_eps: float = 1e-8,
) -> dict[str, Any]:
    decoded_manifest = population_decoded_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        resolution=int(resolution),
    )
    metadata = dict(getattr(runtime, "metadata", {}) or {})
    fingerprint = {
        "provider": str(runtime.provider),
        "model_type": metadata.get("model_type"),
        "condition_mode": metadata.get("condition_mode"),
        "run_dir": str(Path(runtime.run_dir)),
        "dataset_path": str(invocation["dataset_path"]),
        "time_indices": np.asarray(runtime.time_indices, dtype=np.int64).astype(int).tolist(),
        "condition_set_id": str(domain_spec["condition_set"]["condition_set_id"]),
        "root_condition_batch_id": str(domain_spec["condition_set"]["root_condition_batch_id"]),
        "domain": str(domain_spec["domain"]),
        "split": str(domain_spec["split"]),
        "budget_conditions": int(domain_spec["budget_conditions"]),
        "n_realizations": int(n_realizations),
        "generation_seed": int(seed_policy["generation_seed"]),
        "target_contract": [
            {
                "label": str(spec["label"]),
                "rollout_pos": int(spec["rollout_pos"]),
                "time_index": int(spec["time_index"]),
                "conditioning_time_index": int(spec["conditioning_time_index"]),
                "H_target": float(spec["H_target"]),
                "H_condition": float(spec["H_condition"]),
            }
            for spec in target_specs
        ],
        "resolution": int(resolution),
        "pixel_size": float(pixel_size),
        "source_decoded_manifest": decoded_manifest,
        "cache_payload": "population_rollout_metrics_v1",
        "pdf_min_spacing_pixels": 4,
        "transfer_operator": "tran_periodic_tikhonov_transfer",
        "transfer_ridge_lambda": float(ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA),
        "coarse_relative_eps": float(coarse_relative_eps),
    }
    return build_expected_store_manifest(
        store_name=f"conditional_rollout_population_metrics_{str(domain_spec['domain_key'])}",
        store_kind="cache",
        fingerprint=fingerprint,
    )


def _chunk_records(store) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    for path in store.chunks_dir.glob("condition_chunk_*.npz"):
        records.append((int(path.stem.rsplit("_", 1)[-1]), path.stem))
    records.sort(key=lambda item: item[0])
    return records


def _load_population_metadata(store) -> dict[str, Any]:
    if not store.has_chunk("metadata"):
        raise FileNotFoundError(f"Missing population metrics metadata chunk in {store.store_dir}.")
    metadata = store.load_chunk("metadata")
    target_labels = [str(item) for item in np.asarray(metadata["target_labels"], dtype=np.str_).tolist()]
    target_time_indices = np.asarray(metadata["target_time_indices"], dtype=np.int64).reshape(-1)
    target_conditioning_time_indices = np.asarray(
        metadata["target_conditioning_time_indices"], dtype=np.int64
    ).reshape(-1)
    target_rollout_positions = np.asarray(metadata["target_rollout_positions"], dtype=np.int64).reshape(-1)
    target_h = np.asarray(metadata["target_H"], dtype=np.float64).reshape(-1)
    conditioning_h = np.asarray(metadata["conditioning_H"], dtype=np.float64).reshape(-1)
    return {
        "condition_set": condition_set_from_metadata_arrays(metadata),
        "seed_policy": seed_policy_from_metadata_arrays(metadata),
        "domain": str(np.asarray(metadata["population_domain"]).item()),
        "split": str(np.asarray(metadata["population_split"]).item()),
        "domain_key": str(np.asarray(metadata["population_domain_key"]).item()),
        "resolution": int(np.asarray(metadata["resolution"], dtype=np.int64).item()),
        "pixel_size": float(np.asarray(metadata["pixel_size"], dtype=np.float64).item()),
        "n_realizations": int(np.asarray(metadata["n_realizations"], dtype=np.int64).item()),
        "sample_indices": np.asarray(metadata["sample_indices"], dtype=np.int64).reshape(-1),
        "target_labels": target_labels,
        "target_time_indices": target_time_indices.astype(np.int64).tolist(),
        "target_conditioning_time_indices": target_conditioning_time_indices.astype(np.int64).tolist(),
        "target_rollout_positions": target_rollout_positions.astype(np.int64).tolist(),
        "target_H": target_h.astype(float).tolist(),
        "conditioning_H": conditioning_h.astype(float).tolist(),
        "candidate_tiers": np.asarray(metadata["candidate_tiers"], dtype=np.int64).reshape(-1).astype(int).tolist(),
        "coarse_relative_eps": float(np.asarray(metadata["coarse_relative_eps"], dtype=np.float64).item()),
        "transfer_ridge_lambda": float(np.asarray(metadata["transfer_ridge_lambda"], dtype=np.float64).item()),
        "transfer_operator": str(np.asarray(metadata["transfer_operator"]).item()),
    }


def _shared_conditioning_contract(target_specs: list[dict[str, Any]]) -> tuple[int, float]:
    conditioning_tidx = int(target_specs[0]["conditioning_time_index"])
    conditioning_h = float(target_specs[0]["H_condition"])
    for spec in target_specs:
        if int(spec["conditioning_time_index"]) != conditioning_tidx or not np.isclose(
            float(spec["H_condition"]), conditioning_h
        ):
            raise ValueError(
                "Population metrics require target specs with the same conditioning_time_index and H_condition."
            )
    return conditioning_tidx, conditioning_h


def _append_pdf_values(
    segments: list[np.ndarray],
    offsets: np.ndarray,
    index: tuple[int, ...],
    values: np.ndarray,
    cursor: int,
) -> int:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    offsets[index + (0,)] = int(cursor)
    offsets[index + (1,)] = int(cursor) + int(vals.shape[0])
    segments.append(vals)
    return int(cursor) + int(vals.shape[0])


def _pdf_sample_indices(
    *,
    resolution: int,
    R_e1: np.ndarray,
    R_e2: np.ndarray,
    min_spacing_pixels: int = 4,
) -> tuple[np.ndarray, int]:
    spacing = decorrelation_spacing_from_curves(
        np.asarray(R_e1, dtype=np.float64),
        np.asarray(R_e2, dtype=np.float64),
        min_spacing_pixels=int(min_spacing_pixels),
    )
    spacing_pixels = int(min(int(spacing["spacing_pixels"]), int(resolution)))
    return subsample_grid_indices(int(resolution), int(spacing_pixels)), spacing_pixels


def _empty_pdf_values(segments: list[np.ndarray]) -> np.ndarray:
    if not segments:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(segments).astype(np.float32, copy=False)


def _per_condition_residual_stats(
    transferred_by_condition: np.ndarray,
    coarse_reference_fields: np.ndarray,
    *,
    relative_eps: float,
) -> dict[str, np.ndarray]:
    residuals = np.asarray(transferred_by_condition, dtype=np.float64) - np.asarray(
        coarse_reference_fields,
        dtype=np.float64,
    )[:, None, :]
    targets = np.asarray(coarse_reference_fields, dtype=np.float64)
    per_draw_sq = np.sum(np.square(residuals), axis=2, dtype=np.float64)
    total_sq = np.mean(per_draw_sq, axis=1)
    mean_residual = np.mean(residuals, axis=1)
    bias_sq = np.sum(np.square(mean_residual), axis=1, dtype=np.float64)
    spread_sq = np.maximum(total_sq - bias_sq, 0.0)
    target_sq = np.sum(np.square(targets), axis=1, dtype=np.float64)
    denom = np.maximum(target_sq, float(relative_eps))
    return {
        "total_sq": total_sq,
        "total_rel": total_sq / denom,
        "bias_sq": bias_sq,
        "bias_rel": bias_sq / denom,
        "spread_sq": spread_sq,
        "spread_rel": spread_sq / denom,
        "target_sq": target_sq,
    }


def _population_metrics_chunk_payload(
    *,
    split_fields_by_tidx: dict[int, np.ndarray],
    decoded_chunk: dict[str, np.ndarray],
    target_specs: list[dict[str, Any]],
    resolution: int,
    pixel_size: float,
    coarse_relative_eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    condition_indices = np.asarray(decoded_chunk["sample_indices"], dtype=np.int64).reshape(-1)
    decoded_rollout = np.asarray(decoded_chunk["decoded_rollout_fields"], dtype=np.float32)
    n_conditions = int(condition_indices.shape[0])
    n_realizations = int(decoded_rollout.shape[1])
    n_targets = int(len(target_specs))
    max_rollout_pos = max(int(spec["rollout_pos"]) for spec in target_specs)
    if int(decoded_rollout.shape[2]) <= int(max_rollout_pos):
        raise ValueError(
            "Population decoded cache does not contain enough rollout positions: "
            f"got {decoded_rollout.shape[2]}, need index {int(max_rollout_pos)}."
        )

    reference_rollout_e1 = np.zeros((n_conditions, n_targets, int(resolution)), dtype=np.float32)
    reference_rollout_e2 = np.zeros_like(reference_rollout_e1)
    reference_recoarsened_e1 = np.zeros_like(reference_rollout_e1)
    reference_recoarsened_e2 = np.zeros_like(reference_rollout_e1)
    generated_rollout_e1 = np.zeros((n_conditions, n_targets, n_realizations, int(resolution)), dtype=np.float32)
    generated_rollout_e2 = np.zeros_like(generated_rollout_e1)
    generated_recoarsened_e1 = np.zeros_like(generated_rollout_e1)
    generated_recoarsened_e2 = np.zeros_like(generated_rollout_e1)
    valid_realization_counts = np.full((n_conditions, n_targets), n_realizations, dtype=np.int64)

    coarse_total_sq = np.zeros((n_conditions, n_targets), dtype=np.float64)
    coarse_total_rel = np.zeros_like(coarse_total_sq)
    coarse_bias_sq = np.zeros_like(coarse_total_sq)
    coarse_bias_rel = np.zeros_like(coarse_total_sq)
    coarse_spread_sq = np.zeros_like(coarse_total_sq)
    coarse_spread_rel = np.zeros_like(coarse_total_sq)
    coarse_target_sq = np.zeros_like(coarse_total_sq)

    pdf_reference_rollout_segments: list[np.ndarray] = []
    pdf_reference_recoarsened_segments: list[np.ndarray] = []
    pdf_generated_rollout_segments: list[np.ndarray] = []
    pdf_generated_recoarsened_segments: list[np.ndarray] = []
    pdf_reference_rollout_offsets = np.zeros((n_conditions, n_targets, 2), dtype=np.int64)
    pdf_reference_recoarsened_offsets = np.zeros_like(pdf_reference_rollout_offsets)
    pdf_generated_rollout_offsets = np.zeros((n_conditions, n_targets, n_realizations, 2), dtype=np.int64)
    pdf_generated_recoarsened_offsets = np.zeros_like(pdf_generated_rollout_offsets)
    pdf_rollout_spacing_pixels = np.zeros((n_conditions, n_targets), dtype=np.int64)
    pdf_recoarsened_spacing_pixels = np.zeros_like(pdf_rollout_spacing_pixels)
    pdf_ref_rollout_cursor = 0
    pdf_ref_recoarsened_cursor = 0
    pdf_gen_rollout_cursor = 0
    pdf_gen_recoarsened_cursor = 0

    conditioning_tidx, _ = _shared_conditioning_contract(target_specs)
    coarse_reference_fields = np.asarray(
        decoded_chunk.get(
            "conditioning_fields",
            np.asarray(split_fields_by_tidx[int(conditioning_tidx)][condition_indices], dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(n_conditions, -1)
    for row_idx in range(n_conditions):
        coarse_e1, coarse_e2 = directional_correlation(
            coarse_reference_fields[row_idx].reshape(int(resolution), int(resolution))
        )
        reference_recoarsened_e1[row_idx, :, :] = np.asarray(coarse_e1, dtype=np.float32)
        reference_recoarsened_e2[row_idx, :, :] = np.asarray(coarse_e2, dtype=np.float32)

    for target_idx, spec in enumerate(target_specs):
        target_time_index = int(spec["time_index"])
        reference_fields = np.asarray(
            split_fields_by_tidx[target_time_index][condition_indices],
            dtype=np.float32,
        ).reshape(n_conditions, -1)
        for row_idx in range(n_conditions):
            ref_e1, ref_e2 = directional_correlation(
                reference_fields[row_idx].reshape(int(resolution), int(resolution))
            )
            reference_rollout_e1[row_idx, target_idx, :] = np.asarray(ref_e1, dtype=np.float32)
            reference_rollout_e2[row_idx, target_idx, :] = np.asarray(ref_e2, dtype=np.float32)

        generated_fields = np.asarray(decoded_rollout[:, :, int(spec["rollout_pos"]), :], dtype=np.float32).reshape(
            n_conditions * n_realizations,
            -1,
        )
        for batch_idx, field in enumerate(generated_fields):
            cond_idx = int(batch_idx // n_realizations)
            real_idx = int(batch_idx % n_realizations)
            gen_e1, gen_e2 = directional_correlation(field.reshape(int(resolution), int(resolution)))
            generated_rollout_e1[cond_idx, target_idx, real_idx, :] = np.asarray(gen_e1, dtype=np.float32)
            generated_rollout_e2[cond_idx, target_idx, real_idx, :] = np.asarray(gen_e2, dtype=np.float32)

        transferred_fields = np.asarray(
            recoarsen_fields_to_scale(
                generated_fields,
                resolution=int(resolution),
                source_H=float(spec["H_target"]),
                target_H=float(spec["H_condition"]),
                pixel_size=float(pixel_size),
                ridge_lambda=float(ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA),
            ),
            dtype=np.float32,
        )
        transferred_by_condition = transferred_fields.reshape(n_conditions, n_realizations, -1)
        stats = _per_condition_residual_stats(
            transferred_by_condition,
            coarse_reference_fields,
            relative_eps=float(coarse_relative_eps),
        )
        coarse_total_sq[:, target_idx] = stats["total_sq"]
        coarse_total_rel[:, target_idx] = stats["total_rel"]
        coarse_bias_sq[:, target_idx] = stats["bias_sq"]
        coarse_bias_rel[:, target_idx] = stats["bias_rel"]
        coarse_spread_sq[:, target_idx] = stats["spread_sq"]
        coarse_spread_rel[:, target_idx] = stats["spread_rel"]
        coarse_target_sq[:, target_idx] = stats["target_sq"]

        for batch_idx, field in enumerate(transferred_fields):
            cond_idx = int(batch_idx // n_realizations)
            real_idx = int(batch_idx % n_realizations)
            gen_e1, gen_e2 = directional_correlation(field.reshape(int(resolution), int(resolution)))
            generated_recoarsened_e1[cond_idx, target_idx, real_idx, :] = np.asarray(gen_e1, dtype=np.float32)
            generated_recoarsened_e2[cond_idx, target_idx, real_idx, :] = np.asarray(gen_e2, dtype=np.float32)

        generated_by_condition = generated_fields.reshape(n_conditions, n_realizations, -1)
        for row_idx in range(n_conditions):
            rollout_idx, rollout_spacing = _pdf_sample_indices(
                resolution=int(resolution),
                R_e1=reference_rollout_e1[row_idx, target_idx],
                R_e2=reference_rollout_e2[row_idx, target_idx],
            )
            pdf_rollout_spacing_pixels[row_idx, target_idx] = int(rollout_spacing)
            pdf_ref_rollout_cursor = _append_pdf_values(
                pdf_reference_rollout_segments,
                pdf_reference_rollout_offsets,
                (row_idx, target_idx),
                reference_fields[row_idx, rollout_idx],
                pdf_ref_rollout_cursor,
            )
            for real_idx in range(n_realizations):
                pdf_gen_rollout_cursor = _append_pdf_values(
                    pdf_generated_rollout_segments,
                    pdf_generated_rollout_offsets,
                    (row_idx, target_idx, real_idx),
                    generated_by_condition[row_idx, real_idx, rollout_idx],
                    pdf_gen_rollout_cursor,
                )

            recoarsened_idx, recoarsened_spacing = _pdf_sample_indices(
                resolution=int(resolution),
                R_e1=reference_recoarsened_e1[row_idx, target_idx],
                R_e2=reference_recoarsened_e2[row_idx, target_idx],
            )
            pdf_recoarsened_spacing_pixels[row_idx, target_idx] = int(recoarsened_spacing)
            pdf_ref_recoarsened_cursor = _append_pdf_values(
                pdf_reference_recoarsened_segments,
                pdf_reference_recoarsened_offsets,
                (row_idx, target_idx),
                coarse_reference_fields[row_idx, recoarsened_idx],
                pdf_ref_recoarsened_cursor,
            )
            for real_idx in range(n_realizations):
                pdf_gen_recoarsened_cursor = _append_pdf_values(
                    pdf_generated_recoarsened_segments,
                    pdf_generated_recoarsened_offsets,
                    (row_idx, target_idx, real_idx),
                    transferred_by_condition[row_idx, real_idx, recoarsened_idx],
                    pdf_gen_recoarsened_cursor,
                )

    return {
        "sample_indices": condition_indices.astype(np.int64),
        "reference_rollout_e1": reference_rollout_e1,
        "reference_rollout_e2": reference_rollout_e2,
        "reference_recoarsened_e1": reference_recoarsened_e1,
        "reference_recoarsened_e2": reference_recoarsened_e2,
        "generated_rollout_e1": generated_rollout_e1,
        "generated_rollout_e2": generated_rollout_e2,
        "generated_recoarsened_e1": generated_recoarsened_e1,
        "generated_recoarsened_e2": generated_recoarsened_e2,
        "valid_realization_counts": valid_realization_counts,
        "coarse_total_sq": coarse_total_sq,
        "coarse_total_rel": coarse_total_rel,
        "coarse_bias_sq": coarse_bias_sq,
        "coarse_bias_rel": coarse_bias_rel,
        "coarse_spread_sq": coarse_spread_sq,
        "coarse_spread_rel": coarse_spread_rel,
        "coarse_target_sq": coarse_target_sq,
        "pdf_reference_rollout_values": _empty_pdf_values(pdf_reference_rollout_segments),
        "pdf_reference_rollout_offsets": pdf_reference_rollout_offsets,
        "pdf_generated_rollout_values": _empty_pdf_values(pdf_generated_rollout_segments),
        "pdf_generated_rollout_offsets": pdf_generated_rollout_offsets,
        "pdf_reference_recoarsened_values": _empty_pdf_values(pdf_reference_recoarsened_segments),
        "pdf_reference_recoarsened_offsets": pdf_reference_recoarsened_offsets,
        "pdf_generated_recoarsened_values": _empty_pdf_values(pdf_generated_recoarsened_segments),
        "pdf_generated_recoarsened_offsets": pdf_generated_recoarsened_offsets,
        "pdf_rollout_spacing_pixels": pdf_rollout_spacing_pixels,
        "pdf_recoarsened_spacing_pixels": pdf_recoarsened_spacing_pixels,
    }


def store_population_domain_metrics_cache(
    *,
    runtime,
    invocation: dict[str, Any],
    seed_policy: dict[str, int],
    domain_spec: dict[str, Any],
    target_specs: list[dict[str, Any]],
    split_fields_by_tidx: dict[int, np.ndarray],
    resolution: int,
    pixel_size: float,
    n_realizations: int,
    coarse_relative_eps: float = 1e-8,
) -> dict[str, Any]:
    decoded_manifest = population_decoded_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        resolution=int(resolution),
    )
    decoded_dir = population_decoded_store_dir(invocation["output_dir"], domain_key=str(domain_spec["domain_key"]))
    decoded_metadata = load_population_decoded_metadata(decoded_dir, expected_manifest=decoded_manifest)
    store_dir = population_store_dir(invocation["output_dir"], domain_key=str(domain_spec["domain_key"]))
    manifest = population_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        target_specs=target_specs,
        resolution=int(resolution),
        pixel_size=float(pixel_size),
        n_realizations=int(n_realizations),
        coarse_relative_eps=float(coarse_relative_eps),
    )
    store = prepare_resumable_store(store_dir, expected_manifest=manifest)
    sample_indices = np.asarray(decoded_metadata["sample_indices"], dtype=np.int64)
    if not store.has_chunk("metadata"):
        store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(domain_spec["condition_set"]),
                **seed_policy_to_metadata_arrays(seed_policy),
                "population_domain": np.asarray(str(domain_spec["domain"])),
                "population_split": np.asarray(str(domain_spec["split"])),
                "population_domain_key": np.asarray(str(domain_spec["domain_key"])),
                "resolution": np.asarray(int(resolution), dtype=np.int64),
                "pixel_size": np.asarray(float(pixel_size), dtype=np.float64),
                "n_realizations": np.asarray(int(n_realizations), dtype=np.int64),
                "sample_indices": sample_indices,
                "target_labels": np.asarray([str(spec["label"]) for spec in target_specs], dtype=np.str_),
                "target_time_indices": np.asarray([int(spec["time_index"]) for spec in target_specs], dtype=np.int64),
                "target_conditioning_time_indices": np.asarray(
                    [int(spec["conditioning_time_index"]) for spec in target_specs],
                    dtype=np.int64,
                ),
                "target_rollout_positions": np.asarray(
                    [int(spec["rollout_pos"]) for spec in target_specs],
                    dtype=np.int64,
                ),
                "target_H": np.asarray([float(spec["H_target"]) for spec in target_specs], dtype=np.float64),
                "conditioning_H": np.asarray([float(spec["H_condition"]) for spec in target_specs], dtype=np.float64),
                "candidate_tiers": np.asarray(domain_spec["candidate_tiers"], dtype=np.int64),
                "coarse_relative_eps": np.asarray(float(coarse_relative_eps), dtype=np.float64),
                "transfer_operator": np.asarray("tran_periodic_tikhonov_transfer"),
                "transfer_ridge_lambda": np.asarray(
                    float(ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA),
                    dtype=np.float64,
                ),
            },
        )
    for chunk_start, chunk_name, decoded_chunk in iter_population_decoded_chunks(decoded_dir, expected_manifest=decoded_manifest):
        if store.has_chunk(chunk_name):
            continue
        store.write_chunk(
            chunk_name,
            _population_metrics_chunk_payload(
                split_fields_by_tidx=split_fields_by_tidx,
                decoded_chunk=decoded_chunk,
                target_specs=target_specs,
                resolution=int(resolution),
                pixel_size=float(pixel_size),
                coarse_relative_eps=float(coarse_relative_eps),
            ),
            metadata={
                "chunk_start": int(chunk_start),
                "chunk_count": int(np.asarray(decoded_chunk["sample_indices"], dtype=np.int64).shape[0]),
            },
        )
    store.mark_complete(
        status_updates={
            "n_conditions": int(sample_indices.shape[0]),
            "n_realizations": int(n_realizations),
        }
    )
    return {
        "store_dir": str(store.store_dir),
        "manifest": manifest,
        "condition_set": domain_spec["condition_set"],
        "sample_indices": sample_indices.astype(np.int64),
    }


def load_population_domain_metrics(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
    include_pdf_samples: bool = False,
    include_coarse_stats: bool = False,
) -> dict[str, Any]:
    manifest = load_store_manifest(Path(store_dir))
    if manifest is None:
        raise FileNotFoundError(f"Missing population metrics store manifest: {store_dir}.")
    if expected_manifest is not None and manifest != expected_manifest:
        raise FileNotFoundError(
            "Population metrics store manifest does not match the current request. "
            f"Rerun --phases {POPULATION_METRICS_CACHE_PHASE} for {store_dir}."
        )
    store = prepare_resumable_store(
        Path(store_dir),
        expected_manifest=manifest if expected_manifest is None else expected_manifest,
    )
    if not store.complete_path.exists():
        raise FileNotFoundError(f"Population metrics store is incomplete: {store_dir}.")
    metadata = _load_population_metadata(store)
    array_keys = (
        "reference_rollout_e1",
        "reference_rollout_e2",
        "reference_recoarsened_e1",
        "reference_recoarsened_e2",
        "generated_rollout_e1",
        "generated_rollout_e2",
        "generated_recoarsened_e1",
        "generated_recoarsened_e2",
        "valid_realization_counts",
    )
    coarse_keys = (
        "coarse_total_sq",
        "coarse_total_rel",
        "coarse_bias_sq",
        "coarse_bias_rel",
        "coarse_spread_sq",
        "coarse_spread_rel",
        "coarse_target_sq",
    )
    arrays: dict[str, list[np.ndarray]] = {"sample_indices": []}
    arrays.update({key: [] for key in array_keys})
    if include_coarse_stats:
        arrays.update({key: [] for key in coarse_keys})
    pdf_value_keys = (
        "pdf_reference_rollout_values",
        "pdf_generated_rollout_values",
        "pdf_reference_recoarsened_values",
        "pdf_generated_recoarsened_values",
    )
    pdf_offset_keys = (
        "pdf_reference_rollout_offsets",
        "pdf_generated_rollout_offsets",
        "pdf_reference_recoarsened_offsets",
        "pdf_generated_recoarsened_offsets",
    )
    pdf_spacing_keys = ("pdf_rollout_spacing_pixels", "pdf_recoarsened_spacing_pixels")
    pdf_values: dict[str, list[np.ndarray]] = {key: [] for key in pdf_value_keys}
    pdf_offsets: dict[str, list[np.ndarray]] = {key: [] for key in pdf_offset_keys}
    pdf_spacings: dict[str, list[np.ndarray]] = {key: [] for key in pdf_spacing_keys}
    pdf_cursors = {key: 0 for key in pdf_value_keys}
    offset_value_key = {
        "pdf_reference_rollout_offsets": "pdf_reference_rollout_values",
        "pdf_generated_rollout_offsets": "pdf_generated_rollout_values",
        "pdf_reference_recoarsened_offsets": "pdf_reference_recoarsened_values",
        "pdf_generated_recoarsened_offsets": "pdf_generated_recoarsened_values",
    }
    expected_start = 0
    for chunk_start, chunk_name in _chunk_records(store):
        if int(chunk_start) != int(expected_start):
            raise FileNotFoundError(
                f"Missing population metrics chunk in {store_dir} at conditions [{expected_start}, {chunk_start})."
            )
        chunk = store.load_chunk(chunk_name)
        arrays["sample_indices"].append(np.asarray(chunk["sample_indices"], dtype=np.int64))
        for key in array_keys:
            arrays[key].append(np.asarray(chunk[key]))
        if include_coarse_stats:
            missing_coarse = [key for key in coarse_keys if key not in chunk]
            if missing_coarse:
                raise FileNotFoundError(
                    "Population metrics cache is missing root coarse-consistency statistics. "
                    f"Rerun --phases {POPULATION_METRICS_CACHE_PHASE} for {store_dir}."
                )
            for key in coarse_keys:
                arrays[key].append(np.asarray(chunk[key], dtype=np.float64))
        if include_pdf_samples:
            missing = [
                key
                for key in (*pdf_value_keys, *pdf_offset_keys, *pdf_spacing_keys)
                if key not in chunk
            ]
            if missing:
                raise FileNotFoundError(
                    "Population metrics cache is missing one-point PDF samples. "
                    f"Rerun --phases {POPULATION_METRICS_CACHE_PHASE} for {store_dir}."
                )
            for key in pdf_value_keys:
                values = np.asarray(chunk[key], dtype=np.float32).reshape(-1)
                pdf_values[key].append(values)
            for key in pdf_offset_keys:
                offsets = np.asarray(chunk[key], dtype=np.int64).copy()
                value_key = offset_value_key[key]
                offsets += int(pdf_cursors[value_key])
                pdf_offsets[key].append(offsets)
            for key in pdf_spacing_keys:
                pdf_spacings[key].append(np.asarray(chunk[key], dtype=np.int64))
            for key in pdf_value_keys:
                pdf_cursors[key] += int(np.asarray(chunk[key]).reshape(-1).shape[0])
        expected_start += int(np.asarray(chunk["sample_indices"], dtype=np.int64).shape[0])

    combined = {
        "metadata": metadata,
        "sample_indices": (
            np.concatenate(arrays["sample_indices"], axis=0)
            if arrays["sample_indices"]
            else np.zeros((0,), dtype=np.int64)
        ),
    }
    for key, values in arrays.items():
        if key == "sample_indices":
            continue
        combined[key] = np.concatenate(values, axis=0) if values else np.zeros((0,), dtype=np.float32)
    if include_pdf_samples:
        for key in pdf_value_keys:
            combined[key] = (
                np.concatenate(pdf_values[key], axis=0)
                if pdf_values[key]
                else np.zeros((0,), dtype=np.float32)
            )
        for key in pdf_offset_keys:
            combined[key] = (
                np.concatenate(pdf_offsets[key], axis=0)
                if pdf_offsets[key]
                else np.zeros((0,), dtype=np.int64)
            )
        for key in pdf_spacing_keys:
            combined[key] = (
                np.concatenate(pdf_spacings[key], axis=0)
                if pdf_spacings[key]
                else np.zeros((0,), dtype=np.int64)
            )
    return combined
