from __future__ import annotations

"""Token-latent decode and field-metric evaluation for kNN conditional reports."""

from pathlib import Path
from typing import Any

import numpy as np

from data.transform_utils import apply_inverse_transform
from scripts.csp.conditional_eval.field_metrics import (
    evaluate_pair_field_metrics,
    infer_pixel_size_from_grid,
    sample_reference_latent_draws,
)
from scripts.csp.token_decode_runtime import (
    AUTO_COARSE_DECODE_BATCH_CAP,
    resolve_decode_point_batch_size,
    resolve_decode_sample_batch_size,
    resolve_requested_jax_device,
)


def decode_token_latents_to_fields(
    latents: np.ndarray,
    *,
    dataset_path: Path,
    fae_checkpoint_path: Path,
    load_token_fae_decode_context_fn,
    decode_token_latent_batch_fn,
    requested_device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    requested = str(requested_device)
    initial_device_kind, decode_device = resolve_requested_jax_device(requested, auto_preference="gpu")
    first_context = load_token_fae_decode_context_fn(
        dataset_path=Path(dataset_path),
        fae_checkpoint_path=Path(fae_checkpoint_path),
        decode_device=decode_device,
        decode_device_kind=initial_device_kind,
        jit_decode=bool(initial_device_kind == "gpu"),
    )

    def decode_context_factory(device_kind: str, jit_decode: bool):
        resolved_device_kind, resolved_device = resolve_requested_jax_device(
            str(device_kind),
            auto_preference=str(device_kind),
        )
        return load_token_fae_decode_context_fn(
            dataset_path=Path(dataset_path),
            fae_checkpoint_path=Path(fae_checkpoint_path),
            decode_device=resolved_device,
            decode_device_kind=resolved_device_kind,
            jit_decode=bool(jit_decode),
        )

    lat_arr = np.asarray(latents, dtype=np.float32)
    result = decode_token_latent_batch_fn(
        latents=lat_arr,
        decode_context_factory=decode_context_factory,
        grid_coords=np.asarray(first_context.grid_coords, dtype=np.float32),
        requested_device=requested,
        auto_preference="gpu",
        sample_batch_size=resolve_decode_sample_batch_size(
            int(lat_arr.shape[0]),
            requested_device=requested,
            auto_cap=int(AUTO_COARSE_DECODE_BATCH_CAP),
        ),
        point_batch_size=resolve_decode_point_batch_size(None, grid_size=int(first_context.grid_coords.shape[0])),
        stage_label="knn_reference_field_metrics_token_decode",
        clip_bounds=first_context.clip_bounds,
    )
    decoded_fields = np.asarray(
        apply_inverse_transform(
            np.asarray(result["fields_log"], dtype=np.float32),
            getattr(first_context, "transform_info"),
        ),
        dtype=np.float32,
    )
    return decoded_fields, {
        "resolution": int(first_context.resolution),
        "grid_coords": np.asarray(first_context.grid_coords, dtype=np.float32),
        "resolved_device": str(result["resolved_device"]),
        "jit_decode": bool(result["jit_decode"]),
        "sample_batch_size": int(result["sample_batch_size"]),
        "point_batch_size": int(result["point_batch_size"]),
    }


def compute_token_pair_field_metrics(
    *,
    args,
    pair_label: str,
    pair_meta: dict[str, object],
    pair_sample_payload: dict[str, object],
    test_sample_indices: np.ndarray,
    corpus_z_fine: np.ndarray,
    reference_sampling_seed: int,
    representative_seed: int,
    output_dir: Path,
    dataset_path: Path,
    fae_checkpoint_path: Path,
    load_token_fae_decode_context_fn,
    decode_token_latent_batch_fn,
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
    requested_device = "cpu" if bool(getattr(args, "nogpu", False)) else "auto"
    generated_fields_flat, decode_meta = decode_token_latents_to_fields(
        generated_latents.reshape(-1, *generated_latents.shape[2:]),
        dataset_path=Path(dataset_path),
        fae_checkpoint_path=Path(fae_checkpoint_path),
        load_token_fae_decode_context_fn=load_token_fae_decode_context_fn,
        decode_token_latent_batch_fn=decode_token_latent_batch_fn,
        requested_device=requested_device,
    )
    reference_fields_flat, _reference_decode_meta = decode_token_latents_to_fields(
        reference_latents.reshape(-1, *reference_latents.shape[2:]),
        dataset_path=Path(dataset_path),
        fae_checkpoint_path=Path(fae_checkpoint_path),
        load_token_fae_decode_context_fn=load_token_fae_decode_context_fn,
        decode_token_latent_batch_fn=decode_token_latent_batch_fn,
        requested_device=requested_device,
    )
    generated_fields = generated_fields_flat.reshape(generated_latents.shape[0], generated_latents.shape[1], -1)
    reference_fields = reference_fields_flat.reshape(reference_latents.shape[0], reference_latents.shape[1], -1)
    field_metrics, figure_manifest = evaluate_pair_field_metrics(
        pair_label=pair_label,
        pair_display_label=str(pair_meta["display_label"]),
        pair_h_value=float(pair_meta["H_fine"]),
        test_sample_indices=np.asarray(test_sample_indices, dtype=np.int64),
        conditions=np.asarray(pair_sample_payload["latent_ecmmd_conditions"], dtype=np.float32),
        reference_fields=reference_fields,
        generated_fields=generated_fields,
        resolution=int(decode_meta["resolution"]),
        pixel_size=float(
            infer_pixel_size_from_grid(
                grid_coords=np.asarray(decode_meta["grid_coords"], dtype=np.float32),
                resolution=int(decode_meta["resolution"]),
            )
        ),
        min_spacing_pixels=4,
        representative_seed=int(representative_seed),
        n_plot_conditions=int(max(0, getattr(args, "n_plot_conditions", 0))),
        output_dir=output_dir,
    )
    return field_metrics, figure_manifest
