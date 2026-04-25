"""Latent encoding utilities shared by maintained FAE evaluation surfaces."""

from __future__ import annotations

import numpy as np

from mmsfm.fae.fae_latent_utils import make_fae_apply_fns


def compute_latent_codes(
    autoencoder,
    params,
    batch_stats,
    fields: np.ndarray,
    coords: np.ndarray,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode time-indexed fields and return latent codes with shape ``(T, N, K)``.

    The returned latent representation matches the maintained downstream
    transport boundary used by ``make_fae_apply_fns``. Standard FAEs therefore
    return vector latents, while transformer-token FAEs return flattened token
    latents.
    """

    fields_np = np.asarray(fields, dtype=np.float32)
    coords_np = np.asarray(coords, dtype=np.float32)
    if fields_np.ndim != 4:
        raise ValueError(f"Expected fields with shape (T, N, P, C); got {fields_np.shape}.")
    if coords_np.ndim != 2:
        raise ValueError(f"Expected coords with shape (P, D); got {coords_np.shape}.")

    encode_fn, _ = make_fae_apply_fns(
        autoencoder,
        params,
        batch_stats,
        decode_mode="standard",
    )

    t_count, n_count, n_points, _channels = fields_np.shape
    if coords_np.shape[0] != n_points:
        raise ValueError(
            "Coordinate/field point-count mismatch: "
            f"coords has {coords_np.shape[0]} points but fields use {n_points}."
        )

    all_z: list[np.ndarray] = []
    for time_index in range(t_count):
        fields_at_time = fields_np[time_index]
        parts: list[np.ndarray] = []
        for start in range(0, n_count, max(int(batch_size), 1)):
            fields_batch = fields_at_time[start : start + int(batch_size)]
            coords_batch = np.broadcast_to(
                coords_np[None, ...],
                (fields_batch.shape[0], *coords_np.shape),
            )
            parts.append(np.asarray(encode_fn(fields_batch, coords_batch), dtype=np.float32))
        all_z.append(np.concatenate(parts, axis=0))

    return np.asarray(np.stack(all_z, axis=0), dtype=np.float32)
