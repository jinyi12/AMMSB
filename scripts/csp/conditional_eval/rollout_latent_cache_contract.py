from __future__ import annotations

from typing import Any

import numpy as np


ROLLOUT_LATENT_KNOTS_KEY = "sampled_rollout_latents"
ROLLOUT_LATENT_DENSE_KEY = "sampled_rollout_dense_latents"
ROLLOUT_DENSE_TIME_COORDINATES_KEY = "rollout_dense_time_coordinates"
ROLLOUT_DENSE_TIME_SEMANTICS_KEY = "rollout_dense_time_semantics"


def rollout_dense_metadata_from_chunk(
    metadata_chunk: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if ROLLOUT_DENSE_TIME_COORDINATES_KEY in metadata_chunk:
        payload[ROLLOUT_DENSE_TIME_COORDINATES_KEY] = np.asarray(
            metadata_chunk[ROLLOUT_DENSE_TIME_COORDINATES_KEY],
            dtype=np.float32,
        ).reshape(-1)
    if ROLLOUT_DENSE_TIME_SEMANTICS_KEY in metadata_chunk:
        payload[ROLLOUT_DENSE_TIME_SEMANTICS_KEY] = str(
            np.asarray(metadata_chunk[ROLLOUT_DENSE_TIME_SEMANTICS_KEY]).item()
        )
    return payload


def sample_rollout_latent_chunk_payload(
    *,
    runtime: Any,
    test_sample_indices: np.ndarray,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
) -> dict[str, np.ndarray]:
    knots = np.asarray(
        runtime.sample_full_rollout_knots(
            np.asarray(test_sample_indices, dtype=np.int64),
            int(n_realizations),
            int(seed),
            drift_clip_norm,
        ),
        dtype=np.float32,
    )
    return {ROLLOUT_LATENT_KNOTS_KEY: knots}
