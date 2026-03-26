"""Helpers for vector and token-sequence latent tensors."""

from __future__ import annotations

import jax.numpy as jnp


def squared_l2_per_sample(latents: jnp.ndarray) -> jnp.ndarray:
    """Return the squared L2 norm of each sample across all non-batch axes."""
    if latents.ndim < 2:
        raise ValueError(
            f"Latents must include a batch axis and at least one latent axis. Got shape {latents.shape}."
        )
    latent_axes = tuple(range(1, latents.ndim))
    return jnp.sum(jnp.square(latents), axis=latent_axes)
