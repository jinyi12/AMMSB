from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def validate_level_data(
    level_data: jax.Array,
    zt: jax.Array,
    *,
    ndim: int,
    shape_spec: str,
    name: str = "latent_data",
    context: str = "bridge matching",
) -> tuple[int, ...]:
    """Validate a multilevel latent archive aligned with stored fine-to-coarse `zt`."""
    level_shape = tuple(int(dim) for dim in level_data.shape)
    if len(level_shape) != int(ndim):
        raise ValueError(f"{name} must have shape {shape_spec}, got {level_shape}.")

    validate_data_zt(zt)
    zt_np = np.asarray(zt)
    if level_shape[0] != zt_np.shape[0]:
        raise ValueError(f"{name} and zt disagree on T: {level_shape[0]} versus {zt_np.shape[0]}.")
    if level_shape[0] < 2:
        raise ValueError(f"Need at least two scale levels for {context}.")

    try:
        level_np = np.asarray(level_data)
    except Exception:
        return level_shape
    if not np.all(np.isfinite(level_np)):
        raise ValueError(f"{name} contains non-finite values.")
    return level_shape


def validate_data_zt(zt: jax.Array) -> None:
    """Validate the stored data-time grid aligned with fine-to-coarse level order."""
    zt_np = np.asarray(zt)
    if zt_np.ndim != 1:
        raise ValueError(f"zt must be 1-D, got {zt_np.shape}.")
    if zt_np.shape[0] < 2:
        raise ValueError("Need at least two zt knots.")
    if not np.all(np.isfinite(zt_np)):
        raise ValueError("zt contains non-finite values.")
    if not np.all(np.diff(zt_np) > 0.0):
        raise ValueError(
            "zt must be strictly increasing in stored data order: "
            "zt[0]=finest level, zt[-1]=coarsest level."
        )


def reverse_level_order(level_data: jax.Array) -> jax.Array:
    """Flip the leading level axis between stored fine-to-coarse and generation coarse-to-fine views."""
    return jnp.asarray(level_data)[::-1]


def generation_zt_from_data_zt(zt: jax.Array) -> jax.Array:
    """Map stored fine-to-coarse zt to an increasing coarse-to-fine generation grid."""
    validate_data_zt(zt)
    zt_arr = jnp.asarray(zt, dtype=jnp.float32)
    return zt_arr[-1] - zt_arr[::-1]


def generation_view_from_data(
    level_data: jax.Array,
    zt: jax.Array,
    *,
    ndim: int,
    shape_spec: str,
    name: str = "latent_data",
    context: str = "bridge matching",
) -> tuple[jax.Array, jax.Array]:
    """Convert stored fine-to-coarse level data into coarse-to-fine generation order."""
    validate_level_data(
        level_data,
        zt,
        ndim=ndim,
        shape_spec=shape_spec,
        name=name,
        context=context,
    )
    level_jax = jnp.asarray(level_data, dtype=jnp.float32)
    return reverse_level_order(level_jax), generation_zt_from_data_zt(zt)


def select_level_batch(level_data: jax.Array, batch_size: int | None, key: jax.Array) -> jax.Array:
    """Select a batch along the sample axis of a `(T, N, ...)` latent archive."""
    n_samples = int(level_data.shape[1])
    if batch_size is None or int(batch_size) >= n_samples:
        return level_data
    indices = jax.random.choice(key, n_samples, shape=(int(batch_size),), replace=False)
    return jnp.take(level_data, indices, axis=1)
