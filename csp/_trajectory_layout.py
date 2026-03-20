from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


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
