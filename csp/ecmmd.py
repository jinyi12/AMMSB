from __future__ import annotations

import jax
import jax.numpy as jnp


def standardize(conditions: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Standardize conditioning vectors columnwise."""
    cond = jnp.asarray(conditions)
    mean = jnp.mean(cond, axis=0, keepdims=True)
    std = jnp.std(cond, axis=0, keepdims=True)
    return (cond - mean) / jnp.maximum(std, jnp.asarray(eps, dtype=cond.dtype))


def build_knn(conditions: jax.Array, k: int) -> jax.Array:
    """Build directed K-NN indices on the standardized conditions."""
    cond = standardize(conditions)
    n = int(cond.shape[0])
    if n < 2:
        raise ValueError("Need at least two conditions to build a K-NN graph.")

    k_eff = max(1, min(int(k), n - 1))
    sqdist = jnp.sum((cond[:, None, :] - cond[None, :, :]) ** 2, axis=-1)
    sqdist = sqdist.at[jnp.diag_indices(n)].set(jnp.inf)
    _, idx = jax.lax.top_k(-sqdist, k_eff)
    return idx


def rbf(x: jax.Array, y: jax.Array, bandwidth: jax.Array | float) -> jax.Array:
    """RBF kernel on the last axis."""
    sqdist = jnp.sum((x - y) ** 2, axis=-1)
    bw_sq = jnp.maximum(jnp.asarray(bandwidth, dtype=sqdist.dtype) ** 2, jnp.asarray(1e-12, dtype=sqdist.dtype))
    return jnp.exp(-sqdist / bw_sq)


def median_bandwidth(real_targets: jax.Array, generated_targets: jax.Array, eps: float = 1e-3) -> jax.Array:
    """Median heuristic on direct realization distances, clipped away from zero."""
    real = jnp.asarray(real_targets)
    generated = jnp.asarray(generated_targets)
    distances = jnp.linalg.norm(real - generated, axis=-1)
    return jnp.maximum(jnp.median(distances), jnp.asarray(eps, dtype=distances.dtype))


def ecmmd_loss(
    conditions: jax.Array,
    real_targets: jax.Array,
    generated_targets: jax.Array,
    *,
    k_neighbors: int = 5,
) -> jax.Array:
    """Compute the intervalwise ECMMD^2 estimator on a minibatch."""
    cond = jnp.asarray(conditions)
    real = jnp.asarray(real_targets)
    generated = jnp.asarray(generated_targets)

    if cond.ndim != 2 or real.ndim != 2 or generated.ndim != 2:
        raise ValueError("conditions, real_targets, and generated_targets must all have shape (batch, dim).")
    if cond.shape[0] != real.shape[0] or cond.shape[0] != generated.shape[0]:
        raise ValueError("conditions, real_targets, and generated_targets must share the same batch size.")
    if real.shape[1] != generated.shape[1]:
        raise ValueError("real_targets and generated_targets must share the same response dimension.")

    nn_idx = build_knn(cond, k_neighbors)
    bandwidth = median_bandwidth(real, generated)

    y_i = real[:, None, :]
    y_j = real[nn_idx]
    z_i = generated[:, None, :]
    z_j = generated[nn_idx]
    h = (
        rbf(y_i, y_j, bandwidth)
        + rbf(z_i, z_j, bandwidth)
        - rbf(y_i, z_j, bandwidth)
        - rbf(z_i, y_j, bandwidth)
    )
    return jnp.mean(h)
