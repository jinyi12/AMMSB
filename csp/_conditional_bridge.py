from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp


BridgeConditionMode = Literal["coarse_only", "previous_state", "global_and_previous"]
BRIDGE_CONDITION_MODES = ("coarse_only", "previous_state", "global_and_previous")


def validate_bridge_condition_mode(condition_mode: str) -> BridgeConditionMode:
    mode = str(condition_mode)
    if mode not in BRIDGE_CONDITION_MODES:
        raise ValueError(
            f"condition_mode must be one of {BRIDGE_CONDITION_MODES}, got {condition_mode!r}."
        )
    return mode  # type: ignore[return-value]


def bridge_condition_uses_global_state(condition_mode: str) -> bool:
    mode = validate_bridge_condition_mode(condition_mode)
    return mode in {"coarse_only", "global_and_previous"}


def bridge_condition_dim(
    latent_dim: int,
    num_intervals: int,
    condition_mode: str,
    *,
    include_interval_embedding: bool = True,
) -> int:
    latent_dim_int = int(latent_dim)
    num_intervals_int = int(num_intervals)
    mode = validate_bridge_condition_mode(condition_mode)
    if latent_dim_int <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if num_intervals_int <= 0:
        raise ValueError(f"num_intervals must be positive, got {num_intervals}.")

    if mode in {"coarse_only", "previous_state"}:
        dim = latent_dim_int
    else:
        dim = 2 * latent_dim_int

    if include_interval_embedding:
        dim += num_intervals_int
    return int(dim)


def local_interval_time(
    t: jax.Array | float,
    t_start: jax.Array | float,
    t_end: jax.Array | float,
) -> jax.Array:
    t_arr = jnp.asarray(t)
    t_start_arr = jnp.asarray(t_start, dtype=t_arr.dtype)
    t_end_arr = jnp.asarray(t_end, dtype=t_arr.dtype)
    denom = jnp.maximum(t_end_arr - t_start_arr, jnp.asarray(1e-12, dtype=t_arr.dtype))
    return (t_arr - t_start_arr) / denom


def sample_brownian_bridge_state(
    x_start: jax.Array,
    x_end: jax.Array,
    t: jax.Array | float,
    t_start: jax.Array | float,
    t_end: jax.Array | float,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    """Sample an exact Brownian bridge state between two endpoints."""
    x_start_arr = jnp.asarray(x_start)
    x_end_arr = jnp.asarray(x_end, dtype=x_start_arr.dtype)
    t_arr = jnp.asarray(t, dtype=x_start_arr.dtype)
    t_start_arr = jnp.asarray(t_start, dtype=x_start_arr.dtype)
    t_end_arr = jnp.asarray(t_end, dtype=x_start_arr.dtype)
    sigma_arr = jnp.asarray(sigma, dtype=x_start_arr.dtype)

    alpha = (t_end_arr - t_arr) / (t_end_arr - t_start_arr)
    beta = (t_arr - t_start_arr) / (t_end_arr - t_start_arr)
    mean_t = alpha * x_start_arr + beta * x_end_arr
    var_t = jnp.maximum(
        sigma_arr * sigma_arr * (t_arr - t_start_arr) * (t_end_arr - t_arr) / (t_end_arr - t_start_arr),
        jnp.asarray(0.0, dtype=x_start_arr.dtype),
    )
    noise = jax.random.normal(key, x_start_arr.shape, dtype=x_start_arr.dtype)
    return mean_t + jnp.sqrt(var_t) * noise


def brownian_bridge_target(
    x_t: jax.Array,
    x_end: jax.Array,
    t: jax.Array | float,
    t_end: jax.Array | float,
) -> jax.Array:
    """Analytical Brownian bridge drift target."""
    x_t_arr = jnp.asarray(x_t)
    return (jnp.asarray(x_end, dtype=x_t_arr.dtype) - x_t_arr) / (
        jnp.asarray(t_end, dtype=x_t_arr.dtype) - jnp.asarray(t, dtype=x_t_arr.dtype)
    )


def sample_truncated_interval_time(
    *,
    time_key: jax.Array,
    batch_size: int,
    dtype: jnp.dtype,
    t_start: jax.Array,
    t_end: jax.Array,
    endpoint_epsilon: float,
    state_rank: int,
) -> jax.Array:
    """Sample interval time away from the endpoints and broadcast to a state rank."""
    interval_length = jnp.asarray(t_end - t_start, dtype=dtype)
    epsilon = jnp.minimum(
        jnp.asarray(max(float(endpoint_epsilon), 0.0), dtype=dtype),
        0.5 * interval_length,
    )
    span = interval_length - 2.0 * epsilon
    midpoint = jnp.asarray(t_start, dtype=dtype) + 0.5 * interval_length
    u = jax.random.uniform(time_key, (int(batch_size), 1), dtype=dtype)
    truncated = jnp.asarray(t_start, dtype=dtype) + epsilon + u * span
    sampled = jnp.where(span > 0.0, truncated, midpoint)
    return sampled.reshape((int(batch_size),) + (1,) * max(int(state_rank), 0))


def make_bridge_condition(
    global_state: jax.Array,
    previous_state: jax.Array,
    *,
    interval_idx: int | jax.Array,
    num_intervals: int,
    condition_mode: str,
    include_interval_embedding: bool = True,
) -> jax.Array:
    previous_arr = jnp.asarray(previous_state)
    global_arr = jnp.asarray(global_state, dtype=previous_arr.dtype)
    mode = validate_bridge_condition_mode(condition_mode)
    num_intervals_int = int(num_intervals)

    if previous_arr.shape[-1] != global_arr.shape[-1]:
        raise ValueError(
            "global_state and previous_state must agree on latent dimension, "
            f"got {global_arr.shape} and {previous_arr.shape}."
        )

    if mode == "coarse_only":
        parts = [global_arr]
    elif mode == "previous_state":
        parts = [previous_arr]
    else:
        parts = [global_arr, previous_arr]

    if include_interval_embedding:
        interval_arr = jnp.asarray(interval_idx, dtype=jnp.int32)
        interval_embed = jax.nn.one_hot(interval_arr, num_intervals_int, dtype=previous_arr.dtype)
        interval_embed = jnp.broadcast_to(interval_embed, previous_arr.shape[:-1] + (num_intervals_int,))
        parts.append(interval_embed)

    return jnp.concatenate(parts, axis=-1)


def validate_bridge_condition_dim(
    drift_net: object,
    *,
    latent_dim: int,
    num_intervals: int,
    condition_mode: str,
    include_interval_embedding: bool = True,
) -> None:
    if not hasattr(drift_net, "condition_dim"):
        return

    expected = bridge_condition_dim(
        latent_dim,
        num_intervals,
        condition_mode,
        include_interval_embedding=include_interval_embedding,
    )
    actual = int(getattr(drift_net, "condition_dim"))
    if actual != expected:
        raise ValueError(
            "Drift condition_dim does not match the requested sequential conditioning setup: "
            f"expected {expected}, got {actual}."
        )
