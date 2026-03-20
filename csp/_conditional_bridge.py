from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp


BridgeConditionMode = Literal["previous_state", "global_and_previous"]
BRIDGE_CONDITION_MODES = ("previous_state", "global_and_previous")


def validate_bridge_condition_mode(condition_mode: str) -> BridgeConditionMode:
    mode = str(condition_mode)
    if mode not in BRIDGE_CONDITION_MODES:
        raise ValueError(
            f"condition_mode must be one of {BRIDGE_CONDITION_MODES}, got {condition_mode!r}."
        )
    return mode  # type: ignore[return-value]


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

    if mode == "previous_state":
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

    if mode == "previous_state":
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
