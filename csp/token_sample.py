from __future__ import annotations

import numpy as np

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from ._trajectory_layout import generation_zt_from_data_zt, validate_data_zt
from .sde import SigmaFn, interval_save_times
from .token_dit import TokenConditionalDiT, integrate_token_conditional_interval, make_token_bridge_condition


def _resolve_forward_sampling_adjoint(
    adjoint: diffrax.AbstractAdjoint | None,
) -> diffrax.AbstractAdjoint:
    """Use a forward-only adjoint for rollout sampling unless the caller overrides it."""
    if adjoint is None:
        return diffrax.DirectAdjoint()
    return adjoint


def _sample_token_conditional_trajectory_impl(
    drift_net: TokenConditionalDiT,
    initial_state: jax.Array,
    global_condition: jax.Array,
    generation_zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str,
    interval_offset: int = 0,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    initial_state_arr = jnp.asarray(initial_state)
    global_condition_arr = jnp.asarray(global_condition, dtype=initial_state_arr.dtype)
    trajectory = [initial_state_arr]
    x = initial_state_arr
    rollout_num_intervals = int(len(generation_zt) - 1)
    interval_offset_int = int(interval_offset)
    for i in range(rollout_num_intervals):
        key, subkey = jax.random.split(key)
        condition = make_token_bridge_condition(
            global_condition_arr,
            x,
            interval_idx=interval_offset_int + i,
            condition_mode=condition_mode,
        )
        x = integrate_token_conditional_interval(
            drift_net,
            x,
            condition,
            generation_zt[i],
            generation_zt[i + 1],
            dt0,
            subkey,
            sigma_fn,
            adjoint=adjoint,
            time_mode="local",
        )
        trajectory.append(x)
    return jnp.stack(trajectory[::-1], axis=0)


def _sample_token_conditional_trajectory_dense_impl(
    drift_net: TokenConditionalDiT,
    initial_state: jax.Array,
    global_condition: jax.Array,
    generation_zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str,
    interval_offset: int = 0,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> tuple[jax.Array, jax.Array]:
    initial_state_arr = jnp.asarray(initial_state)
    global_condition_arr = jnp.asarray(global_condition, dtype=initial_state_arr.dtype)
    trajectory = [initial_state_arr]
    dense_segments: list[jax.Array] = []
    x = initial_state_arr
    rollout_num_intervals = int(len(generation_zt) - 1)
    interval_offset_int = int(interval_offset)
    for i in range(rollout_num_intervals):
        key, subkey = jax.random.split(key)
        condition = make_token_bridge_condition(
            global_condition_arr,
            x,
            interval_idx=interval_offset_int + i,
            condition_mode=condition_mode,
        )
        save_times = interval_save_times(generation_zt[i], generation_zt[i + 1], dt0, dtype=initial_state_arr.dtype)
        interval_path = integrate_token_conditional_interval(
            drift_net,
            x,
            condition,
            generation_zt[i],
            generation_zt[i + 1],
            dt0,
            subkey,
            sigma_fn,
            adjoint=adjoint,
            time_mode="local",
            save_times=save_times,
        )
        x = interval_path[-1]
        trajectory.append(x)
        dense_segments.append(interval_path if i == 0 else interval_path[1:])
    knots = jnp.stack(trajectory[::-1], axis=0)
    dense_path = jnp.concatenate(dense_segments, axis=0)[::-1]
    return knots, dense_path


@eqx.filter_jit
def _sample_token_conditional_batch_impl(
    drift_net: TokenConditionalDiT,
    z_batch: jax.Array,
    generation_zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str,
    global_condition_batch: jax.Array,
    interval_offset: int = 0,
    adjoint: diffrax.AbstractAdjoint,
) -> jax.Array:
    keys = jax.random.split(key, z_batch.shape[0])
    return jax.vmap(
        lambda z, global_cond, subkey: _sample_token_conditional_trajectory_impl(
            drift_net,
            z,
            global_cond,
            generation_zt,
            sigma_fn,
            dt0,
            subkey,
            condition_mode=condition_mode,
            interval_offset=interval_offset,
            adjoint=adjoint,
        )
    )(z_batch, global_condition_batch, keys)


def sample_token_conditional_trajectory(
    drift_net: TokenConditionalDiT,
    z: jax.Array,
    zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str = "global_and_previous",
    global_condition: jax.Array | None = None,
    interval_offset: int = 0,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    z_arr = jnp.asarray(z)
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(z_arr.dtype)
    global_arr = z_arr if global_condition is None else jnp.asarray(global_condition, dtype=z_arr.dtype)
    return _sample_token_conditional_trajectory_impl(
        drift_net,
        z_arr,
        global_arr,
        generation_zt,
        sigma_fn,
        dt0,
        key,
        condition_mode=condition_mode,
        interval_offset=interval_offset,
        adjoint=_resolve_forward_sampling_adjoint(adjoint),
    )


def sample_token_conditional_batch(
    drift_net: TokenConditionalDiT,
    z_batch: jax.Array,
    zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str = "global_and_previous",
    global_condition_batch: jax.Array | None = None,
    interval_offset: int = 0,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    z_batch_arr = jnp.asarray(z_batch)
    if z_batch_arr.ndim != 3:
        raise ValueError(
            f"z_batch must have shape (N, L, D), got {tuple(np.asarray(z_batch_arr).shape)}."
        )
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(z_batch_arr.dtype)
    global_batch_arr = (
        z_batch_arr if global_condition_batch is None else jnp.asarray(global_condition_batch, dtype=z_batch_arr.dtype)
    )
    if global_batch_arr.shape != z_batch_arr.shape:
        raise ValueError(
            "global_condition_batch must match z_batch shape, "
            f"got {global_batch_arr.shape} and {z_batch_arr.shape}."
        )
    return _sample_token_conditional_batch_impl(
        drift_net,
        z_batch_arr,
        generation_zt,
        sigma_fn,
        dt0,
        key,
        condition_mode=condition_mode,
        global_condition_batch=global_batch_arr,
        interval_offset=interval_offset,
        adjoint=_resolve_forward_sampling_adjoint(adjoint),
    )


def sample_token_conditional_dense_batch_from_keys(
    drift_net: TokenConditionalDiT,
    z_batch: jax.Array,
    zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    keys: jax.Array,
    *,
    condition_mode: str = "global_and_previous",
    global_condition_batch: jax.Array | None = None,
    interval_offset: int = 0,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> tuple[jax.Array, jax.Array]:
    z_batch_arr = jnp.asarray(z_batch)
    key_batch_arr = jnp.asarray(keys)
    if z_batch_arr.ndim != 3:
        raise ValueError(
            f"z_batch must have shape (N, L, D), got {tuple(np.asarray(z_batch_arr).shape)}."
        )
    if key_batch_arr.ndim != 2 or key_batch_arr.shape[0] != z_batch_arr.shape[0]:
        raise ValueError(
            "keys must have shape (N, 2) aligned with z_batch, "
            f"got keys={key_batch_arr.shape}, z_batch={z_batch_arr.shape}."
        )
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(z_batch_arr.dtype)
    global_batch_arr = (
        z_batch_arr if global_condition_batch is None else jnp.asarray(global_condition_batch, dtype=z_batch_arr.dtype)
    )
    if global_batch_arr.shape != z_batch_arr.shape:
        raise ValueError(
            "global_condition_batch must match z_batch shape, "
            f"got {global_batch_arr.shape} and {z_batch_arr.shape}."
        )
    return jax.vmap(
        lambda z, global_cond, subkey: _sample_token_conditional_trajectory_dense_impl(
            drift_net,
            z,
            global_cond,
            generation_zt,
            sigma_fn,
            dt0,
            subkey,
            condition_mode=condition_mode,
            interval_offset=interval_offset,
            adjoint=_resolve_forward_sampling_adjoint(adjoint),
        )
    )(z_batch_arr, global_batch_arr, key_batch_arr)


__all__ = [
    "sample_token_conditional_dense_batch_from_keys",
    "sample_token_conditional_batch",
    "sample_token_conditional_trajectory",
]
