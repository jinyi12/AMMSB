from __future__ import annotations

import numpy as np

import diffrax
import jax
import jax.numpy as jnp

from ._conditional_bridge import make_bridge_condition, validate_bridge_condition_dim
from ._trajectory_layout import generation_zt_from_data_zt, validate_data_zt
from .sde import ConditionalDriftNet, DriftNet, SigmaFn, integrate_conditional_interval, integrate_interval


def _validate_tau_knots(tau_knots: jax.Array) -> None:
    tau_np = np.asarray(tau_knots)
    if tau_np.ndim != 1:
        raise ValueError(f"tau_knots must be 1-D, got {tau_np.shape}.")
    if tau_np.shape[0] < 2:
        raise ValueError("Need at least two tau knots to sample a trajectory.")
    if not np.all(np.diff(tau_np) < 0.0):
        raise ValueError("tau_knots must be strictly decreasing, e.g. tau = 1 - zt.")


def _sample_trajectory_impl(
    drift_net: DriftNet,
    x_T: jax.Array,
    tau_knots: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    trajectory = [jnp.asarray(x_T)]
    x = trajectory[0]
    for i in range(len(tau_knots) - 1, 0, -1):
        key, subkey = jax.random.split(key)
        x = integrate_interval(
            drift_net,
            x,
            tau_knots[i],
            tau_knots[i - 1],
            dt0,
            subkey,
            sigma_fn,
            adjoint=adjoint,
        )
        trajectory.append(x)
    return jnp.stack(trajectory[::-1], axis=0)


def _sample_conditional_trajectory_impl(
    drift_net: ConditionalDriftNet,
    initial_state: jax.Array,
    global_condition: jax.Array,
    generation_zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    include_interval_embedding: bool = True,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    initial_state_arr = jnp.asarray(initial_state)
    global_condition_arr = jnp.asarray(global_condition, dtype=initial_state_arr.dtype)
    trajectory = [initial_state_arr]
    x = initial_state_arr
    rollout_num_intervals = int(len(generation_zt) - 1)
    condition_num_intervals_int = rollout_num_intervals if condition_num_intervals is None else int(condition_num_intervals)
    interval_offset_int = int(interval_offset)
    for i in range(rollout_num_intervals):
        key, subkey = jax.random.split(key)
        condition = make_bridge_condition(
            global_condition_arr,
            x,
            interval_idx=interval_offset_int + i,
            num_intervals=condition_num_intervals_int,
            condition_mode=condition_mode,
            include_interval_embedding=include_interval_embedding,
        )
        x = integrate_conditional_interval(
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


def sample_trajectory(
    drift_net: DriftNet,
    x_T: jax.Array,
    tau_knots: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    """Sample a single latent trajectory from coarse to fine and return it fine-to-coarse."""
    tau_arr = jnp.asarray(tau_knots, dtype=jnp.asarray(x_T).dtype)
    _validate_tau_knots(tau_arr)
    return _sample_trajectory_impl(drift_net, x_T, tau_arr, sigma_fn, dt0, key, adjoint=adjoint)


def sample_batch(
    drift_net: DriftNet,
    x_T_batch: jax.Array,
    tau_knots: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    """Sample a batch of latent trajectories from coarse endpoints."""
    x_batch = jnp.asarray(x_T_batch)
    tau_arr = jnp.asarray(tau_knots, dtype=x_batch.dtype)
    _validate_tau_knots(tau_arr)
    keys = jax.random.split(key, x_batch.shape[0])
    return jax.vmap(
        lambda x, subkey: _sample_trajectory_impl(
            drift_net,
            x,
            tau_arr,
            sigma_fn,
            dt0,
            subkey,
            adjoint=adjoint,
        )
    )(x_batch, keys)


def sample_conditional_trajectory(
    drift_net: ConditionalDriftNet,
    z: jax.Array,
    zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str = "global_and_previous",
    global_condition: jax.Array | None = None,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    include_interval_embedding: bool = True,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    """Sample a sequential conditional latent trajectory and return it in stored fine-to-coarse order.

    Args:
        z: Initial state for the final stored level of the provided `zt` grid.
        zt: Stored data-time grid aligned with latent archives, increasing from
            finest level to coarsest level.
    """
    z_arr = jnp.asarray(z)
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(z_arr.dtype)
    validate_bridge_condition_dim(
        drift_net,
        latent_dim=int(z_arr.shape[-1]),
        num_intervals=(
            int(generation_zt.shape[0] - 1) if condition_num_intervals is None else int(condition_num_intervals)
        ),
        condition_mode=condition_mode,
        include_interval_embedding=include_interval_embedding,
    )
    global_arr = z_arr if global_condition is None else jnp.asarray(global_condition, dtype=z_arr.dtype)
    return _sample_conditional_trajectory_impl(
        drift_net,
        z_arr,
        global_arr,
        generation_zt,
        sigma_fn,
        dt0,
        key,
        condition_mode=condition_mode,
        condition_num_intervals=condition_num_intervals,
        interval_offset=interval_offset,
        include_interval_embedding=include_interval_embedding,
        adjoint=adjoint,
    )


def sample_conditional_batch(
    drift_net: ConditionalDriftNet,
    z_batch: jax.Array,
    zt: jax.Array,
    sigma_fn: SigmaFn,
    dt0: float,
    key: jax.Array,
    *,
    condition_mode: str = "global_and_previous",
    global_condition_batch: jax.Array | None = None,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    include_interval_embedding: bool = True,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    """Sample sequential conditional latent trajectories in stored fine-to-coarse order."""
    z_batch_arr = jnp.asarray(z_batch)
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(z_batch_arr.dtype)
    validate_bridge_condition_dim(
        drift_net,
        latent_dim=int(z_batch_arr.shape[-1]),
        num_intervals=(
            int(generation_zt.shape[0] - 1) if condition_num_intervals is None else int(condition_num_intervals)
        ),
        condition_mode=condition_mode,
        include_interval_embedding=include_interval_embedding,
    )
    global_batch_arr = (
        z_batch_arr if global_condition_batch is None else jnp.asarray(global_condition_batch, dtype=z_batch_arr.dtype)
    )
    if global_batch_arr.shape != z_batch_arr.shape:
        raise ValueError(
            "global_condition_batch must match z_batch shape, "
            f"got {global_batch_arr.shape} and {z_batch_arr.shape}."
        )
    keys = jax.random.split(key, z_batch_arr.shape[0])
    return jax.vmap(
        lambda z, global_cond, subkey: _sample_conditional_trajectory_impl(
            drift_net,
            z,
            global_cond,
            generation_zt,
            sigma_fn,
            dt0,
            subkey,
            condition_mode=condition_mode,
            condition_num_intervals=condition_num_intervals,
            interval_offset=interval_offset,
            include_interval_embedding=include_interval_embedding,
            adjoint=adjoint,
        )
    )(z_batch_arr, global_batch_arr, keys)
