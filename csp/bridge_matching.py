from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._conditional_bridge import (
    brownian_bridge_target,
    local_interval_time,
    make_bridge_condition,
    sample_brownian_bridge_state,
    sample_truncated_interval_time,
    validate_bridge_condition_dim,
)
from ._trajectory_layout import generation_view_from_data, select_level_batch
from .sde import ConditionalDriftNet


def _paired_generation_view(
    condition_latent_data: jax.Array,
    target_latent_data: jax.Array,
    zt: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return matched coarse-to-fine views for detached-target bridge regression."""
    condition_shape = tuple(int(dim) for dim in condition_latent_data.shape)
    target_shape = tuple(int(dim) for dim in target_latent_data.shape)
    if condition_shape != target_shape:
        raise ValueError(
            "condition_latent_data and target_latent_data must share the same shape. "
            f"Got {condition_shape} and {target_shape}."
        )
    condition_canonical, zt_canonical = generation_view_from_data(
        condition_latent_data,
        zt,
        ndim=3,
        shape_spec="(T, N, K)",
        name="condition_latent_data",
        context="detached bridge regression",
    )
    target_canonical, _zt_canonical_target = generation_view_from_data(
        target_latent_data,
        zt,
        ndim=3,
        shape_spec="(T, N, K)",
        name="target_latent_data",
        context="detached bridge regression",
    )
    return condition_canonical, target_canonical, zt_canonical


def sample_brownian_bridge(
    x_start: jax.Array,
    x_end: jax.Array,
    t: jax.Array | float,
    t_start: jax.Array | float,
    t_end: jax.Array | float,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    return sample_brownian_bridge_state(x_start, x_end, t, t_start, t_end, sigma, key)


def bridge_target(
    x_t: jax.Array,
    x_end: jax.Array,
    t: jax.Array | float,
    t_end: jax.Array | float,
) -> jax.Array:
    return brownian_bridge_target(x_t, x_end, t, t_end)


def _bridge_matching_pass_loss(
    drift,
    batch: jax.Array,
    zt_canonical: jax.Array,
    *,
    key: jax.Array,
    sigma: float,
    condition_mode: str,
    include_interval_embedding: bool,
    endpoint_epsilon: float,
) -> jax.Array:
    num_intervals = int(batch.shape[0] - 1)
    global_condition = batch[0]
    interval_indices = jnp.arange(num_intervals, dtype=jnp.int32)
    time_key, bridge_key = jax.random.split(key, 2)
    time_keys = jax.random.split(time_key, num_intervals)
    bridge_keys = jax.random.split(bridge_key, num_intervals)

    def _interval_loss(
        interval_idx: jax.Array,
        interval_time_key: jax.Array,
        interval_bridge_key: jax.Array,
    ) -> jax.Array:
        x_start = batch[interval_idx]
        x_end = batch[interval_idx + 1]
        t_start = zt_canonical[interval_idx]
        t_end = zt_canonical[interval_idx + 1]
        condition = make_bridge_condition(
            global_condition,
            x_start,
            interval_idx=interval_idx,
            num_intervals=num_intervals,
            condition_mode=condition_mode,
            include_interval_embedding=include_interval_embedding,
        )
        t = sample_truncated_interval_time(
            time_key=interval_time_key,
            batch_size=int(batch.shape[1]),
            dtype=batch.dtype,
            t_start=t_start,
            t_end=t_end,
            endpoint_epsilon=endpoint_epsilon,
            state_rank=int(x_start.ndim - 1),
        )
        x_t = sample_brownian_bridge(
            x_start,
            x_end,
            t,
            t_start,
            t_end,
            sigma,
            interval_bridge_key,
        )
        target = bridge_target(x_t, x_end, t, t_end)
        pred = jax.vmap(drift)(
            local_interval_time(t, t_start, t_end).squeeze(-1),
            x_t,
            condition,
        )
        return jnp.mean(jnp.sum(jnp.square(pred - target), axis=-1))

    interval_losses = jax.vmap(_interval_loss)(interval_indices, time_keys, bridge_keys)
    return jnp.mean(interval_losses)


def _detached_bridge_matching_pass_loss(
    drift,
    condition_batch: jax.Array,
    target_batch: jax.Array,
    zt_canonical: jax.Array,
    *,
    key: jax.Array,
    sigma: float,
    condition_mode: str,
    include_interval_embedding: bool,
    endpoint_epsilon: float,
) -> jax.Array:
    num_intervals = int(condition_batch.shape[0] - 1)
    global_condition = condition_batch[0]
    interval_indices = jnp.arange(num_intervals, dtype=jnp.int32)
    time_key, bridge_key = jax.random.split(key, 2)
    time_keys = jax.random.split(time_key, num_intervals)
    bridge_keys = jax.random.split(bridge_key, num_intervals)

    def _interval_loss(
        interval_idx: jax.Array,
        interval_time_key: jax.Array,
        interval_bridge_key: jax.Array,
    ) -> jax.Array:
        x_start_condition = condition_batch[interval_idx]
        x_start_target = target_batch[interval_idx]
        x_end_target = target_batch[interval_idx + 1]
        t_start = zt_canonical[interval_idx]
        t_end = zt_canonical[interval_idx + 1]
        condition = make_bridge_condition(
            global_condition,
            x_start_condition,
            interval_idx=interval_idx,
            num_intervals=num_intervals,
            condition_mode=condition_mode,
            include_interval_embedding=include_interval_embedding,
        )
        t = sample_truncated_interval_time(
            time_key=interval_time_key,
            batch_size=int(condition_batch.shape[1]),
            dtype=condition_batch.dtype,
            t_start=t_start,
            t_end=t_end,
            endpoint_epsilon=endpoint_epsilon,
            state_rank=int(x_start_target.ndim - 1),
        )
        x_t = sample_brownian_bridge(
            x_start_target,
            x_end_target,
            t,
            t_start,
            t_end,
            sigma,
            interval_bridge_key,
        )
        target = bridge_target(x_t, x_end_target, t, t_end)
        pred = jax.vmap(drift)(
            local_interval_time(t, t_start, t_end).squeeze(-1),
            x_t,
            condition,
        )
        return jnp.mean(jnp.sum(jnp.square(pred - target), axis=-1))

    interval_losses = jax.vmap(_interval_loss)(interval_indices, time_keys, bridge_keys)
    return jnp.mean(interval_losses)


def estimate_monte_carlo_bridge_matching_loss(
    static_model: ConditionalDriftNet,
    params: ConditionalDriftNet,
    latent_data: jax.Array,
    zt: jax.Array,
    sigma: float,
    *,
    key: jax.Array,
    mc_passes: int,
    mc_chunk_size: int = 8,
    batch_size: int | None = None,
    condition_mode: str = "global_and_previous",
    include_interval_embedding: bool = True,
    endpoint_epsilon: float = 1e-3,
) -> jax.Array:
    """Average repeated bridge-matching draws over an already encoded latent bundle.

    This is intended for joint FAE+CSP training where a matched `(T, B, K)` latent
    trajectory tensor is already available and the bridge expectation should be
    reduced by Monte Carlo averaging without re-encoding the same batch.
    """
    mc_passes_int = int(mc_passes)
    mc_chunk_size_int = int(mc_chunk_size)
    if mc_passes_int < 1:
        raise ValueError(f"mc_passes must be >= 1, got {mc_passes}.")
    if mc_chunk_size_int < 1:
        raise ValueError(f"mc_chunk_size must be >= 1, got {mc_chunk_size}.")

    latent_canonical, zt_canonical = generation_view_from_data(
        latent_data,
        zt,
        ndim=3,
        shape_spec="(T, N, K)",
        context="bridge matching",
    )
    latent_dim = int(latent_canonical.shape[-1])
    num_intervals = int(latent_canonical.shape[0] - 1)
    validate_bridge_condition_dim(
        static_model,
        latent_dim=latent_dim,
        num_intervals=num_intervals,
        condition_mode=condition_mode,
        include_interval_embedding=include_interval_embedding,
    )
    drift = eqx.combine(params, static_model)
    batch_key, mc_key = jax.random.split(key)
    batch = select_level_batch(latent_canonical, batch_size, batch_key)
    pass_keys = jax.random.split(mc_key, mc_passes_int)

    total_loss = jnp.asarray(0.0, dtype=batch.dtype)
    for start in range(0, mc_passes_int, mc_chunk_size_int):
        stop = min(start + mc_chunk_size_int, mc_passes_int)
        chunk_losses = jax.vmap(
            lambda pass_key: _bridge_matching_pass_loss(
                drift,
                batch,
                zt_canonical,
                key=pass_key,
                sigma=sigma,
                condition_mode=condition_mode,
                include_interval_embedding=include_interval_embedding,
                endpoint_epsilon=endpoint_epsilon,
            )
        )(pass_keys[start:stop])
        total_loss = total_loss + jnp.sum(chunk_losses)

    return total_loss / jnp.asarray(mc_passes_int, dtype=batch.dtype)


def estimate_monte_carlo_detached_bridge_matching_loss(
    static_model: ConditionalDriftNet,
    params: ConditionalDriftNet,
    condition_latent_data: jax.Array,
    target_latent_data: jax.Array,
    zt: jax.Array,
    sigma: float,
    *,
    key: jax.Array,
    mc_passes: int,
    mc_chunk_size: int = 8,
    batch_size: int | None = None,
    condition_mode: str = "global_and_previous",
    include_interval_embedding: bool = True,
    endpoint_epsilon: float = 1e-3,
) -> jax.Array:
    """Average detached-target bridge-matching draws over matched latent bundles."""
    mc_passes_int = int(mc_passes)
    mc_chunk_size_int = int(mc_chunk_size)
    if mc_passes_int < 1:
        raise ValueError(f"mc_passes must be >= 1, got {mc_passes}.")
    if mc_chunk_size_int < 1:
        raise ValueError(f"mc_chunk_size must be >= 1, got {mc_chunk_size}.")

    condition_canonical, target_canonical, zt_canonical = _paired_generation_view(
        condition_latent_data,
        target_latent_data,
        zt,
    )
    latent_dim = int(condition_canonical.shape[-1])
    num_intervals = int(condition_canonical.shape[0] - 1)
    validate_bridge_condition_dim(
        static_model,
        latent_dim=latent_dim,
        num_intervals=num_intervals,
        condition_mode=condition_mode,
        include_interval_embedding=include_interval_embedding,
    )
    drift = eqx.combine(params, static_model)
    batch_key, mc_key = jax.random.split(key)
    condition_batch = _select_batch(condition_canonical, batch_size, batch_key)
    target_batch = _select_batch(target_canonical, batch_size, batch_key)
    pass_keys = jax.random.split(mc_key, mc_passes_int)

    total_loss = jnp.asarray(0.0, dtype=condition_batch.dtype)
    for start in range(0, mc_passes_int, mc_chunk_size_int):
        stop = min(start + mc_chunk_size_int, mc_passes_int)
        chunk_losses = jax.vmap(
            lambda pass_key: _detached_bridge_matching_pass_loss(
                drift,
                condition_batch,
                target_batch,
                zt_canonical,
                key=pass_key,
                sigma=sigma,
                condition_mode=condition_mode,
                include_interval_embedding=include_interval_embedding,
                endpoint_epsilon=endpoint_epsilon,
            )
        )(pass_keys[start:stop])
        total_loss = total_loss + jnp.sum(chunk_losses)

    return total_loss / jnp.asarray(mc_passes_int, dtype=condition_batch.dtype)


def make_bridge_matching_loss_fn(
    static_model: ConditionalDriftNet,
    latent_data: jax.Array | None = None,
    zt: jax.Array | None = None,
    sigma: float = 1.0,
    *,
    batch_size: int | None = 256,
    condition_mode: str = "global_and_previous",
    include_interval_embedding: bool = True,
    endpoint_epsilon: float = 1e-3,
) -> Callable[..., jax.Array]:
    """Return a pure loss function for sequential conditional Brownian bridge matching.

    Args:
        sigma: Brownian reference diffusion coefficient.
        The loss uses an equal-weight interval average each step to remove the
        avoidable variance from sampling a single interval for the whole batch.
    """
    def loss_fn(
        params: ConditionalDriftNet,
        key: jax.Array,
        latent_canonical: jax.Array,
        zt_canonical: jax.Array,
    ) -> jax.Array:
        latent_dim = int(latent_canonical.shape[-1])
        num_intervals = int(latent_canonical.shape[0] - 1)
        validate_bridge_condition_dim(
            static_model,
            latent_dim=latent_dim,
            num_intervals=num_intervals,
            condition_mode=condition_mode,
            include_interval_embedding=include_interval_embedding,
        )
        drift = eqx.combine(params, static_model)
        batch_key, pass_key = jax.random.split(key, 2)
        batch = select_level_batch(latent_canonical, batch_size, batch_key)
        return _bridge_matching_pass_loss(
            drift,
            batch,
            zt_canonical,
            key=pass_key,
            sigma=sigma,
            condition_mode=condition_mode,
            include_interval_embedding=include_interval_embedding,
            endpoint_epsilon=endpoint_epsilon,
        )

    if latent_data is None and zt is None:
        return loss_fn
    if latent_data is None or zt is None:
        raise ValueError("latent_data and zt must both be provided for the legacy loss-factory form.")
    latent_canonical, zt_canonical = generation_view_from_data(
        latent_data,
        zt,
        ndim=3,
        shape_spec="(T, N, K)",
        context="bridge matching",
    )

    def legacy_loss_fn(params: ConditionalDriftNet, key: jax.Array) -> jax.Array:
        return loss_fn(params, key, latent_canonical, zt_canonical)

    return legacy_loss_fn


def train_bridge_matching(
    drift_net: ConditionalDriftNet,
    latent_data: jax.Array,
    zt: jax.Array,
    sigma: float,
    *,
    lr: float = 1e-4,
    num_steps: int = 10_000,
    batch_size: int | None = 256,
    seed: int = 0,
    condition_mode: str = "global_and_previous",
    include_interval_embedding: bool = True,
    endpoint_epsilon: float = 1e-3,
    return_losses: bool = False,
    progress_every: int | None = None,
    progress_fn: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> ConditionalDriftNet | tuple[ConditionalDriftNet, jax.Array]:
    """Train a sequential paired-data conditional drift via bridge matching regression.

    The stored training tuples and zt grid follow the repo-wide data order:
    fine-to-coarse. The reverse coarse-to-fine conditional-generation direction
    is constructed internally.
    """
    latent_jax, zt_canonical = generation_view_from_data(
        latent_data,
        zt,
        ndim=3,
        shape_spec="(T, N, K)",
        context="bridge matching",
    )
    params, static = eqx.partition(drift_net, eqx.is_inexact_array)
    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)
    loss_fn = make_bridge_matching_loss_fn(
        static,
        sigma=sigma,
        batch_size=batch_size,
        condition_mode=condition_mode,
        include_interval_embedding=include_interval_embedding,
        endpoint_epsilon=endpoint_epsilon,
    )

    @jax.jit
    def step(
        params: ConditionalDriftNet,
        opt_state: optax.OptState,
        key: jax.Array,
        latent_canonical: jax.Array,
        zt_canonical: jax.Array,
    ) -> tuple[ConditionalDriftNet, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, key, latent_canonical, zt_canonical)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    key = jax.random.PRNGKey(seed)
    losses = []
    batch_size_effective = int(latent_jax.shape[1]) if batch_size is None else min(int(batch_size), int(latent_jax.shape[1]))
    interval_samples_per_step = batch_size_effective * int(latent_jax.shape[0] - 1)
    num_steps_int = int(num_steps)
    steady_step_times: deque[float] = deque(maxlen=20)
    train_start = time.perf_counter()

    for step_idx in range(num_steps_int):
        step_start = time.perf_counter()
        key, subkey = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, subkey, latent_jax, zt_canonical)
        loss_value = float(jax.device_get(loss))
        step_seconds = time.perf_counter() - step_start
        losses.append(loss_value)

        if step_idx > 0:
            steady_step_times.append(step_seconds)

        should_report = (
            progress_fn is not None
            and progress_every is not None
            and progress_every > 0
            and (
                step_idx == 0
                or (step_idx + 1) % int(progress_every) == 0
                or (step_idx + 1) == num_steps_int
            )
        )
        if should_report:
            elapsed_seconds = time.perf_counter() - train_start
            eta_step_seconds = (
                float(np.mean(np.asarray(steady_step_times, dtype=np.float64)))
                if steady_step_times
                else step_seconds
            )
            steps_per_second = 0.0 if eta_step_seconds <= 0.0 else 1.0 / eta_step_seconds
            samples_per_second = steps_per_second * float(interval_samples_per_step)
            remaining_steps = max(num_steps_int - (step_idx + 1), 0)
            progress_fn(
                {
                    "step": int(step_idx + 1),
                    "num_steps": num_steps_int,
                    "loss": loss_value,
                    "step_seconds": float(step_seconds),
                    "elapsed_seconds": float(elapsed_seconds),
                    "eta_seconds": float(remaining_steps * eta_step_seconds),
                    "steps_per_second": float(steps_per_second),
                    "samples_per_second": float(samples_per_second),
                    "batch_size": batch_size_effective,
                    "interval_samples_per_step": interval_samples_per_step,
                    "is_warmup_step": bool(step_idx == 0),
                }
            )

    trained_model = eqx.combine(params, static)
    if return_losses:
        return trained_model, jnp.asarray(losses, dtype=latent_jax.dtype)
    return trained_model
