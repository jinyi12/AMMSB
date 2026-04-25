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
    local_interval_time,
    make_bridge_condition,
    validate_bridge_condition_dim,
)
from ._trajectory_layout import generation_zt_from_data_zt, reverse_level_order, validate_data_zt
from .sde import ConditionalDriftNet


def _validate_bridge_matching_inputs(latent_data: jax.Array, zt: jax.Array) -> None:
    latent_np = np.asarray(latent_data)

    if latent_np.ndim != 3:
        raise ValueError(f"latent_data must have shape (T, N, K), got {latent_np.shape}.")
    validate_data_zt(zt)
    zt_np = np.asarray(zt)
    if latent_np.shape[0] != zt_np.shape[0]:
        raise ValueError(
            f"latent_data and zt disagree on T: {latent_np.shape[0]} versus {zt_np.shape[0]}."
        )
    if latent_np.shape[0] < 2:
        raise ValueError("Need at least two scale levels for bridge matching.")
    if not np.all(np.isfinite(latent_np)):
        raise ValueError("latent_data contains non-finite values.")


def _generation_view(
    latent_data: jax.Array,
    zt: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return a coarse-to-fine view for training from stored fine-to-coarse data."""
    _validate_bridge_matching_inputs(latent_data, zt)
    latent_jax = jnp.asarray(latent_data, dtype=jnp.float32)
    return reverse_level_order(latent_jax), generation_zt_from_data_zt(zt)


def _select_batch(latent_data: jax.Array, batch_size: int | None, key: jax.Array) -> jax.Array:
    n_samples = int(latent_data.shape[1])
    if batch_size is None or batch_size >= n_samples:
        return latent_data
    indices = jax.random.choice(key, n_samples, shape=(int(batch_size),), replace=False)
    return latent_data[:, indices, :]


def sample_brownian_bridge(
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


def bridge_target(
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


def _sample_truncated_interval_time(
    *,
    time_key: jax.Array,
    batch_size: int,
    dtype: jnp.dtype,
    t_start: jax.Array,
    t_end: jax.Array,
    endpoint_epsilon: float,
) -> jax.Array:
    interval_length = jnp.asarray(t_end - t_start, dtype=dtype)
    epsilon = jnp.minimum(
        jnp.asarray(max(float(endpoint_epsilon), 0.0), dtype=dtype),
        0.5 * interval_length,
    )
    span = interval_length - 2.0 * epsilon
    midpoint = jnp.asarray(t_start, dtype=dtype) + 0.5 * interval_length
    u = jax.random.uniform(time_key, (int(batch_size), 1), dtype=dtype)
    truncated = jnp.asarray(t_start, dtype=dtype) + epsilon + u * span
    return jnp.where(span > 0.0, truncated, midpoint)


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
        batch_key, time_key, bridge_key = jax.random.split(key, 3)
        batch = _select_batch(latent_canonical, batch_size, batch_key)
        global_condition = batch[0]
        interval_indices = jnp.arange(num_intervals, dtype=jnp.int32)
        time_keys = jax.random.split(time_key, num_intervals)
        bridge_keys = jax.random.split(bridge_key, num_intervals)

        def _interval_loss(interval_idx: jax.Array, interval_time_key: jax.Array, interval_bridge_key: jax.Array) -> jax.Array:
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
            t = _sample_truncated_interval_time(
                time_key=interval_time_key,
                batch_size=int(batch.shape[1]),
                dtype=batch.dtype,
                t_start=t_start,
                t_end=t_end,
                endpoint_epsilon=endpoint_epsilon,
            )
            x_t = sample_brownian_bridge(x_start, x_end, t, t_start, t_end, sigma, interval_bridge_key)
            target = bridge_target(x_t, x_end, t, t_end)
            pred = jax.vmap(drift)(local_interval_time(t, t_start, t_end).squeeze(-1), x_t, condition)
            return jnp.mean(jnp.sum(jnp.square(pred - target), axis=-1))

        interval_losses = jax.vmap(_interval_loss)(interval_indices, time_keys, bridge_keys)
        return jnp.mean(interval_losses)

    if latent_data is None and zt is None:
        return loss_fn
    if latent_data is None or zt is None:
        raise ValueError("latent_data and zt must both be provided for the legacy loss-factory form.")
    latent_canonical, zt_canonical = _generation_view(latent_data, zt)

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
    _validate_bridge_matching_inputs(latent_data, zt)
    latent_jax, zt_canonical = _generation_view(latent_data, zt)
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
