from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .conditional_models import IntervalConditionalModelStack
from .ecmmd import ecmmd_loss


def _validate_training_inputs(latent_data: jax.Array) -> None:
    latent_np = np.asarray(latent_data)
    if latent_np.ndim != 3:
        raise ValueError(f"latent_data must have shape (T, N, K), got {latent_np.shape}.")
    if latent_np.shape[0] < 2:
        raise ValueError(f"Expected at least two scale levels, got {latent_np.shape[0]}.")
    if not np.all(np.isfinite(latent_np)):
        raise ValueError("latent_data contains non-finite values.")


def _select_batch(latent_data: jax.Array, batch_size: int | None, key: jax.Array) -> jax.Array:
    n_samples = int(latent_data.shape[1])
    if batch_size is None or batch_size >= n_samples:
        return latent_data
    indices = jax.random.choice(key, n_samples, shape=(int(batch_size),), replace=False)
    return latent_data[:, indices, :]


def _sample_training_aux(
    model: Any,
    key: jax.Array,
    *,
    batch_size: int,
    dtype: jnp.dtype,
) -> jax.Array | None:
    aux_noise_dim = int(getattr(model, "aux_noise_dim", 0))
    if aux_noise_dim <= 0:
        return None
    aux = jax.random.normal(key, (int(batch_size), aux_noise_dim), dtype=dtype)
    return aux


def _sample_model_batch(
    model: Any,
    conditions: jax.Array,
    key: jax.Array,
) -> jax.Array:
    batch_size = int(conditions.shape[0])
    aux = _sample_training_aux(model, key, batch_size=batch_size, dtype=conditions.dtype)
    if aux is None:
        return jax.vmap(model)(conditions)
    return jax.vmap(model)(conditions, aux)


def make_interval_conditional_ecmmd_loss_fn(
    static_model: IntervalConditionalModelStack,
    latent_data: jax.Array,
    *,
    k_neighbors: int = 8,
    batch_size: int | None = 256,
) -> Callable[[IntervalConditionalModelStack, jax.Array], jax.Array]:
    num_intervals = len(static_model.interval_models)
    if int(latent_data.shape[0]) != num_intervals + 1:
        raise ValueError(
            f"latent_data has {latent_data.shape[0]} levels, but the ensemble expects {num_intervals + 1}."
        )

    def loss_fn(params: IntervalConditionalModelStack, key: jax.Array) -> jax.Array:
        ensemble = eqx.combine(params, static_model)
        batch_key, draw_key = jax.random.split(key)
        latent_batch = _select_batch(latent_data, batch_size, batch_key)
        interval_keys = jax.random.split(draw_key, num_intervals)
        interval_losses = []
        for coarse_level in range(1, num_intervals + 1):
            coarse = latent_batch[coarse_level]
            fine = latent_batch[coarse_level - 1]
            model = ensemble.model_for_coarse_level(coarse_level)
            generated = _sample_model_batch(model, coarse, interval_keys[coarse_level - 1])
            interval_losses.append(ecmmd_loss(coarse, fine, generated, k_neighbors=k_neighbors))
        return jnp.mean(jnp.stack(interval_losses, axis=0))

    return loss_fn


def train_interval_conditional_ecmmd(
    ensemble: IntervalConditionalModelStack,
    latent_data: jax.Array,
    *,
    k_neighbors: int = 8,
    lr: float = 5e-4,
    num_steps: int = 1000,
    batch_size: int | None = 256,
    seed: int = 0,
    return_losses: bool = False,
    progress_every: int | None = None,
    progress_fn: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> IntervalConditionalModelStack | tuple[IntervalConditionalModelStack, jax.Array]:
    latent_jax = jnp.asarray(latent_data, dtype=jnp.float32)
    _validate_training_inputs(latent_jax)

    params, static = eqx.partition(ensemble, eqx.is_inexact_array)
    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)
    loss_fn = make_interval_conditional_ecmmd_loss_fn(
        static,
        latent_jax,
        k_neighbors=k_neighbors,
        batch_size=batch_size,
    )

    @jax.jit
    def step(
        params: IntervalConditionalModelStack,
        opt_state: optax.OptState,
        key: jax.Array,
    ) -> tuple[IntervalConditionalModelStack, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    key = jax.random.PRNGKey(seed)
    losses = []
    batch_size_effective = int(latent_jax.shape[1]) if batch_size is None else min(int(batch_size), int(latent_jax.shape[1]))
    num_steps_int = int(num_steps)
    steady_step_times: deque[float] = deque(maxlen=20)
    train_start = time.perf_counter()

    for step_idx in range(num_steps_int):
        step_start = time.perf_counter()
        key, subkey = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, subkey)
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
            samples_per_second = steps_per_second * float(batch_size_effective)
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
                    "is_warmup_step": bool(step_idx == 0),
                }
            )

    trained_model = eqx.combine(params, static)
    if return_losses:
        return trained_model, jnp.asarray(losses, dtype=latent_jax.dtype)
    return trained_model
