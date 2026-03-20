from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ecmmd import ecmmd_loss
from .sde import DriftNet, SigmaFn, integrate_interval


def _validate_training_inputs(latent_data: jax.Array, tau_knots: jax.Array) -> None:
    latent_np = np.asarray(latent_data)
    tau_np = np.asarray(tau_knots)

    if latent_np.ndim != 3:
        raise ValueError(f"latent_data must have shape (T, N, K), got {latent_np.shape}.")
    if tau_np.ndim != 1:
        raise ValueError(f"tau_knots must be 1-D, got {tau_np.shape}.")
    if latent_np.shape[0] != tau_np.shape[0]:
        raise ValueError(
            f"latent_data and tau_knots disagree on T: {latent_np.shape[0]} versus {tau_np.shape[0]}."
        )
    if latent_np.shape[0] < 2:
        raise ValueError("Need at least two scale levels for CSP training.")
    if not np.all(np.isfinite(latent_np)):
        raise ValueError("latent_data contains non-finite values.")
    if not np.all(np.isfinite(tau_np)):
        raise ValueError("tau_knots contains non-finite values.")
    if not np.all(np.diff(tau_np) < 0.0):
        raise ValueError("tau_knots must be strictly decreasing, e.g. tau = 1 - zt.")


def _select_batch(latent_data: jax.Array, batch_size: int | None, key: jax.Array) -> jax.Array:
    n_samples = int(latent_data.shape[1])
    if batch_size is None or batch_size >= n_samples:
        return latent_data
    indices = jax.random.choice(key, n_samples, shape=(int(batch_size),), replace=False)
    return latent_data[:, indices, :]


def make_loss_fn(
    static_model: DriftNet,
    latent_data: jax.Array,
    tau_knots: jax.Array,
    sigma_fn: SigmaFn,
    *,
    k_neighbors: int = 5,
    dt0: float = 0.01,
    batch_size: int | None = 256,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> Callable[[DriftNet, jax.Array], jax.Array]:
    """Return a pure loss function over model parameters and a PRNG key."""
    num_levels = int(latent_data.shape[0])
    num_intervals = num_levels - 1

    def loss_fn(params: DriftNet, key: jax.Array) -> jax.Array:
        drift = eqx.combine(params, static_model)
        batch_key, sim_key = jax.random.split(key)
        latent_batch = _select_batch(latent_data, batch_size, batch_key)
        interval_keys = jax.random.split(sim_key, num_intervals)
        interval_losses = []

        for i in range(1, num_levels):
            coarse = latent_batch[i]
            fine = latent_batch[i - 1]
            sample_keys = jax.random.split(interval_keys[i - 1], coarse.shape[0])
            generated = jax.vmap(
                lambda y0, subkey: integrate_interval(
                    drift,
                    y0,
                    tau_knots[i],
                    tau_knots[i - 1],
                    dt0,
                    subkey,
                    sigma_fn,
                    adjoint=adjoint,
                )
            )(coarse, sample_keys)
            interval_losses.append(ecmmd_loss(coarse, fine, generated, k_neighbors=k_neighbors))

        return jnp.mean(jnp.stack(interval_losses, axis=0))

    return loss_fn


def train(
    drift_net: DriftNet,
    latent_data: jax.Array,
    tau_knots: jax.Array,
    sigma_fn: SigmaFn,
    *,
    k_neighbors: int = 5,
    dt0: float = 0.01,
    lr: float = 1e-4,
    num_steps: int = 10_000,
    batch_size: int | None = 256,
    seed: int = 0,
    adjoint: diffrax.AbstractAdjoint | None = None,
    return_losses: bool = False,
    progress_every: int | None = None,
    progress_fn: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> DriftNet | tuple[DriftNet, jax.Array]:
    """Train a shared reverse SDE on paired latent trajectories."""
    latent_jax = jnp.asarray(latent_data, dtype=jnp.float32)
    tau_jax = jnp.asarray(tau_knots, dtype=jnp.float32)
    _validate_training_inputs(latent_jax, tau_jax)

    params, static = eqx.partition(drift_net, eqx.is_inexact_array)
    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)
    loss_fn = make_loss_fn(
        static,
        latent_jax,
        tau_jax,
        sigma_fn,
        k_neighbors=k_neighbors,
        dt0=dt0,
        batch_size=batch_size,
        adjoint=adjoint,
    )

    @jax.jit
    def step(
        params: DriftNet,
        opt_state: optax.OptState,
        key: jax.Array,
    ) -> tuple[DriftNet, optax.OptState, jax.Array]:
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
