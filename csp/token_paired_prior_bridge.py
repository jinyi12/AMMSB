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

from ._trajectory_layout import generation_view_from_data, generation_zt_from_data_zt, select_level_batch, validate_data_zt
from .paired_prior_bridge import (
    DEFAULT_THETA_FEATURE_CLIP,
    _interval_sigma,
    _validate_bridge_hyperparameters,
    bridge_logsnr,
    matched_prior_time_from_bridge_logsnr,
    resolve_prior_logsnr_max_from_checkpoint_path,
    resolve_prior_logsnr_max_from_checkpoint_payload,
    sample_ve_paired_interval,
)
from .sde import interval_save_times
from .token_dit import TokenConditionalDiT, integrate_token_conditional_interval, make_token_bridge_condition


PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE = "paired_prior_bridge_token_dit"
PAIRED_PRIOR_TOKEN_DIT_TRAINING_OBJECTIVE = "paired_prior_bridge_state_prediction"


def _resolve_forward_sampling_adjoint(
    adjoint: diffrax.AbstractAdjoint | None,
) -> diffrax.AbstractAdjoint:
    """Use a forward-only adjoint for rollout sampling unless the caller overrides it."""
    if adjoint is None:
        return diffrax.DirectAdjoint()
    return adjoint


def _sample_theta(
    *,
    key: jax.Array,
    batch_size: int,
    dtype: jnp.dtype,
    theta_trim: float,
) -> jax.Array:
    theta_trim_arr = jnp.asarray(float(theta_trim), dtype=dtype)
    span = jnp.asarray(1.0, dtype=dtype) - 2.0 * theta_trim_arr
    u = jax.random.uniform(key, (int(batch_size), 1, 1), dtype=dtype)
    return theta_trim_arr + u * span


def _token_paired_prior_pass_metrics(
    drift: TokenConditionalDiT,
    batch: jax.Array,
    *,
    key: jax.Array,
    delta_v: float,
    theta_trim: float,
    prior_logsnr_max: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    num_intervals = int(batch.shape[0] - 1)
    interval_indices = jnp.arange(num_intervals, dtype=jnp.int32)
    theta_key, noise_key = jax.random.split(key, 2)
    theta_keys = jax.random.split(theta_key, num_intervals)
    noise_keys = jax.random.split(noise_key, num_intervals)

    def _interval_metrics(
        interval_idx: jax.Array,
        interval_theta_key: jax.Array,
        interval_noise_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        x_prev = batch[interval_idx]
        x_next = batch[interval_idx + 1]
        theta = _sample_theta(
            key=interval_theta_key,
            batch_size=int(batch.shape[1]),
            dtype=batch.dtype,
            theta_trim=theta_trim,
        )
        x_t = sample_ve_paired_interval(x_prev, x_next, theta, delta_v, interval_noise_key)
        logsnr = bridge_logsnr(theta[..., 0], delta_v)
        condition = jax.vmap(
            lambda previous_tokens: make_token_bridge_condition(
                previous_tokens,
                previous_tokens,
                interval_idx=interval_idx,
                condition_mode="previous_state",
            )
        )(x_prev)
        drift_pred = jax.vmap(drift)(logsnr.squeeze(-1), x_t, condition)
        denom = jnp.maximum(1.0 - theta, jnp.asarray(1e-6, dtype=batch.dtype))
        next_hat = x_t + denom * drift_pred
        state_residual = x_next - next_hat
        drift_target = (x_next - x_t) / denom

        state_loss = jnp.mean(jnp.sum(jnp.square(state_residual) / jnp.square(denom), axis=(-2, -1)))
        drift_mse = jnp.mean(jnp.sum(jnp.square(drift_pred - drift_target), axis=(-2, -1)))
        matched_prior_time = matched_prior_time_from_bridge_logsnr(logsnr, prior_logsnr_max)
        return (
            state_loss,
            drift_mse,
            jnp.mean(logsnr),
            jnp.mean(matched_prior_time),
        )

    interval_metrics = jax.vmap(_interval_metrics)(interval_indices, theta_keys, noise_keys)
    return tuple(jnp.mean(metric, axis=0) for metric in interval_metrics)


def train_token_paired_prior_bridge(
    drift_net: TokenConditionalDiT,
    latent_data: jax.Array,
    zt: jax.Array,
    delta_v: float,
    prior_logsnr_max: float,
    *,
    lr: float = 1e-4,
    num_steps: int = 10_000,
    batch_size: int | None = 256,
    seed: int = 0,
    theta_trim: float = 0.05,
    return_history: bool = False,
    progress_every: int | None = None,
    progress_fn: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> TokenConditionalDiT | tuple[TokenConditionalDiT, dict[str, jax.Array]]:
    _validate_bridge_hyperparameters(delta_v, theta_trim, DEFAULT_THETA_FEATURE_CLIP)
    latent_canonical, _generation_zt = generation_view_from_data(
        latent_data,
        zt,
        ndim=4,
        shape_spec="(T, N, L, D)",
        context="the token paired prior bridge",
    )

    params, static = eqx.partition(drift_net, eqx.is_inexact_array)
    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)

    def _loss_and_aux(
        params_inner,
        key_inner: jax.Array,
        latent_canonical_inner: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        drift = eqx.combine(params_inner, static)
        batch_key, pass_key = jax.random.split(key_inner)
        batch = select_level_batch(latent_canonical_inner, batch_size, batch_key)
        state_loss, drift_mse, mean_logsnr, mean_prior_time = _token_paired_prior_pass_metrics(
            drift,
            batch,
            key=pass_key,
            delta_v=delta_v,
            theta_trim=theta_trim,
            prior_logsnr_max=prior_logsnr_max,
        )
        return state_loss, (drift_mse, mean_logsnr, mean_prior_time)

    @jax.jit
    def step(
        params_inner,
        opt_state_inner: optax.OptState,
        key_inner: jax.Array,
        latent_canonical_inner: jax.Array,
    ) -> tuple[TokenConditionalDiT, optax.OptState, jax.Array, jax.Array, jax.Array, jax.Array]:
        (loss, aux), grads = jax.value_and_grad(_loss_and_aux, has_aux=True)(
            params_inner,
            key_inner,
            latent_canonical_inner,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state_inner, params=params_inner)
        new_params = optax.apply_updates(params_inner, updates)
        drift_mse, mean_logsnr, mean_prior_time = aux
        return new_params, new_opt_state, loss, drift_mse, mean_logsnr, mean_prior_time

    key = jax.random.PRNGKey(seed)
    state_losses: list[jax.Array] = []
    drift_mses: list[jax.Array] = []
    mean_logsnr_history: list[jax.Array] = []
    mean_prior_time_history: list[jax.Array] = []
    batch_size_effective = (
        int(latent_canonical.shape[1])
        if batch_size is None
        else min(int(batch_size), int(latent_canonical.shape[1]))
    )
    interval_samples_per_step = batch_size_effective * int(latent_canonical.shape[0] - 1)
    num_steps_int = int(num_steps)
    steady_step_times: deque[float] = deque(maxlen=20)
    train_start = time.perf_counter()

    for step_idx in range(num_steps_int):
        step_start = time.perf_counter()
        key, subkey = jax.random.split(key)
        params, opt_state, loss, drift_mse, mean_logsnr, mean_prior_time = step(
            params,
            opt_state,
            subkey,
            latent_canonical,
        )
        state_losses.append(loss)
        drift_mses.append(drift_mse)
        mean_logsnr_history.append(mean_logsnr)
        mean_prior_time_history.append(mean_prior_time)

        if progress_fn is not None and progress_every is not None and (
            step_idx == 0 or (step_idx + 1) % int(progress_every) == 0 or step_idx == num_steps_int - 1
        ):
            step_seconds = time.perf_counter() - step_start
            if step_idx > 0:
                steady_step_times.append(step_seconds)
            effective_step_seconds = (
                float(np.mean(np.asarray(steady_step_times, dtype=np.float64)))
                if steady_step_times
                else step_seconds
            )
            elapsed_seconds = time.perf_counter() - train_start
            remaining_steps = max(num_steps_int - (step_idx + 1), 0)
            eta_seconds = remaining_steps * effective_step_seconds
            progress_fn(
                {
                    "step": step_idx + 1,
                    "num_steps": num_steps_int,
                    "loss": float(loss),
                    "drift_mse": float(drift_mse),
                    "step_seconds": float(step_seconds),
                    "steps_per_second": float(1.0 / max(step_seconds, 1e-12)),
                    "samples_per_second": float(interval_samples_per_step / max(step_seconds, 1e-12)),
                    "elapsed_seconds": float(elapsed_seconds),
                    "eta_seconds": float(eta_seconds),
                    "is_warmup_step": bool(step_idx == 0),
                }
            )

    trained_model = eqx.combine(params, static)
    if not return_history:
        return trained_model

    history = {
        "state_loss": jnp.asarray(state_losses, dtype=jnp.float32),
        "drift_mse": jnp.asarray(drift_mses, dtype=jnp.float32),
        "mean_bridge_logsnr": jnp.asarray(mean_logsnr_history, dtype=jnp.float32),
        "mean_prior_time_match": jnp.asarray(mean_prior_time_history, dtype=jnp.float32),
    }
    return trained_model, history


class _TokenBridgeLogSNRDriftWrapper(eqx.Module):
    drift_net: TokenConditionalDiT
    delta_v: float
    theta_feature_clip: float
    interval_length: float

    def __call__(self, theta: jax.Array | float, y: jax.Array, condition) -> jax.Array:
        logsnr = bridge_logsnr(theta, self.delta_v, theta_clip=self.theta_feature_clip)
        # The learned token controller is parameterized in local interval time theta,
        # while rollout integrates over absolute interval time tau.
        return self.drift_net(logsnr, y, condition) / jnp.asarray(self.interval_length, dtype=y.dtype)


def sample_token_paired_prior_conditional_trajectory(
    drift_net: TokenConditionalDiT,
    coarse_state: jax.Array,
    zt: jax.Array,
    delta_v: float,
    dt0: float,
    key: jax.Array,
    *,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    theta_feature_clip: float = DEFAULT_THETA_FEATURE_CLIP,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    _validate_bridge_hyperparameters(delta_v, 0.0, theta_feature_clip)
    coarse_state_arr = jnp.asarray(coarse_state, dtype=jnp.float32)
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(coarse_state_arr.dtype)
    return _sample_token_paired_prior_conditional_trajectory_impl(
        drift_net,
        coarse_state_arr,
        generation_zt,
        delta_v,
        dt0,
        key,
        condition_num_intervals=condition_num_intervals,
        interval_offset=interval_offset,
        theta_feature_clip=theta_feature_clip,
        adjoint=_resolve_forward_sampling_adjoint(adjoint),
    )


def _sample_token_paired_prior_conditional_trajectory_impl(
    drift_net: TokenConditionalDiT,
    coarse_state_arr: jax.Array,
    generation_zt: jax.Array,
    delta_v: float,
    dt0: float,
    key: jax.Array,
    *,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    theta_feature_clip: float = DEFAULT_THETA_FEATURE_CLIP,
    adjoint: diffrax.AbstractAdjoint,
) -> jax.Array:
    rollout_num_intervals = int(generation_zt.shape[0] - 1)
    condition_num_intervals_int = (
        rollout_num_intervals if condition_num_intervals is None else int(condition_num_intervals)
    )
    interval_offset_int = int(interval_offset)

    trajectory = [coarse_state_arr]
    x = coarse_state_arr
    for interval_idx in range(rollout_num_intervals):
        key, subkey = jax.random.split(key)
        tau_start = generation_zt[interval_idx]
        tau_end = generation_zt[interval_idx + 1]
        interval_length = float(jnp.maximum(tau_end - tau_start, jnp.asarray(1e-12, dtype=coarse_state_arr.dtype)))
        sigma_value = _interval_sigma(delta_v, tau_start, tau_end, dtype=coarse_state_arr.dtype)
        wrapped_drift = _TokenBridgeLogSNRDriftWrapper(
            drift_net=drift_net,
            delta_v=float(delta_v),
            theta_feature_clip=float(theta_feature_clip),
            interval_length=interval_length,
        )

        def sigma_fn(theta: jax.Array | float) -> jax.Array:
            del theta
            return jnp.asarray(sigma_value, dtype=coarse_state_arr.dtype)

        condition = make_token_bridge_condition(
            x,
            x,
            interval_idx=interval_offset_int + interval_idx,
            condition_mode="previous_state",
        )
        x = integrate_token_conditional_interval(
            wrapped_drift,
            x,
            condition,
            tau_start,
            tau_end,
            dt0,
            subkey,
            sigma_fn,
            adjoint=adjoint,
            time_mode="local",
        )
        trajectory.append(x)
    return jnp.stack(trajectory[::-1], axis=0)


def _sample_token_paired_prior_conditional_batch_impl(
    drift_net: TokenConditionalDiT,
    coarse_batch: jax.Array,
    generation_zt: jax.Array,
    delta_v: float,
    dt0: float,
    key: jax.Array,
    *,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    theta_feature_clip: float = DEFAULT_THETA_FEATURE_CLIP,
    adjoint: diffrax.AbstractAdjoint,
) -> jax.Array:
    keys = jax.random.split(key, int(coarse_batch.shape[0]))
    return jax.vmap(
        lambda coarse_state, subkey: _sample_token_paired_prior_conditional_trajectory_impl(
            drift_net,
            coarse_state,
            generation_zt,
            delta_v,
            dt0,
            subkey,
            condition_num_intervals=condition_num_intervals,
            interval_offset=interval_offset,
            theta_feature_clip=theta_feature_clip,
            adjoint=adjoint,
        )
    )(coarse_batch, keys)


def sample_token_paired_prior_conditional_batch(
    drift_net: TokenConditionalDiT,
    coarse_batch: jax.Array,
    zt: jax.Array,
    delta_v: float,
    dt0: float,
    key: jax.Array,
    *,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    theta_feature_clip: float = DEFAULT_THETA_FEATURE_CLIP,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> jax.Array:
    coarse_batch_arr = jnp.asarray(coarse_batch, dtype=jnp.float32)
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(coarse_batch_arr.dtype)
    return _sample_token_paired_prior_conditional_batch_impl(
        drift_net,
        coarse_batch_arr,
        generation_zt,
        delta_v,
        dt0,
        key,
        condition_num_intervals=condition_num_intervals,
        interval_offset=interval_offset,
        theta_feature_clip=theta_feature_clip,
        adjoint=_resolve_forward_sampling_adjoint(adjoint),
    )


def sample_token_paired_prior_conditional_dense_batch_from_keys(
    drift_net: TokenConditionalDiT,
    coarse_batch: jax.Array,
    zt: jax.Array,
    delta_v: float,
    dt0: float,
    keys: jax.Array,
    *,
    condition_num_intervals: int | None = None,
    interval_offset: int = 0,
    theta_feature_clip: float = DEFAULT_THETA_FEATURE_CLIP,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> tuple[jax.Array, jax.Array]:
    coarse_batch_arr = jnp.asarray(coarse_batch, dtype=jnp.float32)
    key_batch_arr = jnp.asarray(keys)
    if key_batch_arr.ndim != 2 or key_batch_arr.shape[0] != coarse_batch_arr.shape[0]:
        raise ValueError(
            "keys must have shape (N, 2) aligned with coarse_batch, "
            f"got keys={key_batch_arr.shape}, coarse_batch={coarse_batch_arr.shape}."
        )
    _validate_bridge_hyperparameters(delta_v, 0.0, theta_feature_clip)
    validate_data_zt(zt)
    generation_zt = generation_zt_from_data_zt(zt).astype(coarse_batch_arr.dtype)
    rollout_num_intervals = int(generation_zt.shape[0] - 1)
    condition_num_intervals_int = (
        rollout_num_intervals if condition_num_intervals is None else int(condition_num_intervals)
    )
    interval_offset_int = int(interval_offset)
    resolved_adjoint = _resolve_forward_sampling_adjoint(adjoint)

    def _sample_one(coarse_state_arr: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        trajectory = [coarse_state_arr]
        dense_segments: list[jax.Array] = []
        x = coarse_state_arr
        local_key = key
        for interval_idx in range(rollout_num_intervals):
            local_key, subkey = jax.random.split(local_key)
            tau_start = generation_zt[interval_idx]
            tau_end = generation_zt[interval_idx + 1]
            interval_length = float(jnp.maximum(tau_end - tau_start, jnp.asarray(1e-12, dtype=coarse_state_arr.dtype)))
            sigma_value = _interval_sigma(delta_v, tau_start, tau_end, dtype=coarse_state_arr.dtype)
            wrapped_drift = _TokenBridgeLogSNRDriftWrapper(
                drift_net=drift_net,
                delta_v=float(delta_v),
                theta_feature_clip=float(theta_feature_clip),
                interval_length=interval_length,
            )

            def sigma_fn(theta: jax.Array | float) -> jax.Array:
                del theta
                return jnp.asarray(sigma_value, dtype=coarse_state_arr.dtype)

            condition = make_token_bridge_condition(
                x,
                x,
                interval_idx=interval_offset_int + interval_idx,
                condition_mode="previous_state",
            )
            save_times = interval_save_times(tau_start, tau_end, dt0, dtype=coarse_state_arr.dtype)
            interval_path = integrate_token_conditional_interval(
                wrapped_drift,
                x,
                condition,
                tau_start,
                tau_end,
                dt0,
                subkey,
                sigma_fn,
                adjoint=resolved_adjoint,
                time_mode="local",
                save_times=save_times,
            )
            x = interval_path[-1]
            trajectory.append(x)
            dense_segments.append(interval_path if interval_idx == 0 else interval_path[1:])
        knots = jnp.stack(trajectory[::-1], axis=0)
        dense_path = jnp.concatenate(dense_segments, axis=0)[::-1]
        return knots, dense_path

    return jax.vmap(_sample_one)(coarse_batch_arr, key_batch_arr)


__all__ = [
    "DEFAULT_THETA_FEATURE_CLIP",
    "PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE",
    "PAIRED_PRIOR_TOKEN_DIT_TRAINING_OBJECTIVE",
    "bridge_logsnr",
    "matched_prior_time_from_bridge_logsnr",
    "resolve_prior_logsnr_max_from_checkpoint_path",
    "resolve_prior_logsnr_max_from_checkpoint_payload",
    "sample_token_paired_prior_conditional_batch",
    "sample_token_paired_prior_conditional_dense_batch_from_keys",
    "sample_token_paired_prior_conditional_trajectory",
    "train_token_paired_prior_bridge",
]
