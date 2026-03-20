from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class _MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        key: jax.Array,
    ) -> None:
        widths = (int(input_dim), *(int(h) for h in hidden_dims), int(output_dim))
        keys = jax.random.split(key, len(widths) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_features, out_features, key=subkey)
            for in_features, out_features, subkey in zip(widths[:-1], widths[1:], keys, strict=True)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jnp.asarray(x)
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        return self.layers[-1](h)


class ConditionalMLP(eqx.Module):
    """Simple conditional MLP with optional auxiliary noise."""

    net: _MLP
    condition_dim: int
    output_dim: int
    aux_noise_dim: int
    hidden_dims: tuple[int, ...]

    def __init__(
        self,
        condition_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        aux_noise_dim: int = 4,
        *,
        key: jax.Array,
    ) -> None:
        condition_dim = int(condition_dim)
        output_dim = int(output_dim)
        aux_noise_dim = int(aux_noise_dim)
        hidden_dims = tuple(int(h) for h in hidden_dims)
        if aux_noise_dim < 0:
            raise ValueError(f"aux_noise_dim must be non-negative, got {aux_noise_dim}.")

        self.net = _MLP(
            input_dim=condition_dim + aux_noise_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            key=key,
        )
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.aux_noise_dim = aux_noise_dim
        self.hidden_dims = hidden_dims

    def _resolve_aux(self, aux: jax.Array | None, *, dtype: jnp.dtype) -> jax.Array:
        if self.aux_noise_dim == 0:
            return jnp.zeros((0,), dtype=dtype)
        if aux is None:
            return jnp.zeros((self.aux_noise_dim,), dtype=dtype)
        aux_arr = jnp.asarray(aux, dtype=dtype).reshape(-1)
        if aux_arr.shape[0] != self.aux_noise_dim:
            raise ValueError(f"Expected aux shape ({self.aux_noise_dim},), got {aux_arr.shape}.")
        return aux_arr

    def __call__(self, condition: jax.Array, aux: jax.Array | None = None) -> jax.Array:
        condition_arr = jnp.asarray(condition)
        aux_arr = self._resolve_aux(aux, dtype=condition_arr.dtype)
        features = jnp.concatenate([condition_arr, aux_arr], axis=0)
        return self.net(features)


class IntervalConditionalModelStack(eqx.Module):
    """Interval models ordered by coarse level: (1->0), (2->1), ..., (L->L-1)."""

    interval_models: tuple[Any, ...]
    latent_dim: int
    hidden_dims: tuple[int, ...]
    aux_noise_dim: int

    def model_for_coarse_level(self, coarse_level: int) -> Any:
        coarse_level_int = int(coarse_level)
        if coarse_level_int < 1 or coarse_level_int > len(self.interval_models):
            raise ValueError(f"Expected coarse_level in [1, {len(self.interval_models)}], got {coarse_level}.")
        return self.interval_models[coarse_level_int - 1]


def build_interval_conditional_mlp_stack(
    latent_dim: int,
    *,
    num_intervals: int,
    hidden_dims: Sequence[int] = (128, 128),
    aux_noise_dim: int = 4,
    key: jax.Array,
) -> IntervalConditionalModelStack:
    latent_dim_int = int(latent_dim)
    num_intervals_int = int(num_intervals)
    hidden_dims_tuple = tuple(int(h) for h in hidden_dims)
    aux_noise_dim_int = int(aux_noise_dim)
    if num_intervals_int <= 0:
        raise ValueError(f"num_intervals must be positive, got {num_intervals}.")

    keys = jax.random.split(key, num_intervals_int)
    interval_models = tuple(
        ConditionalMLP(
            condition_dim=latent_dim_int,
            output_dim=latent_dim_int,
            hidden_dims=hidden_dims_tuple,
            aux_noise_dim=aux_noise_dim_int,
            key=subkey,
        )
        for subkey in keys
    )
    return IntervalConditionalModelStack(
        interval_models=interval_models,
        latent_dim=latent_dim_int,
        hidden_dims=hidden_dims_tuple,
        aux_noise_dim=aux_noise_dim_int,
    )


def _validate_stack_for_sampling(stack: IntervalConditionalModelStack, states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != int(stack.latent_dim):
        raise ValueError(f"states must have shape (n_conditions, {stack.latent_dim}), got {arr.shape}.")
    return arr


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
    return jax.random.normal(key, (int(batch_size), aux_noise_dim), dtype=dtype)


def sample_interval_conditionals(
    stack: IntervalConditionalModelStack,
    condition_states: np.ndarray,
    *,
    coarse_level: int,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    conditions = _validate_stack_for_sampling(stack, condition_states)
    model = stack.model_for_coarse_level(int(coarse_level))
    repeated = np.repeat(conditions, int(n_realizations), axis=0)
    key = jax.random.PRNGKey(int(seed))
    aux = _sample_training_aux(model, key, batch_size=repeated.shape[0], dtype=jnp.float32)
    if aux is None:
        generated = jax.vmap(model)(jnp.asarray(repeated, dtype=jnp.float32))
    else:
        generated = jax.vmap(model)(jnp.asarray(repeated, dtype=jnp.float32), aux)
    return np.asarray(generated, dtype=np.float32).reshape(conditions.shape[0], int(n_realizations), stack.latent_dim)


def sample_rollouts(
    stack: IntervalConditionalModelStack,
    coarse_states: np.ndarray,
    *,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    coarse = _validate_stack_for_sampling(stack, coarse_states)
    repeated = jnp.asarray(np.repeat(coarse, int(n_realizations), axis=0), dtype=jnp.float32)
    trajectory = [repeated]
    current = repeated
    key = jax.random.PRNGKey(int(seed))
    num_intervals = len(stack.interval_models)
    for coarse_level in range(num_intervals, 0, -1):
        model = stack.model_for_coarse_level(coarse_level)
        key, subkey = jax.random.split(key)
        aux = _sample_training_aux(model, subkey, batch_size=current.shape[0], dtype=current.dtype)
        if aux is None:
            current = jax.vmap(model)(current)
        else:
            current = jax.vmap(model)(current, aux)
        trajectory.append(current)

    stacked = jnp.stack(trajectory[::-1], axis=1)
    return np.asarray(stacked, dtype=np.float32).reshape(
        coarse.shape[0],
        int(n_realizations),
        num_intervals + 1,
        stack.latent_dim,
    )
